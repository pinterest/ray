"""Iceberg manifest writing + snapshot-producer injection.

ISOLATES all coupling to PyIceberg *private* APIs. Pinned to ``pyiceberg==0.11.0``
(see AGENTS/spec). If the pyiceberg pin changes, this module is the only place to
revisit, and the ``commit_spill_enabled`` flag provides a fallback.
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

from ray.data._internal.datasource.iceberg_commit_spill import read_spill_bin

if TYPE_CHECKING:
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile, ManifestFile
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.snapshots import SnapshotSummaryCollector


def estimate_manifest_entry_size(data_file) -> int:
    """Rough proxy for the bytes a DataFile contributes to a manifest, used to
    size spill bins so each becomes ~one target-sized manifest."""
    total = 256  # fixed per-entry overhead
    for bounds in (data_file.lower_bounds, data_file.upper_bounds):
        if bounds:
            for value in bounds.values():
                total += len(value) + 8
    for counts in (
        data_file.column_sizes,
        data_file.value_counts,
        data_file.null_value_counts,
        data_file.nan_value_counts,
    ):
        if counts:
            total += 16 * len(counts)
    return total


def _avro_codec(table_metadata: "TableMetadata") -> str:
    from pyiceberg.table import TableProperties

    return table_metadata.properties.get(
        TableProperties.WRITE_AVRO_COMPRESSION,
        TableProperties.WRITE_AVRO_COMPRESSION_DEFAULT,
    )


def _new_summary_collector(table_metadata: "TableMetadata") -> "SnapshotSummaryCollector":
    """Construct a summary collector mirroring stock ``_SnapshotProducer._summary``.

    The Pinterest pyiceberg fork adds a ``changed-identity-partition`` summary
    (gated by ``write.summary.identity-partition-limit``) that stock upstream lacks.
    We pass that limit when the installed build defines it, so the produced summary
    matches stock on both the fork and any pin without the feature.
    """
    from pyiceberg.table import TableProperties
    from pyiceberg.table.snapshots import SnapshotSummaryCollector

    kwargs = {
        "partition_summary_limit": int(
            table_metadata.properties.get(
                TableProperties.WRITE_PARTITION_SUMMARY_LIMIT,
                TableProperties.WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT,
            )
        )
    }
    identity_limit_prop = getattr(
        TableProperties, "WRITE_CHANGED_IDENTITY_PARTITION_LIMIT", None
    )
    if identity_limit_prop is not None:
        kwargs["identity_partition_limit"] = int(
            table_metadata.properties.get(
                identity_limit_prop,
                getattr(TableProperties, "WRITE_CHANGED_IDENTITY_PARTITION_LIMIT_DEFAULT", 0),
            )
        )
    return SnapshotSummaryCollector(**kwargs)


def _add_update_metrics(dst, src) -> None:
    """Fold one pyiceberg ``UpdateMetrics`` into another (all fields are ints)."""
    for name, value in vars(src).items():
        setattr(dst, name, getattr(dst, name) + value)


def merge_summary_collectors(
    collectors: List["SnapshotSummaryCollector"], table_metadata: "TableMetadata"
) -> "SnapshotSummaryCollector":
    """Combine per-bin summary collectors into one, summing global and per-partition
    metrics so the merged ``build()`` matches a single-pass stock collection."""
    merged = _new_summary_collector(table_metadata)
    for ssc in collectors:
        _add_update_metrics(merged.metrics, ssc.metrics)
        for partition_path, partition_metrics in ssc.partition_metrics.items():
            _add_update_metrics(merged.partition_metrics[partition_path], partition_metrics)
        # Fork-only: union the changed identity-partition set (see _new_summary_collector).
        changed_identity_partitions = getattr(ssc, "changed_identity_partitions", None)
        if changed_identity_partitions:
            merged.changed_identity_partitions.update(changed_identity_partitions)
    return merged


def _spec_id_of(data_file, default_spec_id: int) -> int:
    """Resolve a DataFile's partition spec, matching stock fast-append.

    Stock pyiceberg 0.11.0 stores ``spec_id`` as a pydantic field. Pinterest's
    Record-based DataFile keeps it as a runtime ``_spec_id`` that ``from_args``
    drops (it is not an Avro/manifest field). ``_FastAppendFiles._manifests``
    never reads ``data_file.spec_id`` — it uses ``table_metadata.spec()``.
    """
    spec_id = getattr(data_file, "_spec_id", None)
    if spec_id is None:
        # Property access raises on Pinterest DataFile when _spec_id was never set.
        try:
            spec_id = data_file.spec_id
        except AttributeError:
            spec_id = None
    if spec_id is None:
        spec_id = default_spec_id
        try:
            data_file.spec_id = spec_id
        except (AttributeError, TypeError):
            pass
    return spec_id


def write_manifests_for_bin(
    bin_path: str,
    table_metadata: "TableMetadata",
    io: "FileIO",
    snapshot_id: int,
    output_prefix: str,
) -> Tuple[List["ManifestFile"], "SnapshotSummaryCollector"]:
    """Write the data files in ``bin_path`` into one manifest per partition spec.

    Runs on a worker (head-node-pinned by the caller). Reads the local spill bin,
    groups by ``spec_id`` (usually one), and writes each group as an Iceberg
    manifest with ADDED entries stamped with ``snapshot_id``.

    Returns the manifests plus a ``SnapshotSummaryCollector`` accumulated from the
    same data files. The injection path never populates the producer's
    ``_added_data_files``, so this collector is how the committed snapshot summary
    (added-records / added-data-files / total-records) gets its added-side values.
    """
    from pyiceberg.manifest import (
        ManifestEntry,
        ManifestEntryStatus,
        write_manifest,
    )

    data_files = read_spill_bin(bin_path)
    default_spec_id = table_metadata.default_spec_id
    by_spec: Dict[int, List] = defaultdict(list)
    for data_file in data_files:
        by_spec[_spec_id_of(data_file, default_spec_id)].append(data_file)

    schema = table_metadata.schema()
    specs = table_metadata.specs()
    codec = _avro_codec(table_metadata)

    summary = _new_summary_collector(table_metadata)
    manifests: List["ManifestFile"] = []
    for i, (spec_id, files) in enumerate(by_spec.items()):
        spec = specs[spec_id]
        output_file = io.new_output(f"{output_prefix}-{spec_id}-{i}.avro")
        writer = write_manifest(
            format_version=table_metadata.format_version,
            spec=spec,
            schema=schema,
            output_file=output_file,
            snapshot_id=snapshot_id,
            avro_compression=codec,
        )
        with writer:
            for data_file in files:
                writer.add(
                    ManifestEntry.from_args(
                        status=ManifestEntryStatus.ADDED,
                        snapshot_id=snapshot_id,
                        data_file=data_file,
                    )
                )
                summary.add_file(data_file=data_file, schema=schema, partition_spec=spec)
        manifests.append(writer.to_manifest_file())
    return manifests, summary


class _InjectedSummaryMixin:
    """Supplies the snapshot summary from a pre-accumulated ``SnapshotSummaryCollector``.

    Stock ``_SnapshotProducer._summary`` derives added-records / added-data-files /
    added-files-size from ``_added_data_files``, which the injection path never
    populates -- so without this the committed snapshot would report zero added rows
    and an undercounted ``total-records``. We mirror stock ``_summary`` exactly, but
    seed the collector from the injected manifests' data files instead. Deletes still
    flow through ``_deleted_data_files`` (populated by the overwrite path).
    """

    _injected_summary: "SnapshotSummaryCollector" = None

    def _summary(self, snapshot_properties=None):
        if self._injected_summary is None:
            return super()._summary(snapshot_properties)

        from pyiceberg.table.snapshots import Summary, update_snapshot_summaries

        table_metadata = self._transaction.table_metadata
        ssc = self._injected_summary
        if len(self._deleted_data_files) > 0:
            specs = table_metadata.specs()
            for data_file in self._deleted_data_files:
                ssc.remove_file(
                    data_file=data_file,
                    partition_spec=specs[data_file.spec_id],
                    schema=table_metadata.schema(),
                )
        previous_snapshot = (
            table_metadata.snapshot_by_id(self._parent_snapshot_id)
            if self._parent_snapshot_id is not None
            else None
        )
        return update_snapshot_summaries(
            summary=Summary(
                operation=self._operation, **ssc.build(), **(snapshot_properties or {})
            ),
            previous_summary=previous_snapshot.summary
            if previous_snapshot is not None
            else None,
        )


class _InjectManifestsMixin(_InjectedSummaryMixin):
    """Overrides the snapshot producer's added-manifest source with pre-written
    manifests instead of buffering DataFiles in ``_added_data_files``."""

    _injected_manifests: List["ManifestFile"] = ()

    def _manifests(self):
        # Feed our pre-written manifests through the producer's own
        # _process_manifests so the merge variant still runs its merge manager
        # and the fast variant passes them through unchanged.
        return self._process_manifests(
            list(self._injected_manifests) + self._existing_manifests()
        )


def _injected_producer_cls(merge_enabled: bool):
    from pyiceberg.table.update.snapshot import _FastAppendFiles, _MergeAppendFiles

    base = _MergeAppendFiles if merge_enabled else _FastAppendFiles
    return type(f"_Injected{base.__name__}", (_InjectManifestsMixin, base), {})


def apply_appended_manifests(
    txn,
    io: "FileIO",
    table_properties: Dict[str, str],
    added_manifests: List["ManifestFile"],
    snapshot_id: int,
    snapshot_properties: Dict[str, str],
    branch: str,
    commit_uuid,
    summary_collector: "SnapshotSummaryCollector" = None,
) -> None:
    """Apply an APPEND snapshot built from ``added_manifests`` to ``txn``.

    Mirrors ``Transaction._append_snapshot_producer``'s fast-vs-merge choice and
    forces the pre-allocated ``snapshot_id``. ``summary_collector`` carries the
    added-side snapshot-summary metrics (see ``_InjectedSummaryMixin``). The caller
    commits the transaction.
    """
    from pyiceberg.table import TableProperties
    from pyiceberg.table.snapshots import Operation
    from pyiceberg.utils.properties import property_as_bool

    merge_enabled = property_as_bool(
        table_properties,
        TableProperties.MANIFEST_MERGE_ENABLED,
        TableProperties.MANIFEST_MERGE_ENABLED_DEFAULT,
    )
    producer_cls = _injected_producer_cls(merge_enabled)
    producer = producer_cls(
        Operation.APPEND,
        txn,
        io,
        commit_uuid=commit_uuid,
        snapshot_properties=snapshot_properties,
        branch=branch,
    )
    producer._injected_manifests = list(added_manifests)
    producer._injected_summary = summary_collector
    producer._snapshot_id = snapshot_id
    with producer:
        # No append_data_file calls: _manifests() supplies our manifests.
        # __exit__ runs _commit(), applying the AddSnapshot update + requirements
        # to the transaction.
        pass


def _identity_partition_delete_props(table_metadata, delete_filter) -> Dict[str, str]:
    """Reproduce the fork's ``identity-partition-only-metadata-delete`` summary stamp.

    Stock ``replace_partitions`` builds its snapshot with ``_OverwritePartitions``,
    a ``_DeleteFiles`` subclass whose ``_summary`` records whether every column the
    delete predicate references is an identity-partition column. Our injected
    producer derives from ``_OverwriteFiles`` instead, which never stamps it, and
    ``pyiceberg.table.partition_metadata`` treats a snapshot that deleted files
    *without* the flag as "cannot tell if a partition was emptied" and gives up. So
    a spill-committed dynamic overwrite would silently disable that optimization.

    Computed from the predicate rather than hardcoded, so a caller that passes a
    non-identity filter gets ``false`` like stock would. Returns nothing on builds
    that lack the feature (see ``_new_summary_collector``).
    """
    try:
        from pyiceberg.table.snapshots import (
            IDENTITY_PARTITION_ONLY_METADATA_DELETE,
            referenced_column_names,
        )
    except ImportError:
        return {}

    from pyiceberg.transforms import IdentityTransform

    schema = table_metadata.schema()
    identity_names = {
        name
        for field in table_metadata.spec().fields
        if isinstance(field.transform, IdentityTransform)
        and (name := schema.find_column_name(field.source_id)) is not None
    }
    referenced = referenced_column_names(delete_filter)
    identity_only = bool(referenced) and referenced.issubset(identity_names)
    return {IDENTITY_PARTITION_ONLY_METADATA_DELETE: str(identity_only).lower()}


class _InjectOverwriteMixin(_InjectedSummaryMixin):
    """Overwrite producer whose *added* side is pre-written manifests.

    `_added_data_files` stays empty; the base `_manifests()` therefore contributes
    only the delete + (rewritten) existing manifests, and we prepend our pre-written
    added manifests. Reuses the producer's own delete/existing handling.
    """

    _injected_manifests: List["ManifestFile"] = ()

    def _manifests(self):
        return list(self._injected_manifests) + super()._manifests()


def _injected_overwrite_cls():
    from pyiceberg.table.update.snapshot import _OverwriteFiles

    return type("_InjectedOverwriteFiles", (_InjectOverwriteMixin, _OverwriteFiles), {})


def apply_overwrite_manifests(
    txn,
    io: "FileIO",
    added_manifests: List["ManifestFile"],
    deleted_data_files: List["DataFile"],
    snapshot_id: int,
    snapshot_properties: Dict[str, str],
    branch: str,
    commit_uuid,
    summary_collector: "SnapshotSummaryCollector" = None,
    delete_filter=None,
) -> None:
    """Stage a single OVERWRITE snapshot on ``txn``: delete ``deleted_data_files`` and
    add ``added_manifests`` (pre-written). ``summary_collector`` carries the added-side
    snapshot-summary metrics (see ``_InjectedSummaryMixin``), and ``delete_filter`` is
    the predicate the deleted files were planned from, used only to reproduce stock's
    summary stamp (see ``_identity_partition_delete_props``). The caller commits the
    transaction."""
    from pyiceberg.table.snapshots import Operation

    # OVERWRITE when the branch already has a snapshot, else APPEND (mirrors
    # UpdateSnapshot.overwrite()).
    operation = (
        Operation.OVERWRITE
        if txn.table_metadata.snapshot_by_name(name=branch) is not None
        else Operation.APPEND
    )
    if delete_filter is not None:
        # Caller-supplied properties win, matching _DeleteFiles._summary.
        snapshot_properties = {
            **_identity_partition_delete_props(txn.table_metadata, delete_filter),
            **snapshot_properties,
        }
    producer_cls = _injected_overwrite_cls()
    producer = producer_cls(
        operation,
        txn,
        io,
        commit_uuid=commit_uuid,
        snapshot_properties=snapshot_properties,
        branch=branch,
    )
    producer._injected_manifests = list(added_manifests)
    producer._injected_summary = summary_collector
    producer._snapshot_id = snapshot_id
    for data_file in deleted_data_files:
        producer.delete_data_file(data_file)
    with producer:
        # No append_data_file / append_manifest calls: _manifests() supplies the
        # added manifests; __exit__ runs _commit(), staging the snapshot on txn.
        pass
