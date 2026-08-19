"""Iceberg manifest writing + snapshot-producer injection.

ISOLATES all coupling to PyIceberg *private* APIs. Pinned to ``pyiceberg==0.11.0``
(see AGENTS/spec). If the pyiceberg pin changes, this module is the only place to
revisit, and the ``commit_spill_enabled`` flag provides a fallback.
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from ray.data._internal.datasource.iceberg_commit_spill import read_spill_bin

if TYPE_CHECKING:
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile, ManifestFile
    from pyiceberg.table.metadata import TableMetadata


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
) -> List["ManifestFile"]:
    """Write the data files in ``bin_path`` into one manifest per partition spec.

    Runs on a worker (head-node-pinned by the caller). Reads the local spill bin,
    groups by ``spec_id`` (usually one), and writes each group as an Iceberg
    manifest with ADDED entries stamped with ``snapshot_id``.
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

    manifests: List["ManifestFile"] = []
    for i, (spec_id, files) in enumerate(by_spec.items()):
        output_file = io.new_output(f"{output_prefix}-{spec_id}-{i}.avro")
        writer = write_manifest(
            format_version=table_metadata.format_version,
            spec=specs[spec_id],
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
        manifests.append(writer.to_manifest_file())
    return manifests


class _InjectManifestsMixin:
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
) -> None:
    """Apply an APPEND snapshot built from ``added_manifests`` to ``txn``.

    Mirrors ``Transaction._append_snapshot_producer``'s fast-vs-merge choice and
    forces the pre-allocated ``snapshot_id``. The caller commits the transaction.
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
    producer._snapshot_id = snapshot_id
    with producer:
        # No append_data_file calls: _manifests() supplies our manifests.
        # __exit__ runs _commit(), applying the AddSnapshot update + requirements
        # to the transaction.
        pass


class _InjectOverwriteMixin:
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
) -> None:
    """Stage a single OVERWRITE snapshot on ``txn``: delete ``deleted_data_files`` and
    add ``added_manifests`` (pre-written). The caller commits the transaction."""
    from pyiceberg.table.snapshots import Operation

    # OVERWRITE when the branch already has a snapshot, else APPEND (mirrors
    # UpdateSnapshot.overwrite()).
    operation = (
        Operation.OVERWRITE
        if txn.table_metadata.snapshot_by_name(name=branch) is not None
        else Operation.APPEND
    )
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
    producer._snapshot_id = snapshot_id
    for data_file in deleted_data_files:
        producer.delete_data_file(data_file)
    with producer:
        # No append_data_file / append_manifest calls: _manifests() supplies the
        # added manifests; __exit__ runs _commit(), staging the snapshot on txn.
        pass
