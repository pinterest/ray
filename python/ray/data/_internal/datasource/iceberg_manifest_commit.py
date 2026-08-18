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
    from pyiceberg.manifest import ManifestFile
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
    by_spec: Dict[int, List] = defaultdict(list)
    for data_file in data_files:
        by_spec[data_file.spec_id].append(data_file)

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
