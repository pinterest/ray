import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

from ray._common.retry import call_with_retry
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.planner.plan_write_op import WRITE_UUID_KWARG_NAME
from ray.data._internal.savemode import SaveMode
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import _resolve_kwargs
from ray.data.datasource.file_datasink import _FileDatasink
from ray.data.datasource.filename_provider import FilenameProvider

if TYPE_CHECKING:
    import pyarrow

WRITE_FILE_MAX_ATTEMPTS = 10
WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS = 32

# Map Ray Data's SaveMode to pyarrow's existing_data_behavior property which is exposed via the
# `pyarrow.dataset.write_dataset` function.
# Docs: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
EXISTING_DATA_BEHAVIOR_MAP = {
    SaveMode.APPEND: "overwrite_or_ignore",
    SaveMode.OVERWRITE: "overwrite_or_ignore",  # delete_matching is not a suitable choice for parallel writes.
    SaveMode.IGNORE: "overwrite_or_ignore",
    SaveMode.ERROR: "error",
}

FILE_FORMAT = "parquet"

# These args are part of https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html#pyarrow.fs.FileSystem.open_output_stream
# and are not supported by ParquetDatasink.
UNSUPPORTED_OPEN_STREAM_ARGS = {"path", "buffer", "metadata"}

# https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
ARROW_DEFAULT_MAX_ROWS_PER_GROUP = 1024 * 1024

# Overrides the root directory that `stream_writes` stages task output in. Defaults to a
# subdirectory of the system temp dir. Note that Ray's object spilling may live on the
# same volume, so staged files are always removed once uploaded -- a leaked stage file
# per task would push a shared volume toward the spill capacity threshold and make Ray
# start failing tasks.
STAGE_DIR_ENV_VAR = "RAY_DATA_PARQUET_WRITE_STAGE_DIR"
DEFAULT_STAGE_DIR_NAME = "ray_data_parquet_write_stage"

# Chunk size for copying a staged file to its destination. Large enough to keep object
# store multipart uploads efficient without materializing the whole file in memory.
STAGE_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024

logger = logging.getLogger(__name__)


def choose_row_group_limits(
    row_group_size: Optional[int],
    min_rows_per_file: Optional[int],
    max_rows_per_file: Optional[int],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Configure `min_rows_per_group`, `max_rows_per_group`, `max_rows_per_file` parameters of Pyarrow's `write_dataset` API based on Ray Data's configuration

    Returns
    -------
    (min_rows_per_group, max_rows_per_group, max_rows_per_file)
    """

    if (
        row_group_size is None
        and min_rows_per_file is None
        and max_rows_per_file is None
    ):
        return None, None, None

    elif row_group_size is None:
        # No explicit row group size provided. We are defaulting to
        # either the caller's min_rows_per_file or max_rows_per_file limits
        # or Arrow's defaults
        min_rows_per_group, max_rows_per_group, max_rows_per_file = (
            min_rows_per_file,
            max_rows_per_file,
            max_rows_per_file,
        )

        # If min_rows_per_group is provided and max_rows_per_group is not,
        # and min_rows_per_group is greater than Arrow's default max_rows_per_group,
        # we set max_rows_per_group to min_rows_per_group to avoid creating too many row groups.
        if (
            min_rows_per_group is not None
            and max_rows_per_group is None
            and min_rows_per_group > ARROW_DEFAULT_MAX_ROWS_PER_GROUP
        ):
            max_rows_per_group, max_rows_per_file = (
                min_rows_per_group,
                min_rows_per_group,
            )

        return min_rows_per_group, max_rows_per_group, max_rows_per_file

    elif row_group_size is not None and (
        min_rows_per_file is None or max_rows_per_file is None
    ):
        return row_group_size, row_group_size, max_rows_per_file

    else:
        # Clamp the requested `row_group_size` so that it is
        # * no smaller than `min_rows_per_file` (`lower`)
        # * no larger than `max_rows_per_file` (or Arrow's default cap) (`upper`)
        # This keeps each row-group within the per-file limits while staying
        # as close as possible to the requested size.
        clamped_group_size = max(
            min_rows_per_file, min(row_group_size, max_rows_per_file)
        )
        return clamped_group_size, clamped_group_size, max_rows_per_file


class ParquetDatasink(_FileDatasink):
    def __init__(
        self,
        path: str,
        *,
        partition_cols: Optional[List[str]] = None,
        arrow_parquet_args_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        arrow_parquet_args: Optional[Dict[str, Any]] = None,
        min_rows_per_file: Optional[int] = None,
        max_rows_per_file: Optional[int] = None,
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        try_create_dir: bool = True,
        open_stream_args: Optional[Dict[str, Any]] = None,
        filename_provider: Optional[FilenameProvider] = None,
        dataset_uuid: Optional[str] = None,
        mode: SaveMode = SaveMode.APPEND,
        stream_writes: bool = False,
    ):
        if arrow_parquet_args_fn is None:
            arrow_parquet_args_fn = lambda: {}  # noqa: E731

        if arrow_parquet_args is None:
            arrow_parquet_args = {}

        self.arrow_parquet_args_fn = arrow_parquet_args_fn
        self.arrow_parquet_args = arrow_parquet_args
        self.min_rows_per_file = min_rows_per_file
        self.max_rows_per_file = max_rows_per_file
        self.partition_cols = partition_cols
        self.stream_writes = stream_writes

        if stream_writes and partition_cols:
            raise ValueError(
                "`stream_writes` doesn't support `partition_cols`, because it writes "
                "one file per task rather than routing rows to per-partition files."
            )

        if self.min_rows_per_file is not None and self.max_rows_per_file is not None:
            assert (
                self.min_rows_per_file <= self.max_rows_per_file
            ), "min_rows_per_file must be less than or equal to max_rows_per_file"

        if open_stream_args is not None:
            intersecting_keys = UNSUPPORTED_OPEN_STREAM_ARGS.intersection(
                set(open_stream_args.keys())
            )
            if intersecting_keys:
                logger.warning(
                    "open_stream_args contains unsupported arguments: %s. These arguments "
                    "are not supported by ParquetDatasink. They will be ignored.",
                    intersecting_keys,
                )

            if "compression" in open_stream_args:
                self.arrow_parquet_args["compression"] = open_stream_args["compression"]

        super().__init__(
            path,
            filesystem=filesystem,
            try_create_dir=try_create_dir,
            open_stream_args=open_stream_args,
            filename_provider=filename_provider,
            dataset_uuid=dataset_uuid,
            file_format=FILE_FORMAT,
            mode=mode,
        )

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> None:
        import pyarrow as pa

        if self.stream_writes:
            return self._write_streaming(blocks, ctx)

        blocks = list(blocks)

        if all(BlockAccessor.for_block(block).num_rows() == 0 for block in blocks):
            return

        blocks = [
            block for block in blocks if BlockAccessor.for_block(block).num_rows() > 0
        ]

        filename = self.filename_provider.get_filename_for_block(
            blocks[0], ctx.kwargs[WRITE_UUID_KWARG_NAME], ctx.task_idx, 0
        )
        write_kwargs = _resolve_kwargs(
            self.arrow_parquet_args_fn, **self.arrow_parquet_args
        )
        user_schema = write_kwargs.pop("schema", None)

        def write_blocks_to_path():
            tables = [BlockAccessor.for_block(block).to_arrow() for block in blocks]
            if user_schema is None:
                output_schema = pa.unify_schemas([table.schema for table in tables])
            else:
                output_schema = user_schema

            self._write_parquet_files(
                tables,
                filename,
                output_schema,
                ctx.kwargs[WRITE_UUID_KWARG_NAME],
                write_kwargs,
            )

        logger.debug(f"Writing {filename} file to {self.path}.")

        call_with_retry(
            write_blocks_to_path,
            description=f"write '{filename}' to '{self.path}'",
            match=self._data_context.retried_io_errors,
            max_attempts=WRITE_FILE_MAX_ATTEMPTS,
            max_backoff_s=WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS,
        )

    def _write_streaming(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> None:
        """Writes each block as a row group, without buffering the whole task's blocks.

        The default write path calls `list(blocks)` and holds every block of the task in
        memory until the last one arrives. When the read, transform, and write operators
        fuse into a single task, that pulls the entire fused pipeline to exhaustion
        before a single row is flushed, so peak memory scales with the task's whole
        output.

        This path instead appends each block to one open `ParquetWriter` and drops the
        reference before pulling the next, bounding peak memory to a single block. The
        output is still exactly one file per task, using the same deterministic filename
        the default path produces, so a task-level retry reopens the same path and
        overwrites it wholesale rather than leaking duplicate rows. A per-block file
        would not be idempotent: a partial first attempt would strand its already
        flushed blocks as duplicate rows for the downstream reader.

        The writer's sink is a LOCAL file, uploaded to the destination once the last
        block is written, rather than the destination stream itself. The reason is
        credential lifetime. Writing straight to an object store keeps that stream open
        for the task's entire duration, and pyarrow's `S3FileSystem` binds AWS
        credentials when it is constructed, so every underlying `UploadPart` reuses the
        credentials captured at open time. Each part is separately signed, but signing
        cannot renew a session token, so a task whose stream straddles the token expiry
        fails with `ExpiredToken`, surfaced as `AWS Error UNKNOWN (HTTP status 400)`.
        That string is not in `DataContext.retried_io_errors`, so it isn't retried, and
        refreshing the filesystem mid-task doesn't help because the open stream keeps
        its original credentials. The default path avoids this by doing all of its work
        before opening any stream; staging locally restores that property, shrinking the
        credential window from the whole task to the few seconds of the final upload.

        Driving the multipart upload directly -- one `UploadPart` per block, each with a
        freshly resolved client -- would remove the local disk requirement and bound
        memory to one part. It is deliberately not done here: pyarrow exposes no way to
        resume an upload (`UploadId` and the ETag list are private to the stream, and
        reopening a stream starts a new upload that truncates the object), so it would
        mean owning part numbering, ETag ordering, the 5MB minimum part size, the
        10,000 part ceiling, and abort-on-failure inside the datasink. Local staging
        gets the same memory bound without that surface.

        Staging also makes each task's output atomic in practice: nothing appears at the
        destination until the local file is complete, so a failed task cannot leave a
        truncated, footerless file behind for the downstream reader. The tradeoff versus
        the default path is that write resilience becomes task-level rather than the
        default per-write `call_with_retry` -- a mid-upload error can't be retried in
        place because the upstream generator is already consumed and can't be rewound,
        so Ray retries the whole task.

        One behavioral difference from the default path: it collects every block, so it
        can `pa.unify_schemas` across them and widen a column that is null-typed in one
        block and concretely typed in another. Holding a single block makes that
        impossible here, so the first block's schema is adopted and later blocks are cast
        to it, which raises when a later block needs a wider type. Callers with
        non-uniform blocks must pass an explicit `schema`.
        """
        import pyarrow.parquet as pq

        write_kwargs = _resolve_kwargs(
            self.arrow_parquet_args_fn, **self.arrow_parquet_args
        )
        user_schema = write_kwargs.pop("schema", None)
        # Only meaningful to `ds.write_dataset`; this path writes one row group per
        # block, so the block boundaries set the row group size.
        write_kwargs.pop("row_group_size", None)

        stage_dir = self._staging_dir(ctx)
        os.makedirs(stage_dir, exist_ok=True)

        # One writer and one staged file per task, opened lazily on the first non-empty
        # block so an all-empty task writes nothing, matching the default path's
        # zero-row skip.
        writer = None
        stage_path = None
        write_path = None
        try:
            for block in blocks:
                accessor = BlockAccessor.for_block(block)
                if accessor.num_rows() == 0:
                    continue

                table = accessor.to_arrow()
                if writer is None:
                    # Pin block_index to 0: one file per task keeps the name stable
                    # across retries. The stage file reuses the name so a retry
                    # overwrites its own stage file instead of colliding with a
                    # concurrent task.
                    filename = self.filename_provider.get_filename_for_block(
                        block, ctx.kwargs[WRITE_UUID_KWARG_NAME], ctx.task_idx, 0
                    )
                    write_path = f"{self.path.rstrip('/')}/{filename}"
                    stage_path = os.path.join(stage_dir, filename)
                    logger.debug(f"Staging {filename} at {stage_path}.")
                    schema = user_schema if user_schema is not None else table.schema
                    writer = pq.ParquetWriter(stage_path, schema, **write_kwargs)
                if not table.schema.equals(writer.schema):
                    table = table.cast(writer.schema)
                writer.write_table(table)
                del table, accessor

            if writer is None:
                return

            # Finalize the local file, writing the parquet footer, before uploading, so
            # what lands at the destination is always complete and readable. Cleared
            # first so the `finally` block doesn't double-close.
            writer, to_close = None, writer
            to_close.close()

            logger.debug(f"Uploading {stage_path} to {write_path}.")
            self._upload_staged_file(stage_path, write_path)
        finally:
            if writer is not None:
                # Failure path: close the writer so the fd isn't leaked. The partial
                # stage file is removed below.
                try:
                    writer.close()
                except Exception:
                    logger.exception(
                        f"Failed to close parquet writer for {stage_path}."
                    )
            # Runs on the success path as well as on failure, so a staged file never
            # outlives the task that wrote it.
            if stage_path is not None:
                try:
                    os.remove(stage_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    # Never mask the original error, but do flag the leak: staged files
                    # may share a volume with Ray's object spilling.
                    logger.exception(f"Failed to remove staged file {stage_path}.")
                # Best effort: removes the per-write directory once the last task on
                # this node finishes. Fails harmlessly while sibling tasks still have
                # files staged, since the directory isn't empty yet.
                try:
                    os.rmdir(stage_dir)
                except OSError:
                    pass

    def _staging_dir(self, ctx: TaskContext) -> str:
        """Returns the directory used to stage this task's output before uploading.

        The path is namespaced by the per-write UUID so that concurrent or successive
        writes -- including separate Ray jobs sharing a node, and separate write calls
        within one job -- never stage into the same directory. Combined with the
        task-deterministic filename, that means a given stage path belongs to exactly
        one write task, so a retry overwrites only its own partial output.
        """
        stage_root = os.environ.get(STAGE_DIR_ENV_VAR) or os.path.join(
            tempfile.gettempdir(), DEFAULT_STAGE_DIR_NAME
        )
        return os.path.join(stage_root, ctx.kwargs[WRITE_UUID_KWARG_NAME])

    def _upload_staged_file(self, stage_path: str, write_path: str) -> None:
        """Copies the finalized local file to `write_path` in chunks.

        The destination stream is opened here, immediately before the transfer, so it
        holds freshly resolved credentials for the seconds the upload takes rather than
        credentials captured when the task started.
        """
        with open(stage_path, "rb") as src, self.open_output_stream(
            write_path
        ) as dest:
            while True:
                chunk = src.read(STAGE_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                dest.write(chunk)

    def _get_basename_template(self, filename: str, write_uuid: str) -> str:
        # Check if write_uuid is present in filename, add if missing
        if write_uuid not in filename and self.mode == SaveMode.APPEND:
            raise ValueError(
                f"Write UUID '{write_uuid}' missing from filename template '{filename}'. This could result in files being overwritten."
                f"Modify your FileNameProvider implementation to include the `write_uuid` into the filename template or change your write mode to SaveMode.OVERWRITE. "
            )
        # Check if filename is already templatized
        if "{i}" in filename:
            # Filename is already templatized, but may need file extension
            if FILE_FORMAT not in filename:
                # Add file extension to templatized filename
                basename_template = f"{filename}.{FILE_FORMAT}"
            else:
                # Already has extension, use as-is
                basename_template = filename
        elif FILE_FORMAT not in filename:
            # No extension and not templatized, add extension and template
            basename_template = f"{filename}-{{i}}.{FILE_FORMAT}"
        else:
            # TODO(@goutamvenkat-anyscale): Add a warning if you pass in a custom
            # filename provider and it isn't templatized.
            # Use pathlib.Path to properly handle filenames with dots
            filename_path = Path(filename)
            stem = filename_path.stem  # filename without extension
            assert "." not in stem, "Filename should not contain a dot"
            suffix = filename_path.suffix  # extension including the dot
            basename_template = f"{stem}-{{i}}{suffix}"
        return basename_template

    def _write_parquet_files(
        self,
        tables: List["pyarrow.Table"],
        filename: str,
        output_schema: "pyarrow.Schema",
        write_uuid: str,
        write_kwargs: Dict[str, Any],
    ) -> None:
        import pyarrow.dataset as ds

        # Make every incoming batch conform to the final schema *before* writing
        for idx, table in enumerate(tables):
            if output_schema and not table.schema.equals(output_schema):
                table = table.cast(output_schema)
            tables[idx] = table

        row_group_size = write_kwargs.pop("row_group_size", None)

        existing_data_behavior = EXISTING_DATA_BEHAVIOR_MAP.get(
            self.mode, "overwrite_or_ignore"
        )

        (
            min_rows_per_group,
            max_rows_per_group,
            max_rows_per_file,
        ) = choose_row_group_limits(
            row_group_size,
            min_rows_per_file=self.min_rows_per_file,
            max_rows_per_file=self.max_rows_per_file,
        )

        basename_template = self._get_basename_template(filename, write_uuid)

        ds.write_dataset(
            data=tables,
            base_dir=self.path,
            schema=output_schema,
            basename_template=basename_template,
            filesystem=self.filesystem,
            partitioning=self.partition_cols,
            format=FILE_FORMAT,
            existing_data_behavior=existing_data_behavior,
            partitioning_flavor="hive",
            use_threads=True,
            min_rows_per_group=min_rows_per_group,
            max_rows_per_group=max_rows_per_group,
            max_rows_per_file=max_rows_per_file,
            file_options=ds.ParquetFileFormat().make_write_options(**write_kwargs),
        )

    @property
    def min_rows_per_write(self) -> Optional[int]:
        return self.min_rows_per_file
