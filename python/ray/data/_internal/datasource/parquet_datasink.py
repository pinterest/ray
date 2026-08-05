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

        Unlike the default `list(blocks)` path, blocks are appended to one open
        `ParquetWriter` and released before the next is pulled, so peak memory is bounded
        by a single block rather than the task's whole output. Output is one file per task
        under the default path's deterministic name, so a task retry overwrites it
        wholesale instead of leaking duplicate rows.

        The writer's sink is a local file, uploaded once complete, rather than the
        destination stream. Writing straight to an object store would hold that stream
        open for the whole task, and pyarrow's `S3FileSystem` binds credentials at
        construction; signing each part can't renew a session token, so a task spanning
        the token expiry fails with `ExpiredToken` as `AWS Error UNKNOWN (HTTP status
        400)` -- absent from `DataContext.retried_io_errors`, so never retried. Staging
        shrinks that window to the final upload. Driving multipart directly (one part per
        block, fresh client each) would avoid the disk entirely, but pyarrow can't resume
        an upload, so it would mean owning part numbering, ETags, size limits, and aborts
        here.

        Two consequences. Resilience is task-level rather than the default per-write
        `call_with_retry`, since a consumed generator can't be rewound. And blocks with
        differing schemas can't be unified as the default path does via `pa.unify_schemas`
        -- the first block's schema wins and later blocks are cast to it, so callers with
        non-uniform blocks must pass an explicit `schema`.
        """
        import pyarrow.parquet as pq

        write_kwargs = _resolve_kwargs(
            self.arrow_parquet_args_fn, **self.arrow_parquet_args
        )
        user_schema = write_kwargs.pop("schema", None)
        # One row group per block, so block boundaries set the row group size.
        write_kwargs.pop("row_group_size", None)

        stage_dir = self._staging_dir(ctx)
        os.makedirs(stage_dir, exist_ok=True)

        # Opened lazily on the first non-empty block, so an all-empty task writes nothing.
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
                    # block_index is pinned to 0 so the name is stable across retries.
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

            # Write the footer before uploading, so what lands is always readable.
            # Cleared first so the `finally` block doesn't double-close.
            writer, to_close = None, writer
            to_close.close()

            logger.debug(f"Uploading {stage_path} to {write_path}.")
            self._upload_staged_file(stage_path, write_path)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    logger.exception(
                        f"Failed to close parquet writer for {stage_path}."
                    )
            # Runs on success as well as failure, so a staged file never outlives its
            # task. Leaks are logged, not raised: they'd mask the original error, but
            # staged files may share a volume with Ray's object spilling.
            if stage_path is not None:
                try:
                    os.remove(stage_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.exception(f"Failed to remove staged file {stage_path}.")
                # Succeeds for whichever task finishes last; the others' dirs aren't
                # empty yet.
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
