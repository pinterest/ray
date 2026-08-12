import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.datasource.parquet_datasink import (
    STAGE_DIR_ENV_VAR,
    STAGE_UPLOAD_CHUNK_BYTES,
    ParquetDatasink,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.planner.plan_write_op import WRITE_UUID_KWARG_NAME
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa


@pytest.fixture
def stage_dir(tmp_path, monkeypatch):
    """Points `stream_writes` staging at a directory this test can inspect."""
    path = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV_VAR, str(path))
    yield path


def _staged_files(stage_dir):
    if not os.path.exists(stage_dir):
        return []
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(stage_dir)
        for filename in filenames
    ]


def _task_ctx(write_uuid="testuuid", task_idx=0):
    return TaskContext(task_idx, "Write", kwargs={WRITE_UUID_KWARG_NAME: write_uuid})


def test_stream_writes_round_trip(ray_start_regular_shared, tmp_path):
    """`stream_writes` output is readable and holds the same rows as the default path."""
    stream_path = tmp_path / "stream"
    default_path = tmp_path / "default"
    ds = ray.data.range(100, override_num_blocks=10)

    ds.write_parquet(str(stream_path), stream_writes=True)
    ds.write_parquet(str(default_path))

    streamed = sorted(
        row["id"] for row in ray.data.read_parquet(str(stream_path)).take_all()
    )
    default = sorted(
        row["id"] for row in ray.data.read_parquet(str(default_path)).take_all()
    )
    assert streamed == list(range(100))
    assert streamed == default
    # One file per write task, same as the default path.
    assert len(os.listdir(stream_path)) == len(os.listdir(default_path))


def test_stream_writes_one_row_group_per_block(tmp_path):
    """Each block is appended as its own row group rather than buffered and merged."""
    datasink = ParquetDatasink(str(tmp_path), stream_writes=True)
    datasink.on_write_start()
    blocks = [pa.table({"id": [i * 10 + j for j in range(10)]}) for i in range(5)]
    datasink.write(iter(blocks), _task_ctx())

    files = [f for f in os.listdir(tmp_path) if f.endswith(".parquet")]
    assert len(files) == 1
    metadata = pq.ParquetFile(os.path.join(tmp_path, files[0])).metadata
    assert metadata.num_rows == 50
    # One row group per block, rather than the single merged group the default path
    # produces by writing every block at once.
    assert metadata.num_row_groups == 5


def test_stream_writes_removes_staged_files_on_success(
    ray_start_regular_shared, tmp_path, stage_dir
):
    """Staged files never outlive the task that wrote them."""
    ds = ray.data.range(100, override_num_blocks=10)
    ds.write_parquet(str(tmp_path / "out"), stream_writes=True)

    assert ray.data.read_parquet(str(tmp_path / "out")).count() == 100
    assert _staged_files(stage_dir) == []


def test_stream_writes_removes_staged_files_on_failure(tmp_path, stage_dir):
    """A failure mid-write still cleans up the partial staged file.

    Driven in-process rather than through `write_parquet`, because `write()` runs in a
    worker process and a patch applied on the driver wouldn't reach it.
    """
    datasink = ParquetDatasink(str(tmp_path / "out"), stream_writes=True)
    datasink.on_write_start()

    def blocks():
        yield pa.table({"id": [0, 1, 2]})
        raise RuntimeError("upstream boom")

    with pytest.raises(RuntimeError, match="upstream boom"):
        datasink.write(blocks(), _task_ctx())

    assert _staged_files(stage_dir) == []


def test_stream_writes_removes_staged_files_on_upload_failure(tmp_path, stage_dir):
    """A failed upload cleans up the fully-staged file it left behind."""

    class FailingUploadDatasink(ParquetDatasink):
        def _upload_staged_file(self, stage_path, write_path):
            assert os.path.exists(stage_path), "expected a finalized stage file"
            raise RuntimeError("upload boom")

    datasink = FailingUploadDatasink(str(tmp_path / "out"), stream_writes=True)
    datasink.on_write_start()

    with pytest.raises(RuntimeError, match="upload boom"):
        datasink.write(iter([pa.table({"id": [0, 1, 2]})]), _task_ctx())

    assert _staged_files(stage_dir) == []


def test_stream_writes_stage_dir_defaults_to_temp(tmp_path):
    """Staging works without the env var set, and still cleans up."""
    datasink = ParquetDatasink(str(tmp_path), stream_writes=True)
    datasink.on_write_start()
    ctx = _task_ctx()
    stage_dir = datasink._staging_dir(ctx)

    datasink.write(iter([pa.table({"id": [0, 1]})]), ctx)

    assert _staged_files(stage_dir) == []
    assert ray.data.read_parquet(str(tmp_path)).count() == 2


def test_stream_writes_staging_dir_is_unique_per_task(tmp_path, stage_dir):
    """Separate writes -- and separate tasks of one write -- stage under separate dirs."""
    datasink = ParquetDatasink(str(tmp_path), stream_writes=True)

    first = datasink._staging_dir(_task_ctx("uuid-one"))
    second = datasink._staging_dir(_task_ctx("uuid-two"))

    assert first != second
    assert str(stage_dir) in first and str(stage_dir) in second
    # A task owns its directory, so it can clean up without racing a sibling.
    assert datasink._staging_dir(_task_ctx("uuid-one", task_idx=1)) != first
    # A retry of the same task reuses it, overwriting only its own partial output.
    assert datasink._staging_dir(_task_ctx("uuid-one")) == first


def test_stream_writes_finished_task_does_not_break_a_sibling(tmp_path, stage_dir):
    """A task finishing mid-write doesn't remove a directory a sibling still needs.

    The writer is opened lazily on the first non-empty block, so a sibling that hasn't
    produced one yet is exactly the task a shared staging directory would strand.
    """
    datasink = ParquetDatasink(str(tmp_path / "out"), stream_writes=True)
    datasink.on_write_start()

    def blocks():
        # No writer open yet: another task of the same write completes here.
        datasink.write(iter([pa.table({"id": [0]})]), _task_ctx(task_idx=1))
        yield pa.table({"id": [1, 2]})

    datasink.write(blocks(), _task_ctx(task_idx=0))

    written = sorted(
        row["id"] for row in ray.data.read_parquet(str(tmp_path / "out")).take_all()
    )
    assert written == [0, 1, 2]
    assert _staged_files(stage_dir) == []


def test_stream_writes_empty_dataset_writes_no_file(ray_start_regular_shared, tmp_path):
    """An all-empty task writes nothing, matching the default path's zero-row skip."""
    out = tmp_path / "out"
    ray.data.range(0).write_parquet(str(out), stream_writes=True)

    assert not os.path.exists(out) or [
        f for f in os.listdir(out) if f.endswith(".parquet")
    ] == []


def test_stream_writes_skips_empty_blocks(tmp_path):
    """Zero-row blocks are skipped without opening a writer for them."""
    datasink = ParquetDatasink(str(tmp_path), stream_writes=True)
    datasink.on_write_start()
    schema = pa.schema([pa.field("id", pa.int64())])
    blocks = [
        pa.table({"id": pa.array([], type=pa.int64())}, schema=schema),
        pa.table({"id": [0, 1, 2]}),
        pa.table({"id": pa.array([], type=pa.int64())}, schema=schema),
    ]
    datasink.write(iter(blocks), _task_ctx())

    written = sorted(
        row["id"] for row in ray.data.read_parquet(str(tmp_path)).take_all()
    )
    assert written == [0, 1, 2]


def test_stream_writes_all_empty_blocks_writes_no_file(tmp_path, stage_dir):
    """A task that sees only empty blocks writes no file and stages nothing."""
    out = tmp_path / "out"
    datasink = ParquetDatasink(str(out), stream_writes=True)
    datasink.on_write_start()
    schema = pa.schema([pa.field("id", pa.int64())])
    empty = pa.table({"id": pa.array([], type=pa.int64())}, schema=schema)

    datasink.write(iter([empty, empty]), _task_ctx())

    assert [f for f in os.listdir(out) if f.endswith(".parquet")] == []
    assert _staged_files(stage_dir) == []


def test_stream_writes_rejects_partition_cols(tmp_path):
    """`stream_writes` writes one file per task, so it can't route partition columns."""
    with pytest.raises(ValueError, match="partition_cols"):
        ParquetDatasink(str(tmp_path), partition_cols=["id"], stream_writes=True)


def test_stream_writes_respects_user_schema(ray_start_regular_shared, tmp_path):
    """An explicit schema is used instead of the first block's inferred one."""
    schema = pa.schema([pa.field("id", pa.int32())])
    datasink = ParquetDatasink(
        str(tmp_path), stream_writes=True, arrow_parquet_args={"schema": schema}
    )
    datasink.on_write_start()
    datasink.write(iter([pa.table({"id": [0, 1]})]), _task_ctx())

    files = [f for f in os.listdir(tmp_path) if f.endswith(".parquet")]
    written_schema = pq.ParquetFile(os.path.join(tmp_path, files[0])).schema_arrow
    assert written_schema.field("id").type == pa.int32()


def test_stream_writes_uses_custom_filename_provider(
    ray_start_regular_shared, tmp_path
):
    """The filename comes from the provider, with block_index pinned for retry safety."""
    ds = ray.data.range(20, override_num_blocks=2)
    ds.write_parquet(str(tmp_path), stream_writes=True)

    files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".parquet"))
    assert len(files) == 2
    # The default provider's name is `{uuid}_{task_idx}_{block_idx}.parquet`; block_idx
    # is pinned to 0 so a retried task rewrites the same path.
    assert all(f.split("_")[-1] == "000000.parquet" for f in files), files


def test_stream_writes_larger_than_upload_chunk(ray_start_regular_shared, tmp_path):
    """Files bigger than one upload chunk transfer completely."""
    num_rows = 64
    row_bytes = STAGE_UPLOAD_CHUNK_BYTES // 16

    datasink = ParquetDatasink(
        str(tmp_path), stream_writes=True, arrow_parquet_args={"compression": "none"}
    )
    datasink.on_write_start()
    blocks = [
        pa.table(
            {
                "id": list(range(i * 16, (i + 1) * 16)),
                "payload": [os.urandom(row_bytes) for _ in range(16)],
            }
        )
        for i in range(num_rows // 16)
    ]
    datasink.write(iter(blocks), _task_ctx())

    assert ray.data.read_parquet(str(tmp_path)).count() == num_rows
    written = sum(
        os.path.getsize(os.path.join(tmp_path, f))
        for f in os.listdir(tmp_path)
        if f.endswith(".parquet")
    )
    assert written > STAGE_UPLOAD_CHUNK_BYTES


def test_stream_writes_divergent_block_schemas(ray_start_regular_shared, tmp_path):
    """Documents a real divergence from the default path.

    The default path collects every block, so it can `pa.unify_schemas` across them and
    widen a column that's null in one block and typed in another. `stream_writes` only
    ever holds one block, so it adopts the first block's schema and casts later blocks to
    it -- a widening cast has nothing to widen to and fails. Passing an explicit `schema`
    is the supported way to write blocks whose schemas differ.
    """
    null_first = pa.table({"id": [0], "value": pa.array([None], type=pa.null())})
    typed_second = pa.table({"id": [1], "value": pa.array([2], type=pa.int64())})

    datasink = ParquetDatasink(str(tmp_path / "stream"), stream_writes=True)
    datasink.on_write_start()
    with pytest.raises(Exception):
        datasink.write(iter([null_first, typed_second]), _task_ctx())

    # An explicit schema is the supported workaround.
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.int64())])
    widened = ParquetDatasink(
        str(tmp_path / "widened"),
        stream_writes=True,
        arrow_parquet_args={"schema": schema},
    )
    widened.on_write_start()
    widened.write(iter([null_first, typed_second]), _task_ctx())
    assert ray.data.read_parquet(str(tmp_path / "widened")).count() == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
