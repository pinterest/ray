import pytest

import ray
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource.datasink import Datasink, WriteResult


def _make_counting_datasink(incremental: bool):
    # Defined inside a function so Ray/cloudpickle serializes the class by value.
    # A module-level class in this test file is not importable on workers
    # (ModuleNotFoundError: test_write_incremental_results).
    class _CountingDatasink(Datasink[int]):
        def __init__(self, incremental: bool):
            self._incremental = incremental
            self.streamed = []
            self.final_write_returns = None
            self.final_num_rows = None

        @property
        def collect_write_results_incrementally(self) -> bool:
            return self._incremental

        def write(self, blocks, ctx: TaskContext) -> int:
            n = 0
            for b in blocks:
                n += b.num_rows
            return n

        def on_write_result(self, write_return: int) -> None:
            self.streamed.append(write_return)

        def on_write_complete(self, write_result: WriteResult) -> None:
            self.final_write_returns = list(write_result.write_returns)
            self.final_num_rows = write_result.num_rows

    return _CountingDatasink(incremental)


@pytest.fixture(scope="module")
def ray_start():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_incremental_streams_results_and_empties_write_returns(ray_start):
    sink = _make_counting_datasink(incremental=True)
    ray.data.range(100, override_num_blocks=5).write_datasink(sink)
    # Each write task streamed its return; none accumulated for on_write_complete.
    assert len(sink.streamed) == 5
    assert sink.final_write_returns == []
    assert sink.final_num_rows == 100


def test_non_incremental_preserves_existing_behavior(ray_start):
    sink = _make_counting_datasink(incremental=False)
    ray.data.range(100, override_num_blocks=5).write_datasink(sink)
    assert sink.streamed == []
    assert len(sink.final_write_returns) == 5
    assert sink.final_num_rows == 100
