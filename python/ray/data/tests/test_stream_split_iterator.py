import pytest
import threading
import time
from unittest.mock import MagicMock, patch

import ray
from ray.exceptions import RayTaskError
from ray.data._internal.iterator.stream_split_iterator import (
    SplitCoordinator,
    StreamSplitDataIterator,
    _DatasetWrapper,
)


@pytest.fixture(scope="function")
def ray_start_4_cpus():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


class TestSyncShutdown:
    """Tests for synchronized shutdown functionality in SplitCoordinator."""

    def test_sync_shutdown_requires_split_idx(self, ray_start_4_cpus):
        """Verify split_idx is required when sync_shutdown is True."""
        ds = ray.data.range(100)
        iterators = ds.streaming_split(2)

        # Get the coordinator actor handle
        coord_actor = iterators[0]._coord_actor

        # Should raise ValueError when split_idx is not provided
        # Ray wraps the exception in RayTaskError
        with pytest.raises(RayTaskError) as exc_info:
            ray.get(coord_actor.shutdown_executor.remote(sync_shutdown=True))
        assert "split_idx is required" in str(exc_info.value)

    def test_sync_shutdown_duplicate_call_fails(self, ray_start_4_cpus):
        """Verify calling shutdown twice from same shard raises ValueError."""
        ds = ray.data.range(100)
        n = 2
        iterators = ds.streaming_split(n)

        coord_actor = iterators[0]._coord_actor

        # Start epochs concurrently (start_epoch uses a barrier)
        epoch_futures = [
            coord_actor.start_epoch.remote(i) for i in range(n)
        ]
        ray.get(epoch_futures)

        results = {}
        errors = {}
        lock = threading.Lock()

        def call_shutdown(idx, delay=0, key=None):
            if key is None:
                key = idx
            time.sleep(delay)
            try:
                ray.get(coord_actor.shutdown_executor.remote(
                    split_idx=idx, sync_shutdown=True
                ))
                with lock:
                    results[key] = "success"
            except Exception as e:
                with lock:
                    errors[key] = str(e)

        # Start shutdown calls:
        # - shard 0 at t=0
        # - shard 0 duplicate at t=0.3 (should fail)
        # - shard 1 at t=0.5 (should complete the barrier)
        threads = [
            threading.Thread(target=call_shutdown, args=(0, 0, "shard0_first")),
            threading.Thread(target=call_shutdown, args=(0, 0.3, "shard0_dup")),
            threading.Thread(target=call_shutdown, args=(1, 0.5, "shard1")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # shard0_first should succeed (completed after barrier)
        assert "shard0_first" in results, f"Expected shard0_first to succeed. Results: {results}, Errors: {errors}"

        # shard0_dup should fail with duplicate error
        assert "shard0_dup" in errors, f"Expected shard0_dup to fail. Results: {results}, Errors: {errors}"
        assert "already called shutdown_executor" in errors["shard0_dup"]

        # shard1 should succeed
        assert "shard1" in results, f"Expected shard1 to succeed. Results: {results}, Errors: {errors}"

    def test_sync_shutdown_before_epoch_ignores(self, ray_start_4_cpus):
        """Verify calling shutdown before start_epoch is ignored."""
        ds = ray.data.range(100)
        iterators = ds.streaming_split(2)

        coord_actor = iterators[0]._coord_actor

        # Call shutdown before any epoch has started (cur_epoch == -1)
        # This should return immediately without error
        ray.get(coord_actor.shutdown_executor.remote(
            split_idx=0, sync_shutdown=True
        ))
        ray.get(coord_actor.shutdown_executor.remote(
            split_idx=1, sync_shutdown=True
        ))

    def test_sync_shutdown_all_shards_required(self, ray_start_4_cpus):
        """Verify shutdown only happens after all N shards call it."""
        ds = ray.data.range(100)
        n = 3
        iterators = ds.streaming_split(n)

        coord_actor = iterators[0]._coord_actor

        # Start epochs concurrently (start_epoch uses a barrier)
        epoch_futures = [coord_actor.start_epoch.remote(i) for i in range(n)]
        ray.get(epoch_futures)

        # Track completion
        completed = []
        lock = threading.Lock()

        def call_shutdown(idx, delay=0):
            time.sleep(delay)
            ray.get(coord_actor.shutdown_executor.remote(
                split_idx=idx, sync_shutdown=True
            ))
            with lock:
                completed.append(idx)

        # Start shutdown calls with delays
        threads = []
        for i in range(n):
            t = threading.Thread(target=call_shutdown, args=(i, i * 0.3))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=10)

        # All should have completed
        assert len(completed) == n
        assert set(completed) == set(range(n))

    def test_sync_shutdown_shard_0_triggers(self, ray_start_4_cpus):
        """Verify only shard 0 actually performs the shutdown."""
        ds = ray.data.range(100)
        n = 2
        iterators = ds.streaming_split(n)

        coord_actor = iterators[0]._coord_actor

        # Start epochs concurrently (start_epoch uses a barrier)
        epoch_futures = [coord_actor.start_epoch.remote(i) for i in range(n)]
        ray.get(epoch_futures)

        completed_order = []
        lock = threading.Lock()

        def call_shutdown(idx, delay=0):
            time.sleep(delay)
            ray.get(coord_actor.shutdown_executor.remote(
                split_idx=idx, sync_shutdown=True
            ))
            with lock:
                completed_order.append(idx)

        # Start shard 1 first, then shard 0
        t1 = threading.Thread(target=call_shutdown, args=(1, 0))
        t0 = threading.Thread(target=call_shutdown, args=(0, 0.2))

        t1.start()
        t0.start()

        t1.join(timeout=10)
        t0.join(timeout=10)

        # Both should have completed
        assert len(completed_order) == 2

    def test_non_sync_shutdown_immediate(self, ray_start_4_cpus):
        """Verify default behavior (no sync) still works."""
        ds = ray.data.range(100)
        n = 2
        iterators = ds.streaming_split(n)

        coord_actor = iterators[0]._coord_actor

        # Start epochs concurrently (start_epoch uses a barrier)
        epoch_futures = [coord_actor.start_epoch.remote(i) for i in range(n)]
        ray.get(epoch_futures)

        # Non-sync shutdown should work immediately from any shard
        ray.get(coord_actor.shutdown_executor.remote(force=True))

    def test_sync_shutdown_force_aggregated(self, ray_start_4_cpus):
        """Verify force=True from any shard results in force shutdown."""
        ds = ray.data.range(100)
        n = 2
        iterators = ds.streaming_split(n)

        coord_actor = iterators[0]._coord_actor

        # Start epochs concurrently (start_epoch uses a barrier)
        epoch_futures = [coord_actor.start_epoch.remote(i) for i in range(n)]
        ray.get(epoch_futures)

        def call_shutdown(idx, force):
            ray.get(coord_actor.shutdown_executor.remote(
                split_idx=idx, sync_shutdown=True, force=force
            ))

        # Shard 0 calls without force, shard 1 calls with force
        threads = [
            threading.Thread(target=call_shutdown, args=(0, False)),
            threading.Thread(target=call_shutdown, args=(1, True)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Shutdown should have completed (force from shard 1 should have been used)


class TestStreamSplitDataIterator:
    """Tests for StreamSplitDataIterator basic functionality."""

    def test_basic_streaming_split(self, ray_start_4_cpus):
        """Test basic streaming split functionality."""
        ds = ray.data.range(100)
        n = 2
        iterators = ds.streaming_split(n)

        assert len(iterators) == n

        results = [[] for _ in range(n)]

        def consume(idx):
            for batch in iterators[idx].iter_batches(batch_size=10):
                results[idx].extend(batch["id"].tolist())

        threads = [threading.Thread(target=consume, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Verify all data was consumed
        all_values = sorted(results[0] + results[1])
        assert all_values == list(range(100))

    def test_world_size(self, ray_start_4_cpus):
        """Test world_size() returns correct value."""
        ds = ray.data.range(100)
        n = 3
        iterators = ds.streaming_split(n)

        for it in iterators:
            assert it.world_size() == n


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
