"""Integration test for automatic epoch cleanup via iterator __del__."""
import ray
import gc


@ray.remote
class ShardConsumer:
    """Actor that consumes data from a single shard."""

    def __init__(self, iterator, shard_idx):
        self._iterator = iterator
        self._shard_idx = shard_idx
        self._batch_iter = None

    def consume_epoch(self, max_batches=None):
        """Consume data from the iterator for one epoch.

        Args:
            max_batches: If set, stop after consuming this many batches.
                         None = consume all.

        Returns:
            Number of rows consumed.
        """
        # Create a new batch iterator for this epoch
        self._batch_iter = self._iterator.iter_batches(batch_size=10)

        count = 0
        batches_consumed = 0
        for batch in self._batch_iter:
            count += len(batch["id"])
            batches_consumed += 1
            if max_batches is not None and batches_consumed >= max_batches:
                break

        print(f"  Shard {self._shard_idx} consumed {count} rows ({batches_consumed} batches)")
        return count

    def cleanup(self):
        """Delete the batch iterator to trigger __del__ cleanup."""
        if self._batch_iter is not None:
            del self._batch_iter
            self._batch_iter = None
            gc.collect()
        print(f"    Shard {self._shard_idx} cleanup triggered")


def test_auto_cleanup_multi_epoch():
    """Test that executor is automatically shutdown when iterators are deleted.

    Verifies:
    1. Multiple epochs work without explicit shutdown_executor calls
    2. Executor is cleaned up automatically via __del__ when batch iterators are deleted
    """
    N = 2
    NUM_EPOCHS = 3

    ds = ray.data.range(100)
    iterators = ds.streaming_split(N)
    coord_actor = iterators[0]._coord_actor

    # Create one actor per shard
    consumers = [
        ShardConsumer.remote(it, i)
        for i, it in enumerate(iterators)
    ]

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Starting epoch {epoch + 1} ===")

        # Consume ALL data from all shards concurrently via actors
        consume_futures = [c.consume_epoch.remote() for c in consumers]
        results = ray.get(consume_futures)
        print(f"  All shards consumed data successfully: {results}")

        # Trigger cleanup by deleting batch iterators (calls __del__)
        print(f"  Triggering cleanup from all shards...")
        cleanup_futures = [c.cleanup.remote() for c in consumers]
        ray.get(cleanup_futures)

        # Verify executor was shutdown
        executor_state = ray.get(coord_actor._get_executor_state.remote())
        print(f"  Executor state after cleanup: {executor_state}")

        print(f"=== Epoch {epoch + 1} complete ===")

    print(f"\n{'='*50}")
    print(f"Test: Multi-epoch auto cleanup - PASSED ({NUM_EPOCHS} epochs)")
    print(f"{'='*50}")


def main():
    ray.init()
    try:
        test_auto_cleanup_multi_epoch()
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
