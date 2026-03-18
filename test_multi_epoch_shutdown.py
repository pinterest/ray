"""Integration test for multi-epoch synchronized shutdown."""
import ray
import threading
import time


def test_multi_epoch_sync_shutdown():
    """Test that we can start and shutdown multiple epochs.

    This tests that the shutdown state variables are properly reset when
    starting a new epoch. Without the fix, the second epoch's sync shutdown
    would fail because shards are already in _shutdown_received_from.
    """
    N = 2  # Number of splits
    NUM_EPOCHS = 3

    ds = ray.data.range(100)
    iterators = ds.streaming_split(N)
    coord_actor = iterators[0]._coord_actor

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Starting epoch {epoch + 1} ===")

        # Consume data from all iterators concurrently
        # Each iterator calls start_epoch internally when iterating
        errors = []
        results = [None] * N

        def consume_data(idx, iterator):
            try:
                count = 0
                for batch in iterator.iter_batches(batch_size=10):
                    count += len(batch["id"])
                results[idx] = count
                print(f"  Shard {idx} consumed {count} rows")
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i, it in enumerate(iterators):
            t = threading.Thread(target=consume_data, args=(i, it))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        if errors:
            for idx, e in errors:
                print(f"  ERROR in shard {idx}: {e}")
            raise errors[0][1]

        print(f"  All shards consumed data successfully")

        # Synchronized shutdown from all shards
        print(f"  Calling sync shutdown from all shards...")
        shutdown_errors = []

        def call_shutdown(idx):
            try:
                ray.get(coord_actor.shutdown_executor.remote(
                    split_idx=idx, sync_shutdown=True
                ))
                print(f"    Shard {idx} shutdown complete")
            except Exception as e:
                shutdown_errors.append((idx, e))
                print(f"    Shard {idx} shutdown ERROR: {e}")

        shutdown_threads = []
        for i in range(N):
            t = threading.Thread(target=call_shutdown, args=(i,))
            shutdown_threads.append(t)
            t.start()

        for t in shutdown_threads:
            t.join(timeout=30)

        if shutdown_errors:
            print(f"\n!!! Shutdown failed at epoch {epoch + 1} !!!")
            for idx, e in shutdown_errors:
                print(f"  Shard {idx} error: {e}")
            raise shutdown_errors[0][1]

        print(f"=== Epoch {epoch + 1} complete ===")

    print(f"\n{'='*50}")
    print(f"Successfully completed {NUM_EPOCHS} epochs!")
    print(f"{'='*50}")


def main():
    ray.init()
    try:
        test_multi_epoch_sync_shutdown()
        print("\nMulti-epoch shutdown test PASSED!")
    except Exception as e:
        print(f"\nMulti-epoch shutdown test FAILED: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
