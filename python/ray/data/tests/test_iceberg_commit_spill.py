import os

from ray.data._internal.datasource.iceberg_commit_spill import (
    DataFileSpiller,
    read_spill_bin,
)


def test_rolls_bins_by_count(tmp_path):
    spiller = DataFileSpiller(
        str(tmp_path), target_size_bytes=10**9, max_files_per_bin=2, size_of=lambda x: 1
    )
    spiller.add([1, 2, 3])
    spiller.add([4, 5])
    bins = spiller.close()
    assert len(bins) == 3  # [1,2], [3,4], [5]
    contents = [read_spill_bin(b) for b in bins]
    assert contents == [[1, 2], [3, 4], [5]]


def test_rolls_bins_by_size(tmp_path):
    spiller = DataFileSpiller(
        str(tmp_path), target_size_bytes=10, max_files_per_bin=10**9, size_of=lambda x: 4
    )
    spiller.add([1, 2, 3, 4, 5])  # rolls after 3 items (12 >= 10)
    bins = spiller.close()
    assert len(bins) == 2
    assert [read_spill_bin(b) for b in bins] == [[1, 2, 3], [4, 5]]


def test_close_is_idempotent_and_cleanup_removes_files(tmp_path):
    spiller = DataFileSpiller(
        str(tmp_path), target_size_bytes=10**9, max_files_per_bin=10**9, size_of=lambda x: 1
    )
    spiller.add([1, 2])
    bins = spiller.close()
    assert spiller.close() == bins  # idempotent
    assert all(os.path.exists(b) for b in bins)
    spiller.cleanup()
    assert not any(os.path.exists(b) for b in bins)


def test_empty_spiller_produces_no_bins(tmp_path):
    spiller = DataFileSpiller(
        str(tmp_path), target_size_bytes=10, max_files_per_bin=10, size_of=lambda x: 1
    )
    assert spiller.close() == []
