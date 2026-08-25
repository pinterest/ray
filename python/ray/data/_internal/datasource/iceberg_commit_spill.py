"""Local-disk spill for Iceberg commit data files.

Spills lists of items (Iceberg ``DataFile`` objects) into size-bounded pickle
bins so the driver never holds all of them in memory at once. Generic and
pyiceberg-free so it can be unit-tested in isolation.
"""
import os
import pickle
from typing import Any, Callable, List


class DataFileSpiller:
    def __init__(
        self,
        spill_dir: str,
        target_size_bytes: int,
        max_files_per_bin: int,
        size_of: Callable[[Any], int],
    ):
        self._dir = spill_dir
        self._target_size_bytes = target_size_bytes
        self._max_files_per_bin = max_files_per_bin
        self._size_of = size_of
        self._bins: List[str] = []
        self._current: List[Any] = []
        self._current_bytes = 0

    def add(self, items: List[Any]) -> None:
        for item in items:
            self._current.append(item)
            self._current_bytes += self._size_of(item)
            if (
                self._current_bytes >= self._target_size_bytes
                or len(self._current) >= self._max_files_per_bin
            ):
                self._flush()

    def _flush(self) -> None:
        if not self._current:
            return
        path = os.path.join(self._dir, f"bin_{len(self._bins):06d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._current, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._bins.append(path)
        self._current = []
        self._current_bytes = 0

    def close(self) -> List[str]:
        self._flush()
        return list(self._bins)

    def cleanup(self) -> None:
        for path in self._bins:
            try:
                os.remove(path)
            except OSError:
                pass


def read_spill_bin(path: str) -> List[Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
