from typing import Callable


class SubDatasetConfig:
    def __init__(
        self,
        on_subdataset_finished: Callable,
        skip_finished_subdatasets: bool,
        fetch_finished_subdatasets: Callable,
        assign_subdataset_index: Callable,
    ) -> None:
        self.on_subdataset_finished = on_subdataset_finished
        self.skip_finished_subdatasets = skip_finished_subdatasets
        self.fetch_finished_subdatasets = fetch_finished_subdatasets
        self.assign_subdataset_index = assign_subdataset_index
