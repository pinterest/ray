import copy
from python.ray.data._internal.logical.interfaces.plan import Plan
from ray.data._internal.logical.interfaces import LogicalPlan, Rule
from ray.data._internal.subdataset_config import SubDatasetConfig


class AddSubdatasetRule(Rule):
    """Rule to enable the subdataset to the LogicalPlans
    1. Add the subDataset_index to Read op
    2. make sure there is no AbstractAllToAll
    """
    def __init__(self, subdataset_config: SubDatasetConfig) -> None:
        pass

    def apply(plan: LogicalPlan) -> LogicalPlan:
        pass
