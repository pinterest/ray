from .autoscaler import Autoscaler
from .autoscaling_actor_pool import AutoscalingActorPool
from .default_autoscaler import DefaultAutoscaler
from .backpressure_aware_autoscaler import BackPressureAwareAutoscaler


def create_autoscaler(topology, resource_manager, execution_id):
    return BackPressureAwareAutoscaler(topology, resource_manager, execution_id)


__all__ = [
    "Autoscaler",
    "DefaultAutoscaler",
    "BackPressureAwareAutoscaler",
    "create_autoscaler",
    "AutoscalingActorPool",
]
