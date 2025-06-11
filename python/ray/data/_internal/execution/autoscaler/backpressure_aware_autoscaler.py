from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict

import math
import time

import ray

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState
    from ray.data._internal.execution.streaming_executor_state import Topology

from .autoscaler import Autoscaler
from .autoscaling_actor_pool import AutoscalingActorPool
from ray.data._internal.execution.autoscaling_requester import get_or_create_autoscaling_requester_actor
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources


class BackPressureAwareAutoscaler(Autoscaler):
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 20
    MIN_GAP_BETWEEN_ACTOR_POOL_SCALE_UPS = 5
    MAX_OUTPUT_QUEUE_SIZE = 5

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        execution_id: str,
    ):
        print("INITIALIZE BackPressureAwareAutoscaler!!!!!!!!!!!!!")

        # Last time when a request was sent to Ray's autoscaler.
        self._last_request_time = 0
        self._last_actor_pool_scale_up_time = 0
        self._operators_with_terminated_scaling = set()

        super().__init__(topology, resource_manager, execution_id)

    def try_trigger_scaling(self):
        self._try_scale_up_cluster()
        self._try_scale_up_actor_pool()

    def log_operator_metrics(self, op, op_state) -> None:
        metrics = op.metrics
        print(f"In Queue size: {sum(len(q) for q in op_state.inqueues)}, [{op.name}]")
        print(f"Out Queue size: {len(op_state.outqueue)}, [{op.name}]\n")

    def _actor_pool_should_scale_up(
        self,
        actor_pool: AutoscalingActorPool,
        op: "PhysicalOperator",
        op_state: "OpState",
    ):
        if op.name in self._operators_with_terminated_scaling:
            return False

        if actor_pool.current_size() < actor_pool.min_size():
            # Scale up, if the actor pool is below min size.
            return True
        elif actor_pool.current_size() >= actor_pool.max_size():
            # Do not scale up, if the actor pool is already at max size.
            return False

        # Do not scale up, if the op does not have more resources.
        if not op_state._scheduling_status.under_resource_limits:
            print(f"OUT OF RESOURCES, {op.name}")
            return False
        # Do not scale up, if the op has enough free slots for the existing inputs.
        if op_state.num_queued() <= actor_pool.num_free_task_slots():
            print(f"ALREADY ENOUGH FREE SLOTS, {op.name}")
            return False

        now = time.time()
        if now - self._last_actor_pool_scale_up_time < self.MIN_GAP_BETWEEN_ACTOR_POOL_SCALE_UPS:
            return False

        # Do not scale up, if the op is completed or no more inputs are coming.
        if op.completed() or (op._inputs_complete and op.internal_queue_size() == 0):
            print(f"IN QUEUE IS EMPTY, {op.name}")
            return False

        # Determine whether to scale up based on the outqueue size
        outqueue_size = len(op_state.outqueue)
        should_scale_up = outqueue_size < self.MAX_OUTPUT_QUEUE_SIZE
        print(f"SCALE UP DECISION: {should_scale_up} , because outqueue size is {outqueue_size}")

        if should_scale_up:
            print(f"Scaling up, [{op.name}] has {actor_pool.current_size()} actors")
            self.log_operator_metrics(op, op_state)
            return True
        else:
            self._operators_with_terminated_scaling.add(op.name)
            print(f"Scaling has stopped for: [{op.name}], fixed at {actor_pool.current_size()} actors.")
            return False

    def _try_scale_up_actor_pool(self):
        for op, state in self._topology.items():
            actor_pools = op.get_autoscaling_actor_pools()
            for actor_pool in actor_pools:
                while True:
                    # Try to scale up or down the actor pool.
                    should_scale_up = self._actor_pool_should_scale_up(
                        actor_pool,
                        op,
                        state,
                    )
                    if should_scale_up:
                        now = time.time()
                        print("UPDATE LAST TIME")
                        self._last_actor_pool_scale_up_time = now

                        if actor_pool.scale_up(1) == 0:
                            break
                    else:
                        break

    def _try_scale_up_cluster(self):
        """Try to scale up the cluster to accomodate the provided in-progress workload.
        This makes a resource request to Ray's autoscaler consisting of the current,
        aggregate usage of all operators in the DAG + the incremental usage of all
        operators that are ready for dispatch (i.e. that have inputs queued). If the
        autoscaler were to grant this resource request, it would allow us to dispatch
        one task for every ready operator.
        Note that this resource request does not take the global resource limits or the
        liveness policy into account; it only tries to make the existing resource usage
        + one more task per ready operator feasible in the cluster.
        """
        # Limit the frequency of autoscaling requests.
        now = time.time()
        if now - self._last_request_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            return

        # # Scale up the cluster, if no ops are allowed to run, but there are still data
        # # in the input queues.
        # no_runnable_op = all(op_state._scheduling_status.runnable is False for _, op_state in self._topology.items())
        # any_has_input = any(op_state.num_queued() > 0 for _, op_state in self._topology.items())
        # if not (no_runnable_op and any_has_input):
        #     return

        self._last_request_time = now

        # Get resource usage for all ops + additional resources needed to launch one
        # more task for each ready op.
        resource_request = []

        def to_bundle(resource: ExecutionResources) -> Dict:
            req = {}
            if resource.cpu:
                req["CPU"] = math.ceil(resource.cpu)
            if resource.gpu:
                req["GPU"] = math.ceil(resource.gpu)
            return req

        for op, state in self._topology.items():
            per_task_resource = op.incremental_resource_usage()
            task_bundle = to_bundle(per_task_resource)
            resource_request.extend([task_bundle] * op.num_active_tasks())
            # Only include incremental resource usage for ops that are ready for
            # dispatch.
            if state.num_queued() > 0:
                # TODO(Clark): Scale up more aggressively by adding incremental resource
                # usage for more than one bundle in the queue for this op?
                resource_request.append(task_bundle)

        self._send_resource_request(resource_request)

    def _send_resource_request(self, resource_request):
        # Make autoscaler resource request.
        actor = get_or_create_autoscaling_requester_actor()
        actor.request_resources.remote(resource_request, self._execution_id)

    def on_executor_shutdown(self):
        # Make request for zero resources to autoscaler for this execution.
        actor = get_or_create_autoscaling_requester_actor()
        actor.request_resources.remote({}, self._execution_id)

    def get_total_resources(self) -> ExecutionResources:
        return ExecutionResources.from_resource_dict(ray.cluster_resources())