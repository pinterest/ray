import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from ray.autoscaler.v2.instance_manager.node_provider import (
    CloudInstance,
    CloudInstanceId,
    CloudInstanceProviderError,
    ICloudInstanceProvider,
)
from ray.core.generated.autoscaler_pb2 import ClusterResourceState
from ray.core.generated.instance_manager_pb2 import Instance as IMInstance
from ray.core.generated.instance_manager_pb2 import (
    InstanceUpdateEvent as IMInstanceUpdateEvent,
)

logger = logging.getLogger(__name__)


class IReconciler(ABC):
    @staticmethod
    @abstractmethod
    def reconcile(*args, **kwargs) -> List[IMInstanceUpdateEvent]:
        """
        Reconcile the status of Instance Manager(IM) instances to a valid state.
        It should yield a list of instance update events that will be applied to the
        instance storage by the IM.

        There should not be any other side effects other than producing the instance
        update events. This method is expected to be idempotent and deterministic.

        Returns:
            A list of instance update event.

        """
        pass


class StuckInstanceReconciler(IReconciler):
    @staticmethod
    def reconcile(
        instances: List[IMInstance],
    ) -> List[IMInstanceUpdateEvent]:
        """
        ***********************
        **STUCK STATES
        ***********************
        For instances that's stuck in the following states, we will try to recover
        them:

        - REQUESTED -> QUEUED
            WHEN: happens when the cloud instance provider is not able to allocate
                cloud instances, and the instance is in REQUESTED state for longer than
                the timeout.
            ACTION: We will retry the allocation by setting it back to QUEUED.

        - ALLOCATED -> STOPPING
            WHEN: happens when the cloud instance provider is responsible for
                installing and running ray on the cloud instance, and the cloud instance
                is not able to start ray for some reason.
            ACTION: We will terminate the instances since it's a leak after a timeout
                for installing/running ray, setting it to STOPPING.

        - RAY_INSTALLING -> RAY_INSTALL_FAILED
            WHEN: this happens when the cloud instance provider is responsible for
                installing ray on the cloud instance, and the cloud instance is not
                able to start ray for some reason (e.g. immediate crash after ray start)
            ACTION: We will fail the installation and terminate the instance after a
                timeout, transitioning to RAY_INSTALL_FAILED.

        # TODO(rickyx): Maybe we should add a TO_STOP status so we could do retry
        # stopping better
        - STOPPING -> STOPPING
            WHEN: the cloud node provider taking long or failed to terminate the cloud
                instance.
            ACTION: We will retry the termination by setting it to STOPPING, triggering
                another termination request to the cloud instance provider.

        ***********************
        **TRANSIENT STATES
        ***********************
        These states are expected to be transient, when they don't, it usually indicates
        a bug in the instance manager or the cloud instance provider as the process loop
        which is none-blocking is now somehow not making progress.

        # TODO(rickyx): When we added the timeout for draining, we should also add
        # timeout for the RAY_STOPPING states.
        - RAY_STOPPING:
            WHEN: this state should be transient, instances in this state are being
                drained through the drain protocol (and the drain request should
                eventually finish).
            ACTION: There's no drain request timeout so we will wait for the drain to
                finish, and print errors if it's taking too long.

        - RAY_INSTALL_FAILED or RAY_STOPPED
            WHEN: these state should be transient, cloud instances should be terminating
                or terminated soon. This could happen if the subscribers for the status
                transition failed to act upon the status update.
            ACTION: Print errors if stuck for too long - this is a bug.

        - QUEUED
            WHEN: if there's many instances queued to be launched, and we limit the
                number of concurrent instance launches, the instance might be stuck
                in QUEUED state for a long time.
            ACTION: No actions, print warnings if needed (or adjust the scaling-up
                configs)

        ***********************
        **LONG LASTING STATES
        ***********************
        These states are expected to be long lasting. So we will not take any actions.

        - RAY_RUNNING:
            Normal state - unlimited time.
        - STOPPED:
            Normal state - terminal, and unlimited time.
        - ALLOCATION_FAILED:
            Normal state - terminal, and unlimited time.

        """
        pass


class RayStateReconciler(IReconciler):
    @staticmethod
    def reconcile(
        ray_cluster_resource_state: ClusterResourceState, instances: List[IMInstance]
    ) -> List[IMInstanceUpdateEvent]:
        """
        Reconcile the instances states for Ray node state changes.

        1. Newly running ray nodes -> RAY_RUNNING status change.
            When an instance in (ALLOCATED or RAY_INSTALLING) successfully launched the
            ray node, we will discover the newly launched ray node and transition the
            instance to RAY_RUNNING.

        2. Newly dead ray nodes -> RAY_STOPPED status change.
            When an instance not in RAY_STOPPED is no longer running the ray node, we
            will transition the instance to RAY_STOPPED.
        """
        pass


class CloudProviderReconciler(IReconciler):
    @staticmethod
    def reconcile(
        provider: ICloudInstanceProvider, instances: List[IMInstance]
    ) -> List[IMInstanceUpdateEvent]:
        """
        Reconcile the instance storage with the node provider.
        This is responsible for transitioning the instance status of:
        1. to ALLOCATED: when a REQUESTED instance could be assigned to an unassigned
            cloud instance.
        2. to STOPPED: when an ALLOCATED instance no longer has the assigned cloud
            instance found in node provider.
        3. to ALLOCATION_FAILED: when a REQUESTED instance failed to be assigned to
            an unassigned cloud instance.
        4. to STOPPING: when an instance being terminated fails to be terminated
            for some reasons, we will retry the termination.
        """
        pass

    @staticmethod
    def _reconcile_allocated(
        non_terminated_cloud_nodes: Dict[CloudInstanceId, CloudInstance]
    ) -> Dict[str, IMInstanceUpdateEvent]:

        """
        For any REQUESTED instances, if there's any unassigned non-terminated
        cloud instance that matches the instance type from the same request,
        assign it and transition it to ALLOCATED.

        """

        # Find all requested instances, by launch request id.

        # Find all non_terminated cloud_nodes by launch request id.

        # For the same request, find any unassigned non-terminated cloud instance
        # that matches the instance type. Assign them to REQUESTED instances
        # and transition them to ALLOCATED.
        pass

    @staticmethod
    def _reconcile_failed_to_allocate(
        self, cloud_provider_errors: List[CloudInstanceProviderError]
    ) -> Dict[str, IMInstanceUpdateEvent]:
        """
        For any REQUESTED instances, if there's errors in allocating the cloud instance,
        transition the instance to ALLOCATION_FAILED.
        """

        # Find all requested instances, by launch request id.

        # Find all launch errors by launch request id.

        # For the same request, transition the instance to ALLOCATION_FAILED.
        # TODO(rickyx): we don't differentiate transient errors (which might be
        # retryable) and permanent errors (which are not retryable).
        pass

    @staticmethod
    def _reconcile_stopped(
        non_terminated_cloud_nodes: Dict[CloudInstanceId, CloudInstance]
    ) -> List[IMInstanceUpdateEvent]:
        """
        For any IM (instance manager) instance with a cloud node id, if the mapped
        cloud instance is no longer running, transition the instance to STOPPED.
        """

        # Find all instances with cloud instance id, by cloud instance id.

        # Check if any matched cloud instance is still running.
        # If not, transition the instance to STOPPED. (The instance will have the cloud
        # instance id removed, and GCed later.)
        pass

    @staticmethod
    def _reconcile_failed_to_terminate(
        cloud_provider_errors: List[CloudInstanceProviderError],
    ) -> List[IMInstanceUpdateEvent]:
        """
        For any STOPPING instances, if there's errors in terminating the cloud instance,
        we will retry the termination by setting it to STOPPING again.
        """
        pass