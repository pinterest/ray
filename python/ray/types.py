from typing import TYPE_CHECKING, TypeVar
from ray.util.annotations import PublicAPI




# TODO(ekl) this is a dummy generic ref type for documentation purposes only.
# We should try to make the Cython ray.ObjectRef properly generic.
# NOTE(sang): Looks like using Generic in Cython is not currently possible.
# We should update Cython > 3.0 for this.

T = TypeVar("T")

if TYPE_CHECKING:
    from ray._raylet import ObjectRef as ObjectRef  # for static type checking
else:
    from ray._raylet import ObjectRef as ObjectRef  # for runtime usage