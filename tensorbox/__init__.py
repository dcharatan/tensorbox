import re
from copy import deepcopy
from dataclasses import dataclass, fields, make_dataclass
from typing import Annotated, Any, TypeVar, Protocol, runtime_checkable

import jaxtyping
from jaxtyping import AbstractArray
from typing import Callable

T = TypeVar("T", bound=type)

TENSORBOX_CLASS_VARIABLES = ("__tensorbox__",)


def is_tensorbox(cls: type) -> bool:
    return getattr(cls, "__tensorbox__", False) in ("base", "specialized")


def _ensure_compatibility(cls: T) -> None:
    # TODO: Ensure that there are no defaults.
    # TODO: Ensure that jaxtyping annotations don't have "batch" anywhere.

    for name, annotation in cls.__annotations__.items():
        # Ensure that the class doesn't have any of the forbidden names defined.
        if name in TENSORBOX_CLASS_VARIABLES:
            raise AttributeError(
                f'A @tensorbox class cannot have an instance variable named "{name}"'
            )

        # Ensure that the class only has jaxtyping annotations defined.
        if not (issubclass(annotation, AbstractArray) or is_tensorbox(annotation)):
            raise AttributeError(
                f'The instance variable "{name}" is not a jaxtyping annotation or a '
                "@tensorbox class. A  @tensorbox class's instance variables must "
                "either be jaxtyping-annotated tensors or other @tensorbox classes."
            )


def _specialize(cls: type, leaf_fn: Callable[[type], type]) -> type:
    """Derive a typing.Protocol from the tensorbox in which the arbitrary batch shape
    (*batch) has been replaced with the specified batch_shape.
    """

    # Recurse on tensorbox children.
    specialized_fields = {
        f.name: (
            _specialize(f.type, leaf_fn) if is_tensorbox(f.type) else leaf_fn(f.type)
        )
        for f in fields(cls)
    }

    # Derive the typing.Protocol, then mark it as being a tensorbox specialization.
    specialized = type(
        f"Specialized{cls.__name__}",
        (Protocol,),
        specialized_fields,
    )
    specialized.__tensorbox__ = "specialized"

    # Allow the type checker to type-check the specialization at runtime.
    return runtime_checkable(specialized)


def _transform_tensorbox(cls: type, leaf_fn: Callable[[type], type]) -> type:
    """Modify the"""

    """Create a transformed copy of the @tensorbox class in which the leaves (jaxtyping
    annotations) have been transformed according to leaf_fn. The leaf_fn should create
    new jaxtyping annotations rather than modifying the existing ones.
    """

    for field in fields(cls):
        if is_tensorbox(field.type):
            _transform_tensorbox(field.type, leaf_fn)
        else:
            field.type = leaf_fn(field.type)

    return cls


def _transform_dim_str(fn: Callable[[str], str]) -> Callable[[type], type]:
    """Generate a leaf_fn for use with _transform_tensorbox. For each leaf (jaxtyping
    annotation), the returned function reads the existing dim_str, modifies it according
    to fn, and then creates a new leaf with the updated dim_str.
    """

    def leaf_fn(cls: type) -> type:
        assert issubclass(cls, AbstractArray)

        # Get the jaxtyping class name.
        match = re.match(r"^(\w+)\[.*\]$", cls.__name__)
        if match is None or len(match.groups()) != 1:
            raise ValueError("Could not parse jaxtyping type.")
        jaxtype_name = match.groups()[0]

        # Exchange the class name for the actual class.
        jaxtype = getattr(jaxtyping, jaxtype_name)

        # Finally, recreate the jaxtyping annotation with a new dim_str.
        return jaxtype[cls.array_type, fn(cls.dim_str)]

    return leaf_fn


def _tensorbox(cls: T) -> T:
    # Ensure that the provided class is compatible with tensorbox, mainly in order to
    # emit helpful errors when the user uses tensorbox incorrectly.
    _ensure_compatibility(cls)

    # Tensorbox uses many of the mechanisms dataclass provides.
    cls = dataclass(cls)

    # Re-write all of the jaxtyping annotations to include a *batch dimension. This is
    # because jaxtyping annotations used with tensorbox are assumed to be for scalar
    # (non-batched) types.
    _transform_tensorbox(cls, _transform_dim_str(lambda x: f"*batch {x}"))

    def specialize(batch_shape: str):
        return _specialize(
            cls, _transform_dim_str(lambda x: x.replace("*batch", batch_shape))
        )

    # This is called when the a tensorbox class is used as an annotation. It re-writes
    # all of the tensorbox's jaxtyping annotations to include the desired batch
    cls.__class_getitem__ = specialize

    # Add a __tensorbox__ attribute so that we can detect @tensorbox classes.
    cls.__tensorbox__ = "base"

    return cls


# This makes VS Code IntelliSense interpret @tensorbox the same way as @dataclass, which
# makes type hints work correctly.
TYPE_CHECKING = False

if TYPE_CHECKING:
    from dataclasses import dataclass as tensorbox
else:
    tensorbox = _tensorbox
