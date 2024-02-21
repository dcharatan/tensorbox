import inspect
import re
from dataclasses import dataclass, fields
from inspect import get_annotations
from typing import Any, Callable, TypeVar, dataclass_transform

import jaxtyping
from beartype.door import is_bearable
from jaxtyping import AbstractArray

T = TypeVar("T", bound=type)

TENSORBOX_CLASS_VARIABLES = ("__tensorbox__",)


class SpecializationMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        for key, value in inspect.get_annotations(self).items():
            # Check if the key is missing.
            if not hasattr(instance, key):
                return False

            # Check if the type is wrong.
            if not is_bearable(getattr(instance, key), value):
                return False

        return True


class Specialization(metaclass=SpecializationMeta):
    pass


def is_tensorbox(cls: type) -> bool:
    """Check if a class is a tensorbox."""
    return getattr(cls, "__tensorbox__", False)


def _ensure_compatibility(cls: T) -> None:
    # TODO: Ensure that there are no defaults.
    # TODO: Ensure that jaxtyping annotations don't have "batch" anywhere.

    for name, annotation in get_annotations(cls).items():
        # Ensure that the class doesn't have any of the forbidden names defined.
        if name in TENSORBOX_CLASS_VARIABLES:
            raise AttributeError(
                f'A @tensorbox class cannot have an instance variable named "{name}"'
            )

        # Ensure that the class only has jaxtyping or tensorbox annotations defined. One
        # could intentionally fool this using a protocol that isn't a tensorbox
        # specialization, but that probably doesn't matter.
        if not (
            issubclass(annotation, AbstractArray)
            or is_tensorbox(annotation)
            or issubclass(annotation, Specialization)
        ):
            raise AttributeError(
                f'The instance variable "{name}" is not a jaxtyping annotation or a '
                "@tensorbox class. A  @tensorbox class's instance variables must "
                "either be jaxtyping-annotated tensors or other @tensorbox classes."
            )


def _specialize(cls: type, leaf_fn: Callable[[type], type]) -> type:
    """Derive a typing.Protocol from the tensorbox in which the arbitrary batch shape
    (*batch) has been replaced with the specified batch_shape.
    """

    annotations = {}

    # Handle leaves (jaxtyping annotations).
    if issubclass(cls, AbstractArray):
        return leaf_fn(cls)

    # Recurse on tensorbox classes.
    elif is_tensorbox(cls):
        for field in fields(cls):
            annotations[field.name] = _specialize(field.type, leaf_fn)

    # Recurse on tensorbox specializations.
    else:
        for name, annotation in cls.__annotations__.items():
            annotations[name] = _specialize(annotation, leaf_fn)

    # Generate the specialization type.
    specialization = type(
        f"{cls.__name__}Specialization",
        (Specialization,),
        annotations,
    )

    # Add type annotations to the specialization. These are used to traverse the
    # specialization and carry out __isinstance__ checks.
    specialization.__annotations__ = annotations

    return specialization


def _transform_tensorbox(cls: type, leaf_fn: Callable[[type], type]) -> type:
    """Transform all of the tensorbox's jaxtyping annotations (leaf nodes) according to
    leaf_fn.
    """

    if issubclass(cls, AbstractArray):
        return leaf_fn(cls)
    elif is_tensorbox(cls):
        for field in fields(cls):
            field.type = _transform_tensorbox(field.type, leaf_fn)
        return cls
    else:
        for key, value in cls.__annotations__.items():
            cls.__annotations__[key] = _transform_tensorbox(value, leaf_fn)
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


@dataclass_transform()
def tensorbox(cls: T) -> T:
    # Ensure that the provided class is compatible with tensorbox, mainly in order to
    # emit helpful errors when the user uses tensorbox incorrectly.
    _ensure_compatibility(cls)

    # Tensorbox uses many of the mechanisms dataclass provides.
    cls = dataclass(cls)

    # Add a __tensorbox__ attribute so that we can detect @tensorbox classes.
    cls.__tensorbox__ = True

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

    return cls
