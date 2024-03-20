import inspect
import re
from dataclasses import dataclass, fields
from inspect import get_annotations
from typing import Any, Callable, TypeVar, dataclass_transform

import jaxtyping
import torch
from beartype.door import is_bearable
from jaxtyping import AbstractArray
from jaxtyping._array_types import _anonymous_variadic_dim, _NamedVariadicDim
from torch import Tensor

T = TypeVar("T", bound=type)

TENSORBOX_CLASS_VARIABLE = "__tensorbox__"
RESERVED_NAMES = (
    TENSORBOX_CLASS_VARIABLE,
    "__class_getitem__",
    "__torch_function__",
    "shape",
)


def is_tensorbox(cls: type) -> bool:
    """Check if a class is a tensorbox."""
    return getattr(cls, TENSORBOX_CLASS_VARIABLE, False)


# Since @tensorbox is dataclass-like, there's no actual tensorbox type. This is just a
# suggestively named Any.
TensorBox = Any


def convert_dim(x: TensorBox, dim: int, extra: int = 0) -> int:
    """Convert negative dims to be relative to the tensorbox's batch shape. The "extra"
    parameter is used for operations like torch.stack where the dim can be one larger
    than usual because it refers to slots around dims, not dims themselves.
    """
    maximum = len(x.shape) + extra
    absolute = dim if dim >= 0 else maximum + dim
    if not (0 <= absolute < maximum):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-maximum}, "
            f"{maximum - 1}], but got {dim})"
        )
    return absolute


def combine_tensorbox(fn: Callable, extra: int) -> Callable:
    def wrapped(
        tensors: list[TensorBox] | tuple[TensorBox],
        dim: int = 0,
        out: Any = None,
    ) -> TensorBox:
        if out is not None:
            raise Exception('tensorbox does not support the "out" parameter')

        # It should be impossible to have no tensors, since then __torch_function__
        # wouldn't be called.
        assert len(tensors) > 0

        # Ensure that the model is actually a tensorbox.
        model = tensors[0]
        assert is_tensorbox(model)

        # Combine each field separately.
        dim = convert_dim(model, dim, extra=extra)
        items = {
            name: fn([getattr(x, name) for x in tensors], dim=dim)
            for name in model.__annotations__.keys()
        }
        return model.__class__(**items)

    return wrapped


TORCH_FUNCTIONS = {
    torch.cat: combine_tensorbox(torch.cat, 0),
    torch.stack: combine_tensorbox(torch.stack, 1),
}


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


def _ensure_compatibility(cls: T) -> None:
    for name, annotation in get_annotations(cls).items():
        # Ensure that the class doesn't have any of the forbidden names defined.
        if name in RESERVED_NAMES:
            raise AttributeError(
                f'A @tensorbox class cannot have an instance variable named "{name}"'
            )

        # Ensure that only specializations are used as annotations.
        if is_tensorbox(annotation):
            raise AttributeError(
                f'The annotation "{name}" is a non-specialized tensorbox. Did you mean '
                f'to use {name}[""]?'
            )

        # Ensure that the class only has jaxtyping or tensorbox annotations defined.
        if not (
            issubclass(annotation, AbstractArray)
            or issubclass(annotation, Specialization)
        ):
            raise AttributeError(
                f'The annotation "{name}" is not a jaxtyping annotation or a '
                "@tensorbox class. A @tensorbox class's instance variables must either "
                "be jaxtyping-annotated Torch tensors or other @tensorbox classes."
            )

        # Ensure that only Torch tensors are used.
        if issubclass(annotation, AbstractArray):
            if annotation.array_type != Tensor:
                raise AttributeError(
                    f"The annotation {name} is not a Torch tensor. A @tensorbox "
                    "class's instance variables must either be jaxtyping-annotated "
                    "Torch tensors or other @tensorbox classes."
                )

        # Ensure that any array annotations have fixed shapes.
        if issubclass(annotation, AbstractArray):
            for dim in annotation.dims:
                if dim is _anonymous_variadic_dim or isinstance(dim, _NamedVariadicDim):
                    raise AttributeError(
                        "Variadic annotations are not allowed in @tensorbox classes."
                    )

        # Ensure that no defaults are provided.
        if hasattr(cls, name):
            raise AttributeError(
                f"The annotation for {name} has a default specified. @tensorbox "
                "classes must not have defaults specified."
            )


def _specialize(cls: type, leaf_fn: Callable[[type], type]) -> type:
    """Derive a class from the tensorbox in which the arbitrary batch shape (*batch) has
    been replaced with the specified batch_shape.
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

    # Add an attribute so that we can detect tensorbox classes.
    setattr(cls, TENSORBOX_CLASS_VARIABLE, True)

    # Re-write all of the jaxtyping annotations to include a *batch dimension. This is
    # because jaxtyping annotations used with tensorbox are assumed to be for scalar
    # (non-batched) types.
    _transform_tensorbox(cls, _transform_dim_str(lambda x: f"*batch {x}"))

    def specialize(batch_shape: str):
        """This is called when a tensorbox class is used as an annotation. It re-writes
        all of the tensorbox's jaxtyping annotations to include the desired batch size.
        """
        return _specialize(
            cls, _transform_dim_str(lambda x: x.replace("*batch", batch_shape))
        )

    def get_shape(self) -> tuple[int, ...]:
        """Return only the batch (common) shape."""
        assert cls == self.__class__
        (name, annotation), *_ = cls.__annotations__.items()

        if issubclass(annotation, AbstractArray):
            # Deduce the batch shape by chopping off the fixed shape in the annotation.
            if len(annotation.dims) == 0:
                # This special case is necessary because -0 is 0, which doesn't index
                # from the right.
                return getattr(self, name).shape
            else:
                return getattr(self, name).shape[: -len(annotation.dims)]

        if issubclass(annotation, Specialization):
            # Traverse nested specializations.
            value = getattr(self, name)
            while issubclass(annotation, Specialization):
                name, annotation = next(iter(annotation.__annotations__.items()))
                value = getattr(value, name)

            # The final nested value's shape will be the following:
            # (*batch, *(spec. 1 shape), ..., *(spec. n shape), *(scalar shape))
            # The specialization and scalar shapes are non-variadic, so we can simply
            # chop them off to get the batch shape.
            return value.shape[: 1 - len(annotation.dims)]

        raise AttributeError(f'@tensorbox class contains invalid annotation "{name}"')

    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """Allow tensorbox classes to be used with PyTorch functions as if they were
        PyTorch tensors.
        """
        handler = TORCH_FUNCTIONS.get(func, None)
        if handler is None:
            raise Exception(
                f"tensorbox does not support the PyTorch {func.__name__} function."
            )
        return TORCH_FUNCTIONS[func](*args, **kwargs)

    # These are the methods @tensorbox adds to the class.
    cls.__class_getitem__ = specialize
    cls.__torch_function__ = classmethod(__torch_function__)
    cls.shape = property(get_shape)

    return cls
