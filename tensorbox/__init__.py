import re
from dataclasses import dataclass, fields
from typing import Annotated, Any, TypeVar

import jaxtyping
from jaxtyping import AbstractArray, Bool, Int64
from torch import Tensor

T = TypeVar("T", bound=type)


RESTRICTED_NAMES = ("shape", "unbind")


def is_tensorbox(cls: type) -> bool:
    return getattr(cls, "__tensorbox__", False)


def _ensure_compatibility(cls: T) -> None:
    for name, annotation in cls.__annotations__.items():
        # Ensure that the class doesn't have any of the forbidden names defined.
        if name in RESTRICTED_NAMES:
            raise AttributeError(
                f'A @tensorbox class cannot have an instance variable named "{name}"'
            )

        # Ensure that the class only has jaxtyping annotations defined.
        if not (issubclass(annotation, AbstractArray) or is_tensorbox(annotation)):
            raise AttributeError(
                f'The instance variable "{name}" is not a jaxtyping annotation. A '
                "@tensorbox class's instance variables must either be "
                "jaxtyping-annotated tensors or other @tensorbox classes."
            )


def _tensorbox(cls: T) -> T:
    _ensure_compatibility(cls)

    cls = dataclass(cls)

    def rewrite_annotations(batch_shape: str) -> Annotated[T, Any]:
        # Use a queue to recurse on @tensorbox fields.
        queue = [cls]

        while queue:
            for field in fields(queue.pop()):
                # Modify jaxtyping annotations to include the batch shape.
                if issubclass(field.type, AbstractArray):
                    # Get the jaxtyping class name.
                    match = re.match(r"^(\w+)\[.*\]$", field.type.__name__)
                    if match is None or len(match.groups()) != 1:
                        raise ValueError("Could not parse jaxtyping type.")
                    jaxtype_name = match.groups()[0]

                    # Exchange the class name for the actual class.
                    jaxtype = getattr(jaxtyping, jaxtype_name)

                    # Finally, recreate the jaxtyping annotation with a new set of dims.
                    combined_shape = f"{batch_shape} {field.type.dim_str}"
                    field.type = jaxtype[field.type.array_type, combined_shape]

                # Recurse on @tensorbox fields.
                if is_tensorbox(field.type):
                    queue.append(field.type)

        return cls

    # This is called when the a tensorbox class is used as an annotation. It re-writes
    # all of the tensorbox's jaxtyping annotations to include the desired batch
    cls.__class_getitem__ = rewrite_annotations

    # Add a __tensorbox__ attribute so that we can detect @tensorbox classes.
    cls.__tensorbox__ = True

    return cls


# This makes VS Code IntelliSense interpret @tensorbox the same way as @dataclass, which
# makes type hints work correctly.
TYPE_CHECKING = False

if TYPE_CHECKING:
    from dataclasses import dataclass as tensorbox
else:
    tensorbox = _tensorbox
