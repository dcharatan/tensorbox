import numpy as np
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from tensorbox import RESERVED_NAMES, tensorbox


@pytest.mark.parametrize("name", RESERVED_NAMES)
def test_reserved_name(name):
    with pytest.raises(AttributeError):
        example = type("Example", (), {})
        example.__annotations__ = {name: Float[Tensor, "3 3"]}
        tensorbox(example)


def test_anonymous_variadic_annotation():
    with pytest.raises(AttributeError):

        @tensorbox
        class Example:
            variadic: Float[Tensor, "... 3"]


def test_named_variadic_annotation():
    with pytest.raises(AttributeError):

        @tensorbox
        class Example:
            variadic: Float[Tensor, " *batch"]


@pytest.mark.parametrize(
    "annotation_type", (int, str, np.ndarray, Tensor, Float[np.ndarray, "3 3"])
)
def test_incorrect_annotation_type(annotation_type):
    with pytest.raises(AttributeError):

        @tensorbox
        class Example:
            bad: annotation_type


def test_default_argument():
    with pytest.raises(AttributeError):

        @tensorbox
        class Example:
            bad: Float[Tensor, "3 3"] = torch.zeros((3, 3), dtype=torch.float32)


def test_valid_tensorbox():
    @tensorbox
    class ExampleA:
        a: Float[Tensor, "3 3"]

    @tensorbox
    class ExampleB:
        a: ExampleA["_ 3"]
        b: Float[Tensor, "3 3"]
