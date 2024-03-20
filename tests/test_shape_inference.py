import torch
from jaxtyping import Float
from torch import Tensor

from tensorbox import tensorbox


def test_basic_shape_inference():

    @tensorbox
    class Example:
        a: Float[Tensor, "_ 3"]

    assert Example(torch.zeros((1, 2, 3, 10, 3))).shape == (1, 2, 3)


def test_scalar_shape_inference():

    @tensorbox
    class Example:
        a: Float[Tensor, ""]

    assert Example(torch.zeros((100, 100))).shape == (100, 100)


def test_nested_shape_inference():

    @tensorbox
    class A:
        a: Float[Tensor, "_ 3"]

    @tensorbox
    class B:
        a: A["2 2"]

    box = B(A(torch.zeros((100, 2, 2, 5, 3))))

    assert box.shape == (100,)
    assert box.a.shape == (100, 2, 2)


def test_twice_nested_shape_inference():

    @tensorbox
    class A:
        a: Float[Tensor, ""]

    @tensorbox
    class B:
        a: A["2 2"]

    @tensorbox
    class C:
        b: B["3"]

    box = C(B(A(torch.zeros((100, 100, 3, 2, 2)))))

    assert box.shape == (100, 100)
    assert box.b.shape == (100, 100, 3)
    assert box.b.a.shape == (100, 100, 3, 2, 2)
