import pytest
import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from tensorbox import tensorbox


def test_basic_function_pass():
    @tensorbox
    class Box:
        a: Float[Tensor, ""]
        b: Int[Tensor, " dim"]
        c: Bool[Tensor, "3 dim"]

    @jaxtyped(typechecker=typechecker)
    def fn(box: Box["batch 10"]):
        pass

    box = Box(
        torch.zeros((5, 10), dtype=torch.float32),
        torch.zeros((5, 10, 5), dtype=torch.int64),
        torch.zeros((5, 10, 3, 5), dtype=torch.bool),
    )
    fn(box)


def test_recursive_function_pass():
    @tensorbox
    class A:
        a: Int[Tensor, "dim 3"]
        b: Float[Tensor, ""]

    @tensorbox
    class B:
        a: A["2"]
        b: A[""]
        c: Bool[Tensor, "3 dim"]

    @jaxtyped(typechecker=typechecker)
    def fn(b: B[" *batch"]):
        pass

    BATCH = (1, 2, 3)
    DIM = 7
    box = B(
        A(
            torch.zeros((*BATCH, 2, DIM, 3), dtype=torch.int64),
            torch.zeros((*BATCH, 2), dtype=torch.float32),
        ),
        A(
            torch.zeros((*BATCH, DIM, 3), dtype=torch.int64),
            torch.zeros(BATCH, dtype=torch.float32),
        ),
        torch.zeros((*BATCH, 3, DIM), dtype=torch.bool),
    )
    fn(box)


def test_basic_function_fail_due_to_batch():
    with pytest.raises(Exception):

        @tensorbox
        class Box:
            a: Float[Tensor, ""]
            b: Int[Tensor, " dim"]
            c: Bool[Tensor, "3 dim"]

        @jaxtyped(typechecker=typechecker)
        def fn(box: Box["batch 10"]):
            pass

        box = Box(
            torch.zeros((5, 9), dtype=torch.float32),
            torch.zeros((5, 10, 5), dtype=torch.int64),
            torch.zeros((5, 10, 3, 5), dtype=torch.bool),
        )
        fn(box)


def test_basic_function_fail_due_to_scalar():
    with pytest.raises(Exception):

        @tensorbox
        class Box:
            a: Float[Tensor, ""]
            b: Int[Tensor, " dim"]
            c: Bool[Tensor, "3 dim"]

        @jaxtyped(typechecker=typechecker)
        def fn(box: Box["batch 10"]):
            pass

        box = Box(
            torch.zeros((5, 10), dtype=torch.float32),
            torch.zeros((5, 10, 4), dtype=torch.int64),
            torch.zeros((5, 10, 3, 5), dtype=torch.bool),
        )
        fn(box)


def test_recursive_function_fail_due_to_nested_scalar():
    with pytest.raises(Exception):

        @tensorbox
        class A:
            a: Int[Tensor, "dim 3"]
            b: Float[Tensor, ""]

        @tensorbox
        class B:
            a: A["2"]
            b: A[""]
            c: Bool[Tensor, "3 dim"]

        @jaxtyped(typechecker=typechecker)
        def fn(b: B[" *batch"]):
            pass

        BATCH = (1, 2, 3)
        DIM = 7
        box = B(
            A(
                torch.zeros((*BATCH, 2, DIM, 2), dtype=torch.int64),
                torch.zeros((*BATCH, 2), dtype=torch.float32),
            ),
            A(
                torch.zeros((*BATCH, DIM, 3), dtype=torch.int64),
                torch.zeros(BATCH, dtype=torch.float32),
            ),
            torch.zeros((*BATCH, 3, DIM), dtype=torch.bool),
        )
        fn(box)
