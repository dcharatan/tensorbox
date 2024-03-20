# `tensorbox`

`tensorbox` allows you to interact with dataclasses of tensors as if they were tensors. Simply use `@tensorbox` instead of `@dataclass`.

```python
from jaxtyping import Float
from tensorbox import tensorbox
from torch import Tensor

# Define a @tensorbox class. The jaxtyping annotations describe each attribute's scalar (unbatched) shape.
@tensorbox
class Gaussians:
    mean: Float[Tensor, "dim"]
    covariance: Float[Tensor, "dim dim"]
    color: Float[Tensor, "3"]

# Define Gaussians with batch size (10, 10) and dim=3.
gaussians = Gaussians(
    torch.zeros((10, 10, 3), dtype=torch.float32),
    torch.zeros((10, 10, 3, 3), dtype=torch.float32),
    torch.zeros((10, 10, 3), dtype=torch.float32),
)

# Define a function that uses Gaussians as input. When a @tensorbox class is subscripted, each attribute's shape becomes the concatenation of the subscript (batch shape) and the attribute's original (scalar) shape. This means fn expects the following shapes:
# - mean: "batch_a batch_b dim"
# - covariances: "batch_a batch_b dim dim"
# - color: "batch_a batch_b 3"
def fn(g: Gaussians["batch_a batch_b"]):
    ...
```

## Features

### Shape Inference

A `@tensorbox` class will automatically infer its batch shape:

```python
@tensorbox
class Camera:
    intrinsics: Float[Tensor, "3 3"]
    extrinsics: Float[Tensor, "4 4"]

cameras = Camera(
    torch.zeros((512, 4, 3, 3), dtype=torch.float32),
    torch.zeros((512, 4, 4, 4), dtype=torch.float32),
)

cameras.shape  # (512, 4)
```

### Nested Tensorboxes

You can define and use nested `@tensorbox` classes as follows:

```python
@tensorbox
class Leaf:
    rgb: Float[Tensor, "3"]
    scale: Float[Tensor, ""]

@tensorbox
class Tree:
    pair: Leaf["2"]

def fn(tree: Tree["*batch"]):
    # tree.pair.rgb has shape (*batch, 2, 3)
    ...
```

### Interaction with PyTorch

`@tensorbox` classes can be used directly with the following `torch` functions:

- `torch.cat`
- `torch.stack`

Note that `dim` arguments are always specified relative to the `@tensorbox` class's batch shape.

## Comparison with TensorDict

`tensorbox` is very similar to [TensorDict](https://github.com/pytorch/tensordict), but has a few key differences:

- It's compatible with `jaxtyping` annotations.
- It's not as feature-complete.
- When creating a tensorbox class instance, you don't have to specify the batch shape—it's automatically inferred.
