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
