import jax
from flax import linen as nn
from typing import Type


class MLPBlock(nn.Module):
    input_dimension: int
    hidden_dimension: int
    activation: Type[nn.Module]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dimension)(x)
        x = self.activation(x)
        x = nn.Dense(self.input_dimension)(x)
        return x
