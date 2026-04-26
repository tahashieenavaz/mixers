import jax
from flax import linen as nn
from .MLPBlock import MLPBlock


class MixerBlock(nn.Module):
    token_dimension: int
    hidden_dimension: int
    tokens_mlp_dimension: int
    channels_mlp_dimension: int

    @nn.compact
    def __call__(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = jax.numpy.swapaxes(x, 1, 2)
        x = MLPBlock(
            input_dimension=self.token_dimension, hidden_dimension=self.hidden_dimension
        )(x)
        x = jax.numpy.swapaxes(x, 1, 2)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = MLPBlock(
            input_dimension=self.hidden_dimension,
            hidden_dimension=self.channels_mlp_dimension,
        )(x)
        x = x + residual

        return x
