import torch
from .MLPBlock import MLPBlock

class MixerBlock(torch.nn.Module):
    def __init__(self, *, token_dimension: int, hidden_dimension: int, tokens_mlp_dimension: int, channels_mlp_dimension: int):
        super().__init__()
        self.token_mlp = MLPBlock(token_dimension, tokens_mlp_dimension)
        self.channel_mlp = MLPBlock(hidden_dimension, channels_mlp_dimension)
        self.alpha = torch.nn.LayerNorm(hidden_dimension)
        self.beta = torch.nn.LayerNorm(hidden_dimension)

    def forward(self, x):
        residual = x
        x = self.alpha(x).transpose(1, 2)
        x = self.token_mlp(x).transpose(1, 2)

        x = residual + x

        residual = x
        x = self.beta(x)
        x = self.channel_mlp(x)

        return residual + x