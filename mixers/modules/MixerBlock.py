import torch
from .MLPBlock import MLPBlock


class MixerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        channels: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        super().__init__()
        self.abel = torch.nn.LayerNorm(channels)
        self.cain = torch.nn.LayerNorm(channels)
        self.token_mixing = MLPBlock(sequence_length, tokens_mlp_dim)
        self.channel_mixing = MLPBlock(channels, channels_mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # token mixing
        x = self.abel(x)
        x = x.transpose(1, 2)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)
        x = residual + x

        # channel mixing
        residual = x
        x = self.cain(x)
        x = self.channel_mixing(x)
        return residual + x
