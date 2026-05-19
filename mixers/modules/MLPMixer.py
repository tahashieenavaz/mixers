import torch
import einops
from typing import Tuple, Union
from .MixerBlock import MixerBlock
from .HeadModule import HeadModule


class MLPMixer(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        num_classes: int,
        num_blocks: int,
        hidden_dimension: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.sequence_length = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )

        self.patch_embedding = torch.nn.Conv2d(
            in_channels, hidden_dimension, kernel_size=patch_size, stride=patch_size
        )

        self.blocks = torch.nn.ModuleList(
            [
                MixerBlock(
                    sequence_length=self.sequence_length,
                    channels=hidden_dimension,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_normalization = torch.nn.LayerNorm(hidden_dimension)

        if self.num_classes > 0:
            self.head = HeadModule(hidden_dimension=hidden_dimension)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(inputs)
        x = einops.rearrange(x, "n c h w -> n (h w) c")

        for block in self.blocks:
            x = block(x)

        x = self.final_normalization(x)
        x = self.aggregate(x)

        if self.num_classes > 0:
            x = self.head(x)

        return x
