import torch
import einops
from .MixerBlock import MixerBlock

class MLPMixer(torch.nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        num_blocks: int,
        hidden_dimension:int,
        tokens_mlp_dimension: int,
        channels_mlp_dimension: int,
        patch_size: int,
        image_size: int,
        image_channels: int = 3
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_blocks = num_blocks

        self.stem = torch.nn.Conv2d(
            in_channels=image_channels,
            out_channels=hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2

        self.blocks = torch.nn.ModuleList([
            MixerBlock(
                num_tokens=num_patches,
                hidden_dimension=hidden_dimension,
                tokens_mlp_dimension=tokens_mlp_dimension,
                channels_mlp_dimension=channels_mlp_dimension,
            )
            for _ in range(num_blocks)
        ])

        self.normalization = torch.nn.LayerNorm(hidden_dimension)
        self.head = torch.nn.Linear(hidden_dimension, num_classes)

        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')

        for block in self.blocks:
            x = block(x)

        x = self.normalization(x)
        x = x.mean(dim=1)
        return self.head(x)