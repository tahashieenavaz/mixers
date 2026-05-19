import torch
from typing import Type
from .modules import MLPMixer


class BaseMixer16(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
        channel_mixing_activation: Type[torch.nn.Module] = torch.nn.GELU,
        token_mixing_activation: Type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=12,
            hidden_dimension=768,
            channels_mlp_dimension=3072,
            tokens_mlp_dimension=384,
            patch_size=16,
            image_channels=3,
            channel_mixing_activation=channel_mixing_activation,
            token_mixing_activation=token_mixing_activation,
        )
