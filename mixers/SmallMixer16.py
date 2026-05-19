from .modules import MLPMixer
from typing import Union, Tuple


class SmallMixer16(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: Union[int, Tuple[int, int]],
    ):
        super().__init__(
            image_channels=3,
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            num_blocks=8,
            hidden_dimension=512,
            channels_mlp_dimension=2048,
            tokens_mlp_dimension=256,
        )
