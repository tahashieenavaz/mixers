from .modules import MLPMixer


class LargeMixer16(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=24,
            hidden_dimension=1024,
            channels_mlp_dimension=4096,
            tokens_mlp_dimension=512,
            patch_size=16,
            image_channels=3,
        )
