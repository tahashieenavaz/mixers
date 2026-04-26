from .modules import MLPMixer


class Small32Mixer(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=8,
            hidden_dimension=512,
            channels_mlp_dimension=2048,
            tokens_mlp_dimension=256,
            patch_size=16,
            image_channels=3,
        )
