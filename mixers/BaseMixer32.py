from .modules import MLPMixer


class BaseMixer32(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=12,
            hidden_dimension=768,
            channels_mlp_dimension=3072,
            tokens_mlp_dimension=384,
            patch_size=32,
            image_channels=3,
        )
