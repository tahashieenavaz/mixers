from .modules import MLPMixer


class HugeMixer14(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=32,
            hidden_dimension=1280,
            channels_mlp_dimension=5120,
            tokens_mlp_dimension=640,
            patch_size=14,
            image_channels=3,
        )
