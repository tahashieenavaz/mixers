# MLP Mixer

The unofficial implementation of MLP Mixer by Tolstikhin, Houlsby, Kolesnikov, Beyer et all based on the official JAX implementation. 

## Installation

You can install this package using pip simply by running following command.

```
pip install mlpmixer
```

## Usage

```py
import torch
from mlpmixer import MLPMixer

images = torch.randn(1, 3, 224, 224)
classifier = MLPMixer(
        num_classes = 10,
        num_blocks = 5,
        hidden_dimension = 512,
        tokens_mlp_dimension = 128,
        channels_mlp_dimension = 128,
        patch_size = 16,
        image_size = 224
)

print(classifier(images))
```