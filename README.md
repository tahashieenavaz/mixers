# MLP Mixer

The unofficial implementation of MLP Mixer by Tolstikhin, Houlsby, Kolesnikov, Beyer et all based on the official JAX implementation. 

## Installation

You can install this package using pip simply by running following command.

```
pip install mixers
```

## Usage

```py
import torch
from mixers import MLPMixer

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

## Citation


@misc{tolstikhin_mlp-mixer_2021,
	title = {{MLP}-{Mixer}: {An} all-{MLP} {Architecture} for {Vision}},
	shorttitle = {{MLP}-{Mixer}},
	url = {http://arxiv.org/abs/2105.01601},
	doi = {10.48550/arXiv.2105.01601},
	urldate = {2025-11-20},
	publisher = {arXiv},
	author = {Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
	month = jun,
	year = {2021},
	note = {arXiv:2105.01601 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Computer Science - Artificial Intelligence},
}
