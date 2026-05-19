# MLP Mixer

An unofficial PyTorch implementation of the MLP-Mixer architecture proposed by *Tolstikhin et al.*, based on the official JAX implementation.

This package provides a simple and flexible interface for experimenting with all-MLP vision models.

## Features

* PyTorch implementation of MLP-Mixer
* Simple and customizable architecture
* Lightweight and easy to integrate

## Installation

You can install this package using pip by running the following command:

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

print(classifier(images))  # torch.Size([1, 10])
```

## Parameters

* `num_classes`: Number of output classes
* `num_blocks`: Number of mixer layers
* `hidden_dimension`: Embedding dimension
* `tokens_mlp_dimension`: Token-mixing MLP size
* `channels_mlp_dimension`: Channel-mixing MLP size
* `patch_size`: Size of image patches
* `image_size`: Input image resolution

## Reference

Paper: http://arxiv.org/abs/2105.01601

## Citation

```bibtex
@misc{2105.01601,
	Title = {MLP-Mixer: An all-MLP Architecture for Vision},
	Author = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Andreas Steiner and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
	Year = {2021},
	Eprint = {arXiv:2105.01601},
}
```

## License

MIT License
