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

Build your own mixer:

```py
import torch
from mixers.modules import MLPMixer

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

Use predefined, paper mentioned, mixers: 

```py
from mixers import BaseMixer16, BaseMixer32
from mixers import SmallMixer16, SmallMixer32
from mixers import LargeMixer16, LargeMixer32
from mixers import HugeMixer14

model = HugeMixer14(image_size=224, num_classes=10)

train_model(model)
```

## Parameters

| # | Model        | Trainable Parameters      |
|---|--------------|--------------------------:|
| 1 | SmallMixer16 | 18.020394                 |
| 2 | SmallMixer32 | 18.596754                 |
| 3 | BaseMixer16  | 59.119162                 |
| 4 | BaseMixer32  | 59.532118                 |
| 5 | LargeMixer16 | 207.181418                |
| 6 | LargeMixer32 | 205.924514                |
| 7 | HugeMixer14  | 431.082762                |

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
