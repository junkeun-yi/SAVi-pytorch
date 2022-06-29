"""STEVE module

From 
    Simple Unsupervised Object-Centric Learning
        for Complex and Naturalistic Videos
    - https://arxiv.org/pdf/2205.14065.pdf
    - https://sites.google.com/view/slot-transformer-for-videos
and
    Illiterate DALL-E Learns to Compose
    - https://arxiv.org/pdf/2110.11405.pdf
    - https://github.com/singhgautam/slate
"""

from .dvae import (dVAE)
from .model_slate import (SLATE, OneHotDictionary)
from .model_steve import (STEVE)
from .utils import (Conv2dBlock, conv2d)