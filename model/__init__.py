"""
Model Package

DINOde: continuous vision-text alignment for open-vocabulary semantic segmentation.

Modules:
- `components.py`: frozen encoders (CLIPTextEncoder, DinoV3HFBackbone) and shared utilities.
- `dinode.py`: the text-conditioned segmentation head (`TextCondHead`) built on an ODE flow
               field, and the training loop wrapper (`FlowTrainer`).
- `pamr.py`: optional PAMR mask refinement applied at inference time.
"""

from .components import (
    CLIPTextEncoder,
    DinoV3HFBackbone,
    build_prompts,
    l2norm,
    l2norm_hw,
)

from .dinode import (
    TextCondHead,
    FlowTrainer,
)

# The `__all__` variable explicitly defines the public interface of this package.
__all__ = [
    'CLIPTextEncoder',
    'DinoV3HFBackbone',
    'TextCondHead',
    'FlowTrainer',
    'build_prompts',
    'l2norm',
    'l2norm_hw',
]
