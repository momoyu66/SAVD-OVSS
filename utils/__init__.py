"""
Utils Package

- `metrics.py`: `MeanIoU`, the streaming confusion-matrix mIoU metric used by all
                eight evaluation protocols.
- `model_utils.py`: `count_parameters`, parameter accounting for the trainable head.
"""

from .metrics import MeanIoU
from .model_utils import count_parameters

# The `__all__` variable explicitly defines the public interface of this package.
__all__ = [
    'MeanIoU',
    'count_parameters',
]
