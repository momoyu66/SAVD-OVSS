"""
Data Processing Package

This package contains modules for processing various datasets, including:
- `coco_stuff`: For handling COCO-Stuff dataset (171 classes, BG exclude).
- `coco_object`: For handling COCO Object dataset (80/81 classes).
- `ade20k`: For handling ADE20K dataset (150 classes, BG exclude).
- `cityscapes`: For handling Cityscapes dataset (19 classes, no BG).
- `pascal_voc`: For handling PASCAL VOC dataset (20/21 classes).
- `pascal_context`: For handling PASCAL Context dataset (59/60 classes).

Evaluation Protocols:
- BG Include: VOC21, Context60, COCO Object (81 classes)
- BG Exclude: VOC20, Context59, COCO Stuff (171), Cityscapes (19), ADE20K (150)
"""

from .coco_stuff.coco_stuff_dataset import COCOStuffDataset
from .coco_object.coco_object_dataset import COCOObjectDataset
from .ade20k.ade20k_dataset import ADE20KDataset
from .cityscapes.cityscapes_dataset import CityscapesDataset
from .pascal_voc.pascal_voc_dataset import PascalVOCDataset
from .pascal_context.pascal_context_dataset import PascalContextDataset

__all__ = [
    'COCOStuffDataset',
    'COCOObjectDataset',
    'ADE20KDataset',
    'CityscapesDataset',
    'PascalVOCDataset',
    'PascalContextDataset',
]
