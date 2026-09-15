"""
COCO Object Dataset Module

COCO Object has 81 classes (including background) or 80 classes (excluding background).
- COCO81: 81 classes (BG include) - includes background as class 0
- COCO80: 80 classes (BG exclude) - background is treated as ignore (255)

Note: COCO Object uses the first 80 object classes from COCO dataset.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Dict, Union
import torch

# Load class names from file
PARENT_PATH = os.path.dirname(os.path.realpath(__file__))

def load_class_names(filename):
    """Load class names from a text file."""
    filepath = os.path.join(PARENT_PATH, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

# Class names for different modes
COCO81_CLASS_NAMES = load_class_names('cls_coco81.txt')  # 81 classes with BG
COCO80_CLASS_NAMES = load_class_names('cls_coco80.txt')  # 80 classes without BG


class COCOObjectDataset(Dataset):
    """PyTorch Dataset for the COCO Object segmentation dataset.

    Supports two evaluation modes:
    - bg_mode='include': 81 classes including background (class 0)
    - bg_mode='exclude': 80 classes, background treated as ignore label (255)
    """

    def __init__(self, processed_data_path: str, transform=None, mask_transform=None,
                 max_samples: Optional[int] = None, bg_mode: str = 'include',
                 dino_cache_dir: Optional[str] = None):
        """
        Initializes the COCOObjectDataset.

        Args:
            processed_data_path (str): Path to the processed .npy file.
            transform: Image transformations to apply.
            mask_transform: Mask transformations to apply.
            max_samples (Optional[int]): Maximum number of samples to load.
            bg_mode (str): Background handling mode.
                          'include' - 81 classes (background is class 0)
                          'exclude' - 80 classes (background becomes ignore label 255)
            dino_cache_dir (Optional[str]): Path to per-image DINOv3 feature cache directory.
        """
        self.processed_data = np.load(processed_data_path, allow_pickle=True)

        if max_samples:
            self.processed_data = self.processed_data[:max_samples]

        self.transform = transform
        self.mask_transform = mask_transform
        self.bg_mode = bg_mode
        self.dino_cache_dir = dino_cache_dir

        # Set class names based on mode
        if bg_mode == 'include':
            self.class_names = COCO81_CLASS_NAMES if COCO81_CLASS_NAMES else self._default_coco81_classes()
            self.num_classes = 81
        else:  # exclude
            self.class_names = COCO80_CLASS_NAMES if COCO80_CLASS_NAMES else self._default_coco80_classes()
            self.num_classes = 80

    def _default_coco81_classes(self):
        """Default class names for COCO81 (with background)."""
        return [
            'background', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable',
            'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def _default_coco80_classes(self):
        """Default class names for COCO80 (without background)."""
        return [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable',
            'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.processed_data)

    def get_class_names(self) -> List[str]:
        """Returns the list of class names used by the dataset."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Returns the number of classes."""
        return self.num_classes

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List[str], Image.Image]]:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict containing the image, mask, and class names.
        """
        sample = self.processed_data[idx]

        image_path = sample['image_path']
        mask_path = sample['mask_path']

        dino_A = None
        dino_cls = None
        if self.dino_cache_dir is not None:
            img_name = os.path.splitext(os.path.basename(str(image_path)))[0]
            cache_path = os.path.join(self.dino_cache_dir, f"{img_name}.npy")
            if os.path.exists(cache_path):
                cached = np.load(cache_path, allow_pickle=True).item()
                dino_A = cached['A'].astype(np.float32)
                dino_cls = cached['cls'].astype(np.float32)

        # image = None if dino_A is not None else Image.open(image_path).convert('RGB')
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        # Handle background mode
        if self.bg_mode == 'exclude':
            new_mask = np.where(mask_array == 0, 255, mask_array - 1)
            new_mask = np.where(mask_array == 255, 255, new_mask)
            mask = Image.fromarray(new_mask.astype(np.uint8))

        # Get class names present in this image
        if self.bg_mode == 'include':
            present_indices = np.unique(mask_array)
            present_indices = [i for i in present_indices if i < len(self.class_names) and i != 255]
        else:
            present_indices = np.unique(np.array(mask))
            present_indices = [i for i in present_indices if i < len(self.class_names) and i != 255]

        current_class_names = [self.class_names[i] for i in present_indices if i < len(self.class_names)]

        return {
            'image': image,
            'mask': mask,
            'class_names': current_class_names,
            'image_id': sample.get('image_id', str(idx)),
            'dataset_idx': idx,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }
