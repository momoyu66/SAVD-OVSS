"""
PASCAL Context Dataset Module

PASCAL Context has 60 classes (including background) or 59 classes (excluding background).
- Context60: 60 classes (BG include) - includes background as class 0
- Context59: 59 classes (BG exclude) - background is treated as ignore (255)
"""

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Dict, Union
import torch

class PascalContextDataset(Dataset):
    """PyTorch Dataset for the preprocessed PASCAL Context segmentation dataset.
    
    This dataset loads image and mask paths, and class names from a preprocessed .npy file.
    """
    
    def __init__(self, processed_data_path: str, transform=None, mask_transform=None, max_samples: Optional[int] = None, dino_cache_dir: Optional[str] = None):
        """
        Initializes the PascalContextDataset.

        Args:
            processed_data_path (str): Path to the processed .npy file.
            transform: Image transformations to apply.
            mask_transform: Mask transformations to apply.
            max_samples (Optional[int]): Maximum number of samples to load.
            dino_cache_dir (Optional[str]): Path to per-image DINOv3 feature cache directory.
        """
        self.processed_data = np.load(processed_data_path, allow_pickle=True)

        if max_samples:
            self.processed_data = self.processed_data[:max_samples]

        self.transform = transform
        self.mask_transform = mask_transform
        self.dino_cache_dir = dino_cache_dir

        # Load class names
        class_names_path = os.path.join(os.path.dirname(processed_data_path), "pascal_context_class_names.json")
        if os.path.exists(class_names_path):
            import json
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Fallback for older processed data
            self.class_names = [
            'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird',
            'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
            'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
            'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
            'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant',
            'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
            'table', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
        ]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.processed_data)
    
    def get_class_names(self) -> List[str]:
        """Returns the list of class names used by the dataset."""
        return self.class_names

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
        class_names = sample['class_names']

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

        return {
            'image': image,
            'mask': mask,
            'class_names': class_names,
            'image_id': sample['image_id'],
            'dataset_idx': idx,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }
