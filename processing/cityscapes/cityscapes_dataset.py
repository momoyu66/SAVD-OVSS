import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import List

class CityscapesDataset(Dataset):
    """Dataset for Cityscapes segmentation with open-vocabulary support."""
    
    def __init__(self, processed_data_path, split='train', transform=None, dino_cache_dir=None):
        """
        Initializes the Cityscapes dataset.

        Args:
            processed_data_path (str): Path to the directory containing processed .npy files.
            split (str): Dataset split ('train' or 'val'). Defaults to 'train'.
            transform: Optional transform to be applied to images.
            dino_cache_dir: Path to per-image DINOv3 feature cache directory.
        """
        self.split = split
        self.transform = transform
        self.dino_cache_dir = dino_cache_dir
        
        # Load processed data
        data_file = os.path.join(processed_data_path, f"cityscapes_{split}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Processed data file not found: {data_file}")
        
        self.data = np.load(data_file, allow_pickle=True)
        
        # Load class names
        class_names_file = os.path.join(processed_data_path, "cityscapes_class_names.json")
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Fallback to default class names
            self.class_names = [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]
        
        print(f"Loaded Cityscapes {split} dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Returns:
            dict: A dictionary containing:
                - 'image': PIL Image
                - 'mask': PIL Image
                - 'image_path': str
                - 'mask_path': str
                - 'class_names': list of class names present in the image
                - 'class_indices': list of class indices
        """
        sample = self.data[idx]

        dino_A = None
        dino_cls = None
        if self.dino_cache_dir is not None:
            img_name = os.path.splitext(os.path.basename(str(sample['image_path'])))[0]
            cache_path = os.path.join(self.dino_cache_dir, f"{img_name}.npy")
            if os.path.exists(cache_path):
                cached = np.load(cache_path, allow_pickle=True).item()
                dino_A = cached['A'].astype(np.float32)
                dino_cls = cached['cls'].astype(np.float32)

        # image = None if dino_A is not None else Image.open(sample['image_path']).convert('RGB')
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path'])

        if self.transform and image is not None:
            image = self.transform(image)

        class_indices = [self.class_names.index(cls) for cls in sample['class_names']
                        if cls in self.class_names]

        return {
            'image': image,
            'mask': mask,
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
            'class_names': sample['class_names'],
            'class_indices': class_indices,
            'dataset_idx': idx,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }
    
    def get_all_class_names(self):
        """Returns all possible class names in the dataset."""
        return self.class_names
    
    def get_class_names(self) -> List[str]:
        """Returns all possible class names in the dataset."""
        return self.class_names




