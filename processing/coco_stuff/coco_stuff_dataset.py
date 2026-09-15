import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Dict, Union
import torch


class COCOStuffDataset(Dataset):
    """PyTorch Dataset for the preprocessed COCO-Stuff segmentation dataset."""

    def __init__(self, processed_data_path: str, transform=None, mask_transform=None, max_samples: Optional[int] = None, dino_cache_dir: Optional[str] = None):
        self.processed_data = np.load(processed_data_path, allow_pickle=True)

        if max_samples:
            self.processed_data = self.processed_data[:max_samples]

        self.transform = transform
        self.mask_transform = mask_transform
        self.dino_cache_dir = dino_cache_dir

        # Load class names
        class_names_path = os.path.join(os.path.dirname(processed_data_path), "coco_stuff_class_names.json")
        if os.path.exists(class_names_path):
            import json
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [] # Should not happen if processor is run

    def __len__(self) -> int:
        return len(self.processed_data)
    
    def get_class_names(self) -> List[str]:
        """Returns the list of class names used by the dataset."""
        return self.class_names
    
    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List[str], Image.Image]]:
        sample = self.processed_data[idx]

        # Try loading cached DINOv3 features (per-image .npy file)
        dino_A = None
        dino_cls = None
        if self.dino_cache_dir is not None:
            img_name = os.path.splitext(os.path.basename(str(sample['image_path'])))[0]
            cache_path = os.path.join(self.dino_cache_dir, f"{img_name}.npy")
            if os.path.exists(cache_path):
                cached = np.load(cache_path, allow_pickle=True).item()
                dino_A = cached['A'].astype(np.float32)
                dino_cls = cached['cls'].astype(np.float32)

        # Only load PIL image if no cached features (saves I/O during training)
        # image = None if dino_A is not None else Image.open(sample['image_path']).convert('RGB')
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path'])

        return {
            'image': image,
            'mask': mask,
            'class_names': sample['class_names'],
            'image_id': sample['image_id'],
            'captions': sample.get('captions', []),
            'dataset_idx': idx,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }
