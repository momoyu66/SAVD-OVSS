import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import pickle
from collections import defaultdict

PASCAL_VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_class_map():
    """Returns a color map for PASCAL VOC classes."""
    # This is a standard color map for PASCAL VOC
    # The last color is for the void/border class (255)
    return np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        [0, 192, 0], [128, 192, 0], [0, 64, 128], [224, 224, 192] # Void class
    ])

def process_pascal_voc(root_dir, output_dir, split):
    """
    Processes PASCAL VOC 2012 segmentation data.

    This function reads the specified data split (e.g., 'train', 'val'),
    parses the segmentation masks to identify the classes present in each image,
    and saves the processed data as a .npy file.

    Args:
        root_dir (str): Root directory of the PASCAL VOC dataset (containing VOCdevkit).
        output_dir (str): Directory to save the processed .npy file.
        split (str): The data split to process ('train' or 'val').
    """
    voc_dir = os.path.join(root_dir, "VOCdevkit", "VOC2012")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    masks_dir = os.path.join(voc_dir, "SegmentationClass")
    imageset_file = os.path.join(voc_dir, "ImageSets", "Segmentation", f"{split}.txt")

    if not os.path.exists(imageset_file):
        raise FileNotFoundError(f"Image set file not found: {imageset_file}")

    with open(imageset_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    processed_data = []
    for image_id in tqdm(image_ids, desc=f"Processing PASCAL VOC {split} set"):
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        mask_path = os.path.join(masks_dir, f"{image_id}.png")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        mask = Image.open(mask_path)
        mask_array = np.array(mask, dtype=np.uint8)

        present_class_indices = np.unique(mask_array)
        
        # Filter out background (0) and void (255)
        present_class_indices = [idx for idx in present_class_indices if idx != 0 and idx != 255]
        
        if not present_class_indices:
            continue

        class_names = [PASCAL_VOC_CLASSES[idx] for idx in present_class_indices]
        
        processed_data.append({
            'image_id': image_id,
            'image_path': image_path,
            'mask_path': mask_path,
            'class_names': class_names
        })
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pascal_voc_{split}.npy")
    np.save(output_path, processed_data)
    
    # Save class names as well
    class_names_path = os.path.join(output_dir, "pascal_voc_class_names.json")
    if not os.path.exists(class_names_path):
        import json
        with open(class_names_path, 'w') as f:
            json.dump(PASCAL_VOC_CLASSES, f)

    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess PASCAL VOC 2012 segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data", help="Root directory of the PASCAL VOC dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/pascal_voc_processed", help="Directory to save processed data.")
    args = parser.parse_args()

    process_pascal_voc(args.root_dir, args.output_dir, "train")
    process_pascal_voc(args.root_dir, args.output_dir, "val")

if __name__ == "__main__":
    main()
