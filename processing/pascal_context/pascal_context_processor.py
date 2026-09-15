"""
PASCAL Context Dataset Processor

PASCAL Context extends PASCAL VOC 2010 with additional annotations for 60 context classes
(including background). The dataset uses the same images as PASCAL VOC but with more
detailed segmentation annotations.

Directory structure expected (similar to PASCAL VOC):
    root_dir/
        VOCdevkit/
            VOC2010/
                JPEGImages/
                    *.jpg
                SegmentationClass/
                    *.png (context annotations as PNG masks)
                ImageSets/
                    Segmentation/
                        train.txt
                        val.txt

Download:
    - Images: PASCAL VOC 2010 from http://host.robots.ox.ac.uk/pascal/VOC/voc2010/
    - Annotations: https://cs.stanford.edu/~roozbeh/pascal-context/
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json

# 60 classes (including background)
PASCAL_CONTEXT_CLASSES = [
    'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird',
    'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
    'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
    'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant',
    'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
    'table', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
]


def process_pascal_context(root_dir, output_dir, split):
    """
    Processes PASCAL Context segmentation data.

    Args:
        root_dir (str): Root directory containing VOCdevkit.
        output_dir (str): Directory to save the processed .npy file.
        split (str): The data split to process ('train' or 'val').
    """
    voc_dir = os.path.join(root_dir, "VOCdevkit", "VOC2010")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    masks_dir = os.path.join(voc_dir, "SegmentationClassContext")
    imageset_file = os.path.join(voc_dir, "ImageSets", "SegmentationContext", f"{split}.txt")

    if not os.path.exists(imageset_file):
        raise FileNotFoundError(f"Image set file not found: {imageset_file}")

    with open(imageset_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    processed_data = []
    # all_class_names = set()
    for image_id in tqdm(image_ids, desc=f"Processing PASCAL Context {split} set"):
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        mask_path = os.path.join(masks_dir, f"{image_id}.png")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        mask = Image.open(mask_path)
        mask_array = np.array(mask, dtype=np.uint8)

        present_class_indices = np.unique(mask_array)
        # all_class_names.update(present_class_indices)

        # Filter out background (0) and void (255)
        present_class_indices = [idx for idx in present_class_indices if idx != 0 and idx != 255]

        if not present_class_indices:
            continue

        class_names = [PASCAL_CONTEXT_CLASSES[idx] for idx in present_class_indices]

        processed_data.append({
            'image_id': image_id,
            'image_path': image_path,
            'mask_path': mask_path,
            'class_names': class_names
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pascal_context_{split}.npy")
    np.save(output_path, processed_data)

    # Save class names as well
    class_names_path = os.path.join(output_dir, "pascal_context_class_names.json")
    if not os.path.exists(class_names_path):
        with open(class_names_path, 'w') as f:
            json.dump(PASCAL_CONTEXT_CLASSES, f)

    # print(f"Unique classes in {split} set: {sorted(all_class_names)}")
    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PASCAL Context segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data/pascal_context",
                        help="Root directory containing VOCdevkit and context annotations.")
    parser.add_argument("--output_dir", type=str, default="./data/pascal_context_processed",
                        help="Directory to save processed data.")
    args = parser.parse_args()

    process_pascal_context(args.root_dir, args.output_dir, "train")
    process_pascal_context(args.root_dir, args.output_dir, "val")


if __name__ == "__main__":
    main()
