import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import scipy.io

def get_ade20k_class_names(root_dir):
    """Loads class names from objectInfo150.txt."""
    # objectInfo150.txt is in ADEChallengeData2016 directory
    ade20k_dir = os.path.join(root_dir, "ADEChallengeData2016")
    info_path = os.path.join(ade20k_dir, 'objectInfo150.txt')
    class_names = ['background']
    with open(info_path, 'r') as f:
        lines = f.readlines()
        # Skip the header line and process the rest
        for line in lines[1:]:  # Skip first line (header)
            parts = line.strip().split()
            if len(parts) > 4:
                class_names.append(parts[4])
    return class_names

def process_ade20k(root_dir, output_dir, split, class_names):
    """Processes ADE20K data for a given split."""
    ade20k_dir = os.path.join(root_dir, "ADEChallengeData2016")
    images_dir = os.path.join(ade20k_dir, "images", split)
    masks_dir = os.path.join(ade20k_dir, "annotations", split)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    processed_data = []
    for image_file in tqdm(image_files, desc=f"Processing ADE20K {split} set"):
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, f"{image_id}.png")

        if not os.path.exists(mask_path):
            continue

        mask = Image.open(mask_path)
        mask_array = np.array(mask, dtype=np.uint8)

        # In ADE20K, pixel values are class indices (0-150)
        # 0 is background, 1-150 are class indices
        present_class_indices = np.unique(mask_array)
        # Filter out invalid indices (> 150)
        present_class_indices = [idx for idx in present_class_indices if idx < len(class_names)]

        if not present_class_indices:
            continue
            
        current_class_names = [class_names[idx] for idx in present_class_indices]

        processed_data.append({
            'image_id': image_id,
            'image_path': image_path,
            'mask_path': mask_path,
            'class_names': current_class_names
        })

    os.makedirs(output_dir, exist_ok=True)

    # HY
    if split == "training":                              
        output_file_name = "ade20k_train.npy"
    elif split == "validation":
        output_file_name = "ade20k_val.npy"
    output_path = os.path.join(output_dir, output_file_name)
    # HY

    np.save(output_path, processed_data)

    # Save class names as well
    class_names_path = os.path.join(output_dir, "ade20k_class_names.json")
    import json
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)

    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess ADE20K segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data/ade20k", help="Root directory containing ADEChallengeData2016.")
    parser.add_argument("--output_dir", type=str, default="./data/ade20k_processed", help="Directory to save processed data.")
    args = parser.parse_args()

    class_names = get_ade20k_class_names(args.root_dir)

    process_ade20k(args.root_dir, args.output_dir, "training", class_names)   
    process_ade20k(args.root_dir, args.output_dir, "validation", class_names)  

if __name__ == "__main__":
    main()
