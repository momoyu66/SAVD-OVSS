import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json

def get_cityscapes_class_names():
    """Returns the Cityscapes class names."""
    # Cityscapes has 34 classes (19 classes + ignore classes)
    # Mapping from trainId to class name
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    return class_names

def process_cityscapes(root_dir, output_dir, split, class_names):
    """Processes Cityscapes data for a given split."""
    # Cityscapes directory structure
    images_dir = os.path.join(root_dir, "leftImg8bit", split)
    masks_dir = os.path.join(root_dir, "gtFine", split)
    
    # Get list of cities
    cities = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
    
    processed_data = []
    
    for city in cities:
        city_images_dir = os.path.join(images_dir, city)
        city_masks_dir = os.path.join(masks_dir, city)
        
        image_files = sorted([f for f in os.listdir(city_images_dir) if f.endswith('.png')])
        
        for image_file in tqdm(image_files, desc=f"Processing Cityscapes {split} - {city}"):
            image_id = os.path.splitext(image_file)[0]
            # Cityscapes filename format: {city}_{seq}_{frame}_{camera}_leftImg8bit.png
            # GT filename format: {city}_{seq}_{frame}_{camera}_gtFine_labelIds.png
            
            image_path = os.path.join(city_images_dir, image_file)
            # Extract base name (remove '_leftImg8bit')
            base_name = image_id.replace('_leftImg8bit', '')
            mask_path = os.path.join(city_masks_dir, f"{base_name}_gtFine_labelTrainIds.png")
            
            mask = Image.open(mask_path)
            mask_array = np.array(mask, dtype=np.uint8)
            
            # Cityscapes uses trainId for training (0-18 valid classes, 255 for ignore)
            # Map trainId to class names
            present_train_ids = np.unique(mask_array)
            # Filter out ignore class (255) and background (negative values)
            present_train_ids = [tid for tid in present_train_ids if 0 <= tid < len(class_names)]
            
            # if not present_train_ids:
            #     continue
            
            current_class_names = [class_names[tid] for tid in present_train_ids]
            
            processed_data.append({
                'image_id': image_id,
                'image_path': image_path,
                'mask_path': mask_path,
                'class_names': current_class_names
            })
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cityscapes_{split}.npy")
    np.save(output_path, processed_data)
    
    # Save class names as well
    class_names_path = os.path.join(output_dir, "cityscapes_class_names.json")
    if not os.path.exists(class_names_path):
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)
    
    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Cityscapes segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data/cityscapes", 
                       help="Root directory of the Cityscapes dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/cityscapes_processed", 
                       help="Directory to save processed data.")
    args = parser.parse_args()
    
    class_names = get_cityscapes_class_names()
    
    # Cityscapes splits: train, val, test
    process_cityscapes(args.root_dir, args.output_dir, "train", class_names)
    process_cityscapes(args.root_dir, args.output_dir, "val", class_names)
    # Note: test set doesn't have ground truth labels, so we skip it
    
    print("\nCityscapes preprocessing completed!")

if __name__ == "__main__":
    main()




