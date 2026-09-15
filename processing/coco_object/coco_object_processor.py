"""
COCO Object Dataset Processor

COCO Object uses the 80 thing (object) classes from COCO dataset.
This processor uses the same COCO-Stuff 2017 dataset directory but only keeps
the first 80 object classes (indices 0-79 in the remapped 171-class format).

Directory structure expected (same as COCO-Stuff):
    root_dir/
        train2017/
            *.jpg
        val2017/
            *.jpg
        annotations/
            train2017/  (*.png masks from COCO-Stuff)
            val2017/    (*.png masks from COCO-Stuff)
            coco.names

Note: COCO Object is a subset of COCO-Stuff, using only the first 80 thing classes.
Background (unlabeled, index 0) is included as class 0 for BG Include protocols.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json

# 81 classes for COCO Object evaluation (WITH background)
# Background + 80 thing classes (first 80 classes from COCO-Stuff 171)
COCO_OBJECT_81_CLASSES = [
    'background',  # index 0
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite',
    'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle',
    'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush'
]

# Mapping from COCO-Stuff 2017 coco.names index to COCO Object 81-class index
# coco.names: 0-181=classes (182 total, person=0, wood=181)
# COCO Object: 0=background, 1-80=thing classes
# Classes to skip: street sign(11), hat(25), shoe(28), eye glasses(29), plate(44),
#                  mirror(65), window(67), desk(68), door(70), blender(82), hair brush(90)
# Note: After post-processing, thing classes (0-89) are shifted +1, stuff classes (91+) become 0 (background)
COCO_STUFF_2017_TO_OBJECT_81 = {
    # === Thing classes (0-89, 11 skipped) -> will be shifted +1 to 1-80 ===
    0: 0,      # person -> 1
    1: 1,      # bicycle -> 2
    2: 2,      # car -> 3
    3: 3,      # motorcycle -> 4
    4: 4,      # airplane -> 5
    5: 5,      # bus -> 6
    6: 6,      # train -> 7
    7: 7,      # truck -> 8
    8: 8,      # boat -> 9
    9: 9,      # traffic light -> 10
    10: 10,    # fire hydrant -> 11
    # 11: SKIP - street sign
    12: 11,    # stop sign -> 12
    13: 12,    # parking meter -> 13
    14: 13,    # bench -> 14
    15: 14,    # bird -> 15
    16: 15,    # cat -> 16
    17: 16,    # dog -> 17
    18: 17,    # horse -> 18
    19: 18,    # sheep -> 19
    20: 19,    # cow -> 20
    21: 20,    # elephant -> 21
    22: 21,    # bear -> 22
    23: 22,    # zebra -> 23
    24: 23,    # giraffe -> 24
    # 25: SKIP - hat
    26: 24,    # backpack -> 25
    27: 25,    # umbrella -> 26
    # 28: SKIP - shoe
    # 29: SKIP - eye glasses
    30: 26,    # handbag -> 27
    31: 27,    # tie -> 28
    32: 28,    # suitcase -> 29
    33: 29,    # frisbee -> 30
    34: 30,    # skis -> 31
    35: 31,    # snowboard -> 32
    36: 32,    # sports ball -> 33
    37: 33,    # kite -> 34
    38: 34,    # baseball bat -> 35
    39: 35,    # baseball glove -> 36
    40: 36,    # skateboard -> 37
    41: 37,    # surfboard -> 38
    42: 38,    # tennis racket -> 39
    43: 39,    # bottle -> 40
    # 44: SKIP - plate
    45: 40,    # wine glass -> 41
    46: 41,    # cup -> 42
    47: 42,    # fork -> 43
    48: 43,    # knife -> 44
    49: 44,    # spoon -> 45
    50: 45,    # bowl -> 46
    51: 46,    # banana -> 47
    52: 47,    # apple -> 48
    53: 48,    # sandwich -> 49
    54: 49,    # orange -> 50
    55: 50,    # broccoli -> 51
    56: 51,    # carrot -> 52
    57: 52,    # hot dog -> 53
    58: 53,    # pizza -> 54
    59: 54,    # donut -> 55
    60: 55,    # cake -> 56
    61: 56,    # chair -> 57
    62: 57,    # couch -> 58
    63: 58,    # potted plant -> 59
    64: 59,    # bed -> 60
    # 65: SKIP - mirror
    66: 60,    # dining table -> 61
    # 67: SKIP - window
    # 68: SKIP - desk
    69: 61,    # toilet -> 62
    # 70: SKIP - door
    71: 62,    # tv -> 63
    72: 63,    # laptop -> 64
    73: 64,    # mouse -> 65
    74: 65,    # remote -> 66
    75: 66,    # keyboard -> 67
    76: 67,    # cell phone -> 68
    77: 68,    # microwave -> 69
    78: 69,    # oven -> 70
    79: 70,    # toaster -> 71
    80: 71,    # sink -> 72
    81: 72,    # refrigerator -> 73
    # 82: SKIP - blender
    83: 73,    # book -> 74
    84: 74,    # clock -> 75
    85: 75,    # vase -> 76
    86: 76,    # scissors -> 77
    87: 77,    # teddy bear -> 78
    88: 78,    # hair drier -> 79
    89: 79,    # toothbrush -> 80
    # 90: SKIP - hair brush
    # === Stuff classes (91-181) -> will be set to 0 (background) ===
    91: 80,    # banner -> 0 (background)
    92: 81,    # blanket -> 0 (background)
    93: 82,    # branch -> 0 (background)
    94: 83,    # bridge -> 0 (background)
    95: 84,    # building-other -> 0 (background)
    96: 85,    # bush -> 0 (background)
    97: 86,    # cabinet -> 0 (background)
    98: 87,    # cage -> 0 (background)
    99: 88,    # cardboard -> 0 (background)
    100: 89,   # carpet -> 0 (background)
    101: 90,   # ceiling-other -> 0 (background)
    102: 91,   # ceiling-tile -> 0 (background)
    103: 92,   # cloth -> 0 (background)
    104: 93,   # clothes -> 0 (background)
    105: 94,   # clouds -> 0 (background)
    106: 95,   # counter -> 0 (background)
    107: 96,   # cupboard -> 0 (background)
    108: 97,   # curtain -> 0 (background)
    109: 98,   # desk-stuff -> 0 (background)
    110: 99,   # dirt -> 0 (background)
    111: 100,  # door-stuff -> 0 (background)
    112: 101,  # fence -> 0 (background)
    113: 102,  # floor-marble -> 0 (background)
    114: 103,  # floor-other -> 0 (background)
    115: 104,  # floor-stone -> 0 (background)
    116: 105,  # floor-tile -> 0 (background)
    117: 106,  # floor-wood -> 0 (background)
    118: 107,  # flower -> 0 (background)
    119: 108,  # fog -> 0 (background)
    120: 109,  # food-other -> 0 (background)
    121: 110,  # fruit -> 0 (background)
    122: 111,  # furniture-other -> 0 (background)
    123: 112,  # grass -> 0 (background)
    124: 113,  # gravel -> 0 (background)
    125: 114,  # ground-other -> 0 (background)
    126: 115,  # hill -> 0 (background)
    127: 116,  # house -> 0 (background)
    128: 117,  # leaves -> 0 (background)
    129: 118,  # light -> 0 (background)
    130: 119,  # mat -> 0 (background)
    131: 120,  # metal -> 0 (background)
    132: 121,  # mirror-stuff -> 0 (background)
    133: 122,  # moss -> 0 (background)
    134: 123,  # mountain -> 0 (background)
    135: 124,  # mud -> 0 (background)
    136: 125,  # napkin -> 0 (background)
    137: 126,  # net -> 0 (background)
    138: 127,  # paper -> 0 (background)
    139: 128,  # pavement -> 0 (background)
    140: 129,  # pillow -> 0 (background)
    141: 130,  # plant-other -> 0 (background)
    142: 131,  # plastic -> 0 (background)
    143: 132,  # platform -> 0 (background)
    144: 133,  # playingfield -> 0 (background)
    145: 134,  # railing -> 0 (background)
    146: 135,  # railroad -> 0 (background)
    147: 136,  # river -> 0 (background)
    148: 137,  # road -> 0 (background)
    149: 138,  # rock -> 0 (background)
    150: 139,  # roof -> 0 (background)
    151: 140,  # rug -> 0 (background)
    152: 141,  # salad -> 0 (background)
    153: 142,  # sand -> 0 (background)
    154: 143,  # sea -> 0 (background)
    155: 144,  # shelf -> 0 (background)
    156: 145,  # sky-other -> 0 (background)
    157: 146,  # skyscraper -> 0 (background)
    158: 147,  # snow -> 0 (background)
    159: 148,  # solid-other -> 0 (background)
    160: 149,  # stairs -> 0 (background)
    161: 150,  # stone -> 0 (background)
    162: 151,  # straw -> 0 (background)
    163: 152,  # structural-other -> 0 (background)
    164: 153,  # table -> 0 (background)
    165: 154,  # tent -> 0 (background)
    166: 155,  # textile-other -> 0 (background)
    167: 156,  # towel -> 0 (background)
    168: 157,  # tree -> 0 (background)
    169: 158,  # vegetable -> 0 (background)
    170: 159,  # wall-brick -> 0 (background)
    171: 160,  # wall-concrete -> 0 (background)
    172: 161,  # wall-other -> 0 (background)
    173: 162,  # wall-panel -> 0 (background)
    174: 163,  # wall-stone -> 0 (background)
    175: 164,  # wall-tile -> 0 (background)
    176: 165,  # wall-wood -> 0 (background)
    177: 166,  # water-other -> 0 (background)
    178: 167,  # waterdrops -> 0 (background)
    179: 168,  # window-blind -> 0 (background)
    180: 169,  # window-other -> 0 (background)
    181: 170,  # wood -> 0 (background)
    255: 255   # ignore
}

# Post-process: shift thing classes +1 (so person=1), set stuff classes to 0 (background)
for k, v in COCO_STUFF_2017_TO_OBJECT_81.items():
    COCO_STUFF_2017_TO_OBJECT_81[k] = v + 1
    if k > 90:
        COCO_STUFF_2017_TO_OBJECT_81[k] = 0

def remap_mask_to_object(mask_array):
    """Remap mask from COCO-Stuff 2017 indices to COCO Object 81-class format."""
    # new_mask = np.full_like(mask_array, 255, dtype=np.uint8)  # Default to ignore
    new_mask = np.zeros_like(mask_array, dtype=np.uint8)

    for old_idx, new_idx in COCO_STUFF_2017_TO_OBJECT_81.items():
        new_mask[mask_array == old_idx] = new_idx

    return new_mask


def process_coco_object(root_dir, output_dir, split):
    """
    Processes COCO Object segmentation data.

    Args:
        root_dir (str): Root directory of COCO-Stuff dataset (same structure).
        output_dir (str): Directory to save the processed .npy file.
        split (str): The data split to process ('train' or 'val').
    """
    images_dir = os.path.join(root_dir, f"{split}2017")
    masks_dir = os.path.join(root_dir, "annotations", f"{split}2017")

    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    # Create remapped mask output directory
    mask_output_dir = os.path.join(output_dir, "masks", split)
    os.makedirs(mask_output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    processed_data = []
    for image_file in tqdm(image_files, desc=f"Processing COCO Object {split} set"):
        image_id_str = os.path.splitext(image_file)[0]

        image_path = os.path.join(images_dir, image_file)
        orig_mask_path = os.path.join(masks_dir, f"{image_id_str}.png")

        if not os.path.exists(orig_mask_path):
            continue

        mask = Image.open(orig_mask_path)
        mask_array = np.array(mask, dtype=np.uint8)

        # Remap mask to 81-class format (keep only object classes)
        remapped_mask = remap_mask_to_object(mask_array)

        # Collect the object classes present in this image (excluding background and ignore).
        # Images whose mask contains no object class are intentionally KEPT: the COCO-Object
        # protocol evaluates the full 5,000-image val2017 split, so dropping them here would
        # change the denominator of the reported mIoU.
        present_class_indices = np.unique(remapped_mask)
        present_class_indices = [idx for idx in present_class_indices if 0 < idx < 81]

        # Save remapped mask
        new_mask_path = os.path.join(mask_output_dir, f"{image_id_str}.png")
        Image.fromarray(remapped_mask).save(new_mask_path)

        # Include background in class names for BG Include protocol
        current_class_names = [COCO_OBJECT_81_CLASSES[idx] for idx in present_class_indices]

        processed_data.append({
            'image_id': image_id_str,
            'image_path': image_path,
            'mask_path': new_mask_path,
            'class_names': current_class_names
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"coco_object_{split}.npy")
    np.save(output_path, processed_data)

    # Save class names (81 classes WITH background)
    class_names_path = os.path.join(output_dir, "coco_object_class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(COCO_OBJECT_81_CLASSES, f)

    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")
    print(f"Using 81 classes (80 objects + background)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO Object segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data/coco_stuff",
                        help="Root directory of COCO-Stuff dataset (contains train2017, val2017, annotations).")
    parser.add_argument("--output_dir", type=str, default="./data/coco_object_processed",
                        help="Directory to save processed data.")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val"],
                        help="Process only this split. If not set, processes both train and val.")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        process_coco_object(args.root_dir, args.output_dir, split)


if __name__ == "__main__":
    main()
