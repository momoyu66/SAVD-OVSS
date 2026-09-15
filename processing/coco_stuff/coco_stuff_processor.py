"""
COCO-Stuff Dataset Processor

COCO-Stuff 2017 dataset has 182 classes in coco.names (0=unlabeled, 1-181=classes).
For evaluation, we use 171 classes following the CLIPer protocol:
- Skip 11 classes: street sign, hat, shoe, eye glasses, plate, mirror, window, desk, door, blender, hair brush
- Background (unlabeled, index 0) is excluded from evaluation

The mask indices are remapped from COCO-Stuff 2017 format to CLIPer 171-class format.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json

# 171 classes for COCO-Stuff evaluation (NO background)
# This follows CLIPer's cls_coco_stuff172.txt (lines 2-172, excluding background)
COCO_STUFF_171_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite',
    'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle',
    'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'banner', 'blanket',
    'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
    'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
    'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other',
    'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
    'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light',
    'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
    'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
    'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
    'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
    'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
]

# Mapping from COCO-Stuff 2017 coco.names index to CLIPer 171-class index
# coco.names: 0-181=classes (182 total, person=0, wood=181)
# CLIPer 171: 0-170 = 171 classes (no background)
# Classes to skip: street sign(11), hat(25), shoe(28), eye glasses(29), plate(44),
#                  mirror(65), window(67), desk(68), door(70), blender(82), hair brush(90)
COCO_STUFF_2017_TO_171 = {
    # === Thing classes (0-90, 11 skipped) ===
    0: 0,      # person
    1: 1,      # bicycle
    2: 2,      # car
    3: 3,      # motorcycle
    4: 4,      # airplane
    5: 5,      # bus
    6: 6,      # train
    7: 7,      # truck
    8: 8,      # boat
    9: 9,      # traffic light
    10: 10,    # fire hydrant
    # 11: SKIP - street sign
    12: 11,    # stop sign
    13: 12,    # parking meter
    14: 13,    # bench
    15: 14,    # bird
    16: 15,    # cat
    17: 16,    # dog
    18: 17,    # horse
    19: 18,    # sheep
    20: 19,    # cow
    21: 20,    # elephant
    22: 21,    # bear
    23: 22,    # zebra
    24: 23,    # giraffe
    # 25: SKIP - hat
    26: 24,    # backpack
    27: 25,    # umbrella
    # 28: SKIP - shoe
    # 29: SKIP - eye glasses
    30: 26,    # handbag
    31: 27,    # tie
    32: 28,    # suitcase
    33: 29,    # frisbee
    34: 30,    # skis
    35: 31,    # snowboard
    36: 32,    # sports ball
    37: 33,    # kite
    38: 34,    # baseball bat
    39: 35,    # baseball glove
    40: 36,    # skateboard
    41: 37,    # surfboard
    42: 38,    # tennis racket
    43: 39,    # bottle
    # 44: SKIP - plate
    45: 40,    # wine glass
    46: 41,    # cup
    47: 42,    # fork
    48: 43,    # knife
    49: 44,    # spoon
    50: 45,    # bowl
    51: 46,    # banana
    52: 47,    # apple
    53: 48,    # sandwich
    54: 49,    # orange
    55: 50,    # broccoli
    56: 51,    # carrot
    57: 52,    # hot dog
    58: 53,    # pizza
    59: 54,    # donut
    60: 55,    # cake
    61: 56,    # chair
    62: 57,    # couch
    63: 58,    # potted plant
    64: 59,    # bed
    # 65: SKIP - mirror
    66: 60,    # dining table
    # 67: SKIP - window
    # 68: SKIP - desk
    69: 61,    # toilet
    # 70: SKIP - door
    71: 62,    # tv
    72: 63,    # laptop
    73: 64,    # mouse
    74: 65,    # remote
    75: 66,    # keyboard
    76: 67,    # cell phone
    77: 68,    # microwave
    78: 69,    # oven
    79: 70,    # toaster
    80: 71,    # sink
    81: 72,    # refrigerator
    # 82: SKIP - blender
    83: 73,    # book
    84: 74,    # clock
    85: 75,    # vase
    86: 76,    # scissors
    87: 77,    # teddy bear
    88: 78,    # hair drier
    89: 79,    # toothbrush
    # 90: SKIP - hair brush
    # === Stuff classes (91-181) ===
    91: 80,    # banner
    92: 81,    # blanket
    93: 82,    # branch
    94: 83,    # bridge
    95: 84,    # building-other
    96: 85,    # bush
    97: 86,    # cabinet
    98: 87,    # cage
    99: 88,    # cardboard
    100: 89,   # carpet
    101: 90,   # ceiling-other
    102: 91,   # ceiling-tile
    103: 92,   # cloth
    104: 93,   # clothes
    105: 94,   # clouds
    106: 95,   # counter
    107: 96,   # cupboard
    108: 97,   # curtain
    109: 98,   # desk-stuff
    110: 99,   # dirt
    111: 100,  # door-stuff
    112: 101,  # fence
    113: 102,  # floor-marble
    114: 103,  # floor-other
    115: 104,  # floor-stone
    116: 105,  # floor-tile
    117: 106,  # floor-wood
    118: 107,  # flower
    119: 108,  # fog
    120: 109,  # food-other
    121: 110,  # fruit
    122: 111,  # furniture-other
    123: 112,  # grass
    124: 113,  # gravel
    125: 114,  # ground-other
    126: 115,  # hill
    127: 116,  # house
    128: 117,  # leaves
    129: 118,  # light
    130: 119,  # mat
    131: 120,  # metal
    132: 121,  # mirror-stuff
    133: 122,  # moss
    134: 123,  # mountain
    135: 124,  # mud
    136: 125,  # napkin
    137: 126,  # net
    138: 127,  # paper
    139: 128,  # pavement
    140: 129,  # pillow
    141: 130,  # plant-other
    142: 131,  # plastic
    143: 132,  # platform
    144: 133,  # playingfield
    145: 134,  # railing
    146: 135,  # railroad
    147: 136,  # river
    148: 137,  # road
    149: 138,  # rock
    150: 139,  # roof
    151: 140,  # rug
    152: 141,  # salad
    153: 142,  # sand
    154: 143,  # sea
    155: 144,  # shelf
    156: 145,  # sky-other
    157: 146,  # skyscraper
    158: 147,  # snow
    159: 148,  # solid-other
    160: 149,  # stairs
    161: 150,  # stone
    162: 151,  # straw
    163: 152,  # structural-other
    164: 153,  # table
    165: 154,  # tent
    166: 155,  # textile-other
    167: 156,  # towel
    168: 157,  # tree
    169: 158,  # vegetable
    170: 159,  # wall-brick
    171: 160,  # wall-concrete
    172: 161,  # wall-other
    173: 162,  # wall-panel
    174: 163,  # wall-stone
    175: 164,  # wall-tile
    176: 165,  # wall-wood
    177: 166,  # water-other
    178: 167,  # waterdrops
    179: 168,  # window-blind
    180: 169,  # window-other
    181: 170,  # wood
    255: 255   # ignore
}


def load_captions(root_dir, split):
    """Loads captions from COCO captions JSON file."""
    captions_path = os.path.join(root_dir, "annotations", f"captions_{split}2017.json")

    if not os.path.exists(captions_path):
        print(f"Warning: Captions file not found at {captions_path}")
        return {}

    with open(captions_path, 'r') as f:
        captions_data = json.load(f)

    captions_dict = {}
    for ann in captions_data['annotations']:
        image_id = ann['image_id']
        if image_id not in captions_dict:
            captions_dict[image_id] = []
        captions_dict[image_id].append(ann['caption'])

    return captions_dict


def remap_mask(mask_array):
    """Remap mask from COCO-Stuff 2017 indices to 171-class format."""
    # new_mask = np.full_like(mask_array, 255, dtype=np.uint8)  # Default to ignore
    new_mask = np.copy(mask_array)

    for old_idx, new_idx in COCO_STUFF_2017_TO_171.items():
        new_mask[mask_array == old_idx] = new_idx

    return new_mask


def process_coco_stuff(root_dir, output_dir, split):
    """Processes COCO-Stuff data for a given split."""
    images_dir = os.path.join(root_dir, f"{split}2017")
    masks_dir = os.path.join(root_dir, "annotations", f"{split}2017")

    # Load captions
    captions_dict = load_captions(root_dir, split)

    # Create remapped mask output directory
    mask_output_dir = os.path.join(output_dir, "masks", split)
    os.makedirs(mask_output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    processed_data = []
    for image_file in tqdm(image_files, desc=f"Processing COCO-Stuff {split} set"):
        image_id_str = os.path.splitext(image_file)[0]
        image_id_int = int(image_id_str)

        image_path = os.path.join(images_dir, image_file)
        orig_mask_path = os.path.join(masks_dir, f"{image_id_str}.png")

        if not os.path.exists(orig_mask_path):
            continue

        mask = Image.open(orig_mask_path)
        mask_array = np.array(mask, dtype=np.uint8)

        # Remap mask to 171-class format
        remapped_mask = remap_mask(mask_array)

        # Save remapped mask
        new_mask_path = os.path.join(mask_output_dir, f"{image_id_str}.png")
        Image.fromarray(remapped_mask).save(new_mask_path)

        # Get present classes (in new 0-170 index)
        present_class_indices = np.unique(remapped_mask)
        present_class_indices = [idx for idx in present_class_indices if idx < 171]

        if not present_class_indices:
            continue

        current_class_names = [COCO_STUFF_171_CLASSES[idx] for idx in present_class_indices]

        captions = captions_dict.get(image_id_int, [])

        processed_data.append({
            'image_id': image_id_str,
            'image_path': image_path,
            'mask_path': new_mask_path,
            'class_names': current_class_names,
            'captions': captions
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"coco_stuff_{split}.npy")
    np.save(output_path, processed_data)

    # Save class names (171 classes, NO background)
    class_names_path = os.path.join(output_dir, "coco_stuff_class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(COCO_STUFF_171_CLASSES, f)

    print(f"Processed {len(processed_data)} samples for the {split} split.")
    print(f"Saved processed data to {output_path}")
    print(f"Using 171 classes (background excluded)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO-Stuff segmentation data.")
    parser.add_argument("--root_dir", type=str, default="./data/coco_stuff",
                        help="Root directory of the COCO-Stuff dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/coco_stuff_processed",
                        help="Directory to save processed data.")
    args = parser.parse_args()

    process_coco_stuff(args.root_dir, args.output_dir, "train")
    process_coco_stuff(args.root_dir, args.output_dir, "val")


if __name__ == "__main__":
    main()
