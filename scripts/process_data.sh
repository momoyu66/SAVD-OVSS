#!/bin/bash
# Preprocess all datasets used by the eight open-vocabulary evaluation protocols.
#
# Prerequisite: the raw datasets must be reachable under ./data (see README.md).
# Each processor writes a `<dataset>_{train,val}.npy` index plus remapped label masks
# into its output directory; train.py / eval.py read only these processed directories.
#
# Usage:
#   bash scripts/process_data.sh

set -e

echo "========================================"
echo "Dataset preprocessing for DINOde"
echo "========================================"

# 1. PASCAL VOC  -> VOC21 (BG include) / VOC20 (BG exclude)
#    Expects ./data/VOCdevkit/VOC2012/
echo ""
echo "[1/6] PASCAL VOC"
python processing/pascal_voc/pascal_voc_processor.py \
    --root_dir ./data \
    --output_dir ./data/voc_processed

# 2. PASCAL Context -> Context60 (BG include) / Context59 (BG exclude)
#    Expects ./data/VOCdevkit/VOC2010/ plus the Context annotations
echo ""
echo "[2/6] PASCAL Context"
python processing/pascal_context/pascal_context_processor.py \
    --root_dir ./data \
    --output_dir ./data/pascal_context_processed

# 3. COCO-Object (81 classes, BG include) - derived from the COCO-Stuff annotations
#    Expects ./data/coco_stuff/{train2017,val2017,annotations}
#    The full 5,000-image val2017 split is kept; hence the directory suffix.
echo ""
echo "[3/6] COCO-Object"
python processing/coco_object/coco_object_processor.py \
    --root_dir ./data/coco_stuff \
    --output_dir ./data/coco_object_processed_5000

# 4. COCO-Stuff (171 classes, no BG) - also the training set
#    Expects ./data/coco_stuff/{train2017,val2017,annotations,annotations_stufftingmaps}
echo ""
echo "[4/6] COCO-Stuff"
python processing/coco_stuff/coco_stuff_processor.py \
    --root_dir ./data/coco_stuff \
    --output_dir ./data/coco_stuff_processed

# 5. Cityscapes (19 classes, no BG)
#    Expects ./data/cityscapes/{leftImg8bit,gtFine}
echo ""
echo "[5/6] Cityscapes"
python processing/cityscapes/cityscapes_processor.py \
    --root_dir ./data/cityscapes \
    --output_dir ./data/cityscapes_processed

# 6. ADE20K (150 classes, BG exclude)
#    Expects ./data/ADEChallengeData2016/
echo ""
echo "[6/6] ADE20K"
python processing/ade20k/ade20k_processor.py \
    --root_dir ./data \
    --output_dir ./data/ade20k_processed

echo ""
echo "========================================"
echo "Preprocessing complete."
echo "========================================"
echo "  PASCAL VOC:     ./data/voc_processed"
echo "  PASCAL Context: ./data/pascal_context_processed"
echo "  COCO-Object:    ./data/coco_object_processed_5000"
echo "  COCO-Stuff:     ./data/coco_stuff_processed"
echo "  Cityscapes:     ./data/cityscapes_processed"
echo "  ADE20K:         ./data/ade20k_processed"
