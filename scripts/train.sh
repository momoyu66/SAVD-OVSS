#!/bin/bash
# Train DINOde on COCO-Stuff captions and evaluate on all eight OV-SS protocols.
#
# Usage:
#   bash scripts/train.sh
#
# Extra arguments are forwarded to the Python entry point, e.g.:
#   bash scripts/train.sh --no_cache
#
# Override the GPU or output directory from the environment:
#   GPU=0 OUTPUT_DIR=outputs/my_run bash scripts/train.sh

set -e

GPU="${GPU:-0}"
CONFIG_FILE="${CONFIG_FILE:-configs/dinode_coco_stuff.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dinode_coco_stuff}"

# Processed dataset directories (produced by scripts/process_data.sh)
COCO_STUFF_DATA_DIR="./data/coco_stuff_processed"
PASCAL_VOC_DATA_DIR="./data/voc_processed"
PASCAL_CONTEXT_DATA_DIR="./data/pascal_context_processed"
COCO_OBJECT_DATA_DIR="./data/coco_object_processed_5000"
CITYSCAPES_DATA_DIR="./data/cityscapes_processed"
ADE20K_DATA_DIR="./data/ade20k_processed"

# Eight evaluation protocols run after every epoch.
#   BG include: pascal_voc (VOC21), pascal_context (Context60), coco_object (81 classes)
#   BG exclude: voc20, context59, coco_stuff (171), cityscapes (19), ade20k (150)
VAL_DATASETS="${VAL_DATASETS:-pascal_voc,voc20,pascal_context,context59,coco_object,coco_stuff,cityscapes,ade20k}"

echo "========================================"
echo "DINOde training"
echo "========================================"

if [ ! -d "$COCO_STUFF_DATA_DIR" ]; then
    echo "Error: processed COCO-Stuff data not found at $COCO_STUFF_DATA_DIR"
    echo "Run 'bash scripts/process_data.sh' first."
    exit 1
fi

echo "Config:      $CONFIG_FILE"
echo "Output:      $OUTPUT_DIR"
echo "Protocols:   $VAL_DATASETS"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --processed_data_dir "$COCO_STUFF_DATA_DIR" \
    --val_dataset "$VAL_DATASETS" \
    --pascal_voc_data_dir "$PASCAL_VOC_DATA_DIR" \
    --pascal_context_data_dir "$PASCAL_CONTEXT_DATA_DIR" \
    --coco_object_data_dir "$COCO_OBJECT_DATA_DIR" \
    --coco_stuff_data_dir "$COCO_STUFF_DATA_DIR" \
    --cityscapes_data_dir "$CITYSCAPES_DATA_DIR" \
    --ade20k_data_dir "$ADE20K_DATA_DIR" \
    "$@"

echo ""
echo "========================================"
echo "Training complete. Results in $OUTPUT_DIR"
echo "========================================"
