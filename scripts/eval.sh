#!/bin/bash
# Evaluate a DINOde checkpoint on the eight open-vocabulary segmentation protocols.
#
# With the defaults below this reproduces the numbers reported in the paper
# (see the results table in README.md).
#
# Usage:
#   bash scripts/eval.sh
#
# Extra arguments are forwarded to the Python entry point, e.g.:
#   bash scripts/eval.sh --no_cache
#
# Override from the environment:
#   GPU=1 CHECKPOINT=outputs/my_run/checkpoints/checkpoint_epoch_009.pth bash scripts/eval.sh

set -e

GPU="${GPU:-0}"
CONFIG_FILE="${CONFIG_FILE:-configs/dinode_eval.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/eccv26_dinode_coco_stuff.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_dinode}"

# Processed dataset directories (produced by scripts/process_data.sh)
COCO_STUFF_DATA_DIR="./data/coco_stuff_processed"
PASCAL_VOC_DATA_DIR="./data/voc_processed"
PASCAL_CONTEXT_DATA_DIR="./data/pascal_context_processed"
COCO_OBJECT_DATA_DIR="./data/coco_object_processed_5000"
CITYSCAPES_DATA_DIR="./data/cityscapes_processed"
ADE20K_DATA_DIR="./data/ade20k_processed"

VAL_DATASETS="${VAL_DATASETS:-pascal_voc,voc20,pascal_context,context59,coco_object,coco_stuff,cityscapes,ade20k}"

echo "========================================"
echo "DINOde evaluation"
echo "========================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Config:      $CONFIG_FILE"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT_DIR"
echo "Protocols:   $VAL_DATASETS"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" python eval.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
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
echo "Evaluation complete. Results in $OUTPUT_DIR/eval_log.txt"
echo "========================================"
