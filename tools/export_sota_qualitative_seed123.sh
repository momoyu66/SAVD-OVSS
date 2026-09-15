#!/usr/bin/env bash
set -euo pipefail

# Export the final h1536 ASFD+VADD prediction for the exact images used by the
# released Talk2DINO qualitative comparison. Run from the repository root.

CONFIG="${CONFIG:-configs/dinode_eval_local.json}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/eccv26_dinode_coco_stuff.pth}"
FINAL_CKPT="${FINAL_CKPT:-experiments/asfd_capacity_h1536_20260906_v1/run/vadd/epoch_01.pth}"
INPUT_ROOT="${INPUT_ROOT:-tools/qualitative_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sota_qualitative_seed123}"

for required in \
    "$CONFIG" \
    "$TEACHER_CKPT" \
    "$FINAL_CKPT" \
    "$INPUT_ROOT/voc/1_img.png" \
    "$INPUT_ROOT/voc/2_img.png" \
    "$INPUT_ROOT/context/3r_image.png" \
    "$INPUT_ROOT/stuff/1r_image.png"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required artifact: $required" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python tools/export_sota_qualitative.py \
    --config "$CONFIG" \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --student-checkpoint "$FINAL_CKPT" \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed 123

tar -czf "${OUTPUT_DIR}.tar.gz" \
    -C "$OUTPUT_DIR" \
    metadata.json \
    voc_1_mask.png voc_1_ours.png \
    voc_2_mask.png voc_2_ours.png \
    context_3r_mask.png context_3r_ours.png \
    stuff_1r_mask.png stuff_1r_ours.png

echo "Packaged paper-ready outputs as: ${OUTPUT_DIR}.tar.gz"

