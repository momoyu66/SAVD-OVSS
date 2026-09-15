#!/usr/bin/env bash
set -euo pipefail

# Export matched qualitative predictions for the introduction teaser.
# Run from the DINOde_ASFD_VADD repository root on the experiment server.

CONFIG="${CONFIG:-configs/dinode_eval_local.json}"
DATA_DIR="${CITYSCAPES_DATA_DIR:-data/cityscapes_processed}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/eccv26_dinode_coco_stuff.pth}"
ASFD_CKPT="${ASFD_CKPT:-experiments/asfd_capacity_h1536_20260906_v1/run/asfd_h1536_seed123.pth}"
FINAL_CKPT="${FINAL_CKPT:-experiments/asfd_capacity_h1536_20260906_v1/run/vadd/epoch_01.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/intro_cityscapes_seed123}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

for required in \
    "$CONFIG" \
    "$DATA_DIR/cityscapes_val.npy" \
    "$TEACHER_CKPT" \
    "$ASFD_CKPT" \
    "$FINAL_CKPT"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required artifact: $required" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_eval() {
    local name="$1"
    local student_ckpt="$2"

    if [[ -n "$student_ckpt" ]]; then
        export DINODE_FLOW_STUDENT="$student_ckpt"
    else
        unset DINODE_FLOW_STUDENT || true
    fi

    python eval_flow_distill.py \
        --config "$CONFIG" \
        --checkpoint "$TEACHER_CKPT" \
        --output_dir "$OUTPUT_ROOT/$name" \
        --val_dataset cityscapes \
        --cityscapes_data_dir "$DATA_DIR" \
        --max_samples "$MAX_SAMPLES" \
        --seed 123 \
        --no_cache
}

run_eval teacher ""
run_eval asfd "$ASFD_CKPT"
run_eval asfd_vadd "$FINAL_CKPT"

unset DINODE_FLOW_STUDENT || true

echo
echo "Export complete. Each timestamped run contains:"
echo "  validation_visualizations/cityscapes/epoch_*/sample_000.png ..."
echo "Copy the same sample index from teacher, asfd, and asfd_vadd."

if command -v tar >/dev/null 2>&1; then
    tar -czf "${OUTPUT_ROOT}.tar.gz" "$OUTPUT_ROOT"
    echo "Packaged matched outputs as: ${OUTPUT_ROOT}.tar.gz"
fi
