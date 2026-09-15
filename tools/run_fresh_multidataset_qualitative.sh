#!/usr/bin/env bash
set -euo pipefail

# Run from the DINOde_ASFD_VADD repository root.
DINO_ROOT="$(pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DINO_ROOT}/outputs/fresh_multidataset_qualitative}"
GLA_ROOT="${GLA_ROOT:-${DINO_ROOT}/external/GLA-CLIP}"
GPU="${GPU:-0}"
CANDIDATES_PER_DATASET="${CANDIDATES_PER_DATASET:-6}"

if [[ ! -f "${DINO_ROOT}/eval_flow_distill.py" ]]; then
  echo "[error] Run this script from the DINOde_ASFD_VADD repository root." >&2
  exit 1
fi
if [[ ! -f "${GLA_ROOT}/eval.py" ]]; then
  echo "[error] Missing ${GLA_ROOT}/eval.py. Extract the complete server kit first." >&2
  exit 1
fi

python - <<'PY'
import importlib
required = ("torch", "numpy", "PIL", "mmcv", "mmengine", "mmseg", "timm", "einops")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing GLA-CLIP runtime dependencies:\n  " + "\n  ".join(missing))
print("[ok] Python dependencies are available")
PY

python tools/prepare_fresh_multidataset_qualitative.py \
  --repo-root "${DINO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --candidates-per-dataset "${CANDIDATES_PER_DATASET}"

STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${DINO_ROOT}/checkpoints/flow_student_asfd_vadd_h1536_seed123.pth}"
if [[ ! -f "${STUDENT_CHECKPOINT}" ]]; then
  FALLBACK_STUDENT="${DINO_ROOT}/experiments/asfd_capacity_h1536_20260906_v1/run/vadd/epoch_01.pth"
  if [[ -f "${FALLBACK_STUDENT}" ]]; then
    STUDENT_CHECKPOINT="${FALLBACK_STUDENT}"
  else
    echo "[error] Cannot find the final h1536 seed-123 student checkpoint." >&2
    exit 1
  fi
fi

EXPECTED_OURS=$((4 * CANDIDATES_PER_DATASET))
mkdir -p "${OUTPUT_ROOT}/ours"
EXISTING_OURS=$(find "${OUTPUT_ROOT}/ours" -maxdepth 1 -type f -name '*_mask.png' | wc -l)
if (( EXISTING_OURS >= EXPECTED_OURS )); then
  echo "[skip] ASFD+VADD already has ${EXISTING_OURS} masks"
else
  CUDA_VISIBLE_DEVICES="${GPU}" python tools/export_sota_qualitative.py \
    --config configs/dinode_eval_local.json \
    --teacher-checkpoint checkpoints/eccv26_dinode_coco_stuff.pth \
    --student-checkpoint "${STUDENT_CHECKPOINT}" \
    --input-root "${OUTPUT_ROOT}" \
    --manifest "${OUTPUT_ROOT}/manifest.json" \
    --ade20k-class-file "${GLA_ROOT}/configs/cls_ade20k.txt" \
    --output-dir "${OUTPUT_ROOT}/ours" \
    --seed 123
fi

declare -A CONFIGS=(
  [voc21]="configs/cfg_qual_voc21.py"
  [context60]="configs/cfg_qual_context60.py"
  [coco_stuff]="configs/cfg_qual_coco_stuff.py"
  [ade20k]="configs/cfg_qual_ade20k.py"
)

run_gla_method() {
  local method_dir="$1"
  local clip_type="$2"
  local dataset="$3"
  shift 3
  export GLA_QUAL_DATA_ROOT="${OUTPUT_ROOT}/datasets/${dataset}"
  export GLA_RAW_PRED_DIR="${OUTPUT_ROOT}/predictions/${method_dir}/${dataset}"
  mkdir -p "${GLA_RAW_PRED_DIR}"
  local existing_count
  existing_count=$(find "${GLA_RAW_PRED_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)
  if (( existing_count >= CANDIDATES_PER_DATASET )); then
    echo "[skip] ${method_dir} on ${dataset} already has ${existing_count} masks"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python eval.py \
    --config "${CONFIGS[${dataset}]}" \
    --work_dir "${OUTPUT_ROOT}/runtime/${method_dir}/${dataset}" \
    --show_dir "${OUTPUT_ROOT}/runtime/${method_dir}/${dataset}/visualize" \
    --CLIP_type "${clip_type}" \
    "$@"
}

cd "${GLA_ROOT}"
for dataset in voc21 context60 coco_stuff ade20k; do
  echo "[run] ClearCLIP on ${dataset}"
  run_gla_method clearclip ClearCLIP "${dataset}"

  echo "[run] NACLIP on ${dataset}"
  run_gla_method naclip NACLIP "${dataset}"

  echo "[run] ProxyCLIP on ${dataset}"
  run_gla_method proxyclip ProxyCLIP "${dataset}"

  echo "[run] GLA-CLIP on ${dataset}"
  run_gla_method gla_clip ProxyCLIP "${dataset}" \
    --token_norm \
    --KV_token_extension \
    --proxy_sim \
    --mini_iters 2 \
    --initial_crit_pos 0.6 \
    --dynamic_beta \
    --beta_alpha 0.3 \
    --dynamic_gamma \
    --gamma_alpha 30
done

cd "${DINO_ROOT}"
python tools/collect_fresh_multidataset_qualitative.py --root "${OUTPUT_ROOT}"

ARCHIVE="${DINO_ROOT}/outputs/fresh_multidataset_qualitative_results.tar.gz"
tar -czf "${ARCHIVE}" \
  --exclude="$(basename "${OUTPUT_ROOT}")/runtime" \
  --exclude="$(basename "${OUTPUT_ROOT}")/ours/runtime" \
  --exclude="$(basename "${OUTPUT_ROOT}")/ours/*_ours.png" \
  -C "$(dirname "${OUTPUT_ROOT}")" \
  "$(basename "${OUTPUT_ROOT}")/manifest.json" \
  "$(basename "${OUTPUT_ROOT}")/results_manifest.json" \
  "$(basename "${OUTPUT_ROOT}")/datasets" \
  "$(basename "${OUTPUT_ROOT}")/predictions" \
  "$(basename "${OUTPUT_ROOT}")/ours"

echo "[done] Download this archive: ${ARCHIVE}"
