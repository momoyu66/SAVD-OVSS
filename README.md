# SAVD: Decision-Aware Text Alignment for Open-Vocabulary Segmentation

This repository contains the research code for **SAVD**, a one-pass student
that replaces the ten-evaluation spherical text flow used by DINOde while
preserving the visual class decisions induced by the teacher.

SAVD is designed for **dynamic-vocabulary OVSS**, where user queries or class
banks are refreshed repeatedly. It accelerates prototype alignment. It does
not accelerate DINOv3 image encoding or the complete segmentation pipeline.

## Method

- **Accelerated Spherical Flow Distillation (ASFD)** learns the terminal
  teacher prototype with a normalized residual student initialized from the
  frozen teacher projection.
- **Visual Anchor Decision Distillation (VADD)** compares teacher and student
  class distributions on unlabeled, frozen DINOv3 patch features.
- The teacher and visual anchors are used only during training. Inference maps
  a new class vocabulary with one student pass.

## Paper results

The paper reports one fixed run with hidden dimension 1536:

| Result | Value |
| --- | ---: |
| Ten-evaluation DINOde teacher, eight-protocol macro mIoU | 49.50 |
| SAVD, eight-protocol macro mIoU | 49.56 |
| Matched ASFD-only continuation | 49.36 |
| Matched VADD gain | +0.193 |
| Independent-anchor decision KL, ASFD-only / SAVD | 0.00496 / 0.00174 |
| Independent-anchor low-margin top-1, ASFD-only / SAVD | 71.06 / 80.56 |
| Cached-embedding alignment speedup | 50.25x |
| Raw-name vocabulary-refresh speedup | 1.001--1.041x |
| Full Cityscapes latency, SAVD / teacher | 407.12 / 411.53 ms/image |

The `results/paper/` directory contains the machine-readable records used for
these claims. The raw-name and full-pipeline measurements show that the main
deployment benefit occurs when aligned prototypes must be regenerated often.

## Repository layout

```text
model/flow_student.py              One-pass spherical student
utils/anchor_split.py              Image-group visual-anchor split
train_flow_distill.py              ASFD endpoint training
train_flow_distill_vadd.py         Matched continuation and VADD training
eval_flow_distill.py               Eight-protocol student evaluation
eval_flow_distill_deploy.py        Student-only deployment path
eval_deployment_audit_*.py         Full-pipeline latency audit
benchmark_flow_efficiency.py       Cached-embedding alignment benchmark
benchmark_vocabulary_refresh.py    Raw-name vocabulary-refresh benchmark
tools/extract_visual_anchors.py    Visual-anchor cache builder
tools/verify_repository.py         Lightweight release verification
assets/taxonomy/                   VADD training class bank
processing/                        Dataset preparation and evaluation loaders
results/paper/                     Paper-facing JSON evidence
```

Model weights, datasets, feature caches, tokens, and generated outputs are not
included.

## Environment

The reported experiments used Python 3.10, PyTorch 2.6.0 with CUDA 11.8,
Transformers 4.56.2, OpenCLIP 2.20.0, and Quadro RTX 6000 GPUs.

Install PyTorch for the local CUDA runtime first, then install the remaining
dependencies:

```bash
conda create -n savd python=3.10
conda activate savd
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python tools/verify_repository.py
```

## External assets

Follow `checkpoints/README.md` and `data/README.md`. The default configuration
expects the following local files:

```text
checkpoints/facebook_dinov3_vitl16_pretrain_lvd1689m_hf/
checkpoints/open_clip_pytorch_model.bin
checkpoints/eccv26_dinode_coco_stuff.pth
data/coco/annotations/captions_train2017.json
data/flow_distill_vadd/coco_train_2k_64anchors_fp16.pth
```

The COCO-Stuff training taxonomy is included at
`assets/taxonomy/coco_stuff_171_classes.json`.

## Build visual anchors

```bash
python tools/extract_visual_anchors.py \
  --config configs/dinode_eval_local.json \
  --image-root data/coco/train2017 \
  --output data/flow_distill_vadd/coco_train_2k_64anchors_fp16.pth \
  --num-images 2000 \
  --anchors-per-image 64 \
  --seed 123
```

## Train

ASFD endpoint training:

```bash
python train_flow_distill.py \
  --config configs/dinode_eval_local.json \
  --checkpoint checkpoints/eccv26_dinode_coco_stuff.pth \
  --captions data/coco/annotations/captions_train2017.json \
  --output checkpoints/flow_student_asfd_h1536_seed123.pth \
  --max-captions 200000 \
  --hidden-dim 1536 \
  --epochs 20 \
  --relation-weight 0.0 \
  --student-geometry ambient_residual \
  --seed 123
```

VADD continuation:

```bash
python train_flow_distill_vadd.py \
  --config configs/dinode_eval_local.json \
  --checkpoint checkpoints/eccv26_dinode_coco_stuff.pth \
  --initial-student checkpoints/flow_student_asfd_h1536_seed123.pth \
  --captions data/coco/annotations/captions_train2017.json \
  --anchors data/flow_distill_vadd/coco_train_2k_64anchors_fp16.pth \
  --class-names assets/taxonomy/coco_stuff_171_classes.json \
  --output checkpoints/flow_student_savd_h1536_seed123.pth \
  --class-weight 0.1 \
  --vadd-weight 0.1 \
  --distill-temperature 2.0 \
  --epochs 1 \
  --max-train-batches 200 \
  --seed 123
```

For the matched ASFD-only control, keep the same command and set
`--vadd-weight 0.0`.

## Evaluate

```bash
export DINODE_FLOW_STUDENT=checkpoints/flow_student_savd_h1536_seed123.pth
python eval_flow_distill.py \
  --config configs/dinode_eval_local.json \
  --checkpoint checkpoints/eccv26_dinode_coco_stuff.pth \
  --output_dir outputs/eval_cityscapes \
  --val_dataset cityscapes \
  --cityscapes_data_dir data/cityscapes_processed \
  --no_cache
```

Run `python eval_flow_distill.py --help` for the switches used by the other
seven evaluation protocols.

## Upstream and usage terms

This implementation builds on
[DINOde](https://github.com/yoon307/DINOde) at commit
`0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`. See `NOTICE.md` before reuse.
The repository contains no DINOv3, OpenCLIP, DINOde, or dataset weights.
