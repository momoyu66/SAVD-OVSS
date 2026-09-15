#!/usr/bin/env python3
"""Export ASFD+VADD predictions for qualitative comparisons.

With ``--manifest``, this script runs the final h1536 seed-123 student on a
fresh multi-dataset shortlist using the same open-vocabulary protocols as the
quantitative evaluation.  It writes raw label maps for a fair, uniformly
styled comparison and optional overlays for quick inspection.

Run from the DINOde_ASFD_VADD repository root.  The resulting ``*_ours.png``
files can be copied back and passed to ``paper_draft/figures/``'s compositor.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_flow_distill import EvalPipeline  # noqa: E402
from model.components import build_prompts  # noqa: E402
from model.flow_student import SingleStepSphericalStudent  # noqa: E402
from processing.coco_stuff.coco_stuff_processor import (  # noqa: E402
    COCO_STUFF_171_CLASSES,
)
from processing.pascal_context.pascal_context_processor import (  # noqa: E402
    PASCAL_CONTEXT_CLASSES,
)
from processing.pascal_voc.pascal_voc_processor import (  # noqa: E402
    PASCAL_VOC_CLASSES,
)


BACKGROUND_PROMPTS = (
    "ground, land, grass, tree, building, wall, sky, lake, water, river, sea, "
    "railway, railroad, keyboard, helmet, cloud, house, mountain, ocean, road, "
    "rock, street, valley, bridge, sign"
)


SAMPLES = (
    ("voc_1", "voc/1_img.png", "pascal_voc"),
    ("voc_2", "voc/2_img.png", "pascal_voc"),
    ("context_3r", "context/3r_image.png", "pascal_context"),
    ("stuff_1r", "stuff/1r_image.png", "coco_stuff"),
)

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final ASFD+VADD on the four SOTA qualitative examples."
    )
    parser.add_argument(
        "--config",
        default="configs/dinode_eval_local.json",
        help="Evaluation JSON config.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoints/eccv26_dinode_coco_stuff.pth",
        help="DINOde teacher/head checkpoint.",
    )
    parser.add_argument(
        "--student-checkpoint",
        default="checkpoints/flow_student_asfd_vadd_h1536_seed123.pth",
        help="Final h1536 ASFD+VADD seed-123 checkpoint.",
    )
    parser.add_argument(
        "--input-root",
        default="tools/qualitative_inputs",
        help="Directory containing voc/, context/, and stuff/ input images.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest with a samples list. Each item needs "
            "name, image, and protocol fields; this enables fresh selected "
            "examples such as Cityscapes without changing the script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sota_qualitative_seed123",
        help="Directory for masks, overlays, and run metadata.",
    )
    parser.add_argument(
        "--ade20k-class-file",
        type=Path,
        default=Path("external/GLA-CLIP/configs/cls_ade20k.txt"),
        help="One ADE20K class name per line, in evaluation order.",
    )
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--max-long-side", type=int, default=2048)
    parser.add_argument(
        "--voc-threshold",
        type=float,
        default=0.26,
        help="Frozen VOC21 confidence threshold from the paper ledger.",
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=0.09,
        help="Frozen Context60 confidence threshold from the paper ledger.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.62,
        help="Mask opacity in the colored foreground region.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Also save dense float32 confidence maps (large; off by default).",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_student(pipeline: EvalPipeline, checkpoint_path: Path) -> Dict:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    state = checkpoint["student_state_dict"]
    geometry = checkpoint.get("student_geometry", "tangent_residual")
    student = SingleStepSphericalStudent(
        init_weight=state["text_flow_init.weight"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        geometry=geometry,
    )
    student.load_state_dict(state)
    student = student.to(pipeline.device).eval()
    for parameter in student.parameters():
        parameter.requires_grad = False

    pipeline.flow_student = student

    def apply_distilled_flow(text_features: torch.Tensor) -> torch.Tensor:
        return pipeline.flow_student(
            text_features.to(device=pipeline.device, dtype=torch.float32)
        )

    pipeline.head.apply_text_flow = apply_distilled_flow
    return checkpoint


def protocol_spec(
    protocol: str, ade20k_classes: List[str] | None = None
) -> Tuple[List[str], int, float | None]:
    if protocol == "pascal_voc":
        # VOC21 uses multiple background concepts, max-pooled to one channel.
        return [BACKGROUND_PROMPTS, *PASCAL_VOC_CLASSES[1:]], 0, None
    if protocol == "pascal_context":
        return list(PASCAL_CONTEXT_CLASSES), 0, None
    if protocol == "coco_stuff":
        return list(COCO_STUFF_171_CLASSES), -1, None
    if protocol == "cityscapes":
        return list(CITYSCAPES_CLASSES), -1, None
    if protocol == "ade20k":
        if not ade20k_classes:
            raise ValueError("ADE20K protocol requires --ade20k-class-file")
        return list(ade20k_classes), -1, None
    raise ValueError(f"Unsupported qualitative protocol: {protocol}")


def encode_protocol(
    pipeline: EvalPipeline,
    protocol: str,
    ade20k_classes: List[str] | None = None,
) -> Tuple[torch.Tensor, int]:
    class_names, _, _ = protocol_spec(protocol, ade20k_classes)
    if class_names and "," in class_names[0]:
        background_names = [name.strip() for name in class_names[0].split(",")]
        background_channels = len(background_names)
        flat_names = background_names + class_names[1:]
    else:
        background_channels = 1
        flat_names = class_names

    with torch.no_grad():
        clip_features = torch.stack(
            [
                pipeline.text_encoder.encode(
                    build_prompts(name, protocol), aggregate="mean"
                ).squeeze(0)
                for name in flat_names
            ],
            dim=0,
        )
        text_features = pipeline.trainer._encode_texts(
            flat_names, cached_clip_embeddings=clip_features
        )
    return text_features, background_channels


def predict(
    pipeline: EvalPipeline,
    image: Image.Image,
    text_features: torch.Tensor,
    background_channels: int,
    threshold: float | None,
    crop_size: int,
    stride: int,
    max_long_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    head_fn = lambda features: pipeline.head(features, text_features)[0]
    with torch.no_grad():
        logits = pipeline._sliding_window_inference(
            image,
            head_fn,
            crop_size=crop_size,
            stride=stride,
            max_long_side=max_long_side,
        )
        logits = F.interpolate(
            logits.unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        if background_channels > 1:
            background = logits[:, :background_channels].max(
                dim=1, keepdim=True
            )[0]
            logits = torch.cat(
                [background, logits[:, background_channels:]], dim=1
            )
        probabilities = F.softmax(logits, dim=1)
        confidence, labels = probabilities.max(dim=1)
        if threshold is not None:
            labels = labels.masked_fill(confidence < threshold, 0)
    return (
        labels.squeeze(0).cpu().numpy().astype(np.uint8),
        confidence.squeeze(0).cpu().numpy().astype(np.float32),
    )


def palette_for(pipeline: EvalPipeline, protocol: str) -> np.ndarray:
    if protocol == "pascal_voc":
        palette = pipeline._get_voc_palette()
    elif protocol == "pascal_context":
        palette = pipeline._get_voc_context_palette()
    elif protocol == "coco_stuff":
        palette = pipeline._get_coco_stuff_palette()
    elif protocol == "cityscapes":
        palette = pipeline._get_cityscapes_palette()
    elif protocol == "ade20k":
        # Entry 0 of the evaluator palette is the ignore color; predictions
        # use contiguous 0--149 train IDs, so take the following 150 colors.
        palette = pipeline._get_ade20k_palette()[1:]
    else:
        raise ValueError(protocol)
    return np.rint(palette * 255.0).astype(np.uint8)


def make_overlay(
    image: Image.Image,
    labels: np.ndarray,
    palette: np.ndarray,
    protocol: str,
    alpha: float,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = palette[np.clip(labels, 0, len(palette) - 1)].astype(np.float32)

    # Match the released comparison style: retain image detail, darken the
    # unselected background, and use a translucent categorical color overlay.
    result = rgb * 0.42
    if protocol in {"pascal_voc", "pascal_context"}:
        foreground = labels != 0
    else:
        foreground = np.ones(labels.shape, dtype=bool)
    blended = (1.0 - alpha) * rgb + alpha * color
    result[foreground] = blended[foreground]
    return np.clip(result, 0, 255).astype(np.uint8)


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing {label}:\n  {formatted}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config_path = Path(args.config)
    teacher_path = Path(args.teacher_checkpoint)
    student_path = Path(args.student_checkpoint)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = SAMPLES
    if args.manifest is not None:
        with args.manifest.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        samples = tuple(
            (item["name"], item["image"], item["protocol"])
            for item in manifest["samples"]
        )

    ade20k_classes = None
    if any(protocol == "ade20k" for _, _, protocol in samples):
        require_files([args.ade20k_class_file], "ADE20K class-name file")
        ade20k_classes = [
            line.strip()
            for line in args.ade20k_class_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(ade20k_classes) != 150:
            raise ValueError(
                f"Expected 150 ADE20K classes, found {len(ade20k_classes)}"
            )

    input_paths = [input_root / relative for _, relative, _ in samples]
    require_files(
        [config_path, teacher_path, student_path, *input_paths], "input files"
    )

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config["flow"]["infer_steps"] = []
    config.setdefault("validation", {})["use_pamr"] = False

    runtime_dir = output_dir / "runtime"
    pipeline = EvalPipeline(
        config, output_dir=str(runtime_dir), num_classes=171, no_cache=True
    )
    pipeline.load_checkpoint_for_eval(str(teacher_path))
    student_checkpoint = load_student(pipeline, student_path)
    pipeline.backbone.eval()
    pipeline.text_encoder.eval()
    pipeline.head.eval()

    encoded: Dict[str, Tuple[torch.Tensor, int]] = {}
    records = []
    for name, relative, protocol in samples:
        image_path = input_root / relative
        image = Image.open(image_path).convert("RGB")
        if protocol not in encoded:
            encoded[protocol] = encode_protocol(
                pipeline, protocol, ade20k_classes=ade20k_classes
            )
        text_features, background_channels = encoded[protocol]

        threshold = None
        if protocol == "pascal_voc":
            threshold = args.voc_threshold
        elif protocol == "pascal_context":
            threshold = args.context_threshold

        labels, confidence = predict(
            pipeline,
            image,
            text_features,
            background_channels,
            threshold,
            crop_size=args.crop_size,
            stride=args.stride,
            max_long_side=args.max_long_side,
        )
        palette = palette_for(pipeline, protocol)
        overlay = make_overlay(
            image, labels, palette, protocol, alpha=args.overlay_alpha
        )

        mask_path = output_dir / f"{name}_mask.png"
        overlay_path = output_dir / f"{name}_ours.png"
        Image.fromarray(labels, mode="L").save(mask_path)
        Image.fromarray(overlay, mode="RGB").save(overlay_path)

        confidence_path = None
        if args.save_confidence:
            confidence_path = output_dir / f"{name}_confidence.npy"
            np.save(confidence_path, confidence)

        records.append(
            {
                "sample": name,
                "input": str(image_path),
                "protocol": protocol,
                "threshold": threshold,
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "confidence": str(confidence_path) if confidence_path else None,
                "size": [image.width, image.height],
            }
        )
        print(f"[saved] {overlay_path}")

    metadata = {
        "method": "ASFD+VADD",
        "student": str(student_path),
        "student_hidden_dim": int(student_checkpoint["hidden_dim"]),
        "student_geometry": student_checkpoint.get(
            "student_geometry", "tangent_residual"
        ),
        "seed": args.seed,
        "teacher": str(teacher_path),
        "config": str(config_path),
        "crop_size": args.crop_size,
        "stride": args.stride,
        "max_long_side": args.max_long_side,
        "overlay_alpha": args.overlay_alpha,
        "samples": records,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    pipeline.writer.close()
    print(f"[done] Exported {len(records)} qualitative predictions to {output_dir}")


if __name__ == "__main__":
    main()
