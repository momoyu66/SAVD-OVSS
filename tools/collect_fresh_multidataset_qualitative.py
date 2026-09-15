#!/usr/bin/env python3
"""Validate and summarize the fresh qualitative prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("clearclip", "naclip", "proxyclip", "gla_clip")
NUM_CLASSES = {
    "voc21": 21,
    "context60": 60,
    "coco_stuff": 171,
    "ade20k": 150,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/fresh_multidataset_qualitative"),
    )
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def normalize_gt(mask: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "ade20k":
        valid = (mask >= 1) & (mask <= 150)
        normalized = np.full(mask.shape, 255, dtype=np.uint8)
        normalized[valid] = mask[valid] - 1
        return normalized, valid
    n_cls = NUM_CLASSES[dataset]
    valid = mask < n_cls
    normalized = np.full(mask.shape, 255, dtype=np.uint8)
    normalized[valid] = mask[valid]
    return normalized, valid


def resize_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if prediction.shape == shape:
        return prediction
    image = Image.fromarray(prediction, mode="L")
    image = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def image_miou(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    num_classes: int,
) -> float:
    prediction = prediction.copy()
    prediction[prediction >= num_classes] = 255
    present = np.unique(target[valid])
    ious = []
    for class_id in present:
        pred_class = (prediction == class_id) & valid
        target_class = (target == class_id) & valid
        union = np.logical_or(pred_class, target_class).sum()
        if union:
            intersection = np.logical_and(pred_class, target_class).sum()
            ious.append(float(intersection / union))
    return float(np.mean(ious)) if ious else 0.0


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    summarized = []
    missing = []
    for sample in manifest["samples"]:
        dataset = sample["dataset"]
        gt_path = root / sample["mask"]
        gt_raw = load_mask(gt_path)
        gt, valid = normalize_gt(gt_raw, dataset)
        prediction_paths = {
            "ours": root / "ours" / f"{sample['name']}_mask.png",
            **{
                method: root
                / "predictions"
                / method
                / dataset
                / f"{sample['gla_image_stem']}.png"
                for method in METHODS
            },
        }

        metrics = {}
        paths = {}
        for method, path in prediction_paths.items():
            paths[method] = path.relative_to(root).as_posix()
            if not path.is_file():
                missing.append(str(path))
                continue
            pred = resize_prediction(load_mask(path), gt.shape)
            metrics[method] = image_miou(
                pred, gt, valid, NUM_CLASSES[dataset]
            )

        summarized.append(
            {
                **sample,
                "predictions": paths,
                "selection_only_image_miou": metrics,
            }
        )

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing qualitative predictions:\n  {formatted}")

    result = {
        **manifest,
        "methods": [*METHODS, "ours"],
        "metric_note": (
            "Per-image mIoU is for selecting visually informative figure rows "
            "only; datasets, protocols, and official aggregate metrics "
            "remain unchanged."
        ),
        "samples": summarized,
    }
    result_path = root / "results_manifest.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nCandidate summary (per-image mIoU; selection aid only):")
    for sample in summarized:
        values = sample["selection_only_image_miou"]
        formatted = " ".join(
            f"{method}={100.0 * values[method]:.1f}" for method in (*METHODS, "ours")
        )
        print(f"  {sample['name']}: {formatted}")
    print(f"Results manifest: {result_path}")


if __name__ == "__main__":
    main()
