#!/usr/bin/env python3
"""Build paper-ready qualitative comparisons from raw train-ID masks.

The script uses one dataset-specific palette for every method, overlays the
same class with the same color, and never draws contours or bounding boxes.
It can render all six candidates for visual auditing and the final four-row
comparison selected after that audit.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = (
    ("clearclip", "ClearCLIP"),
    ("naclip", "NACLIP"),
    ("proxyclip", "ProxyCLIP"),
    ("gla_clip", "GLA-CLIP"),
    ("ours", "Ours (ASFD+VADD)"),
)

DATASET_TITLES = {
    "voc21": "PASCAL VOC",
    "context60": "PASCAL Context",
    "coco_stuff": "COCO-Stuff",
    "ade20k": "ADE20K",
}

PALETTE_METHODS = {
    "voc21": ("_get_voc_palette", "voc_palette"),
    "context60": ("_get_voc_context_palette", "CONTEXT_PALETTE"),
    "coco_stuff": ("_get_coco_stuff_palette", "STUFF_PALETTE"),
    "ade20k": ("_get_ade20k_palette", "ade20k_palette"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--palette-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--select",
        nargs="*",
        default=("voc21_05", "context60_04", "coco_stuff_02", "ade20k_02"),
    )
    parser.add_argument("--alpha", type=float, default=0.68)
    return parser.parse_args()


def literal_palette(source: Path, method_name: str, variable_name: str) -> np.ndarray:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != method_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == variable_name for t in statement.targets):
                continue
            value = statement.value
            if isinstance(value, ast.Call):
                value = value.args[0]
            palette = np.asarray(ast.literal_eval(value), dtype=np.uint8)
            if palette.ndim != 2 or palette.shape[1] != 3:
                raise ValueError(f"Invalid palette extracted for {method_name}")
            return palette
    raise ValueError(f"Could not extract {variable_name} from {method_name}")


def load_palettes(source: Path) -> dict[str, np.ndarray]:
    return {
        dataset: literal_palette(source, *definition)
        for dataset, definition in PALETTE_METHODS.items()
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    safe = mask.astype(np.int64).copy()
    invalid = (safe == 255) | (safe < 0) | (safe >= len(palette))
    safe[invalid] = 0
    colored = palette[safe]
    colored[invalid] = 255
    return colored.astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, palette: np.ndarray, dataset: str, alpha: float) -> np.ndarray:
    if mask.shape != image.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
        )
    colored = colorize(mask, palette)
    valid = mask != 255
    if dataset in {"voc21", "context60", "ade20k"}:
        valid &= mask != 0
    result = image.astype(np.float32).copy()
    result[valid] = (1.0 - alpha) * result[valid] + alpha * colored[valid]
    return np.clip(result, 0, 255).astype(np.uint8)


def panel(sample: dict, key: str, root: Path, palettes: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    image = load_rgb(root / sample["image"])
    if key == "image":
        return image
    path = sample["mask"] if key == "gt" else sample["predictions"][key]
    mask = load_mask(root / path)
    return overlay(image, mask, palettes[sample["dataset"]], sample["dataset"], alpha)


def render_grid(
    samples: list[dict],
    columns: tuple[tuple[str, str], ...],
    root: Path,
    palettes: dict[str, np.ndarray],
    output: Path,
    alpha: float,
    show_scores: bool,
    dpi: int,
) -> None:
    rows, cols = len(samples), len(columns)
    fig = plt.figure(figsize=(2.2 * cols + 0.85, 1.72 * rows + 0.55), facecolor="white")
    grid = fig.add_gridspec(
        rows,
        cols,
        left=0.068,
        right=0.997,
        bottom=0.018,
        top=0.93,
        wspace=0.018,
        hspace=0.025,
    )
    for row, sample in enumerate(samples):
        for col, (key, title) in enumerate(columns):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(panel(sample, key, root, palettes, alpha), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#d0d5dc")
            if row == 0:
                color = "#0b5cad" if key == "ours" else "#111827"
                weight = "bold" if key == "ours" else "semibold"
                ax.set_title(title, fontsize=12.2, fontweight=weight, color=color, pad=7)
            if col == 0:
                row_label = sample["name"] if show_scores else DATASET_TITLES[sample["dataset"]]
                ax.text(
                    -0.055,
                    0.5,
                    row_label,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=9.5 if show_scores else 11.0,
                    fontweight="semibold",
                    color="#374151",
                )
            if show_scores and key not in {"image", "gt"}:
                score = 100.0 * sample["selection_only_image_miou"][key]
                ax.text(
                    0.985,
                    0.025,
                    f"{score:.1f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.62, linewidth=0),
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "results_manifest.json").read_text(encoding="utf-8"))
    samples = data["samples"]
    palettes = load_palettes(args.palette_source.resolve())

    if args.audit:
        audit_columns = (("image", "Input Image"), ("gt", "Ground Truth"), *METHODS)
        for dataset in DATASET_TITLES:
            subset = [sample for sample in samples if sample["dataset"] == dataset]
            render_grid(
                subset,
                audit_columns,
                root,
                palettes,
                output_dir / f"audit_{dataset}.png",
                args.alpha,
                show_scores=True,
                dpi=220,
            )

    lookup = {sample["name"]: sample for sample in samples}
    selected = [lookup[name] for name in args.select]
    if {sample["dataset"] for sample in selected} != set(DATASET_TITLES):
        raise ValueError("Final selection must contain one sample from every dataset")
    selected.sort(key=lambda sample: list(DATASET_TITLES).index(sample["dataset"]))

    main_columns = (("image", "Input Image"), *METHODS)
    for suffix, dpi in (("png", 600), ("pdf", 300)):
        render_grid(
            selected,
            main_columns,
            root,
            palettes,
            output_dir / f"qualitative_comparison.{suffix}",
            args.alpha,
            show_scores=False,
            dpi=dpi,
        )

    gt_columns = (("image", "Input Image"), ("gt", "Ground Truth"), *METHODS)
    for suffix, dpi in (("png", 600), ("pdf", 300)):
        render_grid(
            selected,
            gt_columns,
            root,
            palettes,
            output_dir / f"qualitative_comparison_with_gt.{suffix}",
            args.alpha,
            show_scores=False,
            dpi=dpi,
        )

    selection = {
        sample["dataset"]: {
            "sample": sample["name"],
            "image_id": sample["image_id"],
            "image_miou": sample["selection_only_image_miou"],
        }
        for sample in selected
    }
    (output_dir / "selected_samples.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved qualitative figures to {output_dir}")


if __name__ == "__main__":
    main()
