#!/usr/bin/env python3
"""Build a fresh, prediction-blind qualitative shortlist.

The script ranks validation images using ground-truth annotation complexity
only.  It selects several candidates from each paper benchmark so that all
comparison methods can later be run on exactly the same images.  Final figure
rows are chosen only after every method has produced a prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    protocol: str
    processed_dir: str
    npy_name: str
    image_subdir: str
    mask_subdir: str
    mask_suffix: str
    split_file: str | None
    valid_max: int
    background_id: int | None


DATASETS = (
    DatasetSpec(
        key="voc21",
        protocol="pascal_voc",
        processed_dir="data/pascal_voc_processed",
        npy_name="pascal_voc_val.npy",
        image_subdir="JPEGImages",
        mask_subdir="SegmentationClass",
        mask_suffix=".png",
        split_file="ImageSets/Segmentation/val.txt",
        valid_max=20,
        background_id=0,
    ),
    DatasetSpec(
        key="context60",
        protocol="pascal_context",
        processed_dir="data/pascal_context_processed",
        npy_name="pascal_context_val.npy",
        image_subdir="JPEGImages",
        mask_subdir="SegmentationClassContext",
        mask_suffix=".png",
        split_file="ImageSets/SegmentationContext/val.txt",
        valid_max=59,
        background_id=0,
    ),
    DatasetSpec(
        key="coco_stuff",
        protocol="coco_stuff",
        processed_dir="data/coco_stuff_processed",
        npy_name="coco_stuff_val.npy",
        image_subdir="val2017",
        mask_subdir="annotations/val2017",
        mask_suffix="_labelTrainIds.png",
        split_file=None,
        valid_max=170,
        background_id=None,
    ),
    DatasetSpec(
        key="ade20k",
        protocol="ade20k",
        processed_dir="data/ade20k_processed",
        npy_name="ade20k_val.npy",
        image_subdir="images/validation",
        mask_subdir="annotations/validation",
        mask_suffix=".png",
        split_file=None,
        valid_max=150,
        background_id=0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="DINOde_ASFD_VADD repository root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/fresh_multidataset_qualitative"),
    )
    parser.add_argument("--candidates-per-dataset", type=int, default=6)
    parser.add_argument(
        "--exclude-first",
        type=int,
        default=25,
        help="Skip an initial validation prefix used by earlier quick exports.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("tools/qualitative_inputs"),
        help="Existing paper-image directory; byte-identical images are excluded.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_hashes(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {
        sha256(candidate)
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in suffixes
    }


def boundary_density(mask: np.ndarray, valid: np.ndarray) -> float:
    valid_h = valid[:, 1:] & valid[:, :-1]
    valid_v = valid[1:, :] & valid[:-1, :]
    changes = ((mask[:, 1:] != mask[:, :-1]) & valid_h).sum()
    changes += ((mask[1:, :] != mask[:-1, :]) & valid_v).sum()
    return float(changes / max(1, int(valid_h.sum() + valid_v.sum())))


def annotation_metrics(mask: np.ndarray, spec: DatasetSpec) -> dict[str, float | int]:
    valid = (mask != 255) & (mask <= spec.valid_max)
    if spec.background_id is not None:
        foreground = valid & (mask != spec.background_id)
    else:
        foreground = valid

    labels, counts = np.unique(mask[foreground], return_counts=True)
    valid_pixels = max(1, int(valid.sum()))
    fractions = counts.astype(np.float64) / valid_pixels
    small_classes = int(((fractions >= 0.001) & (fractions <= 0.06)).sum())
    medium_classes = int(((fractions > 0.06) & (fractions <= 0.25)).sum())
    boundary = boundary_density(mask, valid)
    foreground_fraction = float(foreground.sum() / valid_pixels)

    # Favor diverse scenes containing several non-dominant regions and thin
    # semantic boundaries.  Extremely tiny regions below 0.1% are ignored so
    # isolated annotation noise cannot dominate the shortlist.
    score = (
        2.5 * len(labels)
        + 4.0 * small_classes
        + 2.0 * medium_classes
        + 150.0 * boundary
        + 4.0 * min(foreground_fraction, 0.75)
    )
    return {
        "score": float(score),
        "class_count": int(len(labels)),
        "small_class_count": small_classes,
        "medium_class_count": medium_classes,
        "boundary_density": boundary,
        "foreground_fraction": foreground_fraction,
    }


def normalized_extension(path: Path, fallback: str) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png"} else fallback


def select_candidates(
    spec: DatasetSpec,
    repo_root: Path,
    count: int,
    exclude_first: int,
    excluded_hashes: set[str],
) -> list[dict]:
    processed_dir = repo_root / spec.processed_dir
    data_path = processed_dir / spec.npy_name
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    records = np.load(data_path, allow_pickle=True)

    ranked: list[dict] = []
    for index in range(min(exclude_first, len(records)), len(records)):
        item = records[index]
        image_path = Path(str(item["image_path"]))
        mask_path = Path(str(item["mask_path"]))
        if not image_path.is_file() or not mask_path.is_file():
            continue
        if excluded_hashes and sha256(image_path) in excluded_hashes:
            continue
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
        metrics = annotation_metrics(mask, spec)
        ranked.append(
            {
                "source_index": index,
                "image_id": str(item.get("image_id", image_path.stem)),
                "image_path": image_path,
                "mask_path": mask_path,
                **metrics,
            }
        )

    ranked.sort(key=lambda row: (-float(row["score"]), int(row["source_index"])))
    if len(ranked) < count:
        raise RuntimeError(f"Only {len(ranked)} usable samples found for {spec.key}")
    return ranked[:count]


def copy_dataset(
    spec: DatasetSpec,
    selected: Iterable[dict],
    output_root: Path,
) -> list[dict]:
    dataset_root = output_root / "datasets" / spec.key
    image_dir = dataset_root / spec.image_subdir
    mask_dir = dataset_root / spec.mask_subdir
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    split_ids = []
    for rank, row in enumerate(selected):
        image_id = str(row["image_id"])
        image_ext = normalized_extension(row["image_path"], ".jpg")
        image_name = image_id + image_ext
        # MMSegmentation's COCOStuffDataset hard-codes
        # ``_labelTrainIds.png`` while the other evaluation datasets use
        # plain ``.png`` masks.  Keep the copied mini dataset faithful to
        # each dataset adapter instead of inheriting the source filename.
        mask_name = image_id + spec.mask_suffix
        image_dst = image_dir / image_name
        mask_dst = mask_dir / mask_name
        shutil.copy2(row["image_path"], image_dst)
        shutil.copy2(row["mask_path"], mask_dst)
        split_ids.append(image_id)

        sample_name = f"{spec.key}_{rank:02d}"
        samples.append(
            {
                "name": sample_name,
                "dataset": spec.key,
                "protocol": spec.protocol,
                "source_index": row["source_index"],
                "image_id": image_id,
                "image": image_dst.relative_to(output_root).as_posix(),
                "mask": mask_dst.relative_to(output_root).as_posix(),
                "gla_image_stem": image_dst.stem,
                "score": row["score"],
                "class_count": row["class_count"],
                "small_class_count": row["small_class_count"],
                "medium_class_count": row["medium_class_count"],
                "boundary_density": row["boundary_density"],
                "foreground_fraction": row["foreground_fraction"],
            }
        )

    if spec.split_file is not None:
        split_path = dataset_root / spec.split_file
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("\n".join(split_ids) + "\n", encoding="utf-8")
    return samples


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    excluded = reference_hashes(repo_root / args.reference_dir)

    all_samples = []
    for spec in DATASETS:
        selected = select_candidates(
            spec,
            repo_root,
            args.candidates_per_dataset,
            args.exclude_first,
            excluded,
        )
        samples = copy_dataset(spec, selected, output_root)
        all_samples.extend(samples)
        print(f"[{spec.key}] selected {len(samples)} fresh candidates")
        for sample in samples:
            print(
                f"  {sample['name']} id={sample['image_id']} "
                f"classes={sample['class_count']} small={sample['small_class_count']} "
                f"score={sample['score']:.3f}"
            )

    manifest = {
        "selection_rule": (
            "Prediction-blind ranking by GT class diversity, non-dominant "
            "region count, foreground coverage, and semantic-boundary density."
        ),
        "candidates_per_dataset": args.candidates_per_dataset,
        "excluded_validation_prefix": args.exclude_first,
        "datasets": [spec.key for spec in DATASETS],
        "samples": all_samples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
