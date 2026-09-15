#!/usr/bin/env python3
"""Select fresh, visually informative Cityscapes examples from ground truth.

Selection uses annotation complexity only: class diversity, the number and
area of traffic-object classes, and semantic-boundary density.  It never looks
at ASFD+VADD or baseline predictions.  The script excludes the first ten
validation items used by the earlier introduction experiment and balances the
shortlist across Cityscapes cities.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


THING_IDS = np.array([11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/cityscapes_processed")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/new_cityscapes_qualitative/dataset"),
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--exclude-first", type=int, default=10)
    parser.add_argument("--max-per-city", type=int, default=4)
    return parser.parse_args()


def boundary_density(mask: np.ndarray) -> float:
    valid_h = (mask[:, 1:] != 255) & (mask[:, :-1] != 255)
    valid_v = (mask[1:, :] != 255) & (mask[:-1, :] != 255)
    change_h = (mask[:, 1:] != mask[:, :-1]) & valid_h
    change_v = (mask[1:, :] != mask[:-1, :]) & valid_v
    denom = max(1, int(valid_h.sum() + valid_v.sum()))
    return float((change_h.sum() + change_v.sum()) / denom)


def sample_score(mask: np.ndarray) -> dict:
    valid = mask != 255
    valid_labels = np.unique(mask[valid])
    thing_mask = np.isin(mask, THING_IDS)
    thing_labels = np.unique(mask[thing_mask])
    thing_fraction = float(thing_mask.sum() / max(1, valid.sum()))
    boundaries = boundary_density(mask)
    score = (
        2.0 * len(valid_labels)
        + 5.0 * len(thing_labels)
        + 120.0 * boundaries
        + 25.0 * min(thing_fraction, 0.20)
    )
    return {
        "score": float(score),
        "class_count": int(len(valid_labels)),
        "thing_class_count": int(len(thing_labels)),
        "thing_fraction": thing_fraction,
        "boundary_density": boundaries,
    }


def main() -> None:
    args = parse_args()
    data_file = args.processed_dir / "cityscapes_val.npy"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    data = np.load(data_file, allow_pickle=True)

    ranked = []
    for index in range(args.exclude_first, len(data)):
        item = data[index]
        image_path = Path(str(item["image_path"]))
        mask_path = Path(str(item["mask_path"]))
        if not image_path.is_file() or not mask_path.is_file():
            continue
        with Image.open(mask_path) as image:
            mask = np.asarray(image, dtype=np.uint8)
        metrics = sample_score(mask)
        city = image_path.parent.name
        ranked.append(
            {
                "index": index,
                "city": city,
                "image_id": str(item.get("image_id", image_path.stem)),
                "image_path": image_path,
                "mask_path": mask_path,
                **metrics,
            }
        )

    ranked.sort(key=lambda row: (-row["score"], row["index"]))
    selected = []
    city_counts = {}
    for row in ranked:
        if city_counts.get(row["city"], 0) >= args.max_per_city:
            continue
        selected.append(row)
        city_counts[row["city"]] = city_counts.get(row["city"], 0) + 1
        if len(selected) == args.count:
            break
    if len(selected) < args.count:
        selected_indices = {row["index"] for row in selected}
        for row in ranked:
            if row["index"] in selected_indices:
                continue
            selected.append(row)
            if len(selected) == args.count:
                break

    samples = []
    for rank, row in enumerate(selected):
        city = row["city"]
        image_name = row["image_path"].name
        mask_name = row["mask_path"].name
        image_rel = Path("leftImg8bit") / "val" / city / image_name
        mask_rel = Path("gtFine") / "val" / city / mask_name
        image_dst = args.output_root / image_rel
        mask_dst = args.output_root / mask_rel
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image_path"], image_dst)
        shutil.copy2(row["mask_path"], mask_dst)

        name = f"sample_{rank:02d}"
        samples.append(
            {
                "name": name,
                "protocol": "cityscapes",
                "source_index": row["index"],
                "image_id": row["image_id"],
                "city": city,
                "image": image_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "score": row["score"],
                "class_count": row["class_count"],
                "thing_class_count": row["thing_class_count"],
                "thing_fraction": row["thing_fraction"],
                "boundary_density": row["boundary_density"],
            }
        )

    manifest = {
        "selection_rule": (
            "Ground-truth class diversity + traffic-object diversity/area + "
            "semantic-boundary density; predictions were not inspected."
        ),
        "excluded_validation_prefix": args.exclude_first,
        "samples": samples,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Selected {len(samples)} fresh Cityscapes examples:")
    for sample in samples:
        print(
            f"  {sample['name']} idx={sample['source_index']} "
            f"city={sample['city']} classes={sample['class_count']} "
            f"things={sample['thing_class_count']} score={sample['score']:.3f}"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
