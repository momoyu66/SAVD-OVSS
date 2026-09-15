"""Extract normalized DINOv3 patch anchors from unlabeled images."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.components import DinoV3HFBackbone  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample normalized DINOv3 patch features from unlabeled images."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/dinode_eval_local.json",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--output",
        default=(
            "data/flow_distill_vadd/"
            "coco_train_2k_64anchors_fp16.pth"
        ),
    )
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--anchors-per-image", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def find_images(root):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


@torch.inference_mode()
def main():
    args = parse_args()
    started = time.time()

    image_root = Path(args.image_root).expanduser().resolve()
    output_path = Path(args.output).expanduser()

    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)

    backbone_cfg = config["backbone"]
    image_paths = find_images(image_root)
    if len(image_paths) < args.num_images:
        raise ValueError(
            f"requested {args.num_images} images, found {len(image_paths)}"
        )

    rng = random.Random(args.seed)
    # Preserve the sampled order. This is the exact image-selection rule used
    # to build the released 2,000-image cache (seed 123).
    selected_paths = rng.sample(image_paths, args.num_images)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(args.seed)

    backbone = DinoV3HFBackbone(
        model_id=backbone_cfg["model_id"],
        device=args.device,
        image_size=backbone_cfg["image_size"],
    ).eval()

    anchor_batches = []
    path_batches = [
        selected_paths[start : start + args.batch_size]
        for start in range(0, len(selected_paths), args.batch_size)
    ]
    for batch_paths in tqdm(path_batches, desc="Extracting"):
        pixel_batches = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                pixel_batches.append(
                    backbone.preprocess(image.convert("RGB"))[
                        "pixel_values"
                    ]
                )

        patch_grid, _ = backbone.forward_grid(
            torch.cat(pixel_batches, dim=0)
        )
        batch_tokens = F.normalize(
            patch_grid.flatten(2).transpose(1, 2).float(),
            dim=-1,
        )

        for image_path, patch_tokens in zip(batch_paths, batch_tokens):
            if len(patch_tokens) < args.anchors_per_image:
                raise ValueError(
                    f"{image_path} has only {len(patch_tokens)} patches"
                )

            indices = torch.randperm(
                len(patch_tokens),
                generator=torch_generator,
            )[: args.anchors_per_image].to(patch_tokens.device)
            anchor_batches.append(
                patch_tokens[indices].cpu().half()
            )

    anchor_cache = torch.cat(anchor_batches, dim=0)
    anchor_count = args.num_images * args.anchors_per_image
    if anchor_cache.shape != (anchor_count, 1024):
        raise ValueError(
            "unexpected anchor shape: "
            f"{tuple(anchor_cache.shape)}"
        )
    if not anchor_cache.isfinite().all():
        raise ValueError("anchor cache contains non-finite values")

    metadata = {
        "seed": args.seed,
        "num_images": args.num_images,
        "anchors_per_image": args.anchors_per_image,
        "anchor_count": anchor_count,
        "feature_dim": 1024,
        "dtype": "float16",
        "image_root": str(image_root),
        "backbone_model_id": backbone_cfg["model_id"],
        "image_size": backbone_cfg["image_size"],
        "normalization": "per-patch L2 normalization",
        "selected_images": [path.name for path in selected_paths],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "anchors": anchor_cache,
            "metadata": metadata,
        },
        output_path,
    )

    elapsed = time.time() - started
    size_mib = (
        anchor_cache.numel()
        * anchor_cache.element_size()
        / 1024**2
    )
    stored_norms = anchor_cache[:8192].float().norm(dim=-1)

    print("saved:", output_path)
    print("shape:", tuple(anchor_cache.shape))
    print("dtype:", anchor_cache.dtype)
    print("finite:", bool(anchor_cache.isfinite().all()))
    print(
        "stored norm min/mean/max:",
        stored_norms.min().item(),
        stored_norms.mean().item(),
        stored_norms.max().item(),
    )
    print("elapsed seconds:", round(elapsed, 2))
    print("cache MiB:", round(size_mib, 2))
    print("VISUAL ANCHOR EXTRACTION: OK")


if __name__ == "__main__":
    main()
