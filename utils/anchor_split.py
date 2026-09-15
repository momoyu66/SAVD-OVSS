"""Deterministic image-group holdout for visual-anchor caches."""

import hashlib
import json
import random

import torch


def split_anchors(data, val_images=200, seed=20260906):
    """Split consecutive per-image anchor groups without image leakage."""
    anchors = data["anchors"]
    metadata = data["metadata"]
    names = metadata["selected_images"]
    image_count = int(metadata["num_images"])
    anchors_per_image = int(metadata["anchors_per_image"])

    if len(names) != image_count or len(set(names)) != image_count:
        raise ValueError("missing or duplicate image identities")
    if anchors.ndim != 2 or anchors.shape != (
        image_count * anchors_per_image,
        1024,
    ):
        raise ValueError("cache does not match consecutive image-group layout")
    if not 0 < val_images < image_count:
        raise ValueError("val_images must lie between 0 and num_images")
    if not torch.isfinite(anchors).all():
        raise ValueError("visual-anchor cache contains non-finite values")

    order = list(range(image_count))
    random.Random(seed).shuffle(order)
    val_ids = sorted(order[:val_images])
    train_ids = sorted(order[val_images:])
    grouped = anchors.reshape(image_count, anchors_per_image, 1024)
    train = grouped[train_ids].reshape(-1, 1024).contiguous()
    validation = grouped[val_ids].reshape(-1, 1024).contiguous()
    manifest = {
        "split_seed": seed,
        "anchors_per_image": anchors_per_image,
        "layout_assumption": (
            "consecutive anchors_per_image rows per selected_images entry"
        ),
        "train_image_indices": train_ids,
        "val_image_indices": val_ids,
        "train_images": [names[index] for index in train_ids],
        "val_images": [names[index] for index in val_ids],
        "train_anchor_count": len(train),
        "val_anchor_count": len(validation),
        "scope": (
            "visual anchors held out from stage-2 gradient updates; "
            "not an end-to-end unseen-image claim"
        ),
    }
    manifest["split_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return train, validation, manifest
