import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.components import CLIPTextEncoder
from model.dinode import TextCondHead
from model.flow_student import (
    SingleStepSphericalStudent,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/dinode_eval_local.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "eccv26_dinode_coco_stuff.pth"
        ),
    )
    parser.add_argument(
        "--captions",
        default=(
            "data/coco/annotations/"
            "captions_train2017.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--max-captions",
        type=int,
        default=200000,
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1536,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--relation-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--student-geometry",
        type=str,
        choices=(
            "ambient_residual",
            "tangent_residual",
        ),
        default="ambient_residual",
        help=(
            "Residual geometry for the single-step "
            "student."
        ),
    )

    return parser.parse_args()


def build_teacher(args, device):
    with open(
        args.config,
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    head_cfg = config["head"]
    flow_cfg = config.get("flow", {})
    text_cfg = config["text_encoder"]

    text_encoder = CLIPTextEncoder(
        model_name=text_cfg["model_name"],
        pretrained=text_cfg["pretrained"],
        device=str(device),
    )

    teacher = TextCondHead(
        in_channels=head_cfg["in_channels"],
        tau=head_cfg["tau"],
        use_text_flow=True,
        use_cls_flow=True,
        use_cls_mlp=head_cfg.get(
            "use_cls_mlp",
            False,
        ),
        text_flow_steps=flow_cfg.get(
            "steps",
            10,
        ),
        text_flow_depth=flow_cfg.get(
            "depth",
            4,
        ),
        text_flow_dt=flow_cfg.get("dt"),
        topk=head_cfg.get("topk", 20),
        debug_interval=1000000,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"],
    )
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return (
        text_encoder,
        teacher,
        config,
    )


def load_captions(args):
    with open(
        args.captions,
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    captions = [
        item["caption"].strip()
        for item in data["annotations"]
        if item.get("caption", "").strip()
    ]

    rng = random.Random(args.seed)
    rng.shuffle(captions)

    count = min(
        args.max_captions,
        len(captions),
    )
    captions = captions[:count]

    if args.val_count >= len(captions):
        raise ValueError(
            "val-count must be smaller than max-captions"
        )

    print("selected captions:", len(captions))
    print("train captions:", len(captions) - args.val_count)
    print("validation captions:", args.val_count)

    return captions


@torch.no_grad()
def build_memory_cache(
    captions,
    text_encoder,
    teacher,
    batch_size,
    device,
):
    count = len(captions)

    clip_cache = torch.empty(
        count,
        768,
        dtype=torch.float16,
    )
    target_cache = torch.empty(
        count,
        1024,
        dtype=torch.float16,
    )

    started = time.time()

    for start in range(
        0,
        count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            count,
        )

        batch_captions = captions[start:end]

        clip_features = text_encoder.encode(
            batch_captions,
            aggregate="none",
        ).to(
            device=device,
            dtype=torch.float32,
        )

        targets = teacher.apply_text_flow(
            clip_features
        )

        clip_cache[start:end].copy_(
            clip_features.cpu().half()
        )
        target_cache[start:end].copy_(
            targets.cpu().half()
        )

        if (
            start == 0
            or end == count
            or end % (batch_size * 20) == 0
        ):
            elapsed = time.time() - started
            print(
                "encoded:",
                f"{end}/{count}",
                "elapsed:",
                f"{elapsed:.1f}s",
                flush=True,
            )

    print(
        "cache CLIP MiB:",
        round(
            clip_cache.numel()
            * clip_cache.element_size()
            / 1024**2,
            2,
        ),
    )
    print(
        "cache target MiB:",
        round(
            target_cache.numel()
            * target_cache.element_size()
            / 1024**2,
            2,
        ),
    )

    return clip_cache, target_cache


def relation_loss(student, teacher):
    if student.shape[0] < 2:
        return student.new_zeros(())

    student_gram = student @ student.T
    teacher_gram = teacher @ teacher.T

    mask = ~torch.eye(
        student.shape[0],
        dtype=torch.bool,
        device=student.device,
    )

    return F.mse_loss(
        student_gram[mask],
        teacher_gram[mask],
    )


@torch.no_grad()
def evaluate(
    model,
    clip_features,
    targets,
    batch_size,
    device,
):
    model.eval()
    cosine_values = []

    for start in range(
        0,
        len(clip_features),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(clip_features),
        )

        text = clip_features[start:end].to(
            device=device,
            dtype=torch.float32,
        )
        target = targets[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        output = model(text)

        cosine = F.cosine_similarity(
            output,
            target,
            dim=-1,
        )

        cosine_values.append(
            cosine.cpu()
        )

    cosine_values = torch.cat(
        cosine_values
    )

    return {
        "mean": cosine_values.mean().item(),
        "min": cosine_values.min().item(),
        "p05": torch.quantile(
            cosine_values,
            0.05,
        ).item(),
        "p50": torch.quantile(
            cosine_values,
            0.50,
        ).item(),
    }


def save_checkpoint(
    path,
    model,
    args,
    config,
    epoch,
    metrics,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    torch.save(
        {
            "student_state_dict": state,
            "hidden_dim": args.hidden_dim,
            "input_dim": 768,
            "output_dim": 1024,
            "relation_weight": (
                args.relation_weight
            ),
            "student_geometry": (
                args.student_geometry
            ),
            "epoch": epoch,
            "metrics": metrics,
            "teacher_checkpoint": (
                args.checkpoint
            ),
            "teacher_config": config,
        },
        path,
    )


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    captions = load_captions(args)

    (
        text_encoder,
        teacher,
        config,
    ) = build_teacher(args, device)

    init_weight = (
        teacher.text_flow_init.weight
        .detach()
        .cpu()
        .clone()
    )

    clip_cache, target_cache = (
        build_memory_cache(
            captions,
            text_encoder,
            teacher,
            args.encode_batch_size,
            device,
        )
    )

    del text_encoder
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    train_count = (
        len(captions) - args.val_count
    )

    train_clip = clip_cache[:train_count]
    train_targets = target_cache[:train_count]
    val_clip = clip_cache[train_count:]
    val_targets = target_cache[train_count:]

    model = SingleStepSphericalStudent(
        init_weight=init_weight,
        hidden_dim=args.hidden_dim,
        geometry=args.student_geometry,
    ).to(device)

    print(
        "trainable parameters:",
        model.trainable_parameter_count(),
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    initial = evaluate(
        model,
        val_clip,
        val_targets,
        args.batch_size,
        device,
    )
    print("initial validation:", initial)

    best_mean = -1.0

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()

        permutation = torch.randperm(
            train_count,
            generator=generator,
        )

        total_loss = 0.0
        total_endpoint = 0.0
        total_relation = 0.0
        steps = 0

        for start in range(
            0,
            train_count,
            args.batch_size,
        ):
            indices = permutation[
                start:start + args.batch_size
            ]

            text = train_clip[indices].to(
                device=device,
                dtype=torch.float32,
            )
            target = train_targets[indices].to(
                device=device,
                dtype=torch.float32,
            )

            output = model(text)

            endpoint = (
                1.0
                - F.cosine_similarity(
                    output,
                    target,
                    dim=-1,
                ).mean()
            )

            relation = relation_loss(
                output,
                target,
            )

            loss = (
                endpoint
                + args.relation_weight
                * relation
            )

            optimizer.zero_grad(
                set_to_none=True,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            total_endpoint += endpoint.item()
            total_relation += relation.item()
            steps += 1

        metrics = evaluate(
            model,
            val_clip,
            val_targets,
            args.batch_size,
            device,
        )

        print(
            f"epoch {epoch:02d}: "
            f"loss={total_loss / steps:.6f} "
            f"endpoint={total_endpoint / steps:.6f} "
            f"relation={total_relation / steps:.6f} "
            f"val_mean={metrics['mean']:.6f} "
            f"val_p05={metrics['p05']:.6f} "
            f"val_min={metrics['min']:.6f}",
            flush=True,
        )

        if metrics["mean"] > best_mean:
            best_mean = metrics["mean"]
            save_checkpoint(
                args.output,
                model,
                args,
                config,
                epoch,
                metrics,
            )
            print(
                "saved best:",
                args.output,
                flush=True,
            )

    print("best validation mean:", best_mean)
    print("FLOW DISTILL TRAINING: OK")


if __name__ == "__main__":
    main()
