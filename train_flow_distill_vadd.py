import argparse
import gc
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from model.components import build_prompts
from model.flow_student import SingleStepSphericalStudent
from train_flow_distill import (
    build_memory_cache,
    build_teacher,
    evaluate,
    load_captions,
)
from utils.anchor_split import split_anchors


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
        "--initial-student",
        required=True,
    )
    parser.add_argument(
        "--captions",
        default=(
            "data/coco/annotations/"
            "captions_train2017.json"
        ),
    )
    parser.add_argument(
        "--anchors",
        required=True,
    )
    parser.add_argument(
        "--class-names",
        default=(
            "assets/taxonomy/"
            "coco_stuff_171_classes.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--max-captions",
        type=int,
        default=50000,
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
        "--anchor-batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--vadd-weight",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--vadd-weighting",
        choices=["entropy", "uniform"],
        default="uniform",
        help=(
            "Use entropy-confidence weighted KL or "
            "uniform mean KL for visual anchors."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--anchor-split-seed",
        type=int,
        default=20260906,
        help="Fixed image-group split seed for visual anchors.",
    )
    parser.add_argument(
        "--anchor-val-images",
        type=int,
        default=200,
        help="Number of source images held out from VADD updates.",
    )
    parser.add_argument(
        "--student-geometry",
        choices=(
            "auto",
            "ambient_residual",
            "tangent_residual",
        ),
        default="auto",
        help=(
            "Geometry inherited from the initial "
            "student. Explicit values must agree "
            "with checkpoint metadata."
        ),
    )

    return parser.parse_args()


@torch.no_grad()
def build_class_cache(
    class_names,
    text_encoder,
    teacher,
    device,
):
    clip_features = torch.stack(
        [
            text_encoder.encode(
                build_prompts(
                    class_name,
                    "coco_stuff",
                ),
                aggregate="mean",
            ).squeeze(0)
            for class_name in class_names
        ],
        dim=0,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    teacher_features = teacher.apply_text_flow(
        clip_features
    )

    return (
        clip_features.cpu().half(),
        teacher_features.cpu().half(),
    )


def visual_decision_loss(
    anchors,
    teacher_text,
    student_text,
    tau,
    distill_temperature,
    vadd_weighting="entropy",
):
    teacher_logits = (
        anchors @ teacher_text.T
    ) / tau
    student_logits = (
        anchors @ student_text.T
    ) / tau

    teacher_scaled = (
        teacher_logits
        / distill_temperature
    )
    student_scaled = (
        student_logits
        / distill_temperature
    )

    with torch.no_grad():
        teacher_probability = F.softmax(
            teacher_scaled,
            dim=-1,
        )
        teacher_log_probability = F.log_softmax(
            teacher_scaled,
            dim=-1,
        )

        entropy = -(
            teacher_probability
            * teacher_log_probability
        ).sum(dim=-1)

        confidence = (
            1.0
            - entropy
            / math.log(
                teacher_probability.shape[-1]
            )
        ).clamp(min=0.05)

    if vadd_weighting == "entropy":
        loss_weight = confidence
    elif vadd_weighting == "uniform":
        loss_weight = torch.ones_like(confidence)
    else:
        raise ValueError(
            "vadd_weighting must be "
            "'entropy' or 'uniform'"
        )

    student_log_probability = F.log_softmax(
        student_scaled,
        dim=-1,
    )

    per_anchor_kl = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)

    decision_loss = (
        per_anchor_kl * loss_weight
    ).sum() / loss_weight.sum().clamp_min(1e-6)

    decision_loss = (
        decision_loss
        * distill_temperature**2
    )

    with torch.no_grad():
        agreement = (
            teacher_logits.argmax(dim=-1)
            == student_logits.argmax(dim=-1)
        ).float().mean()

        mean_confidence = confidence.mean()

    return (
        decision_loss,
        agreement,
        mean_confidence,
    )


@torch.no_grad()
def evaluate_decisions(
    model,
    class_clip,
    class_targets,
    anchors,
    anchor_batch_size,
    tau,
    distill_temperature,
    device,
    max_anchors=8192,
    vadd_weighting="entropy",
):
    model.eval()

    class_clip_device = class_clip.to(
        device=device,
        dtype=torch.float32,
    )
    class_target_device = class_targets.to(
        device=device,
        dtype=torch.float32,
    )
    student_text = model(class_clip_device)

    class_cosine = F.cosine_similarity(
        student_text,
        class_target_device,
        dim=-1,
    )

    losses = []
    agreements = []
    confidences = []
    weights = []

    count = min(max_anchors, len(anchors))

    for start in range(
        0,
        count,
        anchor_batch_size,
    ):
        end = min(
            start + anchor_batch_size,
            count,
        )
        anchor_batch = anchors[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        loss, agreement, confidence = (
            visual_decision_loss(
                anchor_batch,
                class_target_device,
                student_text,
                tau,
                distill_temperature,
                vadd_weighting=vadd_weighting,
            )
        )

        batch_weight = end - start
        losses.append(loss.cpu())
        agreements.append(agreement.cpu())
        confidences.append(confidence.cpu())
        weights.append(batch_weight)

    total_weight = float(sum(weights))

    return {
        "class_mean": class_cosine.mean().item(),
        "class_min": class_cosine.min().item(),
        "decision_kl": sum(
            value.item() * weight
            for value, weight
            in zip(losses, weights)
        ) / total_weight,
        "decision_agreement": sum(
            value.item() * weight
            for value, weight
            in zip(agreements, weights)
        ) / total_weight,
        "teacher_confidence": sum(
            value.item() * weight
            for value, weight
            in zip(confidences, weights)
        ) / total_weight,
    }


def save_checkpoint(
    path,
    model,
    initial_checkpoint,
    args,
    teacher_config,
    epoch,
    metrics,
    anchor_split,
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
            "hidden_dim": initial_checkpoint[
                "hidden_dim"
            ],
            "input_dim": 768,
            "output_dim": 1024,
            "relation_weight": 0.0,
            "vadd_weight": args.vadd_weight,
            "vadd_weighting": args.vadd_weighting,
            "vadd_reduction": (
                "uniform_mean"
                if args.vadd_weighting == "uniform"
                else "entropy_normalized"
            ),
            "student_geometry": model.geometry,
            "class_weight": args.class_weight,
            "tau": args.tau,
            "distill_temperature": (
                args.distill_temperature
            ),
            "epoch": epoch,
            "metrics": metrics,
            "teacher_checkpoint": args.checkpoint,
            "teacher_config": teacher_config,
            "initial_student": args.initial_student,
            "anchor_cache": args.anchors,
            "anchor_split_seed": args.anchor_split_seed,
            "anchor_val_images": args.anchor_val_images,
            "anchor_split_sha256": anchor_split[
                "split_sha256"
            ],
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

    text_encoder, teacher, config = build_teacher(
        args,
        device,
    )

    clip_cache, target_cache = build_memory_cache(
        captions,
        text_encoder,
        teacher,
        args.encode_batch_size,
        device,
    )

    with open(
        args.class_names,
        encoding="utf-8",
    ) as file:
        class_names = json.load(file)

    assert len(class_names) == 171, (
        f"expected 171 COCO-Stuff classes, "
        f"found {len(class_names)}"
    )

    class_clip, class_targets = build_class_cache(
        class_names,
        text_encoder,
        teacher,
        device,
    )

    del text_encoder
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    anchor_data = torch.load(
        args.anchors,
        map_location="cpu",
        weights_only=False,
    )
    train_anchors, val_anchors, anchor_split = (
        split_anchors(
            anchor_data,
            val_images=args.anchor_val_images,
            seed=args.anchor_split_seed,
        )
    )
    del anchor_data

    split_path = Path(args.output).with_name(
        Path(args.output).stem + "_anchor_split.json"
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(anchor_split, indent=2),
        encoding="utf-8",
    )
    print(
        "visual-anchor split:",
        len(anchor_split["train_images"]),
        "train images,",
        len(anchor_split["val_images"]),
        "validation images; hash=",
        anchor_split["split_sha256"],
    )

    initial_checkpoint = torch.load(
        args.initial_student,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_geometry = initial_checkpoint.get(
        "student_geometry"
    )

    if args.student_geometry == "auto":
        if checkpoint_geometry is None:
            resolved_geometry = (
                "tangent_residual"
            )
            print(
                "WARNING: initial checkpoint has no "
                "student_geometry; using legacy "
                "tangent_residual compatibility."
            )
        else:
            resolved_geometry = checkpoint_geometry
    else:
        resolved_geometry = args.student_geometry

    valid_geometries = {
        "ambient_residual",
        "tangent_residual",
    }

    if resolved_geometry not in valid_geometries:
        raise ValueError(
            "invalid resolved student geometry: "
            f"{resolved_geometry!r}"
        )

    if (
        checkpoint_geometry is not None
        and checkpoint_geometry
        != resolved_geometry
    ):
        raise ValueError(
            "student geometry mismatch: checkpoint="
            f"{checkpoint_geometry!r}, requested="
            f"{resolved_geometry!r}"
        )

    model = SingleStepSphericalStudent(
        init_weight=initial_checkpoint[
            "student_state_dict"
        ]["text_flow_init.weight"],
        hidden_dim=initial_checkpoint[
            "hidden_dim"
        ],
        geometry=resolved_geometry,
    ).to(device)

    model.load_state_dict(
        initial_checkpoint["student_state_dict"]
    )

    train_count = len(captions) - args.val_count
    train_clip = clip_cache[:train_count]
    train_targets = target_cache[:train_count]
    val_clip = clip_cache[train_count:]
    val_targets = target_cache[train_count:]

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    caption_initial = evaluate(
        model,
        val_clip,
        val_targets,
        args.batch_size,
        device,
    )
    decision_initial = evaluate_decisions(
        model,
        class_clip,
        class_targets,
        val_anchors,
        args.anchor_batch_size,
        args.tau,
        args.distill_temperature,
        device,
        max_anchors=len(val_anchors),
        vadd_weighting=args.vadd_weighting,
    )

    print("trainable parameters:", model.trainable_parameter_count())
    print("VADD settings:", vars(args))
    print(
        "resolved student geometry:",
        model.geometry,
    )
    print("initial caption:", caption_initial)
    print("initial decision:", decision_initial)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_count,
            generator=generator,
        )

        total = 0.0
        total_caption = 0.0
        total_class = 0.0
        total_vadd = 0.0
        total_agreement = 0.0
        steps = 0

        class_clip_device = class_clip.to(
            device=device,
            dtype=torch.float32,
        )
        class_target_device = class_targets.to(
            device=device,
            dtype=torch.float32,
        )

        for start in range(
            0,
            train_count,
            args.batch_size,
        ):
            if (
                args.max_train_batches > 0
                and steps >= args.max_train_batches
            ):
                break

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

            anchor_indices = torch.randint(
                len(train_anchors),
                (args.anchor_batch_size,),
                generator=generator,
            )
            anchor_batch = train_anchors[
                anchor_indices
            ].to(
                device=device,
                dtype=torch.float32,
            )

            caption_output = model(text)
            student_class = model(
                class_clip_device
            )

            caption_loss = (
                1.0
                - F.cosine_similarity(
                    caption_output,
                    target,
                    dim=-1,
                ).mean()
            )
            class_loss = (
                1.0
                - F.cosine_similarity(
                    student_class,
                    class_target_device,
                    dim=-1,
                ).mean()
            )
            vadd_loss, agreement, _ = (
                visual_decision_loss(
                    anchor_batch,
                    class_target_device,
                    student_class,
                    args.tau,
                    args.distill_temperature,
                    vadd_weighting=args.vadd_weighting,
                )
            )

            loss = (
                caption_loss
                + args.class_weight * class_loss
                + args.vadd_weight * vadd_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            total += loss.item()
            total_caption += caption_loss.item()
            total_class += class_loss.item()
            total_vadd += vadd_loss.item()
            total_agreement += agreement.item()
            steps += 1

        caption_metrics = evaluate(
            model,
            val_clip,
            val_targets,
            args.batch_size,
            device,
        )
        decision_metrics = evaluate_decisions(
            model,
            class_clip,
            class_targets,
            val_anchors,
            args.anchor_batch_size,
            args.tau,
            args.distill_temperature,
            device,
            max_anchors=len(val_anchors),
            vadd_weighting=args.vadd_weighting,
        )

        score = (
            caption_metrics["mean"]
            + 0.5 * decision_metrics["class_mean"]
            + 0.1 * decision_metrics[
                "decision_agreement"
            ]
            - 0.01 * decision_metrics[
                "decision_kl"
            ]
        )

        metrics = {
            "caption": caption_metrics,
            "decision": decision_metrics,
            "selection_score": score,
        }

        print(
            f"epoch {epoch:02d}: "
            f"total={total / steps:.6f} "
            f"caption={total_caption / steps:.6f} "
            f"class={total_class / steps:.6f} "
            f"vadd={total_vadd / steps:.6f} "
            f"train_agree={total_agreement / steps:.6f} "
            f"val_caption={caption_metrics['mean']:.6f} "
            f"val_class={decision_metrics['class_mean']:.6f} "
            f"val_kl={decision_metrics['decision_kl']:.6f} "
            f"val_agree={decision_metrics['decision_agreement']:.6f} "
            f"score={score:.6f}",
            flush=True,
        )

        last_path = Path(args.output).with_name(
            Path(args.output).stem + "_last.pth"
        )
        save_checkpoint(
            last_path,
            model,
            initial_checkpoint,
            args,
            config,
            epoch,
            metrics,
            anchor_split,
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                args.output,
                model,
                initial_checkpoint,
                args,
                config,
                epoch,
                metrics,
                anchor_split,
            )
            print("saved best:", args.output, flush=True)

    print("best selection score:", best_score)
    print("FLOW DISTILL VADD TRAINING: OK")


if __name__ == "__main__":
    main()
