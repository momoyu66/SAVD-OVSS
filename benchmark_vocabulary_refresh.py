import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_flow_efficiency import load_student, load_teacher
from model.components import CLIPTextEncoder, build_prompts, l2norm


PROTOCOLS = {
    "cityscapes": {
        "count": 19,
        "path": "data/cityscapes_processed/cityscapes_class_names.json",
        "drop_background": False,
    },
    "context59": {
        "count": 59,
        "path": "data/pascal_context_processed/pascal_context_class_names.json",
        "drop_background": True,
    },
    "ade20k": {
        "count": 150,
        "path": "data/ade20k_processed/ade20k_class_names.json",
        "drop_background": True,
    },
    "coco_stuff": {
        "count": 171,
        "path": "data/coco_stuff_processed/coco_stuff_class_names.json",
        "drop_background": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark model-resident, uncached vocabulary refresh from raw "
            "class names through prompt construction, OpenCLIP, and alignment."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/dinode_eval_local.json",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoints/eccv26_dinode_coco_stuff.pth",
    )
    parser.add_argument(
        "--student-checkpoint",
        default=(
            "checkpoints/"
            "flow_student_asfd_vadd_h1536_seed123.pth"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/vocabulary_refresh_h1536.json",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
    )
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def load_names(protocol):
    spec = PROTOCOLS[protocol]
    path = Path(spec["path"])
    names = json.loads(path.read_text(encoding="utf-8"))

    if spec["drop_background"] and len(names) == spec["count"] + 1:
        first = str(names[0]).strip().lower()
        if first not in {"background", "__background__", "bg"}:
            raise RuntimeError(
                f"Refusing to drop unrecognized first class {names[0]!r} "
                f"from {path}"
            )
        names = names[1:]

    if len(names) != spec["count"]:
        raise RuntimeError(
            f"{protocol}: expected {spec['count']} classes, found "
            f"{len(names)} in {path}"
        )
    return names, path


@torch.inference_mode()
def encode_uncached_vocabulary(
    names,
    protocol,
    text_encoder,
    device,
    clip_batch_size,
):
    """Return CLIP class embeddings and measured common-prefix stages."""
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()

    prompt_groups = [build_prompts(name, protocol) for name in names]
    templates_per_class = len(prompt_groups[0])
    if any(len(group) != templates_per_class for group in prompt_groups):
        raise RuntimeError("Every class must use the same template count")
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    after_prompts = time.perf_counter_ns()

    # Tokenization is deliberately inside the timed vocabulary-refresh path.
    tokens = text_encoder.tokenizer(flat_prompts)
    after_tokenization = time.perf_counter_ns()

    chunks = []
    for start_index in range(0, len(flat_prompts), clip_batch_size):
        batch_tokens = tokens[
            start_index : start_index + clip_batch_size
        ].to(device, non_blocking=False)
        encoded = text_encoder.model.encode_text(batch_tokens)
        chunks.append(l2norm(encoded.float()))

    prompt_features = torch.cat(chunks, dim=0)
    clip_features = prompt_features.reshape(
        len(names), templates_per_class, -1
    ).mean(dim=1)

    torch.cuda.synchronize(device)
    after_clip = time.perf_counter_ns()

    timings = {
        "prompt_build_ms": (after_prompts - start) / 1e6,
        "tokenization_ms": (after_tokenization - after_prompts) / 1e6,
        "clip_encode_ms": (after_clip - after_tokenization) / 1e6,
        "common_prefix_ms": (after_clip - start) / 1e6,
        "prompt_count": len(flat_prompts),
        "templates_per_class": templates_per_class,
    }
    return clip_features, timings


@torch.inference_mode()
def time_alignment(function, clip_features, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    output = function(clip_features)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if not output.isfinite().all():
        raise RuntimeError("Alignment produced non-finite values")
    return elapsed_ms, output


@torch.inference_mode()
def verify_batched_clip_path(
    names,
    protocol,
    text_encoder,
    device,
    clip_batch_size,
):
    """Check the batched path against the released per-class encode path."""
    subset = names[:2]
    batched, _ = encode_uncached_vocabulary(
        subset,
        protocol,
        text_encoder,
        device,
        clip_batch_size,
    )
    reference = torch.stack(
        [
            text_encoder.encode(
                build_prompts(name, protocol), aggregate="mean"
            ).squeeze(0)
            for name in subset
        ],
        dim=0,
    ).float()
    max_abs = float((batched - reference).abs().max())
    min_cosine = float(
        F.cosine_similarity(batched, reference, dim=-1).min()
    )
    if min_cosine < 0.9999:
        raise RuntimeError(
            f"Batched CLIP path mismatch for {protocol}: "
            f"min cosine={min_cosine}"
        )
    return {
        "classes_checked": len(subset),
        "max_abs_error": max_abs,
        "min_cosine": min_cosine,
    }


@torch.inference_mode()
def one_trial(
    names,
    protocol,
    text_encoder,
    teacher,
    student,
    device,
    clip_batch_size,
    teacher_first,
):
    clip_features, common = encode_uncached_vocabulary(
        names,
        protocol,
        text_encoder,
        device,
        clip_batch_size,
    )

    functions = {
        "teacher_10_step": teacher.apply_text_flow,
        "student_single_step": student,
    }
    order = (
        ["teacher_10_step", "student_single_step"]
        if teacher_first
        else ["student_single_step", "teacher_10_step"]
    )
    alignment = {}
    outputs = {}
    for name in order:
        alignment[name], outputs[name] = time_alignment(
            functions[name], clip_features, device
        )

    if outputs["teacher_10_step"].shape != outputs["student_single_step"].shape:
        raise RuntimeError("Teacher/student output shapes do not match")

    return {
        **common,
        "teacher_alignment_ms": alignment["teacher_10_step"],
        "student_alignment_ms": alignment["student_single_step"],
        "teacher_total_ms": (
            common["common_prefix_ms"] + alignment["teacher_10_step"]
        ),
        "student_total_ms": (
            common["common_prefix_ms"] + alignment["student_single_step"]
        ),
    }


def summarize_trials(trials):
    timing_keys = [
        "prompt_build_ms",
        "tokenization_ms",
        "clip_encode_ms",
        "common_prefix_ms",
        "teacher_alignment_ms",
        "student_alignment_ms",
        "teacher_total_ms",
        "student_total_ms",
    ]
    summary = {
        key: summarize([trial[key] for trial in trials])
        for key in timing_keys
    }
    summary["alignment_speedup_ratio_of_means"] = (
        summary["teacher_alignment_ms"]["mean_ms"]
        / summary["student_alignment_ms"]["mean_ms"]
    )
    summary["inclusive_speedup_ratio_of_means"] = (
        summary["teacher_total_ms"]["mean_ms"]
        / summary["student_total_ms"]["mean_ms"]
    )
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    teacher, config = load_teacher(
        args.config, args.teacher_checkpoint, device
    )
    student, student_checkpoint = load_student(
        args.student_checkpoint, device
    )
    text_config = config["text_encoder"]
    text_encoder = CLIPTextEncoder(
        model_name=text_config["model_name"],
        pretrained=text_config["pretrained"],
        device=str(device),
    ).eval()

    metadata = {
        "definition": (
            "model-resident, uncached-vocabulary refresh from raw class names; "
            "includes prompt construction, CPU tokenization, OpenCLIP text "
            "encoding, and teacher/student text alignment; excludes model and "
            "checkpoint loading, DINOv3 image encoding, dense matching, and "
            "segmentation post-processing"
        ),
        "paired_common_prefix": (
            "Within each trial, both methods use the same measured prompt, "
            "tokenization, OpenCLIP prefix and the exact same CLIP embeddings. "
            "Inclusive totals are the measured common prefix plus the respective "
            "measured alignment stage."
        ),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "clip_batch_size": args.clip_batch_size,
        "warmup": args.warmup,
        "trials": args.trials,
        "seed": args.seed,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "student_geometry": student_checkpoint.get("student_geometry"),
        "text_encoder": text_config,
    }

    results = {}
    for protocol in args.protocols:
        names, class_path = load_names(protocol)
        equivalence = verify_batched_clip_path(
            names,
            protocol,
            text_encoder,
            device,
            args.clip_batch_size,
        )

        for warmup_index in range(args.warmup):
            one_trial(
                names,
                protocol,
                text_encoder,
                teacher,
                student,
                device,
                args.clip_batch_size,
                teacher_first=(warmup_index % 2 == 0),
            )

        trials = []
        for trial_index in range(args.trials):
            trial = one_trial(
                names,
                protocol,
                text_encoder,
                teacher,
                student,
                device,
                args.clip_batch_size,
                teacher_first=(trial_index % 2 == 0),
            )
            trials.append(trial)

        summary = summarize_trials(trials)
        results[protocol] = {
            "class_count": len(names),
            "class_names_path": str(class_path),
            "templates_per_class": trials[0]["templates_per_class"],
            "prompt_count": trials[0]["prompt_count"],
            "batched_vs_released_clip_path": equivalence,
            "summary": summary,
            "raw_trials": trials,
        }

        teacher_total = summary["teacher_total_ms"]
        student_total = summary["student_total_ms"]
        print("\n" + "=" * 88)
        print(
            f"{protocol}: K={len(names)}, prompts={trials[0]['prompt_count']}"
        )
        print(
            "teacher total: "
            f"{teacher_total['mean_ms']:.3f} mean / "
            f"{teacher_total['median_ms']:.3f} median / "
            f"{teacher_total['p95_ms']:.3f} p95 ms"
        )
        print(
            "student total: "
            f"{student_total['mean_ms']:.3f} mean / "
            f"{student_total['median_ms']:.3f} median / "
            f"{student_total['p95_ms']:.3f} p95 ms"
        )
        print(
            "inclusive speedup: "
            f"{summary['inclusive_speedup_ratio_of_means']:.3f}x"
        )
        print(
            "alignment-only speedup: "
            f"{summary['alignment_speedup_ratio_of_means']:.3f}x"
        )

    payload = {"metadata": metadata, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nsaved:", output_path)
    print("VOCABULARY REFRESH BENCHMARK: OK")


if __name__ == "__main__":
    main()
