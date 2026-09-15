import argparse
import json
import os
import platform
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TextCondHead
from model.flow_student import SingleStepSphericalStudent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only the DINOde/ASFD text-alignment modules. "
            "The DINOv3 image backbone and dense segmentation head are "
            "deliberately excluded."
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
        default="results/flow_efficiency_h1536.json",
    )
    parser.add_argument(
        "--class-counts",
        type=int,
        nargs="+",
        default=[19, 59, 150, 171],
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_teacher(config_path, checkpoint_path, device):
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    head_config = config["head"]
    flow_config = config.get("flow", {})

    head = TextCondHead(
        in_channels=head_config["in_channels"],
        tau=head_config["tau"],
        use_text_flow=head_config.get("use_text_flow", True),
        use_cls_flow=head_config.get("use_cls_flow", True),
        use_cls_mlp=head_config.get("use_cls_mlp", False),
        text_flow_steps=flow_config.get("steps", 10),
        text_flow_depth=flow_config.get("depth", 4),
        text_flow_dt=flow_config.get("dt"),
        topk=head_config.get("topk", 20),
        debug_interval=head_config.get("debug_interval", 100),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    head.load_state_dict(checkpoint["model_state_dict"])
    head = head.to(device).eval()

    for parameter in head.parameters():
        parameter.requires_grad_(False)

    return head, config


def load_student(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["student_state_dict"]
    student_geometry = checkpoint.get(
        "student_geometry",
        "tangent_residual",
    )

    student = SingleStepSphericalStudent(
        init_weight=state["text_flow_init.weight"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        geometry=student_geometry,
    )
    student.load_state_dict(state)
    student = student.to(device).eval()

    for parameter in student.parameters():
        parameter.requires_grad_(False)

    return student, checkpoint


def count_parameters(modules):
    parameters = []
    seen = set()

    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)

    return {
        "parameters": int(sum(p.numel() for p in parameters)),
        "bytes": int(sum(p.numel() * p.element_size() for p in parameters)),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


@torch.inference_mode()
def benchmark_cuda(function, inputs, warmup, trials, inner_repeats):
    for _ in range(warmup):
        output = function(inputs)

    torch.cuda.synchronize()
    values = []

    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(inner_repeats):
            output = function(inputs)
        end.record()

        torch.cuda.synchronize()
        values.append(
            float(start.elapsed_time(end)) / inner_repeats
        )

    assert output.isfinite().all()

    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "p95_ms": percentile(values, 0.95),
        "output_shape": list(output.shape),
    }


def format_row(name, nfe, result, speedup):
    return (
        f"{name:22s} "
        f"NFE={nfe:2d}  "
        f"mean={result['mean_ms']:9.4f} ms  "
        f"median={result['median_ms']:9.4f} ms  "
        f"p95={result['p95_ms']:9.4f} ms  "
        f"speedup={speedup:7.2f}x"
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # This changes only the step size used by the custom-step teacher.
    # It makes both the 5-step and 10-step solvers integrate from t=0 to t=1.
    os.environ["DINODE_RESCALED_STEPS"] = "1"

    teacher, config = load_teacher(
        args.config,
        args.teacher_checkpoint,
        device,
    )
    student, student_checkpoint = load_student(
        args.student_checkpoint,
        device,
    )

    teacher_text_parameters = count_parameters(
        [teacher.text_flow_init, teacher.text_flow_net]
    )
    student_parameters = count_parameters([student])
    student_residual_parameters = count_parameters(
        [student.down, student.up]
    )

    metadata = {
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "warmup": args.warmup,
        "trials": args.trials,
        "inner_repeats": args.inner_repeats,
        "seed": args.seed,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "student_hidden_dim": int(student_checkpoint["hidden_dim"]),
        "student_geometry": student_checkpoint.get(
            "student_geometry",
            "tangent_residual",
        ),
        "student_vadd_weighting": student_checkpoint.get(
            "vadd_weighting",
        ),
        "student_vadd_reduction": student_checkpoint.get(
            "vadd_reduction",
        ),
        "teacher_configured_steps": int(teacher.text_flow_steps),
        "teacher_configured_dt": float(teacher.text_flow_dt),
        "scope": (
            "text alignment only; DINOv3 image backbone, CLIP encoding, "
            "GCF/CLS flow, dense logits, and post-processing excluded"
        ),
    }

    parameter_counts = {
        "teacher_text_branch": teacher_text_parameters,
        "student_total_including_frozen_init": student_parameters,
        "student_trainable_residual": student_residual_parameters,
        "teacher_to_student_total_ratio": (
            teacher_text_parameters["parameters"]
            / student_parameters["parameters"]
        ),
        "teacher_to_student_residual_ratio": (
            teacher_text_parameters["parameters"]
            / student_residual_parameters["parameters"]
        ),
    }

    results = {}

    print("=" * 88)
    print("Text-alignment efficiency benchmark")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Parameter counts:")
    print(json.dumps(parameter_counts, indent=2))

    for class_count in args.class_counts:
        text_features = F.normalize(
            torch.randn(
                class_count,
                768,
                device=device,
                dtype=torch.float32,
            ),
            dim=-1,
        )

        methods = {
            "teacher_10_step": (
                10,
                lambda value: teacher.apply_text_flow_with_steps(
                    value, 10
                ),
            ),
            "teacher_5_step": (
                5,
                lambda value: teacher.apply_text_flow_with_steps(
                    value, 5
                ),
            ),
            "asfd_vadd_single_step": (
                1,
                student,
            ),
        }

        class_results = {}
        for name, (nfe, function) in methods.items():
            class_results[name] = benchmark_cuda(
                function,
                text_features,
                args.warmup,
                args.trials,
                args.inner_repeats,
            )
            class_results[name]["nfe"] = nfe

        student_time = class_results[
            "asfd_vadd_single_step"
        ]["mean_ms"]

        for result in class_results.values():
            result["speedup_vs_student"] = (
                result["mean_ms"] / student_time
            )

        results[str(class_count)] = class_results

        print("\n" + "-" * 88)
        print(f"Classes: {class_count}")
        for name in [
            "teacher_10_step",
            "teacher_5_step",
            "asfd_vadd_single_step",
        ]:
            result = class_results[name]
            print(
                format_row(
                    name,
                    result["nfe"],
                    result,
                    result["speedup_vs_student"],
                )
            )

    payload = {
        "metadata": metadata,
        "parameter_counts": parameter_counts,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("saved:", output_path)
    print("FLOW EFFICIENCY BENCHMARK: OK")


if __name__ == "__main__":
    main()
