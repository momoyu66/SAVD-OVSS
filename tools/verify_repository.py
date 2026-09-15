"""Run lightweight checks that do not require model weights or CUDA."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "NOTICE.md",
    "model/flow_student.py",
    "model/dinode.py",
    "train_flow_distill.py",
    "train_flow_distill_vadd.py",
    "eval_flow_distill.py",
    "eval_flow_distill_deploy.py",
    "benchmark_flow_efficiency.py",
    "benchmark_vocabulary_refresh.py",
    "configs/dinode_eval_local.json",
    "assets/taxonomy/coco_stuff_171_classes.json",
    "results/paper/asfd_only_seed123.json",
    "results/paper/savd_seed123.json",
    "results/paper/decision_metrics_independent.json",
    "results/paper/vocabulary_refresh.json",
    "results/paper/semantic_disjoint_summary.json",
]

PRIVATE_MARKERS = (
    "$HOME",
    "C:\\Users\\moyu",
    "D:\\Projects\\xuezhang",
    "zystu2025@",
)


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required release files: " + ", ".join(missing))

    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))

    taxonomy = load_json("assets/taxonomy/coco_stuff_171_classes.json")
    if len(taxonomy) != 171:
        raise ValueError(f"expected 171 taxonomy entries, found {len(taxonomy)}")

    savd = load_json("results/paper/savd_seed123.json")
    if savd.get("seed") != 123:
        raise ValueError("paper result must use the fixed seed-123 run")
    if not math.isclose(savd["means"]["vadd"], 49.557025, abs_tol=1e-9):
        raise ValueError("unexpected SAVD eight-protocol macro mIoU")

    decision = load_json("results/paper/decision_metrics_independent.json")
    metrics = decision["by_seed"]["123"]
    expected = {
        ("asfd_only", "decision_kl_t2"): 0.00496,
        ("asfd_vadd", "decision_kl_t2"): 0.00174,
        ("asfd_only", "ambiguous_top1_agreement"): 0.7106,
        ("asfd_vadd", "ambiguous_top1_agreement"): 0.8056,
    }
    for (method, key), value in expected.items():
        if not math.isclose(metrics[method][key], value, abs_tol=5e-5):
            raise ValueError(f"unexpected independent-anchor metric: {method}.{key}")

    text_suffixes = {".py", ".json", ".md", ".txt", ".sh", ".csv", ".yaml", ".yml"}
    leaked = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in PRIVATE_MARKERS):
            leaked.append(str(path.relative_to(ROOT)))
    if leaked:
        raise ValueError("private machine paths remain in: " + ", ".join(leaked))

    print("required release files:", len(REQUIRED))
    print("taxonomy classes:", len(taxonomy))
    print("paper seed:", savd["seed"])
    print("paper macro mIoU:", savd["means"]["vadd"])
    print("private-path audit: OK")
    print("SAVD RELEASE: OK")


if __name__ == "__main__":
    main()
