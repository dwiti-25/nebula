"""Generate compact, provenance-labelled figures for the Nebula report.

This script performs no simulation. It reads preserved result artifacts and
produces PDF/PNG figures suitable for the LaTeX report.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "figures"
TRAINING = ROOT / "results" / "v3_tt_1000_training.jsonl"
BENCHMARK = ROOT / "results" / "benchmark_report.json"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#334155",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rolling_median(values: np.ndarray, width: int = 25) -> np.ndarray:
    return np.array([np.median(values[max(0, i - width + 1): i + 1]) for i in range(len(values))])


def training_overview(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: int(row["total_evaluation_count"]))
    evaluations = np.array([int(row["total_evaluation_count"]) for row in rows])
    rewards = np.array([float(row["reward"]) for row in rows])
    successes = np.array([bool(row.get("spec_satisfied")) for row in rows])
    stages = Counter("no stage failure" if row.get("success") else (row.get("failure_stage") or "unknown") for row in rows)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    axes[0].scatter(evaluations, rewards, s=8, alpha=0.22, color="#2563EB", label="Step reward")
    axes[0].plot(evaluations, rolling_median(rewards), color="#0F172A", linewidth=1.8,
                 label="Rolling median (25 steps)")
    axes[0].set_xlabel("Cumulative real-SPICE evaluations")
    axes[0].set_ylabel("Reward v3")
    axes[0].set_title("Reward history")
    axes[0].legend(frameon=False, fontsize=8)

    cumulative = np.cumsum(successes)
    axes[1].step(evaluations, cumulative, where="post", color="#15803D", linewidth=1.8)
    axes[1].set_xlabel("Cumulative real-SPICE evaluations")
    axes[1].set_ylabel("Cumulative satisfying steps")
    axes[1].set_title("Observed target satisfaction")

    order = [name for name in ("dc", "ac", "ctle_transient", "channel", "transient", "noise", "hd3", "no stage failure", "unknown") if stages[name]]
    vals = [stages[name] for name in order]
    colors = ["#15803D" if name == "no stage failure" else "#64748B" for name in order]
    axes[2].barh(order, vals, color=colors)
    for index, value in enumerate(vals):
        axes[2].text(value, index, f" {value}", va="center", fontsize=8)
    axes[2].set_xlabel("Logged policy steps")
    axes[2].set_title("Outcome by first failed stage")
    axes[2].grid(axis="x")
    axes[2].grid(axis="y", visible=False)

    fig.suptitle("PPO v3 real-backend training evidence", fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, -0.015,
             "Seed 101; 100 updates; 692 total evaluations. One recorded run, not a convergence or superiority estimate.",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout()
    save(fig, "ppo-v3-training-overview")


def parameter_trajectory(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: int(row["total_evaluation_count"]))
    evaluations = np.array([int(row["total_evaluation_count"]) for row in rows])
    names = (
        ("rload_ohm", r"$R_{LOAD}$", "ohm"),
        ("rdeg_ohm", r"$R_{DEG}$", "ohm"),
        ("cdeg_f", r"$C_{DEG}$", "pF"),
        ("itail_a", r"$I_{TAIL}$", "mA"),
        ("dfe_tap_v", "DFE tap", "V"),
        ("mos_width_um", "MOS width", "um"),
        ("mos_length_um", "MOS length", "um"),
        ("mos_multiplier", "MOS multiplier", "integer"),
    )
    scales = {"cdeg_f": 1e12, "itail_a": 1e3}
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 10), sharex=True)
    for axis, (key, title, unit) in zip(axes.flat, names):
        values = np.array([float(row["parameters"][key]) for row in rows]) * scales.get(key, 1.0)
        axis.plot(evaluations, values, color="#2563EB", linewidth=0.8, alpha=0.75)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(unit)
    for axis in axes[-1]:
        axis.set_xlabel("Cumulative real-SPICE evaluations")
    fig.suptitle("PPO v3 eight-parameter trajectory", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01,
             "Recorded policy steps from the seed-101 run; trajectories show exploration, not causal parameter importance.",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    save(fig, "ppo-v3-parameter-trajectory")


def optimizer_comparison() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    trial = payload["trials"]["no_warm_start_headtohead_sec20"]
    methods = (("Random Search", "random_search"), ("CEM", "cem"), ("PPO", "ppo"))
    successes = [round(trial["methods"][key]["native_success_rate"] * trial["methods"][key]["n_evaluations"])
                 for _, key in methods]
    budgets = [trial["methods"][key]["n_evaluations"] for _, key in methods]
    labels = [name for name, _ in methods]

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(labels, successes, color=("#2563EB", "#7C3AED", "#DC2626"), width=0.58)
    axis.set_ylim(0, max(2, max(successes) + 0.5))
    axis.set_ylabel("Feasible designs")
    axis.set_title("Matched-budget, no-warm-start comparison", fontweight="bold")
    for bar, success, budget in zip(bars, successes, budgets):
        axis.text(bar.get_x() + bar.get_width()/2, success + 0.06, f"{success}/{budget}",
                  ha="center", fontweight="bold")
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    fig.text(0.5, 0.015,
             "One seed (123); equal evaluation count; PPO remains grid-local while RS/CEM draw continuously.",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "optimizer-comparison")


def evaluation_efficiency() -> None:
    baseline, cached = 1200, 1085
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(("Uncached", "Exact-key cache"), (baseline, cached),
                    color=("#64748B", "#15803D"), width=0.58)
    axis.set_ylim(0, 1350)
    axis.set_ylabel("Expensive evaluator calls")
    axis.set_title("Evaluation-count reduction in the controlled cache study", fontweight="bold")
    for bar, value in zip(bars, (baseline, cached)):
        axis.text(bar.get_x() + bar.get_width()/2, value + 22, str(value), ha="center", fontweight="bold")
    axis.annotate("9.58% fewer calls", xy=(1, cached), xytext=(0.53, 1260),
                  arrowprops={"arrowstyle": "->", "color": "#334155"}, ha="center")
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    fig.text(0.5, 0.015,
             "20-seed synthetic study; bit-identical action/reward trajectories. Real-SPICE wall-clock speedup was not measured.",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "evaluation-efficiency")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    rows = load_jsonl(TRAINING)
    training_overview(rows)
    parameter_trajectory(rows)
    optimizer_comparison()
    evaluation_efficiency()
    manifest = {
        "generated_without_simulation": True,
        "sources": {
            "ppo-v3-training-overview": str(TRAINING.relative_to(ROOT)),
            "ppo-v3-parameter-trajectory": str(TRAINING.relative_to(ROOT)),
            "optimizer-comparison": str(BENCHMARK.relative_to(ROOT)),
            "evaluation-efficiency": "docs/RL_RUNTIME_EFFICIENCY.md, controlled 20-seed study",
        },
    }
    (OUTPUT / "report-graph-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
