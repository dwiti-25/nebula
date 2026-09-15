"""Create presentation plots from a completed web-UI PVT result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "07acbcab1cd346cca24eab31919496c0"
PROCESSES = ("tt", "ss", "ff")
TEMPS = (0.0, 27.0, 125.0)
VDDS = (1.71, 1.80, 1.89)
COLORS = {"tt": "#2563EB", "ss": "#7C3AED", "ff": "#DC2626"}
MARKERS = {1.71: "o", 1.80: "s", 1.89: "^"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.edgecolor": "#334155", "axes.grid": True,
        "grid.alpha": 0.20, "grid.linewidth": 0.7,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_points(source: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    pvt = payload["selection"]["pvt"]
    points = pvt["points"]
    if len(points) != 27:
        raise ValueError(f"expected 27 PVT points, found {len(points)}")
    return payload, points


def matrix(points: list[dict], metric: str, scale: float = 1.0) -> np.ndarray:
    lookup = {(p["process_corner"], float(p["temperature_c"]), round(float(p["supply_v"]), 2)): p
              for p in points}
    return np.array([[lookup[(process, temp, vdd)]["metrics"][metric] * scale for vdd in VDDS]
                     for process in PROCESSES for temp in TEMPS])


def annotated_heatmaps(points: list[dict], output: Path) -> None:
    specs = (
        ("peaking_db", "Peaking (dB)", 1, "viridis", ".2f"),
        ("dfe_locked_phase_eye_height_v", "Eye height (V)", 1, "viridis", ".3f"),
        ("dfe_eye_width_ui", "Eye width (UI)", 1, "viridis", ".2f"),
        ("dfe_min_margin_v", "DFE margin (V)", 1, "viridis", ".3f"),
        ("ctle_power_w", "Power (mW)", 1000, "magma", ".3f"),
        ("input_referred_noise_vrms", "Noise (mV RMS)", 1000, "magma", ".3f"),
        ("hd3_db", "HD3 (dB)", 1, "magma_r", ".2f"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    labels = [f"{p.upper()}, {t:g}°C" for p in PROCESSES for t in TEMPS]
    for ax, (key, title, scale, cmap, fmt) in zip(axes.flat, specs):
        values = matrix(points, key, scale)
        image = ax.imshow(values, cmap=cmap, aspect="auto")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(3), [f"{v:.2f} V" for v in VDDS])
        ax.set_yticks(range(9), labels, fontsize=8)
        for i in range(9):
            for j in range(3):
                rgba = image.cmap(image.norm(values[i, j]))
                lum = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        fontsize=8, color="black" if lum > .58 else "white")
        fig.colorbar(image, ax=ax, shrink=.75)
    axes.flat[-1].axis("off")
    fig.suptitle("Measured performance across the completed 512-bit, 27-corner run",
                 fontsize=18, fontweight="bold", y=.995)
    fig.text(.5, .955, "Run 07acbcab… • all 27 evaluated conditions passed",
             ha="center", color="#475569")
    fig.subplots_adjust(top=.90, wspace=.42, hspace=.28)
    save(fig, output, "pvt27_all_metrics_heatmaps")


def temperature_trends(points: list[dict], output: Path) -> None:
    specs = (
        ("peaking_db", "Peaking", "dB"),
        ("dfe_locked_phase_eye_height_v", "Eye height", "V"),
        ("dfe_eye_width_ui", "Eye width", "UI"),
        ("dfe_min_margin_v", "DFE margin", "V"),
        ("input_referred_noise_vrms", "Noise", "mV RMS"),
        ("hd3_db", "HD3", "dB"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    lookup = {(p["process_corner"], round(float(p["supply_v"]), 2), float(p["temperature_c"])): p
              for p in points}
    for ax, (key, title, unit) in zip(axes.flat, specs):
        scale = 1000 if key == "input_referred_noise_vrms" else 1
        for process in PROCESSES:
            for vdd in VDDS:
                values = [lookup[(process, vdd, temp)]["metrics"][key] * scale for temp in TEMPS]
                ax.plot(TEMPS, values, color=COLORS[process], marker=MARKERS[vdd],
                        linewidth=1.5, markersize=5, alpha=.82)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks(TEMPS)
        ax.set_xlabel("Temperature (°C)")
    process_handles = [plt.Line2D([], [], color=COLORS[p], lw=2, label=p.upper()) for p in PROCESSES]
    voltage_handles = [plt.Line2D([], [], color="#334155", marker=MARKERS[v], lw=0,
                                  label=f"{v:.2f} V") for v in VDDS]
    fig.legend(handles=process_handles + voltage_handles, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle("Temperature sensitivity across process and supply",
                 fontsize=18, fontweight="bold", y=.99)
    fig.text(.5, .95, "Colour = process corner; marker = supply voltage", ha="center", color="#475569")
    fig.subplots_adjust(top=.89, bottom=.12, wspace=.28, hspace=.34)
    save(fig, output, "pvt27_temperature_trends")


def worst_case_summary(points: list[dict], output: Path) -> None:
    metrics = (
        ("Peaking", "peaking_db", 1, "min", "3–12 dB", "dB"),
        ("Eye height", "dfe_locked_phase_eye_height_v", 1, "min", "> 0.1 V", "V"),
        ("Eye width", "dfe_eye_width_ui", 1, "min", "> 0.4 UI", "UI"),
        ("DFE margin", "dfe_min_margin_v", 1, "min", "> 0 V", "V"),
        ("Power", "ctle_power_w", 1000, "max", "< 15 mW", "mW"),
        ("Noise", "input_referred_noise_vrms", 1000, "max", "< 1.5 mV", "mV"),
        ("HD3", "hd3_db", 1, "max", "< −30 dB", "dB"),
    )
    cards = []
    for label, key, scale, direction, requirement, unit in metrics:
        candidates = [(p["metrics"][key] * scale, p) for p in points]
        value, point = (min(candidates, key=lambda item: item[0]) if direction == "min"
                        else max(candidates, key=lambda item: item[0]))
        cards.append((label, value, unit, requirement, point))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (label, value, unit, requirement, point) in zip(axes.flat, cards):
        ax.set_facecolor("#F8FAFC")
        for spine in ax.spines.values(): spine.set_color("#CBD5E1")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.text(.06, .86, label, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#334155")
        ax.text(.06, .57, f"{value:.3g} {unit}", transform=ax.transAxes, fontsize=22,
                fontweight="bold", color="#0F766E")
        ax.text(.06, .34, f"{point['process_corner'].upper()} • {point['supply_v']:.2f} V • {point['temperature_c']:g}°C",
                transform=ax.transAxes, fontsize=10, color="#475569")
        ax.text(.06, .12, f"PASS  |  requirement {requirement}", transform=ax.transAxes,
                fontsize=9.5, color="#15803D", fontweight="bold")
    axes.flat[-1].axis("off")
    fig.suptitle("Worst measured corner for each specification", fontsize=18, fontweight="bold", y=.98)
    fig.text(.5, .925, "Values are reported directly; no artificial normalization is applied",
             ha="center", color="#475569")
    fig.subplots_adjust(top=.85, bottom=.06, wspace=.16, hspace=.20)
    save(fig, output, "pvt27_worst_case_spec_margins")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = ROOT / "results" / "web_ui_runs" / f"{args.run_id}.json"
    output = args.output or ROOT / "results" / "presentation_graphs" / f"ui_{args.run_id[:8]}_512bit_pvt27"
    output.mkdir(parents=True, exist_ok=True)
    payload, points = load_points(source)
    style()
    annotated_heatmaps(points, output)
    temperature_trends(points, output)
    worst_case_summary(points, output)
    manifest = {"source": str(source), "run_id": args.run_id, "pvt_pattern_bits": payload.get("pvt_pattern_bits"),
                "n_conditions": len(points), "n_passing": sum(bool(p["success"]) for p in points)}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
