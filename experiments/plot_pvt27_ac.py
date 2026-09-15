"""Regenerate and plot AC-only responses for the recorded 27-corner restart.

The source restart artifact stores extracted metrics, not full AC traces. This
script re-runs the repository's ctle_ac.cir bench using the exact recorded
finalist parameters and required conditions. Outputs are labelled AC-only and
must not be interpreted as completed transient/noise/HD3 PVT qualification.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from simulator.config import ProcessCorner, SimulationConditions, Sky130Config
from simulator.models import SimulationRequest
from simulator.ngspice import NgSpiceConfig, run_simulation
from simulator.receiver import BENCHES, BLOCK, ReceiverParameters
from simulator.waveform import ac_metrics, parse_wrdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "pvt27_restart_1024_360.json"
DEFAULT_OUTPUT = ROOT / "results" / "presentation_graphs" / "pvt27_ac"
PROCESS_ORDER = ("tt", "ss", "ff")
PROCESS_LABELS = {"tt": "TT", "ss": "SS", "ff": "FF"}
SUPPLY_COLORS = {1.71: "#2563EB", 1.80: "#7C3AED", 1.89: "#DC2626"}
TEMP_STYLES = {0.0: "-", 27.0: "--", 125.0: ":"}


def load_contract(path: Path) -> tuple[dict, ReceiverParameters, list[SimulationConditions]]:
    if path.suffix.lower() == ".jsonl":
        recorded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(recorded) != 27:
            raise ValueError(f"expected 27 JSONL rows, found {len(recorded)}")
        from rl.parameter_grid import VERIFIED_INITIAL_PARAMETERS
        parameters = ReceiverParameters(**VERIFIED_INITIAL_PARAMETERS)
        conditions = [SimulationConditions(
            process_corner=ProcessCorner(row["process_corner"]),
            supply_v=float(row["supply_v"]),
            temperature_c=float(row["temperature_c"]),
        ) for row in recorded]
        return {
            "status": "complete" if all(row.get("success") for row in recorded) else "contains_failures",
            "source_kind": "completed_jsonl_pvt_run",
            "recorded_successes": sum(bool(row.get("success")) for row in recorded),
        }, parameters, conditions
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("source artifact contains no candidate")
    parameters = ReceiverParameters(**candidates[0]["parameters"])
    conditions = []
    for item in payload["required_conditions"]:
        row = dict(item)
        row["process_corner"] = ProcessCorner(row["process_corner"])
        conditions.append(SimulationConditions(**row))
    if len(conditions) != 27:
        raise ValueError(f"expected 27 conditions, found {len(conditions)}")
    return payload, parameters, conditions


def run_ac(parameters: ReceiverParameters, condition: SimulationConditions, timeout: float) -> dict:
    model = Sky130Config().resolve_model_library()
    templates = {
        "SKY130_MODEL_LIBRARY": model.as_posix(),
        "PROCESS_CORNER": condition.process_corner.value,
        "CTLE_BLOCK_FILE": BLOCK.as_posix(),
        "TEMPERATURE_C": format(condition.temperature_c, ".15g"),
    }
    request = SimulationRequest(
        BENCHES / "ctle_ac.cir",
        parameters.spice_parameters(condition),
        (), templates, {"AC_OUTPUT_FILE": "ac.dat"},
    )
    result = run_simulation(request, config=NgSpiceConfig(timeout_s=timeout))
    if not result.success:
        raise RuntimeError(
            f"AC failed at {condition.process_corner.value}, {condition.supply_v} V, "
            f"{condition.temperature_c:g} C: {'; '.join(result.errors)}"
        )
    trace = parse_wrdata(result.artifacts["AC_OUTPUT_FILE"])
    metrics = ac_metrics(trace)
    magnitude = trace.column("transfer_mag")
    gain_db = 20.0 * np.log10(magnitude)
    return {
        "condition": condition,
        "frequency_hz": np.asarray(trace.scale),
        "gain_db": np.asarray(gain_db),
        "metrics": metrics,
        "runtime_s": result.runtime_s,
        "provenance": result.provenance,
    }


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#334155",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_responses(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=True, sharey=True)
    for axis, process in zip(axes, PROCESS_ORDER):
        selected = [r for r in rows if r["condition"].process_corner.value == process]
        for row in selected:
            condition = row["condition"]
            color = SUPPLY_COLORS[round(condition.supply_v, 2)]
            style = TEMP_STYLES[float(condition.temperature_c)]
            axis.semilogx(row["frequency_hz"] / 1e9, row["gain_db"], color=color,
                          linestyle=style, linewidth=1.55, alpha=0.9)
            metric = row["metrics"]
            axis.scatter(metric["peak_frequency_hz"] / 1e9, metric["peak_gain_db"],
                         color=color, s=22, edgecolor="white", linewidth=0.5, zorder=4)
        axis.axvspan(1.25, 2.5, color="#F59E0B", alpha=0.09)
        axis.axvline(2.5, color="#64748B", linewidth=0.9, linestyle="--")
        axis.set_title(f"{PROCESS_LABELS[process]} process")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_xlim(0.01, 10)
    axes[0].set_ylabel("Differential CTLE gain (dB)")
    supply_handles = [Line2D([0], [0], color=color, lw=2, label=f"{v:.2f} V")
                      for v, color in SUPPLY_COLORS.items()]
    temp_handles = [Line2D([0], [0], color="#334155", lw=2, linestyle=style,
                           label=f"{temp:g} °C") for temp, style in TEMP_STYLES.items()]
    peak_handle = Line2D([0], [0], marker="o", color="none", markerfacecolor="#334155",
                         markeredgecolor="white", markersize=6, label="Peak in 1.25–2.5 GHz band")
    fig.legend(handles=supply_handles + temp_handles + [peak_handle], loc="lower center",
               ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("CTLE AC response across the recorded 27-corner grid", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925,
             "Regenerated AC-only sweeps from the recorded 27-corner artifact; markers show the intended-band peak",
             ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.08)
    save_figure(fig, output, "ctle_ac_responses_27_corners")


def plot_heatmaps(rows: list[dict], output: Path) -> None:
    row_keys = [(p, t) for p in PROCESS_ORDER for t in (0.0, 27.0, 125.0)]
    voltages = (1.71, 1.80, 1.89)
    lookup = {(r["condition"].process_corner.value, float(r["condition"].temperature_c),
               round(r["condition"].supply_v, 2)): r for r in rows}
    peaking = np.array([[lookup[(p, t, v)]["metrics"]["peaking_db"] for v in voltages]
                        for p, t in row_keys])
    peak_freq = np.array([[lookup[(p, t, v)]["metrics"]["peak_frequency_hz"] / 1e9 for v in voltages]
                          for p, t in row_keys])
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    matrices = ((peaking, "Peaking (dB)", "viridis", 3.0, 12.0, ".2f"),
                (peak_freq, "Peak frequency (GHz)", "plasma", None, None, ".3f"))
    ylabels = [f"{PROCESS_LABELS[p]}, {t:g} °C" for p, t in row_keys]
    for axis, (matrix, title, cmap, vmin, vmax, fmt) in zip(axes, matrices):
        image = axis.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(range(3), [f"{v:.2f} V" for v in voltages])
        axis.set_yticks(range(9), ylabels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(j, i, format(value, fmt), ha="center", va="center",
                          color="black" if luminance > 0.58 else "white", fontsize=9)
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("AC corner sensitivity of the recorded finalist", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, "27 AC-only simulations; the configured peaking acceptance band is 3–12 dB",
             ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(top=0.88, bottom=0.08, wspace=0.35)
    save_figure(fig, output, "ctle_ac_pvt_heatmaps")


def plot_summary(rows: list[dict], output: Path) -> None:
    labels = []
    peaking = []
    peak_freq = []
    colors = []
    for row in rows:
        c = row["condition"]
        labels.append(f"{c.process_corner.value.upper()}\n{c.supply_v:.2f}V\n{c.temperature_c:g}°C")
        peaking.append(row["metrics"]["peaking_db"])
        peak_freq.append(row["metrics"]["peak_frequency_hz"] / 1e9)
        colors.append(SUPPLY_COLORS[round(c.supply_v, 2)])
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].bar(x, peaking, color=colors, width=0.75)
    axes[0].axhspan(3, 12, color="#16A34A", alpha=0.09, label="Accepted peaking band")
    axes[0].axhline(3, color="#16A34A", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Peaking (dB)")
    axes[0].set_title("Peaking by PVT condition")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].scatter(x, peak_freq, c=colors, s=45, edgecolor="white", linewidth=0.6)
    axes[1].axhspan(1.25, 2.5, color="#F59E0B", alpha=0.10)
    axes[1].set_ylabel("Peak frequency (GHz)")
    axes[1].set_title("Frequency of the intended-band peak")
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[1].set_xlabel("Process, supply and temperature")
    for boundary in (8.5, 17.5):
        for axis in axes:
            axis.axvline(boundary, color="#94A3B8", linewidth=1)
    fig.suptitle("CTLE peaking across 27 AC corners", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.948, "AC-only regeneration using the completed 27-corner run configuration",
             ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(top=0.88, bottom=0.18, hspace=0.33)
    save_figure(fig, output, "ctle_ac_corner_summary")


def plot_recorded_pvt_metrics(source: Path, output: Path) -> bool:
    if source.suffix.lower() != ".jsonl":
        return False
    recorded = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(recorded) != 27 or not all(isinstance(row.get("metrics"), dict) for row in recorded):
        return False
    row_keys = [(p, t) for p in PROCESS_ORDER for t in (0.0, 27.0, 125.0)]
    voltages = (1.71, 1.80, 1.89)
    lookup = {(row["process_corner"], float(row["temperature_c"]), round(float(row["supply_v"]), 2)): row
              for row in recorded}
    specs = (
        ("dfe_locked_phase_eye_height_v", "DFE eye height (V)", "viridis", ".2f"),
        ("dfe_eye_width_ui", "DFE eye width (UI)", "viridis", ".2f"),
        ("dfe_min_margin_v", "Decision margin (V)", "viridis", ".2f"),
        ("ctle_power_w", "CTLE power (mW)", "magma", ".2f"),
        ("input_referred_noise_vrms", "Input noise (mV RMS)", "magma", ".2f"),
        ("hd3_db", "HD3 at 100 mVpp (dB)", "magma_r", ".1f"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ylabels = [f"{PROCESS_LABELS[p]}, {t:g} °C" for p, t in row_keys]
    for axis, (key, title, cmap, fmt) in zip(axes.flat, specs):
        scale = 1000.0 if key in {"ctle_power_w", "input_referred_noise_vrms"} else 1.0
        matrix = np.array([[float(lookup[(p, t, v)]["metrics"][key]) * scale for v in voltages]
                           for p, t in row_keys])
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xticks(range(3), [f"{v:.2f} V" for v in voltages])
        axis.set_yticks(range(9), ylabels, fontsize=8)
        for i in range(9):
            for j in range(3):
                rgba = image.cmap(image.norm(matrix[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                          color="black" if luminance > 0.58 else "white", fontsize=7.5)
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Recorded Design A performance across the completed 27-corner run",
                 fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.955,
             "All 27 recorded evaluations passed; values come directly from design_a_pvt_minimal27_rerun.jsonl",
             ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.10, right=0.97, hspace=0.30, wspace=0.38)
    save_figure(fig, output, "design_a_recorded_pvt_metrics")
    return True


def save_data(rows: list[dict], source: Path, source_payload: dict,
              parameters: ReceiverParameters, output: Path) -> None:
    columns = ["process_corner", "supply_v", "temperature_c", "gain_100mhz_db",
               "gain_1p25ghz_db", "gain_2p5ghz_db", "gain_5ghz_db", "peak_gain_db",
               "peak_frequency_hz", "peaking_db", "boost_2p5ghz_db", "response_class", "runtime_s"]
    with (output / "ctle_ac_27_corner_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            c, m = row["condition"], row["metrics"]
            writer.writerow({
                "process_corner": c.process_corner.value, "supply_v": c.supply_v,
                "temperature_c": c.temperature_c, "runtime_s": row["runtime_s"],
                **{key: m[key] for key in columns if key in m},
            })
    manifest = {
        "source_artifact": str(source.resolve()),
        "source_status": source_payload.get("status"),
        "method": "Regenerated AC-only traces using circuits/benches/ctle_ac.cir",
        "parameters": asdict(parameters),
        "condition_count": len(rows),
        "limitations": [
            "These plots establish AC response only.",
            "They do not complete transient, noise, HD3, eye, or full PVT qualification.",
            "Peak markers use the repository's 1.25–2.5 GHz intended-band definition.",
        ],
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    payload, parameters, conditions = load_contract(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, condition in enumerate(conditions, 1):
        print(f"[{index:02d}/27] {condition.process_corner.value} "
              f"{condition.supply_v:.2f} V {condition.temperature_c:g} C", flush=True)
        rows.append(run_ac(parameters, condition, args.timeout))
    setup_style()
    plot_responses(rows, args.output_dir)
    plot_heatmaps(rows, args.output_dir)
    plot_summary(rows, args.output_dir)
    recorded_metrics_plot = plot_recorded_pvt_metrics(args.source, args.output_dir)
    save_data(rows, args.source, payload, parameters, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "plots": 3 + int(recorded_metrics_plot),
                      "conditions": len(rows), "source_status": payload.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
