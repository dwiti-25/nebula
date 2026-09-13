"""Render measured CTLE and behavioral-DFE eye data as SVG/PNG artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


def render_eye_diagram(capture: dict, output: str | Path) -> dict[str, object]:
    """Render two explicitly-labelled panels and persist the source data."""

    requested = Path(output)
    base = requested.with_suffix("")
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    data_path = base.with_suffix(".json")
    for path in (svg_path, png_path, data_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(11.2, 4.8), dpi=130, layout="constrained")
    ctle_axis, dfe_axis = figure.subplots(1, 2)
    for trace in capture.get("ctle_traces", []):
        ctle_axis.plot(trace["phase_ui"], trace["voltage_v"], color="#18a999",
                       alpha=0.22, linewidth=0.7)
    ctle_axis.set(title="CTLE output eye (transistor-level SPICE)", xlabel="Phase (UI)",
                  ylabel="Differential output (V)", xlim=(0, 2))

    points = capture.get("dfe_points", [])
    for expected, color, label in ((0, "#4c78a8", "expected 0"),
                                    (1, "#e45756", "expected 1")):
        selected = [point for point in points if point["expected_bit"] == expected]
        dfe_axis.scatter([point["phase_ui"] for point in selected],
                         [point["voltage_v"] for point in selected],
                         s=4, alpha=0.20, color=color, label=label, rasterized=True)
    locked = float(capture.get("locked_phase_ui", 0.5))
    dfe_axis.axvline(locked, color="#f2cf5b", linewidth=1.2, label="locked phase")
    dfe_axis.axhline(0.0, color="#777", linewidth=0.6)
    dfe_axis.set(title="Behavioral 1-tap DFE sampling eye", xlabel="Sampling phase (UI)",
                 ylabel="Corrected sample (V)", xlim=(0, 1))
    dfe_axis.legend(frameon=False, fontsize=8)
    for axis in (ctle_axis, dfe_axis):
        axis.grid(alpha=0.18, linewidth=0.6)

    metrics = capture.get("metrics", {})
    conditions = capture.get("conditions", {})
    channel = capture.get("channel", {})
    subtitle = (
        f"{conditions.get('process_corner', '?').upper()} / {conditions.get('supply_v', '?')} V / "
        f"{conditions.get('temperature_c', '?')} C | channel {str(channel.get('checksum', 'unknown'))[:12]} | "
        f"eye H={metrics.get('dfe_locked_phase_eye_height_v', 'n/a')} V, "
        f"W={metrics.get('dfe_eye_width_ui', 'n/a')} UI, errors={metrics.get('dfe_error_count', 'n/a')}"
    )
    figure.suptitle("NEBULA receiver eye diagram\n" + subtitle, fontsize=11)
    figure.text(0.01, 0.005,
                "DFE panel contains corrected sample points, not a continuous transistor-level DFE waveform. "
                "Short deterministic PRBS evidence is not a BER claim.", fontsize=7, color="#555")

    with svg_path.open("wb") as stream:
        FigureCanvasSVG(figure).print_svg(stream, metadata={"Title": "NEBULA receiver eye diagram"})
    with png_path.open("wb") as stream:
        FigureCanvasAgg(figure).print_png(stream, metadata={"Title": "NEBULA receiver eye diagram"})
    data_path.write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return {"status": "produced", "svg_path": str(svg_path), "png_path": str(png_path),
            "data_path": str(data_path), "semantics": capture["plot_semantics"],
            "point_counts": {"ctle_traces": len(capture.get("ctle_traces", [])),
                             "dfe_samples": len(points)}}
