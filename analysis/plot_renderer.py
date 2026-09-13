"""Deterministic Matplotlib rendering for dashboard chart specifications."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from threading import RLock
from typing import Any, Mapping

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


STYLE_VERSION = 1
_CACHE: dict[str, bytes] = {}
_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class PlotRequest:
    plot_id: str
    evidence_hash: str
    format: str = "svg"
    style_version: int = STYLE_VERSION
    width: float = 7.2
    height: float = 4.2
    description: str = ""


def evidence_hash(chart: Mapping[str, Any]) -> str:
    encoded = json.dumps(chart, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _figure(chart: Mapping[str, Any], request: PlotRequest) -> Figure:
    figure = Figure(figsize=(request.width, request.height), dpi=120, layout="constrained")
    axis = figure.subplots()
    axis.set_title(str(chart.get("title", request.plot_id)), loc="left", fontweight="bold")
    axis.set_xlabel(str(chart.get("x_label", "")))
    axis.set_ylabel(str(chart.get("y_label", "")))
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    kind = chart.get("type")
    if kind == "line":
        for series in chart.get("series", []):
            values = [float(value) if value is not None else float("nan") for value in series.get("values", [])]
            axis.plot(series.get("x_values", range(1, len(values) + 1)), values, marker="o", markersize=2.5,
                      linewidth=1.5, label=series.get("name", "series"))
        if len(chart.get("series", [])) > 1:
            axis.legend(frameon=False)
    elif kind == "bar":
        categories = [str(value) for value in chart.get("categories", [])]
        values = [float(value or 0) for value in chart.get("values", [])]
        axis.bar(categories, values, color="#208a8a")
        axis.tick_params(axis="x", rotation=20)
    elif kind == "scatter":
        points = chart.get("points", [])
        axis.scatter([float(p["x"]) for p in points], [float(p["y"]) for p in points],
                     color="#208a8a", edgecolor="white", linewidth=0.5)
        for point in points:
            axis.annotate(str(point.get("label", "")), (float(point["x"]), float(point["y"])),
                          xytext=(4, 3), textcoords="offset points", fontsize=7)
    elif kind == "heatmap":
        rows = list(chart.get("rows", []))
        conditions = list(chart.get("conditions", []))
        matrix = [[float("nan") for _ in conditions] for _ in rows]
        for cell in chart.get("cells", []):
            try:
                row = rows.index(cell["process"])
                column = conditions.index([cell["supply_v"], cell["temperature_c"]])
            except ValueError:
                try:
                    column = conditions.index((cell["supply_v"], cell["temperature_c"]))
                except ValueError:
                    continue
            matrix[row][column] = 1.0 if cell.get("passed") else -1.0
        from matplotlib import colormaps
        colors = colormaps["RdYlGn"].with_extremes(bad="#b8bec5")
        axis.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap=colors)
        axis.set_yticks(range(len(rows)), rows)
        axis.set_xticks(range(len(conditions)), chart.get("columns", []), rotation=45, ha="right", fontsize=7)
    else:
        raise ValueError(f"unsupported chart type: {kind!r}")
    note = chart.get("note")
    if note:
        figure.text(0.01, 0.005, str(note), fontsize=7, color="#555555", wrap=True)
    return figure


def render_chart(chart: Mapping[str, Any], *, format: str = "svg") -> bytes:
    if format not in {"svg", "png"}:
        raise ValueError("format must be 'svg' or 'png'")
    digest = evidence_hash(chart)
    request = PlotRequest(str(chart.get("id", "plot")), digest, format=format,
                          description=str(chart.get("note", chart.get("title", ""))))
    cache_key = json.dumps(request.__dict__, sort_keys=True)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    figure = _figure(chart, request)
    output = io.BytesIO()
    metadata = {"Title": str(chart.get("title", request.plot_id)),
                "Description": request.description, "Creator": "NEBULA", "Date": None}
    if format == "svg":
        FigureCanvasSVG(figure).print_svg(output, metadata=metadata)
    else:
        FigureCanvasAgg(figure).print_png(output, metadata={k: v for k, v in metadata.items() if v is not None})
    rendered = output.getvalue()
    with _CACHE_LOCK:
        _CACHE[cache_key] = rendered
        while len(_CACHE) > 128:
            del _CACHE[next(iter(_CACHE))]
    return rendered
