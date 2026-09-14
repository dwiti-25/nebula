"""Recorded execution provenance, rendered without Manim or Graphviz."""
from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
from html import escape
import json
from pathlib import Path
import time
import uuid
from threading import RLock

def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped

ACTIVE = ContextVar("receiver_run_graph", default=None)

class RunGraph:
    def __init__(self, config, path=None, event_path=None):
        self.lock = RLock()
        self.path = Path(path) if path else None
        self.event_path = Path(event_path) if event_path else None
        self.started = time.perf_counter()
        self.data = {"schema_version": 1, "run_id": uuid.uuid4().hex, "status": "running",
                     "nodes": [], "edges": [], "limitations": [
                         "Synthetic backend is software-only evidence; DFE is behavioral.",
                         "Area is partial MOS channel area, not total layout area.",
                         "Cached stage times describe the original evaluation, not work repeated in this run."]}
        self.root = self.node("configuration", "Run configuration", "recorded", config)
        self.data["limitations"].append("Uncached evaluations and ngspice process invocations are different counts; startup/version probes are excluded.")
        self.count = 0
        self.cache_hits = 0
        self.spice_invocations = 0
        self.origins = {}
        self.save()

    @synchronized
    def node(self, kind, label, status, detail=None, parent=None):
        ident = f"n{len(self.data['nodes'])}"
        safe_detail = json.loads(json.dumps(detail or {}, default=str))
        self.data["nodes"].append({"id": ident, "kind": kind, "label": label, "status": status, "detail": safe_detail})
        if parent is not None:
            self.data["edges"].append({"source": parent, "target": ident, "relation": "produced"})
        return ident

    @synchronized
    def evaluation(self, evaluation, *, phase="candidate", elapsed_s=None):
        self.count += 1
        self.cache_hits += int(evaluation.cache_hit)
        if self.path:
            event_path = self.event_path or self.path.with_name(self.path.name.replace(".graph.json", ".events.jsonl"))
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event = {"event_type": "evaluation", "run_id": self.data["run_id"], "phase": phase,
                "evaluation_count": self.count, "cache_hits": self.cache_hits,
                "cache_hit": evaluation.cache_hit, "elapsed_s": elapsed_s,
                "evaluation_id": evaluation.evaluation_id, "raw_metrics": dict(evaluation.metrics),
                "parameters": asdict(evaluation.parameters), "failure_stage": evaluation.failed_stage,
                "conditions": evaluation.conditions.to_dict(), "fidelity": evaluation.fidelity.name,
                "backend": self.data["nodes"][0]["detail"].get("backend", "unknown")}
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, default=str) + "\n")
        ident = self.node("evaluation", f"{phase}: evaluation {self.count}",
            "cached" if evaluation.cache_hit else "passed" if evaluation.success else "failed",
            {"evaluation_id": evaluation.evaluation_id, "parameters": asdict(evaluation.parameters),
             "conditions": evaluation.conditions.to_dict(), "fidelity": evaluation.fidelity.name,
             "metrics": dict(evaluation.metrics), "failed_stage": evaluation.failed_stage,
             "cache_hit": evaluation.cache_hit, "elapsed_s": elapsed_s,
             "original_stage_runtime_s": evaluation.runtime_s, "provenance": evaluation.provenance}, getattr(self, "model_node", self.root))
        origin = self.origins.get(evaluation.evaluation_id)
        if evaluation.cache_hit and origin:
            self.data["edges"].append({"source": origin, "target": ident, "relation": "cache_reuse"})
        else:
            self.origins[evaluation.evaluation_id] = ident
        previous = ident
        for stage in evaluation.stages:
            previous = self.node("stage", stage.name, "cached" if evaluation.cache_hit else "passed" if stage.success else "failed", asdict(stage), previous)
        if evaluation.failed_stage and not any(stage.name.startswith("synthetic") for stage in evaluation.stages):
            expected = ["dc", "ac", "ctle_transient"]
            if evaluation.fidelity.value >= 2:
                expected.append("channel")
            if evaluation.fidelity.value >= 3:
                expected.extend(["noise", "hd3"])
            if evaluation.fidelity.value >= 2:
                expected.append("transient")
            observed = {stage.name for stage in evaluation.stages}
            self.node("skipped_stages", "Stages not executed", "skipped", {
                "stages": [stage for stage in expected if stage not in observed],
                "reason": f"Stopped at {evaluation.failed_stage}; missing measurements are not zero"}, ident)
        if self.count <= 3 or self.count % 20 == 0:
            self.save()
        return ident

    def finish(self, result=None, error=None):
        self.data.update(status="failed" if error else "completed", elapsed_s=time.perf_counter() - self.started,
                         spice_invocations=self.spice_invocations,
                         evaluation_requests=self.count, cache_hits=self.cache_hits,
                         uncached_evaluations=self.count - self.cache_hits)
        if error:
            self.node("error", "Run failed", "failed", {"message": str(error)}, self.root)
        elif result is not None:
            selection_node = self.node("selection", "Selection and verification", "recorded", {
                k: v for k, v in result.items() if k not in ("execution_graph", "final_specification")}, getattr(self, "filter_node", self.root))
            selection = result.get("selection", {})
            self.node("pvt", "PVT verification", "recorded" if selection.get("pvt") or selection.get("pvt_summary") else "not_performed",
                      {"selection": selection, "note": "Only recorded conditions count as verified."}, selection_node)
            artifacts_produced = bool(result.get("final_specification") or result.get("eye_diagram", {}).get("status") == "produced")
            self.node("artifact", "Final report / schematic / eye diagram", "produced" if artifacts_produced else "not_produced",
                      {"schematic_path": result.get("schematic_path"), "report": result.get("final_specification"),
                       "eye_diagram": result.get("eye_diagram")}, selection_node)
        self.save()

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
            if self.data["status"] != "running":
                self.path.with_suffix(".svg").write_text(render_svg(self.data), encoding="utf-8")

def observe(evaluation, *, phase="candidate", elapsed_s=None):
    graph = ACTIVE.get()
    if graph is not None:
        graph.evaluation(evaluation, phase=phase, elapsed_s=elapsed_s)

def observe_spice_invocation(parameters):
    graph = ACTIVE.get()
    if graph is not None:
        with graph.lock:
            graph.spice_invocations += 1
            graph.data["spice_invocations"] = graph.spice_invocations

def record_pipeline(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        path = kwargs.pop("graph_output", None)
        graph = RunGraph(kwargs, path)
        token = ACTIVE.set(graph)
        try:
            result = function(*args, **kwargs)
            graph.finish(result)
            result["execution_graph"] = graph.data
            return result
        except Exception as exc:
            graph.finish(error=exc)
            raise
        finally:
            ACTIVE.reset(token)
    return wrapped

def render_svg(graph, limit=180):
    """Bounded, escaped SVG overview; full node details remain in JSON."""
    nodes = graph.get("nodes", [])[:limit]
    height = 78 + 48 * len(nodes)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{height}" viewBox="0 0 1050 {height}" role="img" aria-label="Recorded run execution graph">',
             '<rect width="100%" height="100%" fill="#101820"/>',
             '<g font-family="sans-serif" font-size="13" fill="#e0ecf0">',
             f'<text x="18" y="25">Recorded run · {escape(str(graph.get("status")))} · {len(graph.get("nodes", []))} nodes (showing {len(nodes)})</text>']
    positions = {n["id"]: (260 if n["kind"] == "stage" else 25, 44 + i * 48) for i, n in enumerate(nodes)}
    for edge in graph.get("edges", []):
        if edge["source"] in positions and edge["target"] in positions:
            x, y = positions[edge["source"]]; tx, ty = positions[edge["target"]]
            parts.append(f'<path d="M{x+5},{y+32} L{x+5},{ty+16} L{tx},{ty+16}" fill="none" stroke="#506c80"/>')
    for node in nodes:
        x, y = positions[node["id"]]
        color = {"failed": "#74372e", "passed": "#174a3d", "cached": "#463d6e"}.get(node["status"], "#203a50")
        title = escape(json.dumps(node.get("detail", {}), default=str))
        label = escape(f'{node["id"]} · {node["label"]} · {node["status"]}')
        parts.append(f'<g><title>{title}</title><rect x="{x}" y="{y}" width="750" height="32" rx="5" fill="{color}"/><text x="{x+10}" y="{y+21}">{label}</text></g>')
    parts.append('</g></svg>')
    return "".join(parts)
