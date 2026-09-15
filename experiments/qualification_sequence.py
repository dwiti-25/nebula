"""Restart fixed designs: nominal 512/1024 bits, then selectable 512/1024-bit PVT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from analysis.final_specification import build_final_specification_report
from analysis.pvt_selection import PVTPointResult, summarize_pvt_results
from analysis.target_assessment import final_violations
from rl.target_spec import TargetSpec
from simulator.cache import EvaluationCache
from simulator.channel import ChannelPortMap
from simulator.config import QUALIFICATION_PVT_GRID, SimulationConditions, ProcessCorner
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver, SYNTHETIC_CHANNEL
from simulator.ngspice import NgSpiceConfig


def run_sequence(data, *, output, cache, evaluator=evaluate_receiver, max_candidates=3,
                 timeout_s=360, conditions=QUALIFICATION_PVT_GRID, workers=2, source=None,
                 runtime_identity=None, pvt_pattern_bits=512):
    if pvt_pattern_bits not in (512, 1024):
        raise ValueError("pvt_pattern_bits must be 512 or 1024")
    pvt_fidelity = EvaluationFidelity.CANDIDATE if pvt_pattern_bits == 512 else EvaluationFidelity.FINAL
    settings = NgSpiceConfig(timeout_s=timeout_s)
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if not conditions:
        raise ValueError("PVT conditions must not be empty")
    for condition in conditions:
        condition.validate()
    target = TargetSpec(**data.get("target", TargetSpec.from_existing_thresholds().as_dict()))
    channel = data.get("channel", {})
    kwargs = {"cache": cache, "channel_path": channel.get("path", SYNTHETIC_CHANNEL),
              "channel_port_map": ChannelPortMap(**channel.get("port_map", {}))}
    selection = data.get("selection", {})
    candidates = [selection.get("selected"), *selection.get("alternatives", []),
                  *data.get("saved_candidates", []), *data.get("candidates", [])]
    unique = {}
    for candidate in candidates:
        if candidate and candidate.get("parameters"):
            unique.setdefault(json.dumps(candidate["parameters"], sort_keys=True), candidate)
    if not unique:
        raise ValueError("No saved candidates found; supply the original UI report with its graph/events files")
    report = {"schema_version": 2, "status": "running", "source_checkpoint": data.get("checkpoint_path", data.get("source_checkpoint")),
              "source_report": source, "simulator_timeout_s": settings.timeout_s,
              "runtime_identity": runtime_identity,
              "target": target.as_dict(), "channel": channel,
              "required_conditions": [c.to_dict() for c in conditions],
              "pvt_fidelity": pvt_fidelity.name, "pvt_pattern_bits": pvt_pattern_bits, "nominal_long_pattern_bits": 1024,
              "candidates": [], "limitations": ["Behavioral DFE, ideal tail source; no physical receiver area/power claim.",
                   ("Corners use 512-bit CANDIDATE fidelity (noise and 100mVpp HD3 included); 1024-bit / three-amplitude HD3 testing is nominal only."
                    if pvt_pattern_bits == 512 else "Corners use 1024-bit FINAL fidelity, including noise and three-amplitude HD3 testing."),
                   "1024-bit PRBS7 is a repeated short pattern; zero errors is not a BER guarantee.",
                   "Tuning, jitter and mismatch are not established by this sequence."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    def save():
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    reusable = {}
    def key(parameters, condition, fidelity):
        return json.dumps([asdict(parameters), condition.to_dict(), fidelity.name], sort_keys=True)
    if runtime_identity and data.get("runtime_identity") == runtime_identity:
        for candidate in data.get("candidates", []):
            for row in [*candidate.get("nominal", []), *candidate.get("pvt", [])]:
                evaluation = row.get("evaluation", {})
                if row.get("retryable", True) or not evaluation:
                    continue
                previous = ReceiverParameters(**evaluation["parameters"])
                raw_condition = evaluation["conditions"]
                condition = SimulationConditions(**{**raw_condition, "process_corner": ProcessCorner(raw_condition["process_corner"])})
                reusable[key(previous, condition, EvaluationFidelity[evaluation["fidelity"].upper()])] = row
    def measure(parameters, condition, fidelity):
        previous = reusable.get(key(parameters, condition, fidelity))
        if previous is not None:
            return {**previous, "reused": True}
        started = time.perf_counter()
        result = evaluator(parameters, condition, fidelity, ngspice=settings, **kwargs)
        violations = final_violations(result.metrics, target)
        return {"passed": result.success and not violations, "violations": violations,
                "retryable": any(s.retryable for s in result.stages),
                "wall_seconds": time.perf_counter() - started, "evaluation": result.to_dict()}
    save()
    for candidate in list(unique.values())[:max_candidates]:
        item = {"design_id": candidate.get("design_id"), "parameters": candidate["parameters"],
                "nominal": [], "pvt": [], "status": "nominal_screening"}
        report["candidates"].append(item)
        parameters = ReceiverParameters(**item["parameters"])
        for fidelity in (EvaluationFidelity.CANDIDATE, EvaluationFidelity.FINAL):
            measured = measure(parameters, SimulationConditions(), fidelity)
            item["nominal"].append(measured)
            save()
            print(json.dumps({"design": item["design_id"], "stage": fidelity.name, "passed": measured["passed"],
                              "failed_stage": measured["evaluation"].get("failed_stage")}), flush=True)
            if not measured["passed"]:
                item["status"] = "incomplete_nominal" if measured["retryable"] else "rejected_nominal"
                break
        if item["status"] in ("rejected_nominal", "incomplete_nominal"):
            save()
            continue
        item["status"] = "pvt_running"
        save()
        # Bound concurrency at two; record each point in grid order. Early
        # nominal rejection above avoids spending PVT on unsuitable designs.
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(measure, parameters, c, pvt_fidelity) for c in conditions]
            for condition, future in zip(conditions, futures):
                row = {"condition": condition.to_dict(), **future.result()}
                item["pvt"].append(row)
                save()
                print(json.dumps({"design": item["design_id"], "pvt_done": len(item["pvt"]), "passed": row["passed"]}), flush=True)
        points = [PVTPointResult(r["condition"]["process_corner"], r["condition"]["supply_v"],
                  r["condition"]["temperature_c"], r["passed"], r["evaluation"].get("failed_stage")) for r in item["pvt"]]
        item["status"] = ("qualified" if all(p.success for p in points) else
                          "incomplete_execution" if any(r["retryable"] for r in item["pvt"]) else "rejected_pvt")
        item["specification"] = build_final_specification_report(design_id=item["design_id"],
            parameters=item["parameters"], nominal_metrics=item["nominal"][-1]["evaluation"]["metrics"],
            nominal_source=str(output), target=target, pvt_result=summarize_pvt_results(item["design_id"], points))
        save()
    report["status"] = "complete"
    report["qualified_designs"] = [c["design_id"] for c in report["candidates"] if c["status"] == "qualified"]
    report["next_step"] = "package qualified designs; tuning and area remain separate" if report["qualified_designs"] else "diagnose nominal/PVT failures; no qualified selection"
    save()
    return report


def load_restart_input(path, condition_set="saved"):
    """Recover fixed designs and exact conditions without invoking a policy."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("backend", "real") != "real":
        raise ValueError("Restart requires a real-SPICE report")
    if not data.get("channel", {}).get("path"):
        raise ValueError("Saved channel is missing; refusing to substitute another channel")
    channel_path = Path(data["channel"]["path"])
    from simulator.provenance import sha256_file
    if data["channel"].get("checksum") and sha256_file(channel_path) != data["channel"]["checksum"]:
        raise ValueError("Saved channel checksum differs from the current channel")
    graph = data.get("execution_graph", {})
    graph_path = path.with_suffix(".graph.json")
    if not graph.get("nodes") and graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if not data.get("saved_candidates"):
        for node in graph.get("nodes", []):
            if node.get("kind") == "filter":
                detail = node["detail"]
                accepted = detail.get("accepted_episodes", [])
                data["saved_candidates"] = [
                    {"design_id": f"pipeline_ep{c['episode']}", "parameters": c["parameters"]}
                    for c in detail.get("candidates", []) if c["episode"] in accepted]
    rows = data.get("required_conditions", [])
    events = path.with_suffix(".events.jsonl")
    if not rows and events.exists():
        rows = [e["conditions"] for e in (json.loads(line) for line in events.read_text().splitlines())
                if e.get("phase") == "pvt"]
    if condition_set != "saved":
        from experiments.run_autockt_pipeline import PVT_CONDITION_SETS
        conditions = PVT_CONDITION_SETS[condition_set]
    else:
        unique = {json.dumps(r, sort_keys=True): r for r in rows}
        conditions = tuple(SimulationConditions(**{**r, "process_corner": ProcessCorner(r["process_corner"])})
                           for r in unique.values())
        if not conditions:
            raise ValueError("No saved PVT conditions; explicitly choose minimal27 or full36")
    return data, conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--simulator-timeout-seconds", type=float, default=360)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--pvt-pattern-bits", type=int, choices=(512, 1024), default=512,
                        help="Corner pattern length; 1024 also enables FINAL three-amplitude HD3. Nominal long validation stays 1024.")
    parser.add_argument("--pvt-condition-set", choices=("saved", "minimal27", "full36"), default="saved")
    parser.add_argument("--export-schematic", type=Path,
                        help="Export the first qualified design only; never overwrite an existing file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report filename")
    if args.export_schematic and args.export_schematic.exists():
        parser.error("Schematic already exists; choose a new filename")
    data, conditions = load_restart_input(args.input, args.pvt_condition_set)
    # A larger time budget does not change completed electrical measurements.
    # Reuse only when simulator, PDK, circuit, channel, target and implementation
    # fingerprints still match. Old UI reports lack this identity and rerun.
    from simulator.provenance import sha256_file, stable_fingerprint, spice_dependency_fingerprint
    from simulator.ngspice import ngspice_identity
    from simulator.config import Sky130Config
    from simulator.receiver import ROOT, BLOCK, BENCHES
    identity = ngspice_identity(NgSpiceConfig(timeout_s=args.simulator_timeout_seconds))
    identity.pop("timeout_s", None)
    identity.update(model=spice_dependency_fingerprint(Sky130Config().resolve_model_library()),
                    block=spice_dependency_fingerprint(BLOCK),
                    files={str(p.relative_to(ROOT)): sha256_file(p) for p in
                           [*BENCHES.glob("*.cir"), *(ROOT / "simulator").glob("*.py"),
                            ROOT / "analysis" / "target_assessment.py", Path(__file__)]},
                    channel=data["channel"], channel_checksum=sha256_file(Path(data["channel"]["path"])),
                    target=data.get("target"))
    result = run_sequence(data, output=args.output, cache=EvaluationCache(args.cache),
                          timeout_s=args.simulator_timeout_seconds, conditions=conditions,
                          workers=args.workers, source=str(args.input.resolve()),
                          runtime_identity=stable_fingerprint(identity), pvt_pattern_bits=args.pvt_pattern_bits)
    if args.export_schematic and result["qualified_designs"]:
        from experiments.export_final_schematic import render_final_schematic
        candidate = next(c for c in result["candidates"] if c["status"] == "qualified")
        schematic = render_final_schematic(ReceiverParameters(**candidate["parameters"]),
            achieved_metrics=candidate["nominal"][-1]["evaluation"]["metrics"],
            target_description=str(result["target"]), source_description=str(args.output.resolve()))
        args.export_schematic.parent.mkdir(parents=True, exist_ok=True)
        with args.export_schematic.open("x", encoding="utf-8") as stream:
            stream.write(schematic)
        result["schematic_path"] = str(args.export_schematic.resolve())
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0 if result["qualified_designs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
