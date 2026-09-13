"""Fail-closed finalist sequence: 512 bits -> 1024 bits -> agreed 36-point PVT."""
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
from simulator.config import QUALIFICATION_PVT_GRID, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver, SYNTHETIC_CHANNEL
from simulator.ngspice import NgSpiceConfig


def run_sequence(data, *, output, cache, evaluator=evaluate_receiver, max_candidates=3):
    target = TargetSpec(**data.get("target", TargetSpec.from_existing_thresholds().as_dict()))
    channel = data.get("channel", {})
    kwargs = {"cache": cache, "channel_path": channel.get("path", SYNTHETIC_CHANNEL),
              "channel_port_map": ChannelPortMap(**channel.get("port_map", {}))}
    selection = data.get("selection", {})
    candidates = [selection.get("selected"), *selection.get("alternatives", [])]
    unique = {}
    for candidate in candidates:
        if candidate and candidate.get("parameters"):
            unique.setdefault(json.dumps(candidate["parameters"], sort_keys=True), candidate)
    report = {"schema_version": 1, "status": "running", "source_checkpoint": data.get("checkpoint_path"),
              "target": target.as_dict(), "channel": channel,
              "required_conditions": [c.to_dict() for c in QUALIFICATION_PVT_GRID],
              "pvt_fidelity": "CANDIDATE", "pvt_pattern_bits": 512, "nominal_long_pattern_bits": 1024,
              "candidates": [], "limitations": ["Behavioral DFE, ideal tail source; no physical receiver area/power claim.",
                   "Corners use 512-bit CANDIDATE fidelity (noise and 100mVpp HD3 included); 1024-bit / three-amplitude HD3 testing is nominal only.",
                   "1024-bit PRBS7 is a repeated short pattern; zero errors is not a BER guarantee.",
                   "Tuning, jitter and mismatch are not established by this sequence."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    def save():
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    def measure(parameters, condition, fidelity):
        started = time.perf_counter()
        result = evaluator(parameters, condition, fidelity, ngspice=NgSpiceConfig(timeout_s=180), **kwargs)
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
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(measure, parameters, c, EvaluationFidelity.CANDIDATE) for c in QUALIFICATION_PVT_GRID]
            for condition, future in zip(QUALIFICATION_PVT_GRID, futures):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    result = run_sequence(json.loads(args.input.read_text()), output=args.output, cache=EvaluationCache(args.cache))
    return 0 if result["qualified_designs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
