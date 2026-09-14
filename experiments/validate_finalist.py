"""Separate, resumable FINAL validation; never loads or trains a policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from analysis.target_assessment import final_violations
from rl.target_spec import TargetSpec
from simulator.cache import EvaluationCache
from simulator.config import PVT_GRID, QUALIFICATION_PVT_GRID, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver
from simulator.runtime import time_budget, expired
from simulator.receiver import SYNTHETIC_CHANNEL
from simulator.channel import ChannelPortMap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Parameter JSON or pipeline result JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Content-addressed cache; reuse on reruns")
    parser.add_argument("--full60", action="store_true")
    parser.add_argument("--full36", action="store_true", help="Agreed TT/SS/FF qualification grid")
    parser.add_argument("--seconds", type=float, default=None, help="Validation-only time budget")
    args = parser.parse_args(argv)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if "selection" in data:
        selected = data["selection"].get("selected")
        if selected is None:
            parser.error("input has no selected candidate")
        parameters = selected["parameters"]
    else:
        parameters = data.get("parameters", data)
    target_values = data.get("requested_target", data.get("target"))
    target = TargetSpec(**target_values) if target_values else TargetSpec.from_existing_thresholds()
    channel = data.get("channel", {})
    channel_kwargs = {"channel_path": channel.get("path", SYNTHETIC_CHANNEL),
                      "channel_port_map": ChannelPortMap(**channel.get("port_map", {}))}
    if args.full36 and args.full60:
        parser.error("select only one grid")
    conditions = QUALIFICATION_PVT_GRID if args.full36 else PVT_GRID if args.full60 else (SimulationConditions(),)
    cache = EvaluationCache(args.cache)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation protects previous evidence. Resume with a new
    # output name and the same cache: completed identities are reused safely.
    with args.output.open("x", encoding="utf-8") as stream, time_budget(args.seconds):
        for condition in conditions:
            if expired():
                break
            result = evaluate_receiver(ReceiverParameters(**parameters), condition, EvaluationFidelity.FINAL,
                                       cache=cache, **channel_kwargs)
            violations = final_violations(result.metrics, target)
            row = {"condition": condition.to_dict(), "success": result.success and not violations,
                   "violations": violations, "evaluation": asdict(result)}
            rows.append(row)
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
    complete = len(rows) == len(conditions)
    passed = complete and all(row["success"] for row in rows)
    print(json.dumps({"complete": complete, "passed": passed, "evaluated": len(rows),
                      "required": len(conditions), "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
