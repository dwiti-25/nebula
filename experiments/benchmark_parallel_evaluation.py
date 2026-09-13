"""Compare identical uncached real simulation batches, without training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time
from simulator.config import Sky130Config
from simulator.ngspice import NgSpiceConfig, ngspice_identity
from simulator.receiver import ReceiverParameters, EvaluationFidelity, evaluate_receiver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fidelity", choices=("SCREENING", "TRAINING"), default="SCREENING")
    args = parser.parse_args(argv)
    identity = ngspice_identity()
    model = str(Sky130Config().resolve_model_library())
    base = ReceiverParameters(rload_ohm=2342.472156, rdeg_ohm=822.355863,
                              cdeg_f=9.99937586e-13, itail_a=0.00060283317, dfe_tap_v=-0.01109)
    batch = [base, replace(base, rload_ohm=base.rload_ohm * 1.01)]
    def evaluate(parameters):
        return evaluate_receiver(parameters, fidelity=EvaluationFidelity[args.fidelity],
                                 ngspice=NgSpiceConfig(timeout_s=60))
    runs = []
    values = []
    for workers in (1, 2):
        started = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(evaluate, batch))
        runs.append({"workers": workers, "seconds": time.perf_counter() - started,
                     "success": [r.success for r in results], "failed_stages": [r.failed_stage for r in results]})
        values.append([(r.success, r.failed_stage, r.metrics) for r in results])
    identical = values[0] == values[1]
    report = {"runs": runs, "identical_results": identical, "fidelity": args.fidelity,
              "fraction_faster": 1 - runs[1]["seconds"] / runs[0]["seconds"],
              "simulator": identity, "model": model,
              "limitation": "Small fixed-order batch, not evidence of better policy accuracy or general speedup."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0 if identical and all(runs[0]["success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
