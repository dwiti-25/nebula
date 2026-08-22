"""Command-line entry point for one Stage 1 receiver evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulator import (
    EvaluationCache, EvaluationFidelity, NgSpiceConfig, ProcessCorner,
    ReceiverParameters, SimulationConditions, Sky130Config, evaluate_receiver,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the hierarchical SKY130 receiver wrapper")
    parser.add_argument("--rload", type=float, default=1000.0, help="load resistance in ohms")
    parser.add_argument("--rdeg", type=float, default=1000.0, help="degeneration resistance in ohms")
    parser.add_argument("--cdeg", type=float, default=0.5e-12, help="degeneration capacitance in farads")
    parser.add_argument("--itail", type=float, default=100e-6, help="tail current in amperes")
    parser.add_argument("--dfe-tap", type=float, default=0.0, help="one-tap DFE value in volts")
    parser.add_argument("--corner", choices=[item.value for item in ProcessCorner], default="tt")
    parser.add_argument("--temperature", type=float, default=27.0)
    parser.add_argument("--supply", type=float, default=1.8)
    parser.add_argument("--fidelity", choices=[item.name.lower() for item in EvaluationFidelity], default="training")
    parser.add_argument("--channel", type=Path, default=Path("channels/synthetic_regression.s4p"))
    parser.add_argument("--model-library", type=Path)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument(
        "--timeout", type=float,
        help="per-ngspice timeout in seconds (default: 180 for final, 60 otherwise)",
    )
    parser.add_argument("--cache", type=Path, default=Path(".nebula-cache"))
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    parameters = ReceiverParameters(args.rload, args.rdeg, args.cdeg, args.itail, args.dfe_tap)
    conditions = SimulationConditions(ProcessCorner(args.corner), args.temperature, args.supply)
    fidelity = EvaluationFidelity[args.fidelity.upper()]
    timeout = args.timeout if args.timeout is not None else (180.0 if fidelity >= EvaluationFidelity.FINAL else 60.0)
    try:
        ngspice = NgSpiceConfig(args.ngspice, timeout_s=timeout)
    except (TypeError, ValueError) as exc:
        _parser().error(str(exc))
    evaluation = evaluate_receiver(
        parameters,
        conditions,
        fidelity,
        channel_path=args.channel,
        sky130=Sky130Config(args.model_library),
        ngspice=ngspice,
        cache=None if args.no_cache else EvaluationCache(args.cache),
    )
    print(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True))
    return 0 if evaluation.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
