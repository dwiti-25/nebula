"""Checkpointed, reproducible pre-RL random-search baseline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Callable

from simulator import (
    ChannelPortMap, EvaluationCache, EvaluationFidelity, NgSpiceConfig,
    ProcessCorner, ReceiverParameters, SimulationConditions, Sky130Config, evaluate_receiver,
    validate_s4p_channel,
)
from simulator.channel import load_s4p
from simulator.ngspice import ngspice_identity
from simulator.provenance import (
    git_identity, sha256_file, spice_dependency_fingerprint,
    spice_dependency_manifest, stable_fingerprint,
)
from simulator.receiver import SYNTHETIC_CHANNEL
from simulator.waveform import require_numpy
from simulator.rl_adapter import (
    ACTION_BOUNDS, ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, REWARD_VERSION,
    normalized_action_to_parameters, reward_v1,
)


SEARCH_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 1
PREFLIGHT_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA_VERSION = 1
SAMPLING_POLICIES = ("uniform", "constraint_aware_v1")


def _sample_actions(
    count: int,
    seed: int,
    policy: str,
    conditions: SimulationConditions,
    version: str = "v1",
) -> list[tuple[float, ...]]:
    """Generate deterministic actions, optionally rejecting obvious Stage-1 dead zones."""
    if policy not in SAMPLING_POLICIES:
        raise ValueError(f"unsupported sampling policy: {policy}")
    from simulator.design_schema import parameter_bounds
    generator = random.Random(seed)
    actions: list[tuple[float, ...]] = []
    attempts = 0
    while len(actions) < count:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("sampling policy could not produce enough actions")
        action = tuple(generator.uniform(-1.0, 1.0) for _ in range(len(parameter_bounds(version))))
        if policy == "constraint_aware_v1":
            parameters = normalized_action_to_parameters(action, version=version)
            # First-order resistive-load common-mode estimate.  This rejects
            # combinations that cannot meet the DC headroom gate even before
            # device-model simulation; the evaluator remains authoritative.
            estimated_common_mode_v = (
                conditions.supply_v - 0.5 * parameters.rload_ohm * parameters.itail_a
            )
            degeneration_time_constant_s = parameters.rdeg_ohm * parameters.cdeg_f
            if not 0.25 <= estimated_common_mode_v <= conditions.supply_v - 0.25:
                continue
            # The Stage-1 source-degeneration zero must be close enough to the
            # 1.25--2.5 GHz target band to make an AC pass plausible.
            if not 0.1e-9 <= degeneration_time_constant_s <= 1.0e-9:
                continue
            # Extremely large first-tap corrections are useful endpoint tests,
            # but are poor random-baseline seeds for this modest synthetic ISI.
            if abs(parameters.dfe_tap_v) > 0.2:
                continue
        actions.append(action)
    return actions


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            index = int(json.loads(line)["candidate_index"])
            if index in completed:
                raise ValueError(f"duplicate candidate index {index} in checkpoint")
            completed.add(index)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint row {line_number} in {path}") from exc
    return completed


def summarize_receiver_search(output: str | Path) -> dict[str, object]:
    """Validate and summarize a receiver-search checkpoint without rerunning it."""
    source = Path(output)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"search manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_id = manifest.get("manifest_id")
    if not isinstance(expected_manifest_id, str):
        raise ValueError("search manifest has no manifest_id")
    fingerprint_input = dict(manifest)
    del fingerprint_input["manifest_id"]
    if stable_fingerprint(fingerprint_input) != expected_manifest_id:
        raise ValueError("search manifest fingerprint does not match its contents")

    rows: list[dict[str, object]] = []
    if source.is_file():
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in checkpoint row {line_number}") from exc
            if row.get("manifest_id") != expected_manifest_id:
                raise ValueError(f"checkpoint row {line_number} belongs to another manifest")
            rows.append(row)

    count = int(manifest["count"])
    indices = [int(row["candidate_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("checkpoint contains duplicate candidate indices")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError("checkpoint contains an out-of-range candidate index")
    rewards = []
    for row in rows:
        reward = float(row["reward"])
        if not math.isfinite(reward):
            raise ValueError("checkpoint contains a non-finite reward")
        rewards.append(reward)
    ranked = sorted(
        rows, key=lambda row: (-float(row["reward"]), int(row["candidate_index"])),
    )[:10]
    failure_counts = Counter(
        str(row.get("failed_stage") or "success") for row in rows
    )
    evaluation_ids = [str(row.get("evaluation_id") or "") for row in rows]
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "manifest_id": expected_manifest_id,
        "requested_candidates": count,
        "completed_candidates": len(rows),
        "missing_candidate_indices": sorted(set(range(count)) - set(indices)),
        "complete": len(rows) == count,
        "successful_candidates": sum(bool(row.get("success")) for row in rows),
        "failure_stage_counts": dict(sorted(failure_counts.items())),
        "unique_evaluation_ids": len(set(evaluation_ids)),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "top_candidates": [
            {
                "candidate_index": int(row["candidate_index"]),
                "evaluation_id": row.get("evaluation_id"),
                "success": bool(row.get("success")),
                "reward": float(row["reward"]),
                "parameters": row.get("parameters"),
            }
            for row in ranked
        ],
    }


def preflight_receiver_search(
    *,
    channel_path: str | Path,
    channel_port_map: ChannelPortMap = ChannelPortMap(),
    sky130: Sky130Config = Sky130Config(),
    ngspice: NgSpiceConfig = NgSpiceConfig(),
) -> dict[str, object]:
    """Resolve and fingerprint every external input before a costly search."""
    numpy = require_numpy()
    model_path = sky130.resolve_model_library()
    executable_path = ngspice.resolve_executable()
    channel = load_s4p(channel_path, port_map=channel_port_map)
    channel_metrics = validate_s4p_channel(channel)
    maximum_spacing_hz = float(numpy.max(numpy.diff(channel.frequency_hz)))
    unwrap_delay_limit_s = 0.5 / maximum_spacing_hz
    delay_fraction = channel_metrics["channel_bulk_delay_s"] / unwrap_delay_limit_s
    warnings = []
    if delay_fraction > 0.8:
        warnings.append(
            "fitted bulk delay approaches the phase-unwrapping ambiguity limit; "
            "qualify or resample the channel with an independent RF tool"
        )
    return {
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "ready_for_search": True,
        "sky130_model_path": str(model_path),
        "sky130_model_checksum": spice_dependency_fingerprint(model_path),
        "sky130_model_root_checksum": sha256_file(model_path),
        "sky130_model_dependency_count": len(spice_dependency_manifest(model_path)),
        "ngspice_executable": str(executable_path),
        **ngspice_identity(ngspice),
        "channel_path": str(channel.path),
        "channel_checksum": channel.checksum,
        "channel_port_map": asdict(channel_port_map),
        "channel_frequency_points": int(len(channel.frequency_hz)),
        "channel_maximum_spacing_hz": maximum_spacing_hz,
        "phase_unwrap_delay_limit_s": unwrap_delay_limit_s,
        "bulk_delay_fraction_of_unwrap_limit": delay_fraction,
        "channel_metrics": channel_metrics,
        "warnings": warnings,
        "qualification_notice": (
            "Built-in checks are software preflight only; independent RF-tool and "
            "model-owner qualification remain required."
        ),
    }


def verify_receiver_search(
    output: str | Path,
    *,
    top: int = 5,
    metric_rtol: float = 1e-6,
    metric_atol: float = 1e-9,
    evaluator: Callable[..., object] = evaluate_receiver,
    evaluator_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Replay top candidates without cache and compare their complete metric sets."""
    if top <= 0:
        raise ValueError("verification candidate count must be positive")
    if metric_rtol < 0 or metric_atol < 0 or not all(map(math.isfinite, (metric_rtol, metric_atol))):
        raise ValueError("metric tolerances must be finite and nonnegative")
    source = Path(output)
    summary = summarize_receiver_search(source)
    if not summary["complete"]:
        raise ValueError("cannot verify an incomplete receiver-search checkpoint")
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_index = {int(row["candidate_index"]): row for row in rows}
    selected = sorted(
        by_index.values(),
        key=lambda row: (-float(row["reward"]), int(row["candidate_index"])),
    )[:top]

    conditions_data = dict(manifest["conditions"])
    conditions_data["process_corner"] = ProcessCorner(conditions_data["process_corner"])
    conditions = SimulationConditions(**conditions_data)
    port_map = ChannelPortMap(**manifest["channel_port_map"])
    fidelity = EvaluationFidelity[str(manifest["fidelity"]).upper()]
    effective_kwargs = dict(evaluator_kwargs or {})
    effective_kwargs.pop("cache", None)

    input_verification: dict[str, object]
    if evaluator is evaluate_receiver:
        sky130 = effective_kwargs.get("sky130") or Sky130Config()
        ngspice = effective_kwargs.get("ngspice") or NgSpiceConfig()
        input_verification = preflight_receiver_search(
            channel_path=manifest["channel_path"], channel_port_map=port_map,
            sky130=sky130, ngspice=ngspice,
        )
        checks = {
            "sky130_model_checksum": input_verification["sky130_model_checksum"],
            "channel_checksum": input_verification["channel_checksum"],
            "ngspice_executable": input_verification["ngspice_executable"],
            "ngspice_version": input_verification["ngspice_version"],
            "initialization_checksum": input_verification["initialization_checksum"],
            "solver": input_verification["solver"],
        }
        mismatched_inputs = {
            name: {"manifest": manifest.get(name), "current": value}
            for name, value in checks.items() if manifest.get(name) != value
        }
        if mismatched_inputs:
            raise ValueError(f"verification inputs do not match manifest: {mismatched_inputs}")
    else:
        input_verification = {"test_double": True}

    results = []
    for stored in selected:
        parameters = ReceiverParameters(**stored["parameters"])
        replay = evaluator(
            parameters, conditions, fidelity=fidelity,
            channel_path=manifest["channel_path"], channel_port_map=port_map,
            **effective_kwargs,
        )
        mismatches: list[str] = []
        if replay.evaluation_id != stored.get("evaluation_id"):
            mismatches.append("evaluation_id")
        if bool(replay.success) != bool(stored.get("success")):
            mismatches.append("success")
        if replay.failed_stage != stored.get("failed_stage"):
            mismatches.append("failed_stage")
        stored_metrics = stored.get("metrics") or {}
        replay_metrics = replay.metrics
        for name, expected in stored_metrics.items():
            if name not in replay_metrics:
                mismatches.append(f"metric:{name}:missing")
                continue
            actual = replay_metrics[name]
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    matches = math.isclose(
                        float(actual), float(expected), rel_tol=metric_rtol, abs_tol=metric_atol,
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                mismatches.append(f"metric:{name}")
        results.append({
            "candidate_index": int(stored["candidate_index"]),
            "stored_evaluation_id": stored.get("evaluation_id"),
            "replay_evaluation_id": replay.evaluation_id,
            "passed": not mismatches,
            "mismatches": mismatches,
        })
    return {
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "metric_rtol": metric_rtol,
        "metric_atol": metric_atol,
        "verified_candidates": len(results),
        "passed": all(item["passed"] for item in results),
        "results": results,
        "input_verification": input_verification,
    }


def run_receiver_search(
    *,
    count: int = 100,
    seed: int = 0,
    output: str | Path = "results/receiver_random_search.jsonl",
    channel_path: str | Path = SYNTHETIC_CHANNEL,
    channel_port_map: ChannelPortMap = ChannelPortMap(),
    conditions: SimulationConditions = SimulationConditions(),
    fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
    sampling_policy: str = "uniform",
    cache_path: str | Path | None = ".nebula-cache",
    evaluator: Callable[..., object] = evaluate_receiver,
    evaluator_kwargs: dict[str, object] | None = None,
    version: str = "v1",
) -> list[dict[str, object]]:
    from simulator.design_schema import parameter_bounds
    from rl.parameter_grid import build_parameter_grids, quantize_parameters
    grids = build_parameter_grids(version=version) if version == "v3" else None
    if not 1 <= count <= 500:
        raise ValueError("baseline count must be between 1 and 500")
    if sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f"unsupported sampling policy: {sampling_policy}")
    destination = Path(output)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    effective_kwargs = dict(evaluator_kwargs or {})
    if cache_path is not None:
        effective_kwargs.setdefault("cache", EvaluationCache(cache_path))
    runtime_identity: dict[str, object]
    if evaluator is evaluate_receiver:
        sky130 = effective_kwargs.get("sky130") or Sky130Config()
        ngspice = effective_kwargs.get("ngspice") or NgSpiceConfig()
        model_path = sky130.resolve_model_library()
        runtime_identity = {
            "sky130_model_checksum": spice_dependency_fingerprint(model_path),
            "sky130_model_root_checksum": sha256_file(model_path),
            "sky130_model_dependency_count": len(spice_dependency_manifest(model_path)),
            **ngspice_identity(ngspice),
            **git_identity(Path(__file__).resolve().parents[1]),
        }
    else:
        runtime_identity = {"test_double": True}
    repository_root = Path(__file__).resolve().parents[1]
    result_files = (
        sorted((repository_root / "simulator").glob("*.py"))
        + sorted((repository_root / "circuits" / "benches").glob("*.cir"))
        + sorted((repository_root / "circuits" / "blocks").glob("*.spice"))
        + [Path(__file__).resolve()]
    )
    manifest = {
        "search_schema_version": SEARCH_SCHEMA_VERSION,
        "action_schema_version": 3 if version == "v3" else ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_version": REWARD_VERSION,
        "count": count,
        "seed": seed,
        "sampling_policy": sampling_policy,
        "fidelity": fidelity.name.lower(),
        "conditions": conditions.to_dict(),
        "channel_port_map": asdict(channel_port_map),
        "action_bounds": parameter_bounds(version),
        "parameter_grid": {name: list(grid.values) for name, grid in grids.items()} if grids else None,
        "channel_path": str(Path(channel_path).resolve()),
        "channel_checksum": sha256_file(channel_path),
        "python_version": sys.version,
        "numpy_version": require_numpy().__version__,
        "cache_policy": "disabled" if cache_path is None else str(Path(cache_path).resolve()),
        "evaluator_kwargs": {key: str(value) for key, value in effective_kwargs.items() if key != "cache"},
        "implementation_checksum": stable_fingerprint({
            str(path.relative_to(repository_root)): sha256_file(path)
            for path in result_files
        }),
        **runtime_identity,
    }
    # Normalize tuples/enums exactly as they will appear on disk before
    # fingerprinting and resume comparison.
    manifest = json.loads(json.dumps(manifest, sort_keys=True, default=str))
    manifest["manifest_id"] = stable_fingerprint(manifest)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing search manifest does not match this requested run")
    else:
        _atomic_json(manifest_path, manifest)

    completed = _completed_indices(destination)
    if any(index < 0 or index >= count for index in completed):
        raise ValueError("checkpoint contains an out-of-range candidate index")
    actions = _sample_actions(count, seed, sampling_policy, conditions, version=version)
    rows: list[dict[str, object]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    for index, action in enumerate(actions):
        if index in completed:
            continue
        parameters: ReceiverParameters = normalized_action_to_parameters(action, version=version)
        if grids:
            parameters = quantize_parameters(parameters, grids)
        start_time = time.perf_counter()
        evaluation = evaluator(
            parameters, conditions, fidelity=fidelity, channel_path=channel_path,
            channel_port_map=channel_port_map, **effective_kwargs,
        )
        wall_clock_s = time.perf_counter() - start_time
        row = {
            "candidate_index": index,
            "action": action,
            "parameters": asdict(parameters),
            "success": evaluation.success,
            "failed_stage": evaluation.failed_stage,
            "evaluation_id": evaluation.evaluation_id,
            "reward": reward_v1(evaluation),
            "metrics": evaluation.metrics,
            "provenance": evaluation.provenance,
            "violations": [
                violation for stage in evaluation.stages for violation in stage.violations
            ],
            "manifest_id": manifest["manifest_id"],
            "wall_clock_s": wall_clock_s,
        }
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
    return rows


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run/resume the receiver pre-RL random baseline")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/receiver_random_search.jsonl"))
    parser.add_argument("--channel", type=Path, default=SYNTHETIC_CHANNEL)
    parser.add_argument(
        "--channel-ports", type=int, nargs=4, metavar=("TXP", "TXN", "RXP", "RXN"),
        default=(1, 2, 3, 4),
    )
    parser.add_argument("--fidelity", choices=("training", "candidate"), default="training")
    parser.add_argument(
        "--sampling-policy", choices=SAMPLING_POLICIES, default="uniform",
        help="deterministic full-space or Stage-1 constraint-aware random sampling",
    )
    parser.add_argument("--model-library", type=Path,
                        help="explicit SKY130 sky130.lib.spice path")
    parser.add_argument("--ngspice-executable", type=Path,
                        help="explicit ngspice/ngspice_con executable path")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="timeout in seconds for each ngspice invocation")
    parser.add_argument("--solver", choices=("auto", "klu", "sparse"), default="auto")
    parser.add_argument("--cache", type=Path, default=Path(".nebula-cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="validate and summarize an existing checkpoint without simulation")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate and fingerprint model/tool/channel inputs without simulation")
    parser.add_argument("--verify-top", type=int,
                        help="re-simulate this many top checkpoint candidates with cache disabled")
    parser.add_argument("--metric-rtol", type=float, default=1e-6)
    parser.add_argument("--metric-atol", type=float, default=1e-9)
    parser.add_argument("--rl-version", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args()
    selected_modes = sum((bool(args.summary_only), bool(args.preflight_only), args.verify_top is not None))
    if selected_modes > 1:
        parser.error("--summary-only, --preflight-only, and --verify-top are mutually exclusive")
    if args.summary_only:
        summary = summarize_receiver_search(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["complete"] else 2
    evaluator_kwargs = {
        "sky130": Sky130Config(args.model_library),
        "ngspice": NgSpiceConfig(
            executable=args.ngspice_executable,
            timeout_s=args.timeout,
            solver=args.solver,
        ),
    }
    if args.preflight_only:
        try:
            report = preflight_receiver_search(
                channel_path=args.channel,
                channel_port_map=ChannelPortMap(*args.channel_ports),
                sky130=evaluator_kwargs["sky130"],
                ngspice=evaluator_kwargs["ngspice"],
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            print(json.dumps({
                "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
                "ready_for_search": False,
                "error": str(exc),
            }, indent=2, sort_keys=True))
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.verify_top is not None:
        try:
            report = verify_receiver_search(
                args.output, top=args.verify_top,
                metric_rtol=args.metric_rtol, metric_atol=args.metric_atol,
                evaluator_kwargs=evaluator_kwargs,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            print(json.dumps({
                "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
                "passed": False, "error": str(exc),
            }, indent=2, sort_keys=True))
            return 2
        verification_path = args.output.with_suffix(args.output.suffix + ".verification.json")
        _atomic_json(verification_path, report)
        print(json.dumps({"verification": str(verification_path), **report}, indent=2, sort_keys=True))
        return 0 if report["passed"] else 3
    rows = run_receiver_search(
        version=args.rl_version,
        count=args.count, seed=args.seed, output=args.output, channel_path=args.channel,
        channel_port_map=ChannelPortMap(*args.channel_ports),
        fidelity=EvaluationFidelity[args.fidelity.upper()],
        sampling_policy=args.sampling_policy,
        cache_path=None if args.no_cache else args.cache,
        evaluator_kwargs=evaluator_kwargs,
    )
    summary = summarize_receiver_search(args.output)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "new_evaluations": len(rows), "output": str(args.output),
        "summary": str(summary_path), **summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
