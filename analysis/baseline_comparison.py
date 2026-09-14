"""Comparison/analysis framework for the Random Search and CEM baselines.

This module loads *existing* results/*.jsonl files written by
experiments/receiver_search.py (random search) and experiments/train_cem.py
(Cross-Entropy Method). It performs no simulation of any kind -- it is pure
post-hoc analysis over rows already on disk.

Two on-disk row schemas exist for full-receiver (not CTLE-only) searches:

  RECEIVER_SEARCH_SCHEMA -- experiments/receiver_search.py rows. Has
    "candidate_index", full "parameters" (rload_ohm/rdeg_ohm/cdeg_f/itail_a/
    dfe_tap_v), "metrics" (the complete measured metric set), "evaluation_id",
    "reward", "success", "failed_stage". Richest schema; only file currently
    using it with a documented seed is
    results/receiver_random_search_20_seed123.jsonl.

  CEM_SCHEMA -- experiments/train_cem.py rows. Has "iteration", "evaluation",
    "reward", "action", "success", "failure_stage". Older rows do NOT carry
    "parameters", "metrics", or "wall_clock_s" -- those three fields were
    added to train_cem.py in this session (additive, no behavior change) so
    the *next* CEM run produces them. For rows written before that change,
    "parameters" is reconstructed here from the logged normalized "action"
    via the existing, unmodified
    simulator.rl_adapter.normalized_action_to_parameters (pure arithmetic,
    not a simulation); "metrics" and "wall_clock_s" cannot be recovered
    after the fact and are reported as unavailable rather than fabricated.

A third schema (uppercase RLOAD/RDEG/CDEG/ITAIL_VAL keys, gain_*_db metrics)
belongs to a pre-receiver-integration, CTLE-block-only sweep
(experiments/sweep.py-era). It is a different circuit scope (CTLE alone, not
the integrated CTLE+channel+sampler+DFE receiver) and is detected and
excluded from receiver-level comparison, not silently merged in.

Nothing here invents a metric that is not present in the source rows. Where
a requested comparison axis (e.g. wall-clock time for pre-instrumentation
runs) has no data, the summary says so explicitly via `caveats` /
`wall_clock_available=False` rather than omitting the field or guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional

from simulator.rl_adapter import normalized_action_to_parameters
from dataclasses import asdict as _dc_asdict
from analysis.artifact_store import read_packaged_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECEIVER_SEARCH_SCHEMA = "receiver_search_v2"
CEM_SCHEMA = "cem_v1"
CTLE_LEGACY_SCHEMA = "ctle_legacy_block_only"
RAW_EVALUATION_SCHEMA = "raw_receiver_evaluation_dump"
UNKNOWN_SCHEMA = "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        packaged = read_packaged_bytes(PROJECT_ROOT, relative)
        if packaged is None:
            raise FileNotFoundError(path)
        text = packaged.decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def detect_schema(row: dict[str, Any]) -> str:
    """Identify which on-disk row schema `row` belongs to, from its keys alone."""

    if "RLOAD" in row and "CDEG" in row:
        return CTLE_LEGACY_SCHEMA
    if (
        "candidate_index" in row
        and isinstance(row.get("parameters"), dict)
        and "rload_ohm" in row["parameters"]
        and "reward" in row
    ):
        return RECEIVER_SEARCH_SCHEMA
    if "iteration" in row and "evaluation" in row and "action" in row and "reward" in row:
        return CEM_SCHEMA
    if "stages" in row and "parameters" in row and "action" not in row and "reward" not in row:
        return RAW_EVALUATION_SCHEMA
    return UNKNOWN_SCHEMA


@dataclass(frozen=True)
class EvaluationRecord:
    """One SPICE evaluation, normalized across schemas."""

    source_file: str
    method: str  # "random_search" | "cem"
    index: int  # 0-based position within the file
    reward: float
    success: bool
    failed_stage: Optional[str]
    parameters: Optional[dict[str, float]]
    parameters_reconstructed: bool  # True if recovered from `action`, not logged directly
    wall_clock_s: Optional[float]
    evaluation_id: Optional[str]
    metrics: Optional[dict[str, float]] = None  # raw per-metric SPICE values, when the source row logged them


@dataclass(frozen=True)
class LoadedRun:
    path: str
    schema: str
    records: tuple[EvaluationRecord, ...]
    skipped_rows: int  # rows present in the file but not the detected schema (e.g. blank/mixed)


def load_run(path: str | Path) -> LoadedRun:
    """Load and normalize one results/*.jsonl file. Read-only; no simulation."""

    resolved = Path(path)
    rows = _read_jsonl(resolved)
    if not rows:
        return LoadedRun(str(resolved), UNKNOWN_SCHEMA, (), 0)

    schema = detect_schema(rows[0])
    records: list[EvaluationRecord] = []
    skipped = 0

    for index, row in enumerate(rows):
        row_schema = detect_schema(row)
        if row_schema != schema:
            skipped += 1
            continue

        if schema == RECEIVER_SEARCH_SCHEMA:
            records.append(EvaluationRecord(
                source_file=str(resolved), method="random_search", index=index,
                reward=float(row["reward"]), success=bool(row["success"]),
                failed_stage=row.get("failed_stage"),
                parameters=row.get("parameters"),
                parameters_reconstructed=False,
                wall_clock_s=row.get("wall_clock_s"),
                evaluation_id=row.get("evaluation_id"),
                metrics=row.get("metrics"),
            ))
        elif schema == CEM_SCHEMA:
            parameters = row.get("parameters")
            reconstructed = False
            if parameters is None and row.get("action") is not None:
                try:
                    parameters = _dc_asdict(normalized_action_to_parameters(row["action"]))
                    reconstructed = True
                except (TypeError, ValueError):
                    parameters = None
            records.append(EvaluationRecord(
                source_file=str(resolved), method="cem", index=index,
                reward=float(row["reward"]), success=bool(row["success"]),
                failed_stage=row.get("failure_stage"),
                parameters=parameters,
                parameters_reconstructed=reconstructed,
                wall_clock_s=row.get("wall_clock_s"),
                evaluation_id=row.get("evaluation_id"),
                metrics=row.get("metrics"),
            ))
        else:
            # CTLE-legacy and raw-evaluation-dump rows are recognized but
            # deliberately not turned into EvaluationRecords: they are a
            # different circuit scope / schema, not a receiver-level
            # random-search or CEM trial.
            skipped += 1

    return LoadedRun(str(resolved), schema, tuple(records), skipped)


@dataclass(frozen=True)
class RunSummary:
    source_file: str
    method: str
    schema: str
    n_evaluations: int
    n_successes: int
    success_rate: Optional[float]
    best_reward: Optional[float]
    best_index: Optional[int]
    best_parameters: Optional[dict[str, float]]
    best_parameters_reconstructed: Optional[bool]
    evaluations_to_first_success: Optional[int]
    running_best_reward: tuple[float, ...]
    reward_mean: Optional[float]
    reward_stdev: Optional[float]
    failure_stage_counts: dict[str, int]
    wall_clock_available: bool
    wall_clock_total_s: Optional[float]
    wall_clock_mean_s: Optional[float]
    caveats: tuple[str, ...] = field(default_factory=tuple)


def summarize(loaded: LoadedRun, method: str, extra_caveats: tuple[str, ...] = ()) -> RunSummary:
    """Compute the comparison metrics for one loaded run.

    Every number here is derived only from fields already present in the
    source rows. `wall_clock_available` is False (and the two wall-clock
    fields are None) whenever any row in the run lacks `wall_clock_s` --
    true for every run recorded before this session's train_cem.py /
    receiver_search.py timing instrumentation was added.
    """

    caveats = list(extra_caveats)
    records = loaded.records
    n = len(records)
    if loaded.skipped_rows:
        caveats.append(
            f"{loaded.skipped_rows} row(s) in {loaded.path} did not match the "
            f"detected schema ({loaded.schema}) and were excluded"
        )
    if n == 0:
        return RunSummary(
            source_file=loaded.path, method=method, schema=loaded.schema,
            n_evaluations=0, n_successes=0, success_rate=None,
            best_reward=None, best_index=None, best_parameters=None,
            best_parameters_reconstructed=None, evaluations_to_first_success=None,
            running_best_reward=(), reward_mean=None, reward_stdev=None,
            failure_stage_counts={}, wall_clock_available=False,
            wall_clock_total_s=None, wall_clock_mean_s=None,
            caveats=tuple(caveats + ["no usable records in this file"]),
        )

    successes = [r for r in records if r.success]
    rewards = [r.reward for r in records]

    running_best: list[float] = []
    best_so_far = -math.inf
    first_success_index: Optional[int] = None
    for position, record in enumerate(records):
        best_so_far = max(best_so_far, record.reward)
        running_best.append(best_so_far)
        if record.success and first_success_index is None:
            first_success_index = position + 1  # 1-indexed: "N evaluations to first success"

    best_record = max(records, key=lambda r: r.reward)
    best_index = records.index(best_record)

    failure_stage_counts: dict[str, int] = {}
    for record in records:
        if not record.success:
            key = record.failed_stage or "unspecified"
            failure_stage_counts[key] = failure_stage_counts.get(key, 0) + 1

    wall_clocks = [r.wall_clock_s for r in records if r.wall_clock_s is not None]
    wall_clock_available = len(wall_clocks) == n and n > 0
    if wall_clocks and not wall_clock_available:
        caveats.append(
            f"wall_clock_s present for only {len(wall_clocks)}/{n} rows "
            "(partial instrumentation coverage) -- not used for aggregate timing"
        )

    reward_mean = sum(rewards) / n
    reward_stdev = (
        math.sqrt(sum((r - reward_mean) ** 2 for r in rewards) / n) if n > 1 else 0.0
    )

    return RunSummary(
        source_file=loaded.path, method=method, schema=loaded.schema,
        n_evaluations=n, n_successes=len(successes),
        success_rate=len(successes) / n,
        best_reward=best_record.reward, best_index=best_index,
        best_parameters=best_record.parameters,
        best_parameters_reconstructed=best_record.parameters_reconstructed,
        evaluations_to_first_success=first_success_index,
        running_best_reward=tuple(running_best),
        reward_mean=reward_mean, reward_stdev=reward_stdev,
        failure_stage_counts=failure_stage_counts,
        wall_clock_available=wall_clock_available,
        wall_clock_total_s=sum(wall_clocks) if wall_clock_available else None,
        wall_clock_mean_s=(sum(wall_clocks) / len(wall_clocks)) if wall_clock_available else None,
        caveats=tuple(caveats),
    )


def summary_to_dict(summary: RunSummary) -> dict[str, Any]:
    payload = {
        "source_file": summary.source_file,
        "method": summary.method,
        "schema": summary.schema,
        "n_evaluations": summary.n_evaluations,
        "n_successes": summary.n_successes,
        "success_rate": summary.success_rate,
        "best_reward": summary.best_reward,
        "best_index": summary.best_index,
        "best_parameters": summary.best_parameters,
        "best_parameters_reconstructed_from_action": summary.best_parameters_reconstructed,
        "evaluations_to_first_success": summary.evaluations_to_first_success,
        "reward_mean": summary.reward_mean,
        "reward_stdev": summary.reward_stdev,
        "failure_stage_counts": summary.failure_stage_counts,
        "wall_clock_available": summary.wall_clock_available,
        "wall_clock_total_s": summary.wall_clock_total_s,
        "wall_clock_mean_s": summary.wall_clock_mean_s,
        "caveats": list(summary.caveats),
    }
    return payload


# ---------------------------------------------------------------------------
# Known result files. This registry is the only place file-specific
# provenance notes live; the loader/summarizer above are generic.
# ---------------------------------------------------------------------------

RANDOM_SEARCH_RUNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "results/receiver_random_search_20_seed123.jsonl",
        (
            "documented seed=123, uniform sampling policy, n=20, full-receiver "
            "schema with per-candidate metrics; the only random-search run with "
            "a manifest.json and summary.json on disk",
        ),
    ),
)

_CEM_INIT_CAVEAT = (
    "experiments/train_cem.py seeds its initial sampling distribution at a fixed "
    "mean=[0.3697, 0.2767, 0.3333, 0.7802, -0.0277] (normalized action space) "
    "with std=0.15 per dimension -- a Gaussian localized in a specific, "
    "already-decent region, not the uniform(-1, 1)^5 coverage "
    "receiver_search.py's 'uniform' sampling_policy uses. Some or all of CEM's "
    "higher raw success rate below reflects this warm-start neighborhood "
    "advantage, not only CEM's population/elite refinement mechanism -- the two "
    "runs are not a controlled comparison of search *algorithm* alone."
)

CEM_RUNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "results/cem_baseline_3x10.jsonl",
        (
            "configured for 3 iterations x population 10 = 30 planned evaluations; "
            "file contains 13 rows (iterations 1-2 plus one row of iteration 3) -- "
            "this run appears interrupted/incomplete, not a completed 30-eval trial",
            _CEM_INIT_CAVEAT,
        ),
    ),
    (
        "results/cem_near_good_v2.jsonl",
        (
            "rows 1-5 are byte-identical to the first 5 rows of "
            "results/cem_baseline_3x10.jsonl (same seed/mean/std init) -- an "
            "earlier checkpoint of the same run, not an independent trial; kept "
            "for completeness but should not be double-counted alongside "
            "cem_baseline_3x10.jsonl in an aggregate",
        ),
    ),
    (
        "results/cem_near_good.jsonl",
        ("byte-identical to results/cem_smoke.jsonl -- same run logged twice",),
    ),
    (
        "results/cem_smoke.jsonl",
        ("byte-identical to results/cem_near_good.jsonl -- same run logged twice",),
    ),
)

EXCLUDED_OUT_OF_SCOPE: tuple[tuple[str, str], ...] = (
    (
        "results/random_search.jsonl",
        "CTLE-block-only sweep (uppercase RLOAD/RDEG/CDEG/ITAIL_VAL keys, "
        "gain_*_db metrics) -- pre-receiver-integration circuit scope, not "
        "comparable to full-receiver random search",
    ),
    (
        "results/ctle_search_20.jsonl",
        "CTLE-block-only sweep, same schema/scope note as random_search.jsonl",
    ),
    (
        "results/test_search.jsonl",
        "CTLE-block-only sweep fixture (identical schema to ctle_search_20.jsonl)",
    ),
    (
        "results/receiver_search.jsonl",
        "1-row raw-ReceiverEvaluation-dump fixture (no candidate_index/action/"
        "reward fields) -- superseded by experiments/receiver_search.py's "
        "current row schema",
    ),
    (
        "results/receiver_search_1.jsonl",
        "1-row raw-ReceiverEvaluation-dump fixture, same note as receiver_search.jsonl",
    ),
    (
        "results/receiver_search_3.jsonl",
        "3-row raw-ReceiverEvaluation-dump fixture, no documented seed/manifest",
    ),
    (
        "results/receiver_search_10.jsonl",
        "10-row raw-ReceiverEvaluation-dump fixture, no documented seed/manifest",
    ),
)


def build_report(repository_root: str | Path = ".") -> dict[str, Any]:
    """Load every known random-search/CEM result file and summarize it.

    Returns a JSON-serializable dict: per-file summaries for both methods,
    the excluded-file registry (with reasons), and a headline comparison
    built only from the single documented-seed random-search run and the
    (incomplete) primary CEM run -- the only two runs specific enough to
    compare apples-to-apples.
    """

    root = Path(repository_root)
    random_search_summaries = []
    for relative_path, notes in RANDOM_SEARCH_RUNS:
        loaded = load_run(root / relative_path)
        random_search_summaries.append(summary_to_dict(summarize(loaded, "random_search", notes)))

    cem_summaries = []
    for relative_path, notes in CEM_RUNS:
        loaded = load_run(root / relative_path)
        cem_summaries.append(summary_to_dict(summarize(loaded, "cem", notes)))

    headline = None
    if random_search_summaries and cem_summaries:
        rs = random_search_summaries[0]
        cem = cem_summaries[0]
        headline = {
            "random_search_file": rs["source_file"],
            "cem_file": cem["source_file"],
            "spice_evaluation_count": {"random_search": rs["n_evaluations"], "cem": cem["n_evaluations"]},
            "success_rate": {"random_search": rs["success_rate"], "cem": cem["success_rate"]},
            "evaluations_to_first_success": {
                "random_search": rs["evaluations_to_first_success"],
                "cem": cem["evaluations_to_first_success"],
            },
            "best_reward": {"random_search": rs["best_reward"], "cem": cem["best_reward"]},
            "wall_clock_available": {
                "random_search": rs["wall_clock_available"],
                "cem": cem["wall_clock_available"],
            },
            "sample_size_caveat": (
                f"random_search n={rs['n_evaluations']} vs cem n={cem['n_evaluations']} "
                "(cem run is incomplete, see its caveats) -- not a like-for-like "
                "sample size; treat any rate comparison as directional only"
            ),
            "initialization_asymmetry_caveat": _CEM_INIT_CAVEAT,
        }

    return {
        "random_search_runs": random_search_summaries,
        "cem_runs": cem_summaries,
        "excluded_out_of_scope": [
            {"file": file, "reason": reason} for file, reason in EXCLUDED_OUT_OF_SCOPE
        ],
        "headline_comparison": headline,
        "not_yet_measured": [
            "wall-clock/design time for any run recorded before this session's "
            "timing instrumentation (train_cem.py, receiver_search.py) -- no "
            "existing row of receiver_random_search_20_seed123.jsonl or "
            "cem_baseline_3x10.jsonl carries a timestamp or duration",
            "a completed (30-evaluation) CEM run at the same n as the seed=123 "
            "random search, for a like-for-like success-rate comparison",
            "PPO/AutoCkt results in this same normalized comparison framework "
            "-- deferred until the running real-SPICE PPO confirmation "
            "experiment finishes and its result file is finalized",
            "per-metric (eye height / eye width / margin / power) design-quality "
            "comparison for CEM's successful candidates -- CEM's historical rows "
            "never logged simulator metrics, only reward_v1's scalar reward",
            "a CEM run initialized with the same uniform(-1, 1)^5 coverage random "
            "search uses (rather than train_cem.py's fixed near-good Gaussian "
            "mean) -- needed to isolate the search-algorithm effect from the "
            "initialization effect, see initialization_asymmetry_caveat above",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Random Search vs. CEM baseline comparison")
    lines.append("")
    lines.append(
        "Generated from existing results/*.jsonl files only. No new SPICE "
        "evaluation was run to produce this report."
    )
    lines.append("")

    lines.append("## Random search runs")
    lines.append("")
    lines.append("| file | n | successes | success rate | best reward | evals to first success | wall-clock |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for summary in report["random_search_runs"]:
        lines.append(
            f"| {summary['source_file']} | {summary['n_evaluations']} | "
            f"{summary['n_successes']} | "
            f"{_fmt_pct(summary['success_rate'])} | "
            f"{_fmt_num(summary['best_reward'])} | "
            f"{_fmt_num(summary['evaluations_to_first_success'])} | "
            f"{'measured (mean %.2fs)' % summary['wall_clock_mean_s'] if summary['wall_clock_available'] else 'not recorded'} |"
        )
    lines.append("")

    lines.append("## CEM runs")
    lines.append("")
    lines.append("| file | n | successes | success rate | best reward | evals to first success | wall-clock |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for summary in report["cem_runs"]:
        lines.append(
            f"| {summary['source_file']} | {summary['n_evaluations']} | "
            f"{summary['n_successes']} | "
            f"{_fmt_pct(summary['success_rate'])} | "
            f"{_fmt_num(summary['best_reward'])} | "
            f"{_fmt_num(summary['evaluations_to_first_success'])} | "
            f"{'measured (mean %.2fs)' % summary['wall_clock_mean_s'] if summary['wall_clock_available'] else 'not recorded'} |"
        )
    lines.append("")

    lines.append("## Caveats per run")
    lines.append("")
    for summary in report["random_search_runs"] + report["cem_runs"]:
        if summary["caveats"]:
            lines.append(f"- **{summary['source_file']}**:")
            for caveat in summary["caveats"]:
                lines.append(f"  - {caveat}")
    lines.append("")

    lines.append("## Excluded (out of scope)")
    lines.append("")
    for item in report["excluded_out_of_scope"]:
        lines.append(f"- `{item['file']}` -- {item['reason']}")
    lines.append("")

    lines.append("## Not yet measured")
    lines.append("")
    for item in report["not_yet_measured"]:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: Optional[float]) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _main() -> int:
    report = build_report()
    output_json = Path("results/baseline_comparison_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    output_markdown = Path("docs/baseline-comparison.md")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"json": str(output_json), "markdown": str(output_markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
