"""Minimal, competition-demo local UI around experiments/run_autockt_pipeline.py.

Runnable from a clean checkout with one command:

    python experiments/web_ui.py

then open the URL it prints (http://127.0.0.1:8000 by default). Use
--host/--port to change the bind address, e.g.
`python experiments/web_ui.py --port 8001`. The server binds to
127.0.0.1 by default and is not exposed externally unless --host is
explicitly overridden.

DESIGN: this server never imports or calls PPO/simulator/reward/PVT/
candidate-selection code itself. Every "Run NEBULA" click launches the
existing, unmodified `python -m experiments.run_autockt_pipeline ...` CLI
as a SEPARATE SUBPROCESS (the exact same entry point used from the command
line and in the real-SPICE smoke test) and only reads back its own JSON
`--output` file -- the pipeline stays the single source of truth, nothing
here duplicates optimization/selection logic.

Subprocess isolation is also a deliberate safety choice, not just
convenience: a real-SPICE run can (rarely, per docs/autockt-mapping.md
sec 23's smoke-test investigation) crash the Python process with a native
SIGSEGV. Because the pipeline runs in a child process, that kind of crash
only ends the run being demonstrated -- it cannot take down this UI
server, and is reported to the judge as a clear, specific status rather
than the whole demo dying silently.

Each run writes to a NEW file under results/web_ui_runs/<run_id>.json
(directory created on first use) -- historical results/*.json files are
never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

# When this file is launched directly (``python experiments/web_ui.py``),
# Python places ``experiments/`` rather than the repository root on
# sys.path. Add the root before importing sibling top-level packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.v3_dashboard import build_dashboard
from analysis.artifact_store import packaged_file_exists
from analysis.plot_renderer import render_chart
from analysis.run_evidence import build_run_dashboard

RUNS_DIR = PROJECT_ROOT / "results" / "web_ui_runs"
HOST = "127.0.0.1"  # localhost-only default -- pass --host to expose beyond this machine
DEFAULT_PORT = 8000
TAIL_CHARS = 4000  # stdout/stderr tail kept per run, to bound memory/response size
# FINAL AUDIT gap E: bounded outer-process handling for a hung (not
# crashed) pipeline subprocess. A SIGSEGV already terminates the
# subprocess immediately and is handled below without any timeout; this
# guards the separate risk of a subprocess that never returns at all
# (e.g. a wedged ngspice process). Generous enough not to interrupt a
# legitimate --pvt-condition-set minimal27 run (~121.8 min historically
# for one design's 27-point sweep, docs/autockt-mapping.md sec 22, times
# however many nominally-feasible candidates reach that stage) while
# still catching a genuine hang rather than waiting forever.
RUN_TIMEOUT_S = 7200.0  # legacy/default bound for non-full-PVT work
MINIMAL27_FIXED_OVERHEAD_S = 1800.0
MINIMAL27_PER_CANDIDATE_S = 3.0 * 3600.0

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()

TARGET_MODES = ("trivial", "hard", "custom")
BACKENDS = ("synthetic", "real")
RL_VERSIONS = ("v1", "v2", "v3")
CHANNELS = {
    "synthetic": ("channels/synthetic_regression.s4p", (1, 2, 3, 4)),
    "ieee802_reference": ("channels/ieee802_ibm_20db_thru.s4p", (1, 3, 2, 4)),
}
# Kept as a plain string tuple (not imported from experiments.run_autockt_
# pipeline) so this server never has to import torch/simulator/rl at
# startup -- see the module docstring's design note. Cross-checked against
# the pipeline's own PVT_CONDITION_SETS keys by
# tests/test_web_ui.py::PvtOptionsMatchPipelineTests so the two cannot
# silently drift apart.
PVT_CONDITION_SETS = ("none", "smoke", "minimal27", "full36")
TRADE_OFF_PREFERENCES = (
    "most_robust", "lowest_power", "strongest_eye_height", "widest_eye", "largest_margin", "balanced", "lowest_partial_mos_area", "lowest_noise",
)
TARGET_SPEC_FIELDS = (
    "dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v", "ctle_power_w",
)

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Run orchestration -- builds the exact CLI invocation and executes it as a
# subprocess. No pipeline logic lives here.
# ---------------------------------------------------------------------------

def _validate_request(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    import math
    for name in ("search_seconds", "max_evaluations"):
        value = payload.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or (name == "max_evaluations" and int(value) != value)):
            problems.append(f"{name} must be a finite positive {'integer' if name == 'max_evaluations' else 'number'}")
    if payload.get("workers", 1) not in (1, 2):
        problems.append("workers must be 1 or 2")
    if payload.get("workflow", "infer") not in ("infer", "train"):
        problems.append("workflow must be infer or train")
    if payload.get("workflow") == "train":
        if payload.get("target_mode") == "custom":
            problems.append("UI training currently uses trivial/hard targets; use CLI target pools for generalization")
        try:
            if not 1 <= int(payload.get("updates", 0)) <= 100:
                problems.append("training updates must be 1–100")
        except (TypeError, ValueError):
            problems.append("training updates must be an integer")
    if payload.get("target_mode") not in TARGET_MODES:
        problems.append(f"target_mode must be one of {TARGET_MODES}")
    if payload.get("target_mode") == "custom":
        target = payload.get("target") or {}
        for field in TARGET_SPEC_FIELDS:
            value = target.get(field)
            if not isinstance(value, (int, float)):
                problems.append(f"target.{field} must be a number")
    if payload.get("backend") not in BACKENDS:
        problems.append(f"backend must be one of {BACKENDS}")
    if payload.get("rl_version", "v1") not in RL_VERSIONS:
        problems.append(f"rl_version must be one of {RL_VERSIONS}")
    if payload.get("channel_id", "synthetic") not in CHANNELS:
        problems.append(f"channel_id must be one of {tuple(CHANNELS)}")
    try:
        episodes = int(payload.get("episodes", 0))
        if episodes < 1:
            problems.append("episodes must be >= 1")
    except (TypeError, ValueError):
        problems.append("episodes must be an integer")
    try:
        horizon = int(payload.get("horizon", 0))
        if horizon < 1:
            problems.append("horizon must be >= 1")
    except (TypeError, ValueError):
        problems.append("horizon must be an integer")
    if payload.get("backend") == "real" and not payload.get("checkpoint") and payload.get("workflow") != "train":
        problems.append("backend='real' requires a checkpoint (an untrained policy against real SPICE "
                         "has no learned behavior to demonstrate -- see docs/autockt-mapping.md sec 20)")
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = (PROJECT_ROOT / checkpoint).resolve()
        inside_project = PROJECT_ROOT in checkpoint_path.parents
        if not inside_project or not (
            checkpoint_path.is_file() or packaged_file_exists(PROJECT_ROOT, checkpoint)
        ):
            problems.append(f"checkpoint not found inside the project: {checkpoint}")
    if payload.get("pvt_condition_set") not in PVT_CONDITION_SETS:
        problems.append(f"pvt_condition_set must be one of {PVT_CONDITION_SETS}")
    if payload.get("trade_off_preference") not in TRADE_OFF_PREFERENCES:
        problems.append(f"trade_off_preference must be one of {TRADE_OFF_PREFERENCES}")
    try:
        timeout = payload.get("simulator_timeout_seconds", 360)
        if isinstance(timeout, bool) or not 1 <= float(timeout) <= 3600:
            raise ValueError
    except (TypeError, ValueError):
        problems.append("simulator_timeout_seconds must be between 1 and 3600")
    if payload.get("pvt_pattern_bits", 512) not in (512, 1024):
        problems.append("pvt_pattern_bits must be 512 or 1024")
    return problems


def _build_argv(payload: dict[str, Any], *, output_path: Path, schematic_path: Path) -> list[str]:
    channel_path, channel_ports = CHANNELS[payload.get("channel_id", "synthetic")]
    runtime_args = []
    if payload.get("search_seconds") is not None:
        runtime_args += ["--search-seconds", str(payload["search_seconds"])]
    if payload.get("workflow") == "train":
        version = payload.get("rl_version", "v3")
        argv = [sys.executable, "-m", "experiments.train_autockt", "--backend", payload["backend"],
            "--rl-version", version, "--target-mode", payload["target_mode"],
            "--updates", str(int(payload["updates"])), "--episodes-per-update", str(int(payload["episodes"])),
            "--horizon", str(int(payload["horizon"])), "--max-evaluations", str(int(payload["updates"]) * int(payload["episodes"]) * (int(payload["horizon"]) + 1)),
            "--initial-indices-source", payload.get("initial_indices_source", "verified"),
            "--channel", channel_path, "--channel-ports", *(str(p) for p in channel_ports),
            "--output", str(output_path.with_suffix(".training.jsonl")),
            "--events-output", str(output_path.with_suffix(".events.jsonl")),
            "--graph-output", str(output_path.with_suffix(".graph.json")), "--summary-output", str(output_path),
            "--save-final-policy", str(output_path.with_name(f"{output_path.stem}_{version}_policy.pt")),
            "--save-full-checkpoint", str(output_path.with_name(f"{output_path.stem}_{version}_full.pt"))]
        if payload.get("evaluation_cache"):
            argv.append("--evaluation-cache")
        if payload.get("max_evaluations") is not None:
            argv[argv.index("--max-evaluations") + 1] = str(payload["max_evaluations"])
        return argv + runtime_args + ["--workers", str(payload.get("workers", 1))]
    argv = [
        sys.executable, "-m", "experiments.run_autockt_pipeline",
        "--backend", payload["backend"],
        "--rl-version", payload.get("rl_version", "v1"),
        "--episodes", str(int(payload["episodes"])),
        "--horizon", str(int(payload["horizon"])),
        "--initial-indices-source", payload.get("initial_indices_source", "verified"),
        "--pvt-condition-set", payload["pvt_condition_set"],
        "--simulator-timeout-seconds", str(payload.get("simulator_timeout_seconds", 360)),
        "--pvt-pattern-bits", str(payload.get("pvt_pattern_bits", 512)),
        "--trade-off-preference", payload["trade_off_preference"],
        "--channel", channel_path,
        "--channel-ports", *(str(port) for port in channel_ports),
        "--output", str(output_path),
        "--export-schematic", str(schematic_path),
        "--eye-diagram-output", str(output_path.with_name(f"{output_path.stem}_eye.svg")),
    ]
    if payload["target_mode"] == "custom":
        argv += ["--target-json", json.dumps({f: float(payload["target"][f]) for f in TARGET_SPEC_FIELDS})]
    else:
        argv += ["--target-mode", payload["target_mode"]]
    if payload.get("checkpoint"):
        argv += ["--checkpoint", payload["checkpoint"]]
    if payload.get("measure_hd3_noise"):
        argv += ["--measure-hd3-noise"]
    if payload.get("evaluation_cache"):
        argv += ["--evaluation-cache"]
    return argv + runtime_args + ["--max-evaluations", str(payload.get("max_evaluations") or 1000)]


def _estimate_timeout_s(payload: dict[str, Any]) -> float:
    """Return a configuration-aware watchdog bound.

    Full PVT is run for every nominally-feasible episode candidate, not just
    one design. The old fixed two-hour bound was shorter than a measured
    27-corner sweep and could kill healthy work. Three hours per possible
    candidate plus setup overhead is deliberately conservative; this remains
    a hang guard, not a prediction of normal runtime.
    """

    if payload.get("pvt_condition_set") not in ("minimal27", "full36"):
        return RUN_TIMEOUT_S
    episodes = max(1, int(payload.get("episodes", 1)))
    factor = 36 / 27 if payload.get("pvt_condition_set") == "full36" else 1
    timeout_scale = max(1, float(payload.get("simulator_timeout_seconds", 360)) / 180)
    return MINIMAL27_FIXED_OVERHEAD_S + episodes * MINIMAL27_PER_CANDIDATE_S * factor * timeout_scale

# ---------------------------------------------------------------------------
# PVT progress parsing ("Add visible PVT progress to web UI"): the pipeline
# subprocess (experiments/run_autockt_pipeline.py::_pvt_progress_evaluator)
# prints one flush=True JSON line per PVT (design, condition) pair to its
# own stdout -- the SAME stream this server already captures, no new IPC.
# These two functions are pure (no I/O, no locking) so they are testable
# directly with plain strings/dicts, independent of subprocess plumbing.
# ---------------------------------------------------------------------------

_PROGRESS_EVENT_KEY = "nebula_progress_event"


def _parse_progress_line(line: str) -> Optional[dict[str, Any]]:
    """Best-effort parse of one stdout line as a pipeline progress event.
    Never raises: any blank, non-JSON, non-object, or unrecognized line is
    simply not a progress event (None) -- malformed or missing pipeline
    output can never break a run (requirement: fall back gracefully)."""
    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(event, dict) or _PROGRESS_EVENT_KEY not in event:
        return None
    return event


def _apply_progress_event(entry: dict[str, Any], event: dict[str, Any]) -> None:
    """Mutates one _RUNS entry in place from one parsed progress event.
    Caller must hold _RUNS_LOCK.
    """
    kind = event.get(_PROGRESS_EVENT_KEY)
    if kind == "candidate_generation_complete":
        total = event.get("pvt_conditions_total") or 0
        if total:
            entry["pvt_conditions_total"] = total
            entry["pvt_conditions_completed"] = 0
        # total == 0 means pvt_conditions_set == "none" (or zero feasible
        # candidates) -- leave every pvt_* field at its default (None), so
        # a "none" run never shows misleading PVT progress.
    elif kind == "pvt_condition_start":
        entry["pvt_current_corner"] = event.get("corner")
        entry["pvt_current_temperature_c"] = event.get("temperature_c")
        entry["pvt_current_supply_v"] = event.get("supply_v")
    elif kind == "pvt_condition_complete":
        # Only a definitively finished (successful or failed) condition
        # advances the count -- a "start" event above never does.
        if entry.get("pvt_conditions_completed") is not None:
            entry["pvt_conditions_completed"] += 1
        entry["pvt_current_corner"] = event.get("corner")
        entry["pvt_current_temperature_c"] = event.get("temperature_c")
        entry["pvt_current_supply_v"] = event.get("supply_v")
    # any other/unrecognized event kind is ignored, not an error.


def _consume_stdout_line(run_id: str, line: str) -> None:
    event = _parse_progress_line(line)
    if event is None:
        return
    with _RUNS_LOCK:
        entry = _RUNS.get(run_id)
        if entry is not None:
            _apply_progress_event(entry, event)


def _drain_subprocess_with_progress(
    proc: "subprocess.Popen[bytes]", run_id: str, deadline: float,
) -> tuple[str, str, bool]:
    """Drain both child pipes concurrently and apply stdout progress events.

    Reader threads work for Windows pipe handles as well as POSIX file
    descriptors, while ensuring neither child pipe can fill and deadlock.
    """
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def read_stream(stream, chunks: list[bytes], *, progress: bool) -> None:
        if stream is None:
            return
        for chunk in iter(stream.readline, b""):
            chunks.append(chunk)
            if progress:
                _consume_stdout_line(run_id, chunk.decode("utf-8", errors="replace"))

    readers = [
        threading.Thread(target=read_stream, args=(proc.stdout, stdout_chunks),
                         kwargs={"progress": True}, daemon=True),
        threading.Thread(target=read_stream, args=(proc.stderr, stderr_chunks),
                         kwargs={"progress": False}, daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    if not timed_out:
        for reader in readers:
            reader.join(timeout=2.0)

    return (
        b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        timed_out,
    )


def _execute_run(
    run_id: str, argv: list[str], output_path: Path, schematic_path: Path,
    timeout_s: Optional[float] = None,
) -> None:
    timeout_s = RUN_TIMEOUT_S if timeout_s is None else timeout_s
    with _RUNS_LOCK:
        _RUNS[run_id]["status"] = "running"
        _RUNS[run_id]["started_at"] = time.monotonic()

    timed_out = False
    try:
        import os
        proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        deadline = time.monotonic() + timeout_s
        stdout, stderr, timed_out = _drain_subprocess_with_progress(proc, run_id, deadline)
        if timed_out:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False)
            else:
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.wait()  # reader threads drain the closed pipes after termination
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        returncode = proc.returncode
        launch_error: Optional[str] = None
    except OSError as exc:
        stdout, stderr, returncode = "", "", None
        launch_error = f"failed to launch the pipeline subprocess: {exc}"

    finished_at = time.monotonic()

    with _RUNS_LOCK:
        entry = _RUNS[run_id]
        entry["finished_at"] = finished_at
        entry["returncode"] = returncode
        entry["stdout_tail"] = (stdout or "")[-TAIL_CHARS:]
        entry["stderr_tail"] = (stderr or "")[-TAIL_CHARS:]

        if launch_error is not None:
            entry["status"] = "failed"
            entry["error"] = launch_error
        elif timed_out:
            entry["status"] = "failed"
            entry["error"] = (
                f"the pipeline subprocess did not finish within {timeout_s / 60:.0f} minutes and was "
                "killed (not a crash -- a hang, e.g. a wedged ngspice process). This was NOT retried "
                "automatically. If this is unexpected for the configuration you ran, investigate before retrying."
            )
        elif returncode == 0 and output_path.is_file():
            try:
                entry["result"] = json.loads(output_path.read_text(encoding="utf-8"))
                entry["status"] = "completed"
            except (OSError, json.JSONDecodeError) as exc:
                entry["status"] = "failed"
                entry["error"] = f"pipeline exited cleanly but its output file could not be read: {exc}"
        elif returncode is not None and (returncode < 0 or returncode == 11):
            entry["status"] = "failed"
            entry["error"] = (
                f"the pipeline subprocess was terminated by signal {abs(returncode)} "
                f"(e.g. 11 = SIGSEGV) -- a native-level crash, not a Python error. "
                "This can occur intermittently during real-SPICE runs; see "
                "docs/autockt-mapping.md sec 23 for the investigation. See the "
                "stderr/stdout tails below for whatever was captured before the crash. "
                "Retrying the same run is a reasonable next step."
            )
        else:
            entry["status"] = "failed"
            entry["error"] = f"the pipeline exited with a non-zero status ({returncode}); see stderr below"


def _start_run(payload: dict[str, Any]) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output_path = RUNS_DIR / f"{run_id}.json"
    schematic_path = RUNS_DIR / f"{run_id}_schematic.spice"
    argv = _build_argv(payload, output_path=output_path, schematic_path=schematic_path)
    timeout_s = _estimate_timeout_s(payload)

    with _RUNS_LOCK:
        _RUNS[run_id] = {
            "run_id": run_id, "status": "queued", "command": argv,
            "output_path": str(output_path), "schematic_path": str(schematic_path),
            "started_at": None, "finished_at": None, "returncode": None,
            "timeout_s": timeout_s,
            "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            "pvt_condition_set": payload.get("pvt_condition_set"),
            "pvt_conditions_total": None, "pvt_conditions_completed": None,
            "pvt_current_corner": None, "pvt_current_temperature_c": None, "pvt_current_supply_v": None,
        }

    thread = threading.Thread(
        target=_execute_run, args=(run_id, argv, output_path, schematic_path, timeout_s), daemon=True,
    )
    thread.start()
    return run_id


def _status_payload(run_id: str) -> Optional[dict[str, Any]]:
    with _RUNS_LOCK:
        entry = _RUNS.get(run_id)
        if entry is None:
            return None
        entry = dict(entry)  # shallow copy for a consistent snapshot

    if entry["started_at"] is None:
        elapsed_s = 0.0
    elif entry["finished_at"] is not None:
        elapsed_s = entry["finished_at"] - entry["started_at"]
    else:
        elapsed_s = time.monotonic() - entry["started_at"]

    payload = {
        "run_id": entry["run_id"], "status": entry["status"], "elapsed_s": round(elapsed_s, 1),
        "command": " ".join(entry["command"]), "returncode": entry["returncode"], "error": entry["error"],
        "stdout_tail": entry["stdout_tail"], "stderr_tail": entry["stderr_tail"], "result": entry["result"],
        "timeout_s": entry.get("timeout_s", RUN_TIMEOUT_S),
    }
    if entry["result"] is not None and entry["result"].get("schematic_path"):
        payload["schematic_path"] = entry["result"]["schematic_path"]

    # PVT progress fields -- included only "when available" (requirement
    # 1): a "none" PVT run, or one where the pipeline hasn't yet reported
    # candidate_generation_complete, never sets pvt_conditions_total, so
    # none of this appears and the UI shows only the generic running state
    # (requirement 6/9).
    total = entry.get("pvt_conditions_total")
    if total:
        completed = entry.get("pvt_conditions_completed") or 0
        payload["pvt_conditions_total"] = total
        payload["pvt_conditions_completed"] = completed
        payload["pvt_progress_fraction"] = round(completed / total, 4)
        if entry.get("pvt_current_corner") is not None:
            payload["pvt_current_corner"] = entry["pvt_current_corner"]
            payload["pvt_current_temperature_c"] = entry["pvt_current_temperature_c"]
            payload["pvt_current_supply_v"] = entry["pvt_current_supply_v"]
    if entry.get("pvt_condition_set") in ("minimal27", "full36"):
        # Full 27-point sweep: no ETA (none of this is based on measured
        # data), just an explicit "this is long-running" label.
        payload["pvt_long_running_full_sweep"] = True

    return payload


def _available_checkpoints() -> list[str]:
    results_dir = PROJECT_ROOT / "results"
    checkpoints = set(
        str(p.relative_to(PROJECT_ROOT)) for p in results_dir.glob("*.pt")
    ) if results_dir.is_dir() else set()
    checkpoints.update(str(p.relative_to(PROJECT_ROOT)) for p in RUNS_DIR.glob("*_policy.pt")
                       if p.is_relative_to(PROJECT_ROOT))
    packaged_checkpoint = "results/autockt_mixed_target_confirmation_policy.pt"
    if packaged_file_exists(PROJECT_ROOT, packaged_checkpoint):
        checkpoints.add(packaged_checkpoint)
    return sorted(checkpoints)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NebulaUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        sys.stderr.write(f"[web_ui] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, INDEX_HTML)
        elif re.fullmatch(r"/api/run-graph/[0-9a-f]{32}\.(json|svg)", path):
            name, fmt = path.rsplit("/", 1)[1].rsplit(".", 1)
            graph_path = RUNS_DIR / f"{name}.graph.json"
            if not graph_path.is_file():
                self._send_json(404, {"error": "No recorded execution graph for this run"})
                return
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            if fmt == "json" and "limit" in parse_qs(urlparse(self.path).query):
                # UI overview stays bounded; the download without ?limit retains everything.
                graph["total_nodes"] = len(graph["nodes"])
                graph["nodes"] = graph["nodes"][:180]
                ids = {node["id"] for node in graph["nodes"]}
                graph["edges"] = [edge for edge in graph["edges"] if edge["source"] in ids and edge["target"] in ids]
            if fmt == "json":
                self._send_json(200, graph)
            else:
                from analysis.run_graph import render_svg
                self._send_bytes(200, render_svg(graph).encode("utf-8"), "image/svg+xml")
        elif path == "/api/checkpoints":
            self._send_json(200, {"checkpoints": _available_checkpoints()})
        elif path == "/api/evidence":
            self._send_json(200, build_dashboard(PROJECT_ROOT))
        elif path == "/api/evidence/runs":
            event_runs = {p.stem.removesuffix(".events") for p in RUNS_DIR.glob("*.events.jsonl")}
            result_runs = {p.stem for p in RUNS_DIR.glob("*.json") if _RUN_ID_RE.fullmatch(p.stem)}
            self._send_json(200, {"runs": sorted(event_runs | result_runs)})
        elif path.startswith("/api/result/"):
            run_id = path[len("/api/result/"):]
            if not _RUN_ID_RE.fullmatch(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            result_path = RUNS_DIR / f"{run_id}.json"
            if not result_path.is_file():
                self._send_json(404, {"error": "no saved result for this run"})
                return
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._send_json(500, {"error": f"could not read saved result: {exc}"})
                return
            elapsed_s = None
            graph_path = RUNS_DIR / f"{run_id}.graph.json"
            if graph_path.is_file():
                try:
                    elapsed_s = json.loads(graph_path.read_text(encoding="utf-8")).get("elapsed_s")
                except (OSError, json.JSONDecodeError):
                    pass
            self._send_json(200, {
                "run_id": run_id, "status": "completed", "elapsed_s": elapsed_s,
                "stdout_tail": "", "stderr_tail": "", "result": result,
            })
        elif path.startswith("/api/evidence/"):
            run_id = path[len("/api/evidence/"):]
            if not _RUN_ID_RE.match(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            evidence = build_run_dashboard(RUNS_DIR / f"{run_id}.events.jsonl")
            if not evidence["events"]:
                self._send_json(404, {"error": "no event evidence for this run"})
                return
            self._send_json(200, evidence)
        elif re.fullmatch(r"/api/plot/[0-9a-f]{32}/[a-z0-9_-]+\.(svg|png)", path):
            _, _, _, run_id, filename = path.split("/")
            plot_id, format = filename.rsplit(".", 1)
            evidence = build_run_dashboard(RUNS_DIR / f"{run_id}.events.jsonl")
            chart = next((item for item in evidence["charts"] if item.get("id") == plot_id), None)
            if chart is None:
                self._send_json(404, {"error": "unknown run plot"})
                return
            self._send_bytes(200, render_chart(chart, format=format),
                             "image/svg+xml" if format == "svg" else "image/png")
        elif path.startswith("/api/plot/"):
            name = path[len("/api/plot/"):]
            if not re.fullmatch(r"[a-z0-9_-]+\.(svg|png)", name):
                self._send_json(400, {"error": "invalid plot id"})
                return
            plot_id, format = name.rsplit(".", 1)
            dashboard = build_dashboard(PROJECT_ROOT)
            chart = next((item for item in dashboard["charts"] if item.get("id") == plot_id), None)
            if chart is None:
                self._send_json(404, {"error": "unknown plot"})
                return
            body = render_chart(chart, format=format)
            self._send_bytes(200, body, "image/svg+xml" if format == "svg" else "image/png")
        elif path.startswith("/api/status/"):
            run_id = path[len("/api/status/"):]
            if not _RUN_ID_RE.match(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            payload = _status_payload(run_id)
            if payload is None:
                self._send_json(404, {"error": "unknown run_id"})
                return
            self._send_json(200, payload)
        elif path.startswith("/api/schematic/"):
            run_id = path[len("/api/schematic/"):]
            if not _RUN_ID_RE.match(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            schematic_path = RUNS_DIR / f"{run_id}_schematic.spice"
            if not schematic_path.is_file():
                self._send_json(404, {"error": "no schematic for this run (not yet finished, or PVT/nominal "
                                                 "selection found no feasible design)"})
                return
            self._send_json(200, {"schematic": schematic_path.read_text(encoding="utf-8")})
        elif re.fullmatch(r"/api/eye/[0-9a-f]{32}\.(svg|png|json)", path):
            name, fmt = path.rsplit("/", 1)[1].rsplit(".", 1)
            eye_path = RUNS_DIR / f"{name}_eye.{fmt}"
            if not eye_path.is_file():
                self._send_json(404, {"error": "no measured eye diagram for this run"})
                return
            if fmt == "json":
                self._send_json(200, json.loads(eye_path.read_text(encoding="utf-8")))
            else:
                self._send_bytes(200, eye_path.read_bytes(),
                                 "image/svg+xml" if fmt == "svg" else "image/png")
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/run":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        problems = _validate_request(payload)
        if problems:
            self._send_json(400, {"error": "invalid request", "problems": problems})
            return

        run_id = _start_run(payload)
        self._send_json(202, {"run_id": run_id})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NEBULA -- Automated RF/Analog Receiver Sizing</title>
<style>
  :root {
    --bg: #0f1419; --panel: #161b22; --border: #2a323d; --text: #e6edf3; --muted: #8b96a3;
    --accent: #4fd1c5; --accent-dark: #2c9c92; --pass: #3fb950; --fail: #f85149; --notclaimed: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 20px; letter-spacing: 0.3px; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: 380px 1fr; gap: 0; min-height: calc(100vh - 74px); }
  .panel { padding: 20px 24px; }
  .left { border-right: 1px solid var(--border); overflow-y: auto; }
  fieldset { border: 1px solid var(--border); border-radius: 8px; margin: 0 0 16px; padding: 14px 16px; }
  legend { padding: 0 6px; color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
  label:first-child { margin-top: 0; }
  input, select { width: 100%; background: #0d1117; border: 1px solid var(--border); color: var(--text);
                  border-radius: 6px; padding: 7px 9px; font-size: 13px; }
  input:focus, select:focus { outline: 1px solid var(--accent); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  #customTargetFields { display: none; }
  button#runBtn { width: 100%; margin-top: 6px; padding: 11px; border: none; border-radius: 8px;
                  background: var(--accent); color: #06201d; font-weight: 700; font-size: 14px; cursor: pointer; }
  button#runBtn:disabled { background: #33403f; color: #7c8b89; cursor: not-allowed; }
  button#runBtn:not(:disabled):hover { background: var(--accent-dark); color: #eafffb; }
  .status-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .status-idle { background: #2a323d; color: var(--muted); }
  .status-running { background: #2c2a12; color: var(--notclaimed); }
  .status-completed { background: #0d2a17; color: var(--pass); }
  .status-failed { background: #2a1414; color: var(--fail); }
  #statusLine { display: flex; align-items: center; gap: 10px; margin: 16px 0; font-size: 13px; color: var(--muted); }
  #resultsEmpty { color: var(--muted); font-size: 14px; padding: 40px 0; text-align: center; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 14px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .verdict { padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
  .verdict-PASS { background: #0d2a17; color: var(--pass); }
  .verdict-FAIL { background: #2a1414; color: var(--fail); }
  .verdict-NOT-CLAIMED { background: #2a2210; color: var(--notclaimed); }
  .param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
  .param-grid div { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }
  .param-grid .k { color: var(--muted); font-size: 11px; }
  .param-grid .v { font-family: ui-monospace, monospace; font-size: 13px; margin-top: 2px; }
  pre.log { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px;
            font-size: 11.5px; max-height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .error-box { background: #2a1414; border: 1px solid #6e2323; color: #ffb4ac; border-radius: 8px;
               padding: 12px 14px; font-size: 13px; margin-bottom: 16px; }
  .links a { color: var(--accent); text-decoration: none; font-size: 13px; margin-right: 16px; }
  .links a:hover { text-decoration: underline; }
  .hint { color: var(--muted); font-size: 11.5px; margin-top: 4px; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
  .chart { background:#0d1117; border:1px solid var(--border); border-radius:8px; padding:10px; min-height:245px; }
  .chart h3 { font-size:12px; margin:0 0 6px; color:var(--text); }
  .chart img { width:100%; min-height:220px; background:white; border-radius:5px; }
  .chart-note { color:var(--muted); font-size:10.5px; margin-top:5px; }
  .axis-label { fill:var(--muted); font-size:9px; }
  .evidence-meta { color:var(--muted); font-size:11px; margin-top:10px; }
  code { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>NEBULA</h1>
  <p>Automated RL-driven PCIe Gen-2 receiver (CTLE+DFE) sizing -- real SPICE, end-to-end pipeline demo</p>
</header>
<main>
  <div class="panel left">
    <fieldset>
      <legend>Target specification</legend>
      <label for="targetMode">Preset</label>
      <select id="targetMode">
        <option value="trivial">Trivial (existing thresholds)</option>
        <option value="hard">Hard target</option>
        <option value="custom">Custom (enter values)</option>
      </select>
      <div id="customTargetFields">
        <label for="tEyeHeight">Min eye height (V)</label>
        <input id="tEyeHeight" type="number" step="any" value="0.1">
        <label for="tEyeWidth">Min eye width (UI, 0-1)</label>
        <input id="tEyeWidth" type="number" step="any" value="0.4">
        <label for="tMargin">Min DFE margin (V)</label>
        <input id="tMargin" type="number" step="any" value="0.0">
        <label for="tPower">Max CTLE power (W)</label>
        <input id="tPower" type="number" step="any" value="0.015">
      </div>
    </fieldset>

    <fieldset>
      <legend>Candidate generation</legend>
      <label for="backend">Backend</label>
      <select id="backend">
        <option value="synthetic">Synthetic (no SPICE -- fast dry run)</option>
        <option value="real">Real ngspice (slow, actual SPICE)</option>
      </select>
      <label for="channelId">Four-port channel</label>
      <select id="channelId">
        <option value="synthetic">Synthetic regression — software tests only</option>
        <option value="ieee802_reference">IEEE/IBM lossy THRU — public reference, −6.31 dB at 2.5 GHz</option>
      </select>
      <p class="hint">The IEEE/IBM model is a qualified, realistic reference path for training and comparison.
        It is not PCI-SIG compliance evidence. Port map 1/3 → 2/4 is applied automatically.</p>
      <label for="checkpoint">PPO checkpoint</label>
      <select id="checkpoint"><option value="">(untrained policy -- synthetic backend only)</option></select>
      <label for="rlVersion">PPO version</label>
      <select id="rlVersion">
        <option value="v1">PPO v1 — historical baseline</option>
        <option value="v2">PPO v2 — corrected reset + validity state + reward v2</option>
        <option value="v3">PPO v3 — eight-head MOS + passive/bias sizing</option>
      </select>
      <p class="hint" id="rlVersionDescription"></p>
      <label style="display:flex; align-items:center; gap:8px; margin-top:12px; cursor:pointer">
        <input id="evaluationCache" type="checkbox" style="width:auto">
        <span style="margin:0">Reuse exact repeated evaluations</span>
      </label>
      <div class="row">
        <div>
          <label for="episodes">Episodes</label>
          <input id="episodes" type="number" min="1" value="3">
        </div>
        <div>
          <label for="horizon">Horizon</label>
          <input id="horizon" type="number" min="1" value="4">
        </div>
      </div>
      <label for="searchSeconds">Search time budget (seconds; blank = unlimited)</label>
      <input id="searchSeconds" type="number" min="1" placeholder="Unlimited">
      <label for="simulatorTimeout">PVT / final SPICE timeout per invocation (seconds)</label>
      <input id="simulatorTimeout" type="number" min="1" max="3600" value="360">
      <label for="pvtPatternBits">PVT pattern length</label>
      <select id="pvtPatternBits"><option value="512">512 bits (default)</option><option value="1024">1024 bits (slower; FINAL fidelity)</option></select>
      <p class="hint">1024-bit PVT also includes three-amplitude HD3 at corners. A selected real design must pass nominal 1024-bit validation in either mode. Training screening is unchanged.</p>
      <label for="maxEvaluations">Evaluation budget (blank = workflow default)</label>
      <input id="maxEvaluations" type="number" min="1" placeholder="Workflow default">
      <label for="workers">Training workers (inference is serial)</label>
      <select id="workers"><option value="1">1</option><option value="2">2</option></select>
      <label for="initSource">Initial state</label>
      <select id="initSource">
        <option value="verified">Verified (warm start)</option>
        <option value="grid-center">Grid center (unbiased)</option>
      </select>
      <label style="display:flex; align-items:center; gap:8px; margin-top:12px; cursor:pointer">
        <input id="measureHd3Noise" type="checkbox" style="width:auto">
        <span style="margin:0">Measure HD3 &amp; input-referred noise (real backend only)</span>
      </label>
      <p class="hint">Runs one extra real-SPICE evaluation (FINAL fidelity) of the selected design.
        Off by default: adds real SPICE time; without it, HD3/noise show as NOT CLAIMED.</p>
    </fieldset>

    <fieldset>
      <legend>PVT-aware selection</legend>
      <label for="pvtSet">Condition set</label>
      <select id="pvtSet">
        <option value="none">None -- NOMINAL-ONLY (not PVT-robust; only TT/1.8V/27C is checked)</option>
        <option value="smoke">Smoke -- 2 conditions (nominal + 1 stress corner; not a robustness proof)</option>
        <option value="minimal27">27-point subset -- TT/SS/FF (SLOW; incomplete PVT coverage)</option>
        <option value="full36">Agreed 36-condition grid -- TT/SS/FF (expensive finalist validation)</option>
      </select>
      <p class="hint">None, smoke and minimal27 do NOT establish PVT robustness. Only a passing agreed 36-condition TT/SS/FF grid establishes coverage of this configured PVT grid.
        Smaller sweeps are screening results, not full PVT validation.</p>
      <label for="tradeOff">Trade-off preference (used on PVT ties)</label>
      <select id="tradeOff">
        <option value="most_robust">Most robust</option>
        <option value="lowest_power">Lowest power</option>
        <option value="strongest_eye_height">Strongest eye height</option>
        <option value="widest_eye">Widest eye</option>
        <option value="largest_margin">Largest margin</option>
        <option value="balanced">Balanced</option>
        <option value="lowest_partial_mos_area">Lowest partial MOS channel area (not total layout)</option>
        <option value="lowest_noise">Lowest measured noise (only when available)</option>
      </select>
      <p class="hint">"smoke" spends real SPICE only on nominally-feasible candidates; it is a two-condition screen.</p>
    </fieldset>

    <label for="workflow">Operation</label>
    <select id="workflow"><option value="infer">Generate and verify circuits</option><option value="train">Train a new PPO checkpoint</option></select>
    <label for="trainingUpdates">Training updates (training only)</label>
    <input id="trainingUpdates" type="number" value="2" min="1" max="100">
    <p class="hint">Training uses Episodes per update, Horizon, Backend, Channel and PPO version above. It creates new policy/full-checkpoint files; it does not overwrite the selected checkpoint. Real training spends SPICE time. PVT and final HD3/noise are run afterward using Generate and verify.</p>
    <button id="runBtn">Run NEBULA</button>
    <div id="statusLine"><span id="statusBadge" class="status-badge status-idle">idle</span><span id="elapsed"></span><span id="pvtProgress"></span></div>
  </div>

  <div class="panel">
    <div class="card" id="evidenceCard">
      <h2>Measured v3 performance evidence</h2>
      <div id="evidenceSummary" class="param-grid"></div>
      <div id="chartGrid" class="chart-grid" style="margin-top:14px"></div>
      <div id="evidenceMeta" class="evidence-meta">Loading v3 evidence…</div>
      <label for="runHistory">Recorded runs and results</label>
      <select id="runHistory"><option value="">Select a previous run</option></select>
    </div>
    <div id="resultsEmpty">Configure a target and click <strong>Run NEBULA</strong> to see results here.</div>
    <div id="errorBox" class="error-box" style="display:none"></div>
    <div id="results" style="display:none">
      <div class="card">
        <h2>Final circuit parameters</h2>
        <div id="paramGrid" class="param-grid"></div>
      </div>
      <div class="card">
        <h2>Measured specification</h2>
        <table><thead><tr><th>Metric</th><th>Measured</th><th>Requirement</th><th>Verdict</th></tr></thead>
        <tbody id="specRows"></tbody></table>
      </div>
      <div class="card" id="pvtCard" style="display:none">
        <h2>PVT robustness</h2>
        <div id="pvtSummary"></div>
        <div style="overflow-x:auto; margin-top:12px">
          <table><thead><tr><th>Corner</th><th>VDD</th><th>Temp.</th><th>Status</th><th>Peaking</th><th>Eye height</th><th>Eye width</th><th>DFE margin</th><th>Power</th><th>Noise</th><th>HD3</th></tr></thead>
          <tbody id="pvtRows"></tbody></table>
        </div>
      </div>
      <div class="card" id="eyeCard" style="display:none">
        <h2>Selected-design eye diagram</h2>
        <div id="eyeStatus" class="evidence-meta"></div>
        <img id="eyeImage" alt="Measured CTLE and behavioral-DFE eye diagram" style="max-width:100%; margin-top:10px">
        <div class="links" id="eyeLinks" style="margin-top:10px"></div>
      </div>
      <div class="card">
        <h2>Run summary</h2>
        <div id="runSummary" class="param-grid"></div>
        <div class="links" style="margin-top:12px">
          <a href="#" id="schematicLink">View schematic</a>
          <a href="#" id="jsonLink">View raw pipeline output (JSON)</a>
        </div>
        <pre class="log" id="schematicView" style="display:none; margin-top:10px"></pre>
      </div>
      <div class="card">
        <h2>Process log</h2>
        <label>stdout</label>
        <pre class="log" id="stdoutLog"></pre>
        <label>stderr</label>
        <pre class="log" id="stderrLog"></pre>
      </div>
    </div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let pollTimer = null;

const RL_VERSION_DESCRIPTIONS = {
  v1: 'Historical AutoCkt-compatible five-parameter baseline. Preserves the original reset, state and reward behavior for reproducibility.',
  v2: 'Improved five-parameter model. Measures the initial circuit state, encodes metric validity and failure stage, and uses the corrected dense reward. Existing v2 checkpoints remain compatible.',
  v3: 'Eight-head PPO: RLOAD, RDEG, CDEG, ITAIL, behavioral DFE tap, matched MOS width, length and integer multiplier. Measured reset, per-metric validity, channel features and strict peaking-aware reward. Requires a v3 policy; v1/v2 weights are incompatible. Reports partial MOS channel area, NOT total layout area. Synthetic runs are software tests only.'
};

function updateRlDescription() {
  $('rlVersionDescription').textContent = RL_VERSION_DESCRIPTIONS[$('rlVersion').value];
}
$('rlVersion').addEventListener('change', updateRlDescription);
updateRlDescription();

$('targetMode').addEventListener('change', () => {
  $('customTargetFields').style.display = $('targetMode').value === 'custom' ? 'block' : 'none';
});

fetch('/api/checkpoints').then(r => r.json()).then(data => {
  const sel = $('checkpoint');
  for (const path of data.checkpoints) {
    const opt = document.createElement('option');
    opt.value = path; opt.textContent = path;
    sel.appendChild(opt);
  }
  if (data.checkpoints.length) sel.value = data.checkpoints[0];
  const v3 = data.checkpoints.find(path => path.replaceAll('\\\\', '/').endsWith('/ppo_v3_tt_1000_policy.pt'));
  if (v3) { sel.value = v3; $('rlVersion').value = 'v3'; updateRlDescription(); }
});

function renderChart(chart) {
  const box = document.createElement('div'); box.className = 'chart';
  const h = document.createElement('h3'); h.textContent = chart.title; box.appendChild(h);
  const image = document.createElement('img');
  image.src = `/api/plot/${encodeURIComponent(chart.id)}.svg`;
  image.alt = chart.note || `${chart.title}; ${chart.x_label || ''} by ${chart.y_label || ''}`;
  image.loading = 'lazy'; box.appendChild(image);
  const links = document.createElement('div'); links.className = 'chart-note';
  links.innerHTML = `<a href="/api/plot/${encodeURIComponent(chart.id)}.svg" download>SVG</a> · `+
                    `<a href="/api/plot/${encodeURIComponent(chart.id)}.png" download>PNG</a>`;
  box.appendChild(links);
  return box;
}

function refreshEvidence() { return fetch('/api/evidence').then(r=>r.json()).then(data => {
  const s=data.summary;
  $('evidenceSummary').innerHTML =
    `<div><div class="k">PPO evaluations</div><div class="v">${s.ppo_evaluations}</div></div>`+
    `<div><div class="k">Logged target successes</div><div class="v">${s.ppo_logged_successes}</div></div>`+
    `<div><div class="k">TT/SS/FF qualification</div><div class="v">${s.pvt_total ? s.pvt_passes+'/'+s.pvt_total+' passed; '+s.pvt_total+'/36 measured' : 'Not validated'}</div></div>`+
    `<div><div class="k">Nominal measured points</div><div class="v">${s.feasible_designs}</div></div>`;
  $('chartGrid').replaceChildren();
  for (const chart of data.charts) $('chartGrid').appendChild(renderChart(chart));
  $('evidenceMeta').textContent = `${data.charts.length} v3 graphs from ${data.sources.length} sources. `+
    data.limitations.join(' ');
}).catch(err => { $('evidenceMeta').textContent='Evidence could not be loaded: '+err; }); }
refreshEvidence();
setInterval(refreshEvidence, 30000);

function setBadge(status) {
  const badge = $('statusBadge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status;
}

fetch('/api/evidence/runs').then(r => r.json()).then(data => {
  for (const id of data.runs) $('runHistory').add(new Option(id, id));
});
$('runHistory').addEventListener('change', () => {
  const runId = $('runHistory').value;
  if (!runId) return;
  fetch(`/api/result/${runId}`).then(async response => {
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not load saved run');
    setBadge(payload.status);
    $('elapsed').textContent = payload.elapsed_s == null ? '' : `${payload.elapsed_s.toFixed(1)}s recorded runtime`;
    $('pvtProgress').textContent = '';
    renderResult(payload);
    await renderRunEvidence(runId);
  }).catch(err => showError(String(err)));
});

function buildPayload() {
  const targetMode = $('targetMode').value;
  const payload = {
    workflow: $('workflow').value,
    updates: parseInt($('trainingUpdates').value, 10),
    target_mode: targetMode,
    backend: $('backend').value,
    channel_id: $('channelId').value,
    rl_version: $('rlVersion').value,
    evaluation_cache: $('evaluationCache').checked,
    checkpoint: $('checkpoint').value || null,
    episodes: parseInt($('episodes').value, 10),
    horizon: parseInt($('horizon').value, 10),
    search_seconds: $('searchSeconds').value ? Number($('searchSeconds').value) : null,
    simulator_timeout_seconds: Number($('simulatorTimeout').value),
    pvt_pattern_bits: Number($('pvtPatternBits').value),
    max_evaluations: $('maxEvaluations').value ? Number($('maxEvaluations').value) : null,
    workers: Number($('workers').value),
    initial_indices_source: $('initSource').value,
    pvt_condition_set: $('pvtSet').value,
    trade_off_preference: $('tradeOff').value,
    measure_hd3_noise: $('measureHd3Noise').checked,
  };
  if (targetMode === 'custom') {
    payload.target = {
      dfe_locked_phase_eye_height_v: parseFloat($('tEyeHeight').value),
      dfe_eye_width_ui: parseFloat($('tEyeWidth').value),
      dfe_min_margin_v: parseFloat($('tMargin').value),
      ctle_power_w: parseFloat($('tPower').value),
    };
  }
  return payload;
}

function showError(msg) {
  $('errorBox').style.display = 'block';
  $('errorBox').textContent = msg;
  $('results').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
}

function verdictClass(v) { return 'verdict verdict-' + v.replace(/ /g, '-'); }

async function renderRunEvidence(runId) {
  let panel = $('runEvidence');
  if (!panel) {
    panel = document.createElement('section'); panel.id = 'runEvidence'; panel.className = 'card';
    $('results').parentNode.appendChild(panel);
  }
  panel.replaceChildren();
  const title = document.createElement('h3'); title.textContent = 'This run: execution graph and measured progress'; panel.appendChild(title);
  const response = await fetch(`/api/run-graph/${runId}.json?limit=180`);
  if (!response.ok) { panel.appendChild(document.createTextNode('No recorded graph available for this historical run.')); return; }
  const graph = await response.json();
  const note = document.createElement('p'); note.textContent = `${graph.status}: ${graph.evaluation_requests ?? 'in progress'} evaluation requests, ${graph.cache_hits ?? 0} cache hits. Expand a node for actual parameters, metrics, timing and provenance. Synthetic values are not SPICE evidence.`; panel.appendChild(note);
  const links = document.createElement('p');
  for (const ext of ['json', 'svg']) { const a = document.createElement('a'); a.href = `/api/run-graph/${runId}.${ext}`; a.textContent = `Download ${ext.toUpperCase()}  `; a.download = ''; links.appendChild(a); } panel.appendChild(links);
  const image = document.createElement('img'); image.src = `/api/run-graph/${runId}.svg`; image.alt = 'Recorded execution graph'; image.style.maxWidth = '100%'; panel.appendChild(image);
  for (const node of graph.nodes.slice(0, 180)) {
    const details = document.createElement('details'); const summary = document.createElement('summary');
    summary.textContent = `${node.id}: ${node.label} — ${node.status}`;
    const pre = document.createElement('pre'); pre.style.whiteSpace = 'pre-wrap'; pre.textContent = JSON.stringify(node.detail, null, 2);
    details.append(summary, pre); panel.appendChild(details);
  }
  const evidence = await fetch(`/api/evidence/${runId}`);
  if (evidence.ok) for (const chart of (await evidence.json()).charts) {
    const img = document.createElement('img'); img.src = `/api/plot/${runId}/${chart.id}.svg`; img.alt = chart.title; img.style.maxWidth = '100%'; panel.appendChild(img);
  }
}

function renderResult(payload) {
  $('errorBox').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
  $('results').style.display = 'block';

  const result = payload.result;
  if (result.workflow === 'train') {
    $('eyeCard').style.display = 'none';
    $('paramGrid').textContent = `Training complete. Policy: ${result.policy_path}. Full checkpoint: ${result.full_checkpoint_path}. Training evaluations: ${result.total_evaluations}. This is not a final verified circuit.`;
    $('specRows').replaceChildren(); $('pvtCard').style.display = 'none';
    $('runSummary').textContent = `${result.rl_version} / ${result.backend} / ${payload.elapsed_s}s`;
    $('stdoutLog').textContent = payload.stdout_tail || ''; $('stderrLog').textContent = payload.stderr_tail || '';
    $('jsonLink').onclick = e => { e.preventDefault(); showJson(result); };
    $('schematicLink').onclick = e => { e.preventDefault(); showJson({note: 'Training does not export a verified schematic. Run inference with the saved policy.'}); };
    fetch('/api/checkpoints').then(r => r.json()).then(data => {
      const current = $('checkpoint').value;
      $('checkpoint').replaceChildren(new Option('(untrained policy — synthetic only)', ''));
      for (const path of data.checkpoints) $('checkpoint').add(new Option(path, path));
      $('checkpoint').value = current;
    });
    return;
  }
  const selected = result.selection && result.selection.selected;

  const paramGrid = $('paramGrid');
  paramGrid.innerHTML = '';
  if (selected) {
    for (const [k, v] of Object.entries(selected.parameters)) {
      const div = document.createElement('div');
      div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
      paramGrid.appendChild(div);
    }
  } else {
    paramGrid.innerHTML = '<div class="k">No feasible design was selected by this run.</div>';
  }

  const specRows = $('specRows');
  specRows.innerHTML = '';
  const rows = (result.final_specification && result.final_specification.rows) || [];
  for (const row of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.metric}</td><td>${row.measured ?? 'n/a'}</td><td>${row.requirement}</td>` +
                    `<td><span class="${verdictClass(row.verdict)}">${row.verdict}</span></td>`;
    specRows.appendChild(tr);
  }

  const pvt = selected && result.selection.pvt;
  const pvtRows = $('pvtRows');
  pvtRows.replaceChildren();
  if (pvt) {
    $('pvtCard').style.display = 'block';
    $('pvtSummary').innerHTML =
      `<div class="param-grid"><div><div class="k">Pass rate</div><div class="v">${pvt.n_passing}/${pvt.n_conditions}` +
      ` (${(pvt.pass_rate*100).toFixed(0)}%)</div></div>` +
      `<div><div class="k">Met minimum</div><div class="v">${pvt.met_minimum_pass_rate}</div></div></div>`;
    const number = (value, digits, scale=1, suffix='') =>
      value == null ? 'n/a' : `${(Number(value)*scale).toFixed(digits)}${suffix}`;
    for (const point of (pvt.points || [])) {
      const m = point.metrics || {};
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td>${String(point.process_corner || '').toUpperCase()}</td>` +
        `<td>${number(point.supply_v, 2, 1, ' V')}</td>` +
        `<td>${number(point.temperature_c, 0, 1, '°C')}</td>` +
        `<td><span class="${point.success ? 'verdict verdict-PASS' : 'verdict verdict-FAIL'}">${point.success ? 'PASS' : 'FAIL'}</span></td>` +
        `<td>${number(m.peaking_db, 2, 1, ' dB')}</td>` +
        `<td>${number(m.dfe_locked_phase_eye_height_v, 3, 1, ' V')}</td>` +
        `<td>${number(m.dfe_eye_width_ui, 2, 1, ' UI')}</td>` +
        `<td>${number(m.dfe_min_margin_v, 3, 1, ' V')}</td>` +
        `<td>${number(m.ctle_power_w, 3, 1000, ' mW')}</td>` +
        `<td>${number(m.input_referred_noise_vrms, 3, 1000, ' mV')}</td>` +
        `<td>${number(m.hd3_db, 2, 1, ' dB')}</td>`;
      pvtRows.appendChild(tr);
    }
  } else {
    $('pvtCard').style.display = 'none';
  }

  const eye = result.eye_diagram;
  $('eyeCard').style.display = 'block';
  $('eyeStatus').textContent = eye && eye.status === 'produced'
    ? 'Generated from one selected-design real-SPICE transient. The second panel is behavioral DFE sample data, not a transistor-level DFE waveform.'
    : ((eye && eye.reason) || 'Eye diagram was not requested or could not be measured.');
  $('eyeImage').style.display = eye && eye.status === 'produced' ? 'block' : 'none';
  $('eyeLinks').replaceChildren();
  if (eye && eye.status === 'produced') {
    $('eyeImage').src = `/api/eye/${payload.run_id}.svg`;
    for (const ext of ['svg', 'png', 'json']) {
      const link = document.createElement('a');
      link.href = `/api/eye/${payload.run_id}.${ext}`;
      link.download = ''; link.textContent = `Download ${ext.toUpperCase()} `;
      $('eyeLinks').appendChild(link);
    }
  }

  $('runSummary').innerHTML =
    `<div><div class="k">Candidates generated</div><div class="v">${result.n_candidates_generated}</div></div>` +
    `<div><div class="k">Nominally feasible</div><div class="v">${result.n_nominally_feasible}</div></div>` +
    `<div><div class="k">Backend</div><div class="v">${result.backend}</div></div>` +
    `<div><div class="k">Channel loss @ 2.5 GHz</div><div class="v">${result.channel ? result.channel.metrics.channel_loss_2p5ghz_db.toFixed(2)+' dB' : 'n/a'}</div></div>` +
    `<div><div class="k">PPO version</div><div class="v">${result.rl_version || 'v1'}</div></div>` +
    `<div><div class="k">Elapsed</div><div class="v">${payload.elapsed_s == null ? 'not recorded' : payload.elapsed_s.toFixed(1)+'s'}</div></div>`;

  $('schematicLink').onclick = (e) => {
    e.preventDefault();
    fetch(`/api/schematic/${payload.run_id}`).then(r => r.json()).then(d => {
      const view = $('schematicView');
      view.textContent = d.schematic || d.error;
      view.style.display = 'block';
    });
  };
  $('jsonLink').onclick = (e) => { e.preventDefault(); showJson(result); };

  $('stdoutLog').textContent = payload.stdout_tail || '(empty)';
  $('stderrLog').textContent = payload.stderr_tail || '(empty)';
}

function showJson(result) {
  const view = $('schematicView');
  view.textContent = JSON.stringify(result, null, 2);
  view.style.display = 'block';
}

function pvtProgressText(payload) {
  if (!payload.pvt_conditions_total) return '';
  let text = ` — PVT evaluation: ${payload.pvt_conditions_completed} / ${payload.pvt_conditions_total} conditions`;
  if (payload.pvt_current_corner) {
    text += ` (currently: ${payload.pvt_current_corner.toUpperCase()} / ` +
            `${payload.pvt_current_temperature_c}°C / ${payload.pvt_current_supply_v}V)`;
  }
  if (payload.pvt_long_running_full_sweep) {
    text += ' — requested PVT sweep, long-running';
  }
  return text;
}

function poll(runId) {
  fetch(`/api/status/${runId}`).then(r => r.json()).then(payload => {
    setBadge(payload.status);
    $('elapsed').textContent = payload.elapsed_s + 's elapsed';
    $('pvtProgress').textContent = pvtProgressText(payload);
    if (payload.status === 'running' || payload.status === 'queued') {
      pollTimer = setTimeout(() => poll(runId), 1200);
      return;
    }
    $('runBtn').disabled = false;
    renderRunEvidence(runId).catch(err => console.warn('Run evidence unavailable', err));
    if (payload.status === 'completed') {
      refreshEvidence();
      renderResult(payload);
    } else {
      showError((payload.error || 'run failed') + '\n\nstderr tail:\n' + (payload.stderr_tail || '(empty)'));
      $('stdoutLog') && ($('stdoutLog').textContent = payload.stdout_tail || '(empty)');
    }
  }).catch(err => {
    setBadge('failed');
    showError('Lost contact with the NEBULA UI server: ' + err);
    $('runBtn').disabled = false;
  });
}

$('runBtn').addEventListener('click', () => {
  if (pollTimer) clearTimeout(pollTimer);
  $('errorBox').style.display = 'none';
  $('results').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
  $('runBtn').disabled = true;
  setBadge('queued');
  $('elapsed').textContent = '';
  $('pvtProgress').textContent = '';

  fetch('/api/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(buildPayload()),
  }).then(async (r) => {
    const data = await r.json();
    if (!r.ok) {
      $('runBtn').disabled = false;
      showError('Could not start run:\n' + (data.problems || [data.error]).join('\n'));
      setBadge('idle');
      return;
    }
    poll(data.run_id);
  }).catch(err => {
    $('runBtn').disabled = false;
    setBadge('idle');
    showError('Could not reach the NEBULA UI server: ' + err);
  });
});
</script>
</body>
</html>
"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """http.server.HTTPServer already sets allow_reuse_address = 1 (and
    ThreadingHTTPServer inherits it), so SO_REUSEADDR is already active by
    default -- this subclass just makes that explicit and pins it, so a
    server that exited (cleanly or otherwise) never leaves the port
    unusable for the next `python experiments/web_ui.py` invocation.
    """

    allow_reuse_address = True


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NEBULA local UI server for experiments/run_autockt_pipeline.py.")
    parser.add_argument("--host", default=HOST,
                         help=f"bind address (default: {HOST}, localhost-only -- do not expose externally "
                              "unless you specifically intend to)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default: {DEFAULT_PORT})")
    return parser.parse_args(argv)


def build_server(host: str, port: int) -> ReusableThreadingHTTPServer:
    """Constructs (and binds) the server without starting it -- separated
    from main() so tests can verify host/port/reuse behavior without
    calling the blocking serve_forever().
    """

    return ReusableThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    server = build_server(args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"NEBULA UI running at {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
