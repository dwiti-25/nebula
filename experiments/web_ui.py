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
import os
import re
import select
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
RUN_TIMEOUT_S = 7200.0  # 2 hours

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()

TARGET_MODES = ("trivial", "hard", "custom")
BACKENDS = ("synthetic", "real")
# Kept as a plain string tuple (not imported from experiments.run_autockt_
# pipeline) so this server never has to import torch/simulator/rl at
# startup -- see the module docstring's design note. Cross-checked against
# the pipeline's own PVT_CONDITION_SETS keys by
# tests/test_web_ui.py::PvtOptionsMatchPipelineTests so the two cannot
# silently drift apart.
PVT_CONDITION_SETS = ("none", "smoke", "minimal27")
TRADE_OFF_PREFERENCES = (
    "most_robust", "lowest_power", "strongest_eye_height", "widest_eye", "largest_margin", "balanced",
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
    if payload.get("backend") == "real" and not payload.get("checkpoint"):
        problems.append("backend='real' requires a checkpoint (an untrained policy against real SPICE "
                         "has no learned behavior to demonstrate -- see docs/autockt-mapping.md sec 20)")
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = (PROJECT_ROOT / checkpoint).resolve()
        if PROJECT_ROOT not in checkpoint_path.parents or not checkpoint_path.is_file():
            problems.append(f"checkpoint not found inside the project: {checkpoint}")
    if payload.get("pvt_condition_set") not in PVT_CONDITION_SETS:
        problems.append(f"pvt_condition_set must be one of {PVT_CONDITION_SETS}")
    if payload.get("trade_off_preference") not in TRADE_OFF_PREFERENCES:
        problems.append(f"trade_off_preference must be one of {TRADE_OFF_PREFERENCES}")
    return problems


def _build_argv(payload: dict[str, Any], *, output_path: Path, schematic_path: Path) -> list[str]:
    argv = [
        sys.executable, "-m", "experiments.run_autockt_pipeline",
        "--backend", payload["backend"],
        "--episodes", str(int(payload["episodes"])),
        "--horizon", str(int(payload["horizon"])),
        "--initial-indices-source", payload.get("initial_indices_source", "verified"),
        "--pvt-condition-set", payload["pvt_condition_set"],
        "--trade-off-preference", payload["trade_off_preference"],
        "--output", str(output_path),
        "--export-schematic", str(schematic_path),
    ]
    if payload["target_mode"] == "custom":
        argv += ["--target-json", json.dumps({f: float(payload["target"][f]) for f in TARGET_SPEC_FIELDS})]
    else:
        argv += ["--target-mode", payload["target_mode"]]
    if payload.get("checkpoint"):
        argv += ["--checkpoint", payload["checkpoint"]]
    if payload.get("measure_hd3_noise"):
        argv += ["--measure-hd3-noise"]
    return argv


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
    """Reads proc.stdout and proc.stderr CONCURRENTLY (via select) until
    both hit EOF or `deadline` (time.monotonic()) passes -- concurrently,
    so a child that fills its stderr pipe's OS buffer while this only read
    stdout could never deadlock this server. Every complete stdout line is
    parsed as a possible progress event and applied live (under
    _RUNS_LOCK) as it streams in, so /api/status reflects progress WHILE
    the run is still going, not only after it exits. Returns
    (stdout_text, stderr_text, timed_out) -- does not kill, wait(), or
    close the streams; the caller still owns all of that.

    Uses os.read() (one raw syscall, returns as soon as ANY data is
    available) rather than the stream's own buffered .read(n) -- the
    latter can keep issuing further raw reads to try to fill the full `n`
    bytes even after select() reported the fd readable, which blocks past
    what select() promised and defeats the whole point of multiplexing.
    Popen must therefore be constructed WITHOUT text=True (binary pipes);
    decoding happens here instead.
    """
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_buffer = ""
    readers: dict[Any, list[bytes]] = {proc.stdout: stdout_chunks, proc.stderr: stderr_chunks}
    open_readers = list(readers)
    timed_out = False

    while open_readers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select(open_readers, [], [], min(remaining, 1.0))
        for stream in ready:
            chunk = os.read(stream.fileno(), 4096)
            if chunk == b"":
                open_readers.remove(stream)
                continue
            readers[stream].append(chunk)
            if stream is proc.stdout:
                stdout_buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in stdout_buffer:
                    line, stdout_buffer = stdout_buffer.split("\n", 1)
                    _consume_stdout_line(run_id, line)

    if stdout_buffer:
        _consume_stdout_line(run_id, stdout_buffer)

    return (
        b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        timed_out,
    )


def _execute_run(run_id: str, argv: list[str], output_path: Path, schematic_path: Path) -> None:
    with _RUNS_LOCK:
        _RUNS[run_id]["status"] = "running"
        _RUNS[run_id]["started_at"] = time.monotonic()

    timed_out = False
    try:
        proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + RUN_TIMEOUT_S
        stdout, stderr, timed_out = _drain_subprocess_with_progress(proc, run_id, deadline)
        if timed_out:
            proc.kill()
            extra_stdout, extra_stderr = proc.communicate()  # reap the now-terminated process
            stdout += (extra_stdout or b"").decode("utf-8", errors="replace")
            stderr += (extra_stderr or b"").decode("utf-8", errors="replace")
        else:
            proc.wait()
            proc.stdout.close()
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
                f"the pipeline subprocess did not finish within {RUN_TIMEOUT_S / 60:.0f} minutes and was "
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
        elif returncode is not None and returncode < 0:
            entry["status"] = "failed"
            entry["error"] = (
                f"the pipeline subprocess was terminated by signal {-returncode} "
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

    with _RUNS_LOCK:
        _RUNS[run_id] = {
            "run_id": run_id, "status": "queued", "command": argv,
            "output_path": str(output_path), "schematic_path": str(schematic_path),
            "started_at": None, "finished_at": None, "returncode": None,
            "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            "pvt_condition_set": payload.get("pvt_condition_set"),
            "pvt_conditions_total": None, "pvt_conditions_completed": None,
            "pvt_current_corner": None, "pvt_current_temperature_c": None, "pvt_current_supply_v": None,
        }

    thread = threading.Thread(target=_execute_run, args=(run_id, argv, output_path, schematic_path), daemon=True)
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
    if entry.get("pvt_condition_set") == "minimal27":
        # Full 27-point sweep: no ETA (none of this is based on measured
        # data), just an explicit "this is long-running" label.
        payload["pvt_long_running_full_sweep"] = True

    return payload


def _available_checkpoints() -> list[str]:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in results_dir.glob("*.pt")
    )


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

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, INDEX_HTML)
        elif path == "/api/checkpoints":
            self._send_json(200, {"checkpoints": _available_checkpoints()})
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
      <label for="checkpoint">PPO checkpoint</label>
      <select id="checkpoint"><option value="">(untrained policy -- synthetic backend only)</option></select>
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
        <option value="minimal27">Full 27-point sweep -- TT/SS/FF x VDD+/-5% x 0-125C (SLOW, ~2h/design)</option>
      </select>
      <p class="hint">"None" and "smoke" do NOT establish PVT robustness -- only "Full 27-point sweep" does
        (see docs/autockt-mapping.md sec 22's 27/27 result). The manual real-SPICE demo run used "None".</p>
      <label for="tradeOff">Trade-off preference (used on PVT ties)</label>
      <select id="tradeOff">
        <option value="most_robust">Most robust</option>
        <option value="lowest_power">Lowest power</option>
        <option value="strongest_eye_height">Strongest eye height</option>
        <option value="widest_eye">Widest eye</option>
        <option value="largest_margin">Largest margin</option>
        <option value="balanced">Balanced</option>
      </select>
      <p class="hint">"smoke" spends real SPICE only on nominally-feasible candidates -- not the full 27-point robustness sweep.</p>
    </fieldset>

    <button id="runBtn">Run NEBULA</button>
    <div id="statusLine"><span id="statusBadge" class="status-badge status-idle">idle</span><span id="elapsed"></span><span id="pvtProgress"></span></div>
  </div>

  <div class="panel">
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
});

function setBadge(status) {
  const badge = $('statusBadge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status;
}

function buildPayload() {
  const targetMode = $('targetMode').value;
  const payload = {
    target_mode: targetMode,
    backend: $('backend').value,
    checkpoint: $('checkpoint').value || null,
    episodes: parseInt($('episodes').value, 10),
    horizon: parseInt($('horizon').value, 10),
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

function renderResult(payload) {
  $('errorBox').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
  $('results').style.display = 'block';

  const result = payload.result;
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
  if (pvt) {
    $('pvtCard').style.display = 'block';
    $('pvtSummary').innerHTML =
      `<div class="param-grid"><div><div class="k">Pass rate</div><div class="v">${pvt.n_passing}/${pvt.n_conditions}` +
      ` (${(pvt.pass_rate*100).toFixed(0)}%)</div></div>` +
      `<div><div class="k">Met minimum</div><div class="v">${pvt.met_minimum_pass_rate}</div></div></div>`;
  } else {
    $('pvtCard').style.display = 'none';
  }

  $('runSummary').innerHTML =
    `<div><div class="k">Candidates generated</div><div class="v">${result.n_candidates_generated}</div></div>` +
    `<div><div class="k">Nominally feasible</div><div class="v">${result.n_nominally_feasible}</div></div>` +
    `<div><div class="k">Backend</div><div class="v">${result.backend}</div></div>` +
    `<div><div class="k">Elapsed</div><div class="v">${payload.elapsed_s}s</div></div>`;

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
    text += ' — full 27-point sweep, long-running';
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
    if (payload.status === 'completed') {
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
