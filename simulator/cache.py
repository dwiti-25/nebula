"""Small content-addressed JSON cache for scalar evaluations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping
from contextlib import contextmanager


class EvaluationCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, evaluation_id: str) -> Path:
        return self.root / evaluation_id[:2] / f"{evaluation_id}.json"

    def get(self, evaluation_id: str) -> dict[str, object] | None:
        path = self._path(evaluation_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) and value.get("evaluation_id") == evaluation_id else None

    @contextmanager
    def _key_lock(self, evaluation_id: str):
        """Serialize same-key commits with an atomic, bounded lock file."""
        lock_path = self._path(evaluation_id).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        acquired = False
        for _ in range(400):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                    stream.write(f"{os.getpid()} {time.time():.6f}\n")
                acquired = True
                break
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 30.0
                    if stale:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.01)
            except PermissionError:
                # Windows can report sharing violations as PermissionError
                # rather than FileExistsError for an existing lock file.
                # The owner may unlink between the failing open and an exists
                # check, so any sharing violation under a valid cache parent is
                # treated as transient contention.
                time.sleep(0.01)
        if not acquired:
            raise TimeoutError(f"timed out locking cache key {evaluation_id}")
        try:
            yield
        finally:
            for attempt in range(20):
                try:
                    lock_path.unlink()
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.01 * (attempt + 1))

    def put(self, evaluation_id: str, value: Mapping[str, object], *,
            replace_existing: bool = False) -> Path:
        path = self._path(evaluation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**value, "evaluation_id": evaluation_id}
        # A fixed ``.tmp`` name races when several RL workers finish the same
        # evaluation together.  A same-directory unique file plus os.replace
        # gives every worker a private write and an atomic commit on Windows
        # and POSIX.  Identical content makes last-writer-wins harmless.
        with self._key_lock(evaluation_id):
            existing = self.get(evaluation_id)
            if existing is not None and not replace_existing:
                return path
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{evaluation_id}.", suffix=".tmp", dir=path.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True, default=str)
                    stream.flush()
                    os.fsync(stream.fileno())
                for attempt in range(20):
                    try:
                        os.replace(temporary, path)
                        break
                    except PermissionError:
                        if attempt == 19:
                            # An antivirus/indexer can briefly hold the target.
                            # An already-valid identical entry is success.
                            existing = self.get(evaluation_id)
                            if existing == payload:
                                break
                            raise
                        time.sleep(0.01 * (attempt + 1))
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        return path
