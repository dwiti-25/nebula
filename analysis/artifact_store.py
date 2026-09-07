"""Read-only access to files preserved in the tracked Phase-2 package."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from zipfile import ZipFile

ARTIFACT_RELATIVE_PATH = Path("artifacts/NEBULA_phase2_experimental_artifacts.zip")
ARCHIVE_PREFIX = "NEBULA_phase2_experimental_artifacts/"


def read_packaged_bytes(root: str | Path, relative: str | Path) -> Optional[bytes]:
    archive_path = Path(root) / ARTIFACT_RELATIVE_PATH
    if not archive_path.is_file():
        return None
    member = ARCHIVE_PREFIX + Path(relative).as_posix()
    with ZipFile(archive_path) as archive:
        try:
            return archive.read(member)
        except KeyError:
            return None


def packaged_file_exists(root: str | Path, relative: str | Path) -> bool:
    return read_packaged_bytes(root, relative) is not None

