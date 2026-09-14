from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from zipfile import ZipFile


class Phase2ArtifactIntegrityTests(unittest.TestCase):
    def test_every_archived_member_matches_the_tracked_sha256_ledger(self):
        root = Path(__file__).resolve().parents[1]
        archive_path = root / "artifacts" / "NEBULA_phase2_experimental_artifacts.zip"
        checksum_path = root / "artifacts" / "PHASE2_SHA256SUMS.txt"
        expected = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, member = line.split(maxsplit=1)
            expected[member.strip()] = digest
        with ZipFile(archive_path) as archive:
            members = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
            self.assertEqual(members, set(expected))
            mismatches = [
                member for member, digest in expected.items()
                if hashlib.sha256(archive.read(member)).hexdigest() != digest
            ]
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()

