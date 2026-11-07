"""Cross-footing: README numbers must recompute from the raw run output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import crossfoot


def test_crossfoot_passes():
    failures = crossfoot.check()
    assert not failures, "\n".join(failures)
