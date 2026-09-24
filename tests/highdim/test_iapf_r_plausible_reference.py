"""Mathematical and endpoint checks for explicitly optional R reconstructions."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_local_equation15_no_fire_boundary_and_actual_consumer():
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no independent R reconstruction evidence")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(ROOT / "tests/reference_iapf_plausible_choices.R"), str(ROOT)],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                          OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
