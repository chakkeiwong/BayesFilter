from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_affine_oracle_2026_08_25.py"


def test_affine_oracle_is_cpu_hidden_and_numpy_free() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "import numpy" not in source
    assert "triangular_solve" in source
    assert "whitened_covariance" in source


def test_affine_oracle_help() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    result = subprocess.run((sys.executable, str(RUNNER), "--help"), cwd=ROOT, env=env, capture_output=True, text=True, timeout=30.0)
    assert result.returncode == 0
    assert "--pilot-root" in result.stdout
