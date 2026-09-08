from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_measure_audit_2026_08_25.py"


def test_measure_audit_is_cpu_hidden_and_numpy_free() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "import numpy" not in source
    assert "canonical_protocol_hash" in source
    assert "terminal_weights_match_last_increment" in source


def test_measure_audit_help_does_not_import_tensorflow() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    completed = subprocess.run(
        (sys.executable, str(RUNNER), "--help"),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0
    assert "--pilot-root" in completed.stdout
