from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_smc_u_replication_2026_08_25.py"
)


def test_replication_runner_is_numpy_free_and_fixture_first() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "numpy" not in imports
    assert "FIXTURE_REPLICATES = 64" in source
    assert "FIXTURE_PARTICLES = 128" in source
    assert "if fixture[\"status\"] != \"PASS\"" in source
    assert "random-walk" in source
    assert "MUTATION_SCALE = 0.05" in source
    assert "--q20-seeds" in source
    assert "--q20-particles" in source


def test_replication_runner_has_no_hmc_or_promotion_claim() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "HamiltonianMonteCarlo(" not in source
    assert "sample_chain(" not in source
    assert "posterior correctness" in source
    assert "default promotion" in source


def test_replication_help_runs_cpu_hidden() -> None:
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
    assert "--fixture-only" in completed.stdout
