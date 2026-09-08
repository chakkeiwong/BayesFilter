from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py"
)


def test_phase4_runner_is_batch_native_gpu_xla_and_has_no_numpy_or_hmc() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "numpy" not in imports
    assert 'os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "jit_compile=True" in source
    assert "WeightedForwardKLNeuTraTrainer" in source
    assert "tf.map_fn(" not in source
    assert "tf.vectorized_map(" not in source
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source
    assert '"hmc_launched": False' in source


def test_phase4_runner_has_disjoint_split_and_frozen_selection() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "TRAIN_COUNT = 60" in source
    assert "VALIDATION_COUNT = 20" in source
    assert "AUDIT_COUNT = 20" in source
    assert "selection_frozen_before_audit" in source
    assert "m0_protocol_hash" in source
    assert "--m0-root" in source


def test_phase4_help_runs_without_importing_tensorflow() -> None:
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
    assert "--output-root" in completed.stdout


def test_neutra_split_uses_pilot_mode_axis() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MODE_AXIS = 2" in source
    assert "theta[:, MODE_AXIS]" in source


def test_neutra_tuning_profile_has_distinct_capacity_and_learning_rate_arms() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'choices=("screen", "tuning", "capacity")' in source
    assert '"compact_low_lr"' in source
    assert '"wider_mid_lr"' in source
    assert '"high_capacity"' in source


def test_affine_precondition_is_explicit_and_composes_logdet() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'choices=("identity", "affine")' in source
    assert "affine_weighted_moment_oracle" in source
    assert '"logdet"' in source


def test_neutra_reports_full_bank_moments_separately_from_validation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"full_bank"' in source
    assert "full_log_weights" in source
