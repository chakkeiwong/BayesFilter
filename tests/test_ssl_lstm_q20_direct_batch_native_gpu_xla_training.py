from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_direct_batch_native_gpu_xla_training_2026_07_30.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-plan-2026-07-30.md"
)


def test_runner_is_syntax_valid_and_has_no_numpy_runtime_import() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "numpy" not in imports
    assert "TF_FORCE_GPU_ALLOW_GROWTH" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "configure_tensorflow_gpu_memory_growth(" in source
    assert "jit_compile=True" in source
    assert "bound_batch_native_neutra_training_target(" in source
    assert "require_batch_native_neutra_target(" in source
    assert '"scalar_target": Path(' in source
    assert "material-budget-ledger.json" in source


def test_material_modes_and_update_caps_match_reviewed_plan() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    plan = PLAN.read_text(encoding="utf-8")

    assert 'MATERIAL_MODES = {"mechanics", "tuning-arm", "final-stream"}' in source
    assert "MATERIAL_CAP_SECONDS = 18000.0" in source
    assert "TUNING_STEPS = 100" in source
    assert "FINAL_MAX_STEPS = 1000" in source
    assert "BATCH_SIZE = 100" in source
    assert "18,000 cumulative wall seconds" in plan
    assert "400 total tuning updates" in plan
    assert "2,000 total final updates" in plan


def test_runner_has_no_hmc_launch_or_row_mapping_call() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "tf.map_fn(" not in source
    assert "tf.vectorized_map(" not in source
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source
    assert '"hmc_launched": False' in source


def test_final_requires_repository_tuning_hash_and_exact_source_scope() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "tuning selection hash mismatch" in source
    assert "tuning selection source scope mismatch" in source
    assert "tuning binding closure mismatch" in source
    assert "selected tuning result hash mismatch" in source
    assert "validated_selection(args, binding)" in source


def test_runner_imports_cpu_hidden_in_subprocess() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"

    completed = subprocess.run(
        (
            sys.executable,
            str(RUNNER),
            "--help",
        ),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout
