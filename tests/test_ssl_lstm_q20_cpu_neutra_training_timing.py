from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/benchmark_ssl_lstm_q20_cpu_neutra_training_2026_07_22.py"
)
PLAN = (
    ROOT
    / "docs/plans/bayesfilter-ssl-lstm-q20-cpu-training-timing-plan-2026-07-22.md"
)


def test_cpu_timing_harness_caps_threads_before_tensorflow_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tensorflow_import = source.index("import tensorflow as tf")
    assert source.index('"CUDA_VISIBLE_DEVICES": "-1"') < tensorflow_import
    assert source.index('"TF_NUM_INTRAOP_THREADS"') < tensorflow_import
    assert source.index('"TF_NUM_INTEROP_THREADS"') < tensorflow_import
    assert "CONFIGURED_TF_COMPUTE_THREADS" in source
    assert "if CONFIGURED_TF_COMPUTE_THREADS > 50" in source
    assert "if len(affinity) > 50" in source
    assert 'os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1"' in source
    assert "if not IS_POOL_WORKER_IMPORT" in source


def test_cpu_timing_harness_matches_requested_problem() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Q = 20" in source
    assert "BATCH_SIZE = 100" in source
    assert "HIDDEN_LAYERS = (32, 32)" in source
    assert "WORKER_COUNT = 8" in source
    assert "jit_compile=False" in source
    assert "native thread count exceeds the strict 50-thread cap" in source
    assert "TRAINING_STEPS = (250, 1250, 2000)" in source
    assert "trained state is discarded" in source


def test_plan_records_full_timing_and_nonclaim_contract() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert "4 + 1 + 8 * (1 + 1) = 21" in text
    assert "fails if the complete process-tree sum exceeds 50" in text
    assert "validation/support checkpoint time" in text
    assert "No transport-quality" in text
    assert "taskset -c 0-49" in text
