"""Static tests for the German-credit target canary contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "docs/benchmarks/probe_weighted_neutra_german_credit_target_2026_08_13.py"


def test_probe_is_gpu_xla_memory_growth_and_batch_native() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)' in source
    assert "@tf.function(jit_compile=True" in source
    assert "evaluate_batch_native_value_score" in source
    assert "batch-size" in source
    assert 'if hasattr(value, "as_list"):' in source
