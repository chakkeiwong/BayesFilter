"""Static checks for the NeuTra curriculum control campaign harness."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs/benchmarks/run_neutra_curriculum_control_target_2026_08_15.py"
CAMPAIGN = ROOT / "docs/benchmarks/run_neutra_curriculum_control_campaign_2026_08_15.py"


def test_target_runner_uses_search_tournament_and_fresh_exact_audit() -> None:
    source = TARGET.read_text(encoding="utf-8")
    assert "search_neutra_curriculum(" in source
    assert "select_neutra_protocol(" in source
    assert "tune_neutra_curriculum_probe(" in source
    assert "train_neutra_curriculum_protocol(" in source
    assert "base._proposal_audit(" in source
    assert 'output / "progress.json"' in source
    assert "dense_iaf_five_stage_variable_groups(transport)" in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "jit_compile=True" in source
    assert "import numpy" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source


def test_campaign_is_bounded_to_gaussian_and_banana_with_one_hour_cap() -> None:
    source = CAMPAIGN.read_text(encoding="utf-8")
    assert 'TARGETS = ("gaussian", "banana")' in source
    assert "TIMEOUT_SECONDS = 3600.0" in source
    assert '"--resume"' in source
    assert '"TF_FORCE_GPU_ALLOW_GROWTH": "true"' in source
