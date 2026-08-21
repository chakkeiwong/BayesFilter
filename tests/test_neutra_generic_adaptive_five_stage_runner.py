"""Static checks for the generic adaptive five-stage repair harnesses."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_adaptive_five_stage_model_2026_08_15.py"
CAMPAIGN = ROOT / "docs/benchmarks/run_neutra_generic_adaptive_five_stage_campaign_2026_08_15.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_runner_uses_generic_adaptive_controller_on_gpu_xla() -> None:
    source = MODEL_RUNNER.read_text(encoding="utf-8")
    assert "NeuTraAdaptiveStagePolicy(" in source
    assert "train_neutra_five_stage(" in source
    assert 'optimizer_state_policy="carry_selected"' not in source
    assert '"carry_selected" if args.route == "adaptive_carry"' in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "jit_compile=True" in source
    assert "import numpy" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source


def test_campaign_has_matched_ceiling_two_targets_and_two_seeds() -> None:
    runner = _load(CAMPAIGN, "adaptive_five_stage_campaign")
    assert runner.TARGETS == ("gaussian", "banana")
    assert runner.ROUTES == ("adaptive_reset", "adaptive_carry", "cold")
    assert runner.SEEDS == (0, 1)
    assert runner.CAMPAIGN_TIMEOUT_SECONDS == 2700.0
    args = type("Args", (), {"python": "python", "device": "0"})()
    command = runner._command(args, "gaussian", "adaptive_reset", 0, Path("out"))
    values = {command[index]: command[index + 1] for index in range(0, len(command) - 1) if command[index].startswith("--")}
    selected_ceiling = (
        int(values["--affine-updates"])
        + int(values["--simple-updates"])
        + 3 * int(values["--progressive-updates"])
        + int(values["--joint-updates"])
    )
    assert selected_ceiling == int(values["--cold-updates"]) == 3000
