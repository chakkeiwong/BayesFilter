from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_trial0_alternative_confirmation_2026_07_16.py"
)
POLICY_PATH = (
    ROOT
    / "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-policy-2026-07-16.json"
)


def load_runner():
    name = "ssl_lstm_neutra_trial0_alternative_confirmation_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_trial0_policy_binds_surviving_trial_and_untouched_streams() -> None:
    payload = runner.load_policy(POLICY_PATH)
    assert payload["selected_hyperparameters"] == {
        "learning_rate": 0.0011219623709077644,
        "initialization_scale": 0.02,
        "gradient_clip_norm": 5.0,
    }
    assert [stream.label for stream in runner.FRESH_STREAMS] == ["fresh-g", "fresh-h"]
    seeds = {
        seed
        for stream in runner.FRESH_STREAMS
        for seed in (
            stream.initialization_seed,
            stream.training_seed,
            stream.validation_seed,
        )
    }
    assert len(seeds) == 6
    assert not seeds & runner.EXCLUDED_SEED_ROWS


def test_trial0_runner_preserves_support_controller_resume_and_hmc_boundary() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "selected trial record is not surviving trial 0" in source
    assert "alternative policy does not match trial 0" in source
    assert "alternative policy changed support eligibility" in source
    assert "trial-2 confirmation result hash mismatch" in source
    assert "resume checkpoint history hash mismatch" in source
    assert "resume requires a newly authorized cumulative cap above the prior cap" in source
    assert "replace=resume and checkpoint_path.exists()" in source
    assert "resume numerical source binding drift" in source
    assert "previous_summary_sha256" in source
    assert "authorized_resource_extension_only" in source
    assert "checkpoint_probes" in source
    assert "joint_training_checkpoint_payload" in source
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert "run_hmc" not in source.lower()
    assert "fixed_transport_hmc" not in source.lower()


def test_stream_payload_matches_json_roundtrip_representation() -> None:
    stream = runner.FRESH_STREAMS[1]
    assert runner.stream_payload(stream) == {
        "label": "fresh-h",
        "initialization_seed": [20260716, 7106],
        "training_seed": [20260716, 8106],
        "validation_seed": [20260716, 8206],
    }
