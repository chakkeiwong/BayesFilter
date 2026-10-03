"""CPU-only mechanics; these tests do not establish q20 learning quality."""
import importlib.util
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_training_protocol import TrainingSession
from tests.test_q20_production_repair import session


PATH = Path(__file__).resolve().parents[1] / "docs/benchmarks/diagnose_q20_training_continuation_2026_09_18.py"
spec = importlib.util.spec_from_file_location("q20_continuation_diagnostic", PATH)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_shadow_probe_preserves_exact_actual_adam_and_rng_path():
    actual, config, bridge = session()
    actual.advance(3)
    saved = actual.checkpoint()
    reference = TrainingSession.restore(saved, bridge=bridge, config=config,
        expected_scope=actual.scope, preflight_seed=(71, 2))
    result = diagnostic.probe_step(actual, rtol=1e-9, atol=1e-10)
    reference.advance(1)
    assert result["passed"]
    assert len(result["comparisons"]) == 3
    assert actual.checkpoint()["optimizer"] == reference.checkpoint()["optimizer"]
    assert actual.rng_index == reference.rng_index == 4
    for a, b in zip(actual.trainer.variables, reference.trainer.variables, strict=True):
        tf.debugging.assert_equal(a, b)


def test_gradient_disagreement_vetoes_probe(monkeypatch):
    actual, _, _ = session()
    original = actual.trainer.train_step
    from dataclasses import replace
    def wrong_norm(seed):
        result = original(seed)
        return replace(result, gradient_norm=result.gradient_norm + 1.)
    monkeypatch.setattr(actual.trainer, "train_step", wrong_norm)
    with pytest.raises(tf.errors.InvalidArgumentError):
        diagnostic.probe_step(actual, rtol=1e-9, atol=1e-10)


def test_failed_probe_never_launches_continuation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import bayesfilter.inference.q20_campaign_runtime as runtime
    calls = []
    class FailedProbeCampaign:
        def __init__(self, root, **kwargs):
            root.mkdir()
            self.path = root / "campaign.json"
            self.state = {"stages": {}, "attempts": []}
        @contextmanager
        def locked(self):
            yield self
        def save(self):
            self.path.write_text(json.dumps(self.state))
        def remaining(self, diagnostic=False):
            return 1000.
        def execute(self, stage, command, **kwargs):
            calls.append(stage)
            folder = tmp_path / "failed-attempt"
            (folder / "worker").mkdir(parents=True)
            (folder / "worker/result.json").write_text(json.dumps({"status": "failed", "error": "parity"}))
            return {"status": "failed", "directory": str(folder)}
    monkeypatch.setattr(runtime, "Campaign", FailedProbeCampaign)
    allowance = tmp_path / "allowance.json"
    allowance.write_text('{}')
    result = diagnostic.coordinate({"allowance": str(allowance), "source_root": str(tmp_path),
        "config": {"execution": {"termination_grace_seconds": 1}},
        "caps": {"clipping": 100., "continuation": 100.}}, tmp_path / "campaign")
    assert result == 1
    assert calls == ["clipping"]
