"""CPU mechanics tests; chunk equivalence is not posterior validation."""
from dataclasses import replace
import json

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc import FullChainHMCConfig
from bayesfilter.inference.hmc_bootstrap_checkpoint import CheckpointedBootstrapRunner, require_chunk_health
from bayesfilter.inference.hmc_preparation import HMCPreparationProgress, HMCPreparationBudgetExceeded, HMCPreparationFailure
from bayesfilter.inference.hmc_kernel_tuning import _classify_bootstrap_screen
from tests.test_hmc_bootstrap_initialization import Gaussian, inputs


def config(xla=False):
    return FullChainHMCConfig(num_results=8, num_burnin_steps=4, step_size=.15,
        num_leapfrog_steps=3, seed=(723, 198), use_xla=xla,
        target_scope=Gaussian.target_scope, target_status_trace_policy="per_chain_step")


class Interrupt(HMCPreparationProgress):
    def phase(self, stage, payload=None):
        super().phase(stage, payload)
        if stage == "bootstrap_chunk_complete":
            raise HMCPreparationBudgetExceeded("injected interruption after durable checkpoint")


@pytest.mark.parametrize("xla", [False, True])
def test_resume_matches_uninterrupted_chunk_stream(tmp_path, xla):
    adapter, initial = Gaussian(), tf.zeros([2], tf.float64)
    def runner(name, progress=None, **kw):
        return CheckpointedBootstrapRunner(tmp_path / name, sources={"toy": "frozen"},
            progress=progress or HMCPreparationProgress(tmp_path / (name + "-progress")),
            seconds_per_transition=.01, **kw)
    direct = runner("direct")(adapter, initial, config(xla))
    with pytest.raises(HMCPreparationBudgetExceeded, match="injected"):
        runner("partial", Interrupt(tmp_path / "interrupted"))(adapter, initial, config(xla))
    resumed_runner = runner("resumed", resume_from=tmp_path / "partial")
    resumed = resumed_runner(adapter, initial, config(xla))
    tf.debugging.assert_equal(direct.samples, resumed.samples)
    tf.nest.map_structure(tf.debugging.assert_equal, direct.trace, resumed.trace)
    events = resumed_runner.progress.events
    assert sum(e["details"].get("resumed", False) for e in events) == 1
    assert resumed.metadata["burnin_transitions"] == 4
    assert resumed.samples.shape == (8, 2)
    assert len(list((tmp_path / "resumed").glob("*/chunk-*.json"))) == 3


@pytest.mark.parametrize("defect", ["source", "config", "checksum", "start"])
def test_resume_rejects_changed_scope_or_corrupt_chunk(tmp_path, defect):
    args = dict(sources={"toy": "frozen"}, progress=HMCPreparationProgress(None), seconds_per_transition=.01)
    cfg, initial = config(), tf.zeros([2], tf.float64)
    CheckpointedBootstrapRunner(tmp_path / "original", **args)(Gaussian(), initial, cfg)
    if defect == "source":
        args["sources"] = {"toy": "changed"}
    elif defect == "config":
        cfg = replace(cfg, step_size=.2)
    elif defect == "start":
        initial = tf.ones([2], tf.float64)
    else:
        path = next((tmp_path / "original").glob("*/chunk-*.json"))
        row = json.loads(path.read_text())
        row["wall_seconds"] += 1
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="checkpoint|resume"):
        CheckpointedBootstrapRunner(tmp_path / "resume", resume_from=tmp_path / "original", **args)(Gaussian(), initial, cfg)


def test_cost_gate_defers_before_first_numerical_call(tmp_path):
    progress = HMCPreparationProgress(tmp_path / "progress", max_wall_time_seconds=1.)
    runner = CheckpointedBootstrapRunner(tmp_path / "chunks", sources={"toy": "frozen"},
        progress=progress, seconds_per_transition=10.)
    with pytest.raises(HMCPreparationBudgetExceeded), progress:
        runner(Gaussian(), tf.zeros([2], tf.float64), config())
    assert not list((tmp_path / "chunks").glob("*/chunk-*.json"))
    assert json.loads(progress.path.read_text())["status"] == "deferred"


def test_discarded_invalid_proposal_is_a_veto(tmp_path, monkeypatch):
    from bayesfilter.inference import hmc
    original = hmc.ReusableFullChainHMCRunner.run
    def bad(self, **kwargs):
        result = original(self, **kwargs)
        return replace(result, trace={**result.trace, "target_score_finite": tf.constant([False, True, True, True])})
    monkeypatch.setattr(hmc.ReusableFullChainHMCRunner, "run", bad)
    runner = CheckpointedBootstrapRunner(tmp_path / "chunks", sources={"toy": "frozen"},
        progress=HMCPreparationProgress(None), seconds_per_transition=.01)
    with pytest.raises(HMCPreparationFailure, match="health veto"):
        runner(Gaussian(), tf.zeros([2], tf.float64), config())
    assert len(list((tmp_path / "chunks").glob("*/chunk-*.json"))) == 1


def test_high_acceptance_only_passes_explicit_startup_role():
    _, _, cfg = inputs()
    diag = dict(acceptance_rate=1., runtime_finite=True, log_accept_ratio_finite=True,
                samples_all_finite=True, target_log_prob_finite=True)
    assert _classify_bootstrap_screen(cfg, diagnostics=diag, screen_error=None)[0] == "repair"
    startup = replace(cfg, acceptance_role="warmup_startup_only")
    assert _classify_bootstrap_screen(startup, diagnostics=diag, screen_error=None)[:2] == (
        "passed", "finite_startup_for_adaptation_only")
    assert _classify_bootstrap_screen(startup, diagnostics={**diag, "acceptance_rate": .2}, screen_error=None)[0] == "repair"
    assert _classify_bootstrap_screen(startup, diagnostics={**diag, "log_accept_ratio_finite": False}, screen_error=None)[0] == "hard_veto"
