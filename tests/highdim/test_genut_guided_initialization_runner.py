from __future__ import annotations

import inspect

from bayesfilter.highdim import genut_guided_proposal_tf as proposal
from docs.benchmarks import run_genut_guided_initialization_phase1 as runner


def test_phase1_campaign_partitions_and_budget_are_bounded() -> None:
    assert runner.PARTICLE_COUNT == 96
    assert runner.HORIZONS == (1, 10)
    assert len(runner.DGP_SEEDS) == 3
    assert len(runner.PARTICLE_SEEDS) == 4
    assert runner.GUIDED_RHOS == (0.0, 0.1, 0.25, 0.5)
    assert len(set(runner.DGP_SEEDS)) == len(runner.DGP_SEEDS)
    assert set(runner.DGP_SEEDS).isdisjoint(runner.PARTICLE_SEEDS)


def test_phase1_summary_is_explicitly_descriptive() -> None:
    rows = [
        {
            "horizon": 1,
            "method": "guided_rho_0.25",
            "value_error": value,
            "score_error_l2": abs(value) + 1.0,
            "elapsed_seconds": 0.1,
            "ess": [80.0],
            "maximum_normalized_weight": [0.03],
        }
        for value in (-0.2, 0.1, 0.3)
    ]
    summary = runner._summaries(rows)[0]  # noqa: SLF001
    assert summary["run_count"] == 3
    assert summary["inference_status"] == "descriptive_only_underpowered_pilot"


def test_runner_uses_standard_score_without_autodiff_or_guided_jvp() -> None:
    runner_source = inspect.getsource(runner)
    proposal_source = inspect.getsource(proposal)
    assert "finite_value_standard_score_guided_proposal" in runner_source
    for disallowed in (
        "GradientTape",
        "ForwardAccumulator",
        "_restore_cloud_jvp_core",
        "finite_value_score_guided_proposal_forward_jvp",
    ):
        assert disallowed not in runner_source
        assert disallowed not in proposal_source


def test_cpu_reference_policy_requires_hidden_gpu(monkeypatch) -> None:
    monkeypatch.setattr(runner.tf.config, "list_physical_devices", lambda kind: [])
    monkeypatch.setattr(
        runner.tf.config,
        "list_logical_devices",
        lambda kind: [type("Device", (), {"name": "/device:CPU:0"})()],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    policy = runner._configure_cpu_reference()  # noqa: SLF001
    assert policy["trust_basis"] == "cpu_reference_gpu_intentionally_hidden"
    assert policy["memory_policy"]["cuda_visible_devices"] == "-1"
    assert policy["jit_compile"] is True
    assert policy["tf32_execution_enabled"] is False


def test_rho_one_replay_diagnostics_are_paired_and_fail_closed() -> None:
    baseline = {
        "horizon": 1,
        "dgp_seed": 11,
        "particle_seed": 22,
        "method": "iid_bootstrap",
        "value": -3.0,
        "score": [1.0, 2.0],
    }
    guided = {**baseline, "method": "guided_rho_1", "value": -3.000001}
    replay = runner._rho_one_replay_diagnostics([baseline, guided])  # noqa: SLF001
    assert replay["passed"] is True
    assert replay["pair_count"] == 1
