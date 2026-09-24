"""Orchestration checks for the bounded frozen-map assessment."""
import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf


def module():
    path = Path(__file__).resolve().parents[1]/"docs/benchmarks/run_q20_configured_hmc_2026_09_24.py"
    spec = importlib.util.spec_from_file_location("q20_configured_hmc", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_protocol_keeps_identity_mass_and_separate_hmc_budget():
    config = module().protocol()
    assert config["role"] == "development"
    assert config["jit_compile"] is True
    assert config["tuning"]["l_grid"] == [3, 5, 9]
    assert config["tuning"]["repair_reserve_units"] == 20
    assert config["posterior"]["warmup_min"] >= 2000
    assert config["posterior"]["retained_rhat"] <= 1.01


def test_nomination_prefers_replicated_family_without_ranking_metrics():
    loaded = module()
    comparisons = []
    for family in ("iaf16", "naf16"):
        for seed in range(3):
            comparisons.append({"family": family, "root_seed": seed,
                "fixed_checkpoint_training_improvement": False,
                "checkpoint": f"{family}-{seed}-checkpoint",
                "finalized": f"{family}-{seed}-finalized"})
    report = {"families": {"iaf16": {"replicated_local_improvement": False},
        "naf16": {"replicated_local_improvement": True}}, "comparisons": comparisons}
    ordered = loaded.nominate(report)
    assert [row["family"] for row in ordered[:3]] == ["naf16"]*3
    assert {row["root_seed"] for row in ordered[:3]} == {0, 1, 2}


def test_curvature_warm_start_recovers_known_anisotropic_hessian():
    class Target:
        def log_prob_and_grad(self, z):
            precision = tf.constant([1., 4., 9., 16.], tf.float64)
            return -.5*tf.reduce_sum(z*z*precision, axis=-1), -z*precision
    z = tf.constant([[.1, -.2, .3, -.4], [2., 3., -1., 1.]], tf.float64)
    result = module().curvature_warm_start(Target(), z)
    assert result["epsilon"] == pytest.approx(.125, rel=1.e-8)
    for row in result["curvature_eigenvalues"]:
        assert row == pytest.approx([1., 4., 9., 16.], rel=1.e-8)


@pytest.mark.parametrize("family", ["iaf", "naf_dsf"])
def test_frozen_map_reaches_public_tuning_replay_and_posterior_controller(tmp_path, family):
    """Tiny known-target CPU smoke; short thresholds carry no posterior claim."""
    from tests.test_q20_production_repair import four_dimensional_bridge, tiny_protocol
    from bayesfilter.inference.q20_hmc_qualification import qualify_bridge, attach_qualification
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
    from bayesfilter.inference import load_hmc_candidate_retained_runner
    loaded = module()
    config = tiny_protocol()
    config["tuning"].update(l_grid=[3], max_repairs_per_family=4, total_budget_units=30,
        practical_region=[.41, .99], repair_region=[.405, .995], max_wall_seconds=120.)
    bridge = four_dimensional_bridge()
    qualify_bridge(config, bridge, tmp_path/"qualification", betas=[1.])
    attach_qualification(bridge, tmp_path/"qualification/result.json", config, betas=[1.])
    cfg = NeuTraTransportConfig(4, family, (4,), 1, "elu", (21, 72), 2., mixture_components=3)
    flow = NeuTraTransport(cfg)
    finalized = {"frozen_transport": flow.frozen_payload(target_signature=bridge.fixed_beta_adapter(1.).adapter_signature())}
    result = loaded.tune_map(config, bridge, finalized, tmp_path/"tuning", max_seconds=120.)
    assert result["verified_members"]
    member_path = next(iter(result["verified_members"].values()))
    member = load_hmc_candidate_retained_runner(member_path, adapter=bridge.fixed_beta_adapter(1.))
    assert member._binding._spec["target_lineage"]["method"] == "neutra"
    assert member._binding.config.chain_mode == "batched"
    outcome = loaded.assess_member(config, bridge, member_path, tmp_path/"posterior", max_seconds=120., chunk_seconds=0.)
    assert Path(outcome["result"]).exists()
    assert list((tmp_path/"posterior/chunks").iterdir())
