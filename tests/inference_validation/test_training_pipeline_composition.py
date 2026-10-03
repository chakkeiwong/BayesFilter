"""Tiny CPU training/serialization/tuning composition, never map-quality evidence."""
import pytest
import tensorflow as tf

from bayesfilter.inference import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_training import NeuTraReverseKLTrainer, NeuTraTrainerConfig
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
from bayesfilter.testing.inference_validation.procedures import execute_pipeline


def training_pipeline_composition(tmp_path, target_id, device="cpu_reference"):
    target = ValidationTarget(target_id, jit_compile=device == "gpu")
    observed = []
    class BatchTarget:
        def batch_value_and_score(self, theta):
            observed.append(tuple(theta.shape))
            assert theta.shape[0] > 1
            return target.log_prob_and_grad(theta)
    trainer = NeuTraReverseKLTrainer(BatchTarget(), NeuTraTrainerConfig(
        dimension=2, family="dense_iaf", hidden_layers=(4, 4), jit_compile=device == "gpu"))
    z = tf.random.stateless_normal((8, 2), (20260923, 37), dtype=tf.float64)
    trainer.train_step(z)
    assert observed and all(shape == (8, 2) for shape in observed)
    for signature, program in trainer._compiled_train_step.programs.items():
        assert program.input_signature == signature
        assert program.experimental_get_tracing_count() == 1
    payload = trainer.frozen_transport_payload(transport_id="m37-mechanics-"+target_id,
                                              target_signature=target.adapter_signature())
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target.adapter_signature())
    expected, logdet = trainer.forward_and_logdet(z)
    tf.debugging.assert_equal(loaded.transport.forward_batch(z), expected)
    tf.debugging.assert_equal(loaded.transport.log_abs_det_jacobian_batch(z), logdet)
    tf.debugging.assert_near(loaded.transport.inverse_theta_to_z_batch(expected), z, atol=1e-12)
    design = ValidationDesign(design_id="m37-composition-"+target_id, engine="accuracy",
        scenario=ScenarioSpec(target_id, "fixed_transport"), replications=1, draws=64,
        measurement_draws=64, seed=2026092337, budget_seconds=180, device=device,
        purpose="one-update graph/batch/freeze/retune call chain only",
        numerical_provenance="tiny composition canary; CPU runs are explicit reference exceptions; no learned-quality claim",
        l_grid=(2, 3), step_size=.8, posterior_cap=128,
        options={"transport_payload": payload, "posterior_members": "selected", "member_rule": "first_verified",
                 "global_quantities": ["left_mode_probability"] if target_id == "mixture" else [],
                 "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
                 "search": {"pilot_enabled": False, "refinement_rounds": 0,
                            "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)}})
    result = execute_pipeline(design, tmp_path)
    assert result["verified_candidate_ids"]
    assert len(result["members"]) == len(result["verified_candidate_ids"])
    member = next(m for m in result["members"] if m["status"] == "assessed")
    assert member["warmup_exclusion_matches"]
    if target_id == "mixture":
        targets = member["posterior"]["config"]["assessment_policy"]["precision"]["targets"]
        assert any(t["name"] == "left_mode_probability" for t in targets)
    return {"target": target_id, "device": device, "jit_compile": device == "gpu",
            "batch_size": 8, "updates": 1, "verified_count": len(result["verified_candidate_ids"]),
            "exact_frozen_forward_logdet_parity": True, "posterior_passed": member["posterior"]["passed"],
            "scope": "mechanics_only_not_training_quality"}


@pytest.mark.parametrize("target_id", ["banana", "mixture"])
def test_batch_training_frozen_reload_public_retuning(tmp_path, target_id):
    training_pipeline_composition(tmp_path, target_id)
