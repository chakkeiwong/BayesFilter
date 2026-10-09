"""Framework-free coverage and model configuration inventory for v7 validation.

Settings are explicit diagnostic hypotheses; existence here is not a pass.
The caller supplies evidence counts, seeds and budgets before execution.
"""
from __future__ import annotations

import math


# Every root-cause group has its oracle, execution tier and exact test source.
COVERAGE = (
    ("finite_trial_estimand", "regression", "all T draws; starts/chunks are not replicates", "tests/test_hmc_acceptance_trials.py"),
    ("uncertainty_boundaries", "calibration", "known bounded-law means and simultaneous error/delivery", "tests/test_hmc_acceptance_statistics.py"),
    ("full_candidate_cohort", "calibration", "100 actual pairs and independent verification", "tests/test_hmc_acceptance_calibration.py"),
    ("temporal_preparation", "calibration", "per-start exact window contrasts; no admission effect", "tests/test_hmc_acceptance_calibration.py"),
    ("independent_verification", "regression", "adverse fresh evidence blocks that member", "tests/test_hmc_acceptance_adversarial.py"),
    ("candidate_epsilon_retention", "regression", "opposite L repairs and all verified exports", "tests/test_hmc_acceptance_trials.py"),
    ("rhat_separation", "regression", "high/unavailable report cannot change admission", "tests/test_hmc_acceptance_adversarial.py"),
    ("recovery_seed_cost", "regression", "same trial/stream; one score; all attempts charged", "tests/test_hmc_acceptance_trials.py"),
    ("native_chunk_accounting", "regression", "exact charged extents and missing charges, including checksum-consistent mutations", "tests/test_hmc_acceptance_accounting.py"),
    ("trial_seed_validation_reuse", "regression", "independent seed derivation; every row and charge checked after reuse", "tests/test_hmc_trial_validation_reuse.py"),
    ("serial_trial_execution_compatibility", "regression", "withdrawn experimental worker configuration cannot silently reload as serial", "tests/test_hmc_trial_execution_compatibility.py"),
    ("emitted_analysis_reuse", "regression", "cached emission equals cold raw reconstruction; changed evidence cannot reuse it", "tests/test_hmc_emitted_analysis_reuse.py"),
    ("scoped_reader_replay", "regression", "fresh public member/checkpoint readers agree; current content and predecessor inventories remain checked", "tests/test_hmc_scoped_replay.py"),
    ("confirmation_denominator", "regression", "exact binomial reference, complete declared search and actual independent seeds; missing outcomes remain unsuccessful", "tests/test_hmc_release_confirmation.py"),
    ("confirmation_execution", "regression", "fixed slot/seed inventory, bounded memory-readiness recovery, all costs and failures preserved; fake children are mechanics only", "tests/test_hmc_confirmation_execution.py"),
    ("real_filter_process_recovery", "regression", "QR and nonlinear uninterrupted raw trials equal recovered trials", "tests/test_hmc_acceptance_ssm_recovery.py"),
    ("target_scope_identity", "regression", "changed target/map/horizon cannot reuse evidence", "tests/test_hmc_acceptance_protocol.py"),
    ("live_target_probe", "regression", "exact eager/graph values with current data/constants; changed callbacks, source, geometry and starts remain rejected", "tests/test_hmc_live_target_probe.py"),
    ("health_and_precision", "regression", "nonfinite/divergence/immobility preserved", "tests/test_hmc_acceptance_adversarial.py"),
    ("target_gradient_jacobian", "regression", "independent density/finite-difference mutation oracle", "tests/test_hmc_acceptance_adversarial.py"),
    ("metropolis_reversibility", "regression", "explicit momenta and independent energy/state oracle", "tests/inference_validation/test_kernel_power.py"),
    ("state_space_reference", "regression", "actual Kalman/sigma-point value and total score", "tests/inference_validation/test_ssm_campaign.py"),
    ("model_delivery_and_caps", "scheduled_stress", "verified exports on K0, QR/nonlinear and exact/residual maps; explicit underinformed abstention", "tests/test_hmc_acceptance_model_integrations.py"),
    ("local_acceptance_global_failure", "scheduled_stress", "qualified mixture remains verified while exact mode mass and binary posterior checks expose missed exploration", "tests/test_hmc_acceptance_model_integrations.py"),
    ("supplied_map", "regression", "density/Jacobian/score and identical model starts", "tests/inference_validation/test_residual_funnel_map.py"),
    ("execution_parity", "scheduled_stress", "same explicit momenta, scoped GPU and stable graphs", "tests/inference_validation/test_ssm_xla_full_chain.py"),
    ("trusted_device_parity", "scheduled_stress", "actual CPU/GPU placement, explicit momenta and equal trial-vector decisions", "tests/test_hmc_acceptance_gpu_parity.py"),
    ("posterior_assessment", "scheduled_stress", "independent reference; no acceptance-as-convergence", "tests/inference_validation/test_multimodel_pipeline.py"),
)

MODEL_IDS = ("gaussian", "correlated_gaussian", "lgssm_qr", "nonlinear", "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
             "funnel_exact", "funnel_residual", "funnel_centered", "mixture", "student_t", "cauchy", "positive", "simplex")


def model_configuration(case_id, *, policy_payload, evidence_rungs, seed, wall_seconds,
                        classification, expected_outcome, epsilon_by_l=None):
    """Freeze a case; no implicit evidence allocation or positive-test relabeling."""
    if case_id not in MODEL_IDS:
        raise ValueError("unknown acceptance validation model")
    specs = {
        "gaussian":("gaussian",2,1.35), "correlated_gaussian":("rotated_gaussian",2,.3),
        "lgssm_qr":("ssm_lgssm_qr",2,.3), "nonlinear":("ssm_nonlinear",3,.2),
        "k0":("ssm_campaign_location",1,1.55), "k1":("ssm_campaign_interior",2,.4),
        "k2":("ssm_campaign_near_unit",2,.8), "k3":("ssm_campaign_small_noise",2,.8),
        "k4":("ssm_campaign_two_noises",2,.8), "k5":("ssm_campaign_two_noises_long",2,.8),
        "k6":("ssm_campaign_multivariate",18,.001), "k7":("ssm_campaign_nonlinear",2,.8),
        "funnel_exact":("funnel_noncentered",3,1.25), "funnel_residual":("funnel_noncentered",3,1.),
        "funnel_centered":("funnel",3,.1), "mixture":("mixture",2,1.35),
        "student_t":("student_t",2,.8), "cauchy":("cauchy",2,.8),
        "positive":("gamma",1,.6), "simplex":("dirichlet",2,.6)}
    target, dimension, epsilon = specs[case_id]
    geometry={"kind":"explicit_diagonal","center":[0.]*dimension,"scale":[1.]*dimension}
    parameters, data, route = {}, None, "ordinary"
    if case_id=="correlated_gaussian":
        parameters={"condition":25.,"angle":.6}
    if case_id=="lgssm_qr":
        data=[.18,.05,.16,.11]
        geometry={"kind":"explicit_diagonal","center":[.5,-1.5],"scale":[.5,.4]}
    if case_id=="nonlinear":
        data=[.10,.04,.16]
        geometry={"kind":"explicit_diagonal","center":[.6,.4,.7],"scale":[.1,.1,.1]}
    if case_id=="k0":
        geometry={"kind":"quadratic_location"}
        data=[.4+.03*math.sin(i*.3) for i in range(32)]
    if case_id in {"k1","k2","k3"}:
        rho=.97 if case_id=="k2" else .6
        geometry={"kind":"explicit_diagonal","center":[math.atanh(rho/.999),math.log(.02 if case_id=="k3" else .2)],
                  "scale":[.1,.2] if case_id=="k2" else [.2,.2] if case_id=="k3" else [1.,1.]}
        data=[.1+.04*math.sin(i*.2) for i in range(32)]
    if case_id in {"k4","k5"}:
        geometry={"kind":"explicit_diagonal","center":[math.log(.2)]*2,"scale":[.5]*2}
        data=[.1+.04*math.sin(i*.2) for i in range(16 if case_id=="k4" else 128)]
    if case_id=="k7":
        geometry={"kind":"explicit_diagonal","center":[.7,.8],"scale":[.2,.2]}
        data=[.1+.04*math.sin(i*.2) for i in range(16)]
    if case_id in {"funnel_exact","funnel_residual"}:
        route="fixed_transport"
        geometry={"kind":"exact" if case_id=="funnel_exact" else "residual"}
        parameters={"scale":3.}
    if case_id=="mixture":
        geometry["center"]=[-5.,0.]
        parameters={"separation":5.,"weight":.3}
    return {"schema":"bayesfilter.acceptance_model_config.v1","case_id":classification+"-"+case_id,
        "target":target,"parameters":parameters,"data":data,"route":route,"geometry":geometry,
        "active_starts":[[s]*dimension for s in [-1.,-.3,.4,1.]],
        "epsilon_by_l":epsilon_by_l or [[1,[epsilon]]],"epsilon_domain":[.0001,1.95],
        "repair_factor":1.3,"max_repairs_per_family":0,"policy":dict(policy_payload),
        "evidence_rungs":list(evidence_rungs),"seed":list(seed),"wall_seconds":wall_seconds,
        "expected_outcome":expected_outcome,"classification":classification,
        "provenance":"Frozen diagnostic hypotheses from the October 2 matrix: explicit steps/scales/starts; deterministic synthetic data; K0 uses native quadratic score preparation; supplied funnel maps have no learning claim. Evidence counts and budget are caller-declared. No posterior or default claim."}
