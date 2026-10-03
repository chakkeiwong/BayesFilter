# Actual oversized-step Gaussian counterexample

```json
{
  "commit": "9329cadf3296211ccfa9e2539235225e041ad36a",
  "command": [
    "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
    "docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/gaussian_oversized_step.py"
  ],
  "environment": "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
  "device": "CPU; GPUs intentionally hidden; non-XLA engineering exception",
  "target": "two-dimensional standard Gaussian; no observed data",
  "root_seed": [
    20260914,
    11
  ],
  "exact_config": {
    "measurement_num_results": 64,
    "verification_num_results": 64,
    "num_warmup_steps": 0,
    "seed": [
      20260914,
      11
    ],
    "acceptance_policy": {
      "schema": "bayesfilter.hmc_acceptance_policy.v5",
      "target": 0.7,
      "practical_region": [
        0.65,
        0.75
      ],
      "repair_region": [
        0.55,
        0.85
      ],
      "chain_count": 4,
      "block_count": 4,
      "min_block_size": 16,
      "min_decisions_per_chain": 64,
      "confidence_level": 0.9,
      "min_movement_rate": 0.05,
      "max_repeated_state_fraction": 0.95,
      "min_normalized_return_displacement": 0.0001,
      "path_return_contract": {
        "lags": [
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16
        ],
        "aggregation": "maximum_recurrence_fraction_over_lags_per_chain",
        "minimum_repetitions_at_minimum_evidence": 4,
        "max_fraction": 0.95,
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-10,
        "chain_rule": "veto_if_any_chain_exceeds_max_fraction"
      },
      "max_abs_log_accept_energy_proxy": 1000.0,
      "allowed_cost_stop_reasons": [],
      "dependence_unit": "independently_seeded_chain_mean",
      "uncertainty_method": "two_sided_student_t_independent_chain_means",
      "tuning_decision_role": "working_tuning_compatibility_interval_not_convergence_or_equivalence",
      "diagnostic_roles": {
        "mean_acceptance_probability": "promotion_criterion_and_repair_trigger",
        "movement": "promotion_veto_and_trajectory_repair_trigger",
        "bounded_short_cycle_path_return": "promotion_veto_and_resonance_repair_trigger",
        "native_divergence": "promotion_veto",
        "max_abs_log_accept_energy_proxy": "explanatory_alert_only",
        "signed_log_accept_ratio_tails": "explanatory_alert_only"
      }
    },
    "target_status_trace_policy": "per_chain_step",
    "use_xla": false,
    "non_xla_reason": "CPU numerical reference and persistence regression",
    "chain_mode": "serial"
  },
  "observations": [
    {
      "candidate": {
        "schema": "bayesfilter.hmc_candidate_record.v1",
        "candidate_id": "bridge-test:search-1:candidate:000001",
        "candidate_family_id": "bridge-test:search-1:family:000001",
        "parent_candidate_id": null,
        "creation_ordinal": 1,
        "scope_id": "bridge-test",
        "search_id": "search-1",
        "leapfrog_steps": 3,
        "epsilon": 3.0,
        "target_signature": "dbd7b69f2dac6f85f46806f5baabe5fec98b3615549fc3111f81510a45e03a79",
        "mass_signature": "c7c60ed2cd040983a167b26f14e6e59107a3ff70eb6fcaa133faa217ae176074",
        "coordinate_system": "ordinary",
        "start_bank_signature": "b6fb5bc4d954ef040724128d08bdf8cb2ace7fd95bdead7015903052a0018e62",
        "warmup_protocol": "766e6578a4da70ae1edf7836477aeb811520b4e76b479fac9d29e817bfb80bfb",
        "backend": "tensorflow_probability",
        "dtype": "float64",
        "execution_mode": "tf_function",
        "adapter_signature": "b453e9b9ea00d4b5b8b092a551778d874a0d8d98b7f5276293a98fe46de07762",
        "source_dependency_hash": "cf727f7db1bb4329f096d591b02b157c1007312b1b0bf51be3fc8792a499a240",
        "target_preparation_identity": "45704ccdf43524a6786d5303050b59de0a6ed47b62ffdbaa9b1c7e1163b9f6be",
        "transition_identity": "e0ff32b034964e06f6b77cec1ec34001d672f83bc6e9193e844cec0c69beee6e",
        "candidate_record_hash": "2b1e49b209844906744b8796362a5f7841390d4495025fbebd283740601ea738",
        "use_xla": false
      },
      "seed": [
        1375085941,
        1507755847
      ],
      "decision": "failed",
      "policy_decision": "repair_step_lower",
      "acceptance": 3.191608222971231e-297,
      "hard_vetoes": [
        "movement_gate_failed"
      ],
      "rhat": {
        "passed": false,
        "rhat_definition": "max(rank-normalized split R-hat, folded rank-normalized split R-hat)",
        "max_rank_normalized_split_rhat": 3.588655259038676e+31,
        "max_folded_rank_normalized_split_rhat": 5.8729806392988565e+31,
        "max_finite_rhat": 5.8729806392988565e+31,
        "finite_rhat_count": 1,
        "nonfinite_rhat_count": 1
      }
    }
  ],
  "plan": "docs/plans/bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md",
  "final_status": "complete",
  "repairs": 0,
  "states": {
    "bridge-test:search-1:candidate:000001": "promotion_failed"
  },
  "wall_seconds": 4.7833049219916575
}
```
