# Bounded reproduction results

```json
{
  "question": "Remaining engineering gaps in the full HMC tuning procedure",
  "git_commit": "9329cadf3296211ccfa9e2539235225e041ad36a",
  "command": [
    "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
    "docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/reproduce_gaps.py"
  ],
  "environment": "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
  "cpu_gpu_status": "CPU diagnostics; GPU devices intentionally hidden",
  "data": "synthetic fixtures only",
  "seeds": "No numerical HMC execution; binding fixture seed (20260914, 11)",
  "plan": "docs/plans/bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md",
  "elapsed_seconds": 3.014891487022396,
  "results": {
    "inconclusive_evidence": {
      "initial": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "audit:one:candidate:000001": "validating"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 18,
        "spent": 2,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      },
      "resume_calls": 0,
      "resumed": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "audit:one:candidate:000001": "validating"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 18,
        "spent": 2,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      }
    },
    "inconclusive_conflict": {
      "initial": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "audit:one:candidate:000001": "validating"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 18,
        "spent": 2,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      },
      "resume_calls": 0,
      "resumed": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "audit:one:candidate:000001": "validating"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 18,
        "spent": 2,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      }
    },
    "repair_factor_below_one": {
      "action": {
        "schema": "bayesfilter.hmc_repair_action.v1",
        "repair_action_id": "audit:one:repair:0001",
        "parent_candidate_id": "audit:one:candidate:000001",
        "child_candidate_id": "audit:one:candidate:000002",
        "candidate_family_id": "audit:one:family:000001",
        "source_verification_hash": "1b60e4bb45e734c6eea7a5bece6c3e6bdf86414b93b2d4de6985998f6ad615dd",
        "old_epsilon": 0.4,
        "new_epsilon": 0.2,
        "exact_l": 3,
        "mass_signature": "identity",
        "direction": "repair_step_higher",
        "execution_status": "executed",
        "verification_status": "passed",
        "qualified_repair_status": "executed_and_verified",
        "not_executed_reason": null,
        "allocation_source": "repair_reserve"
      },
      "replay_error": "higher repair did not increase epsilon"
    },
    "insufficient_candidate_reservation": {
      "initial": {
        "completion_status": "partial_budget",
        "final_status": "partial_budget",
        "states": {
          "audit:one:candidate:000001": "screened"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 19,
        "spent": 1,
        "pending": [
          "audit:one:work:000002"
        ],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      },
      "resume_calls": 0,
      "resumed": {
        "completion_status": "partial_budget",
        "final_status": "partial_budget",
        "states": {
          "audit:one:candidate:000001": "screened"
        },
        "verified": [],
        "viable": [
          "audit:one:candidate:000001"
        ],
        "remaining": 19,
        "spent": 1,
        "pending": [
          "audit:one:work:000002"
        ],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      }
    },
    "revisits_failed_epsilon": {
      "completion_status": "complete",
      "final_status": "complete",
      "states": {
        "audit:one:candidate:000001": "promotion_failed",
        "audit:one:candidate:000002": "promotion_failed",
        "audit:one:candidate:000003": "promotion_failed"
      },
      "verified": [],
      "viable": [],
      "remaining": 14,
      "spent": 6,
      "pending": [],
      "candidate_settings": [
        [
          3,
          0.4
        ],
        [
          3,
          0.8
        ],
        [
          3,
          0.4
        ]
      ],
      "repairs": 2
    },
    "duplicate_exploration_settings": {
      "completion_status": "complete",
      "final_status": "complete",
      "states": {
        "audit:one:candidate:000001": "verified",
        "audit:one:candidate:000002": "verified"
      },
      "verified": [
        "audit:one:candidate:000001",
        "audit:one:candidate:000002"
      ],
      "viable": [
        "audit:one:candidate:000001",
        "audit:one:candidate:000002"
      ],
      "remaining": 16,
      "spent": 4,
      "pending": [],
      "candidate_settings": [
        [
          3,
          0.4
        ],
        [
          3,
          0.4
        ]
      ],
      "repairs": 0
    },
    "lower_step_repair_suppressed": {
      "analysis": {
        "decision": "failed",
        "acceptance": 0.010000000000000005,
        "hard_vetoes": [
          "movement_gate_failed"
        ],
        "acceptance_evidence": {
          "schema": "bayesfilter.hmc_acceptance_evidence.v5",
          "evidence_validity": "valid",
          "acceptance_decision": "repair_step_lower",
          "decision": "repair_step_lower",
          "passed": false,
          "promotion_eligible": false,
          "repair_direction": "lower_epsilon",
          "pooled_mean": 0.010000000000000005,
          "chain_mean_uncertainty_interval": [
            0.010000000000000005,
            0.010000000000000005
          ],
          "chain_mean_uncertainty_method": "two_sided_student_t_independent_chain_means",
          "chain_mean_uncertainty_level": 0.9,
          "chain_means": [
            0.010000000000000005,
            0.010000000000000005,
            0.010000000000000005,
            0.010000000000000005
          ],
          "block_means_by_chain": [
            [
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005
            ],
            [
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005
            ],
            [
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005
            ],
            [
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005,
              0.010000000000000005
            ]
          ],
          "realized_acceptance_rate": 0.0,
          "realized_acceptance_rate_by_chain": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "movement_rate_by_chain": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "repeated_state_fraction_by_chain": [
            1.0,
            1.0,
            1.0,
            1.0
          ],
          "normalized_return_displacement_by_chain": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "path_return_fraction_by_chain": [
            1.0,
            1.0,
            1.0,
            1.0
          ],
          "usable_decisions_per_chain": 64,
          "excluded_remainder_per_chain": 0,
          "native_divergence_status": "not_exposed_by_kernel",
          "native_divergence_count": null,
          "min_log_accept_ratio": -4.605170185988091,
          "max_log_accept_ratio": -4.605170185988091,
          "max_abs_log_accept_energy_proxy": 4.605170185988091,
          "negative_proxy_exceedance_count_by_chain": [
            0,
            0,
            0,
            0
          ],
          "positive_proxy_exceedance_count_by_chain": [
            0,
            0,
            0,
            0
          ],
          "negative_proxy_exceedance_rate_by_chain": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "positive_proxy_exceedance_rate_by_chain": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "policy": {
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
          "engineering_invalidity_reasons": [],
          "candidate_promotion_vetoes": [
            "movement_gate_failed"
          ],
          "tuning_repair_triggers": [
            "step_size:lower_epsilon"
          ],
          "candidate_health_alerts": [],
          "diagnostic_followups": [],
          "cost_stop_reasons": [],
          "cost_stop_scope": null,
          "explanatory_notes": [
            "binary_acceptance_is_explanatory_only"
          ],
          "raw_traces_exposed": false,
          "reports_posterior_convergence": false,
          "compatibility_aliases_non_authoritative": true
        }
      },
      "result": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "bridge-test:search-1:candidate:000001": "promotion_failed"
        },
        "verified": [],
        "viable": [],
        "remaining": 19,
        "spent": 1,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ]
        ],
        "repairs": 0
      }
    },
    "lost_measurement_explanation": {
      "has_failure_reason": false,
      "has_acceptance_evidence": false,
      "verification_receipts": 0
    },
    "shared_invalidity_loses_scope": {
      "analysis": {
        "decision": "failed",
        "acceptance": null,
        "hard_vetoes": [
          "nonfinite_state",
          "nonfinite_proposal_displacement",
          "metropolis_state_mismatch",
          "nonfinite_retained_samples"
        ],
        "acceptance_evidence": {
          "schema": "bayesfilter.hmc_acceptance_evidence.v5",
          "evidence_validity": "shared_execution_invalid",
          "acceptance_decision": "unavailable",
          "decision": "unavailable",
          "passed": false,
          "promotion_eligible": false,
          "repair_direction": null,
          "pooled_mean": null,
          "chain_mean_uncertainty_interval": null,
          "chain_mean_uncertainty_method": null,
          "chain_mean_uncertainty_level": null,
          "chain_means": [],
          "block_means_by_chain": [],
          "realized_acceptance_rate": null,
          "realized_acceptance_rate_by_chain": [],
          "movement_rate_by_chain": [],
          "repeated_state_fraction_by_chain": [],
          "normalized_return_displacement_by_chain": [],
          "path_return_fraction_by_chain": [],
          "usable_decisions_per_chain": 0,
          "excluded_remainder_per_chain": 0,
          "native_divergence_status": "not_exposed_by_kernel",
          "native_divergence_count": null,
          "min_log_accept_ratio": null,
          "max_log_accept_ratio": null,
          "max_abs_log_accept_energy_proxy": null,
          "negative_proxy_exceedance_count_by_chain": [],
          "positive_proxy_exceedance_count_by_chain": [],
          "negative_proxy_exceedance_rate_by_chain": [],
          "positive_proxy_exceedance_rate_by_chain": [],
          "policy": {
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
          "engineering_invalidity_reasons": [
            "nonfinite_retained_samples"
          ],
          "candidate_promotion_vetoes": [],
          "tuning_repair_triggers": [],
          "candidate_health_alerts": [],
          "diagnostic_followups": [],
          "cost_stop_reasons": [],
          "cost_stop_scope": null,
          "explanatory_notes": [],
          "raw_traces_exposed": false,
          "reports_posterior_convergence": false,
          "compatibility_aliases_non_authoritative": true
        }
      },
      "result": {
        "completion_status": "complete",
        "final_status": "complete",
        "states": {
          "bridge-test:search-1:candidate:000001": "promotion_failed",
          "bridge-test:search-1:candidate:000002": "verified"
        },
        "verified": [
          "bridge-test:search-1:candidate:000002"
        ],
        "viable": [
          "bridge-test:search-1:candidate:000002"
        ],
        "remaining": 17,
        "spent": 3,
        "pending": [],
        "candidate_settings": [
          [
            3,
            0.4
          ],
          [
            5,
            0.4
          ]
        ],
        "repairs": 0
      }
    },
    "framework_resource_failure": {
      "error_type": "ResourceExhaustedError",
      "error": "injected CPU audit resource failure",
      "result_written": false,
      "saved_evidence": 0
    },
    "reporting_rhat_failure": {
      "error_type": "RuntimeError",
      "error": "injected reporting diagnostic failure",
      "result_written": false,
      "saved_evidence": 0
    },
    "existing_result_overwritten": {
      "overwritten": true,
      "path": "/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/fault-output/overwrite/candidate_set_result.json"
    },
    "ordinary_public_dispatch": {
      "omitted_config": "legacy_executor",
      "ordinary_config": "legacy_executor"
    }
  },
  "limitations": "Fault injection/controller fixtures prove boundary behavior, not real-target incidence or sampler validity"
}
```
