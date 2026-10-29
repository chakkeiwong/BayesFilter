"""Bounded controlled-SMC/KDM supervisor; no numerical implementation here.

Reuses the existing replication supervisor's locks, process identity, atomic
manifests, hashes, accounting and recovery. The old scientific ledger is intact.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import run_replication_master as shared

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
HISTORICAL_EXECUTION_PLAN = 'docs/plans/controlled-smc-kdm-execution-20260929.md'
MASTER = 'docs/plans/controlled-smc-kdm-replication-master-program-20260928.md'
REPLICATION_FIRST_PLAN = 'docs/plans/controlled-smc-kdm-replication-first-execution-20261003.md'
PLAN = REPLICATION_FIRST_PLAN
C0_SOURCE_LOCK_PLAN = 'docs/plans/controlled-smc-kdm-c0-source-lock-20261003.md'
C1_SOURCE_PILOT_PLAN = 'docs/plans/controlled-smc-kdm-c1-source-cost-pilot-20261003.md'
C1_FULL_SOURCE_PLAN = 'docs/plans/controlled-smc-kdm-c1-full-source-20261003.md'
SOURCE_MEMORY_REPAIR_PLAN = 'docs/plans/controlled-smc-kdm-source-memory-repair-20261004.md'
SOURCE_RUNNER = 'docs/benchmarks/run_controlled_smc_author_source.py'
SOURCE_FAILURE_REPLAY_PLAN = 'docs/plans/controlled-smc-lorenz-source-failure-replay-20261004.md'
SOURCE_FAILURE_EVIDENCE = CAMPAIGN/'source-continuation-review-20261004-01/lorenz-source-failure'
C2_SOURCE_RESULTS_PLAN = 'docs/plans/controlled-smc-kdm-source-continuation-repair-20261004.md'
FULL_SOURCE_SPEC = str(CAMPAIGN.relative_to(ROOT)/'full-source-spec-20261003.json')
K0_SOURCE_CONFIG_PLAN = 'docs/plans/controlled-smc-kdm-k0-source-config-20261003.md'
P0_FORWARD_PLAN = 'docs/plans/controlled-smc-p0-forward-replay-20261004.md'
P0_FITTING_PLAN = 'docs/plans/controlled-smc-p0-fitting-forward-20261005.md'
P0_SOLVER_PLAN = 'docs/plans/controlled-smc-p0-solver-semantics-20261005.md'
P0_PERTURB_PLAN = 'docs/plans/controlled-smc-perturbation-experiment-20261006.md'
P0_OFFSET_PLAN = 'docs/plans/controlled-smc-offset-localization-20261006.md'
P0_TF_SOLVER_PLAN = 'docs/plans/controlled-smc-tf-solver-compatibility-20261006.md'
P0_TF_SOLVER_PHASES = ('p0_tf_solver_compatibility','p0_tf_solver_cpu_reference')
P0_PERTURB_PHASES = ('p0_perturb_source','p0_perturb_tf')
P0_FORWARD_PHASES = ('p0_forward_capture','p0_forward_compare','p0_fitting_forward','p0_solver_semantics','p0_solver_system_reference','p0_offset_localization') + P0_PERTURB_PHASES + P0_TF_SOLVER_PHASES
ACTIVE_PHASES = ('c0_source_lock','c1_source_cost_pilot','c1_source_isolation_check','c1_source_full','c1_source_failure_replay','c2_source_results','k0_source_config') + P0_FORWARD_PHASES
PYTHON = '/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
AUTHOR = '.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience/'
REFIT_PLAN = 'docs/plans/controlled-smc-rbf-full-refit-20260930.md'
REFIT_PHASES = ('c2_rbf_full_refit','c2_rbf_refit_pilot','c2_rbf_refit_precision')
LORENZ_PRECISION_PLAN = 'docs/plans/controlled-smc-lorenz-precision-repair-20260930.md'
LORENZ_BATCH_PLAN = 'docs/plans/controlled-smc-lorenz-batch-repair-20260930.md'
LORENZ_PILOT_PLAN = 'docs/plans/controlled-smc-lorenz-figure6-pilot-20260930.md'
LORENZ_REFINEMENT_PLAN = 'docs/plans/controlled-smc-lorenz-refinement-validation-20260930.md'
LORENZ_REFINEMENT_PHASES = ('c1_lorenz_refinement_reference','c2_lorenz_refinement_validation','c1_lorenz_refinement_diagnosis','c1_lorenz_refinement_matched_validation','c2_lorenz_figure6_pilot','c1_lorenz_precision_diagnosis','c2_lorenz_precision_safety','c1_lorenz_trajectory_diagnosis','c2_lorenz_covariance_safety','c1_lorenz_covariance_validation','c2_lorenz_rank_guard','c1_lorenz_covariance_localization','c2_lorenz_batch_validation','c2_lorenz_batch_pilot')
LORENZ_SOURCE_PORT_PLAN = 'docs/plans/controlled-smc-lorenz-source-port-localization-20261001.md'
LORENZ_SOURCE_PORT_PHASES = ('c1_lorenz_author_capture','c2_lorenz_source_port')
LORENZ_MATRIX_PLAN = 'docs/plans/controlled-smc-lorenz-matrix-crossover-20261001.md'
LORENZ_MATRIX_PHASES = ('c1_lorenz_matrix_reference','c2_lorenz_matrix_crossover','c1_lorenz_matrix_crosssolve','c1_lorenz_matrix_fallback','c1_lorenz_failed_source_replay')
LORENZ_BOUNDED_PLAN = 'docs/plans/controlled-smc-lorenz-bounded-spd-20261001.md'
LORENZ_REPAIR_PLAN = 'docs/plans/controlled-smc-lorenz-repair-execution-20261002.md'
LORENZ_REPAIR_PHASES = ('c1_lorenz_recursion_localization','c2_lorenz_first_failure','c2_lorenz_recursion_capture','c1_lorenz_recursion_sensitivity','c2_lorenz_repair_validation','c2_lorenz_absolute_bellman','c2_lorenz_failure_rows')
LORENZ_ROOT_CAUSE_PLAN = 'docs/plans/controlled-smc-lorenz-root-cause-audit-20261002.md'
LORENZ_ROOT_CAUSE_PHASES = ('c1_lorenz_root_cause_audit','c1_lorenz_root_cause_attribution')
LORENZ_BOUNDED_PHASES = ('c1_lorenz_bounded_reference','c2_lorenz_bounded_calibration','c2_lorenz_bounded_eigen_calibration','c2_lorenz_bounded_refined_calibration','c2_lorenz_bounded_safety','c1_lorenz_bounded_validation','c1_lorenz_bounded_localization','c1_lorenz_bounded_integral_reference','c2_lorenz_bounded_integral_calibration','c2_lorenz_bounded_integral_safety','c1_lorenz_bounded_integral_validation','c1_lorenz_bounded_integral_localization','c2_lorenz_bounded_crossover_capture','c1_lorenz_bounded_crossover_reference','c1_lorenz_bounded_reference_arithmetic','c1_lorenz_bounded_condition_reference','c2_lorenz_bounded_condition_calibration','c2_lorenz_bounded_condition_safety','c1_lorenz_bounded_condition_validation', 'c1_lorenz_bounded_trigger_reference','c2_lorenz_bounded_trigger_calibration','c2_lorenz_bounded_trigger_safety', 'c1_lorenz_bounded_trigger_validation','c1_lorenz_bounded_trigger_history','c1_lorenz_bounded_trigger_transition','c1_lorenz_bounded_margin_reference','c2_lorenz_bounded_margin_calibration','c2_lorenz_bounded_margin_safety','c1_lorenz_bounded_margin_validation')
PHASES = {
    'p0_tf_solver_compatibility': dict(resource='gpu',limit=180,total_limit=360,max_attempts=3,script='docs/benchmarks/diagnose_lorenz_solver_compatibility_tf.py',mode='forced_source_branch_solver_diagnostic',dependencies=['p0_offset_localization']),
    'p0_tf_solver_cpu_reference': dict(resource='cpu',limit=180,total_limit=180,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_solver_compatibility_tf.py',mode='forced_source_branch_cpu_graph_reference',dependencies=['p0_tf_solver_compatibility']),
    'p0_offset_localization': dict(resource='gpu',limit=300,total_limit=600,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_offset_tf.py',mode='source_target_solver_and_policy_swap_diagnostic',dependencies=['p0_perturb_tf','p0_fitting_forward']),
    'p0_perturb_source': dict(resource='cpu',limit=1200,total_limit=1500,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_random_perturbation.py',mode='source_perturbation',dependencies=['p0_forward_capture','c0_source_lock']),
    'p0_perturb_tf': dict(resource='gpu',limit=1800,total_limit=2750,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_random_perturbation.py',mode='tf_perturbation',dependencies=['p0_perturb_source']),
    'p0_solver_system_reference': dict(resource='cpu',limit=90,total_limit=90,max_attempts=1,script='docs/benchmarks/reference_lorenz_solver_semantics.py',mode='conditional_solver_system_reference',dependencies=['p0_solver_semantics']),
    'p0_solver_semantics': dict(resource='cpu',limit=90,total_limit=90,max_attempts=1,script='docs/benchmarks/reference_lorenz_solver_semantics.py',mode='conditional_solver_reference',dependencies=['p0_fitting_forward','c1_lorenz_matrix_fallback']),
    'p0_fitting_forward': dict(resource='gpu',limit=600,total_limit=900,max_attempts=2,script='docs/benchmarks/compare_lorenz_fitting_forward_tf.py',mode='fitting_forward_comparison',dependencies=['p0_forward_capture','p0_forward_compare']),
    'p0_forward_capture': dict(resource='cpu',limit=90,total_limit=90,max_attempts=1,script='docs/benchmarks/capture_lorenz_forward_reference.py',mode='fixed_policy_forward_reference',dependencies=['c0_source_lock','c1_lorenz_author_native','c1_source_failure_replay']),
    'p0_forward_compare': dict(resource='gpu',limit=600,total_limit=600,max_attempts=1,script='docs/benchmarks/compare_lorenz_forward_tf.py',mode='fixed_policy_forward_comparison',dependencies=['p0_forward_capture','c1_source_failure_replay']),
    'c1_source_failure_replay': dict(resource='cpu',limit=60,total_limit=60,max_attempts=1,script='docs/benchmarks/replay_controlled_smc_source_failure.py',mode='source_failure_replay',dependencies=['c0_source_lock','c1_source_isolation_check']),
    'c1_source_isolation_check': dict(resource='cpu',limit=600,total_limit=600,max_attempts=2,script='docs/benchmarks/validate_controlled_smc_source_isolation.py',mode='source_isolation',dependencies=['c0_source_lock']),
    'c2_source_results': dict(resource='cpu',limit=600,total_limit=600,max_attempts=1,script='docs/benchmarks/analyze_controlled_smc_author_results.py',mode='source_results',dependencies=['c1_source_full']),
    'k0_source_config': dict(resource='cpu',limit=20,total_limit=20,max_attempts=1,script='docs/benchmarks/lock_kdm_source_config.py',mode='metadata',dependencies=[]),
    'c1_source_full': dict(resource='cpu',limit=259200,total_limit=259200,max_attempts=2,script='docs/benchmarks/run_controlled_smc_author_source.py',mode='full',dependencies=['c0_source_lock','c1_source_cost_pilot']),
    'c1_source_cost_pilot': dict(resource='cpu',limit=205,total_limit=210,max_attempts=2,script='docs/benchmarks/run_controlled_smc_author_source.py',mode='pilot',dependencies=['c0_source_lock']),
    'c0_source_lock': dict(resource='cpu',limit=30,total_limit=30,max_attempts=2,script='docs/benchmarks/lock_controlled_smc_source.py',mode='source_lock',dependencies=[]),
    'c2_lorenz_failure_rows': dict(resource='gpu',limit=90,total_limit=90,max_attempts=2,script='docs/benchmarks/validate_lorenz_failure_rows_tf.py',mode='diagnostic',dependencies=['c2_lorenz_absolute_bellman']),
    'c2_lorenz_absolute_bellman': dict(resource='gpu',limit=90,total_limit=90,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_absolute_bellman_tf.py',mode='diagnostic',dependencies=['c2_lorenz_repair_validation']),
    'c2_lorenz_repair_validation': dict(resource='gpu',limit=120,total_limit=120,max_attempts=1,script='docs/benchmarks/validate_lorenz_repair_acceptance_tf.py',mode='diagnostic',dependencies=['c2_lorenz_recursion_capture','c1_lorenz_recursion_sensitivity']),
    'c1_lorenz_recursion_sensitivity': dict(resource='cpu',limit=20,total_limit=25,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_recursion_sensitivity.py',mode='diagnostic',dependencies=['c1_lorenz_recursion_localization']),
    'c2_lorenz_recursion_capture': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/capture_lorenz_recursion_repair_tf.py',mode='diagnostic',dependencies=['c1_lorenz_recursion_localization','c2_lorenz_first_failure']),
    'c1_lorenz_recursion_localization': dict(resource='cpu',limit=15,total_limit=25,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_recursion_localization.py',mode='diagnostic',dependencies=['c1_lorenz_root_cause_attribution']),
    'c2_lorenz_first_failure': dict(resource='gpu',limit=60,total_limit=120,max_attempts=2,script='docs/benchmarks/validate_lorenz_first_failure_tf.py',mode='diagnostic',dependencies=['c1_lorenz_root_cause_attribution']),
    'c1_lorenz_root_cause_attribution': dict(resource='cpu',limit=15,total_limit=25,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_target_attribution.py',mode='diagnostic',dependencies=['c1_lorenz_root_cause_audit']),
    'c1_lorenz_root_cause_audit': dict(resource='cpu',limit=30,total_limit=30,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_policy_geometry.py',mode='diagnostic',dependencies=['c2_lorenz_bounded_trigger_safety','c2_lorenz_bounded_margin_safety']),
    'c1_lorenz_bounded_margin_reference': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='integral_reference',dependencies=['c1_lorenz_bounded_trigger_transition']),
    'c2_lorenz_bounded_margin_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='integral_calibration',dependencies=['c1_lorenz_bounded_margin_reference']),
    'c2_lorenz_bounded_margin_safety': dict(resource='gpu',limit=900,total_limit=1500,max_attempts=2,script='docs/benchmarks/validate_lorenz_precision_safety_tf.py',mode='safety',dependencies=['c2_lorenz_bounded_margin_calibration']),
    'c1_lorenz_bounded_margin_validation': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='validation',dependencies=['c2_lorenz_bounded_margin_safety']),
    'c1_lorenz_bounded_trigger_transition': dict(resource='cpu',limit=12,total_limit=18,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_safety_history.py',mode='transition',dependencies=['c1_lorenz_bounded_trigger_history']),
    'c1_lorenz_bounded_trigger_history': dict(resource='cpu',limit=10,total_limit=15,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_safety_history.py',mode='history',dependencies=['c2_lorenz_bounded_trigger_safety']),
    'c1_lorenz_bounded_trigger_reference': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='integral_reference',dependencies=['c1_lorenz_bounded_condition_validation']),
    'c2_lorenz_bounded_trigger_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='integral_calibration',dependencies=['c1_lorenz_bounded_trigger_reference']),
    'c2_lorenz_bounded_trigger_safety': dict(resource='gpu',limit=900,total_limit=1500,max_attempts=2,script='docs/benchmarks/validate_lorenz_precision_safety_tf.py',mode='safety',dependencies=['c2_lorenz_bounded_trigger_calibration']),
    'c1_lorenz_bounded_trigger_validation': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='validation',dependencies=['c2_lorenz_bounded_trigger_safety']),
    'c1_lorenz_bounded_condition_reference': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='integral_reference',dependencies=['c1_lorenz_bounded_reference_arithmetic']),
    'c2_lorenz_bounded_condition_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='integral_calibration',dependencies=['c1_lorenz_bounded_condition_reference']),
    'c2_lorenz_bounded_condition_safety': dict(resource='gpu',limit=900,total_limit=1500,max_attempts=2,script='docs/benchmarks/validate_lorenz_precision_safety_tf.py',mode='safety',dependencies=['c2_lorenz_bounded_condition_calibration']),
    'c1_lorenz_bounded_condition_validation': dict(resource='cpu',limit=30,total_limit=40,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='validation',dependencies=['c2_lorenz_bounded_condition_safety']),
    'c1_lorenz_bounded_reference_arithmetic': dict(resource='cpu',limit=30,total_limit=45,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_bounded_crossover_reference.py',mode='crossover',dependencies=['c1_lorenz_bounded_crossover_reference']),
    'c2_lorenz_bounded_crossover_capture': dict(resource='gpu',limit=120,total_limit=120,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_bounded_crossover_tf.py',mode='capture',dependencies=['c1_lorenz_bounded_integral_localization']),
    'c1_lorenz_bounded_crossover_reference': dict(resource='cpu',limit=30,total_limit=30,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_bounded_crossover_reference.py',mode='crossover',dependencies=['c2_lorenz_bounded_crossover_capture']),
    'c1_lorenz_bounded_integral_localization': dict(resource='cpu',limit=30,total_limit=30,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_covariance_localization.py',mode='localization',dependencies=['c1_lorenz_bounded_integral_validation']),
    'c1_lorenz_bounded_integral_reference': dict(resource='cpu',limit=30,total_limit=45,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='integral_reference',dependencies=['c1_lorenz_bounded_localization']),
    'c2_lorenz_bounded_integral_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='integral_calibration',dependencies=['c1_lorenz_bounded_integral_reference']),
    'c2_lorenz_bounded_integral_safety': dict(resource='gpu',limit=900,total_limit=1500,max_attempts=2,script='docs/benchmarks/validate_lorenz_precision_safety_tf.py',mode='safety',dependencies=['c2_lorenz_bounded_integral_calibration']),
    'c1_lorenz_bounded_integral_validation': dict(resource='cpu',limit=45,total_limit=45,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='validation',dependencies=['c2_lorenz_bounded_integral_safety']),
    'c1_lorenz_bounded_localization': dict(resource='cpu',limit=30,total_limit=30,max_attempts=1,script='docs/benchmarks/diagnose_lorenz_covariance_localization.py',mode='localization',dependencies=['c1_lorenz_bounded_validation']),
    'c2_lorenz_bounded_refined_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='refined_calibration',dependencies=['c2_lorenz_bounded_eigen_calibration']),
    'c2_lorenz_bounded_eigen_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='strict_calibration',dependencies=['c2_lorenz_bounded_calibration']),
    'c1_lorenz_bounded_reference': dict(resource='cpu',limit=30,total_limit=45,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='reference',dependencies=['c1_lorenz_failed_source_replay']),
    'c2_lorenz_bounded_calibration': dict(resource='gpu',limit=120,total_limit=180,max_attempts=2,script='docs/benchmarks/validate_lorenz_bounded_calibration_tf.py',mode='calibration',dependencies=['c1_lorenz_bounded_reference']),
    'c2_lorenz_bounded_safety': dict(resource='gpu',limit=900,total_limit=1500,max_attempts=2,script='docs/benchmarks/validate_lorenz_precision_safety_tf.py',mode='safety',dependencies=['c2_lorenz_bounded_refined_calibration']),
    'c1_lorenz_bounded_validation': dict(resource='cpu',limit=45,total_limit=45,max_attempts=2,script='docs/benchmarks/reference_lorenz_bounded_spd.py',mode='validation',dependencies=['c2_lorenz_bounded_safety']),
    'c1_lorenz_failed_source_replay': dict(resource='cpu',limit=45,total_limit=45,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_failed_source_replay.py',mode='failed_replay',dependencies=['c1_lorenz_matrix_fallback','c1_lorenz_author_native']),
    'c1_lorenz_matrix_fallback': dict(resource='cpu',limit=20,total_limit=20,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_matrix_reference.py',mode='fallback',dependencies=['c2_lorenz_matrix_crossover']),
    'c1_lorenz_matrix_reference': dict(resource='cpu',limit=45,total_limit=45,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_matrix_reference.py',mode='form',dependencies=['c1_lorenz_author_capture']),
    'c2_lorenz_matrix_crossover': dict(resource='gpu',limit=600,total_limit=600,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_matrix_crossover_tf.py',mode='cross',dependencies=['c1_lorenz_matrix_reference']),
    'c1_lorenz_matrix_crosssolve': dict(resource='cpu',limit=45,total_limit=45,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_matrix_reference.py',mode='solve',dependencies=['c2_lorenz_matrix_crossover']),
    'c1_lorenz_author_capture': dict(resource='cpu',limit=90,total_limit=90,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_author_capture.py',mode='capture',dependencies=['c1_lorenz_author_native']),
    'c2_lorenz_source_port': dict(resource='gpu',limit=900,total_limit=900,max_attempts=2,script='docs/benchmarks/diagnose_lorenz_source_port_tf.py',mode='localize',dependencies=['c1_lorenz_author_capture']),
    'c1_lorenz_author_native': dict(resource='cpu', limit=90, total_limit=90, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_author_native.py', mode='lorenz_author_native', dependencies=[]),
    'c1_lorenz_covariance_localization': dict(resource='cpu', limit=30, total_limit=30, max_attempts=1, script='docs/benchmarks/diagnose_lorenz_covariance_localization.py', mode='lorenz_covariance_localization', dependencies=['c2_lorenz_rank_guard']),
    'c2_lorenz_rank_guard': dict(resource='gpu', limit=240, total_limit=240, max_attempts=1, script='docs/benchmarks/validate_lorenz_rank_guard_tf.py', mode='lorenz_rank_guard', dependencies=['c2_lorenz_covariance_safety']),
    'c1_lorenz_covariance_validation': dict(resource='cpu', limit=60, total_limit=60, max_attempts=2, script='docs/benchmarks/validate_lorenz_covariance_reference.py', mode='lorenz_covariance_validation', dependencies=['c2_lorenz_covariance_safety']),
    'c2_lorenz_covariance_safety': dict(resource='gpu', limit=512, total_limit=512, max_attempts=1, script='docs/benchmarks/validate_lorenz_precision_safety_tf.py', mode='lorenz_covariance_safety', dependencies=['c1_lorenz_trajectory_diagnosis']),
    'c1_lorenz_trajectory_diagnosis': dict(resource='cpu', limit=45, total_limit=45, max_attempts=1, script='docs/benchmarks/diagnose_lorenz_trajectory.py', mode='lorenz_trajectory_diagnosis', dependencies=['c2_lorenz_precision_safety']),
    'c2_lorenz_precision_safety': dict(resource='gpu', limit=600, total_limit=600, max_attempts=2, script='docs/benchmarks/validate_lorenz_precision_safety_tf.py', mode='lorenz_precision_safety', dependencies=['c1_lorenz_precision_diagnosis']),
    'c1_lorenz_precision_diagnosis': dict(resource='cpu', limit=45, total_limit=45, max_attempts=1, script='docs/benchmarks/diagnose_lorenz_precision.py', mode='lorenz_precision_diagnosis', dependencies=['c2_lorenz_figure6_pilot']),
    'c2_lorenz_batch_validation': dict(resource='gpu', limit=450, total_limit=450, max_attempts=4, script='docs/benchmarks/validate_lorenz_batch_tf.py', mode='validate', dependencies=['c2_lorenz_figure6_pilot','c1_lorenz_covariance_validation']),
    'c2_lorenz_batch_pilot': dict(resource='gpu', limit=450, total_limit=450, max_attempts=2, script='docs/benchmarks/validate_lorenz_batch_tf.py', mode='pilot', dependencies=['c2_lorenz_batch_validation']),
    'c2_lorenz_figure6_pilot': dict(resource='gpu', limit=450, total_limit=450, max_attempts=2, script='docs/benchmarks/run_lorenz_figure6_pilot_tf.py', mode='lorenz_figure6_pilot', dependencies=['c1_lorenz_refinement_matched_validation']),
    'c1_lorenz_refinement_matched_validation': dict(resource='cpu', limit=45, total_limit=45, max_attempts=1, script='docs/benchmarks/validate_lorenz_refinement_matched.py', mode='lorenz_refinement_matched_validation', dependencies=['c1_lorenz_refinement_diagnosis']),
    'c1_lorenz_refinement_diagnosis': dict(resource='cpu', limit=45, total_limit=45, max_attempts=1, script='docs/benchmarks/diagnose_lorenz_refinement_drift.py', mode='lorenz_refinement_diagnosis', dependencies=['c1_lorenz_refinement_reference']),
    'c1_lorenz_refinement_reference': dict(resource='cpu', limit=180, total_limit=180, max_attempts=2, script='docs/benchmarks/reference_lorenz_refinement.py', mode='lorenz_refinement_reference', dependencies=['c2_lorenz_centered_validation']),
    'c2_lorenz_refinement_validation': dict(resource='gpu', limit=450, total_limit=450, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_refinement_tf.py', mode='lorenz_refinement_validation', dependencies=['c1_lorenz_refinement_reference']),
    'c2_rbf_full_refit': dict(resource='gpu', limit=900, total_limit=900, max_attempts=1, script='docs/benchmarks/diagnose_rbf_full_refit_tf.py', mode='rbf_full_refit', dependencies=['c2_rbf_full_retention_precision']),
    'c2_rbf_refit_pilot': dict(resource='gpu', limit=300, total_limit=300, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_refit_pilot', dependencies=['c2_rbf_full_refit']),
    'c2_rbf_refit_precision': dict(resource='gpu', limit=1500, total_limit=1500, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_refit_precision', dependencies=['c2_rbf_refit_pilot']),
    'c2_rbf_full_retention_precision': dict(resource='gpu', limit=2400, total_limit=2400, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_full_retention_precision', dependencies=['c2_rbf_full_retention_pilot']),
    'c2_rbf_full_retention_pilot': dict(resource='gpu', limit=300, total_limit=300, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_full_retention_pilot', dependencies=['c2_rbf_coverage_mechanism']),
    'c2_rbf_coverage_mechanism': dict(resource='gpu', limit=180, total_limit=180, max_attempts=1, script='docs/benchmarks/diagnose_rbf_coverage_mechanism_tf.py', mode='rbf_coverage_mechanism', dependencies=['c2_rbf_policy_coverage']),
    'c2_rbf_policy_coverage': dict(resource='gpu', limit=180, total_limit=180, max_attempts=1, script='docs/benchmarks/diagnose_rbf_policy_coverage_tf.py', mode='rbf_policy_coverage', dependencies=['c2_rbf_fixed_policy_precision']),
    'c2_rbf_fixed_policy_precision': dict(resource='gpu', limit=900, total_limit=900, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_fixed_policy_precision', dependencies=['c2_rbf_fixed_policy_sanity']),
    'c2_rbf_fixed_policy_sanity': dict(resource='gpu', limit=900, total_limit=900, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fixed_policy_tf.py', mode='rbf_fixed_policy_sanity', dependencies=['c1_rbf_paired_diagnosis']),
    'c1_rbf_paired_diagnosis': dict(resource='cpu', limit=10, total_limit=10, max_attempts=1, script='docs/benchmarks/diagnose_rbf_truncation.py', mode='rbf_paired_first_divergence', dependencies=['c1_rbf_paired_reference']),
    'c1_rbf_paired_reference': dict(resource='cpu', limit=39, total_limit=39, max_attempts=1, script='docs/benchmarks/reference_rbf_scipy.py', mode='rbf_paired_reference', dependencies=['c1_rbf_matched_solver']),
    'c2_rbf_paired_validation': dict(resource='gpu', limit=900, total_limit=900, max_attempts=1, script='docs/benchmarks/diagnose_rbf_fit_tf.py', mode='rbf_paired_validation', dependencies=['c1_rbf_paired_reference']),
    'c1_rbf_matched_solver': dict(resource='cpu', limit=20, total_limit=20, max_attempts=1, script='docs/benchmarks/diagnose_rbf_matched_solver.py', mode='rbf_matched_solver', dependencies=['c1_rbf_truncation_diagnosis']),
    'c1_rbf_coefficient_stability': dict(resource='cpu', limit=20, total_limit=20, max_attempts=1, script='docs/benchmarks/diagnose_rbf_coefficient_stability.py', mode='rbf_coefficient_stability', dependencies=['c1_rbf_truncation_diagnosis']),
    'c2_rbf_policy_accuracy': dict(resource='gpu', limit=180, total_limit=360, max_attempts=2, script='docs/benchmarks/diagnose_rbf_fit_tf.py', mode='rbf_policy_accuracy', dependencies=['c1_rbf_truncation_diagnosis']),
    'c1_rbf_truncation_diagnosis': dict(resource='cpu', limit=20, total_limit=20, max_attempts=2, script='docs/benchmarks/diagnose_rbf_truncation.py', mode='rbf_first_divergence', dependencies=['c1_rbf_svd_reference']),
    'c1_rbf_svd_reference': dict(resource='cpu', limit=75, total_limit=80, max_attempts=2, script='docs/benchmarks/reference_rbf_scipy.py', mode='rbf_svd_reference', dependencies=['c1_rbf_svd_diagnosis']),
    'c1_rbf_svd_diagnosis': dict(resource='cpu', limit=30, total_limit=30, max_attempts=1, script='docs/benchmarks/diagnose_rbf_reference_solver.py', mode='rbf_svd_solver', dependencies=['c1_rbf_feasibility_diagnosis']),
    'c1_rbf_solver_diagnosis': dict(resource='cpu', limit=30, total_limit=30, max_attempts=1, script='docs/benchmarks/diagnose_rbf_reference_solver.py', mode='rbf_reference_solver', dependencies=['c1_rbf_feasibility_diagnosis']),
    'c1_rbf_feasibility_diagnosis': dict(resource='cpu', limit=20, total_limit=20, max_attempts=1, script='docs/benchmarks/diagnose_rbf_reference_feasibility.py', mode='rbf_feasibility', dependencies=['c2_rbf_nnls_accuracy']),
    'c1_rbf_scipy_reference': dict(resource='cpu', limit=75, total_limit=95, max_attempts=4, script='docs/benchmarks/reference_rbf_scipy.py', mode='rbf_scipy_reference', dependencies=['c2_rbf_nnls_accuracy','c1_rbf_feasibility_diagnosis']),
    'c2_rbf_nnls_accuracy': dict(resource='gpu', limit=120, total_limit=240, max_attempts=2, script='docs/benchmarks/diagnose_rbf_fit_tf.py', mode='rbf_nnls_accuracy', dependencies=['c1_rbf_nnls_diagnosis']),
    'c2_rbf_nnls_validation': dict(resource='gpu', limit=120, total_limit=240, max_attempts=2, script='docs/benchmarks/diagnose_rbf_fit_tf.py', mode='rbf_nnls_validation', dependencies=['c1_rbf_nnls_diagnosis']),
    'c1_rbf_nnls_diagnosis': dict(resource='cpu', limit=45, total_limit=60, max_attempts=2, script='docs/benchmarks/reference_rbf_fit.py', mode='rbf_nnls_diagnosis', dependencies=['c2_rbf_forward']),
    'c1_rbf_fit_reference': dict(resource='cpu', limit=90, total_limit=180, max_attempts=2, script='docs/benchmarks/reference_rbf_fit.py', mode='rbf_fit_reference', dependencies=['c2_rbf_forward']),
    'c2_rbf_fit_validation': dict(resource='gpu', limit=1800, total_limit=3600, max_attempts=2, script='docs/benchmarks/diagnose_rbf_fit_tf.py', mode='rbf_fit_validation', dependencies=['c1_rbf_svd_reference','c2_rbf_nnls_accuracy','c2_rbf_policy_accuracy']),
    'c1_rbf_consumer_audit': dict(resource='cpu', limit=60, total_limit=120, max_attempts=2, script='docs/benchmarks/diagnose_rbf_source_consumer.py', mode='rbf_source_audit', dependencies=['c2_lorenz_centered_validation']),
    'c2_rbf_forward': dict(resource='gpu', limit=120, total_limit=240, max_attempts=2, script='docs/benchmarks/diagnose_rbf_forward_tf.py', mode='rbf_forward', dependencies=['c1_rbf_consumer_audit']),
    'c1_lorenz_centered_reference': dict(resource='cpu', limit=60, total_limit=120, max_attempts=2, script='docs/benchmarks/reference_lorenz_stable_fit.py', mode='lorenz_centered_reference', dependencies=['c2_lorenz_consumer']),
    'c2_lorenz_centered_validation': dict(resource='gpu', limit=120, total_limit=240, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_stable_fit_tf.py', mode='lorenz_centered_validation', dependencies=['c1_lorenz_centered_reference']),
    'c1_lorenz_solver_reference': dict(resource='cpu', limit=60, total_limit=120, max_attempts=2, script='docs/benchmarks/reference_lorenz_stable_fit.py', mode='lorenz_solver_reference', dependencies=['c2_lorenz_consumer']),
    'c2_lorenz_solver_validation': dict(resource='gpu', limit=120, total_limit=240, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_stable_fit_tf.py', mode='lorenz_solver_validation', dependencies=['c1_lorenz_solver_reference']),
    'c1_lorenz_consumer': dict(resource='cpu', limit=90, total_limit=180, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_source_consumer.py', mode='lorenz_consumer_source', dependencies=['c1_lorenz_normalization']),
    'c2_lorenz_consumer': dict(resource='gpu', limit=300, total_limit=600, max_attempts=3, script='docs/benchmarks/diagnose_lorenz_consumer_tf.py', mode='lorenz_consumer', dependencies=['c1_lorenz_consumer']),
    'c3_guided_validation': dict(resource='gpu', limit=300, total_limit=600, max_attempts=3, script='docs/benchmarks/diagnose_controlled_smc_guided.py', mode='guided_validation', dependencies=['c3_figure4']),
    'c3_guided_matrix': dict(resource='gpu', limit=7200, total_limit=7200, max_attempts=2, script='docs/benchmarks/run_controlled_smc_guided_matrix.py', mode='guided_matrix', dependencies=['c3_guided_validation']),
    'c1_lorenz_normalization': dict(resource='cpu', limit=40, total_limit=80, max_attempts=2, script='docs/benchmarks/diagnose_lorenz_normalization.py', mode='lorenz_normalization', dependencies=['c1_demo']),
    'k2_sequential': dict(resource='gpu', limit=180, total_limit=360, max_attempts=2, script='docs/benchmarks/diagnose_kdm_sequential_tf.py', mode='sequential', dependencies=['k1_iwsg','k1_mixture_kalman']),
    'k1_mixture_kalman': dict(resource='gpu', limit=90, total_limit=120, script='docs/benchmarks/diagnose_kdm_mixture_kalman.py', mode='mixture_kalman', dependencies=['k1_iwsg']),
    'c3_figure4': dict(resource='gpu', limit=10800, total_limit=10800, script='docs/benchmarks/run_controlled_smc_figure4.py', mode='figure4', dependencies=['c2_batch','c3_cost_pilot']),
    'c2_batch': dict(resource='gpu', limit=180, total_limit=240, script='docs/benchmarks/diagnose_controlled_smc_batch.py', mode='batch', dependencies=['c3_cost_pilot']),
    'c3_cost_pilot': dict(resource='gpu', limit=300, total_limit=300, script='docs/benchmarks/diagnose_controlled_smc_figure4_cost.py', mode='cost', dependencies=['c2_full_parity','c2_kalman']),
    'c2_full_parity': dict(resource='gpu', limit=300, script='docs/benchmarks/diagnose_controlled_smc_full_parity.py', mode='full_parity', dependencies=['c1_demo','c2_parity_fp64']),
    'c2_kalman': dict(resource='gpu', limit=120, script='docs/benchmarks/diagnose_controlled_smc_kalman_tf.py', mode='kalman', dependencies=['c2_parity_fp64']),
    'c1_pilot': dict(resource='cpu', limit=45, script='docs/benchmarks/reference_controlled_smc.m', mode='pilot', dependencies=[]),
    'c1_demo': dict(resource='cpu', limit=250, script='docs/benchmarks/reference_controlled_smc.m', mode='demo', dependencies=['c1_pilot']),
    'k1_iwsg': dict(resource='gpu', limit=120, script='docs/benchmarks/diagnose_kdm_iwsg_tf.py', mode='iwsg', dependencies=[]),
    'c2_parity_fp64': dict(resource='gpu', limit=600, script='docs/benchmarks/diagnose_controlled_smc_tf.py', mode='parity', dependencies=['c1_pilot']),
    'c2_full': dict(resource='gpu', limit=600, script='docs/benchmarks/diagnose_controlled_smc_full_tf.py', mode='full', dependencies=['c2_parity_fp64','c1_demo']),
}

def campaign_attempts():
    """Numeric attempt order: attempt100 follows99, not09."""
    return sorted(shared.load_attempts(CAMPAIGN),
                  key=lambda pair:int(pair[0].parent.name[len("attempt"):].split("_",1)[0]))


DECISION_FIELDS = ('status','scope','replication','scientific_verdict',
                   'default_bookkeeping_repair_passed','full_recursive_admission',
                   'candidate_admitted','cross_backend_arithmetic_agreement',
                   'guard_full_scope_parity','comparison_verdict','forward_screen_passed',
                   'fitting_forward_screen_passed','solver_verdicts',
                   'numerical_screen_passed','dtype_screen_passed')


def compact_decision(summary):
    return {key: summary[key] for key in DECISION_FIELDS if key in summary}


def reported_decision(manifest):
    """Read omitted terminal verdicts from preserved, checksum-verified evidence."""
    decision = dict(manifest.get('decision') or {})
    if manifest.get('phase') != 'p0_forward_compare' or manifest.get('status') == 'RUNNING':
        return decision
    if all(key in decision for key in ('comparison_verdict','forward_screen_passed')):
        return decision
    try:
        path = Path(manifest['output_root']) / 'summary.json'
        expected = manifest.get('output_sha256', {}).get('summary.json')
        raw = path.read_bytes()
        if not expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Missing or mismatched summary checksum')
        summary = json.loads(raw)
        for key in ('status','scope','replication'):
            if key in decision and decision[key] != summary.get(key):
                raise ValueError('Summary differs from preserved decision')
        if not all(key in summary for key in ('comparison_verdict','forward_screen_passed')):
            raise ValueError('Summary omits the comparison verdict')
        decision.update(compact_decision(summary))
        decision['summary_verification'] = 'VERIFIED_PRESERVED_SUMMARY'
    except (OSError, ValueError, KeyError) as exc:
        decision.update(comparison_verdict='RESULT_SUMMARY_UNVERIFIED',
                        forward_screen_passed=False, summary_verification=str(exc))
    return decision


def reference_diagnostics():
    """Charge and report independent reference runs; never issue admission."""
    path = CAMPAIGN / 'reference-diagnostics.json'
    if not path.exists():
        return []
    reports, seen = [], set()
    for record in json.loads(path.read_text()):
        charge = record['charged_seconds']
        if (record['id'] in seen or record['resource'] not in ('cpu', 'gpu')
                or not math.isfinite(charge) or charge < 0):
            raise ValueError('Invalid or duplicate reference diagnostic charge')
        seen.add(record['id'])
        report = {key: record[key] for key in
                  ('id', 'resource', 'charged_seconds', 'summary', 'result_file',
                   'plan_file', 'stage', 'next_action', 'after_attempt')}
        try:
            raw = (ROOT / record['summary']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['summary_sha256']:
                raise ValueError('Reference summary checksum mismatch')
            summary = json.loads(raw)
            report['decision'] = compact_decision(summary.get('decision', summary))
            report['summary_verification'] = 'VERIFIED_PRESERVED_SUMMARY'
        except (OSError, ValueError) as exc:
            report['decision'] = {'status': 'EVIDENCE_UNVERIFIED'}
            report['summary_verification'] = str(exc)
        reports.append(report)
    return reports


def supplemental_training():
    """Discover bounded KDM training and avoid charging a terminal ledger twice."""
    ledger_path = CAMPAIGN / 'reference-diagnostics.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    charged = {entry['path'] for record in ledger for entry in record.get('executions', []) if 'path' in entry}
    rows = []
    for path in sorted(CAMPAIGN.glob('kdm-complete-training-20261007-*/manifest.json')):
        manifest = json.loads(path.read_text())
        relative = str(path.parent.relative_to(ROOT))
        elapsed = (min(time.time()-manifest['started_unix_seconds'], manifest['timeout_seconds'])
                   if manifest['status']=='RUNNING' else manifest.get('wall_seconds', 0))
        progress_path = path.parent / 'diagnostic/progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        rows.append(dict(path=relative, resource=manifest['resource'], status=manifest['status'],
                         mode=manifest['mode'], epoch_mode=manifest['epoch_mode'],
                         elapsed_seconds=max(0, elapsed), charged_to_reference_ledger=relative in charged,
                         supervisor_pid=manifest.get('supervisor_pid'), child_pid=manifest.get('child_pid'),
                         progress=progress, returncode=manifest.get('returncode')))
    return rows



def source_variability_runs():
    """Track source-only calibration separately from the original training pair."""
    ledger_path = CAMPAIGN / 'reference-diagnostics.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    charged = {entry['path'] for record in ledger for entry in record.get('executions', []) if 'path' in entry}
    rows = []
    for path in sorted(CAMPAIGN.glob('kdm-source-variability-20261008-*/manifest.json')):
        manifest = json.loads(path.read_text())
        relative = str(path.parent.relative_to(ROOT))
        elapsed = manifest.get('routine_cpu_seconds', 0.0)
        children = []
        for child in sorted(path.parent.glob('unit*-*/manifest.json')):
            item = json.loads(child.read_text())
            duration = (max(0, time.time()-item['started_unix_seconds'])
                        if item['status']=='RUNNING' else item.get('wall_seconds', 0.0))
            elapsed += duration
            progress_path = child.parent / 'diagnostic/progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            children.append(dict(path=str(child.parent.relative_to(ROOT)), kind=item['kind'],
                                 status=item['status'], elapsed_seconds=duration,
                                 child_pid=item.get('child_pid'), progress=progress))
        if manifest['status']!='RUNNING':
            elapsed = manifest['charged_seconds']
        rows.append(dict(path=relative, resource='cpu', status=manifest['status'],
                         charged_to_reference_ledger=relative in charged,
                         charged_seconds=elapsed,
                         remaining_phase_seconds=manifest['phase_cpu_cap_seconds']-elapsed,
                         plan=manifest['plan'], result=manifest['result'],
                         supervisor_pid=manifest.get('supervisor_pid'), children=children))
    return rows


def completed_training_review(references, training):
    """Recognize a terminal review only for the complete runs it actually audits."""
    full_runs = {item['path'] for item in training if item['epoch_mode'] == 'full'}
    for report in reversed(references):
        if (report['summary_verification'] != 'VERIFIED_PRESERVED_SUMMARY'
                or report['decision'].get('status') != 'PASS'
                or report['decision'].get('scope') != 'COMPLETE_SOURCE_AND_TF_DEVELOPMENT_TRAINING'):
            continue
        try:
            summary = json.loads((ROOT / report['summary']).read_text())
            reviewed = summary.get('training_runs', [])
            if ({item['path'] for item in reviewed} != full_runs
                    or {item['mode'] for item in reviewed} != {'source', 'tf'}):
                continue
            if all(item['status'] == 'COMPLETED' and item['returncode'] == 0
                   and hashlib.sha256((ROOT / item['manifest']).read_bytes()).hexdigest() == item['manifest_sha256']
                   and hashlib.sha256((ROOT / item['audit']).read_bytes()).hexdigest() == item['audit_sha256']
                   for item in reviewed):
                return report
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None

def csmc_qualification_runs():
    """Track the bounded downstream phase without charging its terminal ledger twice."""
    ledger_path = CAMPAIGN / 'reference-diagnostics.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    charged = {(entry['path'], record['resource']) for record in ledger
               for entry in record.get('executions', []) if 'path' in entry}
    rows = []
    for path in sorted(CAMPAIGN.glob('csmc-qualification-20261009-*/manifest.json')):
        manifest = json.loads(path.read_text())
        relative = str(path.parent.relative_to(ROOT))
        used = dict(manifest.get('charged_seconds', {'cpu': 0., 'gpu': 0.}))
        for execution in manifest.get('executions', []):
            if execution['status'] == 'RUNNING':
                used[execution['resource']] += max(0., time.time()-execution['started_unix_seconds'])
        rows.append(dict(path=relative, status=manifest['status'], stage=manifest['stage'], plan=manifest['plan'],
                         result=manifest['result_file'], used_seconds=used,
                         uncharged_seconds={r: 0. if (relative,r) in charged else used[r] for r in used},
                         supervisor_pid=manifest['supervisor_pid'], started_unix_seconds=manifest.get('started_unix_seconds',0)))
    return rows


def paired_calibration_runs():
    """Track the bounded paired run, including live usage before ledger close."""
    ledger = json.loads((CAMPAIGN/'reference-diagnostics.json').read_text())
    charged = {(entry['path'], record['resource']) for record in ledger
               for entry in record.get('executions', []) if 'path' in entry}
    rows = []
    for path in sorted(CAMPAIGN.glob('paired-calibration-20261011/calibration-*/manifest.json')):
        manifest = json.loads(path.read_text())
        relative = str(path.parent.relative_to(ROOT))
        elapsed = dict(manifest['resource_wall_seconds'])
        for step in manifest['steps']:
            if step['status'] == 'RUNNING':
                elapsed[step['resource']] += max(0, min(
                    time.time()-step['started_unix_seconds'], step['timeout_seconds']))
        uncharged = {r: 0 if (relative, r) in charged else elapsed[r] for r in elapsed}
        reserved = {r: max(0, manifest['phase_cap_seconds'][r]-
                         manifest['prior_phase_seconds'][r]-elapsed[r])
                    if manifest['status']=='RUNNING' else 0 for r in elapsed}
        rows.append(dict(path=relative, status=manifest['status'],
            plan=manifest['plan_file'], result=manifest['result_file'],
            supervisor_pid=manifest['supervisor_pid'],
            current_step=manifest['steps'][-1] if manifest['steps'] else None,
            completed_units=len(manifest['completed_units']),
            elapsed_seconds=elapsed, uncharged_seconds=uncharged,
            reservation_remaining_seconds=reserved))
    return rows


def status():
    old = shared.snapshot()
    records = campaign_attempts()
    used = {r: sum(shared.charge(m) for _,m in records if m['resource']==r) for r in ('cpu','gpu')}
    references = reference_diagnostics()
    for reference in references:
        used[reference['resource']] += reference['charged_seconds']
    training = supplemental_training()
    variability = source_variability_runs()
    qualification = csmc_qualification_runs()
    paired = paired_calibration_runs()
    for run in paired:
        for resource, charge in run["uncharged_seconds"].items():
            used[resource] += charge
    for run in qualification:
        for resource, charge in run['uncharged_seconds'].items():
            used[resource] += charge
    for run in variability:
        if not run["charged_to_reference_ledger"]:
            used["cpu"] += run["charged_seconds"]
    for run in training:
        if not run["charged_to_reference_ledger"]:
            used[run["resource"]] += run["elapsed_seconds"]
    latest_reference = next((item for item in reversed(references)
                             if item['after_attempt'] == len(records)), None)
    opening = json.loads((CAMPAIGN/'opening-budget.json').read_text())
    # Charge both ledgers if another old-lane job has run since activation.
    grants = opening.get('additional_resource_grants_seconds', {})
    remaining = {r: min(opening[r], old['resources'][r]['remaining_seconds']) + grants.get(r,0)-used[r] for r in used}
    running = [str(p) for p,m in records if m['status']=='RUNNING']
    passed = {m['phase'] for _,m in records if m['status']=='COMPLETE' and m.get('decision',{}).get('status')=='PASS'}
    superseded={'c2_parity':'c2_parity_fp64: preserve FP64 expected-value ingestion','c2_lorenz_solver_validation':'c2_lorenz_centered_validation: exact Gaussian-ratio cancellation repair'}
    superseded.update(c2_rbf_nnls_validation='c2_rbf_nnls_accuracy: calibrated termination',
        c1_rbf_fit_reference='c1_rbf_svd_reference: author NNLS did not converge',
        c1_rbf_scipy_reference='c1_rbf_svd_reference: failed independent optimality checks',
        c1_rbf_solver_diagnosis='c1_rbf_svd_diagnosis: successful fresh factorization')

    passed-=set(superseded)
    validation_complete={'c1_pilot','c1_demo','c2_parity_fp64','c2_full','c2_full_parity','c2_kalman','k1_iwsg'}.issubset(passed)
    terminal_result='docs/plans/controlled-smc-lorenz-refinement-result-20260930.md'
    initial_result='docs/plans/controlled-smc-kdm-first-execution-result-20260929.md'
    precision_result='docs/plans/controlled-smc-lorenz-precision-result-20260930.md'
    if (ROOT/precision_result).exists():terminal_result=precision_result
    native_result='docs/plans/controlled-smc-lorenz-author-native-result-20260930.md'
    if (ROOT/native_result).exists():terminal_result=native_result
    source_port_result='docs/plans/controlled-smc-lorenz-source-port-result-20261001.md'
    if (ROOT/source_port_result).exists():terminal_result=source_port_result
    matrix_result='docs/plans/controlled-smc-lorenz-matrix-crossover-result-20261001.md'
    if (ROOT/matrix_result).exists():terminal_result=matrix_result
    bounded_result='docs/plans/controlled-smc-lorenz-bounded-spd-result-20261001.md'
    if (ROOT/bounded_result).exists():terminal_result=bounded_result
    root_result='docs/plans/controlled-smc-lorenz-root-cause-result-20261002.md'
    if (ROOT/root_result).exists():terminal_result=root_result
    repair_result='docs/plans/controlled-smc-lorenz-repair-result-20261002.md'
    if (ROOT/repair_result).exists():terminal_result=repair_result
    active_plan=REPLICATION_FIRST_PLAN
    full_running=any(m['phase']=='c1_source_full' and m['status']=='RUNNING' for _,m in records)
    full_failed=any(m['phase']=='c1_source_full' and m['status'] not in ('RUNNING','COMPLETE') for _,m in records)
    full_spec=json.loads((ROOT/FULL_SOURCE_SPEC).read_text()) if (ROOT/FULL_SOURCE_SPEC).exists() else {}
    isolation_running=any(m['phase']=='c1_source_isolation_check' and m['status']=='RUNNING' for _,m in records)
    analysis_running=any(m['phase']=='c2_source_results' and m['status']=='RUNNING' for _,m in records)
    analysis_failed=any(m['phase']=='c2_source_results' and m['status'] not in ('RUNNING','COMPLETE') for _,m in records)
    source_failures=source_driver_failures()
    replay_running=any(m['phase']=='c1_source_failure_replay' and m['status']=='RUNNING' for _,m in records)
    p0_running=next((m['phase'] for _,m in records if m['phase'] in P0_FORWARD_PHASES and m['status']=='RUNNING'),None)
    solver_reference_verified=any(m['phase']=='p0_solver_system_reference' and
        m['status']=='COMPLETE' and reported_decision(m).get('scientific_verdict')=='CONDITIONAL_SOLVER_REFERENCE_AGREES'
        for _,m in records)
    active_stage=(p0_running.upper()+'_RUNNING' if p0_running else
                  'P0_TF_SOLVER_DIAGNOSTIC_COMPLETE_PORT_AND_KDM_INCOMPLETE' if any(name in passed for name in P0_TF_SOLVER_PHASES) else
                  'P0_OFFSET_DIAGNOSTIC_COMPLETE_PORT_AND_KDM_INCOMPLETE' if 'p0_offset_localization' in passed else
                  'P0_PERTURBATION_MEASURED_PORT_AND_KDM_INCOMPLETE' if 'p0_perturb_tf' in passed else
                  'P0_SOURCE_PERTURBATION_MEASURED_TF_PENDING' if 'p0_perturb_source' in passed else
                  'P0_SOLVER_REFERENCE_VERIFIED_PORT_AND_KDM_INCOMPLETE' if solver_reference_verified else
                  'P0_SOLVER_REFERENCE_MISMATCH_PORT_AND_KDM_INCOMPLETE' if 'p0_solver_system_reference' in passed else
                  'P0_FITTING_COMPARED_PORT_AND_KDM_INCOMPLETE' if 'p0_fitting_forward' in passed else
                  'C2_SOURCE_ANALYSIS_COMPLETE_PORT_AND_KDM_INCOMPLETE' if 'c2_source_results' in passed else
                  'C2_SOURCE_ANALYSIS_RUNNING' if analysis_running else
                  'C2_SOURCE_ANALYSIS_REPAIR_REQUIRED' if analysis_failed else
                  'C2_SOURCE_RESULT_ASSEMBLY_REQUIRED' if 'c1_source_full' in passed else
                  'C1_SOURCE_PARTIAL_FAILURE_OTHER_DRIVERS_RUNNING' if full_running and source_failures else
                  'C1_FULL_SOURCE_RUNNING' if full_running else
                  'C1_SOURCE_ISOLATION_VERIFICATION_RUNNING' if isolation_running else
                  'C1_FULL_SOURCE_RETRY_READY' if full_failed and sum(m['phase']=='c1_source_full' for _,m in records)==1 and 'c1_source_isolation_check' in passed else
                  'C1_SOURCE_FAILURE_REPLAY_RUNNING' if replay_running else
                  'C1_SOURCE_FAILURE_DIAGNOSED_REPLICATION_INCOMPLETE' if 'c1_source_failure_replay' in passed else
                  'C1_FULL_SOURCE_REPAIR_REQUIRED' if full_failed else
                  'C1_FULL_SOURCE_READY' if 'c1_source_cost_pilot' in passed and full_spec.get('cpu_allocation_status')=='APPROVED' else
                  'C1_FULL_SOURCE_ALLOCATION_REQUIRED' if 'c1_source_cost_pilot' in passed else
                  'C1_SOURCE_COST_PILOT_REQUIRED' if 'c0_source_lock' in passed else 'C0_SOURCE_LOCK_REQUIRED')
    if latest_reference and not running and not old['running']:
        active_stage = (latest_reference['stage']
                        if latest_reference['summary_verification'] == 'VERIFIED_PRESERVED_SUMMARY'
                        and latest_reference['decision'].get('status') == 'PASS'
                        else 'REFERENCE_DIAGNOSTIC_EVIDENCE_REVIEW_REQUIRED')
    training_running = [item["path"] for item in training if item["status"]=="RUNNING"]
    training_review = completed_training_review(references, training)
    if latest_reference and not training_running:
        active_plan = latest_reference['plan_file']
    if training and (training_running or not training_review):
        active_plan = 'docs/plans/controlled-smc-kdm-complete-training-20261007.md'
        active_stage = ('KDM_COMPLETE_DEVELOPMENT_TRAINING_RUNNING' if training_running else
                        'KDM_COMPLETE_TRAINING_RESULTS_REQUIRE_REVIEW')
    training_budget = {resource: limit-sum(item["elapsed_seconds"] for item in training if item["resource"]==resource)
                       for resource,limit in (("cpu",64800),("gpu",57600))}
    variability_running = [item['path'] for item in variability if item['status']=='RUNNING']
    active_variability = next((item for item in reversed(variability)
                               if item['status']=='RUNNING' or not item['charged_to_reference_ledger']), None)
    if active_variability:
        active_plan = active_variability['plan']
        active_stage = ('KDM_SOURCE_VARIABILITY_CALIBRATION_RUNNING' if variability_running
                        else 'KDM_SOURCE_VARIABILITY_RESULT_REVIEW_REQUIRED')
    qualification_running = [item['path'] for item in qualification if item['status']=='RUNNING']
    if qualification:
        latest_qualification = max(qualification,key=lambda item:item['started_unix_seconds'])
        active_plan = 'docs/plans/controlled-smc-downstream-qualification-20261009.md'
        active_stage = latest_qualification['stage']+'_'+latest_qualification['status']
    if qualification_running:
        active_plan = 'docs/plans/controlled-smc-downstream-qualification-20261009.md'
        active_stage = next(item['stage']+'_RUNNING' for item in reversed(qualification) if item['status']=='RUNNING')
    paired_running = [item["path"] for item in paired if item["status"]=="RUNNING"]
    if paired:
        active_plan = paired[-1]["plan"]
        active_stage = "KDM_PAIRED_DEVELOPMENT_CALIBRATION_"+paired[-1]["status"]
    return dict(paired_calibration=paired, csmc_qualification=qualification, source_variability=variability,
                supplemental_training=training, supplemental_training_budget_seconds=training_budget,
                complete_training_review=training_review,
                reference_diagnostics=references, active_reference=latest_reference,
                replication_target='author_behavior_no_worse_v1',author_failure_is_replication_veto=False,paper_result_success_required_for_port=False,source_driver_failures=source_failures,master=MASTER,plan=active_plan,active_plans=[REPLICATION_FIRST_PLAN,C0_SOURCE_LOCK_PLAN,C1_SOURCE_PILOT_PLAN,C1_FULL_SOURCE_PLAN,SOURCE_MEMORY_REPAIR_PLAN,SOURCE_FAILURE_REPLAY_PLAN,C2_SOURCE_RESULTS_PLAN,K0_SOURCE_CONFIG_PLAN,P0_FORWARD_PLAN,P0_FITTING_PLAN,P0_SOLVER_PLAN,P0_PERTURB_PLAN,P0_OFFSET_PLAN,P0_TF_SOLVER_PLAN]+[item['plan_file'] for item in references],historical_plans=[HISTORICAL_EXECUTION_PLAN,LORENZ_REPAIR_PLAN,LORENZ_ROOT_CAUSE_PLAN,LORENZ_BOUNDED_PLAN,LORENZ_MATRIX_PLAN,LORENZ_SOURCE_PORT_PLAN,LORENZ_PRECISION_PLAN,LORENZ_BATCH_PLAN,LORENZ_PILOT_PLAN,LORENZ_REFINEMENT_PLAN,REFIT_PLAN,'docs/plans/controlled-smc-rbf-fitting-validation-20260929.md','docs/plans/controlled-smc-rbf-consumer-audit-20260929.md','docs/plans/controlled-smc-guided-repair-20260929.md','docs/plans/controlled-smc-lorenz-consumer-repair-20260929.md','docs/plans/controlled-smc-lorenz-solver-validation-20260929.md','docs/plans/controlled-smc-lorenz-centered-recursion-20260929.md'],result=(paired[-1]['result'] if paired else active_variability['result'] if active_variability else 'docs/plans/controlled-smc-kdm-complete-training-result-20261007.md' if training and (training_running or not training_review) else latest_reference['result_file'] if latest_reference else 'docs/plans/controlled-smc-tf-solver-compatibility-result-20261006.md' if any(name in passed for name in P0_TF_SOLVER_PHASES) else
                'docs/plans/controlled-smc-offset-localization-result-20261006.md' if 'p0_offset_localization' in passed else
                'docs/plans/controlled-smc-perturbation-result-20261006.md' if 'p0_perturb_tf' in passed else
                'docs/plans/controlled-smc-p0-solver-semantics-result-20261005.md' if 'p0_solver_system_reference' in passed else
                'docs/plans/controlled-smc-p0-fitting-forward-result-20261005.md' if 'p0_fitting_forward' in passed else
                'docs/plans/controlled-smc-kdm-restructure-execution-result-20261003.md' if (ROOT/'docs/plans/controlled-smc-kdm-restructure-execution-result-20261003.md').exists() else None),active_program_stage=active_stage,
                historical_result=terminal_result if (ROOT/terminal_result).exists() else initial_result,
                initial_result=initial_result,current_implemented_tranche_complete=set(ACTIVE_PHASES).issubset(passed),
                validation_tranche_complete=validation_complete,implemented_phases=sorted(ACTIVE_PHASES),historical_phases=sorted(set(PHASES)-set(ACTIVE_PHASES)),superseded_phases=superseded,attempts=len(records),used_seconds=used,
                remaining_seconds=remaining,additional_resource_grants_seconds={r:grants.get(r,0) for r in used},tranche_remaining_seconds={r:opening['tranche'][r]+grants.get(r,0)-used[r] for r in used},
                running=running+old['running']+training_running+variability_running+qualification_running+paired_running,passed=sorted(passed),
                csmc_replication='INCOMPLETE',kdm_replication='INCOMPLETE',iapf='INCOMPLETE_DEFERRED')


def source_driver_failures():
    records=[m for _,m in campaign_attempts() if m['phase']=='c1_source_full']
    if not records:
        return []
    path=Path(records[-1]['output_root'])/'progress.json'
    if not path.exists():
        return []
    try:
        drivers=json.loads(path.read_text())
    except json.JSONDecodeError:
        return []  # Writer may be replacing its live progress report.
    return [dict(driver=d['driver'],status=d['status'],returncode=d.get('returncode'),
                 completed_cases=len(d.get('cases',[])))
            for d in drivers if d['status'] not in ('COMPLETE','PREPARED_NOT_EXECUTED')]


def completed_source_attempt():
    matches=[m for _,m in campaign_attempts() if m['phase']=='c1_source_full']
    if not matches or matches[-1]['status']!='COMPLETE' or matches[-1].get('decision',{}).get('status')!='PASS':
        raise RuntimeError('C2 requires a completed successful full source attempt')
    return Path(matches[-1]['output_root'])


def source_analysis_command(script, output, source_attempt):
    return [PYTHON,'-B',str(script),'--output',str(output),'--source-attempt',str(source_attempt)]


def queue_status(phase_name='c2_source_results'):
    path=CAMPAIGN/f'queue-{phase_name}.json'
    if not path.exists():
        return dict(phase=phase_name,status='NOT_QUEUED')
    record=json.loads(path.read_text())
    result={k:record[k] for k in ('phase','status','pid','started_utc','deadline_epoch','ended_utc','error','result') if k in record}
    result['supervisor_alive']=shared.alive(record.get('pid'),record.get('process_identity'))
    return result


def continuation_readiness():
    current=status()
    if 'c2_source_results' in current['passed']:
        return current
    if current.get('source_driver_failures'):
        raise RuntimeError('C2 is blocked by incomplete or failed source drivers')
    records=campaign_attempts()
    if any(m['phase']=='c2_source_results' for _,m in records):
        raise RuntimeError('C2 already has an attempt; inspect progress before any retry')
    sources=[m for _,m in records if m['phase']=='c1_source_full']
    if not sources or sources[-1]['status'] not in ('RUNNING','COMPLETE'):
        raise RuntimeError('C2 requires a running or successfully completed full source attempt')
    source=sources[-1]
    reserved=0
    if source['status']=='RUNNING':
        if not shared.alive(source.get('supervisor_pid'),source.get('supervisor_identity')):
            raise RuntimeError('Full source supervisor disappeared before completion')
        reserved=max(0,source['timeout_seconds']-shared.charge(source))
    elif source.get('decision',{}).get('status')!='PASS':
        raise RuntimeError('Source attempt failed its completion checks')
    available=min(current['remaining_seconds']['cpu'],current['tranche_remaining_seconds']['cpu'])
    if available-reserved < PHASES['c2_source_results']['limit']:
        raise RuntimeError('No 600-second CPU analysis margin after the full source reservation')
    opening=json.loads((CAMPAIGN/'opening-budget.json').read_text())
    if len(records)>=opening['attempt_limit']:
        raise RuntimeError('No remaining attempt for C2 analysis')
    return current


def source_replay_readiness():
    current=status()
    if 'c1_source_failure_replay' in current['passed']:
        return current
    if any(m['phase']=='c1_source_failure_replay' for _,m in campaign_attempts()):
        raise RuntimeError('Inspect the existing failure replay before a retry')
    if not set(PHASES['c1_source_failure_replay']['dependencies']).issubset(current['passed']):
        raise RuntimeError('Source lock and exact isolation check are required')
    evidence=json.loads((SOURCE_FAILURE_EVIDENCE/'failure.json').read_text())
    if evidence['classification']!='ORIGINAL_SOURCE_NUMERICAL_FAILURE_IN_OCTAVE':
        raise RuntimeError('Replay is only specified for the frozen original-source covariance failure')
    reserved=sum(max(0,m['timeout_seconds']-shared.charge(m))
                 for _,m in campaign_attempts() if m['status']=='RUNNING' and m['resource']=='cpu')
    if min(current['remaining_seconds']['cpu'],current['tranche_remaining_seconds']['cpu'])-reserved<60:
        raise RuntimeError('No remaining 60-second CPU diagnostic allowance')
    return current


def p0_forward_readiness(phase_name):
    current=status()
    if phase_name in current['passed']:
        return current
    records=campaign_attempts()
    if any(m['phase']==phase_name for _,m in records):
        raise RuntimeError('Inspect the existing forward comparison attempt before retrying')
    queue_wait_state(phase_name,current)
    running={m['phase'] for _,m in records if m['status']=='RUNNING'}
    pending=[]
    for name in ('c1_source_failure_replay',)+P0_FORWARD_PHASES:
        if name==phase_name or name in running or name in current['passed']:
            continue
        queued=queue_status(name)
        if queued['status'] in ('WAITING','EXECUTING') and queued.get('supervisor_alive'):
            pending.append(name)
    resource=PHASES[phase_name]['resource']
    reserved=sum(max(0,m['timeout_seconds']-shared.charge(m)) for _,m in records
                 if m['status']=='RUNNING' and m['resource']==resource)
    reserved+=sum(PHASES[name]['limit'] for name in pending if PHASES[name]['resource']==resource)
    available=min(current['remaining_seconds'][resource],current['tranche_remaining_seconds'][resource])
    if available-reserved<PHASES[phase_name]['limit']:
        raise RuntimeError('Insufficient unreserved '+resource+' budget for '+phase_name)
    opening=json.loads((CAMPAIGN/'opening-budget.json').read_text())
    if len(records)+len(pending)+1>opening['attempt_limit']:
        raise RuntimeError('Insufficient attempt budget for queued forward comparisons')
    return current


def p0_forward_inputs(phase_name, output, records=None):
    records=campaign_attempts() if records is None else records
    def completed(name):
        matches=[m for _,m in records if m['phase']==name and m['status']=='COMPLETE'
                 and m.get('decision',{}).get('status')=='PASS']
        if not matches:
            raise RuntimeError('Missing successful input: '+name)
        return Path(matches[-1]['output_root'])
    command=[PYTHON,'-B',str(output/'sources'/Path(PHASES[phase_name]['script']).name),
             '--root',str(ROOT),'--output',str(output)]
    if phase_name in P0_TF_SOLVER_PHASES:
        reference=CAMPAIGN/'solver-compatibility-reference-20261006-01'
        command+=['--reference',str(reference)]
        if phase_name=='p0_tf_solver_cpu_reference':command+=['--cpu-reference']
        index=json.loads((reference/'fixture.json').read_text())
        files=[reference/'fixture.json',reference/'manifest.json']
        files.extend(reference/item['file'] for item in index['arrays'].values())
    elif phase_name=='p0_offset_localization':
        reference=CAMPAIGN/'offset-localization-20261006-03'
        capture=completed('p0_forward_capture')
        conditional=completed('p0_fitting_forward')
        command+=['--reference',str(reference),'--capture',str(capture),'--conditional',str(conditional)]
        files=[reference/'fixture.json',reference/'manifest.json',capture/'fixture.json',conditional/'manifest.json']
        for directory in (reference,capture):
            index=json.loads((directory/'fixture.json').read_text())
            files.extend(directory/item['file'] for item in index['arrays'].values())
        files+=sorted(conditional.glob('case3_stage1_centered_qr*.tensor'))
    elif phase_name in P0_PERTURB_PHASES:
        reference=completed('p0_forward_capture' if phase_name=='p0_perturb_source' else 'p0_perturb_source')
        command+=['--mode','source' if phase_name=='p0_perturb_source' else 'tf','--reference',str(reference)]
        index=json.loads((reference/'fixture.json').read_text())
        files=[reference/'fixture.json',reference/'summary.json']
        files.extend(reference/item['file'] for item in index['arrays'].values())
        if phase_name=='p0_perturb_source':
            source_lock=completed('c0_source_lock')
            command+=['--source-lock',str(source_lock)]
            files.append(source_lock/'source-lock.json')
        else:
            files.append(reference/'source-rows.json')
            previous=[m for _,m in records if m['phase']==phase_name and m['status'] in ('TIMED_OUT','FAILED') and (Path(m['output_root'])/'tf-rows.json').exists()]
            if previous:
                resume=Path(previous[-1]['output_root'])
                command+=['--resume',str(resume)]
                files.extend([resume/'manifest.json',resume/'tf-rows.json'])
    elif phase_name=='p0_forward_capture':
        reference=completed('c1_lorenz_author_native')
        source_lock=completed('c0_source_lock')
        command+=['--reference',str(reference),'--source-lock',str(source_lock)]
        files=[source_lock/'source-lock.json',reference/'summary.json']+sorted(reference.glob('*.mat'))
        files+=sorted(path for path in (reference/'sources').rglob('*') if path.is_file())
    elif phase_name in ('p0_solver_semantics','p0_solver_system_reference'):
        reference=completed('c2_lorenz_matrix_crossover')
        fallback=completed('c1_lorenz_matrix_fallback')
        command+=['--reference',str(reference),'--fallback',str(fallback)]
        if phase_name=='p0_solver_system_reference':
            command+=['--system-lapack','/lib/x86_64-linux-gnu/liblapack.so.3']
        files=[ROOT/'.localresources/code/octave-6.4.0-dMatrix.cc']
        if phase_name=='p0_solver_system_reference':
            files.append(Path('/lib/x86_64-linux-gnu/liblapack.so.3'))
        for directory in (reference,fallback):
            index=json.loads((directory/'fixture.json').read_text())
            files.extend([directory/'fixture.json',directory/'summary.json'])
            files.extend(directory/item['file'] for item in index['arrays'].values())
    elif phase_name=='p0_fitting_forward':
        reference=completed('p0_forward_capture')
        prior_forward=completed('p0_forward_compare')
        command+=['--reference',str(reference)]
        index=json.loads((reference/'fixture.json').read_text())
        files=[reference/'fixture.json',reference/'summary.json',prior_forward/'summary.json']
        files.extend(reference/item['file'] for item in index['arrays'].values())
    elif phase_name=='p0_forward_compare':
        reference=completed('p0_forward_capture')
        failure=completed('c1_source_failure_replay')
        command+=['--reference',str(reference),'--failure',str(failure)]
        files=[]
        for directory,name in ((reference,'fixture.json'),(failure,'failed-policy.json')):
            index=json.loads((directory/name).read_text())
            files.extend([directory/name,directory/'summary.json'])
            files.extend(directory/item['file'] for item in index['arrays'].values())
    else:
        raise ValueError('Not a forward replay phase: '+phase_name)
    return command,{str(path):shared.sha(path) for path in files}


def queue_wait_state(phase_name, current):
    if phase_name=='c2_source_results' and current.get('source_driver_failures'):
        raise RuntimeError('C2 is blocked by incomplete or failed source drivers')
    records=campaign_attempts()
    missing=set(PHASES[phase_name]['dependencies'])-set(current['passed'])
    deadlines=[time.time()+60]
    for dependency in missing:
        matching=[m for _,m in records if m['phase']==dependency]
        if matching:
            latest=matching[-1]
            if latest['status']=='COMPLETE' and latest.get('decision',{}).get('status')=='PASS':
                continue  # A predecessor may finish between status() and this read.
            if latest['status']!='RUNNING':
                raise RuntimeError('Required phase failed or did not pass: '+dependency)
            if not shared.alive(latest.get('supervisor_pid'),latest.get('supervisor_identity')):
                raise RuntimeError('Required supervisor disappeared: '+dependency)
            deadlines.append(latest['started_epoch']+latest['timeout_seconds']+60)
        else:
            pending=queue_status(dependency)
            if pending['status'] not in ('WAITING','EXECUTING') or not pending.get('supervisor_alive'):
                raise RuntimeError('Required phase not passed or queued: '+dependency)
            if 'deadline_epoch' not in pending:
                raise RuntimeError('Dependency queue has no bounded deadline: '+dependency)
            deadlines.append(pending['deadline_epoch']+PHASES[dependency]['limit']+60)
    for _,record in records:
        if record['status']=='RUNNING':
            if not shared.alive(record.get('supervisor_pid'),record.get('supervisor_identity')):
                raise RuntimeError('Numerical supervisor disappeared; inspect and recover its attempt')
            deadlines.append(record['started_epoch']+record['timeout_seconds']+60)
    return not current['running'] and not missing, max(deadlines)


def execute(phase_name):
    if phase_name not in ACTIVE_PHASES:
        raise RuntimeError('Historical phase is not active: '+phase_name)
    phase=PHASES[phase_name]
    with shared.acquire_lock(CAMPAIGN):
        current=status()
        opening=json.loads((CAMPAIGN/'opening-budget.json').read_text())
        if current['running']: raise RuntimeError('Reconcile live attempts before launching')
        if current['attempts']>=opening['attempt_limit']: raise RuntimeError('Tranche attempt limit: reassess before extending')
        if phase.get('max_attempts') is not None and sum(m['phase']==phase_name for _,m in campaign_attempts())>=phase['max_attempts']: raise RuntimeError('Phase attempt limit reached')
        if not set(phase['dependencies']).issubset(current['passed']): raise RuntimeError('Required diagnostic has not passed')
        if phase_name in current['passed']: raise RuntimeError('Completed phase needs an explicit new experiment specification')
        if phase_name in LORENZ_BOUNDED_PHASES and phase['mode']=='validation':
            upstream=[m for _,m in campaign_attempts() if m['phase']==phase['dependencies'][0] and m['status']=='COMPLETE'][-1]
            prerequisite=json.loads((Path(upstream['output_root'])/'summary.json').read_text())
            if not any(row.get('eligible',False) for row in prerequisite.get('candidates',[])):
                raise RuntimeError('Independent validation is ineligible: no candidate survived the numerical/non-harm screen')
        resource=phase['resource']
        timeout=min(phase['limit'],current['remaining_seconds'][resource],current['tranche_remaining_seconds'][resource])
        if phase_name in LORENZ_BOUNDED_PHASES:
            stage_used=sum(shared.charge(m) for _,m in campaign_attempts()
                if m['phase'] in LORENZ_BOUNDED_PHASES and m['resource']==resource)
            timeout=min(timeout,({'cpu':90,'gpu':1800}[resource])-stage_used)
        if phase_name in LORENZ_ROOT_CAUSE_PHASES:
            audit_used=sum(shared.charge(m) for _,m in campaign_attempts() if m['phase'] in LORENZ_ROOT_CAUSE_PHASES)
            timeout=min(timeout,30-audit_used)
        if phase_name in LORENZ_REPAIR_PHASES:
            repair_records=[m for _,m in campaign_attempts() if m['phase'] in LORENZ_REPAIR_PHASES]
            if len(repair_records)>=9:raise RuntimeError('Repair tranche attempt cap reached')
            timeout=min(timeout,{'cpu':30,'gpu':300}[resource]-sum(shared.charge(m) for m in repair_records if m['resource']==resource))
        if 'total_limit' in phase:
            charged=sum(shared.charge(m) for _,m in campaign_attempts() if m['phase']==phase_name)
            timeout=min(timeout,phase['total_limit']-charged)
        if phase_name == 'c1_source_full':
            timeout=full_source_timeout(current)
        if phase_name == 'c2_source_results' and timeout < phase['limit']:
            raise RuntimeError('C2 needs its reviewed 600-second CPU reservation')
        if timeout<1: raise RuntimeError('Budget exhausted')
        if phase_name=='c1_demo':
            pilots=[m for _,m in campaign_attempts() if m['phase']=='c1_pilot' and m['status']=='COMPLETE']
            pilot=json.loads((Path(pilots[-1]['output_root'])/'summary.json').read_text())
            estimate=pilot['algorithm_seconds']*(3000/128)*(1024/256)*1.5+15
            if estimate>timeout: raise RuntimeError(f'Source demo predicted {estimate:.1f}s exceeds bounded {timeout:.1f}s; retain pilot and revise budget before launch')
        if phase_name in ('c2_rbf_full_retention_precision','c2_rbf_refit_precision'):
            prior=CAMPAIGN/('attempt67_c2_rbf_refit_pilot' if phase_name=='c2_rbf_refit_precision' else 'attempt64_c2_rbf_full_retention_pilot')
            timings=json.loads((prior/'progress.json').read_text())
            estimate=sum(row['seconds'] for row in timings)*32+30
            if estimate>timeout:raise RuntimeError(f'Full-retention final predicted {estimate:.1f}s exceeds bounded {timeout:.1f}s; preserve pilot and revise before launch')
        output=CAMPAIGN/f'attempt{current["attempts"]+1:02d}_{phase_name}'
        sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,
                 'docs/benchmarks/diagnose_controlled_smc_reference.py','docs/benchmarks/reference_controlled_smc.m',AUTHOR+'cSMC_Resample.m',AUTHOR+'Model_NeuroScience.m',AUTHOR+'Systematic_Resampling.m']
        if phase_name in ('c1_rbf_consumer_audit','c2_rbf_forward','c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_feasibility_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference','c1_rbf_fit_reference','c1_rbf_nnls_diagnosis','c2_rbf_paired_validation','c2_rbf_policy_accuracy','c2_rbf_nnls_accuracy','c2_rbf_nnls_validation','c2_rbf_fit_validation'):
            sources=[rel for rel in sources if not str(rel).startswith(AUTHOR)]
            sources+=['docs/plans/controlled-smc-rbf-consumer-audit-20260929.md','docs/benchmarks/reference_rbf_consumer.m']
            sources+=[str(Path(AUTHOR).parent/'ToyMultimodal'/name) for name in ('cSMC_Resample.m','DiscreteSample.m','Systematic_Resampling.m','SimulatedData_SigmaSq_1.mat')]
            if phase_name.startswith('c2_rbf_'):sources+=['bayesfilter/replication/rbf_controlled_smc_tf.py']
        if phase_name in ('c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_feasibility_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference','c1_rbf_fit_reference','c1_rbf_nnls_diagnosis','c2_rbf_paired_validation','c2_rbf_policy_accuracy','c2_rbf_nnls_accuracy','c2_rbf_nnls_validation','c2_rbf_fit_validation'):
            sources+=['docs/plans/controlled-smc-rbf-fitting-validation-20260929.md','docs/benchmarks/reference_rbf_fit.m']
            if resource=='gpu':sources+=['bayesfilter/replication/rbf_nnls_tf.py']
        if phase_name in ('c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference'):sources+=['docs/benchmarks/diagnose_rbf_reference_feasibility.py','docs/benchmarks/reference_rbf_svd_nnls.py']
        if resource=='gpu': sources+=['bayesfilter/runtime/gpu_memory_policy.py']
        if phase_name.startswith(('c2_','c3_')): sources+=['bayesfilter/replication/controlled_smc_tf.py']
        if phase_name.startswith('c3_guided_'): sources+=['bayesfilter/replication/gaussian_guided_pf_tf.py','docs/benchmarks/controlled_smc_filter_reference_tf.py','docs/plans/controlled-smc-guided-repair-20260929.md']
        if phase_name=='c2_full_parity': sources+=['docs/benchmarks/export_controlled_smc_full_fixture.m','docs/benchmarks/diagnose_controlled_smc_tf.py']
        if phase_name=='c3_cost_pilot': sources+=['docs/plans/controlled-smc-figure4-cost-pilot-20260929.md',AUTHOR+'NormConst_cSMC.m',AUTHOR+'NormConst_BPF.m',AUTHOR+'NormConst_Results.m']
        if phase_name=='c3_figure4': sources+=['docs/plans/controlled-smc-figure4-execution-20260929.md',AUTHOR+'NormConst_cSMC.m',AUTHOR+'NormConst_BPF.m',AUTHOR+'NormConst_Results.m']
        if phase_name=='c2_batch': sources+=['docs/plans/controlled-smc-repetition-batch-validation-20260929.md']
        if phase_name=='c2_kalman': sources+=['docs/plans/controlled-smc-kalman-validation-20260929.md']
        if phase_name=='k1_mixture_kalman': sources+=['bayesfilter/replication/mixture_kalman_reference_tf.py','docs/plans/kdm-mixture-kalman-validation-20260929.md']
        if phase_name=='k2_sequential': sources+=['bayesfilter/replication/kdm_sequential_tf.py','docs/benchmarks/reference_kdm_sequential.py','docs/plans/kdm-sequential-consumer-validation-20260929.md']
        if phase_name=='c1_lorenz_normalization':
            sources=[rel for rel in sources if not str(rel).startswith(AUTHOR)]
            sources+=['docs/benchmarks/reference_lorenz_normalization.m','docs/plans/controlled-smc-lorenz-normalization-check-20260929.md']
            sources+=[str(Path(AUTHOR).parent/'Lorenz'/name) for name in ('cSMC_Resample.m','RungeKutta4.m','Systematic_Resampling.m','APF_VaryF.m')]
        if phase_name in ('k1_iwsg','k2_sequential'):
            sources+=['bayesfilter/replication/kdm_iwsg_tf.py','docs/benchmarks/reference_kdm_iwsg.py',
                      'docs/plans/kdm-iwsg-primitive-validation-20260929.md',
                      '.localresources/code/younis-mdpf-neurips-2023/src/models/kde_particle_filter/kde_particle_filter.py']
            sources+=[str(p.relative_to(ROOT)) for p in
                      (ROOT/'.localresources/code/younis-mdpf-neurips-2023/packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation').glob('*.py')]
        if phase_name in ('c1_lorenz_consumer','c2_lorenz_consumer','c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            sources=[rel for rel in sources if not str(rel).startswith(AUTHOR)]
            sources+=['docs/benchmarks/reference_lorenz_consumer.m','docs/plans/controlled-smc-lorenz-consumer-repair-20260929.md']
            sources+=[str(Path(AUTHOR).parent/'Lorenz'/name) for name in ('cSMC_Resample.m','RungeKutta4.m','Systematic_Resampling.m','cSMC_VaryF.m')]
            if resource=='gpu': sources+=['bayesfilter/replication/multivariate_controlled_smc_tf.py']
        if phase_name in ('c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            sources+=['docs/plans/controlled-smc-lorenz-solver-validation-20260929.md']
        if phase_name in ('c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            sources+=['docs/plans/controlled-smc-lorenz-centered-recursion-20260929.md']
        if phase_name in ('c1_rbf_coefficient_stability','c1_rbf_matched_solver'):
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,'docs/plans/controlled-smc-rbf-fitting-validation-20260929.md','docs/benchmarks/reference_rbf_svd_nnls.py']
        if phase_name=='c1_rbf_matched_solver':sources+=['docs/benchmarks/diagnose_rbf_coefficient_stability.py']
        if phase_name in ('c1_rbf_paired_reference','c2_rbf_paired_validation','c1_rbf_paired_diagnosis'):sources+=['docs/plans/controlled-smc-rbf-paired-stopping-validation-20260929.md']
        if phase_name in ('c2_rbf_fixed_policy_sanity','c2_rbf_fixed_policy_precision','c2_rbf_policy_coverage','c2_rbf_coverage_mechanism','c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision','c2_rbf_full_refit','c2_rbf_refit_pilot','c2_rbf_refit_precision'):
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,
                     'docs/plans/controlled-smc-rbf-fixed-policy-sanity-20260930.md',
                     'bayesfilter/replication/rbf_controlled_smc_tf.py',
                     'bayesfilter/replication/controlled_smc_tf.py',
                     'bayesfilter/runtime/gpu_memory_policy.py']
        if phase_name in ('c2_rbf_policy_coverage','c2_rbf_coverage_mechanism'):sources+=['docs/benchmarks/diagnose_rbf_fixed_policy_tf.py']
        if phase_name=='c2_rbf_coverage_mechanism':sources+=['docs/benchmarks/diagnose_rbf_policy_coverage_tf.py']
        if phase_name in ('c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision'):sources+=['docs/plans/controlled-smc-rbf-retention-repair-20260930.md']
        if phase_name in REFIT_PHASES:
            sources+=[REFIT_PLAN]
        if phase_name=='c2_rbf_full_refit':
            sources+=['docs/benchmarks/diagnose_rbf_fixed_policy_tf.py','bayesfilter/replication/rbf_nnls_tf.py']
        if phase_name in LORENZ_REFINEMENT_PHASES:
            sources=[rel for rel in sources if not str(rel).startswith(AUTHOR)]
            sources += [LORENZ_REFINEMENT_PLAN,'docs/benchmarks/reference_lorenz_stable_fit.py']
            if phase_name in ('c1_lorenz_precision_diagnosis','c2_lorenz_precision_safety','c1_lorenz_trajectory_diagnosis','c2_lorenz_covariance_safety','c1_lorenz_covariance_validation','c2_lorenz_rank_guard','c1_lorenz_covariance_localization'):sources += [LORENZ_PRECISION_PLAN]
            if phase_name=='c2_lorenz_figure6_pilot':sources += [LORENZ_PILOT_PLAN]
            if phase_name.startswith('c2_lorenz_batch_'):sources += [LORENZ_BATCH_PLAN]
            sources += [str(Path(AUTHOR).parent/'Lorenz'/name) for name in
                ('cSMC_Resample.m','RungeKutta4.m','Systematic_Resampling.m','APF_VaryF.m','cSMC_VaryF.m')]
            if phase_name=='c1_lorenz_covariance_localization':sources+=['docs/benchmarks/validate_lorenz_covariance_reference.py']
            if resource=='gpu':
                sources += ['bayesfilter/replication/lorenz_refinement_tf.py',
                            'bayesfilter/replication/multivariate_controlled_smc_tf.py']
        if phase_name=='c1_lorenz_author_native':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,
                     'docs/plans/controlled-smc-lorenz-author-native-audit-20260930.md',
                     'docs/benchmarks/reference_lorenz_author_native.m']
            sources += [str(Path(AUTHOR).parent/'Lorenz'/name) for name in
                ('cSMC_Resample.m','RungeKutta4.m','Systematic_Resampling.m','cSMC_VaryF.m','SimulatedData.mat')]
        if phase_name in LORENZ_SOURCE_PORT_PHASES:
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,LORENZ_SOURCE_PORT_PLAN]
            sources += [str(Path(AUTHOR).parent/'Lorenz'/name) for name in
                ('cSMC_Resample.m','RungeKutta4.m','Systematic_Resampling.m','cSMC_VaryF.m','SimulatedData.mat')]
            if resource=='cpu':
                sources += ['docs/benchmarks/reference_lorenz_author_capture.m']
            else:
                sources += ['bayesfilter/replication/multivariate_controlled_smc_tf.py',
                            'bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py']
        if phase_name in LORENZ_MATRIX_PHASES:
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,LORENZ_MATRIX_PLAN,
                     str(Path(AUTHOR).parent/'Lorenz/cSMC_Resample.m')]
            if resource=='cpu':
                sources+=['docs/benchmarks/reference_lorenz_matrix_crossover.m']
                if phase['mode']=='fallback':sources+=['.localresources/code/octave-6.4.0-dMatrix.cc']
                if phase['mode']=='failed_replay':
                    sources+=['docs/benchmarks/reference_lorenz_failed_source_replay.m']
                    sources+=[str(Path(AUTHOR).parent/'Lorenz'/name) for name in ('RungeKutta4.m','cSMC_VaryF.m','SimulatedData.mat')]
            else:
                sources+=['bayesfilter/replication/multivariate_controlled_smc_tf.py',
                          'bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py']
        if phase_name in LORENZ_BOUNDED_PHASES:
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,LORENZ_BOUNDED_PLAN,
                     'bayesfilter/replication/multivariate_controlled_smc_tf.py',
                     'bayesfilter/replication/lorenz_refinement_tf.py',
                     'bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py',
                     'docs/benchmarks/reference_lorenz_stable_fit.py',
                     'docs/benchmarks/validate_lorenz_covariance_reference.py',
                     str(Path(AUTHOR).parent/'Lorenz/cSMC_Resample.m')]
        if phase_name in LORENZ_BOUNDED_PHASES:
            sources+=['docs/benchmarks/reference_lorenz_bounded_spd.py','.localresources/code/tensorflow-v2.20.0-xla-self-adjoint-eig-op.cc','.localresources/code/xla-eigh-expander.cc']
        if phase_name=='c1_lorenz_bounded_trigger_transition':
            sources+=[str(Path(AUTHOR).parent/'Lorenz'/name) for name in ('RungeKutta4.m','cSMC_VaryF.m')]
        if phase_name in LORENZ_ROOT_CAUSE_PHASES:
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,LORENZ_ROOT_CAUSE_PLAN,
                'docs/benchmarks/diagnose_lorenz_policy_geometry.py',
                'docs/benchmarks/reference_lorenz_stable_fit.py',
                'bayesfilter/replication/multivariate_controlled_smc_tf.py',
                'bayesfilter/replication/lorenz_refinement_tf.py','bayesfilter/replication/controlled_smc_tf.py']
            sources += [str(Path(AUTHOR).parent/'Lorenz'/name) for name in ('RungeKutta4.m','cSMC_VaryF.m','cSMC_Resample.m')]
        if phase_name in LORENZ_REPAIR_PHASES:
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',MASTER,LORENZ_REPAIR_PLAN,
                'docs/benchmarks/diagnose_lorenz_recursion_localization.py','docs/benchmarks/validate_lorenz_first_failure_tf.py',
                'bayesfilter/runtime/gpu_memory_policy.py',
                'docs/benchmarks/reference_lorenz_stable_fit.py','docs/benchmarks/validate_lorenz_covariance_reference.py',
                'bayesfilter/replication/multivariate_controlled_smc_tf.py',
                'bayesfilter/replication/lorenz_refinement_tf.py','bayesfilter/replication/controlled_smc_tf.py',
                str(CAMPAIGN.relative_to(ROOT)/'pre-r1-multivariate-20261002.py'),str(CAMPAIGN.relative_to(ROOT)/'pre-r1-controller-20261002.py')]
        if phase_name == 'c0_source_lock':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,C0_SOURCE_LOCK_PLAN,
                'docs/reference/controlled-smc-remaining-source-audit-20260929.md']
        if phase_name == 'c1_source_cost_pilot':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,C1_SOURCE_PILOT_PLAN,
                'docs/reference/controlled-smc-remaining-source-audit-20260929.md']
        if phase_name == 'c1_source_full':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,C1_FULL_SOURCE_PLAN,FULL_SOURCE_SPEC,
                SOURCE_MEMORY_REPAIR_PLAN,'docs/reference/controlled-smc-remaining-source-audit-20260929.md']
        if phase_name in ('c1_source_isolation_check','c1_source_failure_replay'):
            sources=[phase['script'],SOURCE_RUNNER,__file__,'scripts/run_replication_master.py',
                PLAN,MASTER,SOURCE_MEMORY_REPAIR_PLAN,SOURCE_FAILURE_REPLAY_PLAN]
        if phase_name == 'c2_source_results':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,
                     C1_FULL_SOURCE_PLAN,FULL_SOURCE_SPEC,C2_SOURCE_RESULTS_PLAN]
        if phase_name == 'k0_source_config':
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,K0_SOURCE_CONFIG_PLAN]
        if phase_name in P0_FORWARD_PHASES:
            phase_plan=P0_TF_SOLVER_PLAN if phase_name in P0_TF_SOLVER_PHASES else P0_OFFSET_PLAN if phase_name=='p0_offset_localization' else P0_PERTURB_PLAN if phase_name in P0_PERTURB_PHASES else P0_SOLVER_PLAN if phase_name in ('p0_solver_semantics','p0_solver_system_reference') else P0_FITTING_PLAN if phase_name=='p0_fitting_forward' else P0_FORWARD_PLAN
            sources=[phase['script'],__file__,'scripts/run_replication_master.py',PLAN,MASTER,phase_plan]
            if phase_name=='p0_solver_system_reference':
                sources+=['docs/benchmarks/reference_octave_system_lapack.py']
            if resource=='gpu':
                sources+=['bayesfilter/replication/multivariate_controlled_smc_tf.py',
                          'bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py']
        for rel in sources:
            if not (ROOT/rel).is_file(): raise RuntimeError('Missing implementation: '+rel)
        output.mkdir(); (output/'sources').mkdir()
        hashes={}
        for rel in sources:
            source=ROOT/rel
            relative=source.relative_to(ROOT)
            target=output/'sources'/(Path(source.name) if str(rel)==phase['script'] else relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes()); hashes[str(source.relative_to(ROOT))]=shared.sha(source)
        script=output/'sources'/Path(phase['script']).name
        if phase_name in ('c0_source_lock','k0_source_config'):
            command=[PYTHON,'-B',str(script),'--output',str(output),'--root',str(ROOT)]
        elif phase_name in ('c1_source_isolation_check','c1_source_failure_replay'):
            locked=[m for _,m in campaign_attempts() if m['phase']=='c0_source_lock' and m['status']=='COMPLETE'][-1]
            source_lock=Path(locked['output_root'])
            command=[PYTHON,'-B',str(script),'--output',str(output),'--root',str(ROOT),
                '--source-lock',str(source_lock),'--source-runner',str(output/'sources'/SOURCE_RUNNER)]
            if phase_name == 'c1_source_failure_replay':
                command+=['--failure',str(SOURCE_FAILURE_EVIDENCE)]
        elif phase_name in ('c1_source_cost_pilot','c1_source_full'):
            source_lock=Path([m['output_root'] for _,m in campaign_attempts()
                if m['phase']=='c0_source_lock' and m['status']=='COMPLETE' and m.get('decision',{}).get('status')=='PASS'][-1])
            command=[PYTHON,'-B',str(script),'--output',str(output),'--root',str(ROOT),
                '--source-lock',str(source_lock),'--mode','pilot','--driver-seconds','45']
            prior_pilots=[m for _,m in campaign_attempts() if m['phase']==phase_name and m['status']!='RUNNING']
            if prior_pilots and phase_name == 'c1_source_cost_pilot':
                prior_output=Path(prior_pilots[-1]['output_root'])
                previous=json.loads((prior_output/'summary.json').read_text())
                unfinished=[x['driver'] for x in previous['drivers'] if x['status']!='COMPLETE']
                if unfinished != ['neuroscience_bpf']:
                    raise RuntimeError('Retry requires a newly reviewed failure classification')
                command[command.index('--driver-seconds')+1]=str(min(130,timeout-2))
                command+=['--only-driver','neuroscience_bpf','--resume-pilot',str(prior_output)]
            if phase_name == 'c1_source_full':
                command=[PYTHON,'-B',str(script),'--output',str(output),'--root',str(ROOT),
                    '--source-lock',str(source_lock),'--mode','full',
                    '--full-spec',str(output/'sources'/FULL_SOURCE_SPEC)]
                interrupted=[m for _,m in campaign_attempts() if m['phase']=='c1_source_full']
                if interrupted:
                    command+=['--isolate-cases','--compare-prefix',interrupted[-1]['output_root']]
        elif phase_name == 'c2_source_results':
            source_attempt=completed_source_attempt()
            command=source_analysis_command(script,output,source_attempt)
        elif resource=='cpu' and phase_name not in LORENZ_REPAIR_PHASES and phase_name not in LORENZ_ROOT_CAUSE_PHASES and phase_name not in LORENZ_MATRIX_PHASES and phase_name not in LORENZ_BOUNDED_PHASES and phase_name not in ('c1_lorenz_author_native','c1_lorenz_author_capture') and phase_name not in LORENZ_REFINEMENT_PHASES and phase_name not in ('c1_rbf_paired_diagnosis','c1_rbf_matched_solver','c1_rbf_coefficient_stability','c1_rbf_truncation_diagnosis','c1_rbf_nnls_diagnosis','c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_feasibility_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference','c1_rbf_fit_reference','c1_rbf_consumer_audit','c1_lorenz_normalization','c1_lorenz_consumer','c1_lorenz_solver_reference','c1_lorenz_centered_reference'):
            command=[PYTHON,'-B',str(output/'sources/diagnose_controlled_smc_reference.py'),'--output',str(output),'--mode',phase['mode']]
        else:
            command=[PYTHON,'-B',str(script),'--output',str(output)]
            if phase_name.startswith('c2_') and phase_name not in LORENZ_REPAIR_PHASES and phase_name not in LORENZ_MATRIX_PHASES and phase_name not in LORENZ_BOUNDED_PHASES and phase_name not in LORENZ_SOURCE_PORT_PHASES and phase_name not in LORENZ_REFINEMENT_PHASES and phase_name not in ('c2_rbf_full_refit','c2_rbf_refit_pilot','c2_rbf_refit_precision','c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision','c2_rbf_coverage_mechanism','c2_rbf_policy_coverage','c2_rbf_fixed_policy_precision','c2_rbf_fixed_policy_sanity','c2_rbf_paired_validation','c2_rbf_policy_accuracy','c2_rbf_nnls_accuracy','c2_rbf_nnls_validation','c2_rbf_fit_validation','c2_rbf_forward','c2_lorenz_consumer','c2_lorenz_solver_validation','c2_lorenz_centered_validation'):
                fixtures=[m for _,m in campaign_attempts() if m['phase']=='c1_pilot' and m['status']=='COMPLETE']
                command+=['--fixture',str(Path(fixtures[-1]['output_root'])/'fixture.json')]
        if phase_name=='c1_rbf_nnls_diagnosis':command+=['--diagnose-only']
        if phase_name=='c2_rbf_nnls_accuracy':command+=['--accuracy-calibration']
        if phase_name=='c1_rbf_svd_diagnosis':command+=['--svd-only']
        if phase_name in ('c1_rbf_svd_reference','c1_rbf_paired_reference'):command+=['--svd-reference']
        if phase_name=='c1_rbf_paired_reference':command+=['--stopping-factor','0.1','--seeds','20261009','20261010']
        inputs={}
        if phase_name in P0_FORWARD_PHASES:
            command,inputs=p0_forward_inputs(phase_name,output)
        if phase_name == 'c2_source_results':
            inputs={str(source_attempt/name):shared.sha(source_attempt/name) for name in ('manifest.json','summary.json')}
        if phase_name in ('c1_source_cost_pilot','c1_source_full','c1_source_isolation_check','c1_source_failure_replay'):
            inputs[str(source_lock/'source-lock.json')]=shared.sha(source_lock/'source-lock.json')
        if phase_name == 'c1_source_failure_replay':
            inputs.update({str(p):shared.sha(p) for p in SOURCE_FAILURE_EVIDENCE.iterdir() if p.is_file()})
        if phase_name == 'c1_source_full' and '--compare-prefix' in command:
            prefix=Path(command[command.index('--compare-prefix')+1])
            inputs[str(prefix/'manifest.json')]=shared.sha(prefix/'manifest.json')
            for case_file in (prefix/'reference-work').rglob('*-cases.csv'):
                inputs[str(case_file)]=shared.sha(case_file)
        if phase_name in LORENZ_REPAIR_PHASES:
            assert Path(command[2]).name==Path(phase['script']).name, 'Repair dispatch mismatch'
            command+=['--root',str(ROOT)]
            for name in ('attempt69_c1_lorenz_refinement_reference','attempt76_c2_lorenz_batch_validation','attempt108_c2_lorenz_bounded_integral_safety','attempt111_c2_lorenz_bounded_crossover_capture','attempt121_c2_lorenz_bounded_trigger_safety'):
                for source in (CAMPAIGN/name).rglob('*'):
                    if source.is_file() and source.suffix in ('.tensor','.json','.bin'):inputs[str(source)]=shared.sha(source)
        if phase_name in LORENZ_ROOT_CAUSE_PHASES:
            assert Path(command[2]).name==Path(phase['script']).name, 'Audit script dispatch mismatch'
            design=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
            command+=['--root',str(ROOT),'--design',str(design)]
            files=[design/'rejected_batch/normal.tensor']
            for name in ('attempt121_c2_lorenz_bounded_trigger_safety','attempt126_c2_lorenz_bounded_margin_safety'):
                candidate=CAMPAIGN/name
                command+=['--candidate',str(candidate)]
                files += [candidate/'summary.json']+list((candidate/'covariance_floor_0.01').glob('*.tensor'))
            inputs.update({str(p):shared.sha(p) for p in files})
        if phase_name in LORENZ_BOUNDED_PHASES:
            reference=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference'
            design=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
            records=campaign_attempts()
            def completed(name):
                return Path([m for _,m in records if m['phase']==name and m['status']=='COMPLETE'][-1]['output_root'])
            files=[reference/'fixture.json']+sorted(reference.glob('*.bin'))+sorted((design/'rejected_batch').glob('*.tensor'))
            if phase['mode'] in ('calibration','strict_calibration','refined_calibration','integral_calibration'):
                local=completed('c1_lorenz_bounded_margin_reference' if 'margin_' in phase_name else 'c1_lorenz_bounded_trigger_reference' if 'trigger_' in phase_name else 'c1_lorenz_bounded_condition_reference' if 'condition_' in phase_name else 'c1_lorenz_bounded_integral_reference' if phase['mode']=='integral_calibration' else 'c1_lorenz_bounded_reference')
                command+=['--reference',str(local)]
                if 'trigger_' in phase_name:command+=['--activation-margin',str(64*2.220446049250313e-16)]
                if 'margin_' in phase_name:command+=['--activation-margin','0.001']
                if phase['mode'] in ('refined_calibration','integral_calibration'):command+=['--eigen-refinement']
                if phase['mode']=='integral_calibration':command+=['--stable-integral']
                files+=[local/'summary.json',local/'calibration.json']
                if phase['mode'] in ('strict_calibration','refined_calibration','integral_calibration'):command+=['--eigen-tolerance',str(8*2.220446049250313e-16)]
            else:
                command+=['--reference',str(reference)]
                if resource=='cpu' and phase['mode'] not in ('localization','crossover'):command+=['--design',str(design),'--mode',phase['mode']]
                if phase['mode'] in ('capture','crossover'):
                    candidate=completed('c2_lorenz_bounded_integral_safety')
                    command+=['--candidate',str(candidate)]
                    files+=[candidate/'summary.json']+sorted(candidate.glob('covariance_floor_*/*.tensor'))
                    if phase['mode']=='crossover':
                        capture=completed('c2_lorenz_bounded_crossover_capture')
                        command+=['--capture',str(capture)]
                        if phase_name=='c1_lorenz_bounded_reference_arithmetic':command+=['--stable-all','--design',str(design)]
                        files+=[capture/'summary.json']+sorted(capture.glob('*.tensor'))
                if phase['mode']=='integral_reference':
                    calibration=completed('c1_lorenz_bounded_reference')
                    command+=['--calibration',str(calibration)]
                    if 'condition_' in phase_name:command+=['--condition-grid']
                    if 'trigger_' in phase_name:command+=['--activation-margin',str(64*2.220446049250313e-16),'--repair-floor','0.01']
                    if 'margin_' in phase_name:command+=['--activation-margin','0.001','--repair-floor','0.01']
                    files+=[calibration/'calibration.json']
                if phase['mode']=='safety':
                    calibration=completed('c2_lorenz_bounded_margin_calibration' if 'margin_' in phase_name else 'c2_lorenz_bounded_trigger_calibration' if 'trigger_' in phase_name else 'c2_lorenz_bounded_condition_calibration' if 'condition_' in phase_name else 'c2_lorenz_bounded_integral_calibration' if 'integral_' in phase_name else 'c2_lorenz_bounded_refined_calibration')
                    command+=['--candidate',str(design),'--calibration',str(calibration)]
                    files+=[calibration/'summary.json']
                if phase['mode'] in ('validation','localization','history','transition'):
                    candidate=completed('c2_lorenz_bounded_margin_safety' if 'margin_' in phase_name else 'c2_lorenz_bounded_trigger_safety' if 'trigger_' in phase_name else 'c2_lorenz_bounded_condition_safety' if 'condition_' in phase_name else 'c2_lorenz_bounded_integral_safety' if 'integral_' in phase_name else 'c2_lorenz_bounded_safety')
                    command+=['--candidate',str(candidate)]
                    if phase['mode']=='localization':
                        command+=['--relative-floor','0.001','--candidate-folder','covariance_floor_0.001']
                        if 'integral_' in phase_name:command+=['--stable-integral']
                    files+=[candidate/'summary.json']+sorted(candidate.glob('covariance_floor_*/*.tensor'))
            inputs.update({str(path):shared.sha(path) for path in files})
        if phase_name in LORENZ_MATRIX_PHASES:
            dependency=('c1_lorenz_author_native' if phase['mode']=='failed_replay' else phase['dependencies'][0])
            records=[m for _,m in campaign_attempts() if m['phase']==dependency and m['status']=='COMPLETE']
            reference=Path(records[-1]['output_root'])
            command+=['--reference',str(reference)]
            if resource=='cpu':command+=['--mode',phase['mode']]
            files=[reference/'summary.json']
            if phase['mode']=='failed_replay':
                candidate=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
                fixture=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference/fixture.json'
                command+=['--candidate',str(candidate),'--fixture',str(fixture)]
                files+=[reference/'manifest.json',fixture,candidate/'failure-diagnostics.json']+sorted(reference.glob('author-*.mat'))+sorted((candidate/'rejected_batch').glob('*.tensor'))
            else:files+=[reference/'fixture.json']+sorted(reference.glob('*.bin'))
            inputs.update({str(path):shared.sha(path) for path in files})
        if phase_name in LORENZ_SOURCE_PORT_PHASES:
            dependency=phase['dependencies'][0]
            records=[m for _,m in campaign_attempts() if m['phase']==dependency and m['status']=='COMPLETE']
            reference=Path(records[-1]['output_root'])
            command+=['--reference',str(reference)]
            files=([reference/'summary.json']+sorted(reference.glob('author-*.mat')) if resource=='cpu'
                   else [reference/'fixture.json']+sorted(reference.glob('*.bin')))
            inputs.update({str(p):shared.sha(p) for p in files})
        if phase_name in LORENZ_REFINEMENT_PHASES:
            if phase_name=='c1_lorenz_covariance_localization':
                reference=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference'
                candidate=CAMPAIGN/'attempt80_c2_lorenz_covariance_safety'
                command+=['--reference',str(reference),'--candidate',str(candidate)]
                for item in [reference/'fixture.json',candidate/'summary.json']+sorted(candidate.glob('nearest_covariance/*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif phase_name=='c2_lorenz_rank_guard':
                healthy=CAMPAIGN/'attempt70_c2_lorenz_refinement_validation'
                candidate=CAMPAIGN/'attempt80_c2_lorenz_covariance_safety'
                design=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
                command+=['--healthy',str(healthy),'--candidate',str(candidate),'--design',str(design)]
                for item in [candidate/'summary.json',candidate/'sources/multivariate_controlled_smc_tf.py']+sorted(candidate.glob('nearest_covariance/*.tensor'))+sorted(design.glob('rejected_batch/*.tensor'))+sorted(healthy.glob('*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif phase_name=='c1_lorenz_covariance_validation':
                reference=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference'
                candidate=CAMPAIGN/'attempt80_c2_lorenz_covariance_safety'
                design=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
                command+=['--reference',str(reference),'--candidate',str(candidate),'--design',str(design)]
                for item in [reference/'fixture.json',candidate/'summary.json']+sorted(candidate.glob('nearest_covariance/*.tensor'))+sorted(design.glob('rejected_batch/*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif phase_name=='c1_lorenz_trajectory_diagnosis':
                candidate=CAMPAIGN/'attempt78_c2_lorenz_precision_safety'
                command+=['--candidate',str(candidate)]
                for item in [candidate/'summary.json']+sorted(candidate.glob('*/*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif phase_name=='c1_lorenz_precision_diagnosis':
                reference=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference'
                candidate=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
                command+=['--reference',str(reference),'--candidate',str(candidate)]
                for item in [reference/'fixture.json',candidate/'summary.json',candidate/'failure-diagnostics.json']+sorted((candidate/'rejected_batch').glob('*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif phase_name in ('c1_lorenz_refinement_diagnosis','c1_lorenz_refinement_matched_validation'):
                reference=CAMPAIGN/'attempt69_c1_lorenz_refinement_reference'
                candidate=CAMPAIGN/'attempt70_c2_lorenz_refinement_validation'
                command+=['--reference',str(reference),'--candidate',str(candidate)]
                for item in [reference/'fixture.json',reference/'summary.json',candidate/'summary.json']+sorted(reference.glob('*.bin'))+sorted(candidate.glob('case*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            elif resource=='cpu':
                references=[m for _,m in campaign_attempts() if m['phase']=='c1_lorenz_consumer' and m['status']=='COMPLETE']
                reference=Path(references[-1]['output_root'])/'fixture.json'
                command+=['--fixture',str(reference)]
                inputs[str(reference)]=shared.sha(reference)
            else:
                references=[m for _,m in campaign_attempts() if m['phase']=='c1_lorenz_refinement_reference' and m['status']=='COMPLETE']
                reference=Path(references[-1]['output_root'])
                command+=['--reference',str(reference)]
                for item in [reference/'fixture.json',reference/'summary.json']+sorted(reference.glob('*.bin')):
                    inputs[str(item)]=shared.sha(item)
            if phase_name in ('c2_lorenz_precision_safety','c2_lorenz_covariance_safety'):
                candidate=CAMPAIGN/'attempt76_c2_lorenz_batch_validation'
                command+=['--candidate',str(candidate)]
                if phase_name=='c2_lorenz_covariance_safety':command+=['--policies','nearest_covariance']
                for item in [candidate/'summary.json',candidate/'failure-diagnostics.json']+sorted((candidate/'rejected_batch').glob('*.tensor')):
                    inputs[str(item)]=shared.sha(item)
            if phase_name.startswith('c2_lorenz_batch_'):
                command+=['--mode',phase['mode']]
                scalar_pilot=CAMPAIGN/'attempt73_c2_lorenz_figure6_pilot'
                command+=['--scalar-pilot',str(scalar_pilot)]
                inputs[str(scalar_pilot/'summary.json')]=shared.sha(scalar_pilot/'summary.json')
            data=ROOT/Path(AUTHOR).parent/'Lorenz/SimulatedData.mat'
            inputs[str(data)]=shared.sha(data)
        if phase_name in ('c2_rbf_fixed_policy_sanity','c2_rbf_fixed_policy_precision','c2_rbf_policy_coverage','c2_rbf_coverage_mechanism','c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision','c2_rbf_full_refit','c2_rbf_refit_pilot','c2_rbf_refit_precision'):
            reference=CAMPAIGN/'attempt57_c1_rbf_paired_reference/fixture.json'
            candidate=CAMPAIGN/'attempt58_c2_rbf_paired_validation'
            previous=json.loads((candidate/'summary.json').read_text())
            required=('backward_fit_valid','fitted_forward_valid','fresh_fitted_consumer_valid')
            flags={row['name']:row['passed'] for row in previous['checks']}
            if not all(flags.get(key,False) for key in required):
                raise RuntimeError('Saved policy has not passed its numerical validity checks')
            command+=['--fixture',str(reference)]
            if phase_name!='c2_rbf_full_refit':command+=['--candidate',str(candidate)]
            for item in [reference,candidate/'summary.json']+[candidate/('fit_'+name+'.tensor') for name in ('beta','weights','knots','constant')]:
                inputs[str(item)]=shared.sha(item)
        if phase_name in ('c2_rbf_refit_pilot','c2_rbf_refit_precision'):
            refit=CAMPAIGN/'attempt66_c2_rbf_full_refit'
            command+=['--refit',str(refit),'--full-retention','--seed-base',
                      '202609340' if phase_name.endswith('_pilot') else '202609350']
            for item in [refit/'summary.json',candidate/'fit_coefficients.tensor']+[refit/('fit_'+name+'.tensor') for name in ('beta','constant','coefficients')]:
                inputs[str(item)]=shared.sha(item)
            if phase_name.endswith('_precision'):
                command+=['--repetitions','4096']
                for name in ('summary.json','progress.json'):
                    item=CAMPAIGN/'attempt67_c2_rbf_refit_pilot'/name
                    inputs[str(item)]=shared.sha(item)
        if phase_name=='c2_rbf_fixed_policy_precision':command+=['--repetitions','4096','--seed-base','202609310']
        if phase_name=='c2_rbf_policy_coverage':
            precision=CAMPAIGN/'attempt61_c2_rbf_fixed_policy_precision/summary.json'
            command+=['--precision',str(precision)]
            inputs[str(precision)]=shared.sha(precision)
        if phase_name=='c2_rbf_coverage_mechanism':
            coverage=CAMPAIGN/'attempt62_c2_rbf_policy_coverage'
            command+=['--coverage',str(coverage)]
            for item in [coverage/'summary.json',coverage/'grid2049.json',coverage/'grid4097.json',candidate/'fit_coefficients.tensor']:
                inputs[str(item)]=shared.sha(item)
        if phase_name in ('c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision'):
            command+=['--full-retention','--seed-base','202609320' if phase_name.endswith('_pilot') else '202609330']
            mechanism=CAMPAIGN/'attempt63_c2_rbf_coverage_mechanism/summary.json'
            for item in (mechanism,candidate/'fit_coefficients.tensor'):inputs[str(item)]=shared.sha(item)
            if phase_name.endswith('_precision'):
                command+=['--repetitions','4096']
                for name in ('summary.json','progress.json'):
                    item=CAMPAIGN/'attempt64_c2_rbf_full_retention_pilot'/name
                    inputs[str(item)]=shared.sha(item)
        if phase_name=='c1_rbf_paired_reference':
            paired=CAMPAIGN/'attempt56_c1_rbf_matched_solver/summary.json'
            if .1 not in json.loads(paired.read_text())['matched_policy_passing_factors']:raise RuntimeError('Paired stopping hypothesis did not pass calibration')
            inputs[str(paired)]=shared.sha(paired)
        if phase_name=='c2_rbf_paired_validation':
            paired=[m for _,m in campaign_attempts() if m['phase']=='c1_rbf_paired_reference' and m['status']=='COMPLETE']
            reference=Path(paired[-1]['output_root'])/'fixture.json'
            inputs[str(reference)]=shared.sha(reference)
            command+=['--fixture',str(reference),'--stopping-factor','0.1','--fresh-seed','20261011']
        if phase_name in ('c1_rbf_coefficient_stability','c1_rbf_matched_solver'):
            reference=CAMPAIGN/'attempt50_c1_rbf_svd_reference/fixture.json'
            diagnosis=CAMPAIGN/'attempt53_c1_rbf_truncation_diagnosis'
            candidate=CAMPAIGN/'attempt54_c2_rbf_policy_accuracy'
            command+=['--fixture',str(reference),'--diagnosis',str(diagnosis),'--candidate',str(candidate)]
            for item in [reference,candidate/'summary.json',candidate/'frozen_coefficients.tensor']+sorted(diagnosis.glob('frozen_seed*.json')):
                inputs[str(item)]=shared.sha(item)
        if phase_name=='c1_rbf_matched_solver':
            stability=CAMPAIGN/'attempt55_c1_rbf_coefficient_stability'
            command+=['--stability',str(stability)]
            for item in [stability/'summary.json']+sorted(stability.glob('reference_seed*.npy')):
                inputs[str(item)]=shared.sha(item)
        if phase_name=='c2_rbf_policy_accuracy':
            diagnosis=CAMPAIGN/'attempt53_c1_rbf_truncation_diagnosis'
            first=diagnosis/'frozen_seed20261006.json';second=diagnosis/'frozen_seed20261007.json'
            command+=['--fixture',str(first),'--policy-calibration',str(second)]
            inputs[str(first)]=shared.sha(first);inputs[str(second)]=shared.sha(second)

        if phase_name in ('c1_rbf_truncation_diagnosis','c1_rbf_paired_diagnosis'):
            reference=CAMPAIGN/'attempt50_c1_rbf_svd_reference/fixture.json'
            candidate=CAMPAIGN/'attempt51_c2_rbf_fit_validation'
            if phase_name=='c1_rbf_paired_diagnosis':
                reference=CAMPAIGN/'attempt57_c1_rbf_paired_reference/fixture.json'
                candidate=CAMPAIGN/'attempt58_c2_rbf_paired_validation'
            command+=['--fixture',str(reference),'--candidate',str(candidate)]
            inputs[str(reference)]=shared.sha(reference)
            for item in candidate.glob('fit_*.tensor'):inputs[str(item)]=shared.sha(item)

        if phase_name in ('c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis'):
            reference=CAMPAIGN/'attempt47_c1_rbf_scipy_reference/nnls_failure_seed20261006_time82.npz'
            command+=['--fixture',str(reference)]
            inputs[str(reference)]=shared.sha(reference)

        if phase_name=='c1_rbf_feasibility_diagnosis':
            reference=CAMPAIGN/'attempt44_c1_rbf_scipy_reference/nnls_failure_seed20261006_time90.npz'
            command+=['--fixture',str(reference)]
            inputs[str(reference)]=shared.sha(reference)

        if phase_name in ('c2_rbf_nnls_accuracy','c2_rbf_nnls_validation'):
            references=[m for _,m in campaign_attempts() if m['phase']=='c1_rbf_nnls_diagnosis' and m['status']=='COMPLETE']
            reference=Path(references[-1]['output_root'])/'fixture.json'
            command+=['--fixture',str(reference)]
            inputs[str(reference)]=shared.sha(reference)
        if phase_name=='c2_rbf_fit_validation':
            calibration=[m for _,m in campaign_attempts() if m['phase']=='c2_rbf_policy_accuracy' and m['status']=='COMPLETE']
            calibration_path=Path(calibration[-1]['output_root'])/'summary.json'
            factor=json.loads(calibration_path.read_text())['selected_stopping_factor']
            if factor is None:raise RuntimeError('No retained-policy-qualified stopping factor')
            command+=['--stopping-factor',str(factor)]
            inputs[str(calibration_path)]=shared.sha(calibration_path)
        if phase_name=='c2_rbf_fit_validation':
            references=[m for _,m in campaign_attempts() if m['phase']=='c1_rbf_svd_reference' and m['status']=='COMPLETE']
            reference=Path(references[-1]['output_root'])/'fixture.json'
            command+=['--fixture',str(reference)]
            inputs[str(reference)]=shared.sha(reference)
        if phase_name=='c2_rbf_forward':
            references=[m for _,m in campaign_attempts() if m['phase']=='c1_rbf_consumer_audit' and m['status']=='COMPLETE']
            reference=Path(references[-1]['output_root'])/'fixture.json'
            command+=['--fixture',str(reference)]
            inputs[str(reference)]=shared.sha(reference)
        if phase_name in ('c2_lorenz_consumer','c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            references=[m for _,m in campaign_attempts() if m['phase']=='c1_lorenz_consumer' and m['status']=='COMPLETE']
            fixture=Path(references[-1]['output_root'])/'fixture.json'
            command+=['--fixture',str(fixture)]
            inputs[str(fixture)]=shared.sha(fixture)
        if phase_name in ('c2_lorenz_solver_validation','c2_lorenz_centered_validation'):
            references=[m for _,m in campaign_attempts() if m['phase']==('c1_lorenz_centered_reference' if phase_name=='c2_lorenz_centered_validation' else 'c1_lorenz_solver_reference') and m['status']=='COMPLETE']
            reference=Path(references[-1]['output_root'])/'reference.json'
            command+=['--reference',str(reference)]
            inputs[str(reference)]=shared.sha(reference)
        if phase_name in ('c1_lorenz_normalization','c1_lorenz_consumer','c2_lorenz_consumer','c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            data=ROOT/Path(AUTHOR).parent/'Lorenz/SimulatedData.mat'
            inputs[str(data)]=shared.sha(data)
        if phase_name=='c3_figure4':
            inputs.update({str(ROOT/AUTHOR/name):shared.sha(ROOT/AUTHOR/name) for name in ('NormConst_cSMC.mat','NormConst_BPF.mat')})
        if phase_name=='c2_batch':
            baseline=CAMPAIGN/'attempt12_c2_full_parity/sources/controlled_smc_tf.py'
            observations=CAMPAIGN/'attempt07_c2_full/observations.json'
            command+=['--baseline',str(baseline),'--observations',str(observations)]
            inputs.update({str(p):shared.sha(p) for p in (baseline,observations)})
        if phase_name=='c2_full_parity':
            references=[m for _,m in campaign_attempts() if m['phase']=='c1_demo' and m['status']=='COMPLETE']
            reference=Path(references[-1]['output_root'])/'author-result.octave'
            command+=['--reference-result',str(reference)]
            inputs[str(reference)]=shared.sha(reference)
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   TF_FORCE_GPU_ALLOW_GROWTH='true',PYTHONDONTWRITEBYTECODE='1',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
        if resource=='cpu': env['CUDA_VISIBLE_DEVICES']='-1'
        else: env.pop('CUDA_VISIBLE_DEVICES',None)
        manifest=dict(schema='controlled_smc_kdm_attempt_v1',phase=phase_name,status='RUNNING',
            resource=resource,cpu_only=resource=='cpu',gpu_intentionally_hidden=resource=='cpu',
            started_utc=shared.stamp(),started_epoch=time.time(),timeout_seconds=timeout,command=command,
            plan=({'k2_sequential':'docs/plans/kdm-sequential-consumer-validation-20260929.md','k1_mixture_kalman':'docs/plans/kdm-mixture-kalman-validation-20260929.md','c3_figure4':'docs/plans/controlled-smc-figure4-execution-20260929.md','c2_batch':'docs/plans/controlled-smc-repetition-batch-validation-20260929.md','c3_cost_pilot':'docs/plans/controlled-smc-figure4-cost-pilot-20260929.md','k1_iwsg':'docs/plans/kdm-iwsg-primitive-validation-20260929.md','c2_kalman':'docs/plans/controlled-smc-kalman-validation-20260929.md'}.get(phase_name,PLAN)),master=MASTER,result_file=str(output/'summary.json'),output_root=str(output),cwd=str(ROOT),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256=hashes,input_artifacts_sha256=inputs,data_sha256=('N/A: constructed identity fixture, preserved in diagnostic source and summary' if phase_name in ('k1_iwsg','k1_mixture_kalman','k2_sequential','c2_kalman') else shared.sha(ROOT/AUTHOR/'Data_NeuroScience.mat')),seeds=('N/A: deterministic constructed fixture' if phase_name=='k1_mixture_kalman' else [20260929]),
            author_commit=('b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18' if phase_name.startswith(('k1_','k2_')) else '2696f6d611aae172d10102cb7bfbfd314bb6b88f'),
            statistics_commit=('N/A: TensorFlow/PyTorch phase' if resource=='gpu' else 'bfb1b9f2e5b38a9200fba77e20d5e5517b839edf'),
            environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_FORCE_GPU_ALLOW_GROWTH')},
            interpreter=command[0],supervisor_pid=os.getpid(),supervisor_identity=shared.proc_identity(os.getpid()))
        if phase_name == 'c0_source_lock':
            manifest.update(plan=C0_SOURCE_LOCK_PLAN,scientific_role='source_provenance_lock',
                data_sha256='Recorded per source file in source-lock.json',
                seeds='N/A: metadata only',statistics_commit='N/A: no source filter execution',
                reference_backend='CPU metadata; CUDA intentionally hidden')
        if phase_name == 'c1_source_cost_pilot':
            manifest.update(plan=C1_SOURCE_PILOT_PLAN,scientific_role='author_source_cost_pilot',
                data_sha256='Recorded per source file in the verified C0 source-lock.json',
                seeds=[202610031,202610032,202610033,202610034],
                reference_backend='Independent Octave CPU source reference; CUDA intentionally hidden',
                source_lock=str(source_lock))
            if '--resume-pilot' in command:
                manifest['input_artifacts_sha256'][str(prior_output/'summary.json')]=shared.sha(prior_output/'summary.json')
                manifest['retry_classification']='reference comparator cost exceeded first per-driver timeout; reuse completed source cases unchanged'
        if phase_name in P0_FORWARD_PHASES:
            phase_plan=P0_TF_SOLVER_PLAN if phase_name in P0_TF_SOLVER_PHASES else P0_OFFSET_PLAN if phase_name=='p0_offset_localization' else P0_PERTURB_PLAN if phase_name in P0_PERTURB_PHASES else P0_SOLVER_PLAN if phase_name in ('p0_solver_semantics','p0_solver_system_reference') else P0_FITTING_PLAN if phase_name=='p0_fitting_forward' else P0_FORWARD_PLAN
            manifest.update(plan=phase_plan,scientific_role=phase['mode'],
                data_sha256=inputs,seeds=[20260929,20260930],jit_compile=resource=='gpu',
                reference_backend=('Independent CPU conditional LAPACK solver on saved Octave matrices; CUDA hidden'
                    if phase_name in ('p0_solver_semantics','p0_solver_system_reference') else
                    'Independent Octave CPU reference; native random draws and saved policies'),
                comparison_acceptance_field=('per-case sensitivity and additional-failure verdict' if phase_name in P0_PERTURB_PHASES else 'scientific_verdict' if phase_name in ('p0_solver_semantics','p0_solver_system_reference','p0_offset_localization')+P0_TF_SOLVER_PHASES else 'solver_verdicts and fitting_forward_screen_passed' if phase_name=='p0_fitting_forward' else 'forward_screen_passed')+'; PASS alone means diagnostic completion',
                author_failure_is_replication_veto=False,author_method_repair_in_scope=False)
        if phase_name == 'c1_source_failure_replay':
            manifest.update(plan=SOURCE_FAILURE_REPLAY_PLAN,scientific_role='frozen_author_source_failure_reference',
                data_sha256=inputs,source_lock=str(source_lock),seeds=[202610131],jit_compile=False,
                reference_backend='Independent Octave CPU reference; NumPy post-run matrix diagnostics only; CUDA hidden')
        if phase_name == 'c1_source_isolation_check':
            manifest.update(plan=SOURCE_MEMORY_REPAIR_PLAN,scientific_role='source_process_isolation_reference_check',
                data_sha256='Verified C0 source-lock.json',source_lock=str(source_lock),
                seeds=[202610131,202610132,202610133,202610134],jit_compile=False,
                reference_backend='Independent Octave CPU reference; CUDA intentionally hidden')
        if phase_name == 'c1_source_full':
            if '--isolate-cases' in command:
                manifest.update(repair_plan=SOURCE_MEMORY_REPAIR_PLAN,
                    retry_classification='process-memory infrastructure repair; exact source and RNG preserved',
                    case_process_isolation=True,analysis_cpu_reserve_seconds=600)
            manifest.update(plan=C1_FULL_SOURCE_PLAN,scientific_role='full_author_source_matrix',
                data_sha256='Verified C0 source-lock.json',seeds=[202610131,202610132,202610133,202610134],
                reference_backend='Independent Octave CPU source reference; CUDA intentionally hidden',
                source_lock=str(source_lock),full_source_spec_sha256=shared.sha(ROOT/FULL_SOURCE_SPEC),
                resource_allocation=dict(reserved_seconds=timeout,
                    remaining_seconds_at_launch=current['remaining_seconds'],
                    additional_resource_grants_seconds=current['additional_resource_grants_seconds'],
                    opening_budget_sha256=shared.sha(CAMPAIGN/'opening-budget.json')))
        if phase_name == 'k0_source_config':
            manifest.update(plan=K0_SOURCE_CONFIG_PLAN,scientific_role='KDM_source_config_preparation',
                data_sha256='N/A: no data generated or numerical run',seeds='N/A: deterministic metadata',
                statistics_commit='N/A: no Octave or filter execution',
                author_commit='b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18',
                reference_backend='CPU metadata and author YAML generator; CUDA intentionally hidden')
        if phase_name == 'c2_source_results':
            manifest.update(plan=C2_SOURCE_RESULTS_PLAN,scientific_role='source_only_statistical_reporting',
                data_sha256=inputs,seeds=[202610035],source_attempt=str(source_attempt),
                statistics_commit='N/A: post-run statistical reporting; source runtime in input manifest',
                reference_backend='NumPy CPU diagnostic reporting only; no filter execution',jit_compile=False)
        if phase_name.startswith('c3_guided_'):
            manifest['plan']='docs/plans/controlled-smc-guided-repair-20260929.md'
            previous=CAMPAIGN/'attempt15_c3_figure4'
            manifest['input_artifacts_sha256'].update({str(p):shared.sha(p) for p in [previous/'rows.jsonl',previous/'observations.json']+sorted((previous/'fit-diagnostics').glob('*policy.tensor'))})
        if phase_name=='c1_lorenz_normalization':
            manifest.update(plan='docs/plans/controlled-smc-lorenz-normalization-check-20260929.md',data_sha256=shared.sha(data),seeds=[20260929,20260930])
        if phase_name in ('c1_lorenz_consumer','c2_lorenz_consumer','c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            manifest.update(plan='docs/plans/controlled-smc-lorenz-consumer-repair-20260929.md',data_sha256=shared.sha(data),seeds=[20260929,20260930])
        if phase_name in ('c1_lorenz_solver_reference','c2_lorenz_solver_validation','c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            manifest['plan']='docs/plans/controlled-smc-lorenz-solver-validation-20260929.md'
            manifest['fresh_forward_seeds']=[20261001,20261002] if resource=='gpu' else []
        if phase_name in ('c1_lorenz_centered_reference','c2_lorenz_centered_validation'):
            manifest['plan']='docs/plans/controlled-smc-lorenz-centered-recursion-20260929.md'
        if phase_name in ('c1_rbf_consumer_audit','c2_rbf_forward','c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_feasibility_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference','c1_rbf_fit_reference','c1_rbf_nnls_diagnosis','c2_rbf_paired_validation','c2_rbf_policy_accuracy','c2_rbf_nnls_accuracy','c2_rbf_nnls_validation','c2_rbf_fit_validation'):
            manifest.update(plan='docs/plans/controlled-smc-rbf-consumer-audit-20260929.md',
                            data_sha256=shared.sha(ROOT/Path(AUTHOR).parent/'ToyMultimodal/SimulatedData_SigmaSq_1.mat'),
                            seeds=[20261003,20261004])
            if resource=='gpu':manifest['fresh_forward_seeds']=[20261005]
        if phase_name in ('c1_rbf_svd_diagnosis','c1_rbf_solver_diagnosis','c1_rbf_feasibility_diagnosis','c1_rbf_paired_reference','c1_rbf_svd_reference','c1_rbf_scipy_reference','c1_rbf_fit_reference','c1_rbf_nnls_diagnosis','c2_rbf_paired_validation','c2_rbf_policy_accuracy','c2_rbf_nnls_accuracy','c2_rbf_nnls_validation','c2_rbf_fit_validation'):
            manifest.update(plan='docs/plans/controlled-smc-rbf-fitting-validation-20260929.md',seeds=[20261006,20261007])
            if resource=='gpu':manifest['fresh_forward_seeds']=[20261008]
        if phase_name in ('c1_rbf_coefficient_stability','c1_rbf_matched_solver'):
            manifest.update(plan='docs/plans/controlled-smc-rbf-fitting-validation-20260929.md',
                            data_sha256=shared.sha(reference),seeds=[20261006,20261007],
                            statistics_commit='N/A: independent SciPy SVD reference and saved TF tensor decoding')
        if phase_name in ('c1_rbf_paired_reference','c2_rbf_paired_validation','c1_rbf_paired_diagnosis'):
            manifest.update(plan='docs/plans/controlled-smc-rbf-paired-stopping-validation-20260929.md',seeds=[20261009,20261010],stopping_factor=.1)
            if resource=='gpu':manifest['fresh_forward_seeds']=[20261011]
        if phase_name in ('c2_rbf_fixed_policy_sanity','c2_rbf_fixed_policy_precision','c2_rbf_policy_coverage','c2_rbf_coverage_mechanism','c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision','c2_rbf_full_refit','c2_rbf_refit_pilot','c2_rbf_refit_precision'):
            manifest.update(plan='docs/plans/controlled-smc-rbf-fixed-policy-sanity-20260930.md',
                            data_sha256=shared.sha(reference),
                            seeds=[202609300,202609301,202609302,202609309],
                            fitting_seeds=[20261009,20261010],
                            scientific_role='fixed_policy_downstream_diagnostic_only',
                            statistics_commit='N/A: eligible TensorFlow diagnostic; no NumPy runtime')
        if phase_name=='c2_rbf_fixed_policy_precision':manifest.update(seeds=[202609310,202609311,202609312,202609319],repetitions_per_arm=4096)
        if phase_name in ('c2_rbf_policy_coverage','c2_rbf_coverage_mechanism'):manifest.update(seeds='N/A: deterministic saved-policy quadrature',scientific_role='ideal_proposal_coverage_explanatory_only')
        if phase_name in ('c2_rbf_full_retention_pilot','c2_rbf_full_retention_precision'):
            seedbase=202609320 if phase_name.endswith('_pilot') else 202609330
            manifest.update(plan='docs/plans/controlled-smc-rbf-retention-repair-20260930.md',
                scientific_role='full_retention_fixed_policy_repair_diagnostic_only',
                seeds=[seedbase,seedbase+1,seedbase+2,seedbase+9],
                repetitions_per_arm=128 if phase_name.endswith('_pilot') else 4096)
        if phase_name in REFIT_PHASES:
            manifest.update(plan=REFIT_PLAN,scientific_role='full_backward_refit_repair_diagnostic_only',
                training_seeds=[20261009,20261010],training_clouds='frozen attempt57')
            if phase_name=='c2_rbf_full_refit':
                manifest.update(seeds=[20261009,20261010],new_randomness=False)
            else:
                seedbase=202609340 if phase_name.endswith('_pilot') else 202609350
                manifest.update(seeds=[seedbase,seedbase+1,seedbase+2,seedbase+9],
                    repetitions_per_arm=128 if phase_name.endswith('_pilot') else 4096)
        if phase_name in LORENZ_REFINEMENT_PHASES:
            manifest.update(plan=LORENZ_REFINEMENT_PLAN,data_sha256=shared.sha(data),
                seeds=[202609360],scientific_role='fixed_input_lorenz_refinement_and_apf_validation',
                result_file=str(output/'summary.json'),
                statistics_commit='N/A: independent SciPy reference or eligible TensorFlow validation',
                source_forcing_values=[2.5,4.8801,8.5],maximum_refinements=4,
                source_apf_counts=[1382,2867,4505,6144])
        if phase_name in ('c1_lorenz_precision_diagnosis','c2_lorenz_precision_safety','c1_lorenz_trajectory_diagnosis','c2_lorenz_covariance_safety','c1_lorenz_covariance_validation','c2_lorenz_rank_guard','c1_lorenz_covariance_localization'):
            manifest.update(plan=LORENZ_PRECISION_PLAN,seeds=[202609380],source_forcing_values=[2.5],scientific_role='independent_rejected_precision_diagnosis')
        if phase_name=='c1_lorenz_trajectory_diagnosis':manifest['scientific_role']='independent_nonfinite_trajectory_diagnosis'
        if phase_name=='c1_lorenz_covariance_validation':manifest.update(scientific_role='independent_repaired_policy_correctness',precision_policy='nearest_covariance')
        if phase_name=='c2_lorenz_rank_guard':manifest.update(scientific_role='validity_guard_no_fire_and_rejection',precision_policy='nearest_covariance_rejected_candidate')
        if phase_name=='c1_lorenz_covariance_localization':manifest.update(scientific_role='rejected_policy_numerical_localization',precision_policy='nearest_covariance_rejected_candidate')
        if phase_name in ('c2_lorenz_precision_safety','c2_lorenz_covariance_safety'):
            manifest.update(scientific_role='optional_precision_policy_nonharm_only',precision_policies=['nearest_covariance'] if phase_name=='c2_lorenz_covariance_safety' else ['nearest_precision','backtrack_precision'],positive_relative_floor='sqrt(float64_epsilon)')
        if phase_name=='c2_lorenz_figure6_pilot':
            manifest.update(plan=LORENZ_PILOT_PLAN,seeds=[202609370],
                scientific_role='fresh_seed_validity_and_execution_cost_only',
                source_forcing_grid_indices=[0,15,39],repetitions_per_point=4,
                result_file='docs/plans/controlled-smc-lorenz-refinement-result-20260930.md')
        if phase_name.startswith('c2_lorenz_batch_'):
            manifest.update(plan=LORENZ_BATCH_PLAN,seeds=[202609380 if phase['mode']=='validate' else 202609390],
                replicas=4,source_forcing_values=[2.5,2.5+6*15/39,8.5],
                source_forcing_grid_indices=[0,15,39],
                scientific_role='independent_repetition_batching_validation_and_cost')
            if phase['mode']=='validate':manifest['seeds'].append(202609360)
        if phase_name=='c1_lorenz_author_native':
            manifest.update(plan='docs/plans/controlled-smc-lorenz-author-native-audit-20260930.md',
                data_sha256=shared.sha(ROOT/Path(AUTHOR).parent/'Lorenz/SimulatedData.mat'),
                seeds=[20260929,20260930],source_forcing_values=[2.5,4.8801,8.5],
                maximum_refinements=4,scientific_role='unchanged_author_source_reference_only',
                algorithm_source_modified=False,reference_backend='Octave CPU; MATLAB execution not checked')
        if phase_name in LORENZ_SOURCE_PORT_PHASES:
            manifest.update(plan=LORENZ_SOURCE_PORT_PLAN,
                data_sha256=shared.sha(ROOT/Path(AUTHOR).parent/'Lorenz/SimulatedData.mat'),
                seeds=[20260929,20260930],source_forcing_values=[2.5,4.8801,8.5],
                maximum_refinements=4,scientific_role='matched_author_source_port_localization_only',
                source_instrumentation='capture only; exact source result equality required',
                reference_backend='Octave CPU' if resource=='cpu' else 'TensorFlow FP64 GPU/XLA; TF32 off')
        if phase_name in LORENZ_MATRIX_PHASES:
            manifest.update(plan=LORENZ_MATRIX_PLAN,
                result_file='docs/plans/controlled-smc-lorenz-matrix-crossover-result-20261001.md',
                data_sha256=shared.sha(ROOT/Path(AUTHOR).parent/'Lorenz/SimulatedData.mat'),
                seeds=[20260929,20260930],source_forcing_values=[2.5,4.8801,8.5],
                diagnostic_times=[100,99],regression_count=32,
                scientific_role='crossed_matrix_arithmetic_localization_only',
                reference_backend='Octave CPU independent reference' if resource=='cpu' else 'TensorFlow FP64 GPU/XLA; TF32 off')
        if phase_name=='c1_lorenz_failed_source_replay':
            manifest.update(diagnostic_times='all times until first invalid precision',regression_count='16 exact healthy refits plus 3 frozen TF refits',
                scientific_role='original_source_fit_on_failed_cloud',source_forcing_values=[2.5,4.8801,8.5],
                seeds='No new randomness; source seeds20260929/30 and failed TF attempt76 seeds as input manifests')
        if phase_name in LORENZ_BOUNDED_PHASES:
            manifest.update(plan=LORENZ_BOUNDED_PLAN,seeds='No new randomness: frozen69 and76 designs',
                scientific_role='optional_bounded_covariance_repair_nonharm_and_accuracy',
                precision_policy='nearest_covariance',
                activation_margin=(0.001 if 'margin_' in phase_name else 64*2.220446049250313e-16 if 'trigger_' in phase_name else None),
                relative_floor_hypotheses=([0.01] if any(tag in phase_name for tag in ('trigger_','margin_')) else [0.01,0.1] if 'condition_' in phase_name else [0.001] if any(tag in phase_name for tag in ('integral_','crossover_','reference_arithmetic')) else [2.220446049250313e-16**.5,1e-6,1e-4,1e-3]),
                historical_floor_baseline=2.220446049250313e-16**.5,
                reference_backend='independent CPU SVD' if resource=='cpu' else 'TensorFlow FP64 GPU/XLA')
        if phase_name in LORENZ_ROOT_CAUSE_PHASES:
            manifest.update(plan=LORENZ_ROOT_CAUSE_PLAN,seeds='No new randomness: frozen attempt76 design',
                scientific_role='saved_policy_geometry_and_call_chain_audit_diagnostic_only',
                data_sha256=shared.sha(design/'rejected_batch/normal.tensor'),
                reference_backend='CPU NumPy/SciPy SVD and mpmath80; TF CPU graph regression diagnostic')
        if phase_name in LORENZ_REPAIR_PHASES:
            manifest.update(plan=LORENZ_REPAIR_PLAN,seeds='Frozen69/76/108/111/121 plus deterministic synthetic regression injections; exact stateless seeds in snapshotted test source; no stochastic selection',
                scientific_role='first_failure_and_recursive_numerical_repair_diagnostic',
                data_sha256=inputs,reference_backend='Independent CPU SciPy SVD' if resource=='cpu' else 'TensorFlow FP64 GPU/XLA; TF32 off')
        path=output/'manifest.json'; shared.atomic_json(path,manifest)
        start=time.monotonic()
        with (output/'command.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest.update(pid=child.pid,process_identity=shared.proc_identity(child.pid));shared.atomic_json(path,manifest)
            try: rc=child.wait(timeout=timeout)
            except BaseException as exc:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=124 if isinstance(exc,subprocess.TimeoutExpired) else 130
        seconds=time.monotonic()-start
        try:
            decision=json.loads((output/'summary.json').read_text()) if (output/'summary.json').exists() else {}
        except (ValueError,OSError) as exc:
            decision={'status':'FAIL','artifact_error':str(exc)}
        manifest['artifact_error']=decision.get('artifact_error')
        if resource=='gpu':
            manifest['gpu_memory_policy']=decision.get('gpu_memory_policy','UNAVAILABLE: inspect command log if initialization failed')
            manifest['tensorflow']=decision.get('tensorflow')
            manifest['jit_compile']=decision.get('jit_compile')
            manifest['device']=decision.get('device')
        manifest.update(status='COMPLETE' if rc==0 and decision.get('status')=='PASS' else 'TIMED_OUT' if rc==124 else 'FAILED',
                        ended_utc=shared.stamp(),returncode=rc,wall_seconds=seconds,charged_seconds=seconds,
                        decision=compact_decision(decision),
                        output_sha256={str(p.relative_to(output)):shared.sha(p) for p in output.rglob('*') if p.is_file() and p!=path})
        shared.atomic_json(path,manifest)
        shared.atomic_json(CAMPAIGN/'status.json',status())
        refresh_checkpoint()
        return {k:manifest[k] for k in ('phase','status','returncode','charged_seconds','decision','output_root')}

def refresh_checkpoint(current=None):
    """Write the replication-first checkpoint without making repair work active."""
    current = status() if current is None else current
    records = campaign_attempts()
    latest = {}
    for _, manifest in records:
        latest[manifest['phase']] = manifest

    def phase_line(name):
        manifest = latest.get(name)
        if manifest is None:
            return name + ': not run'
        if name == 'c2_lorenz_source_port':
            return name + ': completed diagnostic; port contract incomplete'
        decision = manifest.get('decision', {})
        return name + ': ' + manifest.get('status', 'UNKNOWN') + ' (' + decision.get('status', 'no decision') + ')'

    repairs = [m for _, m in records if m['phase'] in LORENZ_REPAIR_PHASES]
    repair_charge = {resource: sum(shared.charge(m) for m in repairs if m['resource'] == resource)
                     for resource in ('cpu', 'gpu')}
    spec = json.loads((ROOT / FULL_SOURCE_SPEC).read_text())
    source_allocation_approved = spec.get('cpu_allocation_status') == 'APPROVED'
    next_action = ('Construct and review the C0 source-matrix specification before launching '
                   'the full author-source baseline. E0 numerical repairs are deferred.')
    if current.get('active_program_stage')=='C1_SOURCE_PARTIAL_FAILURE_OTHER_DRIVERS_RUNNING':
        next_action = 'Accept the intrinsic Lorenz failure as a reference outcome. Finish the independent neuroscience baseline and queued failure capture; compare the TensorFlow consumer on valid and failing reference cases. Full C2 paper statistics are separate; no author-method repair.'
    elif current.get('active_program_stage')=='C1_SOURCE_FAILURE_DIAGNOSED_REPLICATION_INCOMPLETE':
        next_action = 'Use the frozen failure as a matched TensorFlow comparison case. Resolve our translation discrepancies; intrinsic author failures are acceptable and author-method repair remains deferred.'
    elif current.get('running'):
        next_action = 'Monitor ' + ', '.join(current['running']) + '; no overlapping numerical launch.'
    elif current.get('active_program_stage') == 'C1_FULL_SOURCE_REPAIR_REQUIRED':
        next_action = ('Inspect the full-source failure and preserve completed evidence; repair only within the approved '
                       'method and remaining CPU allocation before revising a retry specification. E0 stays deferred.')
    elif 'c2_source_results' in current['passed']:
        next_action = ('Read the C2 result and its separate per-grid inference verdicts; complete source-valid P0/P1 specifications and KDM source/training gaps. E0 stays deferred.')
    elif current.get('active_program_stage') == 'C2_SOURCE_ANALYSIS_REPAIR_REQUIRED':
        next_action = 'Inspect the C2 failure and preserve its evidence; repair within the existing contract and budget.'
    elif 'c1_source_full' in current['passed']:
        next_action = ('Assemble C2 source results under the frozen comparison criteria; then specify source-valid P0/P1 port checks. '
                       'Table 1, multimodal and KDM source/training gaps remain separate. E0 stays deferred.')
    elif latest.get('c1_source_cost_pilot', {}).get('status') == 'COMPLETE':
        next_action = (('Run the reviewed full source command under the approved 72-hour CPU allocation. '
                        if source_allocation_approved else
                        'Full source command and diffs prepared; obtain the costed 72-hour CPU allocation before execution. ') +
                       'K0 config generation is complete; recover remaining KDM settings and training dependencies. E0 stays deferred.')
    elif latest.get('c0_source_lock', {}).get('status') == 'COMPLETE':
        next_action = ('Run the bounded C1 source cost pilot, then size the full source matrices '
                       'against the remaining CPU allocation. E0 stays deferred.')
    elif latest.get('c1_lorenz_author_native', {}).get('status') == 'COMPLETE':
        next_action = ('Complete C0 provenance/runtime/driver lock, then specify the full '
                       'author-source experiment. The six-case source pilot does not close C1.')

    forward = latest.get('p0_forward_compare')
    if forward and forward.get('status') != 'RUNNING' and not current.get('running'):
        next_action = ('Review the completed fixed-policy diagnostic and its failed-policy input gap; '
                       'finish source-to-port fitting parity, freeze P1 statistical margins, and complete '
                       'the separately budgeted KDM protocol. Do not rerun the author failure or start E0.')
    fitting = latest.get('p0_fitting_forward')
    if fitting:
        if fitting.get('status') == 'RUNNING':
            next_action = 'Monitor the bounded fitting/forward comparison; do not overlap numerical work.'
        elif fitting.get('decision', {}).get('status') == 'PASS':
            next_action = 'Inspect the per-solver fitting/forward verdicts and localize any additional invalidity using saved policies. P1 margins and KDM source/training work remain open; no E0 protections.'
        else:
            next_action = 'Inspect the fitting comparison infrastructure failure before a bounded retry under its unchanged plan.'
    solver_reference=latest.get('p0_solver_system_reference')
    if (solver_reference and solver_reference.get('status')=='COMPLETE'
            and reported_decision(solver_reference).get('scientific_verdict')=='CONDITIONAL_SOLVER_REFERENCE_AGREES'
            and not current.get('running')):
        next_action = ('Use the verified source solver reference to specify TensorFlow condition/branch/rank behavior; '
                       'check it on all128 saved matrix/RHS pairs before recursive fitting. Exact branch choice alone '
                       'did not reproduce source coefficients with a different CPU LAPACK. Do not add truncation/floors '
                       'or equate the strict diagnostic screen with P1 no-worse qualification. Freeze P1 margins; KDM training remains separately specified.')
    if 'p0_perturb_tf' in current['passed']:
        next_action = 'Perturbation complete: 174 source/348 TF trials valid; both solvers retain a roughly 1.1% F=4.8801/seed20260930 offset, about 49 times the source median sensitivity. Localize it using saved policies/clouds and the verified source solver reference, then freeze downstream margins on fresh matched seeds. No tolerance relaxation; see controlled-smc-perturbation-result-20261006.md.'
    elif latest.get('p0_perturb_tf'):
        next_action = 'Inspect the preserved TF perturbation failure or timeout; continue only within the unchanged plan and remaining phase budget. No completed-sensitivity claim yet.'
    elif 'p0_perturb_source' in current['passed']:
        next_action = 'Execute p0_perturb_tf using the completed author perturbation tapes under the unchanged 20261006 plan.'
    elif latest.get('p0_perturb_source'):
        next_action = 'Inspect the preserved source perturbation failure before retrying under the unchanged plan and remaining phase budget.'
    if 'p0_offset_localization' in current['passed']:
        offset_decision=reported_decision(latest['p0_offset_localization'])
        if offset_decision.get('scientific_verdict')=='DIRECT_SOLVER_DIFFERENCE_CONFIRMED':
            next_action = ('Localization completed; see controlled-smc-offset-localization-result-20261006.md. '
                           'Next test conditional TF solver compatibility on 128 earlier systems and 101 saved source regressions, then autonomous fitting and independently calibrated P1. '
                           'No author numerical repair, rank-floor change or tolerance relaxation.')
        else:
            next_action = 'Inspect the unresolved direct QR/reference mismatch in the localization diagnostic before any solver or qualification claim.'
    compatibility=latest.get('p0_tf_solver_cpu_reference') or latest.get('p0_tf_solver_compatibility')
    if compatibility:
        verdict=reported_decision(compatibility).get('scientific_verdict')
        if verdict=='FORCED_BRANCH_COMPATIBLE':
            next_action = 'Conditional-operation screen passed; implement and check branch selection before autonomous fitting. This diagnostic does not qualify the port.'
        elif verdict=='FORCED_BRANCH_ARITHMETIC_MISMATCH':
            next_action = ('Inspect source DGELSD arithmetic on saved times20,25,28 before another compatibility repair. GPU and CPU both fail the forced-branch screen; small backward errors do not establish policy agreement. Separately justify P0/P1 criteria using fresh source calibration; no autonomous promotion or tolerance relaxation.'
                if latest.get('p0_tf_solver_cpu_reference') else
                'Execute the planned CPU TensorFlow graph reference to distinguish GPU/XLA behavior from a solver-library difference.')
        else:
            next_action = 'Inspect the solver diagnostic log and controls before any bounded retry; no compatibility claim.'
    reference = current.get('active_reference')
    if reference:
        next_action = (reference['next_action']
                       if reference['summary_verification'] == 'VERIFIED_PRESERVED_SUMMARY'
                       and reference['decision'].get('status') == 'PASS'
                       else 'Inspect the unverified reference diagnostic before scientific interpretation.')
    if current.get('running'):
        next_action = ('Monitor ' + ', '.join(current['running']) + '; no duplicate training launch. ' + (reference['next_action'] if reference else 'No overlapping numerical launch without a bounded phase plan.'))
    if current.get('active_program_stage')=='KDM_SOURCE_VARIABILITY_CALIBRATION_RUNNING':
        next_action = ('Monitor the bounded source-only calibration; each full source unit is '
                       'automatically audited and evaluated after training. No duplicate launch, '
                       'seed selection, numerical tolerance relaxation or replication promotion.')
    if current.get('active_program_stage')=='CSMC_SOURCE_GRID_CALIBRATION_RUNNING':
        next_action = ('Monitor the source-only 40-point forcing grid plus the generating-parameter calibration. Isolated workers retain all failures and bound memory; freeze the final sample size and inference before untouched paired qualification.')
    if current.get('active_program_stage')=='CSMC_DOWNSTREAM_CALIBRATION_RUNNING':
        next_action = ('Monitor the complete adaptive source/TF paired calibration. Check pristine-source and initial-APF controls, failures, moments and cost before freezing untouched qualification. Solver repair is conditional on demonstrated downstream need.')
    if current.get('active_program_stage')=='CSMC_SOURCE_GRID_CALIBRATION_COMPLETE':
        next_action = 'Review completed 41-cell calibration and execute the frozen 320-pair-per-cell Q2 design after executable controls. Margins remain unchanged.'
    if current.get('active_program_stage')=='CSMC_UNTOUCHED_QUALIFICATION_RUNNING':
        next_action = 'Monitor source/GPU workers, control outcomes, resource charges and failure triggers; calculate frozen CIs only at all 41 completed cells.'
    if current.get('active_program_stage')=='CSMC_UNTOUCHED_QUALIFICATION_CANDIDATE_REJECTED':
        next_action = 'Q3: inspect candidate-rejection.json and saved failure tensors, localize the additional TF failure against native source. Used qualification seeds are development; preserve criteria and budget.'
    if current.get('active_program_stage')=='CSMC_UNTOUCHED_QUALIFICATION_COMPLETE':
        next_action = ('Q2 is complete under the historical wrong-planning margins. For cSMC, '
                       'check source-only tails, precision and cost, then decide whether an '
                       'independently justified no-worse bound is testable before fresh '
                       'qualification; keep Q2 as development evidence. KDM remains separate.')
    content = [
        '# Controlled-SMC / KDM active checkpoint',
        'Updated UTC ' + shared.stamp() + '.',
        'Checkout: /home/chakwong/BayesFilter; branch surrogate-hmc; base 89065bc6354801cb368d5e163ba49fd9c4372d10. Preserve unrelated dirty work.',
        'Question: reproduce author behavior with no worse numerical results and no additional failures under matched conditions. Intrinsic author failures are accepted baseline outcomes. E0 is deferred; original iAPF is deferred.',
        'Master: controlled-smc-kdm-replication-master-program-20260928.md; contract: ' + REPLICATION_FIRST_PLAN + '; dispatcher: scripts/run_controlled_smc_kdm.py.',
        'Active cSMC acceptance: controlled-smc-author-relative-acceptance-20261011.md. Fixed 1% likelihood and 5% variance margins were wrong planning; Q2 verdict historical, not retroactively passed. Source repeatability alone is not an acceptance bound. KDM criteria unchanged.',
        'Current stage: ' + current['active_program_stage'] + '.',
        'Current result: ' + str(current.get('result')) + '.',
        'Source outcomes: ' + json.dumps(current.get('source_driver_failures', [])) + '. Completed neuroscience matrices and source-failure details: controlled-smc-p0-forward-result-20261005.md.',
        'Forward decision: ' + json.dumps(reported_decision(forward) if forward else {}) + '.',
        'Fitting decision: ' + json.dumps(reported_decision(fitting) if fitting else {}) + '. Evidence: controlled-smc-p0-fitting-forward-result-20261005.md.',
        'Conditional reference: ' + json.dumps(reported_decision(solver_reference) if solver_reference else {}) + '. Evidence: controlled-smc-p0-solver-semantics-result-20261005.md.',
        'Localization decision: ' + json.dumps(reported_decision(latest.get('p0_offset_localization')) if latest.get('p0_offset_localization') else {}) + '.',
        'TF solver compatibility: ' + json.dumps(reported_decision(compatibility) if compatibility else {}) + '.',
        'Independent reference diagnostics: ' + json.dumps([
            {key: item[key] for key in ('id', 'charged_seconds', 'decision', 'summary_verification')}
            for item in current.get('reference_diagnostics', [])[-2:]]) + '. Full history: reference-diagnostics.json.',
        'Remaining gaps: cSMC author-only repeatability calibration, a fresh downstream decision if feasible, and unresolved selected source experiments; source-intrinsic failures are baseline outcomes. KDM float32 compatibility, accumulated training drift and independent trained-unit qualification remain separate. Exact ill-conditioned coefficients are not an acceptance requirement. Both replications INCOMPLETE.',
        'Next action: ' + next_action,
        'Controlled-SMC downstream phase: ' + json.dumps([{k:item[k] for k in ('path','status','used_seconds')} for item in current.get('csmc_qualification', [])]) + '.',
        'Complete-training phase: ' + json.dumps([{k:item[k] for k in ('path','status','mode','epoch_mode','child_pid','progress')}
                                                  for item in current.get('supplemental_training', [])[-2:]]) + '.',
        'Remaining reserved phase seconds: ' + json.dumps(current.get('supplemental_training_budget_seconds', {})) +
            '; included in the active allocation, not an additional grant.',
        'Source-variability phase: ' + json.dumps([
            {key: item[key] for key in ('path', 'status', 'charged_seconds', 'result')}
            for item in current.get('source_variability', [])]) + '. Child manifests remain in the phase directory.',
        'Attempts ' + str(current['attempts']) + '; running ' + json.dumps(current['running']) + '.',
        'Active allocation remaining seconds ' + json.dumps(current['tranche_remaining_seconds']) + '. Larger inherited balances are not active grants. Routine CPU checks are recorded separately.',
        'Detailed run manifests: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/attempt*/manifest.json. No default, superiority, HMC or robustness promotion claim.',
    ]
    (ROOT / 'docs/plans/controlled-smc-kdm-checkpoint.md').write_text('\n'.join(content) + '\n')

def wait_and_execute(phase_name):
    """Serialize a bounded continuation; the phase lock also rejects direct duplicates."""
    if phase_name not in ACTIVE_PHASES:
        raise RuntimeError('Historical phase is not active: '+phase_name)
    queue_path=CAMPAIGN/f'queue-{phase_name}.json'
    with (CAMPAIGN/f'queue-{phase_name}.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This phase already has a live queue supervisor') from exc
        if phase_name=='c2_source_results':
            continuation_readiness()
        elif phase_name=='c1_source_failure_replay':
            source_replay_readiness()
        elif phase_name in P0_FORWARD_PHASES:
            p0_forward_readiness(phase_name)
        current=status()
        ready,deadline=queue_wait_state(phase_name,current)
        queued=dict(phase=phase_name,status='WAITING',pid=os.getpid(),
                    process_identity=shared.proc_identity(os.getpid()),
                    started_utc=shared.stamp(),deadline_epoch=deadline)
        shared.atomic_json(queue_path,queued)
        try:
            while not ready:
                if time.time()>deadline:
                    raise RuntimeError('Queue exceeded predecessor execution deadline')
                refresh_checkpoint(current)
                time.sleep(15)
                current=status()
                ready,_=queue_wait_state(phase_name,current)
            if phase_name in current['passed']:
                result=dict(phase=phase_name,status='COMPLETE',decision={'status':'ALREADY_COMPLETE'})
            else:
                queued.update(status='EXECUTING')
                shared.atomic_json(queue_path,queued)
                result=execute(phase_name)
            queued.update(status=result['status'],result=result,ended_utc=shared.stamp())
            shared.atomic_json(queue_path,queued)
            refresh_checkpoint()
            return result
        except BaseException as exc:
            queued.update(status='FAILED',error=str(exc),ended_utc=shared.stamp())
            shared.atomic_json(queue_path,queued)
            refresh_checkpoint()
            raise

def bounded_report(phase_name=None):
    """Selected evidence fields for routine inspection under the dispatcher."""
    current=status()
    result={k:current[k] for k in ('attempts','used_seconds','tranche_remaining_seconds','active_program_stage','running')}
    if phase_name is None:return result
    matching=[m for _,m in campaign_attempts() if m['phase']==phase_name]
    if not matching:return dict(**result,phase=phase_name,status='NOT_RUN')
    latest=matching[-1];output=Path(latest['output_root'])
    result.update({k:latest.get(k) for k in ('phase','status','output_root','charged_seconds','plan')})
    progress=output/'progress.json'
    if progress.exists():
        progress_value=json.loads(progress.read_text())
        result['progress']=({k:progress_value[k] for k in
            ('last_completed_case','last_completed_refinement','comparisons') if k in progress_value}
            if phase_name in LORENZ_SOURCE_PORT_PHASES else progress_value)
    path=output/'summary.json'
    if not path.exists():return result
    decision=json.loads(path.read_text())
    result['summary']={k:decision[k] for k in ('status','scope','failure','replication','promotion',
        'comparison_verdict','forward_screen_passed','source_results_exactly_unchanged','source_rng_exactly_unchanged','wall_seconds','fit_first_call_seconds','full_backward_refit','retained_components','forward_component_slots','forward_positive_component_slots','peak_active','max_iterations','max_dual_error','max_complementarity','minimum_qr_margin','heuristic_dominance_verdict','allocator','traces') if k in decision}
    if phase_name in LORENZ_REPAIR_PHASES:
        result['summary'].update({k:decision[k] for k in
            ('default_bookkeeping_repair_passed','full_recursive_admission','full_recursive_admission_reason',
             'candidate_admitted','cross_backend_arithmetic_agreement','guard_full_scope_parity','fail_fast_default') if k in decision})
        checks=decision.get('checks',[])
        if checks:
            result['summary']['check_count']=len(checks)
            result['summary']['failed_checks']=[c.get('name') for c in checks if not c.get('passed')]
        result['summary']['diagnostic_completion_is_not_admission']=True
    if phase_name in LORENZ_SOURCE_PORT_PHASES:
        result['summary'].update({k:decision[k] for k in
            ('scientific_verdict','source_results_exactly_unchanged','captured_regressions','trace_counts','gpu_allocator') if k in decision})
        if 'comparisons' in decision:
            result['comparison_count']=len(decision['comparisons'])
            result['mismatching_comparisons']=sum(not row['passed'] for row in decision['comparisons'])
            result['invalid_route_count']=len(decision.get('invalid_routes',[]))
    if phase_name in LORENZ_MATRIX_PHASES:
        result['summary'].update({k:decision[k] for k in
            ('scientific_verdict','mode','trace_counts','gpu_allocator','finite_status','raw_valid','qr_valid') if k in decision})
        result['comparison_count']=len(decision.get('comparisons',[]))
    if phase_name in LORENZ_BOUNDED_PHASES:
        result['summary'].update({k:decision[k] for k in
            ('eligible_floors','validated_floors','frozen_floor','verdict','activation_margin','validation_checks','case_count','first_invalid','error') if k in decision})
        result['candidates']=[{k:row[k] for k in
            ('floor','activation_margin','eligible','verdict','status','failures','healthy_valid','healthy_protections','rejected_design_valid','rejected_design_finite','iterations','error')
            if k in row} for row in decision.get('candidates',[])]
    if 'full_validation' in decision:
        nested=decision['full_validation']
        result['full_validation']={k:nested[k] for k in ('status','verdict','validated_floors','frozen_floor','candidates') if k in nested}
    checks=decision.get('checks',[])
    result['check_count']=len(checks)
    result['failed_checks']=[row for row in checks if not row.get('passed',False)]
    if 'normalizer_comparisons' in decision:
        result['normalizers']=[{k:row[k] for k in ('arm','mean_normalizer_ratio',
            'simultaneous99_bootstrap_interval','classification','max_ratio')}
            for row in decision['normalizer_comparisons']]
    if 'conditional_heuristics' in decision:
        result['conditional_filtering_mse']=[{k:row[k] for k in
            ('arm','situation','filtering_mean_mse','observed_underperformance')}
            for row in decision['conditional_heuristics']]
    return result


def full_source_timeout(current):
    phase=PHASES['c1_source_full']
    records=[m for _,m in campaign_attempts() if m['phase']=='c1_source_full']
    available=min(current['remaining_seconds']['cpu'],current['tranche_remaining_seconds']['cpu'])
    source_remaining=phase['total_limit']-sum(shared.charge(m) for m in records)
    if not records:
        if min(available,source_remaining)<phase['limit']:
            raise RuntimeError('Initial full source launch requires its approved 72-hour CPU reservation')
        return phase['limit']
    if records[-1]['status'] not in ('FAILED','TIMEOUT','INTERRUPTED'):
        raise RuntimeError('Only a finalized interrupted source attempt is eligible for this repair')
    if 'c1_source_isolation_check' not in current['passed']:
        raise RuntimeError('Source process isolation must pass exact numerical and RNG verification')
    timeout=min(source_remaining,available-PHASES['c2_source_results']['limit'])
    spec=json.loads((ROOT/FULL_SOURCE_SPEC).read_text())
    if timeout<sum(spec['driver_limits_seconds'].values())+60:
        raise RuntimeError('Remaining approved CPU budget cannot cover unchanged source-driver limits and C2 reserve')
    return timeout


def full_source_launch_readiness():
    """Check the approved source reservation without starting any process."""
    current = status()
    phase = PHASES["c1_source_full"]
    opening = json.loads((CAMPAIGN / "opening-budget.json").read_text())
    spec = json.loads((ROOT / FULL_SOURCE_SPEC).read_text())
    if current["running"]:
        raise RuntimeError("A campaign attempt is already running")
    if spec.get("cpu_allocation_status") != "APPROVED":
        raise RuntimeError("The full source CPU allocation has not been recorded")
    if current["attempts"] >= opening["attempt_limit"]:
        raise RuntimeError("Campaign attempt allowance exhausted")
    if "c1_source_full" in current["passed"]:
        raise RuntimeError("Full source phase already passed; assemble C2 instead")
    records = [m for _, m in campaign_attempts() if m["phase"] == "c1_source_full"]
    if len(records) >= phase["max_attempts"]:
        raise RuntimeError("Review the failed source attempt before a bounded retry")
    if not set(phase["dependencies"]).issubset(current["passed"]):
        raise RuntimeError("Source lock and cost pilot must pass before the full run")
    full_source_timeout(current)
    return current


def progress():
    """Bounded live reporting; no framework import, device probe or numerical work."""
    current = status()
    records = campaign_attempts()
    selected = [(p, m) for p, m in records if m["phase"] in ACTIVE_PHASES]
    latest = selected[-1] if selected else None
    running_attempts = [(p, m) for p, m in selected if m["status"] == "RUNNING"]
    if running_attempts:
        latest = running_attempts[-1]
    result = dict(replication_target=current["replication_target"],author_failure_is_replication_veto=False,paper_result_success_required_for_port=False,source_driver_failures=current.get("source_driver_failures",[]),stage=current["active_program_stage"], attempts=current["attempts"],
                  running=current["running"], remaining_seconds={r:min(current["remaining_seconds"][r],current["tranche_remaining_seconds"][r]) for r in ("cpu","gpu")},
                  inherited_remaining_seconds=current["remaining_seconds"],
                  csmc_replication=current["csmc_replication"],
                  kdm_replication=current["kdm_replication"], continuation=queue_status())
    result["reference_diagnostics"] = current.get('reference_diagnostics', [])
    result["source_failure_replay"]=queue_status("c1_source_failure_replay")
    result["forward_comparison"]={name:queue_status(name) for name in P0_FORWARD_PHASES}
    completed_forward=[m for _,m in records if m["phase"]=="p0_forward_compare" and m["status"]!="RUNNING"]
    if completed_forward:
        decision=reported_decision(completed_forward[-1])
        result["forward_comparison_result"]={key:decision[key] for key in
            ("status","comparison_verdict","forward_screen_passed","replication","summary_verification") if key in decision}
    completed_fitting=[m for _,m in records if m["phase"]=="p0_fitting_forward" and m["status"]!="RUNNING"]
    if completed_fitting:
        result["fitting_comparison_result"]=reported_decision(completed_fitting[-1])
    completed_solver=[m for _,m in records if m["phase"]=="p0_solver_system_reference" and m["status"]!="RUNNING"]
    if completed_solver:
        result["solver_reference_result"]=reported_decision(completed_solver[-1])
    completed_offset=[m for _,m in records if m['phase']=='p0_offset_localization' and m['status']!='RUNNING']
    if completed_offset:
        result['offset_localization_result']=reported_decision(completed_offset[-1])
    completed_compatibility=[m for _,m in records if m['phase'] in P0_TF_SOLVER_PHASES and m['status']!='RUNNING']
    if completed_compatibility:
        result['tf_solver_compatibility_result']=reported_decision(completed_compatibility[-1])
    if latest is None:
        return result
    path, manifest = latest
    output = Path(manifest["output_root"])
    result["attempt"] = dict(phase=manifest["phase"], status=manifest["status"],
        manifest=str(path), charged_seconds=shared.charge(manifest),
        supervisor_alive=shared.alive(manifest.get("supervisor_pid"), manifest.get("supervisor_identity")),
        worker_alive=shared.alive(manifest.get("pid"), manifest.get("process_identity")))
    if manifest["phase"] == "c1_source_full":
        drivers = []
        for key, lane, expected in (
                ("lorenz_csmc", "Lorenz", 4000), ("lorenz_apf", "Lorenz", 4000),
                ("neuroscience_csmc", "NeuroScience", 2000),
                ("neuroscience_bpf", "NeuroScience", 2000)):
            source = output / "reference-work" / lane / (key + "-cases.csv")
            completed, nonfinite, partial, last = 0, 0, 0, None
            if source.exists():
                # A writer may be appending the final record. Do not count partial lines.
                for line in source.read_text().splitlines(keepends=True):
                    if not line.endswith("\n"):
                        partial += 1
                        continue
                    row = next(csv.reader([line]))
                    if len(row) != 11:
                        partial += 1
                        continue
                    completed += 1
                    nonfinite += int(row[4]) != 1
                    last = dict(grid_index=int(row[0]), repetition=int(row[1]),
                                logZ=float(row[3]) if int(row[4]) == 1 else None)
            drivers.append(dict(driver=key, completed_cases=completed, expected_cases=expected,
                                nonfinite_cases=nonfinite, partial_or_invalid_rows=partial,
                                last_completed_case=last))
        result["drivers"] = drivers
        result["completed_cases"] = sum(d["completed_cases"] for d in drivers)
        result["expected_cases"] = 12000
        result["source_validity"] = "Finite-real logZ screen only; full domain validity not certified"
    elif manifest["phase"] in P0_FORWARD_PHASES:
        progress_path=output/"progress.json"
        if manifest["status"]=="RUNNING" and progress_path.exists():
            try:
                payload=json.loads(progress_path.read_text())
                result["phase_progress"]={key:payload[key] for key in ("fits","stages","status") if key in payload}
            except (OSError,ValueError):
                result["phase_progress"]={"status":"PROGRESS_WRITE_IN_FLIGHT"}
        result["phase_decision"]=reported_decision(manifest)
    return result


def start_detached(phase_name):
    """Start or reuse a detached source/analysis supervisor without duplicate jobs."""
    if phase_name not in ('c1_source_full','c2_source_results','c1_source_failure_replay')+P0_FORWARD_PHASES:
        raise RuntimeError('Detached start is scoped to the reviewed source and analysis phases')
    with (CAMPAIGN/f'launch-{phase_name}.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another start command is active; inspect progress') from exc
        return _start_detached_locked(phase_name)


def _start_detached_locked(phase_name):
    queued=phase_name in ('c2_source_results','c1_source_failure_replay')+P0_FORWARD_PHASES
    if queued:
        existing=queue_status(phase_name)
        if existing['status'] in ('WAITING','EXECUTING') and existing.get('supervisor_alive'):
            return dict(launch=existing,progress=progress())
        current=(p0_forward_readiness(phase_name) if phase_name in P0_FORWARD_PHASES else
                 continuation_readiness() if phase_name=='c2_source_results' else source_replay_readiness())
        if phase_name in current['passed']:
            return dict(launch=dict(status='ALREADY_COMPLETE',phase=phase_name),progress=progress())
    else:
        full_source_launch_readiness()
    launch_name='source-launch-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns())
    launch_path=CAMPAIGN/(launch_name+'.json')
    log_path=CAMPAIGN/(launch_name+'.log')
    command=[PYTHON,'-B',str(Path(__file__).resolve()),'wait-run' if queued else 'run','--phase',phase_name]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1')
    with log_path.open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(status='STARTING',phase=phase_name,command=command,log=str(log_path),
                started_utc=shared.stamp(),supervisor_pid=child.pid,
                supervisor_identity=shared.proc_identity(child.pid),
                plan=P0_TF_SOLVER_PLAN if phase_name in P0_TF_SOLVER_PHASES else P0_OFFSET_PLAN if phase_name=='p0_offset_localization' else P0_PERTURB_PLAN if phase_name in P0_PERTURB_PHASES else P0_SOLVER_PLAN if phase_name in ('p0_solver_semantics','p0_solver_system_reference') else P0_FITTING_PLAN if phase_name=='p0_fitting_forward' else P0_FORWARD_PLAN if phase_name in P0_FORWARD_PHASES else SOURCE_FAILURE_REPLAY_PLAN if phase_name=='c1_source_failure_replay' else C2_SOURCE_RESULTS_PLAN if queued else C1_FULL_SOURCE_PLAN,
                cpu_only=PHASES[phase_name]['resource']=='cpu',gpu_intentionally_hidden=PHASES[phase_name]['resource']=='cpu',
                supervisor_cpu_only=True)
    shared.atomic_json(launch_path,launch)
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:
        match=[(p,m) for p,m in campaign_attempts()
               if m.get('supervisor_pid')==child.pid
               and m.get('supervisor_identity')==launch['supervisor_identity']]
        pending=queue_status(phase_name) if queued else {}
        if match or pending.get('pid')==child.pid:
            if match:
                launch.update(status=match[-1][1]['status'],manifest=str(match[-1][0]))
            else:
                launch.update(status=pending['status'],queue_file=str(CAMPAIGN/f'queue-{phase_name}.json'))
            shared.atomic_json(launch_path,launch)
            refresh_checkpoint()
            return dict(launch=launch,launch_file=str(launch_path),progress=progress())
        if child.poll() is not None:
            launch.update(status='FAILED_TO_START',returncode=child.returncode)
            shared.atomic_json(launch_path,launch)
            raise RuntimeError('Supervisor exited before registering work; inspect '+str(log_path))
        time.sleep(.2)
    shared.atomic_json(launch_path,launch)
    return dict(launch=launch,launch_file=str(launch_path),
                next_action='Use progress; do not start a duplicate supervisor')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','report','run','wait-run','recover','start','progress','start-qualification','start-source-grid','start-final-qualification'])
    parser.add_argument('--phase',choices=PHASES)
    args=parser.parse_args()
    if args.action in ('start-qualification','start-source-grid','start-final-qualification'):
        current = status()
        if current['running']:
            raise RuntimeError('Existing run: ' + repr(current['running']))
        mode = {'start-qualification':'pilot','start-source-grid':'source-grid','start-final-qualification':'run'}[args.action]
        label = {'pilot':'','source-grid':'grid-','run':'final-'}[mode]
        output = CAMPAIGN / ('csmc-qualification-20261009-' + label + time.strftime('%H%M%S',time.gmtime()))
        script = 'docs/benchmarks/run_lorenz_qualification.py' if mode=='run' else 'docs/benchmarks/run_lorenz_downstream_qualification.py'
        command = [PYTHON,'-B',str(ROOT/script),mode,'--output',str(output)]
        log_path = output.with_suffix('.log')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1')
        with log_path.open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+10
        while not (output/'manifest.json').exists() and child.poll() is None and time.monotonic()<deadline:
            time.sleep(.1)
        result = dict(status='STARTED' if child.poll() is None else 'FAILED_TO_START',
                      supervisor_pid=child.pid,output_root=str(output),log=str(log_path))
    elif args.action in ('run','wait-run','start'):
        if not args.phase: parser.error('--phase required')
        if args.action == 'start':
            result = start_detached(args.phase)
        else:
            result=wait_and_execute(args.phase) if args.action=='wait-run' else execute(args.phase)
    elif args.action=='progress': result=progress()
    elif args.action=='report': result=bounded_report(args.phase)
    elif args.action=='recover': result=shared.recover(CAMPAIGN)
    else:
        result=status()
        shared.atomic_json(CAMPAIGN/'status.json',result)
    refresh_checkpoint()
    print(json.dumps(result,indent=2))
    if args.action in ('run','wait-run') and result['status']!='COMPLETE': raise SystemExit(1)
    if result.get('status')=='FAILED_TO_START': raise SystemExit(1)

if __name__=='__main__': main()
