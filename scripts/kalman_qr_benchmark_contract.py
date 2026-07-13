"""Fail-closed artifact contracts for the Kalman QR repair benchmark."""

from __future__ import annotations

import hashlib
import base64
import binascii
import copy
import json
import math
import os
import platform
import re
import secrets
import stat
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.v4"
METHOD_CONTRACT_VERSION = "measurement-boundaries-phase5-v1"
TIMING_BOUNDARY_VERSION = "separated-trace-execution-materialization-phase5-v1"
FIXTURE_CONTRACT_VERSION = "nested-prefix-base-observation-phase2-v1"
PARAMETER_BATCH_VERSION = "canonical-b16-locked-row-map-phase2-v1"
OBSERVATION_GENERATION_VERSION = "base-model-deterministic-sine-phase2-v1"
DIRECT_PARITY_TOLERANCES = {
    "float32": {
        "value_rtol": 2.0e-4,
        "value_atol": 2.0e-4,
        "score_rtol": 2.0e-4,
        "score_atol": 2.0e-4,
    },
    "float64": {
        "value_rtol": 1.0e-10,
        "value_atol": 1.0e-10,
        "score_rtol": 1.0e-8,
        "score_atol": 1.0e-9,
    },
}
STAGES = (
    "fixture",
    "trace",
    "first_executable_call",
    "warm_execution",
    "materialization",
    "parity",
    "payload_encoding",
    "payload_write",
    "envelope_write",
)
MEASUREMENT_EVENT_STAGES = STAGES[:-1]
PRIMARY_METHOD_IDS = (
    "batch_native_analytical_qr_score",
    "batch_native_autodiff_qr_score",
)
REFERENCE_METHOD_IDS = (
    "scalar_analytical_row_loop",
    "autodiff_row_loop_qr_score",
)
METHOD_IDS = PRIMARY_METHOD_IDS + REFERENCE_METHOD_IDS
METHOD_TERMINAL_STATES = ("passed", "failed", "timed_out", "crashed", "interrupted")
METHOD_STATES = METHOD_TERMINAL_STATES + ("pending", "running")
TOP_LEVEL_STATES = ("complete", "complete_with_failures", "failed", "interrupted")
AGGREGATE_CHECKS = (
    "identity_integrity",
    "record_integrity",
    "finite_output_metadata",
    "expected_dtype_shape",
    "primary_pair_complete",
    "comparator_parity",
)
HARNESS_ONLY_CHECKS = AGGREGATE_CHECKS[:2]
METHOD_LOCAL_CHECKS = AGGREGATE_CHECKS[:4]
PRIMARY_PAIR_CHECKS = AGGREGATE_CHECKS
SOURCE_PATHS = (
    "bayesfilter/__init__.py",
    "bayesfilter/linear/__init__.py",
    "bayesfilter/diagnostics.py",
    "bayesfilter/results_tf.py",
    "bayesfilter/structural.py",
    "bayesfilter/linear/dtypes_tf.py",
    "bayesfilter/linear/types_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "scripts/kalman_qr_benchmark_contract.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
)
SUPERVISOR_PATH = "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"
CONFIG_FIELDS = (
    "method_id",
    "method_contract_version",
    "dimension",
    "parameter_count",
    "timesteps",
    "batch_size",
    "dtype",
    "device",
    "jit_compile",
    "cpu_threads",
    "repeats",
    "subprocess_timeout_seconds",
    "xla_flags",
    "tf32_enabled",
    "jitter",
    "jitter_updates_filtered_covariance",
    "fixture_contract_version",
    "timing_boundary_version",
    "method_options",
)
FIXTURE_FIELDS = (
    "fixture_contract_version",
    "randomness",
    "seed",
    "dimension",
    "parameter_count",
    "timesteps",
    "batch_size",
    "dtype",
    "parameter_batch_version",
    "observation_generation_version",
    "external_input_hashes",
)
RUNTIME_FIELDS = (
    "interpreter",
    "python_implementation",
    "python_version",
    "platform",
    "distributions",
)
FINGERPRINT_FIELDS = (
    "source_fingerprint",
    "config_fingerprint",
    "runtime_fingerprint",
    "fixture_fingerprint",
    "schedule_fingerprint",
)

PHASE6_TRACE_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace.v1"
PHASE6_PILOT_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.pilot.v1"
PHASE6_SCALAR_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.scalar.v1"
PHASE6_FINAL_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.cpu_xla.v1"
PHASE6_ROUTING_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.p150_routing.v3"
PHASE6_BUDGET_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.budget.v1"
PHASE6_ATTESTATION_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.budget_attestation.v1"
)
PHASE6_DEPENDENCY_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.execution_dependencies.v1"
)
PHASE6_HANDOFF_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.handoff.v1"
PHASE6_SCHEDULE_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.schedule.v1"
PHASE6_R1_ARCHIVE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.gate_b_r1_invalid_harness_archive.v1"
)
PHASE6_R2_ARCHIVE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.gate_b_r2_invalid_harness_archive.v1"
)
PHASE6_CHILD_AUTHORITY_SNAPSHOT_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.child_authority_snapshot.v1"
)
PHASE6_CHILD_AUTHORITY_FAILURE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.child_authority_failure.v1"
)
PHASE6_CHILD_AUTHORITY_SHA256_ENV = (
    "BAYESFILTER_PHASE6_AUTHORITY_SNAPSHOT_SHA256"
)
PHASE6_SCHEMAS = (
    PHASE6_TRACE_SCHEMA,
    PHASE6_PILOT_SCHEMA,
    PHASE6_SCALAR_SCHEMA,
    PHASE6_FINAL_SCHEMA,
    PHASE6_ROUTING_SCHEMA,
)
PHASE6_TERMINAL_STATES = (
    "passed",
    "failed",
    "timed_out",
    "crashed",
    "interrupted",
)
PHASE6_IDENTITY_FIELDS = (
    "identity_id",
    "dimension",
    "parameter_count",
    "batch_size",
    "dtype",
    "method_id",
    "operation",
)
PHASE6_LEDGER_FIELDS = (
    "schema",
    "gate",
    "artifact_kind",
    "state",
    "bindings",
    "roster",
    "records",
    "events",
    "update_index",
    "aggregate",
    "nonclaims",
)
PHASE6_RECORD_FIELDS = (
    "identity",
    "state",
    "reason",
    "process",
    "evidence",
    "imported_from",
)
PHASE6_EVENT_FIELDS = (
    "update_index",
    "identity_id",
    "prior_state",
    "new_state",
    "timestamp_utc",
    "evidence_sha256",
)
PHASE6_PROCESS_FIELDS = (
    "command_argv",
    "cwd",
    "environment",
    "pid",
    "pgid",
    "process_start_ticks",
    "started_ns",
    "finished_ns",
    "elapsed_seconds",
    "deadline_seconds",
    "term_sent",
    "kill_sent",
    "reaped",
    "reap_status",
    "process_group_gone",
    "returncode",
    "timed_out",
    "stdout_bytes",
    "stdout_total_bytes",
    "stdout_capture_status",
    "stdout_sha256",
    "stdout_base64",
    "stdout_tail",
    "stderr_bytes",
    "stderr_total_bytes",
    "stderr_capture_status",
    "stderr_sha256",
    "stderr_base64",
    "stderr_tail",
)
PHASE6_RUNNING_PROCESS_FIELDS = (
    "command_argv",
    "cwd",
    "environment",
    "pid",
    "pgid",
    "process_start_ticks",
    "started_ns",
    "deadline_seconds",
)
PHASE6_BLOB_FIELDS = (
    "path",
    "present",
    "byte_count",
    "sha256",
    "base64",
    "strict_json",
)
PHASE6_IMPORTED_FROM_FIELDS = (
    "kind",
    "pilot_artifact_sha256",
    "pilot_record_sha256",
)
PHASE6_EVIDENCE_FIELDS = (
    "classification",
    "child_artifact",
    "payload_sidecar",
    "progress_journal",
    "dependency_manifest_before_builder",
    "dependency_manifest_after_terminal",
    "dependency_coverage_before",
    "dependency_coverage_after",
)
PHASE6_BUDGET_FIELDS = (
    "schema",
    "authority_id",
    "gate",
    "plan",
    "opening_hash_ledger",
    "dependency_discovery",
    "source_hashes",
    "commands",
    "schedules",
    "artifacts",
    "budget",
    "inputs",
    "nonclaims",
)
PHASE6_ATTESTATION_FIELDS = (
    "schema",
    "authority_id",
    "gate",
    "proposal",
    "plan",
    "review",
    "verdict",
    "review_strength",
    "timestamp_utc",
)
PHASE6_PATH_DIGEST_FIELDS = ("path", "sha256")
PHASE6_R1_ARCHIVE_FIELDS = (
    "schema",
    "authority_id",
    "disposition",
    "files",
    "diagnosis",
    "no_live_process",
    "pre_edit_lane_hashes",
    "protected_hashes",
    "nonclaims",
)
PHASE6_R1_ARCHIVE_FILE_FIELDS = (
    "path",
    "byte_count",
    "sha256",
    "role",
    "disposition",
)
PHASE6_R2_ARCHIVE_FIELDS = (
    "schema",
    "authority_id",
    "disposition",
    "files",
    "absent_paths",
    "work_root_entries",
    "budget_state_entries",
    "diagnosis",
    "no_live_process",
    "protected_hashes",
    "nonclaims",
)
PHASE6_R2_ARCHIVE_FILE_FIELDS = (
    "path",
    "byte_count",
    "sha256",
    "role",
    "format",
    "disposition",
)
PHASE6_CHILD_AUTHORITY_SNAPSHOT_FIELDS = (
    "schema",
    "bindings",
    "schedule_row",
    "payload_sha256",
)
PHASE6_OPENING_LEDGER_FIELDS = ("path", "sha256", "entries", "entries_sha256")
PHASE6_OPENING_ENTRY_FIELDS = ("opening_state", "path", "sha256")
PHASE6_SCHEMA_CONTRACTS = {
    PHASE6_TRACE_SCHEMA: ("gate_b", "trace_census", "trace"),
    PHASE6_PILOT_SCHEMA: ("gate_b", "cpu_xla_pilot", "xla"),
    PHASE6_SCALAR_SCHEMA: ("gate_c", "scalar_references", "scalar_reference"),
    PHASE6_FINAL_SCHEMA: ("gate_c", "cpu_xla_final", "xla"),
    PHASE6_ROUTING_SCHEMA: ("gate_c", "p150_routing", "p150_routing"),
}
PHASE6_BINDING_FIELDS = (
    "authority_id",
    "proposal",
    "attestation",
    "schedule",
    "phase45_evidence",
    "authority_inputs",
    "runtime_predecessors",
)
PHASE6_RUNTIME_PREDECESSOR_FIELDS = (
    "kind",
    "producer_ledger_schema",
    "authority_id",
    "schedule_sha256",
    "artifact",
)
PHASE6_SCHEDULE_RECORD_FIELDS = (
    "identity",
    "case_id",
    "attempt_id",
    "config",
    "fingerprints",
    "resume_key",
    "child_command_argv",
)
PHASE6_SCHEDULE_FINGERPRINT_FIELDS = (
    "identity",
    "case_id",
    "config",
    "source_fingerprint",
    "runtime_fingerprint",
    "fixture_fingerprint",
)
PHASE6_TERMINAL_REASON_BY_STATE = {
    "passed": {"child_passed"},
    "failed": {
        "child_nonzero_exit",
        "invalid_child_evidence",
        "authority_revalidation_failed",
        "dependency_provenance_invalid",
        "common_invalidity",
    },
    "timed_out": {"child_execution_deadline_exceeded", "authority_revalidation_failed"},
    "crashed": {"child_signal_exit", "authority_revalidation_failed"},
    "interrupted": {"supervisor_recovery", "outer_termination", "keyboard_interrupt"},
}
PHASE6_CLASSIFICATION_BY_STATE = {
    "passed": {"trace_pass", "method_pass", "scalar_reference_pass"},
    "failed": {
        "common_invalidity",
        "cpu_backend_or_method_failure",
        "method_local_failure",
        "trace_structural_failure",
    },
    "timed_out": {
        "common_invalidity",
        "cpu_backend_or_cell_timeout",
        "trace_timeout",
        "scalar_reference_timeout",
    },
    "crashed": {
        "common_invalidity",
        "cpu_backend_or_method_failure",
        "trace_crash",
        "scalar_reference_crash",
    },
    "interrupted": {"supervisor_interruption"},
}
PHASE6_NOT_LAUNCHED_REASONS = {
    "trace_gate_not_passed",
    "after_smaller_batch_failure",
    "p50_dependency_not_launched",
    "p50_dependency_failed",
    "invalid_dependency_evidence",
    "after_smaller_p150_batch_failure",
    "common_invalidity",
    "global_budget_exhausted",
    "not_in_gate_b_pilot",
}
PHASE6_R1_PLAN_RELATIVE = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md"
)
PHASE6_PLAN_RELATIVE = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-subplan-2026-07-12.md"
)
PHASE6_PLAN_SHA256 = (
    "3af2959c719e62b4fb02d9e7c78b3be86521d7e62b757d35d2e4acede679ba1a"
)
PHASE6_REPAIR_RESULT_RELATIVE = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-result-2026-07-12.md"
)
PHASE6_PLAN_REVIEW_RELATIVE = (
    "docs/reviews/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-subplan-review-final-2026-07-12.md"
)
PHASE6_PLAN_REVIEW_SHA256 = (
    "99523d5660988dc08cf5509391a3f7c6ff0ba51cd4352a9da09272f2d1bc4b27"
)
PHASE6_REPAIR_RESULT_REVIEW_RELATIVE = (
    "docs/reviews/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-result-review-final-2026-07-12.md"
)
PHASE6_R1_AUTHORITY_ID = (
    "ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837"
)
PHASE6_R1_ARCHIVE_RELATIVE = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase6_gateb_r1_invalid_harness_archive_2026-07-12.json"
)
PHASE6_R1_WORK_ROOT = "/tmp/kalman_qr_phase6_cpu_xla"
PHASE6_R2_AUTHORITY_ID = (
    "4807a429ce935c95392f6af62266ef53a2e6165c8b8cc5e0cb415ba80fb26096"
)
PHASE6_R2_WORK_ROOT = "/tmp/kalman_qr_phase6_cpu_xla_gateb_r2"
PHASE6_R2_ARCHIVE_RELATIVE = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase6_gateb_r2_invalid_harness_archive_2026-07-12.json"
)
PHASE6_WORK_ROOT = "/tmp/kalman_qr_phase6_cpu_xla_gateb_r3"
PHASE6_OPENING_HASH_LEDGER = f"{PHASE6_R1_WORK_ROOT}/pre_edit_path_hashes.sha256"
PHASE6_OPENING_HASH_LEDGER_SHA256 = (
    "9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911"
)
PHASE6_SUPERVISOR_RELATIVE = SUPERVISOR_PATH
PHASE6_BENCHMARK_RELATIVE = "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
PHASE6_PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
PHASE6_IMPORT_DISCOVERY_OUTPUT = (
    f"{PHASE6_WORK_ROOT}/import_discovery.json"
)
PHASE6_IMPORT_DISCOVERY_ARGV = [
    PHASE6_PYTHON,
    PHASE6_BENCHMARK_RELATIVE,
    "--phase6-import-discovery",
    "--device",
    "cpu",
    "--cpu-threads",
    "1",
    "--output-json",
    PHASE6_IMPORT_DISCOVERY_OUTPUT,
]
PHASE6_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}
PHASE6_REQUIRED_SOURCE_PATHS = tuple(sorted({*SOURCE_PATHS, SUPERVISOR_PATH}))
PHASE6_OPENING_MUTABLE_PATHS = (
    "scripts/kalman_qr_benchmark_contract.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    PHASE6_SUPERVISOR_RELATIVE,
    "tests/test_kalman_qr_measurement_boundaries.py",
    "tests/test_kalman_qr_parameter_count_scaling_harness.py",
    "tests/test_kalman_qr_benchmark_contract.py",
)
PHASE6_OPENING_FIXED_PATHS = (
    "tests/test_kalman_qr_batch_native_autodiff.py",
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "bayesfilter/__init__.py",
    "bayesfilter/linear/__init__.py",
    "bayesfilter/diagnostics.py",
    "bayesfilter/structural.py",
    "bayesfilter/results_tf.py",
    "bayesfilter/linear/dtypes_tf.py",
    "bayesfilter/linear/types_tf.py",
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-result-2026-07-11.md",
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-result-2026-07-11.md",
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json",
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json",
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json",
    PHASE6_R1_PLAN_RELATIVE,
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-subplan-codex-substitute-review-round5-runtime-2026-07-11.md",
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-subplan-codex-substitute-review-round5-evidence-2026-07-11.md",
)
PHASE6_R1_GATE_B_ARTIFACTS = {
    "trace_output_json": "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json",
    "pilot_output_json": "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json",
}
PHASE6_GATE_B_ARTIFACTS = {
    "trace_output_json": (
        "docs/benchmarks/"
        "kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json"
    ),
    "pilot_output_json": (
        "docs/benchmarks/"
        "kalman_qr_batched_xla_repair_phase6_gateb_r3_cpu_xla_pilot_2026-07-12.json"
    ),
}
PHASE6_GATE_C_ARTIFACTS = {
    "scalar_output_json": "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_scalar_references_2026-07-11.json",
    "routing_output_json": "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_p150_routing_2026-07-11.json",
    "final_output_json": "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_2026-07-11.json",
}
PHASE6_R1_GATE_B_BUDGET_RELATIVE = (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json"
)
PHASE6_R1_GATE_B_ATTESTATION_RELATIVE = (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json"
)
PHASE6_GATE_B_BUDGET_RELATIVE = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_2026-07-12.json"
)
PHASE6_GATE_B_ATTESTATION_RELATIVE = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_attestation_2026-07-12.json"
)
PHASE6_GATE_B_REVIEW_RELATIVE = (
    "docs/reviews/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-budget-review-round1-2026-07-12.md"
)
PHASE6_GATE_B_INPUT_RELATIVES = (
    PHASE6_R2_ARCHIVE_RELATIVE,
    PHASE6_REPAIR_RESULT_RELATIVE,
    PHASE6_PLAN_REVIEW_RELATIVE,
    PHASE6_REPAIR_RESULT_REVIEW_RELATIVE,
)
PHASE6_GATE_C_BUDGET_RELATIVE = (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_remaining_budget_2026-07-11.json"
)
PHASE6_GATE_C_ATTESTATION_RELATIVE = (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gatec_budget_attestation_2026-07-11.json"
)
PHASE6_OPENING_ABSENT_PATHS = (
    "tests/test_kalman_qr_phase6_cpu_xla_gates.py",
    *PHASE6_R1_GATE_B_ARTIFACTS.values(),
    PHASE6_R1_GATE_B_BUDGET_RELATIVE,
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round1-2026-07-11.md",
    PHASE6_R1_GATE_B_ATTESTATION_RELATIVE,
    PHASE6_GATE_C_BUDGET_RELATIVE,
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-budget-review-round1-2026-07-11.md",
    PHASE6_GATE_C_ATTESTATION_RELATIVE,
    *PHASE6_GATE_C_ARTIFACTS.values(),
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-result-2026-07-11.md",
)
PHASE6_GRAPHDEF_MAX_DECODED_BYTES = 16 * 1024 * 1024
PHASE6_GRAPHDEF_MAX_TOTAL_DECODED_BYTES = 256 * 1024 * 1024
PHASE6_TRACE_MAX_JSON_BYTES = 512 * 1024 * 1024
PHASE6_GRAPHDEF_DECODE_CHUNK_CHARS = 1024 * 1024
PHASE6_STREAM_MAX_BYTES = 16 * 1024 * 1024
PHASE6_TRACE_STAGES = (
    "fixture",
    "pre_builder_provenance",
    "selected_method_construction",
    "get_concrete_function",
    "graphdef_extraction",
    "terminal_provenance",
    "envelope_write",
)
PHASE6_NONCLAIMS = (
    "no method ranking",
    "no GPU readiness",
    "no CPU or GPU scalability claim",
    "no HMC or posterior correctness claim",
    "no default, production, or scientific validity claim",
)
PHASE6_R1_ARCHIVE_NONCLAIMS = (
    "r1 child-internal passed is not method or backend evidence",
    "r1 trace ledger is non-final invalid-harness evidence",
    "no CPU-XLA viability or performance claim",
    "no GPU, HMC, posterior, default, production, or scientific validity claim",
)
PHASE6_R2_ARCHIVE_NONCLAIMS = (
    "r2 failed before fixture, trace, concrete-function, XLA, or Kalman target work",
    "r2 is immutable invalid-harness evidence and must not be resumed or imported",
    "no trace-structure, CPU-XLA viability, memory, or performance claim",
    "no GPU, HMC, posterior, default, production, or scientific validity claim",
)
PHASE6_R2_ARCHIVE_FILE_SPECS = (
    (
        "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-subplan-2026-07-12.md",
        "575cf2fc7661bef2be6282ca57570d27bbf4490f2b01ecaad9e5cbb4c5efb004",
        "r2_repair_subplan",
        "markdown",
    ),
    (
        "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-result-2026-07-12.md",
        "12518676d76c1497a070e3258463d66b7d81433918efcf7405d9816671dfc091",
        "r2_repair_result",
        "markdown",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-subplan-codex-review-round3-2026-07-12.md",
        "39f41d15394e721a7a481d36abf65ac340dea656f6a96181f40e1577a0e1b9f5",
        "r2_repair_plan_review",
        "markdown",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-result-review-round2-2026-07-12.md",
        "df3f732227dda254ea89f070da3f8f5058f14a08edc5f8613e9e43bf293d9ee0",
        "r2_repair_result_review",
        "markdown",
    ),
    (
        PHASE6_R1_ARCHIVE_RELATIVE,
        "caacd7144a0e6b7767487d7cc3a48145702983487ac1ab6885f5f97ba2f9607a",
        "r1_archive_input",
        "json",
    ),
    (
        "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json",
        "187594f66a2a87e237d697d52085318731efea986e077b2972e7a1cf44b46359",
        "r2_proposal",
        "json",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-budget-review-round1-2026-07-12.md",
        "c4c7055eb5c416310867c831e93fb3cb111d1da76c6d2b3e6cf25b409d940acf",
        "r2_proposal_review",
        "markdown",
    ),
    (
        "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json",
        "4fa7b0cbef59c826804dc9e156fffe9660aabdb05d26b0d88458893b89d566cd",
        "r2_attestation",
        "json",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-final-skeptical-runtime-audit-2026-07-12.md",
        "b95d56f31c5a5f47eba601df6695070c98e12ebac70e0ebc0f43f207709b566b",
        "r2_skeptical_runtime_audit",
        "markdown",
    ),
    (
        "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-trace-pilot-result-2026-07-12.md",
        "a86c0f35c4ecdb22e372f7a0bcbb2b57cca63cd85151a2bbeea88eb1161899ba",
        "r2_harness_invalid_result",
        "markdown",
    ),
    (
        f"{PHASE6_R2_WORK_ROOT}/import_discovery.json",
        "8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac",
        "r2_import_discovery",
        "json",
    ),
    (
        f"{PHASE6_R2_WORK_ROOT}/budget_state/gate_b-{PHASE6_R2_AUTHORITY_ID}.json",
        "a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f",
        "r2_running_budget_state",
        "json",
    ),
    (
        f"{PHASE6_R2_WORK_ROOT}/budget_state/gate_b-{PHASE6_R2_AUTHORITY_ID}.json.lease",
        "ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456",
        "r2_released_budget_lease",
        "json",
    ),
)
PHASE6_R2_ARCHIVE_ABSENT_PATHS = (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_trace_census_2026-07-12.json",
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_cpu_xla_pilot_2026-07-12.json",
    f"{PHASE6_R2_WORK_ROOT}/trace",
    f"{PHASE6_R2_WORK_ROOT}/pilot",
    f"{PHASE6_R2_WORK_ROOT}/children",
    f"{PHASE6_R2_WORK_ROOT}/progress",
)
PHASE6_R1_ARCHIVE_FILE_SPECS = (
    (
        PHASE6_R1_PLAN_RELATIVE,
        "b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b",
        "proposal_bound_parent_plan",
    ),
    (
        PHASE6_OPENING_HASH_LEDGER,
        PHASE6_OPENING_HASH_LEDGER_SHA256,
        "proposal_bound_opening_hash_ledger",
    ),
    (
        f"{PHASE6_R1_WORK_ROOT}/import_discovery.json",
        "5e24afc0246899fca8bb7924b507e6020ed3b2941f2e4a9bc491b23f360da0bf",
        "proposal_import_discovery",
    ),
    (
        PHASE6_R1_GATE_B_BUDGET_RELATIVE,
        "e1b4cabba3dfd1ca292c4d7842d02ba86273001275b5f0a3b69ed0851a0ec823",
        "r1_proposal",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round1-2026-07-11.md",
        "166bd5594d01371e6b08186f885e1cc3f06130d0072aa4a5363802412dc6e0e8",
        "r1_proposal_review",
    ),
    (
        PHASE6_R1_GATE_B_ATTESTATION_RELATIVE,
        "583e7842c3af2ebe0e00598224a86dd5cbf9c2627f0a22ca0638e25f870cd153",
        "r1_attestation",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-preflight-2026-07-12.md",
        "136c6d9d76a30df1068c2e05df1e5e7632ebf1fa9fe9288dda075a768cf7aa90",
        "r1_skeptical_runtime_preflight",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-subplan-claude-opus-review-round1-2026-07-12.md",
        "5e1c7b199bf1b7d30eab28c93bdf8ecad9dcf841c1bc3538fe07a376e46276ad",
        "r1_subplan_review_round1",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-subplan-claude-opus-review-round2-2026-07-12.md",
        "621fdc772c088092173db9326bf39f8fec77135ded4acbf7dd05a0675efd8d0c",
        "r1_subplan_review_round2",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-subplan-claude-opus-review-round3-2026-07-12.md",
        "b46b984eb625955885501a89b61836159dea2691a40e07962e455211ea3ad5b4",
        "r1_subplan_review_round3",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-subplan-claude-opus-review-round4-2026-07-12.md",
        "e136b18fca7bd87cd940b30627a798798e84d45a74f05cd4de51c572b355ccd0",
        "r1_subplan_review_round4",
    ),
    (
        "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-subplan-post-proposal-artifact-repair-2026-07-12.md",
        "75cc3b96017d76642719b36b5bd268ae9dfe6f1b09fe8b28e9dc3fee85fe3b57",
        "r1_post_proposal_artifact_repair_review",
    ),
    (
        PHASE6_R1_GATE_B_ARTIFACTS["trace_output_json"],
        "9def8eb0f728a2262353002234bcbfe39aaa79bd49632e0737df2e0c8bf830ef",
        "r1_nonfinal_trace_ledger",
    ),
    (
        f"{PHASE6_R1_WORK_ROOT}/trace/3ad6c9faf97fe6395a287e17.json",
        "a4235d0d399bde8148fbf07138aefae17b41cceaf6ecae7dd7e06fd36e333f31",
        "r1_first_trace_child",
    ),
    (
        f"{PHASE6_R1_WORK_ROOT}/trace/3ad6c9faf97fe6395a287e17.jsonl",
        "71585306fc1f579e780c059f99f3e2595fb02aac61030345b4a5a585c02bc365",
        "r1_first_trace_progress_journal",
    ),
    (
        f"{PHASE6_R1_WORK_ROOT}/budget_state/gate_b-ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837.json",
        "7cb269c18d4232851c0f620bc120e024ef303527c91945f11a24a56692a5deea",
        "r1_budget_state",
    ),
    (
        f"{PHASE6_R1_WORK_ROOT}/budget_state/gate_b-ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837.json.lease",
        "f0e1c0ce9876b0da20ef8e2715713a598921bc3d0d35a7d8068d617d7c14384e",
        "r1_budget_lease",
    ),
)
PHASE6_R1_PRE_EDIT_LANE_HASHES = {
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py": (
        "efcd2925a8c12a69b0a6950c6315b1e9c53876019317a6e0fb970ea0d7cd4a6c"
    ),
    "scripts/kalman_qr_benchmark_contract.py": (
        "730e4374ca046dcf7cc1aa76af69d1c7942c08eac17120006577ec37ca1ec06c"
    ),
    SUPERVISOR_PATH: (
        "5de6f8e276abf5e9312207fb5de748d8a55c196547371b835fa764ae48a15de2"
    ),
    "tests/test_kalman_qr_phase6_cpu_xla_gates.py": (
        "5d26b60c5edd7996a1eee334cad6ac606e15a52f6c7df9af0c4ea448171c7268"
    ),
    "tests/test_kalman_qr_phase6_gatea_runtime_controls.py": (
        "d0485a4ed1ae2305a75f15bb8e600ea48e43a22037af5e4d9659f8ce3d0798b6"
    ),
    "tests/test_kalman_qr_phase6_import_discovery_cli.py": (
        "e9b4dada5902835105cab7e6ff7006c6c5ce10197eef2166842a2d83f3190715"
    ),
}
PHASE6_PROTECTED_HASHES = {
    "bayesfilter/linear/kalman_qr_tf.py": (
        "ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b"
    ),
    "bayesfilter/linear/kalman_qr_derivatives_tf.py": (
        "d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57"
    ),
    "bayesfilter/linear/qr_factor_tf.py": (
        "bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401"
    ),
}
PHASE6_PHASE45_EVIDENCE = {
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json": (
        "987815216c6919ee52de69e1511cba4dd9a1827bb24a8099747907fb8134ba4e"
    ),
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json": (
        "a74be199826f12b2c7931e7bb8d82d510826b69fcb7232ca3ad0b255b90ce74d"
    ),
}
PHASE7_NONCLAIMS = (
    "no target numerical claim when scalar evidence is missing",
    "no GPU readiness",
    "no CPU/GPU scalability",
    "no method ranking",
    "no HMC/posterior/default/production/scientific claim",
)


class ContractError(ValueError):
    """Raised when artifact evidence does not satisfy the current contract."""


def _reject_constant(token: str) -> None:
    raise ContractError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite float at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    _validate_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"invalid strict JSON: {exc}") from exc


def read_strict_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read JSON artifact {path}: {exc}") from exc


def read_bounded_phase6_trace_json(path: Path) -> Any:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"cannot stat Phase 6 trace artifact {path}: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > PHASE6_TRACE_MAX_JSON_BYTES
    ):
        raise ContractError("Phase 6 trace artifact violates regular-file or size cap")
    return read_strict_json(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, strict_json_dumps(value, indent=2) + "\n")


def atomic_write_encoded_json(path: Path, encoded: str) -> None:
    """Atomically write already validated strict JSON without re-encoding it."""

    strict_json_loads(encoded)
    atomic_write_text(path, encoded if encoded.endswith("\n") else encoded + "\n")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def durable_json_sha256(value: Any) -> str:
    """Digest the exact bytes emitted by durable_atomic_write_json."""

    raw = (strict_json_dumps(value, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_finite_nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and value >= 0


def validate_stage_events(events: Any) -> bool:
    if not isinstance(events, list) or len(events) != len(MEASUREMENT_EVENT_STAGES):
        return False
    previous_finished: int | None = None
    for index, (event, expected_stage) in enumerate(zip(events, MEASUREMENT_EVENT_STAGES)):
        if not isinstance(event, Mapping) or set(event) != {
            "sequence_index",
            "stage",
            "entered_ns",
            "finished_ns",
        }:
            return False
        entered = event["entered_ns"]
        finished = event["finished_ns"]
        if (
            event["sequence_index"] != index
            or event["stage"] != expected_stage
            or type(entered) is not int
            or type(finished) is not int
            or entered < 0
            or finished < entered
            or (previous_finished is not None and previous_finished > entered)
        ):
            return False
        previous_finished = finished
    return True


def measurement_record_checks(record: Mapping[str, Any]) -> dict[str, bool]:
    measurement = record.get("measurement")
    if not isinstance(measurement, Mapping):
        measurement = {}
    measurement_keys = {
        "timing_boundary_version",
        "requested_repeats",
        "stage_events",
        "durations",
        "synchronization",
        "invocation_counts",
        "graphdef",
        "direct_output_parity",
        "payload_sidecar",
        "envelope_write_measured",
    }
    durations = measurement.get("durations")
    if not isinstance(durations, Mapping):
        durations = {}
    repeats = measurement.get("requested_repeats")
    warm = durations.get("warm_execution_seconds")
    duration_keys = {
        "fixture_seconds",
        "trace_seconds",
        "first_executable_call_seconds",
        "warm_execution_seconds",
        "materialization_seconds",
        "parity_seconds",
        "payload_encoding_seconds",
        "artifact_write_seconds",
    }
    durations_valid = (
        set(durations) == duration_keys
        and type(repeats) is int
        and repeats > 0
        and isinstance(warm, list)
        and len(warm) == repeats
        and all(_is_finite_nonnegative_number(value) for value in warm)
        and all(
            _is_finite_nonnegative_number(durations.get(key))
            for key in duration_keys - {"warm_execution_seconds"}
        )
    )
    events = measurement.get("stage_events")
    event_durations: dict[str, float] = {}
    if validate_stage_events(events):
        event_durations = {
            event["stage"]: (event["finished_ns"] - event["entered_ns"]) / 1.0e9
            for event in events
        }
    duration_event_consistency = durations_valid and bool(event_durations)
    if duration_event_consistency:
        for stage, duration_key in (
            ("fixture", "fixture_seconds"),
            ("trace", "trace_seconds"),
            ("first_executable_call", "first_executable_call_seconds"),
            ("materialization", "materialization_seconds"),
            ("parity", "parity_seconds"),
            ("payload_encoding", "payload_encoding_seconds"),
            ("payload_write", "artifact_write_seconds"),
        ):
            duration_event_consistency = duration_event_consistency and math.isclose(
                float(durations[duration_key]),
                event_durations[stage],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        duration_event_consistency = duration_event_consistency and (
            sum(float(value) for value in warm)
            <= event_durations["warm_execution"] + 1.0e-12
        )
    synchronization = measurement.get("synchronization")
    if not isinstance(synchronization, Mapping):
        synchronization = {}
    synchronization_method = synchronization.get("method")
    expected_scalar_count = (
        0
        if synchronization_method == "tf.experimental.async_wait"
        else 1 + repeats
        if synchronization_method == "scalar_sentinel"
        else None
    )
    counts_valid = (
        set(synchronization) == {
            "method",
            "sentinel_definition",
            "scalar_materialization_count",
            "full_output_materialization_count",
            "parity_residual_materialization_count",
        }
        and expected_scalar_count is not None
        and synchronization.get("scalar_materialization_count") == expected_scalar_count
        and synchronization.get("full_output_materialization_count") == 1
        and synchronization.get("parity_residual_materialization_count") == 1
        and synchronization.get("sentinel_definition")
        == (
            None
            if synchronization_method == "tf.experimental.async_wait"
            else "reduce_sum(value)+reduce_sum(score)"
        )
    )
    invocation = measurement.get("invocation_counts")
    invocation_valid = (
        isinstance(invocation, Mapping)
        and set(invocation) == {
            "before_first_executable_call",
            "after_first_executable_call",
            "after_warm_execution",
            "after_reference_call",
        }
        and invocation.get("before_first_executable_call") == 0
        and invocation.get("after_first_executable_call") == 1
        and invocation.get("after_warm_execution") == 1 + repeats
        and invocation.get("after_reference_call") == 2 + repeats
    )
    graph = measurement.get("graphdef")
    graph_valid = (
        isinstance(graph, Mapping)
        and set(graph) == {"node_count", "serialized_bytes"}
        and type(graph.get("node_count")) is int
        and graph["node_count"] > 0
        and type(graph.get("serialized_bytes")) is int
        and graph["serialized_bytes"] > 0
    )
    parity = measurement.get("direct_output_parity")
    output_metadata = record.get("output_metadata")
    dtype = (
        output_metadata.get("value_dtype")
        if isinstance(output_metadata, Mapping)
        else parity.get("dtype")
        if isinstance(parity, Mapping)
        else None
    )
    expected_tolerance = DIRECT_PARITY_TOLERANCES.get(dtype)
    parity_valid = (
        isinstance(parity, Mapping)
        and set(parity)
        == {
            "passed",
            "dtype",
            "value_rtol",
            "value_atol",
            "score_rtol",
            "score_atol",
            "value_reference_max_abs",
            "score_reference_max_abs",
            "value_max_abs_residual",
            "score_max_abs_residual",
        }
        and expected_tolerance is not None
        and parity.get("dtype") == dtype
        and parity.get("value_rtol") == expected_tolerance["value_rtol"]
        and parity.get("value_atol") == expected_tolerance["value_atol"]
        and parity.get("score_rtol") == expected_tolerance["score_rtol"]
        and parity.get("score_atol") == expected_tolerance["score_atol"]
        and parity.get("passed") is True
        and _is_finite_nonnegative_number(parity.get("value_reference_max_abs"))
        and _is_finite_nonnegative_number(parity.get("score_reference_max_abs"))
        and _is_finite_nonnegative_number(parity.get("value_max_abs_residual"))
        and _is_finite_nonnegative_number(parity.get("score_max_abs_residual"))
        and parity["value_max_abs_residual"]
        <= expected_tolerance["value_atol"]
        + expected_tolerance["value_rtol"] * parity["value_reference_max_abs"]
        and parity["score_max_abs_residual"]
        <= expected_tolerance["score_atol"]
        + expected_tolerance["score_rtol"] * parity["score_reference_max_abs"]
    )
    sidecar = measurement.get("payload_sidecar")
    sidecar_valid = (
        isinstance(sidecar, Mapping)
        and set(sidecar) == {"path", "sha256", "write_count"}
        and isinstance(sidecar.get("path"), str)
        and sidecar["path"].endswith(".payload.json")
        and isinstance(sidecar.get("sha256"), str)
        and len(sidecar["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in sidecar["sha256"])
        and sidecar.get("write_count") == 1
    )
    forbidden_names = " ".join(str(key) for key in measurement).lower()
    return {
        "closed_measurement_schema": set(measurement) == measurement_keys,
        "timing_boundary_identity": measurement.get("timing_boundary_version")
        == TIMING_BOUNDARY_VERSION,
        "ordered_stage_events": validate_stage_events(measurement.get("stage_events")),
        "duration_contract": durations_valid,
        "duration_event_consistency": duration_event_consistency,
        "synchronization_counts": counts_valid,
        "invocation_counts": invocation_valid,
        "graphdef_metadata": graph_valid,
        "direct_output_parity": parity_valid,
        "payload_sidecar_identity": sidecar_valid,
        "outer_envelope_unmeasured": measurement.get("envelope_write_measured") is False,
        "no_compile_subtraction_field": (
            "first_minus_warm" not in forbidden_names
            and "compilation_seconds" not in forbidden_names
        ),
    }


def measurement_record_is_valid(record: Mapping[str, Any]) -> bool:
    checks = measurement_record_checks(record)
    return bool(checks) and all(checks.values())


def payload_sidecar_matches_record(
    record: Mapping[str, Any],
    *,
    expected_path: Path | None = None,
) -> bool:
    measurement = record.get("measurement")
    if not isinstance(measurement, Mapping):
        return False
    sidecar = measurement.get("payload_sidecar")
    if not isinstance(sidecar, Mapping):
        return False
    try:
        path = Path(sidecar["path"])
        if expected_path is not None and path.resolve() != expected_path.resolve():
            return False
        content = path.read_text(encoding="utf-8")
        payload = strict_json_loads(content)
    except (KeyError, OSError, ContractError, TypeError, ValueError):
        return False
    expected_payload = {
        "case_id": record.get("case_id"),
        "method_id": record.get("method_id"),
        "output_metadata": record.get("output_metadata"),
        "outputs": record.get("outputs"),
        "graphdef": measurement.get("graphdef"),
        "direct_output_parity": measurement.get("direct_output_parity"),
    }
    return file_sha256(path) == sidecar.get("sha256") and payload == expected_payload


def source_manifest(repo_root: Path, *, include_supervisor: bool) -> dict[str, Any]:
    relative_paths = list(SOURCE_PATHS)
    if include_supervisor:
        relative_paths.append(SUPERVISOR_PATH)
    files = []
    for relative in sorted(relative_paths):
        path = repo_root / relative
        if not path.is_file():
            raise ContractError(f"missing execution-affecting source path {relative}")
        files.append({"path": relative, "sha256": file_sha256(path)})
    return {"files": files, "source_fingerprint": canonical_sha256(files)}


def _exact_fields(payload: Mapping[str, Any], fields: Sequence[str], label: str) -> dict[str, Any]:
    missing = sorted(set(fields) - set(payload))
    extra = sorted(set(payload) - set(fields))
    if missing or extra:
        raise ContractError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return {field: payload[field] for field in fields}


def config_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    exact = _exact_fields(payload, CONFIG_FIELDS, "config")
    method_id = exact["method_id"]
    if method_id not in METHOD_IDS:
        raise ContractError(f"unknown method_id {method_id!r}")
    if exact["method_contract_version"] != METHOD_CONTRACT_VERSION:
        raise ContractError("stale method_contract_version")
    if exact["timing_boundary_version"] != TIMING_BOUNDARY_VERSION:
        raise ContractError("stale timing_boundary_version")
    method_options = exact["method_options"]
    if not isinstance(method_options, Mapping) or method_options:
        raise ContractError("current method_options must be an empty object")
    return {"config": exact, "config_fingerprint": canonical_sha256(exact)}


def fixture_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    exact = _exact_fields(payload, FIXTURE_FIELDS, "fixture")
    if exact["randomness"] != "deterministic" or exact["seed"] is not None:
        raise ContractError("current fixture must be deterministic with seed=null")
    if exact["external_input_hashes"] != {}:
        raise ContractError("current generated fixture has no external input hashes")
    return {"fixture": exact, "fixture_fingerprint": canonical_sha256(exact)}


def _distribution_for_module(module: str) -> dict[str, str]:
    distributions = metadata.packages_distributions().get(module) or []
    if len(distributions) != 1:
        raise ContractError(f"expected one distribution for {module}, found {distributions}")
    name = distributions[0]
    return {"distribution": name, "version": metadata.version(name)}


def current_runtime_payload() -> dict[str, Any]:
    return {
        "interpreter": str(Path(sys.executable).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "distributions": {
            "numpy": _distribution_for_module("numpy"),
            "tensorflow": _distribution_for_module("tensorflow"),
            "tensorflow_probability": _distribution_for_module("tensorflow_probability"),
        },
    }


def runtime_manifest(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    exact = _exact_fields(payload or current_runtime_payload(), RUNTIME_FIELDS, "runtime")
    distributions = exact["distributions"]
    if not isinstance(distributions, Mapping):
        raise ContractError("runtime distributions must be an object")
    expected = {"tensorflow", "tensorflow_probability", "numpy"}
    if set(distributions) != expected:
        raise ContractError("runtime distributions must identify TensorFlow, TFP, and NumPy")
    for module, record in distributions.items():
        if set(record) != {"distribution", "version"} or not all(record.values()):
            raise ContractError(f"incomplete runtime distribution metadata for {module}")
    return {"runtime": exact, "runtime_fingerprint": canonical_sha256(exact)}


def case_id(config: Mapping[str, Any]) -> str:
    keys = ("dimension", "parameter_count", "timesteps", "batch_size", "dtype", "device")
    return "-".join(f"{key}={config[key]}" for key in keys)


def build_schedule_manifest(
    identities: Iterable[Mapping[str, str]],
    mandatory_checks: Sequence[str],
    *,
    harness_contract_test_only: bool,
) -> dict[str, Any]:
    required_identity_fields = {"case_id", "method_id", *FINGERPRINT_FIELDS[:-1]}
    identity_rows = []
    for identity in identities:
        if set(identity) != required_identity_fields:
            raise ContractError("schedule identity fields do not match the closed schema")
        identity_rows.append(dict(identity))
    if not identity_rows:
        raise ContractError("schedule must contain at least one identity")
    if len({(row["case_id"], row["method_id"]) for row in identity_rows}) != len(identity_rows):
        raise ContractError("schedule identities must be unique")
    if any(row["method_id"] not in METHOD_IDS for row in identity_rows):
        raise ContractError("schedule contains unknown method")
    methods_by_case: dict[str, set[str]] = {}
    for row in identity_rows:
        methods_by_case.setdefault(row["case_id"], set()).add(row["method_id"])
    primary_set = set(PRIMARY_METHOD_IDS)
    primary_pair_complete = all(
        primary_set.issubset(methods) for methods in methods_by_case.values()
    )
    comparison_mode = "primary_pair" if primary_pair_complete else "method_local_only"
    checks = list(mandatory_checks)
    if len(checks) != len(set(checks)) or not set(checks).issubset(AGGREGATE_CHECKS):
        raise ContractError("invalid or duplicate aggregate checks")
    if harness_contract_test_only:
        if primary_pair_complete:
            raise ContractError(
                "harness-only schedules must be method-local and cannot claim a primary pair"
            )
        required = HARNESS_ONLY_CHECKS
    elif primary_pair_complete:
        required = PRIMARY_PAIR_CHECKS
    else:
        required = METHOD_LOCAL_CHECKS
    if tuple(checks) != tuple(required):
        raise ContractError(f"aggregate checks must equal {list(required)}")
    schedule_identity = {
        "schema": SCHEMA,
        "method_contract_version": METHOD_CONTRACT_VERSION,
        "comparison_mode": comparison_mode,
        "primary_pair_complete": primary_pair_complete,
        "comparator_parity_applicable": primary_pair_complete,
        "comparator_parity_reason": (
            None if primary_pair_complete else "primary_method_pair_incomplete"
        ),
        "expected_identities": identity_rows,
        "mandatory_aggregate_checks": checks,
        "harness_contract_test_only": harness_contract_test_only,
    }
    schedule_fingerprint = canonical_sha256(schedule_identity)
    return {**schedule_identity, "schedule_fingerprint": schedule_fingerprint}


def resume_key(
    *,
    case_identity: str,
    method_id: str,
    fingerprints: Mapping[str, str],
) -> str:
    if set(fingerprints) != set(FINGERPRINT_FIELDS):
        raise ContractError("resume key requires all five fingerprints")
    return canonical_sha256(
        {
            "schema": SCHEMA,
            "case_id": case_identity,
            "method_id": method_id,
            "fingerprints": dict(fingerprints),
        }
    )


def new_attempt(progress_dir: Path, case_identity: str, method_id: str) -> tuple[str, Path]:
    progress_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = secrets.token_hex(16)
    safe_case = hashlib.sha256(case_identity.encode("utf-8")).hexdigest()[:16]
    path = progress_dir / f"{safe_case}-{method_id}-{attempt_id}.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return attempt_id, path


def append_progress_event(
    path: Path,
    event: Mapping[str, Any],
    *,
    allowed_stages: Sequence[str] = STAGES,
) -> None:
    required = {
        "attempt_id",
        "case_id",
        "method_id",
        "stage",
        "resume_key",
        *FINGERPRINT_FIELDS,
    }
    if set(event) != required:
        raise ContractError("progress event fields do not match the closed schema")
    if (
        event["method_id"] not in METHOD_IDS
        or tuple(allowed_stages) not in {STAGES, PHASE6_TRACE_STAGES}
        or event["stage"] not in allowed_stages
    ):
        raise ContractError("progress event contains invalid method or stage")
    line = strict_json_dumps(dict(event)) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def recover_last_stage(
    path: Path,
    expected: Mapping[str, Any],
    *,
    allowed_stages: Sequence[str] = STAGES,
) -> str | None:
    expected_keys = {
        "attempt_id",
        "case_id",
        "method_id",
        "resume_key",
        *FINGERPRINT_FIELDS,
    }
    if set(expected) != expected_keys:
        raise ContractError("expected progress identity is incomplete")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read progress journal {path}: {exc}") from exc
    if tuple(allowed_stages) not in {STAGES, PHASE6_TRACE_STAGES}:
        raise ContractError("recovery stage set is not a closed contract")
    last_stage: str | None = None
    for raw_line in content.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            break
        try:
            event = strict_json_loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, ContractError):
            break
        if any(event.get(key) != value for key, value in expected.items()):
            continue
        stage = event.get("stage")
        if stage not in allowed_stages:
            continue
        last_stage = stage
    return last_stage


def phase6_progress_journal_valid(
    blob: Mapping[str, Any],
    *,
    schedule_row: Mapping[str, Any],
    trace: bool,
    expected_last_stage: str | None = None,
    expected_attempt_id: str | None = None,
) -> bool:
    if not phase6_blob_record_valid(blob) or blob.get("present") is not True:
        return False
    encoded = blob.get("base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError):
        return False
    if not raw or not raw.endswith(b"\n"):
        return False
    allowed = PHASE6_TRACE_STAGES if trace else STAGES
    events: list[Mapping[str, Any]] = []
    expected_identity = {
        "case_id": schedule_row.get("case_id"),
        "method_id": schedule_row.get("identity", {}).get("method_id"),
        "resume_key": schedule_row.get("resume_key"),
        **schedule_row.get("fingerprints", {}),
    }
    for line in raw.splitlines():
        try:
            event = strict_json_loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ContractError):
            return False
        if (
            not isinstance(event, Mapping)
            or set(event)
            != {"attempt_id", "case_id", "method_id", "stage", "resume_key", *FINGERPRINT_FIELDS}
            or not isinstance(event.get("attempt_id"), str)
            or not event["attempt_id"]
            or any(event.get(key) != value for key, value in expected_identity.items())
            or event.get("stage") not in allowed
        ):
            return False
        events.append(event)
    stage_indexes = [allowed.index(event["stage"]) for event in events]
    attempt_ids = {event["attempt_id"] for event in events}
    if stage_indexes[-1] == len(allowed) - 1:
        operational_indexes = stage_indexes[:-1]
        ordered_stages = (
            bool(operational_indexes)
            and operational_indexes
            == list(range(operational_indexes[-1] + 1))
        )
    else:
        ordered_stages = stage_indexes == list(range(stage_indexes[-1] + 1))
    return (
        ordered_stages
        and len(attempt_ids) == 1
        and (expected_attempt_id is None or attempt_ids == {expected_attempt_id})
        and (expected_last_stage is None or events[-1]["stage"] == expected_last_stage)
    )


def method_record_reuse_decision(
    record: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_method_id: str,
    expected_fingerprints: Mapping[str, str],
    expected_resume_key: str,
) -> tuple[bool, str]:
    if record.get("schema") != SCHEMA:
        return False, "schema_mismatch"
    if record.get("case_id") != expected_case_id:
        return False, "case_id_mismatch"
    if record.get("method_id") != expected_method_id:
        return False, "method_id_mismatch"
    for field in FINGERPRINT_FIELDS:
        if record.get(field) != expected_fingerprints.get(field):
            return False, f"{field}_mismatch"
    if record.get("method_contract_version") != METHOD_CONTRACT_VERSION:
        return False, "method_contract_version_mismatch"
    if record.get("resume_key") != expected_resume_key:
        return False, "resume_key_mismatch"
    if record.get("state") != "passed":
        return False, "state_not_passed"
    if not record.get("attempt_id"):
        return False, "attempt_id_missing"
    if record.get("last_entered_stage") != "envelope_write":
        return False, "last_entered_stage_invalid"
    if record.get("terminal_stage") != "envelope_write":
        return False, "terminal_stage_invalid"
    if record.get("failure_stage") is not None:
        return False, "passed_record_has_failure_stage"
    if record.get("invoked_method_ids") != [expected_method_id]:
        return False, "invocation_ledger_mismatch"
    if not measurement_record_is_valid(record):
        return False, "measurement_contract_invalid"
    output = record.get("output_metadata")
    if not isinstance(output, Mapping) or output.get("all_finite") is not True:
        return False, "finite_output_metadata_missing"
    if output.get("value_shape") is None or output.get("score_shape") is None:
        return False, "output_shape_metadata_missing"
    if not output.get("value_dtype") or not output.get("score_dtype"):
        return False, "output_dtype_metadata_missing"
    return True, "reusable_exact_match"


def classify_top_level_status(
    expected_identities: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
    aggregate_checks: Mapping[str, bool],
    mandatory_checks: Sequence[str],
    *,
    interrupted: bool = False,
    structural_failure: bool = False,
) -> str:
    if interrupted:
        return "interrupted"
    if structural_failure:
        return "failed"
    expected = [(row["case_id"], row["method_id"]) for row in expected_identities]
    observed = [(row.get("case_id"), row.get("method_id")) for row in records]
    if len(observed) != len(set(observed)) or sorted(observed) != sorted(expected):
        return "failed"
    if any(record.get("state") not in METHOD_TERMINAL_STATES for record in records):
        return "failed"
    if (
        tuple(aggregate_checks) != tuple(mandatory_checks)
        or any(type(value) is not bool for value in aggregate_checks.values())
    ):
        return "failed"
    if all(record["state"] == "passed" for record in records) and all(aggregate_checks.values()):
        return "complete"
    return "complete_with_failures"


def exit_code_for_status(status: str) -> int:
    if status not in TOP_LEVEL_STATES:
        raise ContractError(f"unknown top-level status {status!r}")
    return 0 if status == "complete" else 1


def synthesize_process_record(
    *,
    identity: Mapping[str, Any],
    progress_path: Path,
    timed_out: bool,
    returncode: int | None,
    error_tail: str,
) -> dict[str, Any]:
    last_stage = recover_last_stage(progress_path, identity)
    if timed_out:
        state = "timed_out"
    elif returncode is not None and returncode < 0:
        state = "crashed"
    else:
        state = "failed"
    return {
        "schema": SCHEMA,
        "method_contract_version": METHOD_CONTRACT_VERSION,
        "case_id": identity["case_id"],
        "method_id": identity["method_id"],
        **{field: identity[field] for field in FINGERPRINT_FIELDS},
        "resume_key": identity["resume_key"],
        "attempt_id": identity["attempt_id"],
        "state": state,
        "last_entered_stage": last_stage,
        "terminal_stage": last_stage,
        "failure_stage": last_stage,
        "returncode": returncode,
        "timed_out": timed_out,
        "error_tail": error_tail[-4000:],
        "invoked_method_ids": [],
        "measurement": None,
        "output_metadata": None,
    }


def durable_atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a Phase 6 artifact and durably sync its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    directory_fd: int | None = None
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def durable_atomic_write_json(path: Path, value: Any) -> None:
    durable_atomic_write_text(path, strict_json_dumps(value, indent=2) + "\n")


def phase6_identity(
    *,
    dimension: int,
    parameter_count: int,
    batch_size: int,
    dtype: str,
    method_id: str,
    operation: str,
) -> dict[str, Any]:
    if (
        type(dimension) is not int
        or type(parameter_count) is not int
        or type(batch_size) is not int
        or min(dimension, parameter_count, batch_size) <= 0
        or dtype not in DIRECT_PARITY_TOLERANCES
        or method_id not in METHOD_IDS
        or not isinstance(operation, str)
        or not operation
    ):
        raise ContractError("invalid Phase 6 identity")
    identity_id = (
        f"operation={operation}/dimension={dimension}/parameter_count={parameter_count}/"
        f"batch_size={batch_size}/dtype={dtype}/method={method_id}"
    )
    return {
        "identity_id": identity_id,
        "dimension": dimension,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "dtype": dtype,
        "method_id": method_id,
        "operation": operation,
    }


def _phase6_identity_valid(identity: Any) -> bool:
    if not isinstance(identity, Mapping) or set(identity) != set(PHASE6_IDENTITY_FIELDS):
        return False
    try:
        return dict(identity) == phase6_identity(
            dimension=identity["dimension"],
            parameter_count=identity["parameter_count"],
            batch_size=identity["batch_size"],
            dtype=identity["dtype"],
            method_id=identity["method_id"],
            operation=identity["operation"],
        )
    except (KeyError, ContractError, TypeError):
        return False


def phase6_expected_roster(schema: str) -> list[dict[str, Any]]:
    if schema not in PHASE6_SCHEMA_CONTRACTS:
        raise ContractError(f"unknown Phase 6 ledger schema {schema!r}")
    operation = PHASE6_SCHEMA_CONTRACTS[schema][2]
    if schema == PHASE6_TRACE_SCHEMA:
        cells = (
            (dimension, parameter_count, batch_size, method)
            for dimension in (10, 20, 30)
            for parameter_count in (50, 150)
            for batch_size in (1, 4, 16)
            for method in PRIMARY_METHOD_IDS
        )
    elif schema == PHASE6_PILOT_SCHEMA:
        cells = ((10, 50, 1, method) for method in PRIMARY_METHOD_IDS)
    elif schema == PHASE6_SCALAR_SCHEMA:
        cells = (
            (10, 50, batch_size, method)
            for batch_size in (1, 4)
            for method in REFERENCE_METHOD_IDS
        )
    elif schema == PHASE6_FINAL_SCHEMA:
        cells = (
            (dimension, parameter_count, batch_size, method)
            for dimension in (10, 20, 30)
            for parameter_count in (50, 150)
            for batch_size in (1, 4, 16)
            for method in PRIMARY_METHOD_IDS
        )
    else:
        cells = (
            (dimension, 150, batch_size, method)
            for dimension in (10, 20, 30)
            for batch_size in (1, 4, 16)
            for method in PRIMARY_METHOD_IDS
        )
    return [
        phase6_identity(
            dimension=dimension,
            parameter_count=parameter_count,
            batch_size=batch_size,
            dtype="float32",
            method_id=method,
            operation=operation,
        )
        for dimension, parameter_count, batch_size, method in cells
    ]


def _phase6_bindings_valid(bindings: Any) -> bool:
    if not (
        isinstance(bindings, Mapping)
        and set(bindings) == set(PHASE6_BINDING_FIELDS)
        and isinstance(bindings.get("authority_id"), str)
        and re.fullmatch(r"[0-9a-f]{64}", bindings["authority_id"]) is not None
        and phase6_blob_record_valid(bindings.get("proposal"))
        and phase6_blob_record_valid(bindings.get("attestation"))
        and bindings["proposal"].get("strict_json") is not None
        and bindings["attestation"].get("strict_json") is not None
        and isinstance(bindings.get("schedule"), Mapping)
        and set(bindings["schedule"]) == {"payload", "sha256"}
        and canonical_sha256(bindings["schedule"]["payload"])
        == bindings["schedule"]["sha256"]
        and isinstance(bindings["schedule"]["payload"], Mapping)
        and set(bindings["schedule"]["payload"])
        == {"schema", "ledger_schema", "gate", "records", "schedule_sha256"}
        and bindings["schedule"]["payload"].get("schema") == PHASE6_SCHEDULE_SCHEMA
        and canonical_sha256(
            {
                "schema": bindings["schedule"]["payload"]["schema"],
                "ledger_schema": bindings["schedule"]["payload"]["ledger_schema"],
                "gate": bindings["schedule"]["payload"]["gate"],
                "records": bindings["schedule"]["payload"]["records"],
            }
        )
        == bindings["schedule"]["payload"]["schedule_sha256"]
        and isinstance(bindings["schedule"]["payload"]["records"], list)
        and all(
            isinstance(record, Mapping)
            and set(record) == set(PHASE6_SCHEDULE_RECORD_FIELDS)
            and _phase6_identity_valid(record.get("identity"))
            and isinstance(record.get("case_id"), str)
            and isinstance(record.get("config"), Mapping)
            and isinstance(record.get("fingerprints"), Mapping)
            and set(record["fingerprints"]) == set(FINGERPRINT_FIELDS)
            and isinstance(record.get("resume_key"), str)
            and isinstance(record.get("child_command_argv"), list)
            and bool(record["child_command_argv"])
            and all(isinstance(value, str) for value in record["child_command_argv"])
            for record in bindings["schedule"]["payload"]["records"]
        )
        and isinstance(bindings.get("phase45_evidence"), list)
        and bool(bindings["phase45_evidence"])
        and all(
            phase6_blob_record_valid(record)
            and record.get("present") is True
            and record.get("strict_json") is not None
            for record in bindings["phase45_evidence"]
        )
        and isinstance(bindings.get("authority_inputs"), list)
        and all(
            phase6_blob_record_valid(record)
            and record.get("present") is True
            for record in bindings["authority_inputs"]
        )
        and isinstance(bindings.get("runtime_predecessors"), list)
    ):
        return False
    proposal = bindings["proposal"]["strict_json"]
    attestation = bindings["attestation"]["strict_json"]
    schedule_payload = bindings["schedule"]["payload"]
    proposal_schedules = proposal.get("schedules") if isinstance(proposal, Mapping) else None
    expected_schedule_schemas = (
        {PHASE6_TRACE_SCHEMA, PHASE6_PILOT_SCHEMA}
        if proposal.get("gate") == "gate_b"
        else {PHASE6_SCALAR_SCHEMA, PHASE6_FINAL_SCHEMA}
        if proposal.get("gate") == "gate_c"
        else set()
    )
    proposal_inputs = proposal.get("inputs") if isinstance(proposal, Mapping) else None
    authority_inputs = bindings["authority_inputs"]
    authority_inputs_valid = (
        isinstance(proposal_inputs, list)
        and [
            {"path": blob.get("path"), "sha256": blob.get("sha256")}
            for blob in authority_inputs
        ]
        == proposal_inputs
    )
    ledger_schema = schedule_payload.get("ledger_schema")
    expected_runtime_schemas = {
        PHASE6_TRACE_SCHEMA: [],
        PHASE6_PILOT_SCHEMA: [PHASE6_TRACE_SCHEMA],
        PHASE6_SCALAR_SCHEMA: [],
        PHASE6_FINAL_SCHEMA: [PHASE6_SCALAR_SCHEMA],
        PHASE6_ROUTING_SCHEMA: [PHASE6_SCALAR_SCHEMA],
    }.get(ledger_schema)
    runtime_predecessors = bindings["runtime_predecessors"]
    runtime_predecessors_valid = (
        expected_runtime_schemas is not None
        and [
            predecessor.get("producer_ledger_schema")
            for predecessor in runtime_predecessors
            if isinstance(predecessor, Mapping)
        ]
        == expected_runtime_schemas
        and len(runtime_predecessors) == len(expected_runtime_schemas)
        and all(
            phase6_runtime_predecessor_valid(
                predecessor,
                authority_id=bindings["authority_id"],
                proposal=proposal,
            )
            for predecessor in runtime_predecessors
        )
    )
    return (
        isinstance(proposal, Mapping)
        and isinstance(attestation, Mapping)
        and bindings["authority_id"] == proposal.get("authority_id")
        and bindings["authority_id"] == attestation.get("authority_id")
        and bindings["schedule"]["payload"]["gate"] == proposal.get("gate")
        and bindings["schedule"]["payload"]["gate"] == attestation.get("gate")
        and isinstance(proposal_schedules, Mapping)
        and set(proposal_schedules) == expected_schedule_schemas
        and all(
            isinstance(reviewed_schedule, Mapping)
            and reviewed_schema == reviewed_schedule.get("ledger_schema")
            and all(phase6_schedule_checks(reviewed_schedule).values())
            for reviewed_schema, reviewed_schedule in proposal_schedules.items()
        )
        and schedule_payload.get("ledger_schema") in proposal_schedules
        and proposal_schedules[schedule_payload["ledger_schema"]] == schedule_payload
        and authority_inputs_valid
        and runtime_predecessors_valid
    )


def new_phase6_ledger(
    *,
    schema: str,
    gate: str,
    artifact_kind: str,
    identities: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if schema not in PHASE6_SCHEMAS or not gate or not artifact_kind:
        raise ContractError("invalid Phase 6 ledger identity")
    roster = [dict(identity) for identity in identities]
    expected_gate, expected_kind, _ = PHASE6_SCHEMA_CONTRACTS[schema]
    if gate != expected_gate or artifact_kind != expected_kind:
        raise ContractError("Phase 6 ledger gate/artifact does not match schema")
    if roster != phase6_expected_roster(schema):
        raise ContractError("Phase 6 roster does not match exact schema roster")
    if not _phase6_bindings_valid(bindings):
        raise ContractError("Phase 6 bindings do not match closed schema")
    if bindings["schedule"]["payload"]["gate"] != gate:
        raise ContractError("Phase 6 bindings do not match ledger gate")
    if bindings["schedule"]["payload"]["ledger_schema"] != schema:
        raise ContractError("Phase 6 bindings do not match ledger schema")
    schedule_identities = [
        row["identity"] for row in bindings["schedule"]["payload"]["records"]
    ]
    if schedule_identities != roster:
        raise ContractError("Phase 6 binding schedule does not match exact ledger roster")
    identity_ids = [identity["identity_id"] for identity in roster]
    if len(identity_ids) != len(set(identity_ids)):
        raise ContractError("duplicate Phase 6 roster identity")
    records = [
        {
            "identity": identity,
            "state": "pending",
            "reason": None,
            "process": None,
            "evidence": None,
            "imported_from": None,
        }
        for identity in roster
    ]
    payload = {
        "schema": schema,
        "gate": gate,
        "artifact_kind": artifact_kind,
        "state": "running",
        "bindings": copy.deepcopy(dict(bindings)),
        "roster": roster,
        "records": records,
        "events": [],
        "update_index": 0,
        "aggregate": {},
        "nonclaims": list(PHASE6_NONCLAIMS),
    }
    if not all(phase6_ledger_checks(payload, final=False).values()):
        raise ContractError("constructed Phase 6 ledger failed its contract")
    return payload


def _phase6_record_event_digest(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "process": record.get("process"),
            "evidence": record.get("evidence"),
            "imported_from": record.get("imported_from"),
        }
    )


def phase6_blob_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "present": False,
            "byte_count": 0,
            "sha256": None,
            "base64": None,
            "strict_json": None,
        }
    if path.is_symlink() or not resolved.is_file():
        raise ContractError(f"Phase 6 evidence blob is not a regular file: {path}")
    raw = resolved.read_bytes()
    decoded: Any = None
    try:
        decoded = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ContractError):
        pass
    return {
        "path": str(resolved),
        "present": True,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
        "strict_json": decoded,
    }


def phase6_blob_record_valid(blob: Any) -> bool:
    if not isinstance(blob, Mapping) or set(blob) != set(PHASE6_BLOB_FIELDS):
        return False
    if not isinstance(blob.get("path"), str) or not blob["path"]:
        return False
    if blob.get("present") is False:
        return (
            blob.get("byte_count") == 0
            and blob.get("sha256") is None
            and blob.get("base64") is None
            and blob.get("strict_json") is None
        )
    encoded = blob.get("base64")
    if (
        blob.get("present") is not True
        or type(blob.get("byte_count")) is not int
        or blob["byte_count"] < 0
        or not isinstance(encoded, str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(blob.get("sha256")))
    ):
        return False
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return False
    if (
        len(raw) != blob["byte_count"]
        or hashlib.sha256(raw).hexdigest() != blob["sha256"]
        or base64.b64encode(raw).decode("ascii") != encoded
    ):
        return False
    if blob.get("strict_json") is not None:
        try:
            return strict_json_loads(raw.decode("utf-8")) == blob["strict_json"]
        except (UnicodeDecodeError, ContractError):
            return False
    return True


def phase6_child_authority_snapshot(
    bindings: Mapping[str, Any], schedule_row: Mapping[str, Any]
) -> dict[str, Any]:
    if not _phase6_bindings_valid(bindings):
        raise ContractError("cannot snapshot invalid Phase 6 bindings")
    records = bindings["schedule"]["payload"]["records"]
    if not isinstance(schedule_row, Mapping) or list(records).count(schedule_row) != 1:
        raise ContractError("child authority snapshot requires one reviewed schedule row")
    core = {
        "schema": PHASE6_CHILD_AUTHORITY_SNAPSHOT_SCHEMA,
        "bindings": copy.deepcopy(dict(bindings)),
        "schedule_row": copy.deepcopy(dict(schedule_row)),
    }
    return {**core, "payload_sha256": canonical_sha256(core)}


def phase6_child_authority_snapshot_valid(snapshot: Any) -> bool:
    if (
        not isinstance(snapshot, Mapping)
        or set(snapshot) != set(PHASE6_CHILD_AUTHORITY_SNAPSHOT_FIELDS)
        or snapshot.get("schema") != PHASE6_CHILD_AUTHORITY_SNAPSHOT_SCHEMA
        or not isinstance(snapshot.get("bindings"), Mapping)
        or not isinstance(snapshot.get("schedule_row"), Mapping)
    ):
        return False
    core = {
        "schema": snapshot["schema"],
        "bindings": snapshot["bindings"],
        "schedule_row": snapshot["schedule_row"],
    }
    bindings = snapshot["bindings"]
    return (
        snapshot.get("payload_sha256") == canonical_sha256(core)
        and _phase6_bindings_valid(bindings)
        and list(bindings["schedule"]["payload"]["records"]).count(
            snapshot["schedule_row"]
        )
        == 1
    )


def validate_phase6_child_authority_snapshot(
    snapshot: Mapping[str, Any],
    *,
    repo_root: Path,
    command_argv: Sequence[str],
) -> None:
    """Revalidate one captured launch authority before importing TensorFlow."""

    if not phase6_child_authority_snapshot_valid(snapshot):
        raise ContractError("invalid Phase 6 child authority snapshot")
    if not isinstance(command_argv, Sequence) or isinstance(command_argv, (str, bytes)):
        raise ContractError("invalid Phase 6 child command argv")
    exact_argv = list(command_argv)
    if not exact_argv or any(not isinstance(value, str) for value in exact_argv):
        raise ContractError("invalid Phase 6 child command argv")
    bindings = snapshot["bindings"]
    schedule_row = snapshot["schedule_row"]
    if schedule_row.get("child_command_argv") != exact_argv:
        raise ContractError("Phase 6 child argv differs from its reviewed schedule row")
    options = _phase6_command_options(exact_argv)
    identity = schedule_row.get("identity")
    if not isinstance(options, Mapping) or not isinstance(identity, Mapping):
        raise ContractError("Phase 6 child argv cannot be parsed exactly")
    expected_snapshot_path = str(
        _phase6_child_artifact_paths(identity)["authority_snapshot"]
    )
    if options.get("--phase6-authority-snapshot") != [expected_snapshot_path]:
        raise ContractError("Phase 6 child snapshot path differs from reviewed argv")

    proposal_blob = bindings["proposal"]
    attestation_blob = bindings["attestation"]
    if (
        phase6_blob_record(Path(proposal_blob["path"])) != proposal_blob
        or phase6_blob_record(Path(attestation_blob["path"])) != attestation_blob
    ):
        raise ContractError("Phase 6 proposal or attestation changed before child entry")
    proposal, attestation = validate_phase6_runtime_authority(
        Path(proposal_blob["path"]),
        Path(attestation_blob["path"]),
        expected_gate=bindings["schedule"]["payload"]["gate"],
    )
    if (
        proposal != proposal_blob["strict_json"]
        or attestation != attestation_blob["strict_json"]
    ):
        raise ContractError("Phase 6 runtime authority bytes changed before child entry")
    for field in ("phase45_evidence", "authority_inputs"):
        for blob in bindings[field]:
            if phase6_blob_record(Path(blob["path"])) != blob:
                raise ContractError(f"Phase 6 {field} changed before child entry")
    for predecessor in bindings["runtime_predecessors"]:
        artifact = predecessor["artifact"]
        if phase6_blob_record(Path(artifact["path"])) != artifact:
            raise ContractError("Phase 6 runtime predecessor changed before child entry")

    schedule = bindings["schedule"]["payload"]
    current_source = source_manifest(repo_root, include_supervisor=True)[
        "source_fingerprint"
    ]
    current_runtime = runtime_manifest()["runtime_fingerprint"]
    if (
        not all(phase6_schedule_checks(schedule).values())
        or any(
            row["fingerprints"]["source_fingerprint"] != current_source
            or row["fingerprints"]["runtime_fingerprint"] != current_runtime
            for row in schedule["records"]
        )
    ):
        raise ContractError("Phase 6 source/runtime/schedule drift before child entry")


def phase6_runtime_predecessor_record(path: Path) -> dict[str, Any]:
    artifact = phase6_blob_record(path)
    payload = artifact.get("strict_json")
    if not isinstance(payload, Mapping):
        raise ContractError("Phase 6 runtime predecessor is not strict JSON")
    bindings = payload.get("bindings")
    schedule = bindings.get("schedule") if isinstance(bindings, Mapping) else None
    if not isinstance(schedule, Mapping):
        raise ContractError("Phase 6 runtime predecessor lacks schedule bindings")
    return {
        "kind": "same_authority_runtime_predecessor",
        "producer_ledger_schema": payload.get("schema"),
        "authority_id": bindings.get("authority_id"),
        "schedule_sha256": schedule.get("payload", {}).get("schedule_sha256"),
        "artifact": artifact,
    }


def phase6_runtime_predecessor_valid(
    predecessor: Any,
    *,
    authority_id: str,
    proposal: Mapping[str, Any],
) -> bool:
    if (
        not isinstance(predecessor, Mapping)
        or set(predecessor) != set(PHASE6_RUNTIME_PREDECESSOR_FIELDS)
        or predecessor.get("kind") != "same_authority_runtime_predecessor"
        or predecessor.get("authority_id") != authority_id
        or not phase6_blob_record_valid(predecessor.get("artifact"))
    ):
        return False
    artifact = predecessor["artifact"]
    payload = artifact.get("strict_json")
    producer_schema = predecessor.get("producer_ledger_schema")
    schedules = proposal.get("schedules")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != producer_schema
        or producer_schema not in {PHASE6_TRACE_SCHEMA, PHASE6_SCALAR_SCHEMA}
        or not isinstance(schedules, Mapping)
        or producer_schema not in schedules
        or predecessor.get("schedule_sha256")
        != schedules[producer_schema].get("schedule_sha256")
        or payload.get("bindings", {}).get("authority_id") != authority_id
        or payload.get("bindings", {}).get("schedule", {}).get("payload")
        != schedules[producer_schema]
    ):
        return False
    artifact_key = (
        "trace_output_json" if producer_schema == PHASE6_TRACE_SCHEMA else "scalar_output_json"
    )
    declared_path = proposal.get("artifacts", {}).get(artifact_key)
    try:
        expected_path = str((Path(__file__).resolve().parents[1] / declared_path).resolve())
    except (TypeError, OSError):
        return False
    return (
        artifact.get("path") == expected_path
        and all(phase6_ledger_checks(payload, final=True).values())
    )


def phase6_process_record_valid(
    process: Any,
    *,
    terminal_state: str | None = None,
    terminal_reason: str | None = None,
) -> bool:
    if not isinstance(process, Mapping) or set(process) != set(PHASE6_PROCESS_FIELDS):
        return False
    stdout_tail = process.get("stdout_tail")
    stderr_tail = process.get("stderr_tail")
    stdout_encoded = process.get("stdout_base64")
    stderr_encoded = process.get("stderr_base64")
    try:
        stdout_raw = base64.b64decode(stdout_encoded, validate=True)
        stderr_raw = base64.b64decode(stderr_encoded, validate=True)
    except (binascii.Error, TypeError):
        stdout_raw = None
        stderr_raw = None
    common = (
        isinstance(process.get("command_argv"), list)
        and bool(process["command_argv"])
        and all(isinstance(value, str) for value in process["command_argv"])
        and isinstance(process.get("cwd"), str)
        and isinstance(process.get("environment"), Mapping)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in process["environment"].items())
        and type(process.get("pid")) is int
        and process["pid"] > 0
        and type(process.get("pgid")) is int
        and process["pgid"] > 0
        and type(process.get("process_start_ticks")) is int
        and process["process_start_ticks"] > 0
        and type(process.get("started_ns")) is int
        and type(process.get("finished_ns")) is int
        and process["finished_ns"] >= process["started_ns"] >= 0
        and _is_finite_nonnegative_number(process.get("elapsed_seconds"))
        and _is_finite_nonnegative_number(process.get("deadline_seconds"))
        and all(
            type(process.get(field)) is bool
            for field in ("term_sent", "kill_sent", "reaped", "process_group_gone", "timed_out")
        )
        and process.get("reap_status") in {"reaped_direct_child", "already_gone_not_waitable_after_recovery"}
        and process["reaped"] is (process["reap_status"] == "reaped_direct_child")
        and process["process_group_gone"]
        and (process.get("returncode") is None or type(process["returncode"]) is int)
        and type(process.get("stdout_bytes")) is int
        and process["stdout_bytes"] >= 0
        and (process.get("stdout_total_bytes") is None or type(process["stdout_total_bytes"]) is int)
        and process.get("stdout_capture_status")
        in {"complete", "truncated_at_cap", "unavailable_after_recovery"}
        and type(process.get("stderr_bytes")) is int
        and process["stderr_bytes"] >= 0
        and (process.get("stderr_total_bytes") is None or type(process["stderr_total_bytes"]) is int)
        and process.get("stderr_capture_status")
        in {"complete", "truncated_at_cap", "unavailable_after_recovery"}
        and re.fullmatch(r"[0-9a-f]{64}", str(process.get("stdout_sha256"))) is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(process.get("stderr_sha256"))) is not None
        and stdout_raw is not None
        and stderr_raw is not None
        and len(stdout_raw) == process["stdout_bytes"]
        and len(stderr_raw) == process["stderr_bytes"]
        and hashlib.sha256(stdout_raw).hexdigest() == process["stdout_sha256"]
        and hashlib.sha256(stderr_raw).hexdigest() == process["stderr_sha256"]
        and base64.b64encode(stdout_raw).decode("ascii") == stdout_encoded
        and base64.b64encode(stderr_raw).decode("ascii") == stderr_encoded
        and (
            process["stdout_capture_status"] == "unavailable_after_recovery"
            and process["stdout_total_bytes"] is None
            or process["stdout_capture_status"] != "unavailable_after_recovery"
            and type(process["stdout_total_bytes"]) is int
            and process["stdout_total_bytes"] >= process["stdout_bytes"]
        )
        and (
            process["stderr_capture_status"] == "unavailable_after_recovery"
            and process["stderr_total_bytes"] is None
            or process["stderr_capture_status"] != "unavailable_after_recovery"
            and type(process["stderr_total_bytes"]) is int
            and process["stderr_total_bytes"] >= process["stderr_bytes"]
        )
        and (
            process["stdout_capture_status"] == "complete"
        )
        is (process["stdout_total_bytes"] == process["stdout_bytes"])
        and (
            process["stderr_capture_status"] == "complete"
        )
        is (process["stderr_total_bytes"] == process["stderr_bytes"])
        and isinstance(stdout_tail, str)
        and isinstance(stderr_tail, str)
        and stdout_tail == stdout_raw.decode("utf-8", errors="replace")[-4000:]
        and stderr_tail == stderr_raw.decode("utf-8", errors="replace")[-4000:]
        and len(stdout_tail) <= 4000
        and len(stderr_tail) <= 4000
    )
    if not common or terminal_state is None:
        return common
    if terminal_state == "passed":
        return (
            process["returncode"] == 0
            and process["timed_out"] is False
            and process["stdout_capture_status"] == "complete"
            and process["stderr_capture_status"] == "complete"
        )
    if terminal_state == "timed_out":
        return process["timed_out"] is True and process["returncode"] is not None
    if terminal_state == "crashed":
        return process["timed_out"] is False and process["returncode"] is not None and process["returncode"] < 0
    if terminal_state == "failed":
        return process["timed_out"] is False and (
            process["returncode"] == 0
            if terminal_reason == "invalid_child_evidence"
            else process["returncode"] is not None and process["returncode"] >= 0
            if terminal_reason == "authority_revalidation_failed"
            else process["returncode"] is not None and process["returncode"] > 0
        )
    if terminal_state == "interrupted":
        return process["timed_out"] is False and (
            process["returncode"] is not None
            or process["reap_status"]
            == "already_gone_not_waitable_after_recovery"
        )
    return False


def phase6_running_process_valid(process: Any) -> bool:
    return (
        isinstance(process, Mapping)
        and set(process) == set(PHASE6_RUNNING_PROCESS_FIELDS)
        and isinstance(process.get("command_argv"), list)
        and bool(process["command_argv"])
        and all(isinstance(value, str) for value in process["command_argv"])
        and isinstance(process.get("cwd"), str)
        and isinstance(process.get("environment"), Mapping)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in process["environment"].items())
        and type(process.get("pid")) is int
        and process["pid"] > 0
        and type(process.get("pgid")) is int
        and process["pgid"] > 0
        and type(process.get("process_start_ticks")) is int
        and process["process_start_ticks"] > 0
        and type(process.get("started_ns")) is int
        and process["started_ns"] >= 0
        and _is_finite_nonnegative_number(process.get("deadline_seconds"))
    )


def phase6_evidence_record_valid(evidence: Any, *, terminal_state: str | None = None) -> bool:
    if not isinstance(evidence, Mapping) or set(evidence) != set(PHASE6_EVIDENCE_FIELDS):
        return False
    if not isinstance(evidence.get("classification"), str) or not evidence["classification"]:
        return False
    blobs = [evidence.get(field) for field in ("child_artifact", "payload_sidecar", "progress_journal")]
    if not all(
        phase6_blob_record_valid(evidence.get(field))
        for field in ("child_artifact", "payload_sidecar", "progress_journal")
    ):
        return False
    for field in ("dependency_manifest_before_builder", "dependency_manifest_after_terminal"):
        manifest = evidence.get(field)
        if manifest is not None and (
            not isinstance(manifest, Mapping)
            or manifest.get("schema") != PHASE6_DEPENDENCY_SCHEMA
        ):
            return False
    coverage_valid = type(evidence.get("dependency_coverage_before")) is bool and type(
        evidence.get("dependency_coverage_after")
    ) is bool
    if terminal_state == "passed":
        coverage_valid = (
            coverage_valid
            and evidence["dependency_coverage_before"]
            and evidence["dependency_coverage_after"]
            and evidence["child_artifact"]["present"]
            and evidence["child_artifact"]["strict_json"] is not None
        )
    strict_json_valid = terminal_state == "interrupted" or all(
        blob["strict_json"] is not None
        if blob["present"] and blob["path"].endswith(".json")
        else True
        for blob in blobs
    )
    return coverage_valid and strict_json_valid


def _phase6_discovery_manifest(bindings: Mapping[str, Any]) -> Mapping[str, Any] | None:
    proposal = bindings.get("proposal", {}).get("strict_json")
    if not isinstance(proposal, Mapping):
        return None
    discovery = proposal.get("dependency_discovery")
    manifest = discovery.get("manifest") if isinstance(discovery, Mapping) else None
    return manifest if isinstance(manifest, Mapping) else None


def _phase6_trace_child_valid(
    child: Any,
    *,
    identity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    discovery: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
) -> bool:
    if (
        not isinstance(child, Mapping)
        or set(child)
        != {
            "schema",
            "state",
            "identity",
            "case_id",
            "attempt_id",
            *FINGERPRINT_FIELDS,
            "resume_key",
            "stage",
            "started_ns",
            "finished_ns",
            "elapsed_seconds",
            "command_argv",
            "dependency_manifest_before_builder",
            "dependency_manifest_after_terminal",
            "evidence",
            "error",
            "nonclaims",
        }
        or child.get("schema")
        != "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_child.v1"
        or child.get("state") != "passed"
        or child.get("identity") != identity
        or child.get("case_id") != schedule_row.get("case_id")
        or child.get("resume_key") != schedule_row.get("resume_key")
        or any(
            child.get(field) != schedule_row.get("fingerprints", {}).get(field)
            for field in FINGERPRINT_FIELDS
        )
        or not isinstance(child.get("attempt_id"), str)
        or not child["attempt_id"]
        or child.get("stage") != "terminal_provenance"
        or child.get("command_argv") != schedule_row.get("child_command_argv")
        or child.get("error") is not None
        or child.get("nonclaims") != list(PHASE6_NONCLAIMS)
    ):
        return False
    raw = child.get("evidence")
    if not isinstance(raw, Mapping):
        return False
    graph_record = raw.get("graphdef_bytes")
    try:
        graph_bytes = decode_graphdef_bytes_record(graph_record)
        tokens = graphdef_token_stream(graph_bytes)
    except (ContractError, TypeError):
        return False
    expected_outputs = [
        {"dtype": identity["dtype"], "shape": [identity["batch_size"]], "result_position": "value"},
        {
            "dtype": identity["dtype"],
            "shape": [identity["batch_size"], identity["parameter_count"]],
            "result_position": "score",
        },
    ]
    outputs = raw.get("concrete_outputs")
    outputs_valid = isinstance(outputs, list) and len(outputs) == 2 and all(
        isinstance(output, Mapping)
        and isinstance(output.get("name"), str)
        and output["name"].endswith(":0")
        and {
            "dtype": output.get("dtype"),
            "shape": output.get("shape"),
            "result_position": output.get("result_position"),
        }
        == expected
        for output, expected in zip(outputs, expected_outputs)
    )
    before = child.get("dependency_manifest_before_builder")
    after = child.get("dependency_manifest_after_terminal")
    return (
        raw.get("identity") == identity
        and raw.get("timesteps") == 120
        and raw.get("requested_device") == "cpu"
        and raw.get("cuda_visible_devices") == "-1"
        and raw.get("jit_compile") is False
        and raw.get("tf32_queried") is False
        and raw.get("device_enumeration_api_calls") == 0
        and raw.get("invoked_method_ids") == [identity["method_id"]]
        and raw.get("get_concrete_function_calls") == 1
        and raw.get("concrete_function_invocations") == 0
        and raw.get("structured_user_input")
        == {
            "name": "parameters_batch",
            "dtype": identity["dtype"],
            "shape": [identity["batch_size"], identity["parameter_count"]],
        }
        and outputs_valid
        and raw.get("typed_token_stream") == tokens
        and raw.get("typed_token_stream_sha256") == canonical_sha256(tokens)
        and type(raw.get("top_level_node_count")) is int
        and raw["top_level_node_count"] > 0
        and type(raw.get("function_count")) is int
        and raw["function_count"] >= 0
        and before == evidence.get("dependency_manifest_before_builder")
        and after == evidence.get("dependency_manifest_after_terminal")
        and dependency_manifest_covers(
            discovery, before, required_paths=PHASE6_REQUIRED_SOURCE_PATHS
        )
        and dependency_manifest_covers(
            discovery, after, required_paths=PHASE6_REQUIRED_SOURCE_PATHS
        )
        and phase6_progress_journal_valid(
            evidence.get("progress_journal", {}),
            schedule_row=schedule_row,
            trace=True,
            expected_last_stage="envelope_write",
            expected_attempt_id=child["attempt_id"],
        )
    )


def _phase6_v4_child_valid(
    child: Any,
    *,
    identity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    discovery: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
) -> bool:
    if not isinstance(child, Mapping) or child.get("schema") != SCHEMA:
        return False
    if (
        child.get("method_contract_version") != METHOD_CONTRACT_VERSION
        or child.get("method_id") != identity["method_id"]
        or not isinstance(child.get("attempt_id"), str)
        or not child["attempt_id"]
        or child.get("case_id") != schedule_row.get("case_id")
        or child.get("resume_key") != schedule_row.get("resume_key")
        or any(
            child.get(field) != schedule_row.get("fingerprints", {}).get(field)
            for field in FINGERPRINT_FIELDS
        )
        or child.get("state") != "passed"
        or child.get("returncode") != 0
        or child.get("timed_out") is not False
        or child.get("error") is not None
        or child.get("invoked_method_ids") != [identity["method_id"]]
        or child.get("last_entered_stage") != "envelope_write"
        or child.get("terminal_stage") != "envelope_write"
        or child.get("failure_stage") is not None
        or not measurement_record_is_valid(child)
    ):
        return False
    metadata = child.get("output_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("all_finite") is not True
        or metadata.get("value_shape") != [identity["batch_size"]]
        or metadata.get("score_shape")
        != [identity["batch_size"], identity["parameter_count"]]
        or metadata.get("value_dtype") != identity["dtype"]
        or metadata.get("score_dtype") != identity["dtype"]
    ):
        return False
    device = child.get("device_manifest")
    threads = child.get("cpu_thread_manifest")
    if (
        not isinstance(device, Mapping)
        or device.get("requested_device") != "cpu"
        or device.get("selected_device") != "/CPU:0"
        or device.get("physical_gpus") != []
        or device.get("logical_gpus") != []
        or not isinstance(threads, Mapping)
        or threads.get("requested_cpu_threads") != 1
        or threads.get("tf_intra_op_parallelism_threads") != 1
        or threads.get("tf_inter_op_parallelism_threads") != 1
    ):
        return False
    sidecar = evidence.get("payload_sidecar")
    expected_sidecar = {
        "case_id": child.get("case_id"),
        "method_id": child.get("method_id"),
        "output_metadata": child.get("output_metadata"),
        "outputs": child.get("outputs"),
        "graphdef": child.get("measurement", {}).get("graphdef"),
        "direct_output_parity": child.get("measurement", {}).get("direct_output_parity"),
    }
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("strict_json") != expected_sidecar
        or sidecar.get("sha256")
        != child.get("measurement", {}).get("payload_sidecar", {}).get("sha256")
    ):
        return False
    before = evidence.get("dependency_manifest_before_builder")
    after = evidence.get("dependency_manifest_after_terminal")
    return (
        dependency_manifest_covers(
            discovery, before, required_paths=PHASE6_REQUIRED_SOURCE_PATHS
        )
        and dependency_manifest_covers(
            discovery, after, required_paths=PHASE6_REQUIRED_SOURCE_PATHS
        )
        and phase6_progress_journal_valid(
            evidence.get("progress_journal", {}),
            schedule_row=schedule_row,
            trace=False,
            expected_last_stage="envelope_write",
            expected_attempt_id=child["attempt_id"],
        )
    )


def _phase6_failure_child_valid(
    record: Mapping[str, Any],
    *,
    schedule_row: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> bool:
    identity = record.get("identity")
    evidence = record.get("evidence")
    process = record.get("process")
    if not isinstance(identity, Mapping) or not isinstance(evidence, Mapping):
        return False
    dependency_consistent = all(
        evidence.get(coverage_field)
        == dependency_manifest_covers(
            discovery,
            evidence.get(manifest_field, {}),
            required_paths=PHASE6_REQUIRED_SOURCE_PATHS,
        )
        and (
            evidence.get(manifest_field) is None
            or evidence.get(coverage_field) is True
        )
        for manifest_field, coverage_field in (
            (
                "dependency_manifest_before_builder",
                "dependency_coverage_before",
            ),
            (
                "dependency_manifest_after_terminal",
                "dependency_coverage_after",
            ),
        )
    )
    if (
        not isinstance(process, Mapping)
        or process.get("stdout_capture_status") != "complete"
        or process.get("stderr_capture_status") != "complete"
        or not dependency_consistent
    ):
        return False
    child = evidence.get("child_artifact", {}).get("strict_json")
    if identity.get("operation") == "trace":
        if not isinstance(child, Mapping):
            return (
                record.get("state") in {"timed_out", "crashed"}
                and evidence.get("child_artifact", {}).get("present") is False
                and evidence.get("payload_sidecar", {}).get("present") is False
                and phase6_progress_journal_valid(
                    evidence.get("progress_journal", {}),
                    schedule_row=schedule_row,
                    trace=True,
                    expected_attempt_id=schedule_row.get("attempt_id"),
                )
            )
        exact_fields = {
            "schema",
            "state",
            "identity",
            "case_id",
            "attempt_id",
            *FINGERPRINT_FIELDS,
            "resume_key",
            "stage",
            "started_ns",
            "finished_ns",
            "elapsed_seconds",
            "command_argv",
            "dependency_manifest_before_builder",
            "dependency_manifest_after_terminal",
            "evidence",
            "error",
            "nonclaims",
        }
        return (
            set(child) == exact_fields
            and child.get("schema")
            == "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_child.v1"
            and child.get("state") == "failed"
            and child.get("identity") == identity
            and child.get("case_id") == schedule_row.get("case_id")
            and child.get("attempt_id") == schedule_row.get("attempt_id")
            and child.get("resume_key") == schedule_row.get("resume_key")
            and child.get("command_argv") == schedule_row.get("child_command_argv")
            and child.get("stage") in PHASE6_TRACE_STAGES[:-1]
            and isinstance(child.get("error"), Mapping)
            and child.get("evidence") is None
            and child.get("nonclaims") == list(PHASE6_NONCLAIMS)
            and all(
                child.get(field) == schedule_row.get("fingerprints", {}).get(field)
                for field in FINGERPRINT_FIELDS
            )
            and child.get("dependency_manifest_before_builder")
            == evidence.get("dependency_manifest_before_builder")
            and child.get("dependency_manifest_after_terminal")
            == evidence.get("dependency_manifest_after_terminal")
            and evidence.get("dependency_manifest_after_terminal") is not None
            and evidence.get("dependency_coverage_after") is True
            and (
                child.get("stage") == "fixture"
                or child.get("stage") == "pre_builder_provenance"
                or (
                    evidence.get("dependency_manifest_before_builder") is not None
                    and evidence.get("dependency_coverage_before") is True
                )
            )
            and phase6_progress_journal_valid(
                evidence.get("progress_journal", {}),
                schedule_row=schedule_row,
                trace=True,
                expected_last_stage="envelope_write",
                expected_attempt_id=schedule_row.get("attempt_id"),
            )
        )
    if not isinstance(child, Mapping):
        return (
            record.get("state") in {"timed_out", "crashed"}
            and evidence.get("child_artifact", {}).get("present") is False
            and evidence.get("payload_sidecar", {}).get("present") is False
            and phase6_progress_journal_valid(
                evidence.get("progress_journal", {}),
                schedule_row=schedule_row,
                trace=False,
                expected_attempt_id=schedule_row.get("attempt_id"),
            )
        )
    return (
        child.get("schema") == SCHEMA
        and child.get("method_contract_version") == METHOD_CONTRACT_VERSION
        and child.get("case_id") == schedule_row.get("case_id")
        and child.get("method_id") == identity.get("method_id")
        and child.get("attempt_id") == schedule_row.get("attempt_id")
        and child.get("resume_key") == schedule_row.get("resume_key")
        and child.get("state") == "failed"
        and child.get("returncode") == 1
        and child.get("timed_out") is False
        and child.get("last_entered_stage") == "envelope_write"
        and child.get("terminal_stage") == "envelope_write"
        and child.get("failure_stage") in STAGES[:-1]
        and isinstance(child.get("error"), Mapping)
        and child.get("measurement") is None
        and child.get("output_metadata") is None
        and child.get("outputs") is None
        and evidence.get("dependency_coverage_before") is True
        and evidence.get("dependency_coverage_after") is True
        and all(
            child.get(field) == schedule_row.get("fingerprints", {}).get(field)
            for field in FINGERPRINT_FIELDS
        )
        and phase6_progress_journal_valid(
            evidence.get("progress_journal", {}),
            schedule_row=schedule_row,
            trace=False,
            expected_last_stage="envelope_write",
            expected_attempt_id=schedule_row.get("attempt_id"),
        )
    )


def phase6_imported_pilot_record_valid(
    record: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
) -> bool:
    imported_from = record.get("imported_from")
    if (
        not isinstance(imported_from, Mapping)
        or set(imported_from) != set(PHASE6_IMPORTED_FROM_FIELDS)
        or imported_from.get("kind") != "gate_b_pilot"
        or re.fullmatch(
            r"[0-9a-f]{64}", str(imported_from.get("pilot_artifact_sha256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(imported_from.get("pilot_record_sha256"))
        )
        is None
        or not (
            record.get("state") in PHASE6_TERMINAL_STATES
            or isinstance(record.get("state"), str)
            and record["state"].startswith("not_launched:")
        )
    ):
        return False
    inputs = bindings.get("authority_inputs")
    pilot_blobs = [
        blob
        for blob in inputs
        if isinstance(blob, Mapping)
        and isinstance(blob.get("strict_json"), Mapping)
        and blob["strict_json"].get("schema") == PHASE6_PILOT_SCHEMA
    ] if isinstance(inputs, list) else []
    if len(pilot_blobs) != 1:
        return False
    pilot_blob = pilot_blobs[0]
    pilot = pilot_blob["strict_json"]
    if (
        not phase6_blob_record_valid(pilot_blob)
        or pilot_blob.get("present") is not True
        or pilot_blob.get("sha256") != imported_from["pilot_artifact_sha256"]
        or not all(phase6_ledger_checks(pilot, final=True).values())
    ):
        return False
    originals = [
        candidate
        for candidate in pilot.get("records", [])
        if isinstance(candidate, Mapping)
        and candidate.get("identity") == record.get("identity")
    ]
    if len(originals) != 1:
        return False
    original = originals[0]
    imported_copy = copy.deepcopy(dict(record))
    imported_copy["imported_from"] = None
    return (
        canonical_sha256(original) == imported_from["pilot_record_sha256"]
        and imported_copy == original
    )


def phase6_terminal_record_semantics_valid(
    record: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
) -> bool:
    state = record.get("state")
    if state not in PHASE6_TERMINAL_STATES:
        return True
    if record.get("imported_from") is not None:
        return phase6_imported_pilot_record_valid(record, bindings=bindings)
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    discovery = _phase6_discovery_manifest(bindings)
    if not isinstance(discovery, Mapping):
        return False
    schedule_records = bindings.get("schedule", {}).get("payload", {}).get("records")
    matching_schedule = [
        row
        for row in schedule_records
        if isinstance(row, Mapping) and row.get("identity") == record.get("identity")
    ] if isinstance(schedule_records, list) else []
    if len(matching_schedule) != 1:
        return False
    schedule_row = matching_schedule[0]
    process = record.get("process")
    if (
        not isinstance(process, Mapping)
        or process.get("command_argv") != schedule_row.get("child_command_argv")
        or record.get("reason")
        not in PHASE6_TERMINAL_REASON_BY_STATE.get(state, set())
        or evidence.get("classification")
        not in PHASE6_CLASSIFICATION_BY_STATE.get(state, set())
        or not phase6_process_record_valid(
            process,
            terminal_state=state,
            terminal_reason=record.get("reason"),
        )
        or not phase6_evidence_record_valid(evidence, terminal_state=state)
    ):
        return False
    if not (
        evidence.get("dependency_coverage_before")
        == dependency_manifest_covers(
            discovery,
            evidence.get("dependency_manifest_before_builder", {}),
            required_paths=PHASE6_REQUIRED_SOURCE_PATHS,
        )
        and evidence.get("dependency_coverage_after")
        == dependency_manifest_covers(
            discovery,
            evidence.get("dependency_manifest_after_terminal", {}),
            required_paths=PHASE6_REQUIRED_SOURCE_PATHS,
        )
    ):
        return False
    if state != "passed":
        if (
            state == "interrupted"
            and record.get("reason") in {"supervisor_recovery", "outer_termination", "keyboard_interrupt"}
            and process.get("stdout_capture_status") == "unavailable_after_recovery"
            and process.get("stderr_capture_status") == "unavailable_after_recovery"
        ):
            identity = record.get("identity")
            if not isinstance(identity, Mapping):
                return False
            expected_paths = _phase6_child_artifact_paths(identity)
            expected_blob_paths = {
                "child_artifact": str(expected_paths["artifact"].resolve()),
                "payload_sidecar": str(expected_paths["sidecar"].resolve()),
                "progress_journal": str(expected_paths["journal"].resolve()),
            }
            return (
                evidence.get("classification") == "supervisor_interruption"
                and all(
                    phase6_blob_record_valid(evidence.get(field))
                    and evidence[field].get("path") == expected_path
                    for field, expected_path in expected_blob_paths.items()
                )
            )
        failure_valid = _phase6_failure_child_valid(
            record, schedule_row=schedule_row, discovery=discovery
        )
        if record.get("reason") == "authority_revalidation_failed":
            return evidence.get("classification") == "common_invalidity"
        if record.get("reason") == "invalid_child_evidence":
            return (
                evidence.get("classification") == "common_invalidity"
                and not failure_valid
            )
        return failure_valid
    child = evidence.get("child_artifact", {}).get("strict_json")
    identity = record.get("identity")
    if identity.get("operation") == "trace":
        return _phase6_trace_child_valid(
            child,
            identity=identity,
            evidence=evidence,
            discovery=discovery,
            schedule_row=schedule_row,
        )
    if identity.get("operation") in {"xla", "scalar_reference"}:
        if not _phase6_v4_child_valid(
            child,
            identity=identity,
            evidence=evidence,
            discovery=discovery,
            schedule_row=schedule_row,
        ):
            return False
        return (
            child.get("case_id") == schedule_row.get("case_id")
            and child.get("resume_key") == schedule_row.get("resume_key")
            and all(
                child.get(field) == schedule_row.get("fingerprints", {}).get(field)
                for field in FINGERPRINT_FIELDS
            )
            and schedule_row.get("config", {}).get("timesteps") == 120
            and schedule_row.get("config", {}).get("batch_size") == identity["batch_size"]
            and schedule_row.get("config", {}).get("dimension") == identity["dimension"]
            and schedule_row.get("config", {}).get("parameter_count")
            == identity["parameter_count"]
            and schedule_row.get("config", {}).get("dtype") == identity["dtype"]
            and schedule_row.get("config", {}).get("device") == "cpu"
            and schedule_row.get("config", {}).get("cpu_threads") == 1
            and schedule_row.get("config", {}).get("jit_compile")
            is (identity["operation"] == "xla")
        )
    return identity.get("operation") == "p150_routing"


def transition_phase6_record(
    payload: Mapping[str, Any],
    *,
    identity_id: str,
    new_state: str,
    timestamp_utc: str,
    reason: str | None = None,
    process: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    imported_from: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks = phase6_ledger_checks(payload, final=False)
    if not all(checks.values()):
        raise ContractError(f"cannot transition invalid Phase 6 ledger: {checks}")
    updated = copy.deepcopy(dict(payload))
    matches = [
        record
        for record in updated["records"]
        if record["identity"]["identity_id"] == identity_id
    ]
    if len(matches) != 1:
        raise ContractError(f"unknown or duplicate Phase 6 identity {identity_id!r}")
    record = matches[0]
    prior_state = record["state"]
    not_launched = new_state.startswith("not_launched:") and len(new_state.split(":", 1)[1]) > 0
    imported_terminal = (
        prior_state == "pending"
        and new_state in PHASE6_TERMINAL_STATES
        and isinstance(imported_from, Mapping)
    )
    imported_not_launched = (
        prior_state == "pending" and not_launched and isinstance(imported_from, Mapping)
    )
    legal = (
        (prior_state == "pending" and new_state == "running")
        or (prior_state == "pending" and not_launched)
        or (prior_state == "running" and new_state in PHASE6_TERMINAL_STATES)
        or imported_terminal
        or imported_not_launched
    )
    if not legal:
        raise ContractError(f"illegal Phase 6 transition {prior_state!r} -> {new_state!r}")
    if new_state == "running" and (
        reason is not None or evidence is not None or imported_from is not None
    ):
        raise ContractError("running transition cannot carry terminal reason/evidence")
    if new_state == "running" and not phase6_running_process_valid(process):
        raise ContractError("running transition requires a closed process identity")
    if not_launched and (
        reason != new_state.split(":", 1)[1]
        or process is not None
        or evidence is not None
        or imported_from is not None and not imported_not_launched
    ):
        raise ContractError("not-launched transition requires its exact reason and no process")
    if new_state in PHASE6_TERMINAL_STATES and not isinstance(process, Mapping):
        raise ContractError("terminal child transition requires process evidence")
    if new_state in PHASE6_TERMINAL_STATES and not reason:
        raise ContractError("terminal child transition requires a reason")
    if new_state in PHASE6_TERMINAL_STATES and (
        reason not in PHASE6_TERMINAL_REASON_BY_STATE[new_state]
        or not isinstance(evidence, Mapping)
        or evidence.get("classification") not in PHASE6_CLASSIFICATION_BY_STATE[new_state]
    ):
        raise ContractError("terminal state/reason/classification combination is invalid")
    if prior_state == "running" and imported_from is not None:
        raise ContractError("fresh terminal transition cannot carry imported provenance")
    record.update(
        {
            "state": new_state,
            "reason": reason,
            "process": copy.deepcopy(dict(process)) if process is not None else None,
            "evidence": copy.deepcopy(dict(evidence)) if evidence is not None else None,
            "imported_from": (
                copy.deepcopy(dict(imported_from))
                if imported_from is not None
                else None
            ),
        }
    )
    updated["update_index"] += 1
    updated["events"].append(
        {
            "update_index": updated["update_index"],
            "identity_id": identity_id,
            "prior_state": prior_state,
            "new_state": new_state,
            "timestamp_utc": timestamp_utc,
            "evidence_sha256": _phase6_record_event_digest(record),
        }
    )
    checks = phase6_ledger_checks(updated, final=False)
    if not all(checks.values()):
        raise ContractError(f"transition produced invalid Phase 6 ledger: {checks}")
    return updated


def _phase6_replay_events(payload: Mapping[str, Any]) -> bool:
    roster = payload.get("roster")
    events = payload.get("events")
    if not isinstance(roster, list) or not isinstance(events, list):
        return False
    states = {identity.get("identity_id"): "pending" for identity in roster if isinstance(identity, Mapping)}
    if len(states) != len(roster):
        return False
    roster_order = [identity["identity_id"] for identity in roster]
    active_identity: str | None = None
    closed_prefix = 0
    for expected_index, event in enumerate(events, 1):
        if not isinstance(event, Mapping) or set(event) != set(PHASE6_EVENT_FIELDS):
            return False
        identity_id = event.get("identity_id")
        prior = states.get(identity_id)
        new = event.get("new_state")
        record_by_id = {
            record.get("identity", {}).get("identity_id"): record
            for record in payload.get("records", [])
            if isinstance(record, Mapping)
        }
        imported_direct = (
            prior == "pending"
            and new in PHASE6_TERMINAL_STATES
            and isinstance(record_by_id.get(identity_id, {}).get("imported_from"), Mapping)
        )
        legal = (
            prior == "pending" and new == "running"
            or prior == "pending" and isinstance(new, str) and new.startswith("not_launched:")
            or prior == "running" and new in PHASE6_TERMINAL_STATES
            or imported_direct
        )
        if prior == "pending":
            if active_identity is not None or closed_prefix >= len(roster_order):
                return False
            if identity_id != roster_order[closed_prefix]:
                return False
        elif prior == "running" and active_identity != identity_id:
            return False
        if (
            event.get("update_index") != expected_index
            or event.get("prior_state") != prior
            or not legal
            or not isinstance(event.get("timestamp_utc"), str)
            or not event["timestamp_utc"]
            or not isinstance(event.get("evidence_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", event["evidence_sha256"])
        ):
            return False
        states[identity_id] = new
        if new == "running":
            active_identity = identity_id
        else:
            active_identity = None
            closed_prefix += 1
    records = payload.get("records")
    last_events = {
        event["identity_id"]: event
        for event in events
        if isinstance(event, Mapping) and "identity_id" in event
    }
    return isinstance(records, list) and all(
        isinstance(record, Mapping)
        and states.get(record.get("identity", {}).get("identity_id")) == record.get("state")
        and (
            record.get("state") == "pending"
            and record.get("identity", {}).get("identity_id") not in last_events
            or record.get("state") != "pending"
            and last_events.get(record.get("identity", {}).get("identity_id"), {}).get(
                "evidence_sha256"
            )
            == _phase6_record_event_digest(record)
        )
        for record in records
    )


def _phase6_not_launched_reasons_valid(payload: Mapping[str, Any]) -> bool:
    records = payload.get("records")
    schema = payload.get("schema")
    if not isinstance(records, list):
        return False
    by_key = {
        (
            record["identity"]["dimension"],
            record["identity"]["parameter_count"],
            record["identity"]["batch_size"],
            record["identity"]["method_id"],
        ): record
        for record in records
        if isinstance(record, Mapping) and _phase6_identity_valid(record.get("identity"))
    }
    if len(by_key) != len(records):
        return False
    for record in records:
        state = record.get("state")
        if not isinstance(state, str) or not state.startswith("not_launched:"):
            continue
        reason = state.split(":", 1)[1]
        if reason not in PHASE6_NOT_LAUNCHED_REASONS:
            return False
        if record.get("imported_from") is not None:
            if not phase6_imported_pilot_record_valid(
                record, bindings=payload.get("bindings", {})
            ):
                return False
            continue
        identity = record["identity"]
        dimension = identity["dimension"]
        parameter_count = identity["parameter_count"]
        batch_size = identity["batch_size"]
        method = identity["method_id"]
        if schema == PHASE6_PILOT_SCHEMA:
            if reason not in {
                "trace_gate_not_passed",
                "common_invalidity",
                "global_budget_exhausted",
            }:
                return False
            continue
        if schema in {PHASE6_TRACE_SCHEMA, PHASE6_SCALAR_SCHEMA}:
            if reason not in {"common_invalidity", "global_budget_exhausted"}:
                return False
            continue
        if schema == PHASE6_ROUTING_SCHEMA:
            if reason not in {
                "p50_dependency_not_launched",
                "p50_dependency_failed",
                "invalid_dependency_evidence",
                "after_smaller_p150_batch_failure",
                "common_invalidity",
                "global_budget_exhausted",
            }:
                return False
            continue
        if schema != PHASE6_FINAL_SCHEMA:
            return False
        if reason in {"common_invalidity", "global_budget_exhausted"}:
            continue
        if reason == "not_in_gate_b_pilot":
            if (dimension, parameter_count, batch_size) == (10, 50, 1):
                return False
            continue
        smaller_batch = {4: 1, 16: 4}.get(batch_size)
        if reason == "after_smaller_batch_failure":
            dependency = by_key.get((dimension, parameter_count, smaller_batch, method))
            if smaller_batch is None or dependency is None or dependency.get("state") == "passed":
                return False
        elif reason in {"p50_dependency_not_launched", "p50_dependency_failed"}:
            if parameter_count != 150:
                return False
            dependency = by_key.get((dimension, 50, batch_size, method))
            if dependency is None:
                return False
            dependency_not_launched = str(dependency.get("state", "")).startswith("not_launched:")
            if (reason == "p50_dependency_not_launched") != dependency_not_launched:
                return False
            if reason == "p50_dependency_failed" and dependency.get("state") in {"passed", "pending", "running"}:
                return False
        elif reason == "after_smaller_p150_batch_failure":
            dependency = by_key.get((dimension, 150, smaller_batch, method))
            if parameter_count != 150 or smaller_batch is None or dependency is None or dependency.get("state") == "passed":
                return False
        elif reason == "invalid_dependency_evidence":
            if parameter_count != 150:
                return False
        else:
            return False
    return True


def phase6_ledger_checks(payload: Mapping[str, Any], *, final: bool) -> dict[str, bool]:
    top_closed = isinstance(payload, Mapping) and set(payload) == set(PHASE6_LEDGER_FIELDS)
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    roster = payload.get("roster") if isinstance(payload, Mapping) else None
    records = payload.get("records") if isinstance(payload, Mapping) else None
    schema_contract = PHASE6_SCHEMA_CONTRACTS.get(schema)
    bindings = payload.get("bindings") if isinstance(payload, Mapping) else None
    bindings_valid = _phase6_bindings_valid(bindings)
    bindings_gate_valid = (
        bindings_valid
        and schema_contract is not None
        and bindings["schedule"]["payload"]["gate"] == schema_contract[0]
        and bindings["schedule"]["payload"]["gate"] == payload.get("gate")
        and bindings["schedule"]["payload"]["ledger_schema"] == schema
    )
    roster_valid = (
        schema_contract is not None
        and
        isinstance(roster, list)
        and bool(roster)
        and all(_phase6_identity_valid(identity) for identity in roster)
        and len({identity["identity_id"] for identity in roster}) == len(roster)
        and roster == phase6_expected_roster(schema)
    )
    records_valid = isinstance(records, list) and roster_valid and len(records) == len(roster)
    if records_valid:
        for identity, record in zip(roster, records):
            if (
                not isinstance(record, Mapping)
                or set(record) != set(PHASE6_RECORD_FIELDS)
                or record.get("identity") != identity
                or not isinstance(record.get("state"), str)
            ):
                records_valid = False
                break
            state = record["state"]
            allowed_state = (
                state in {"pending", "running", *PHASE6_TERMINAL_STATES}
                or state.startswith("not_launched:")
            )
            not_launched = state.startswith("not_launched:")
            if not allowed_state:
                records_valid = False
                break
            if state in {"pending", "running"} and (
                record.get("reason") is not None
                or record.get("evidence") is not None
                or record.get("imported_from") is not None
            ):
                records_valid = False
                break
            if state == "pending" and record.get("process") is not None:
                records_valid = False
                break
            if state == "running" and not phase6_running_process_valid(record.get("process")):
                records_valid = False
                break
            if not_launched:
                imported_not_launched = record.get("imported_from") is not None
                if (
                    record.get("reason") != state.split(":", 1)[1]
                    or record.get("process") is not None
                    or record.get("evidence") is not None
                    or imported_not_launched
                    and not phase6_imported_pilot_record_valid(
                        record, bindings=payload.get("bindings", {})
                    )
                ):
                    records_valid = False
                    break
            if (
                state in PHASE6_TERMINAL_STATES
                and record.get("imported_from") is not None
                and not phase6_imported_pilot_record_valid(
                    record, bindings=payload.get("bindings", {})
                )
            ):
                records_valid = False
                break
            if state in PHASE6_TERMINAL_STATES and (
                record.get("reason") not in PHASE6_TERMINAL_REASON_BY_STATE[state]
                or record.get("evidence", {}).get("classification")
                not in PHASE6_CLASSIFICATION_BY_STATE[state]
                or not phase6_process_record_valid(
                    record.get("process"),
                    terminal_state=state,
                    terminal_reason=record.get("reason"),
                )
                or not phase6_evidence_record_valid(
                    record.get("evidence"), terminal_state=state
                )
                or not phase6_terminal_record_semantics_valid(
                    record, bindings=payload.get("bindings", {})
                )
            ):
                records_valid = False
                break
    running_count = (
        sum(record.get("state") == "running" for record in records)
        if isinstance(records, list)
        else 2
    )
    terminal = (
        records_valid
        and all(
            record["state"] in PHASE6_TERMINAL_STATES
            or record["state"].startswith("not_launched:")
            for record in records
        )
    )
    top_state = payload.get("state") if isinstance(payload, Mapping) else None
    top_state_valid = top_state in {"running", "passed", "complete_with_failures", "failed"}
    if final:
        top_state_valid = top_state_valid and top_state != "running" and terminal
    else:
        top_state_valid = top_state_valid and (top_state == "running" or terminal)
    aggregate = payload.get("aggregate") if isinstance(payload, Mapping) else None
    aggregate_valid = isinstance(aggregate, Mapping)
    if terminal and not final and top_state == "running" and aggregate == {}:
        aggregate_valid = True
    elif terminal and aggregate_valid:
        try:
            summary = phase6_terminal_summary(payload)
            aggregate_valid = aggregate.get("terminal_summary") == summary
            expected_state = (
                "failed"
                if summary["has_common_invalidity"]
                else "passed"
                if summary["all_passed"]
                else "complete_with_failures"
            )
            top_state_valid = top_state_valid and top_state == expected_state
        except ContractError:
            aggregate_valid = False
    elif not terminal:
        aggregate_valid = aggregate == {}
    return {
        "closed_schema": top_closed,
        "schema_identity": schema_contract is not None,
        "gate_identity": isinstance(payload, Mapping)
        and schema_contract is not None
        and payload.get("gate") == schema_contract[0],
        "artifact_kind_identity": isinstance(payload, Mapping)
        and schema_contract is not None
        and payload.get("artifact_kind") == schema_contract[1],
        "bindings_closed": bindings_valid and bindings_gate_valid,
        "roster_identity": roster_valid,
        "records_identity": records_valid,
        "at_most_one_running": running_count <= 1,
        "transition_log": isinstance(payload, Mapping)
        and type(payload.get("update_index")) is int
        and payload.get("update_index", -1) >= 0
        and isinstance(payload.get("events"), list)
        and payload.get("update_index") == len(payload.get("events", []))
        and _phase6_replay_events(payload),
        "not_launched_reasons": isinstance(payload, Mapping)
        and _phase6_not_launched_reasons_valid(payload),
        "aggregate_object": aggregate_valid,
        "nonclaims_identity": isinstance(payload, Mapping)
        and payload.get("nonclaims") == list(PHASE6_NONCLAIMS),
        "top_state": top_state_valid,
        "final_terminal": terminal if final else True,
    }


def phase6_terminal_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ContractError("Phase 6 terminal summary requires records")
    states: dict[str, int] = {}
    classifications: dict[str, int] = {}
    for record in records:
        state = record.get("state")
        if state in {"pending", "running"}:
            raise ContractError("Phase 6 terminal summary cannot contain unfinished records")
        states[state] = states.get(state, 0) + 1
        if state in PHASE6_TERMINAL_STATES:
            classification = record.get("evidence", {}).get("classification")
            classifications[classification] = classifications.get(classification, 0) + 1
    return {
        "record_count": len(records),
        "state_counts": dict(sorted(states.items())),
        "classification_counts": dict(sorted(classifications.items())),
        "all_passed": bool(records) and all(record.get("state") == "passed" for record in records),
        "has_common_invalidity": any(
            record.get("reason") == "common_invalidity"
            or record.get("evidence", {}).get("classification") == "common_invalidity"
            for record in records
            if record.get("state") in PHASE6_TERMINAL_STATES
            or record.get("state") == "not_launched:common_invalidity"
        ),
    }


def finalize_phase6_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = phase6_terminal_summary(payload)
    updated = copy.deepcopy(dict(payload))
    updated["aggregate"] = {"terminal_summary": summary}
    if summary["has_common_invalidity"]:
        updated["state"] = "failed"
    elif summary["all_passed"]:
        updated["state"] = "passed"
    else:
        updated["state"] = "complete_with_failures"
    checks = phase6_ledger_checks(updated, final=True)
    if not all(checks.values()):
        raise ContractError(f"invalid final Phase 6 ledger: {checks}")
    return updated


def repository_module_manifest(
    repo_root: Path,
    *,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    source = modules if modules is not None else sys.modules
    entries: list[dict[str, str]] = []
    for module_name, module in source.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(module_name, str) or not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        absolute = path.absolute()
        lexically_inside = False
        try:
            absolute.relative_to(root)
            lexically_inside = True
        except ValueError:
            pass
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            if lexically_inside:
                raise ContractError(
                    f"repository module path escapes or cannot resolve safely: {raw_path}"
                )
            continue
        current = root
        symlink_component = False
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                symlink_component = True
                break
        if symlink_component or path.is_symlink() or not resolved.is_file():
            raise ContractError(f"repository module is not a regular non-symlink file: {raw_path}")
        entries.append(
            {
                "module": module_name,
                "path": relative.as_posix(),
                "sha256": file_sha256(resolved),
            }
        )
    entries.sort(key=lambda row: (row["module"], row["path"]))
    seen: dict[tuple[str, str], str] = {}
    for entry in entries:
        key = (entry["module"], entry["path"])
        previous = seen.setdefault(key, entry["sha256"])
        if previous != entry["sha256"]:
            raise ContractError("conflicting repository module bytes")
    return {
        "schema": PHASE6_DEPENDENCY_SCHEMA,
        "repository_root": str(root),
        "entries": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def dependency_manifest_covers(
    discovery: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    required_paths: Sequence[str] = (),
) -> bool:
    required_keys = {"schema", "repository_root", "entries", "manifest_sha256"}
    if (
        not isinstance(discovery, Mapping)
        or not isinstance(actual, Mapping)
        or set(discovery) != required_keys
        or set(actual) != required_keys
        or discovery.get("schema") != PHASE6_DEPENDENCY_SCHEMA
        or actual.get("schema") != PHASE6_DEPENDENCY_SCHEMA
        or discovery.get("repository_root") != actual.get("repository_root")
    ):
        return False
    for manifest in (discovery, actual):
        entries = manifest.get("entries")
        if (
            not isinstance(entries, list)
            or canonical_sha256(entries) != manifest.get("manifest_sha256")
            or any(
                not isinstance(entry, Mapping)
                or set(entry) != {"module", "path", "sha256"}
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256")))
                for entry in entries
            )
        ):
            return False
    allowed = {
        (entry["module"], entry["path"], entry["sha256"])
        for entry in discovery["entries"]
    }
    observed = {
        (entry["module"], entry["path"], entry["sha256"])
        for entry in actual["entries"]
    }
    discovered_paths = {entry["path"] for entry in discovery["entries"]}
    return observed.issubset(allowed) and set(required_paths).issubset(discovered_paths)


def path_digest_record(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ContractError(f"path digest target is not a regular non-symlink file: {path}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def path_digest_record_matches(record: Any) -> bool:
    if (
        not isinstance(record, Mapping)
        or set(record) != set(PHASE6_PATH_DIGEST_FIELDS)
        or not isinstance(record.get("path"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
    ):
        return False
    path = Path(record["path"])
    try:
        resolved = path.resolve(strict=True)
        return (
            not path.is_symlink()
            and resolved.is_file()
            and str(resolved) == record["path"]
            and file_sha256(resolved) == record["sha256"]
        )
    except (OSError, ContractError):
        return False


def _review_declarations(path: Path) -> tuple[dict[str, str], str | None, bool]:
    try:
        lines = [
            line.strip().strip("`")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError):
        return {}, None, False
    declarations: dict[str, str] = {}
    for line in lines:
        for key in (
            "PROPOSAL_PATH",
            "PROPOSAL_SHA256",
            "RESULT_PATH",
            "RESULT_SHA256",
            "PLAN_PATH",
            "PLAN_SHA256",
        ):
            prefix = f"{key}: "
            if line.startswith(prefix):
                declarations[key] = line[len(prefix) :]
    strength = None
    for line in lines:
        for prefix in ("Review strength: ", "REVIEW_STRENGTH: "):
            if line.startswith(prefix):
                candidate = line[len(prefix) :].strip("`")
                if candidate in {"claude_opus_max", "codex_substitute_weaker"}:
                    strength = candidate
    return declarations, strength, bool(lines) and lines[-1] == "VERDICT: AGREE"


def phase6_gate_b_input_records(
    *, repo_root: Path | None = None
) -> list[dict[str, str]]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    return [path_digest_record(root / relative) for relative in PHASE6_GATE_B_INPUT_RELATIVES]


def phase6_gate_b_inputs_valid(inputs: Any) -> bool:
    if not isinstance(inputs, list):
        return False
    try:
        expected = phase6_gate_b_input_records()
    except (OSError, ContractError):
        return False
    if inputs != expected:
        return False
    root = Path(__file__).resolve().parents[1]
    try:
        archive = read_strict_json(root / PHASE6_R2_ARCHIVE_RELATIVE)
        validate_phase6_r2_archive(archive)
        plan_path = (root / PHASE6_PLAN_RELATIVE).resolve(strict=True)
        result_path = (root / PHASE6_REPAIR_RESULT_RELATIVE).resolve(strict=True)
        plan_review_path = (root / PHASE6_PLAN_REVIEW_RELATIVE).resolve(strict=True)
        result_review_path = (
            root / PHASE6_REPAIR_RESULT_REVIEW_RELATIVE
        ).resolve(strict=True)
        declarations, strength, agree = _review_declarations(result_review_path)
    except (OSError, ContractError):
        return False
    return (
        file_sha256(plan_path) == PHASE6_PLAN_SHA256
        and file_sha256(plan_review_path) == PHASE6_PLAN_REVIEW_SHA256
        and declarations
        == {
            "RESULT_PATH": str(result_path),
            "RESULT_SHA256": file_sha256(result_path),
            "PLAN_PATH": str(plan_path),
            "PLAN_SHA256": PHASE6_PLAN_SHA256,
        }
        and strength in {"claude_opus_max", "codex_substitute_weaker"}
        and agree
    )


def phase6_r1_archive_file_records(
    *, repo_root: Path | None = None
) -> list[dict[str, Any]]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    records = []
    for raw_path, expected_sha256, role in PHASE6_R1_ARCHIVE_FILE_SPECS:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_file():
            raise ContractError(f"r1 archive path is not a regular non-symlink file: {path}")
        actual_sha256 = file_sha256(resolved)
        if actual_sha256 != expected_sha256:
            raise ContractError(f"r1 archive path drifted: {resolved}")
        records.append(
            {
                "path": str(resolved),
                "byte_count": resolved.stat().st_size,
                "sha256": actual_sha256,
                "role": role,
                "disposition": "immutable_invalid_harness_lineage",
            }
        )
    return records


def phase6_r1_archive_checks(archive: Any) -> dict[str, bool]:
    files = archive.get("files") if isinstance(archive, Mapping) else None
    diagnosis = archive.get("diagnosis") if isinstance(archive, Mapping) else None
    try:
        expected_files: list[dict[str, Any]] | None = phase6_r1_archive_file_records()
    except (OSError, ContractError):
        expected_files = None
    expected_difference = {
        "index": 0,
        "schedule": PHASE6_PYTHON,
        "child": str(Path(PHASE6_PYTHON).resolve()),
    }
    expected_reviewed_digests = [
        "4403929e2f58d9027b88c21f8840e265a14666a3a7311eb7a0a833723e137bb3",
        "bd449a78fb19c06e90da00892e814eecfa62623c6bcf6f08f2befca29813c332",
        "165e4870155a999661a7502c79347e4b618dee17006749d702b787b7c2b75565",
        "d1c46aacdc6e15c234a3c3d739837d0fcb6fd8c57dbdc6c78f8c532ef0cc1214",
    ]
    protected_current = False
    try:
        root = Path(__file__).resolve().parents[1]
        protected_current = all(
            file_sha256(root / relative) == expected
            for relative, expected in PHASE6_PROTECTED_HASHES.items()
        )
    except OSError:
        pass
    return {
        "closed_schema": isinstance(archive, Mapping)
        and set(archive) == set(PHASE6_R1_ARCHIVE_FIELDS),
        "schema_identity": isinstance(archive, Mapping)
        and archive.get("schema") == PHASE6_R1_ARCHIVE_SCHEMA,
        "authority_identity": isinstance(archive, Mapping)
        and archive.get("authority_id") == PHASE6_R1_AUTHORITY_ID,
        "disposition_identity": isinstance(archive, Mapping)
        and archive.get("disposition")
        == "invalid_harness_authority_exhausted_never_resume_or_import",
        "files_closed": isinstance(files, list)
        and all(
            isinstance(record, Mapping)
            and set(record) == set(PHASE6_R1_ARCHIVE_FILE_FIELDS)
            for record in files
        ),
        "files_current": expected_files is not None and files == expected_files,
        "diagnosis_closed": isinstance(diagnosis, Mapping)
        and set(diagnosis)
        == {
            "trace_ledger_state",
            "trace_ledger_update_index",
            "first_record_state",
            "first_record_reason",
            "child_state",
            "child_stage",
            "child_returncode",
            "argv_differences",
            "other_argv_elements_equal",
            "reviewed_gate_b_subplan_sha256_history",
            "full_child_validity_recomputed",
            "classification",
        },
        "diagnosis_identity": isinstance(diagnosis, Mapping)
        and diagnosis.get("trace_ledger_state") == "running"
        and diagnosis.get("trace_ledger_update_index") == 2
        and diagnosis.get("first_record_state") == "interrupted"
        and diagnosis.get("first_record_reason") == "supervisor_recovery"
        and diagnosis.get("child_state") == "passed"
        and diagnosis.get("child_stage") == "terminal_provenance"
        and diagnosis.get("child_returncode") == 0
        and diagnosis.get("argv_differences") == [expected_difference]
        and diagnosis.get("other_argv_elements_equal") is True
        and diagnosis.get("reviewed_gate_b_subplan_sha256_history")
        == expected_reviewed_digests
        and diagnosis.get("full_child_validity_recomputed") is False
        and diagnosis.get("classification") == "common_invalidity_not_method_evidence",
        "no_live_process": isinstance(archive, Mapping)
        and archive.get("no_live_process")
        == {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": [],
            "passed": True,
        },
        "pre_edit_hashes": isinstance(archive, Mapping)
        and archive.get("pre_edit_lane_hashes") == PHASE6_R1_PRE_EDIT_LANE_HASHES,
        "protected_hashes": isinstance(archive, Mapping)
        and archive.get("protected_hashes") == PHASE6_PROTECTED_HASHES
        and protected_current,
        "nonclaims": isinstance(archive, Mapping)
        and archive.get("nonclaims") == list(PHASE6_R1_ARCHIVE_NONCLAIMS),
    }


def validate_phase6_r1_archive(archive: Any) -> None:
    checks = phase6_r1_archive_checks(archive)
    if not all(checks.values()):
        raise ContractError(f"invalid Phase 6 r1 archive: {checks}")


def phase6_r2_archive_file_records(
    *, repo_root: Path | None = None
) -> list[dict[str, Any]]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    records = []
    for raw_path, expected_sha256, role, file_format in PHASE6_R2_ARCHIVE_FILE_SPECS:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_file():
            raise ContractError(f"r2 archive path is not a regular non-symlink file: {path}")
        actual_sha256 = file_sha256(resolved)
        if actual_sha256 != expected_sha256:
            raise ContractError(f"r2 archive path drifted: {resolved}")
        records.append(
            {
                "path": str(resolved),
                "byte_count": resolved.stat().st_size,
                "sha256": actual_sha256,
                "role": role,
                "format": file_format,
                "disposition": "immutable_invalid_harness_lineage",
            }
        )
    return records


def phase6_r2_archive_absent_paths(
    *, repo_root: Path | None = None
) -> list[str]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    paths = []
    for raw_path in PHASE6_R2_ARCHIVE_ABSENT_PATHS:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if path.exists() or path.is_symlink():
            raise ContractError(f"r2 archive expected an absent path: {path}")
        paths.append(str(path.absolute()))
    return paths


def phase6_r2_archive_checks(archive: Any) -> dict[str, bool]:
    root = Path(PHASE6_R2_WORK_ROOT)
    budget_dir = root / "budget_state"
    files = archive.get("files") if isinstance(archive, Mapping) else None
    try:
        expected_files: list[dict[str, Any]] | None = phase6_r2_archive_file_records()
        expected_absent: list[str] | None = phase6_r2_archive_absent_paths()
        root_entries = sorted(path.name for path in root.iterdir())
        budget_entries = sorted(path.name for path in budget_dir.iterdir())
        budget_state = read_strict_json(
            budget_dir / f"gate_b-{PHASE6_R2_AUTHORITY_ID}.json"
        )
        lease = read_strict_json(
            budget_dir / f"gate_b-{PHASE6_R2_AUTHORITY_ID}.json.lease"
        )
    except (OSError, ContractError):
        expected_files = None
        expected_absent = None
        root_entries = []
        budget_entries = []
        budget_state = None
        lease = None
    protected_current = all(
        file_sha256(Path(__file__).resolve().parents[1] / relative) == digest
        for relative, digest in PHASE6_PROTECTED_HASHES.items()
    )
    return {
        "closed_schema": isinstance(archive, Mapping)
        and set(archive) == set(PHASE6_R2_ARCHIVE_FIELDS),
        "schema_identity": isinstance(archive, Mapping)
        and archive.get("schema") == PHASE6_R2_ARCHIVE_SCHEMA,
        "authority_identity": isinstance(archive, Mapping)
        and archive.get("authority_id") == PHASE6_R2_AUTHORITY_ID,
        "disposition_identity": isinstance(archive, Mapping)
        and archive.get("disposition")
        == "invalid_harness_authority_exhausted_never_resume_or_import",
        "files_closed": isinstance(files, list)
        and all(
            isinstance(record, Mapping)
            and set(record) == set(PHASE6_R2_ARCHIVE_FILE_FIELDS)
            for record in files
        ),
        "files_current": expected_files is not None and files == expected_files,
        "absent_paths": expected_absent is not None
        and archive.get("absent_paths") == expected_absent,
        "work_root_closed": archive.get("work_root_entries")
        == ["budget_state", "import_discovery.json"]
        == root_entries,
        "budget_state_closed": archive.get("budget_state_entries")
        == [
            f"gate_b-{PHASE6_R2_AUTHORITY_ID}.json",
            f"gate_b-{PHASE6_R2_AUTHORITY_ID}.json.lease",
        ]
        == budget_entries,
        "diagnosis": archive.get("diagnosis")
        == {
            "classification": "common_invalidity_not_method_evidence",
            "failure_stage": "pre_trace_ledger_binding_validation",
            "target_fixture_constructed": False,
            "target_trace_requested": False,
            "target_xla_requested": False,
            "trace_output_present": False,
            "pilot_output_present": False,
            "budget_state": "running",
            "budget_update_index": 0,
            "lease_state": "released",
            "mixed_format_inputs": ["json", "markdown", "markdown", "markdown"],
        }
        and isinstance(budget_state, Mapping)
        and budget_state.get("state") == "running"
        and budget_state.get("update_index") == 0
        and isinstance(lease, Mapping)
        and lease.get("state") == "released",
        "no_live_process": archive.get("no_live_process")
        == {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": [],
            "passed": True,
        },
        "protected_hashes": archive.get("protected_hashes") == PHASE6_PROTECTED_HASHES
        and protected_current,
        "nonclaims": archive.get("nonclaims") == list(PHASE6_R2_ARCHIVE_NONCLAIMS),
    }


def validate_phase6_r2_archive(archive: Any) -> None:
    checks = phase6_r2_archive_checks(archive)
    if not all(checks.values()):
        raise ContractError(f"invalid Phase 6 r2 archive: {checks}")


def phase6_parse_opening_hash_ledger(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ContractError("opening hash ledger must be a regular non-symlink file")
    if file_sha256(resolved) != PHASE6_OPENING_HASH_LEDGER_SHA256:
        raise ContractError("opening hash ledger bytes do not match the frozen entry digest")
    try:
        lines = resolved.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read opening hash ledger: {exc}") from exc
    expected_headers = [
        "# Phase 6 pre-edit path ledger",
        "# Created before Gate A implementation; SHA-256 lines are opening bytes.",
    ]
    if lines[:2] != expected_headers or any(not line for line in lines):
        raise ContractError("opening hash ledger headers or blank-line contract failed")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines[2:]:
        match = re.fullmatch(r"([0-9a-f]{64}|ABSENT)  ([^\s].*)", line)
        if match is None:
            raise ContractError(f"malformed opening hash ledger line: {line!r}")
        opening, raw_relative = match.groups()
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or raw_relative != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ContractError(f"unsafe opening hash ledger path: {raw_relative!r}")
        target = root / relative
        try:
            target.resolve(strict=False).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ContractError(f"opening hash ledger path escapes repository: {raw_relative}") from exc
        if raw_relative in seen:
            raise ContractError(f"duplicate opening hash ledger path: {raw_relative}")
        seen.add(raw_relative)
        entries.append(
            {
                "opening_state": "absent" if opening == "ABSENT" else "present",
                "path": raw_relative,
                "sha256": None if opening == "ABSENT" else opening,
            }
        )
    expected_absent = set(PHASE6_OPENING_ABSENT_PATHS)
    expected_nonhistorical_present = {
        *PHASE6_OPENING_MUTABLE_PATHS,
        *PHASE6_OPENING_FIXED_PATHS,
    }
    by_path = {entry["path"]: entry for entry in entries}
    historical = set(by_path) - expected_nonhistorical_present - expected_absent
    if (
        len(entries) != 144
        or len(historical) != 106
        or any(
            re.fullmatch(r"docs/(benchmarks|plans)/[^/]*2026-07-09[^/]*", relative)
            is None
            for relative in historical
        )
    ):
        raise ContractError("opening hash ledger historical inventory is not the frozen class")
    if not (expected_nonhistorical_present | expected_absent).issubset(by_path):
        missing = sorted((expected_nonhistorical_present | expected_absent) - set(by_path))
        raise ContractError(f"opening hash ledger required coverage mismatch: missing={missing}")
    if any(by_path[relative]["opening_state"] != "absent" for relative in expected_absent):
        raise ContractError("opening hash ledger ABSENT path classification mismatch")
    expected_present = expected_nonhistorical_present | historical
    if any(by_path[relative]["opening_state"] != "present" for relative in expected_present):
        raise ContractError("opening hash ledger present path classification mismatch")
    fixed = set(PHASE6_OPENING_FIXED_PATHS) | historical
    for relative in fixed:
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ContractError(f"fixed opening path is not a regular file: {relative}")
        if file_sha256(target) != by_path[relative]["sha256"]:
            raise ContractError(f"fixed opening path drifted: {relative}")
    for relative in PHASE6_OPENING_MUTABLE_PATHS:
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ContractError(f"mutable opening path is not a regular file: {relative}")
    return entries


def phase6_opening_hash_ledger_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    entries = phase6_parse_opening_hash_ledger(resolved)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "entries": entries,
        "entries_sha256": canonical_sha256(entries),
    }


def phase6_opening_hash_ledger_record_matches(record: Any) -> bool:
    if (
        not isinstance(record, Mapping)
        or set(record) != set(PHASE6_OPENING_LEDGER_FIELDS)
        or not isinstance(record.get("path"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("entries_sha256"))) is None
        or not isinstance(record.get("entries"), list)
        or any(
            not isinstance(entry, Mapping)
            or set(entry) != set(PHASE6_OPENING_ENTRY_FIELDS)
            for entry in record["entries"]
        )
    ):
        return False
    try:
        current = phase6_opening_hash_ledger_record(Path(record["path"]))
    except (OSError, ContractError):
        return False
    return dict(record) == current


def _phase6_command_options(argv: Sequence[str]) -> dict[str, list[str]] | None:
    options: dict[str, list[str]] = {}
    index = 2
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or token in options:
            return None
        values: list[str] = []
        index += 1
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(argv[index])
            index += 1
        options[token] = values
    return options


def _phase6_resolved_repo_path(value: str, relative: str) -> bool:
    try:
        root = Path(__file__).resolve().parents[1]
        expected = root / relative
        actual = Path(value)
        if actual.is_absolute():
            return actual.resolve() == expected.resolve()
        return actual.as_posix() == Path(relative).as_posix()
    except OSError:
        return False


def _phase6_child_artifact_paths(identity: Mapping[str, Any]) -> dict[str, Path]:
    digest = canonical_sha256(identity)[:24]
    root = Path(PHASE6_WORK_ROOT) / identity["operation"] / digest
    return {
        "artifact": root.with_suffix(".json"),
        "markdown": root.with_suffix(".md"),
        "sidecar": root.with_suffix(".payload.json"),
        "journal": root.with_suffix(".jsonl"),
        "dependency_before": root.with_suffix(".dependency-before.json"),
        "dependency_after": root.with_suffix(".dependency-after.json"),
        "authority_snapshot": root.with_suffix(".authority.json"),
    }


def _phase6_schedule_child_argv_valid(row: Mapping[str, Any]) -> bool:
    identity = row.get("identity")
    config = row.get("config")
    fingerprints = row.get("fingerprints")
    argv = row.get("child_command_argv")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(fingerprints, Mapping)
        or not isinstance(argv, list)
        or not all(isinstance(value, str) for value in argv)
    ):
        return False
    paths = _phase6_child_artifact_paths(identity)
    common = [
        PHASE6_PYTHON,
        str((Path(__file__).resolve().parents[1] / PHASE6_BENCHMARK_RELATIVE).resolve()),
        "--dimensions", str(identity["dimension"]),
        "--parameter-counts", str(identity["parameter_count"]),
        "--timesteps", "120",
        "--batch-size", str(identity["batch_size"]),
        "--dtype", identity["dtype"],
        "--device", "cpu",
        "--cpu-threads", "1",
    ]
    identity_args = [
        "--method", identity["method_id"],
        "--case-id", row.get("case_id"),
        "--attempt-id", row.get("attempt_id"),
        "--progress-journal", str(paths["journal"]),
        "--source-fingerprint", fingerprints.get("source_fingerprint"),
        "--config-fingerprint", fingerprints.get("config_fingerprint"),
        "--runtime-fingerprint", fingerprints.get("runtime_fingerprint"),
        "--fixture-fingerprint", fingerprints.get("fixture_fingerprint"),
        "--schedule-fingerprint", fingerprints.get("schedule_fingerprint"),
        "--resume-key", row.get("resume_key"),
        "--phase6-authority-snapshot", str(paths["authority_snapshot"]),
    ]
    if any(not isinstance(value, str) for value in identity_args):
        return False
    if identity["operation"] == "trace":
        expected = [
            *common[:2],
            "--phase6-trace-only",
            *common[2:],
            *identity_args,
            "--output-json", str(paths["artifact"]),
            "--output-md", str(paths["markdown"]),
            "--no-jit-compile",
        ]
    else:
        expected = [
            *common,
            "--repeats", str(config.get("repeats")),
            *identity_args,
            "--plan-path", PHASE6_PLAN_RELATIVE,
            "--output-json", str(paths["artifact"]),
            "--output-md", str(paths["markdown"]),
            "--phase6-dependency-before", str(paths["dependency_before"]),
            "--phase6-dependency-after", str(paths["dependency_after"]),
            "--jit-compile" if identity["operation"] == "xla" else "--no-jit-compile",
            "--tf32-enabled",
        ]
    return argv == expected


def _phase6_command_argv_valid(command: Mapping[str, Any], *, gate: str) -> bool:
    argv = command.get("argv")
    if not isinstance(argv, list) or len(argv) < 3 or not all(isinstance(value, str) for value in argv):
        return False
    try:
        python_valid = Path(argv[0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
    except OSError:
        return False
    if not python_valid or not _phase6_resolved_repo_path(argv[1], PHASE6_SUPERVISOR_RELATIVE):
        return False
    options = _phase6_command_options(argv)
    if options is None:
        return False
    common = {
        "--dimensions": ["10", "20", "30"],
        "--parameter-counts": ["50", "150"],
        "--batch-sizes": ["1", "4", "16"],
        "--timesteps": ["120"],
        "--dtype": ["float32"],
        "--device": ["cpu"],
        "--cpu-threads": ["1"],
    }
    if gate == "gate_b":
        required = {
            "--phase6-pilot": [],
            **common,
            "--jit-compile": [],
            "--trace-child-timeout-seconds": ["60"],
            "--xla-child-timeout-seconds": ["60"],
            "--xla-cell-timeout-seconds": ["160"],
            "--budget-contract": [PHASE6_GATE_B_BUDGET_RELATIVE],
            "--budget-attestation": [PHASE6_GATE_B_ATTESTATION_RELATIVE],
            "--trace-output-json": [PHASE6_GATE_B_ARTIFACTS["trace_output_json"]],
            "--output-json": [PHASE6_GATE_B_ARTIFACTS["pilot_output_json"]],
        }
        return set(options) == set(required) and all(options[key] == value for key, value in required.items())
    name = command.get("name")
    if name == "scalar_references":
        required = {
            "--phase6-scalar-references": [],
            "--dimensions": ["10"],
            "--parameter-counts": ["50"],
            "--batch-sizes": ["1", "4"],
            "--timesteps": ["120"],
            "--dtype": ["float32"],
            "--device": ["cpu"],
            "--cpu-threads": ["1"],
            "--no-jit-compile": [],
            "--child-timeout-seconds": ["60"],
            "--budget-contract": [PHASE6_GATE_C_BUDGET_RELATIVE],
            "--budget-attestation": [PHASE6_GATE_C_ATTESTATION_RELATIVE],
            "--output-json": [PHASE6_GATE_C_ARTIFACTS["scalar_output_json"]],
        }
        return set(options) == set(required) and all(options[key] == value for key, value in required.items())
    if name == "remaining_lattice":
        required = {
            "--phase6-remaining": [],
            **common,
            "--jit-compile": [],
            "--child-timeout-seconds": ["60"],
            "--cell-timeout-seconds": ["160"],
            "--trace-input": [PHASE6_GATE_B_ARTIFACTS["trace_output_json"]],
            "--pilot-input": [PHASE6_GATE_B_ARTIFACTS["pilot_output_json"]],
            "--scalar-reference-input": [PHASE6_GATE_C_ARTIFACTS["scalar_output_json"]],
            "--budget-contract": [PHASE6_GATE_C_BUDGET_RELATIVE],
            "--budget-attestation": [PHASE6_GATE_C_ATTESTATION_RELATIVE],
            "--routing-output-json": [PHASE6_GATE_C_ARTIFACTS["routing_output_json"]],
            "--output-json": [PHASE6_GATE_C_ARTIFACTS["final_output_json"]],
        }
        return set(options) == set(required) and all(options[key] == value for key, value in required.items())
    return False


def phase6_schedule_checks(schedule: Mapping[str, Any]) -> dict[str, bool]:
    ledger_schema = schedule.get("ledger_schema") if isinstance(schedule, Mapping) else None
    schema_contract = PHASE6_SCHEMA_CONTRACTS.get(ledger_schema)
    records = schedule.get("records") if isinstance(schedule, Mapping) else None
    core = {
        "schema": schedule.get("schema"),
        "ledger_schema": ledger_schema,
        "gate": schedule.get("gate"),
        "records": records,
    } if isinstance(schedule, Mapping) else {}
    roster = phase6_expected_roster(ledger_schema) if schema_contract is not None else []
    rows_closed = isinstance(records, list) and all(
        isinstance(row, Mapping)
        and set(row) == set(PHASE6_SCHEDULE_RECORD_FIELDS)
        and _phase6_identity_valid(row.get("identity"))
        and isinstance(row.get("case_id"), str)
        and isinstance(row.get("attempt_id"), str)
        and bool(row["attempt_id"])
        and isinstance(row.get("config"), Mapping)
        and isinstance(row.get("fingerprints"), Mapping)
        and set(row["fingerprints"]) == set(FINGERPRINT_FIELDS)
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(row["fingerprints"].get(field)))
            is not None
            for field in FINGERPRINT_FIELDS
        )
        and re.fullmatch(r"[0-9a-f]{64}", str(row.get("resume_key"))) is not None
        and isinstance(row.get("child_command_argv"), list)
        and bool(row["child_command_argv"])
        and all(isinstance(value, str) for value in row["child_command_argv"])
        for row in records
    )
    row_semantics = rows_closed
    schedule_identity_rows: list[dict[str, Any]] = []
    if row_semantics:
        for row in records:
            identity = row["identity"]
            config = row["config"]
            try:
                config_fingerprint = config_manifest(config)["config_fingerprint"]
            except ContractError:
                row_semantics = False
                break
            expected_case = case_id(config)
            operation = identity["operation"]
            if (
                row["case_id"] != expected_case
                or config_fingerprint != row["fingerprints"]["config_fingerprint"]
                or config.get("method_id") != identity["method_id"]
                or config.get("dimension") != identity["dimension"]
                or config.get("parameter_count") != identity["parameter_count"]
                or config.get("batch_size") != identity["batch_size"]
                or config.get("dtype") != identity["dtype"]
                or config.get("timesteps") != 120
                or config.get("device") != "cpu"
                or config.get("cpu_threads") != 1
                or config.get("jit_compile") is not (operation == "xla")
                or config.get("repeats") != (2 if operation == "xla" else 1)
                or not _phase6_schedule_child_argv_valid(row)
            ):
                row_semantics = False
                break
            schedule_identity_rows.append(
                {
                    "identity": identity,
                    "case_id": row["case_id"],
                    "config": config,
                    "source_fingerprint": row["fingerprints"]["source_fingerprint"],
                    "runtime_fingerprint": row["fingerprints"]["runtime_fingerprint"],
                    "fixture_fingerprint": row["fingerprints"]["fixture_fingerprint"],
                }
            )
        if row_semantics:
            schedule_fingerprint = canonical_sha256(
                {
                    "schema": PHASE6_SCHEDULE_SCHEMA,
                    "ledger_schema": ledger_schema,
                    "gate": schedule["gate"],
                    "records": schedule_identity_rows,
                }
            )
            row_semantics = all(
                row["fingerprints"]["schedule_fingerprint"] == schedule_fingerprint
                and row["resume_key"]
                == resume_key(
                    case_identity=row["case_id"],
                    method_id=row["identity"]["method_id"],
                    fingerprints=row["fingerprints"],
                )
                for row in records
            )
    return {
        "closed_schema": isinstance(schedule, Mapping)
        and set(schedule) == {"schema", "ledger_schema", "gate", "records", "schedule_sha256"},
        "schema_identity": isinstance(schedule, Mapping)
        and schedule.get("schema") == PHASE6_SCHEDULE_SCHEMA,
        "ledger_schema_identity": schema_contract is not None,
        "gate_identity": schema_contract is not None
        and schedule.get("gate") == schema_contract[0],
        "roster_identity": rows_closed
        and [row["identity"] for row in records] == roster,
        "row_semantics": row_semantics,
        "schedule_digest": isinstance(schedule, Mapping)
        and schedule.get("schedule_sha256") == canonical_sha256(core),
    }


def phase6_budget_proposal_checks(
    proposal: Mapping[str, Any],
    *,
    expected_gate: str | None = None,
) -> dict[str, bool]:
    plan = proposal.get("plan") if isinstance(proposal, Mapping) else None
    opening = proposal.get("opening_hash_ledger") if isinstance(proposal, Mapping) else None
    discovery = proposal.get("dependency_discovery") if isinstance(proposal, Mapping) else None
    sources = proposal.get("source_hashes") if isinstance(proposal, Mapping) else None
    commands = proposal.get("commands") if isinstance(proposal, Mapping) else None
    schedules = proposal.get("schedules") if isinstance(proposal, Mapping) else None
    artifacts = proposal.get("artifacts") if isinstance(proposal, Mapping) else None
    budget = proposal.get("budget") if isinstance(proposal, Mapping) else None
    inputs = proposal.get("inputs") if isinstance(proposal, Mapping) else None
    authority_id = proposal.get("authority_id") if isinstance(proposal, Mapping) else None
    digest_records = [plan]
    if isinstance(sources, list):
        digest_records.extend(sources)
    if isinstance(inputs, list):
        digest_records.extend(inputs)
    path_records_valid = all(path_digest_record_matches(record) for record in digest_records)
    opening_record_valid = phase6_opening_hash_ledger_record_matches(opening)
    command_names = (
        [command.get("name") for command in commands]
        if isinstance(commands, list) and all(isinstance(command, Mapping) for command in commands)
        else []
    )
    gate = proposal.get("gate") if isinstance(proposal, Mapping) else None
    repo_root = Path(__file__).resolve().parents[1]
    plan_identity = (
        isinstance(plan, Mapping)
        and plan.get("path") == str((repo_root / PHASE6_PLAN_RELATIVE).resolve())
    )
    opening_identity = (
        isinstance(opening, Mapping)
        and opening.get("path") == str(Path(PHASE6_OPENING_HASH_LEDGER).resolve())
    )
    source_identity = (
        isinstance(sources, list)
        and [record.get("path") for record in sources]
        == [str((repo_root / relative).resolve()) for relative in PHASE6_REQUIRED_SOURCE_PATHS]
    )
    discovery_manifest = discovery.get("manifest") if isinstance(discovery, Mapping) else None
    required_discovery_paths = set(PHASE6_REQUIRED_SOURCE_PATHS)
    discovery_entries = (
        discovery_manifest.get("entries") if isinstance(discovery_manifest, Mapping) else None
    )
    discovery_manifest_valid = (
        isinstance(discovery_manifest, Mapping)
        and set(discovery_manifest)
        == {"schema", "repository_root", "entries", "manifest_sha256"}
        and discovery_manifest.get("schema") == PHASE6_DEPENDENCY_SCHEMA
        and discovery_manifest.get("repository_root") == str(repo_root.resolve())
        and isinstance(discovery_entries, list)
        and canonical_sha256(discovery_entries) == discovery_manifest.get("manifest_sha256")
        and all(
            isinstance(entry, Mapping)
            and set(entry) == {"module", "path", "sha256"}
            and isinstance(entry.get("module"), str)
            and isinstance(entry.get("path"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is not None
            for entry in discovery_entries
        )
        and required_discovery_paths.issubset(
            {entry["path"] for entry in discovery_entries}
        )
    )
    gate_specific = False
    if gate == "gate_b" and isinstance(budget, Mapping) and isinstance(artifacts, Mapping):
        gate_specific = (
            command_names == ["trace_census_and_pilot"]
            and commands[0].get("term_deadline_seconds") == 3000
            and commands[0].get("kill_grace_seconds") == 45
            and artifacts == PHASE6_GATE_B_ARTIFACTS
            and phase6_gate_b_inputs_valid(inputs)
            and isinstance(schedules, Mapping)
            and set(schedules) == {PHASE6_TRACE_SCHEMA, PHASE6_PILOT_SCHEMA}
            and budget.get("child_execution_deadline_seconds") == 60
            and budget.get("child_term_grace_seconds") == 5
            and budget.get("child_kill_reap_grace_seconds") == 5
            and budget.get("child_lifecycle_cap_seconds") == 70
            and budget.get("cell_cap_seconds") == 160
            and budget.get("outer_term_deadline_seconds") == 3000
            and budget.get("outer_kill_grace_seconds") == 45
            and budget.get("hard_ceiling_seconds") == 3045
        )
    elif gate == "gate_c" and isinstance(budget, Mapping) and isinstance(artifacts, Mapping):
        gate_specific = (
            command_names == ["scalar_references", "remaining_lattice"]
            and 0 < commands[0].get("term_deadline_seconds", 0) <= 330
            and 0 < commands[1].get("term_deadline_seconds", 0) <= 2700
            and 0 < commands[0].get("kill_grace_seconds", 0) <= 45
            and 0 < commands[1].get("kill_grace_seconds", 0) <= 45
            and artifacts == PHASE6_GATE_C_ARTIFACTS
            and isinstance(inputs, list)
            and len(inputs) == 2
            and isinstance(schedules, Mapping)
            and set(schedules) == {PHASE6_SCALAR_SCHEMA, PHASE6_FINAL_SCHEMA}
            and [record.get("path") for record in inputs]
            == [
                str((repo_root / PHASE6_GATE_B_ARTIFACTS["trace_output_json"]).resolve()),
                str((repo_root / PHASE6_GATE_B_ARTIFACTS["pilot_output_json"]).resolve()),
            ]
            and 0 < budget.get("child_execution_deadline_seconds", 0) <= 60
            and 0 <= budget.get("child_term_grace_seconds", -1) <= 5
            and 0 <= budget.get("child_kill_reap_grace_seconds", -1) <= 5
            and 0 < budget.get("child_lifecycle_cap_seconds", 0) <= 70
            and 0 < budget.get("cell_cap_seconds", 0) <= 160
            and 0 < budget.get("outer_term_deadline_seconds", 0) <= 2700
            and 0 < budget.get("outer_kill_grace_seconds", 0) <= 45
            and 0 < budget.get("hard_ceiling_seconds", 0) <= 3120
            and commands[1]["term_deadline_seconds"]
            == budget["outer_term_deadline_seconds"]
            and commands[1]["kill_grace_seconds"]
            == budget["outer_kill_grace_seconds"]
            and commands[0]["term_deadline_seconds"]
            + commands[0]["kill_grace_seconds"]
            + commands[1]["term_deadline_seconds"]
            + commands[1]["kill_grace_seconds"]
            <= budget["hard_ceiling_seconds"]
            and budget["child_execution_deadline_seconds"]
            + budget["child_term_grace_seconds"]
            + budget["child_kill_reap_grace_seconds"]
            <= budget["child_lifecycle_cap_seconds"]
        )
    environment_closed = isinstance(commands, list) and all(
        isinstance(command, Mapping)
        and command.get("environment")
        == {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
        for command in commands
    )
    expected_schedule_schemas = (
        {PHASE6_TRACE_SCHEMA, PHASE6_PILOT_SCHEMA}
        if gate == "gate_b"
        else {PHASE6_SCALAR_SCHEMA, PHASE6_FINAL_SCHEMA}
        if gate == "gate_c"
        else set()
    )
    return {
        "closed_schema": isinstance(proposal, Mapping)
        and set(proposal) == set(PHASE6_BUDGET_FIELDS),
        "schema_identity": isinstance(proposal, Mapping)
        and proposal.get("schema") == PHASE6_BUDGET_SCHEMA,
        "authority_identity": isinstance(authority_id, str)
        and re.fullmatch(r"[0-9a-f]{64}", authority_id) is not None,
        "gate_identity": isinstance(proposal, Mapping)
        and isinstance(proposal.get("gate"), str)
        and proposal["gate"] in {"gate_b", "gate_c"}
        and (expected_gate is None or proposal["gate"] == expected_gate),
        "path_digests_current": path_records_valid and opening_record_valid,
        "authoritative_plan_path": plan_identity,
        "opening_hash_ledger_path": opening_identity and opening_record_valid,
        "source_path_set_identity": source_identity,
        "dependency_discovery_closed": isinstance(discovery, Mapping)
        and set(discovery)
        == {
            "schema",
            "kind",
            "command_argv",
            "environment",
            "fixture_constructed",
            "trace_requested",
            "selected_method_constructed",
            "concrete_function_invocations",
            "manifest",
            "nonclaims",
        }
        and discovery.get("schema")
        == "bayesfilter.kalman_qr_batched_xla_repair.phase6.import_discovery.v1"
        and discovery.get("kind") == "import_only_no_fixture_trace_or_execution"
        and discovery.get("command_argv") == PHASE6_IMPORT_DISCOVERY_ARGV
        and discovery.get("environment") == PHASE6_ENVIRONMENT
        and discovery.get("fixture_constructed") is False
        and discovery.get("trace_requested") is False
        and discovery.get("selected_method_constructed") is False
        and discovery.get("concrete_function_invocations") == 0
        and isinstance(discovery.get("manifest"), Mapping)
        and discovery["manifest"].get("schema") == PHASE6_DEPENDENCY_SCHEMA
        and discovery.get("nonclaims") == list(PHASE6_NONCLAIMS),
        "dependency_manifest_identity": discovery_manifest_valid,
        "source_hashes_nonempty": isinstance(sources, list) and bool(sources),
        "commands_closed": isinstance(commands, list)
        and bool(commands)
        and all(
            isinstance(command, Mapping)
            and set(command) == {"name", "argv", "environment", "term_deadline_seconds", "kill_grace_seconds"}
            and isinstance(command.get("name"), str)
            and isinstance(command.get("argv"), list)
            and bool(command["argv"])
            and all(isinstance(value, str) for value in command["argv"])
            and isinstance(command.get("environment"), Mapping)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in command["environment"].items())
            and _is_finite_nonnegative_number(command.get("term_deadline_seconds"))
            and _is_finite_nonnegative_number(command.get("kill_grace_seconds"))
            for command in commands
        ),
        "command_argv_identity": isinstance(commands, list)
        and all(_phase6_command_argv_valid(command, gate=gate) for command in commands),
        "command_environment_identity": environment_closed,
        "schedules_closed": isinstance(schedules, Mapping)
        and set(schedules) == expected_schedule_schemas
        and all(
            schema == schedule.get("ledger_schema")
            and all(phase6_schedule_checks(schedule).values())
            for schema, schedule in schedules.items()
            if isinstance(schema, str) and isinstance(schedule, Mapping)
        )
        and len(schedules)
        == sum(
            isinstance(schema, str) and isinstance(schedule, Mapping)
            for schema, schedule in schedules.items()
        ),
        "gate_specific_budget_and_order": gate_specific,
        "artifact_paths_closed": isinstance(artifacts, Mapping)
        and bool(artifacts)
        and all(isinstance(key, str) and isinstance(value, str) and value for key, value in artifacts.items()),
        "budget_closed": isinstance(budget, Mapping)
        and set(budget)
        == {
            "child_execution_deadline_seconds",
            "child_term_grace_seconds",
            "child_kill_reap_grace_seconds",
            "child_lifecycle_cap_seconds",
            "cell_cap_seconds",
            "outer_term_deadline_seconds",
            "outer_kill_grace_seconds",
            "hard_ceiling_seconds",
        }
        and all(_is_finite_nonnegative_number(value) for value in budget.values()),
        "inputs_list": isinstance(inputs, list)
        and (
            phase6_gate_b_inputs_valid(inputs)
            if gate == "gate_b"
            else True
        ),
        "nonclaims_identity": isinstance(proposal, Mapping)
        and proposal.get("nonclaims") == list(PHASE6_NONCLAIMS),
    }


def validate_phase6_budget_proposal(
    proposal: Mapping[str, Any],
    *,
    expected_gate: str,
) -> None:
    checks = phase6_budget_proposal_checks(proposal, expected_gate=expected_gate)
    if not all(checks.values()):
        raise ContractError(f"invalid Phase 6 budget proposal: {checks}")
    identity_payload = {key: proposal[key] for key in PHASE6_BUDGET_FIELDS if key != "authority_id"}
    if proposal["authority_id"] != canonical_sha256(identity_payload):
        raise ContractError("Phase 6 budget authority_id does not bind proposal bytes")


def phase6_attestation_checks(
    attestation: Mapping[str, Any],
    *,
    proposal_path: Path,
    expected_gate: str,
) -> dict[str, bool]:
    proposal_record = attestation.get("proposal") if isinstance(attestation, Mapping) else None
    plan_record = attestation.get("plan") if isinstance(attestation, Mapping) else None
    review_record = attestation.get("review") if isinstance(attestation, Mapping) else None
    try:
        proposal = read_strict_json(proposal_path)
        validate_phase6_budget_proposal(proposal, expected_gate=expected_gate)
    except ContractError:
        proposal = None
    review_verdict = False
    if isinstance(review_record, Mapping) and path_digest_record_matches(review_record):
        try:
            declared, declared_strength, agree = _review_declarations(
                Path(review_record["path"])
            )
            review_verdict = (
                agree
                and isinstance(proposal, Mapping)
                and declared
                == {
                    "PROPOSAL_PATH": str(proposal_path.resolve()),
                    "PROPOSAL_SHA256": file_sha256(proposal_path.resolve()),
                    "PLAN_PATH": proposal.get("plan", {}).get("path"),
                    "PLAN_SHA256": proposal.get("plan", {}).get("sha256"),
                }
                and declared_strength == attestation.get("review_strength")
            )
        except (OSError, KeyError, UnicodeDecodeError):
            review_verdict = False
    return {
        "closed_schema": isinstance(attestation, Mapping)
        and set(attestation) == set(PHASE6_ATTESTATION_FIELDS),
        "schema_identity": isinstance(attestation, Mapping)
        and attestation.get("schema") == PHASE6_ATTESTATION_SCHEMA,
        "gate_identity": isinstance(attestation, Mapping)
        and attestation.get("gate") == expected_gate,
        "authority_identity": isinstance(attestation, Mapping)
        and isinstance(proposal, Mapping)
        and attestation.get("authority_id") == proposal.get("authority_id"),
        "proposal_digest": path_digest_record_matches(proposal_record)
        and isinstance(proposal_record, Mapping)
        and proposal_record.get("path") == str(proposal_path.resolve()),
        "plan_digest": path_digest_record_matches(plan_record)
        and isinstance(proposal, Mapping)
        and plan_record == proposal.get("plan"),
        "review_digest": path_digest_record_matches(review_record),
        "verdict_agree": isinstance(attestation, Mapping)
        and attestation.get("verdict") == "AGREE"
        and review_verdict,
        "review_strength": isinstance(attestation, Mapping)
        and attestation.get("review_strength")
        in {"claude_opus_max", "codex_substitute_weaker"},
        "timestamp_recorded": isinstance(attestation, Mapping)
        and isinstance(attestation.get("timestamp_utc"), str)
        and bool(attestation["timestamp_utc"]),
    }


def validate_phase6_runtime_authority(
    proposal_path: Path,
    attestation_path: Path,
    *,
    expected_gate: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal = read_strict_json(proposal_path)
    attestation = read_strict_json(attestation_path)
    validate_phase6_budget_proposal(proposal, expected_gate=expected_gate)
    checks = phase6_attestation_checks(
        attestation,
        proposal_path=proposal_path,
        expected_gate=expected_gate,
    )
    if not all(checks.values()):
        raise ContractError(f"invalid Phase 6 runtime attestation: {checks}")
    return proposal, attestation


def graphdef_bytes_record(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > PHASE6_GRAPHDEF_MAX_DECODED_BYTES:
        raise ContractError("GraphDef bytes violate Phase 6 size limits")
    return {
        "encoding": "base64-rfc4648",
        "decoded_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def decode_graphdef_bytes_record(
    record: Mapping[str, Any],
    *,
    prior_total_decoded_bytes: int = 0,
) -> bytes:
    if set(record) != {"encoding", "decoded_bytes", "sha256", "base64"}:
        raise ContractError("GraphDef byte record fields do not match closed schema")
    encoded = record.get("base64")
    declared = record.get("decoded_bytes")
    if (
        record.get("encoding") != "base64-rfc4648"
        or type(declared) is not int
        or declared <= 0
        or declared > PHASE6_GRAPHDEF_MAX_DECODED_BYTES
        or prior_total_decoded_bytes < 0
        or prior_total_decoded_bytes + declared > PHASE6_GRAPHDEF_MAX_TOTAL_DECODED_BYTES
        or not isinstance(encoded, str)
        or len(encoded) != 4 * math.ceil(declared / 3)
        or not re.fullmatch(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", encoded)
        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256")))
    ):
        raise ContractError("GraphDef base64 pre-decode contract failed")
    digest = hashlib.sha256()
    decoded_count = 0
    with tempfile.SpooledTemporaryFile(max_size=PHASE6_GRAPHDEF_MAX_DECODED_BYTES) as spool:
        for offset in range(0, len(encoded), PHASE6_GRAPHDEF_DECODE_CHUNK_CHARS):
            chunk = encoded[offset : offset + PHASE6_GRAPHDEF_DECODE_CHUNK_CHARS]
            try:
                decoded = base64.b64decode(chunk, validate=True)
            except binascii.Error as exc:
                raise ContractError(f"invalid canonical GraphDef base64: {exc}") from exc
            decoded_count += len(decoded)
            if decoded_count > declared or decoded_count > PHASE6_GRAPHDEF_MAX_DECODED_BYTES:
                raise ContractError("GraphDef decoder exceeded declared or hard byte limit")
            digest.update(decoded)
            spool.write(decoded)
        if decoded_count != declared or digest.hexdigest() != record["sha256"]:
            raise ContractError("GraphDef decoded length or digest mismatch")
        spool.seek(0)
        raw = spool.read(declared + 1)
    if len(raw) != declared or base64.b64encode(raw).decode("ascii") != encoded:
        raise ContractError("GraphDef base64 is not canonical")
    return raw


def _protobuf_scalar_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite GraphDef protobuf scalar")
        return value
    return value


def _protobuf_tokens(message: Any) -> tuple[list[dict[str, Any]], dict[str, tuple[Any, ...]]]:
    tokens: list[dict[str, Any]] = []
    accessors: dict[str, tuple[Any, ...]] = {}

    def visit(current: Any, path: str, segments: tuple[Any, ...], inside_const: bool) -> None:
        descriptor = getattr(current, "DESCRIPTOR", None)
        if descriptor is None:
            raise ContractError("value is not a protobuf message")
        current_const = inside_const or (
            descriptor.full_name == "tensorflow.NodeDef" and getattr(current, "op", None) == "Const"
        )
        for field, value in current.ListFields():
            field_path = f"{path}.{field.name}"
            if field.is_repeated:
                if field.message_type is not None and field.message_type.GetOptions().map_entry:
                    for key in sorted(value):
                        item = value[key]
                        key_path = f"{field_path}[{strict_json_dumps(key)}]"
                        key_segments = segments + (("map", field.name, key),)
                        if field.message_type.fields_by_name["value"].message_type is not None:
                            visit(item, key_path, key_segments, current_const)
                        else:
                            tokens.append(
                                {
                                    "path": key_path,
                                    "kind": field.message_type.fields_by_name["value"].type,
                                    "value": _protobuf_scalar_value(item),
                                    "inside_const": current_const,
                                }
                            )
                            accessors[key_path] = key_segments
                else:
                    for index, item in enumerate(value):
                        item_path = f"{field_path}[{index}]"
                        item_segments = segments + (("index", field.name, index),)
                        if field.message_type is not None:
                            visit(item, item_path, item_segments, current_const)
                        else:
                            tokens.append(
                                {
                                    "path": item_path,
                                    "kind": field.type,
                                    "value": _protobuf_scalar_value(item),
                                    "inside_const": current_const,
                                }
                            )
                            accessors[item_path] = item_segments
            elif field.message_type is not None:
                visit(value, field_path, segments + (("field", field.name),), current_const)
            else:
                tokens.append(
                    {
                        "path": field_path,
                        "kind": field.type,
                        "value": _protobuf_scalar_value(value),
                        "inside_const": current_const,
                    }
                )
                accessors[field_path] = segments + (("field", field.name),)

    visit(message, "$", (), False)
    return tokens, accessors


def graphdef_token_stream(raw: bytes) -> list[dict[str, Any]]:
    from tensorflow.core.framework import graph_pb2

    graph = graph_pb2.GraphDef()
    try:
        graph.ParseFromString(raw)
    except Exception as exc:
        raise ContractError(f"cannot parse GraphDef bytes: {exc}") from exc
    tokens, _ = _protobuf_tokens(graph)
    return tokens


def _set_protobuf_scalar(message: Any, segments: tuple[Any, ...], value: int) -> None:
    current = message
    for kind, field_name, *selector in segments[:-1]:
        if kind == "field":
            current = getattr(current, field_name)
        elif kind == "index":
            current = getattr(current, field_name)[selector[0]]
        elif kind == "map":
            current = getattr(current, field_name)[selector[0]]
        else:  # pragma: no cover - internal invariant.
            raise ContractError("unknown protobuf accessor segment")
    kind, field_name, *selector = segments[-1]
    if kind == "field":
        setattr(current, field_name, value)
    elif kind == "index":
        getattr(current, field_name)[selector[0]] = value
    elif kind == "map":
        getattr(current, field_name)[selector[0]] = value
    else:  # pragma: no cover - internal invariant.
        raise ContractError("unknown protobuf accessor segment")


def compare_graphdef_cohort(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from tensorflow.core.framework import graph_pb2

    if len(records) != 6:
        raise ContractError("GraphDef cohort must contain exactly six P/B records")
    identities: list[Mapping[str, Any]] = []
    graphs = []
    token_maps: list[dict[str, dict[str, Any]]] = []
    accessor_maps: list[dict[str, tuple[Any, ...]]] = []
    total = 0
    for record in records:
        identity = record.get("identity")
        graph_record = record.get("graphdef_bytes")
        if not _phase6_identity_valid(identity) or not isinstance(graph_record, Mapping):
            raise ContractError("invalid GraphDef cohort record")
        raw = decode_graphdef_bytes_record(graph_record, prior_total_decoded_bytes=total)
        total += len(raw)
        graph = graph_pb2.GraphDef()
        graph.ParseFromString(raw)
        tokens, accessors = _protobuf_tokens(graph)
        identities.append(identity)
        graphs.append(graph)
        token_maps.append({token["path"]: token for token in tokens})
        accessor_maps.append(accessors)
    fixed_fields = {
        (
            identity["dimension"],
            identity["dtype"],
            identity["method_id"],
            identity["operation"],
        )
        for identity in identities
    }
    identity_ids = {identity["identity_id"] for identity in identities}
    if (
        len(fixed_fields) != 1
        or len(identity_ids) != 6
        or next(iter(fixed_fields))[1] != "float32"
        or next(iter(fixed_fields))[2] not in PRIMARY_METHOD_IDS
        or next(iter(fixed_fields))[3] != "trace"
    ):
        raise ContractError("GraphDef cohort must fix dimension/dtype/method/operation")
    pairs = {(identity["parameter_count"], identity["batch_size"]) for identity in identities}
    if pairs != {(p, b) for p in (50, 150) for b in (1, 4, 16)}:
        raise ContractError("GraphDef cohort does not cover the exact P/B lattice")
    path_sets = [set(token_map) for token_map in token_maps]
    all_paths = sorted(set().union(*path_sets))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sentinel_by_axis = {"B": -1000001, "P": -1000002}
    for path in all_paths:
        tokens = [token_map.get(path) for token_map in token_maps]
        values = [token.get("value") if token is not None else None for token in tokens]
        if all(token == tokens[0] for token in tokens[1:]):
            continue
        kinds = {token.get("kind") for token in tokens if token is not None}
        inside_const = any(token is not None and token.get("inside_const") for token in tokens)
        axis: str | None = None
        if (
            all(token is not None for token in tokens)
            and len(kinds) == 1
            and not inside_const
            and path.endswith(".size")
            and ".shape" in path
        ):
            if all(value == identity["batch_size"] for value, identity in zip(values, identities)):
                axis = "B"
            elif all(value == identity["parameter_count"] for value, identity in zip(values, identities)):
                axis = "P"
        difference = {
            "path": path,
            "values": values,
            "axis": axis,
            "inside_const": inside_const,
            "rule_id": f"static_shape_dimension_{axis}" if axis else "rejected_unclassified_difference",
        }
        if axis is None:
            rejected.append(difference)
            continue
        accepted.append(difference)
        for graph, accessors in zip(graphs, accessor_maps):
            _set_protobuf_scalar(graph, accessors[path], sentinel_by_axis[axis])
    normalized = [graph.SerializeToString(deterministic=True) for graph in graphs]
    normalized_equal = all(raw == normalized[0] for raw in normalized[1:])
    if not normalized_equal:
        rejected.append(
            {
                "path": "$.__canonical_graphdef_bytes__",
                "values": [hashlib.sha256(raw).hexdigest() for raw in normalized],
                "axis": None,
                "inside_const": False,
                "rule_id": "rejected_canonical_bytes_mismatch",
            }
        )
    return {
        "accepted_differences": accepted,
        "rejected_differences": rejected,
        "normalized_graphdef_sha256": hashlib.sha256(normalized[0]).hexdigest(),
        "normalized_graphdefs_equal": normalized_equal,
        "passed": not rejected and normalized_equal,
    }


def evaluate_phase6_trace_census(payload: Mapping[str, Any]) -> dict[str, Any]:
    ledger_checks = phase6_ledger_checks(payload, final=True)
    records = payload.get("records") if isinstance(payload, Mapping) else None
    terminal_records = isinstance(records, list) and len(records) == 36
    total_declared = 0
    graph_records: list[dict[str, Any]] = []
    child_semantics = terminal_records
    if terminal_records:
        for record in records:
            if record.get("state") != "passed":
                child_semantics = False
                continue
            child = record.get("evidence", {}).get("child_artifact", {}).get("strict_json")
            child_evidence = child.get("evidence") if isinstance(child, Mapping) else None
            graph_record = child_evidence.get("graphdef_bytes") if isinstance(child_evidence, Mapping) else None
            if not isinstance(graph_record, Mapping) or type(graph_record.get("decoded_bytes")) is not int:
                child_semantics = False
                continue
            total_declared += graph_record["decoded_bytes"]
            graph_records.append(
                {"identity": record["identity"], "graphdef_bytes": graph_record}
            )
    total_cap = 0 < total_declared <= PHASE6_GRAPHDEF_MAX_TOTAL_DECODED_BYTES
    all_decoded = child_semantics and total_cap
    decoded_total = 0
    if all_decoded:
        for graph_record in graph_records:
            try:
                raw = decode_graphdef_bytes_record(
                    graph_record["graphdef_bytes"],
                    prior_total_decoded_bytes=decoded_total,
                )
            except ContractError:
                all_decoded = False
                break
            decoded_total += len(raw)
    cohorts: list[dict[str, Any]] = []
    cohort_passed = all_decoded
    if all_decoded:
        for dimension in (10, 20, 30):
            for method in PRIMARY_METHOD_IDS:
                cohort_records = [
                    record
                    for record in graph_records
                    if record["identity"]["dimension"] == dimension
                    and record["identity"]["method_id"] == method
                ]
                try:
                    result = compare_graphdef_cohort(cohort_records)
                except ContractError as exc:
                    result = {
                        "accepted_differences": [],
                        "rejected_differences": [
                            {
                                "path": "$.__cohort_evaluator__",
                                "values": [str(exc)],
                                "axis": None,
                                "inside_const": False,
                                "rule_id": "rejected_cohort_contract_error",
                            }
                        ],
                        "normalized_graphdef_sha256": None,
                        "normalized_graphdefs_equal": False,
                        "passed": False,
                    }
                cohorts.append(
                    {"dimension": dimension, "method_id": method, "comparison": result}
                )
                cohort_passed = cohort_passed and result["passed"]
    trace_common_valid = (
        all(ledger_checks.values())
        and child_semantics
        and total_cap
        and all_decoded
        and len(cohorts) == 6
        and cohort_passed
    )
    return {
        "ledger_checks": ledger_checks,
        "record_count": len(records) if isinstance(records, list) else None,
        "declared_decoded_bytes": total_declared,
        "decoded_bytes": decoded_total,
        "global_decoded_cap_passed": total_cap and all_decoded,
        "cohorts": cohorts,
        "trace_common_valid": trace_common_valid,
    }


def directed_float32_comparison(
    candidate: Any,
    reference: Any,
    *,
    expected_shape: Sequence[int],
    output_kind: str,
) -> dict[str, Any]:
    if output_kind not in {"value", "score"}:
        raise ContractError("directed comparison output kind must be value or score")
    if not expected_shape or any(type(value) is not int or value <= 0 for value in expected_shape):
        raise ContractError("directed comparison requires a nonempty positive expected shape")
    tolerance = DIRECT_PARITY_TOLERANCES["float32"]

    def flatten(value: Any) -> tuple[list[int], list[float]]:
        if isinstance(value, list):
            child_rows = [flatten(item) for item in value]
            if not child_rows:
                return [0], []
            child_shapes = [shape for shape, _ in child_rows]
            if any(shape != child_shapes[0] for shape in child_shapes[1:]):
                raise ContractError("ragged numeric output")
            return [len(value), *child_shapes[0]], [number for _, row in child_rows for number in row]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ContractError("non-finite or non-numeric output")
        return [], [float(value)]

    candidate_shape, candidate_values = flatten(candidate)
    reference_shape, reference_values = flatten(reference)
    shape_match = candidate_shape == reference_shape == list(expected_shape)
    residuals = (
        [abs(left - right) for left, right in zip(candidate_values, reference_values)]
        if shape_match
        else []
    )
    rtol = tolerance[f"{output_kind}_rtol"]
    atol = tolerance[f"{output_kind}_atol"]
    passed = shape_match and bool(candidate_values) and len(candidate_values) == len(reference_values) and all(
        residual <= atol + rtol * abs(right)
        for residual, right in zip(residuals, reference_values)
    )
    return {
        "passed": passed,
        "shape": candidate_shape if shape_match else None,
        "candidate_shape": candidate_shape,
        "reference_shape": reference_shape,
        "output_kind": output_kind,
        "expected_shape": list(expected_shape),
        "rtol": rtol,
        "atol": atol,
        "max_abs_residual": max(residuals, default=0.0),
        "reference_max_abs": max((abs(value) for value in reference_values), default=0.0),
    }


def _phase6_passed_child(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if record.get("state") != "passed":
        return None
    child = record.get("evidence", {}).get("child_artifact", {}).get("strict_json")
    return child if isinstance(child, Mapping) else None


def _phase6_compare_children(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    batch_size: int,
    parameter_count: int,
) -> dict[str, Any]:
    candidate_outputs = candidate.get("outputs")
    reference_outputs = reference.get("outputs")
    if not isinstance(candidate_outputs, Mapping) or not isinstance(reference_outputs, Mapping):
        raise ContractError("Phase 6 numerical comparison requires embedded outputs")
    value = directed_float32_comparison(
        candidate_outputs.get("value"),
        reference_outputs.get("value"),
        expected_shape=[batch_size],
        output_kind="value",
    )
    score = directed_float32_comparison(
        candidate_outputs.get("score"),
        reference_outputs.get("score"),
        expected_shape=[batch_size, parameter_count],
        output_kind="score",
    )
    return {"value": value, "score": score, "passed": value["passed"] and score["passed"]}


def phase45_common_correctness_valid(bindings: Mapping[str, Any]) -> bool:
    blobs = bindings.get("phase45_evidence") if isinstance(bindings, Mapping) else None
    if not isinstance(blobs, list) or len(blobs) != 2:
        return False
    repo_root = Path(__file__).resolve().parents[1]
    expected_blob_identity = {
        (str((repo_root / relative).resolve()), digest)
        for relative, digest in PHASE6_PHASE45_EVIDENCE.items()
    }
    observed_blob_identity = {
        (blob.get("path"), blob.get("sha256"))
        for blob in blobs
        if isinstance(blob, Mapping)
    }
    if observed_blob_identity != expected_blob_identity:
        return False
    by_schema = {
        blob.get("strict_json", {}).get("schema"): blob.get("strict_json")
        for blob in blobs
        if isinstance(blob, Mapping) and isinstance(blob.get("strict_json"), Mapping)
    }
    phase4 = by_schema.get("bayesfilter.kalman_qr_batched_xla_repair.phase4.autodiff.v1")
    phase5 = by_schema.get("bayesfilter.kalman_qr_batched_xla_repair.phase5.measurement.v1")
    return (
        isinstance(phase4, Mapping)
        and phase4.get("state") == "passed"
        and isinstance(phase4.get("checks"), Mapping)
        and bool(phase4["checks"])
        and all(value is True for value in phase4["checks"].values())
        and isinstance(phase5, Mapping)
        and phase5.get("state") == "passed"
        and isinstance(phase5.get("checks"), Mapping)
        and bool(phase5["checks"])
        and all(value is True for value in phase5["checks"].values())
    )


def evaluate_phase6_scalar_status(
    scalar_payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
) -> dict[str, Any]:
    scalar_checks = phase6_ledger_checks(scalar_payload, final=True)
    final_checks = phase6_ledger_checks(final_payload, final=True)
    ledgers_valid = all(scalar_checks.values()) and all(final_checks.values())
    scalar_records = scalar_payload.get("records") if isinstance(scalar_payload, Mapping) else None
    final_records = final_payload.get("records") if isinstance(final_payload, Mapping) else None
    if (
        not ledgers_valid
        or not isinstance(scalar_records, list)
        or not isinstance(final_records, list)
    ):
        return {
            "target_scalar_status": "failed_common_or_cpu_xla_backend_unlocalized",
            "comparisons": [],
            "scalar_ledger_checks": scalar_checks,
            "final_ledger_checks": final_checks,
        }
    scalar_by_key = {
        (record["identity"]["batch_size"], record["identity"]["method_id"]): record
        for record in scalar_records
    }
    final_by_key = {
        (
            record["identity"]["dimension"],
            record["identity"]["parameter_count"],
            record["identity"]["batch_size"],
            record["identity"]["method_id"],
        ): record
        for record in final_records
    }
    comparisons: list[dict[str, Any]] = []
    scalar_disagreement = False
    method_disagreements: set[str] = set()
    missing = False
    all_scalar_timed_out = True
    for batch_size in (1, 4):
        analytic_reference_record = scalar_by_key.get((batch_size, REFERENCE_METHOD_IDS[0]))
        autodiff_reference_record = scalar_by_key.get((batch_size, REFERENCE_METHOD_IDS[1]))
        if analytic_reference_record is None or autodiff_reference_record is None:
            missing = True
            all_scalar_timed_out = False
            continue
        all_scalar_timed_out = all_scalar_timed_out and (
            analytic_reference_record.get("state") == "timed_out"
            and autodiff_reference_record.get("state") == "timed_out"
        )
        analytic_reference = _phase6_passed_child(analytic_reference_record)
        autodiff_reference = _phase6_passed_child(autodiff_reference_record)
        if analytic_reference is None or autodiff_reference is None:
            missing = True
            continue
        scalar_pair = _phase6_compare_children(
            analytic_reference,
            autodiff_reference,
            batch_size=batch_size,
            parameter_count=50,
        )
        comparisons.append(
            {
                "batch_size": batch_size,
                "kind": "scalar_reference_pair",
                "candidate_method": REFERENCE_METHOD_IDS[0],
                "reference_method": REFERENCE_METHOD_IDS[1],
                "comparison": scalar_pair,
            }
        )
        if not scalar_pair["passed"]:
            scalar_disagreement = True
            continue
        for primary_method, reference_method in zip(
            PRIMARY_METHOD_IDS, REFERENCE_METHOD_IDS, strict=True
        ):
            primary_record = final_by_key.get((10, 50, batch_size, primary_method))
            primary = _phase6_passed_child(primary_record) if primary_record is not None else None
            reference = (
                analytic_reference
                if reference_method == REFERENCE_METHOD_IDS[0]
                else autodiff_reference
            )
            if primary is None:
                missing = True
                continue
            comparison = _phase6_compare_children(
                primary,
                reference,
                batch_size=batch_size,
                parameter_count=50,
            )
            comparisons.append(
                {
                    "batch_size": batch_size,
                    "kind": "batch_to_scalar_reference",
                    "candidate_method": primary_method,
                    "reference_method": reference_method,
                    "comparison": comparison,
                }
            )
            if not comparison["passed"]:
                method_disagreements.add(primary_method)
    if scalar_disagreement:
        status = "failed_scalar_reference_disagreement_unlocalized"
    elif len(method_disagreements) == 2:
        status = "failed_common_or_cpu_xla_backend_unlocalized"
    elif len(method_disagreements) == 1:
        status = f"failed_method_local:{next(iter(method_disagreements))}"
    elif missing:
        status = "not_checked_timeout" if all_scalar_timed_out else "partial_missing_evidence"
    else:
        status = "passed"
    return {
        "target_scalar_status": status,
        "comparisons": comparisons,
        "scalar_ledger_checks": scalar_checks,
        "final_ledger_checks": final_checks,
    }


def _phase6_handoff_from_derived(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "phase45_common_correctness_valid",
        "dependency_provenance_valid",
        "trace_common_valid",
        "cpu_xla_common_invalidity",
        "target_scalar_status",
        "cpu_xla_lane_local_only",
        "fair_pair_cells",
        "valid_trace_cohorts",
    }
    consistent = isinstance(raw, Mapping) and set(raw) == required
    if consistent:
        consistent = all(
            type(raw[field]) is bool
            for field in (
                "phase45_common_correctness_valid",
                "dependency_provenance_valid",
                "trace_common_valid",
                "cpu_xla_common_invalidity",
                "cpu_xla_lane_local_only",
            )
        ) and isinstance(raw.get("fair_pair_cells"), list) and isinstance(
            raw.get("valid_trace_cohorts"), list
        )
    allowed_scalar_statuses = {
        "passed",
        "partial_missing_evidence",
        "not_checked_timeout",
        "failed_scalar_reference_disagreement_unlocalized",
        "failed_common_or_cpu_xla_backend_unlocalized",
        *{f"failed_method_local:{method}" for method in PRIMARY_METHOD_IDS},
    }
    exact_cell_fields = {"dimension", "parameter_count", "batch_size", "dtype"}
    allowed_lattice = {
        (dimension, parameter_count, batch_size, "float32")
        for dimension in (10, 20, 30)
        for parameter_count in (50, 150)
        for batch_size in (1, 4, 16)
    }

    def exact_cell(cell: Any) -> bool:
        return (
            isinstance(cell, Mapping)
            and set(cell) == exact_cell_fields
            and (
                cell["dimension"],
                cell["parameter_count"],
                cell["batch_size"],
                cell["dtype"],
            )
            in allowed_lattice
        )

    def exact_fair_cell(cell: Any) -> bool:
        return (
            isinstance(cell, Mapping)
            and set(cell) == {*exact_cell_fields, "completed_scalar_comparisons"}
            and type(cell["completed_scalar_comparisons"]) is bool
            and exact_cell({field: cell[field] for field in exact_cell_fields})
        )

    if consistent:
        consistent = (
            raw.get("target_scalar_status") in allowed_scalar_statuses
            and all(exact_fair_cell(cell) for cell in raw["fair_pair_cells"])
            and all(exact_cell(cell) for cell in raw["valid_trace_cohorts"])
            and len(
                {
                    tuple(cell[field] for field in sorted(exact_cell_fields))
                    for cell in raw["valid_trace_cohorts"]
                }
            )
            == len(raw["valid_trace_cohorts"])
        )
    scalar_status = raw.get("target_scalar_status") if isinstance(raw, Mapping) else None
    scope = "blocked"
    selected: Mapping[str, Any] | None = None
    blockers = (
        not consistent
        or not raw.get("phase45_common_correctness_valid", False)
        or not raw.get("dependency_provenance_valid", False)
        or not raw.get("trace_common_valid", False)
        or raw.get("cpu_xla_common_invalidity") is True
        or scalar_status
        in {
            "failed_scalar_reference_disagreement_unlocalized",
            "failed_common_or_cpu_xla_backend_unlocalized",
        }
        or isinstance(scalar_status, str)
        and scalar_status.startswith("failed_method_local:")
    )
    if not blockers:
        fair_cells = [
            cell
            for cell in raw["fair_pair_cells"]
            if isinstance(cell, Mapping)
            and cell.get("dimension") == 10
            and cell.get("parameter_count") == 50
            and cell.get("batch_size") in {1, 4}
            and cell.get("dtype") == "float32"
            and cell.get("completed_scalar_comparisons") is True
        ]
        if scalar_status == "passed" and fair_cells:
            scope = "target_numerical_gate"
            selected_fair = min(
                fair_cells,
                key=lambda cell: (
                    cell["dimension"],
                    cell["parameter_count"],
                    cell["batch_size"],
                    cell["dtype"],
                ),
            )
            selected = {field: selected_fair[field] for field in exact_cell_fields}
        elif scalar_status in {"passed", "partial_missing_evidence", "not_checked_timeout"} and raw[
            "cpu_xla_lane_local_only"
        ]:
            valid = [
                cell
                for cell in raw["valid_trace_cohorts"]
                if exact_cell(cell)
            ]
            canonical = [
                cell
                for cell in valid
                if cell
                == {
                    "dimension": 10,
                    "parameter_count": 50,
                    "batch_size": 1,
                    "dtype": "float32",
                }
            ]
            if valid:
                scope = "diagnostic_smallest_gpu_only"
                selected = canonical[0] if canonical else min(
                    valid,
                    key=lambda cell: (
                        cell["dimension"],
                        cell["parameter_count"],
                        cell["batch_size"],
                        cell["dtype"],
                    ),
                )
    return {
        "schema": PHASE6_HANDOFF_SCHEMA,
        "phase7_scope": scope,
        "selected_phase7_cell": dict(selected) if selected is not None else None,
        "phase7_expansion_authorized": False,
        "phase7_nonclaims": list(PHASE7_NONCLAIMS),
    }


def evaluate_phase6_handoff(final_payload: Mapping[str, Any]) -> dict[str, Any]:
    final_checks = phase6_ledger_checks(final_payload, final=True)
    bindings = final_payload.get("bindings") if isinstance(final_payload, Mapping) else None
    authority_inputs = (
        bindings.get("authority_inputs") if isinstance(bindings, Mapping) else None
    )
    runtime_predecessors = (
        bindings.get("runtime_predecessors") if isinstance(bindings, Mapping) else None
    )
    by_schema: dict[str, Mapping[str, Any]] = {}
    if isinstance(authority_inputs, list):
        for blob in authority_inputs:
            payload = blob.get("strict_json") if isinstance(blob, Mapping) else None
            schema = payload.get("schema") if isinstance(payload, Mapping) else None
            if isinstance(schema, str) and schema not in by_schema:
                by_schema[schema] = payload
            elif isinstance(schema, str):
                by_schema.clear()
                break
    trace_payload = by_schema.get(PHASE6_TRACE_SCHEMA)
    scalar_payload = None
    if isinstance(runtime_predecessors, list) and len(runtime_predecessors) == 1:
        predecessor = runtime_predecessors[0]
        if isinstance(predecessor, Mapping):
            artifact = predecessor.get("artifact")
            candidate = artifact.get("strict_json") if isinstance(artifact, Mapping) else None
            if isinstance(candidate, Mapping) and candidate.get("schema") == PHASE6_SCALAR_SCHEMA:
                scalar_payload = candidate
    trace_evaluation = (
        evaluate_phase6_trace_census(trace_payload)
        if isinstance(trace_payload, Mapping)
        else {"trace_common_valid": False, "cohorts": []}
    )
    scalar_evaluation = (
        evaluate_phase6_scalar_status(scalar_payload, final_payload)
        if isinstance(scalar_payload, Mapping)
        else {
            "target_scalar_status": "failed_common_or_cpu_xla_backend_unlocalized",
            "comparisons": [],
        }
    )
    records = final_payload.get("records") if isinstance(final_payload, Mapping) else None
    fair_pair_cells: list[dict[str, Any]] = []
    cpu_xla_common_invalidity = True
    cpu_xla_lane_local_only = False
    if isinstance(records, list) and all(final_checks.values()):
        cpu_xla_common_invalidity = any(
            record.get("evidence", {}).get("classification") == "common_invalidity"
            for record in records
            if record.get("state") in PHASE6_TERMINAL_STATES
        )
        launched = [record for record in records if record.get("state") in PHASE6_TERMINAL_STATES]
        nonpasses = [record for record in launched if record.get("state") != "passed"]
        lane_local_classes = {
            "cpu_backend_or_method_failure",
            "method_local_failure",
            "cpu_backend_or_cell_timeout",
        }
        cpu_xla_lane_local_only = (
            bool(launched)
            and not cpu_xla_common_invalidity
            and all(
                record.get("evidence", {}).get("classification") in lane_local_classes
                for record in nonpasses
            )
        )
        by_cell: dict[tuple[int, int, int, str], dict[str, Mapping[str, Any]]] = {}
        for record in records:
            identity = record["identity"]
            key = (
                identity["dimension"],
                identity["parameter_count"],
                identity["batch_size"],
                identity["dtype"],
            )
            by_cell.setdefault(key, {})[identity["method_id"]] = record
        completed_scalar_batches = {
            row["batch_size"]
            for row in scalar_evaluation.get("comparisons", [])
            if row.get("kind") == "batch_to_scalar_reference"
            and row.get("comparison", {}).get("passed") is True
        }
        for key, methods in sorted(by_cell.items()):
            if set(methods) != set(PRIMARY_METHOD_IDS):
                continue
            children = [_phase6_passed_child(methods[method]) for method in PRIMARY_METHOD_IDS]
            if any(child is None for child in children):
                continue
            try:
                parity = _phase6_compare_children(
                    children[0],
                    children[1],
                    batch_size=key[2],
                    parameter_count=key[1],
                )
            except ContractError:
                continue
            if parity["passed"]:
                fair_pair_cells.append(
                    {
                        "dimension": key[0],
                        "parameter_count": key[1],
                        "batch_size": key[2],
                        "dtype": key[3],
                        "completed_scalar_comparisons": key[0] == 10
                        and key[1] == 50
                        and key[2] in completed_scalar_batches,
                    }
                )
    valid_trace_cohorts = []
    if trace_evaluation.get("trace_common_valid") is True:
        valid_trace_cohorts = [
            {
                "dimension": dimension,
                "parameter_count": parameter_count,
                "batch_size": batch_size,
                "dtype": "float32",
            }
            for dimension in (10, 20, 30)
            for parameter_count in (50, 150)
            for batch_size in (1, 4, 16)
        ]
    derived = {
        "phase45_common_correctness_valid": isinstance(bindings, Mapping)
        and phase45_common_correctness_valid(bindings),
        "dependency_provenance_valid": all(final_checks.values()),
        "trace_common_valid": trace_evaluation.get("trace_common_valid") is True,
        "cpu_xla_common_invalidity": cpu_xla_common_invalidity,
        "target_scalar_status": scalar_evaluation["target_scalar_status"],
        "cpu_xla_lane_local_only": cpu_xla_lane_local_only,
        "fair_pair_cells": fair_pair_cells,
        "valid_trace_cohorts": valid_trace_cohorts,
    }
    handoff = _phase6_handoff_from_derived(derived)
    return {
        **handoff,
        "derived_inputs": derived,
        "trace_evaluation": trace_evaluation,
        "scalar_evaluation": scalar_evaluation,
        "final_ledger_checks": final_checks,
    }
