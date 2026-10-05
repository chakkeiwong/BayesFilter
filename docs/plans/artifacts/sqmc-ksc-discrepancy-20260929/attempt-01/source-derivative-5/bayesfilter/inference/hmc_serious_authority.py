"""Fail-closed one-use authority for serious typed-identity Phase 7 HMC."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
import fcntl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.inference.hmc_identity_adoption import (
    build_phase5_artifact_reference,
    parse_phase5_artifact_reference,
    parse_phase5_preflight_report,
    verify_phase5_artifact_reference,
    verify_phase5_output_manifest,
)
from bayesfilter.inference.hmc_smoke_authority import (
    ADOPTION_RECORD_PATH,
    PHASE5_MANIFEST_PATH,
    PREFLIGHT_PATH,
    REPO_ROOT,
    V2_CONFIG_PATH,
    ConsumedAttempt1EvidenceSession,
    ConsumedAttempt1EvidenceDriftError,
    AUTHORITY_PATH as PHASE6_AUTHORITY_PATH,
    CLAIM_PATH as PHASE6_CLAIM_PATH,
    INFRASTRUCTURE_FAILURE_PATH as PHASE6_INFRASTRUCTURE_FAILURE_PATH,
    INFRASTRUCTURE_MANIFEST_PATH as PHASE6_INFRASTRUCTURE_MANIFEST_PATH,
    LOG_PATH as PHASE6_LOG_PATH,
    OUTPUT_MANIFEST_PATH as PHASE6_TERMINAL_MANIFEST_PATH,
    PRIVATE_SAMPLES_PATH as PHASE6_PRIVATE_SAMPLES_PATH,
    PROPOSAL_MANIFEST_PATH as PHASE6_PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH as PHASE6_PROPOSAL_PATH,
    PinnedSmokeOutputDirectories,
    SecureSmokeOutputSession,
    SmokeOutputReservationError,
    artifact_file_sha256,
    build_file_reference,
    build_verified_implementation_source_bundle,
    create_durable_launch_claim_with_consumed_evidence,
    implementation_source_bundle_hash,
    parse_file_reference,
    verify_consumed_attempt1_evidence,
    parse_launch_claim,
    parse_smoke_authority,
    parse_smoke_authority_proposal,
    parse_smoke_authority_proposal_manifest,
    parse_smoke_output_manifest,
    parse_smoke_progress,
    parse_smoke_terminal_result,
    verify_file_reference,
    verify_implementation_reference_inventory,
    write_phase6_json,
)


HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_historical_result_archive_manifest.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_authority_proposal.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_authority_proposal_manifest.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_authority.v1"
)
HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_launch_claim.v1"
)
HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_result.v1"
)
HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_failure.v1"
)
HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_progress.v1"
)
HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_output_manifest.v1"
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_infrastructure_failure.v1"
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase7_serious_infrastructure_manifest.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_authority_proposal.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_authority_proposal_manifest.v1"
)
HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_authority.v1"
)
HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_launch_claim.v1"
)
HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_result.v1"
)
HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_failure.v1"
)
HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_progress.v1"
)
HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_output_manifest.v1"
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_infrastructure_failure.v1"
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V2 = (
    "bayesfilter.hmc_phase7_serious_attempt2_infrastructure_manifest.v1"
)

# Keep the established public symbol names as the active controller contract.
# The consumed attempt-1 parsers below use the explicit legacy aliases.
ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1 = HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1
ATTEMPT1_HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1 = HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1
ATTEMPT1_HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1 = HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1
ATTEMPT1_HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1
)
ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V2
)
HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V2
)
HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1 = HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V2
HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V2
)
HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1 = HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V2
HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1 = HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V2
HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1 = HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V2
HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V2
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V2
)
HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1 = (
    HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V2
)

SERIOUS_AUTHORITY_DECISION = (
    "AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS_ATTEMPT2"
)
ATTEMPT1_SERIOUS_AUTHORITY_DECISION = (
    "AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS"
)
SERIOUS_AUTHORITY_STATUS_PENDING = "pending_human_serious_approval"
SERIOUS_AUTHORITY_STATUS_APPROVED = "approved_one_serious_launch_only"
SERIOUS_PASS_DECISION = "PASS_PHASE7_SERIOUS_ATTEMPT2_TO_PHASE8_APPROVAL_BOUNDARY"
SERIOUS_BLOCK_DECISION = "BLOCK_PHASE7_SERIOUS_ATTEMPT2"
SERIOUS_INFRASTRUCTURE_BLOCK_DECISION = (
    "BLOCK_PHASE7_SERIOUS_ATTEMPT2_LAUNCHER_INFRASTRUCTURE"
)
ATTEMPT1_SERIOUS_PASS_DECISION = "PASS_PHASE7_SERIOUS_TO_PHASE8_APPROVAL_BOUNDARY"
ATTEMPT1_SERIOUS_BLOCK_DECISION = "BLOCK_PHASE7_SERIOUS"
ATTEMPT1_SERIOUS_INFRASTRUCTURE_BLOCK_DECISION = (
    "BLOCK_PHASE7_SERIOUS_LAUNCHER_INFRASTRUCTURE"
)
TRANSITION_IDENTITY_HASH = (
    "sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a"
)
SERIOUS_EXECUTION_IDENTITY_HASH = (
    "sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4"
)
SERIOUS_CONFIG_HASH = (
    "sha256:79bcaa2b5977cadeb14607d6256e2eda31efb63d9d9a69d3008603fe14e3a450"
)
PHASE5_PREFLIGHT_ARTIFACT_HASH = (
    "sha256:e8e13bd7c7fc635424dd4401cf835dae6367f1be37386f3df45caaa3ef4a497e"
)
SERIOUS_PARAMETER_NAMES = (
    "a11_raw",
    "a22_raw",
    "a33_raw",
    "a44_raw",
    "a21_raw",
    "a31_raw",
    "a32_raw",
    "a41_raw",
    "a42_raw",
    "a43_raw",
    "log_q1",
    "log_q2",
    "log_q3",
    "log_q4",
    "log_r1",
    "log_r2",
    "log_r3",
    "log_r4",
)
SERIOUS_DIAGNOSTIC_DEFINITIONS = {
    "rank_transform": "Blom average-rank normal score",
    "rhat": "max(rank-normalized split R-hat, folded rank-normalized split R-hat)",
    "bulk_ess": "split-chain cross-chain ESS of rank-normalized draws",
    "tail_ess": "minimum split-chain cross-chain ESS of pooled q05/q95 indicators",
    "autocorrelation_truncation": "TFP initial positive pairs",
    "quantile_interpolation": "linear",
}
SERIOUS_DIAGNOSTIC_NONCLAIMS = (
    "all-parameter HMC convergence screen only",
    "no posterior recovery claim",
    "no sampler superiority claim",
    "no production or default readiness claim",
)

PUBLIC_ROOT = REPO_ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
PHASE7_SUBPLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase7-serious-attempt2-repair-subplan-2026-07-13.md"
)
ATTEMPT1_PHASE7_SUBPLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase7-serious-subplan-2026-07-11.md"
)
ATTEMPT1_RESULT_NOTE_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase7-serious-attempt1-infrastructure-result-2026-07-13.md"
)
PHASE6_RESULT_NOTE_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase6-smoke-result-2026-07-11.md"
)
PHASE6_RESULT_PATH = PUBLIC_ROOT / "phase6_smoke_attempt2_result.json"
PHASE6_PROGRESS_PATH = PUBLIC_ROOT / "phase6_smoke_attempt2_progress.json"
PHASE6_OUTPUT_MANIFEST_PATH = (
    PUBLIC_ROOT / "phase6_smoke_attempt2_output_manifest.json"
)
ATTEMPT1_PUBLIC_RESULT_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "burnin_sampling.json"
)
HISTORICAL_RESULT_PATH = ATTEMPT1_PUBLIC_RESULT_PATH
HISTORICAL_ARCHIVE_PATH = (
    PUBLIC_ROOT / "phase7_historical_pre_migration_burnin_sampling.json"
)
HISTORICAL_ARCHIVE_MANIFEST_PATH = (
    PUBLIC_ROOT / "phase7_historical_pre_migration_burnin_sampling_manifest.json"
)
ATTEMPT1_PROPOSAL_PATH = PUBLIC_ROOT / "phase7_serious_authority_proposal.json"
ATTEMPT1_PROPOSAL_MANIFEST_PATH = PUBLIC_ROOT / (
    "phase7_serious_authority_proposal_manifest.json"
)
ATTEMPT1_AUTHORITY_PATH = PUBLIC_ROOT / "phase7_serious_authority.json"
ATTEMPT1_CLAIM_PATH = PUBLIC_ROOT / "phase7_serious_launch_claim.json"
ATTEMPT1_OUTPUT_MANIFEST_PATH = PUBLIC_ROOT / "phase7_serious_output_manifest.json"
ATTEMPT1_INFRASTRUCTURE_FAILURE_PATH = (
    PUBLIC_ROOT / "phase7_serious_infrastructure_failure.json"
)
ATTEMPT1_INFRASTRUCTURE_MANIFEST_PATH = (
    PUBLIC_ROOT / "phase7_serious_infrastructure_manifest.json"
)
ATTEMPT1_PUBLIC_PROGRESS_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "burnin_sampling_progress.json"
)
ATTEMPT1_PRIVATE_SAMPLES_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "private_diagnostics/phase7_retained_samples.npz"
)
ATTEMPT1_LOG_PATH = REPO_ROOT / (
    "docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/"
    "phase7_serious.log"
)

PROPOSAL_PATH = PUBLIC_ROOT / "phase7_serious_attempt2_authority_proposal.json"
PROPOSAL_MANIFEST_PATH = PUBLIC_ROOT / (
    "phase7_serious_attempt2_authority_proposal_manifest.json"
)
AUTHORITY_PATH = PUBLIC_ROOT / "phase7_serious_attempt2_authority.json"
CLAIM_PATH = PUBLIC_ROOT / "phase7_serious_attempt2_launch_claim.json"
OUTPUT_MANIFEST_PATH = PUBLIC_ROOT / "phase7_serious_attempt2_output_manifest.json"
INFRASTRUCTURE_FAILURE_PATH = (
    PUBLIC_ROOT / "phase7_serious_attempt2_infrastructure_failure.json"
)
INFRASTRUCTURE_MANIFEST_PATH = (
    PUBLIC_ROOT / "phase7_serious_attempt2_infrastructure_manifest.json"
)
PUBLIC_RESULT_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "phase7_serious_attempt2_burnin_sampling.json"
)
PUBLIC_PROGRESS_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "phase7_serious_attempt2_burnin_sampling_progress.json"
)
PRIVATE_SAMPLES_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "private_diagnostics/phase7_serious_attempt2_retained_samples.npz"
)
LOG_PATH = REPO_ROOT / (
    "docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/"
    "phase7_serious_attempt2.log"
)
LAUNCHER_PATH = REPO_ROOT / "scripts/run_hmc_phase7_typed_identity_serious.py"
PROPOSAL_BUILDER_PATH = REPO_ROOT / (
    "scripts/build_hmc_phase7_serious_authority_proposal.py"
)
AUTHORITY_BUILDER_PATH = REPO_ROOT / "scripts/build_hmc_phase7_serious_authority.py"
ARCHIVE_BUILDER_PATH = REPO_ROOT / "scripts/archive_hmc_phase7_historical_result.py"
AUTHORITY_MODULE_PATH = Path(__file__).resolve()
AUTHORITY_TEST_PATH = REPO_ROOT / "tests/test_hmc_serious_authority.py"
CONTROLLER_PATH = REPO_ROOT / (
    "bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py"
)
CONTROLLER_TEST_PATH = REPO_ROOT / "tests/test_deterministic_lgssm_hmc_phase7_tf.py"
BENCHMARK_DRIVER_PATH = REPO_ROOT / (
    "docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
)

HISTORICAL_RESULT_RAW_SHA256 = (
    "3b34cf56062950a9ba835f6b4839421510a8921545a0edd36203f39eac4ec0d6"
)
HISTORICAL_RESULT_EMBEDDED_HASH = (
    "sha256:e7ae6f73b92f66e2a346823b3323f419cab57cbc3731dd5e267b7af8e60269bd"
)
HISTORICAL_RESULT_BYTE_COUNT = 2378

PHASE6_TERMINAL_EXPECTATIONS = {
    PHASE6_PROPOSAL_PATH: (
        "7a5c093a42d7b373d1711c29ed073eb46954f3517d4246878a5d1ff20df40880",
        30498,
        "sha256:d2aff98cb93b85527bd71a206af5244aa18e373ae8a3bd7897b8fc3c841d0395",
        parse_smoke_authority_proposal,
    ),
    PHASE6_PROPOSAL_MANIFEST_PATH: (
        "e15cd087fa40e91acb875d88d948fc185a0e6bf1eabc17841111aa9048a7d503",
        847,
        "sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0",
        parse_smoke_authority_proposal_manifest,
    ),
    PHASE6_AUTHORITY_PATH: (
        "0ec56084480028932761b29ed16d18310919323f1a879c20a8806d8937154a66",
        1712,
        "sha256:1f3b8f6b92fda72221fa5036ad752c997d75e4e975b0e0c83afe116eef5e0e9b",
        parse_smoke_authority,
    ),
    PHASE6_CLAIM_PATH: (
        "af813e923269b7547c7cdaab8c2d46f256c2620181dba4d067030fbc6d1608ae",
        1967,
        "sha256:7c3b9ec793eb5dffc5f8b0471ba839cbda7684b2d794c172c51c7df50e93f5ca",
        parse_launch_claim,
    ),
    PHASE6_RESULT_PATH: (
        "ebd870daba9aaff60708327efba3c9fcaeec0624e396be59b34ba61c245e3397",
        16505,
        "sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d",
        parse_smoke_terminal_result,
    ),
    PHASE6_PROGRESS_PATH: (
        "251d91ddf7f7c86c8a35b1b01a2e1ae91afa138a82bf5b44bb255f1a34bbe730",
        1560,
        "sha256:698818a54380c2f2207c35a122201c000111a63c8d52c9d256c98e9051370e05",
        parse_smoke_progress,
    ),
    PHASE6_TERMINAL_MANIFEST_PATH: (
        "6faa154774f99b7cadd7ea9665501b34c329c9663793ea9f3487e8d0507a0384",
        4100,
        "sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8",
        parse_smoke_output_manifest,
    ),
    PHASE6_LOG_PATH: (
        "1bb925e287b15ef927219c709e5f887a9708d7125d9c61d15842037476ebe08e",
        5494,
        None,
        None,
    ),
    PHASE6_PRIVATE_SAMPLES_PATH: (
        "d46514c6fad6dd0b55f9563f9686fee0436034a448677e295062ec899c24393f",
        4392,
        None,
        None,
    ),
    PHASE6_INFRASTRUCTURE_FAILURE_PATH: (
        hashlib.sha256(b"").hexdigest(),
        0,
        None,
        None,
    ),
    PHASE6_INFRASTRUCTURE_MANIFEST_PATH: (
        hashlib.sha256(b"").hexdigest(),
        0,
        None,
        None,
    ),
}

ATTEMPT1_TERMINAL_EXPECTATIONS = {
    ATTEMPT1_PROPOSAL_PATH: (
        "ec5ccd3a006d56e76ed2789288d05b1fb411859dc4f9f019e1d342aa7efa9ebd",
        33316,
        0o600,
        "sha256:5ee3beb04b32e892c34fd49ebb2ac3a7a7498a964aebb3df11196544a994a5eb",
        "proposal",
    ),
    ATTEMPT1_PROPOSAL_MANIFEST_PATH: (
        "28d4335fc1f2a4939db0b1c1bf6a55b13f636a3d5d0a4e37518db0a236575b9b",
        851,
        0o600,
        "sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330",
        "proposal_manifest",
    ),
    ATTEMPT1_AUTHORITY_PATH: (
        "ee58976627310dd8eb3c491f7b810e1240bccb1a6620bc0d5316c6e6d1949b52",
        1638,
        0o600,
        "sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d",
        "authority",
    ),
    ATTEMPT1_CLAIM_PATH: (
        "9dd3a18b045efd863e0fc5c2a60ffca7b1daa104b540b2a947d320b2312b423e",
        2053,
        0o400,
        "sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c",
        "claim",
    ),
    ATTEMPT1_INFRASTRUCTURE_FAILURE_PATH: (
        "06eb4261f452754f658f1acf07c0d47190875286985a92393173122678dfa4f6",
        1227,
        0o400,
        "sha256:f571a196112af7b0fa04b63bda24fe30e4ece7cf8a2fe11dc63429bce7baf173",
        "infrastructure_failure",
    ),
    ATTEMPT1_INFRASTRUCTURE_MANIFEST_PATH: (
        "2ce6fce8d09d4dc6723988f47a8e0ba17e555646065530abdfc57657ee2213d8",
        3500,
        0o400,
        "sha256:7c8a1fc2a886ca1ae9930e1099c3e0f36830875fac47451183b57ab04f7e440e",
        "infrastructure_manifest",
    ),
    ATTEMPT1_OUTPUT_MANIFEST_PATH: (
        hashlib.sha256(b"").hexdigest(),
        0,
        0o400,
        None,
        "empty_output_manifest_reservation",
    ),
    ATTEMPT1_PUBLIC_RESULT_PATH: (
        hashlib.sha256(b"").hexdigest(),
        0,
        0o400,
        None,
        "empty_public_result_reservation",
    ),
    ATTEMPT1_LOG_PATH: (
        hashlib.sha256(b"").hexdigest(),
        0,
        0o400,
        None,
        "empty_log_reservation",
    ),
}
ATTEMPT1_ABSENT_PATHS = (
    ATTEMPT1_PUBLIC_PROGRESS_PATH,
    ATTEMPT1_PRIVATE_SAMPLES_PATH,
)
ATTEMPT1_DOCUMENT_EXPECTATIONS = {
    ATTEMPT1_PHASE7_SUBPLAN_PATH: (
        "99fc680721acdb1a1d0502d91320f4459c186e4050e49d38a4cbf9b75d480be9",
        20325,
    ),
    ATTEMPT1_RESULT_NOTE_PATH: (
        "b1e3c028e4121e04f9b29ab4bf2743f548fd8bc1d9bdb95514e967adffb1bd2b",
        7287,
    ),
}

SERIOUS_PROPOSAL_NONCLAIMS = (
    "attempt-2 proposal and no-runtime evidence only",
    "human serious-run approval not yet recorded",
    "not a worker, HMC transition, burn-in, or sampling authority",
    "not Phase 8 or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
SERIOUS_AUTHORITY_NONCLAIMS = (
    "one attempt-2 serious typed-identity two-worker CPU/XLA Phase 7 launch only",
    "authority is consumed permanently before attempt-2 output or worker creation",
    "attempt-1 evidence remains immutable and is never replaced",
    "not Phase 8 or NeuTra authority",
    "not posterior recovery, superiority, production, default, GPU, or scientific evidence",
)
SERIOUS_NONCLAIMS = (
    "serious Phase 7 convergence screen executed",
    "posterior recovery remains unevaluated until separately authorized Phase 8",
    "no sampler superiority claim",
    "not production, default, or GPU evidence",
    "not NeuTra or broad scientific-validity evidence",
)
SERIOUS_FAILURE_NONCLAIMS = (
    "serious Phase 7 launch attempted after permanent authority consumption",
    "candidate or engineering failure is not automatically evidence against the target or scientific direction",
    "posterior recovery and Phase 8 remain unevaluated",
    "no sampler superiority, production, default, GPU, NeuTra, or broad scientific claim",
)
SERIOUS_INFRASTRUCTURE_NONCLAIMS = (
    "attempt-2 serious authority was consumed before a launcher infrastructure failure",
    "attempt-1 terminal evidence preservation status is recorded, not assumed",
    "any valid primary serious controller result is preserved without overwrite",
    "not Phase 8 or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
ATTEMPT1_SERIOUS_PROPOSAL_NONCLAIMS = (
    "proposal and no-runtime evidence only",
    "human serious-run approval not yet recorded",
    "not a worker, HMC transition, burn-in, or sampling authority",
    "not Phase 8 or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
ATTEMPT1_SERIOUS_AUTHORITY_NONCLAIMS = (
    "one serious typed-identity two-worker CPU/XLA Phase 7 launch only",
    "authority is consumed permanently before historical live-result replacement or worker creation",
    "not Phase 8 or NeuTra authority",
    "not posterior recovery, superiority, production, default, GPU, or scientific evidence",
)
ATTEMPT1_SERIOUS_INFRASTRUCTURE_NONCLAIMS = (
    "serious authority was consumed before a launcher infrastructure failure",
    "historical archive preservation status is recorded, not assumed",
    "any valid primary serious controller result is preserved without overwrite",
    "not Phase 8 or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
_ATTEMPT1_PROPOSAL_REFERENCE_SCHEMAS = {
    "v2_config_reference": "bayesfilter.deterministic_lgssm_hmc_phase7_config.v2",
    "adoption_record_reference": "bayesfilter.hmc_identity_baseline_adoption_record.v1",
    "preflight_reference": "bayesfilter.hmc_identity_phase5_preflight_report.v1",
    "phase5_manifest_reference": "bayesfilter.hmc_identity_phase5_output_manifest.v1",
    "phase6_result_reference": "bayesfilter.hmc_phase6_smoke_result.v1",
    "phase6_progress_reference": "bayesfilter.hmc_phase6_smoke_progress.v1",
    "phase6_output_manifest_reference": "bayesfilter.hmc_phase6_smoke_output_manifest.v1",
    "historical_archive_manifest_reference": HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1,
}

_HEX = frozenset("0123456789abcdef")
_PATH_FIELDS = (
    "claim_path",
    "log_path",
    "public_result_path",
    "public_progress_path",
    "output_manifest_path",
    "infrastructure_failure_path",
    "infrastructure_manifest_path",
    "private_samples_path",
)
@dataclass
class _PreparedSeriousRegistration:
    token: object
    evidence: "SeriousInheritedEvidenceSession"
    directories: PinnedSmokeOutputDirectories
    claim_fd: int | None = None
    claim_identity: tuple[int, int] | None = None


_PREPARED_TOKENS: dict[int, _PreparedSeriousRegistration] = {}


class SeriousInheritedEvidenceDriftError(ConsumedAttempt1EvidenceDriftError):
    """Signal drift in evidence that must remain pinned through serious teardown."""


class SeriousPostClaimPreparationError(RuntimeError):
    """Carry a consumed launch into emergency-only supervision."""

    def __init__(
        self, *, context: "Phase7SeriousLaunchContext", cause: BaseException
    ) -> None:
        super().__init__(f"serious post-claim preparation failed: {cause}")
        self.context = context
        self.cause = cause
        self.__cause__ = cause


class SeriousInheritedEvidenceSession:
    """Pin Phase 5/6/archive evidence plus the complete attempt-1 session."""

    _BASE_PATHS = (
        V2_CONFIG_PATH,
        ADOPTION_RECORD_PATH,
        PREFLIGHT_PATH,
        PHASE5_MANIFEST_PATH,
        PHASE6_RESULT_PATH,
        PHASE6_PROGRESS_PATH,
        *PHASE6_TERMINAL_EXPECTATIONS,
        HISTORICAL_ARCHIVE_PATH,
        HISTORICAL_ARCHIVE_MANIFEST_PATH,
        PHASE6_RESULT_NOTE_PATH,
        ATTEMPT1_PHASE7_SUBPLAN_PATH,
        ATTEMPT1_RESULT_NOTE_PATH,
        PHASE7_SUBPLAN_PATH,
        *ATTEMPT1_TERMINAL_EXPECTATIONS,
    )

    _EXPECTED_MODES = {
        V2_CONFIG_PATH: 0o600,
        ADOPTION_RECORD_PATH: 0o600,
        PREFLIGHT_PATH: 0o600,
        PHASE5_MANIFEST_PATH: 0o600,
        PHASE6_PROPOSAL_PATH: 0o600,
        PHASE6_PROPOSAL_MANIFEST_PATH: 0o600,
        PHASE6_AUTHORITY_PATH: 0o600,
        PHASE6_CLAIM_PATH: 0o400,
        PHASE6_RESULT_PATH: 0o400,
        PHASE6_PROGRESS_PATH: 0o400,
        PHASE6_TERMINAL_MANIFEST_PATH: 0o400,
        PHASE6_LOG_PATH: 0o400,
        PHASE6_PRIVATE_SAMPLES_PATH: 0o400,
        PHASE6_INFRASTRUCTURE_FAILURE_PATH: 0o400,
        PHASE6_INFRASTRUCTURE_MANIFEST_PATH: 0o400,
        HISTORICAL_ARCHIVE_PATH: 0o400,
        HISTORICAL_ARCHIVE_MANIFEST_PATH: 0o400,
        PHASE6_RESULT_NOTE_PATH: 0o644,
        ATTEMPT1_PHASE7_SUBPLAN_PATH: 0o644,
        ATTEMPT1_RESULT_NOTE_PATH: 0o644,
        PHASE7_SUBPLAN_PATH: 0o644,
        **{
            path: expectation[2]
            for path, expectation in ATTEMPT1_TERMINAL_EXPECTATIONS.items()
        },
        PROPOSAL_PATH: 0o600,
        PROPOSAL_MANIFEST_PATH: 0o600,
        AUTHORITY_PATH: 0o600,
    }

    def __init__(self, *, extra_paths: Sequence[Path] = ()) -> None:
        self.attempt1 = ConsumedAttempt1EvidenceSession.open()
        self.parent_entries: dict[Path, tuple[int, tuple[int, int]]] = {}
        self.entries: list[
            tuple[Path, int, int, tuple[int, ...], bytes]
        ] = []
        self._retired_path_invariants: set[Path] = set()
        self._closed = False
        try:
            paths = tuple(dict.fromkeys((*self._BASE_PATHS, *extra_paths)))
            for path in paths:
                self._pin_path(path)
            self.verify()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _signature(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_uid,
            info.st_gid,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    @staticmethod
    def _retired_signature_matches(
        info: os.stat_result, signature: tuple[int, ...]
    ) -> bool:
        """Allow only the metadata transition from replacing the sole link."""

        return (
            signature[3] == 1
            and info.st_nlink == 0
            and (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_size,
                info.st_mtime_ns,
            )
            == (
                signature[0],
                signature[1],
                signature[2],
                signature[4],
                signature[5],
                signature[6],
                signature[7],
            )
            and info.st_ctime_ns >= signature[8]
        )

    @classmethod
    def open(
        cls, *, extra_paths: Sequence[Path] = ()
    ) -> "SeriousInheritedEvidenceSession":
        return cls(extra_paths=extra_paths)

    def __enter__(self) -> "SeriousInheritedEvidenceSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _pin_path(self, path: Path) -> None:
        if self._closed:
            raise RuntimeError("serious inherited evidence session is closed")
        path = Path(os.path.abspath(os.fspath(path)))
        if any(entry[0] == path for entry in self.entries):
            return
        if path.resolve(strict=True) != path or path.is_symlink():
            raise ValueError("serious inherited evidence path contains a symlink")
        parent_entry = self.parent_entries.get(path.parent)
        if parent_entry is None:
            if path.parent.resolve(strict=True) != path.parent or path.parent.is_symlink():
                raise ValueError("serious inherited evidence parent contains a symlink")
            parent_fd = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_info = os.fstat(parent_fd)
            parent_identity = (parent_info.st_dev, parent_info.st_ino)
            self.parent_entries[path.parent] = (parent_fd, parent_identity)
        else:
            parent_fd, _parent_identity = parent_entry
        fd = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            info = os.fstat(fd)
            expected_mode = self._EXPECTED_MODES.get(path)
            if expected_mode is None:
                raise ValueError("serious inherited evidence path is not reviewed")
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or info.st_gid != os.getgid()
                or stat.S_IMODE(info.st_mode) != expected_mode
            ):
                raise ValueError(
                    "serious inherited evidence owner, link, or mode mismatch"
                )
            data = os.pread(fd, info.st_size, 0)
            self.entries.append((path, parent_fd, fd, self._signature(info), data))
        except BaseException:
            os.close(fd)
            raise

    def pin_additional(self, path: str | Path) -> None:
        self._pin_path(Path(path))
        self.verify()

    def read_pinned_bytes(self, path: str | Path) -> bytes:
        target = Path(os.path.abspath(os.fspath(path)))
        if target.resolve(strict=True) != target or target.is_symlink():
            raise ValueError("pinned serious evidence path contains a symlink")
        self.verify()
        for entry_path, _parent_fd, _fd, _signature, data in self.entries:
            if entry_path == target:
                return data
        raise KeyError(f"path is not pinned serious evidence: {target}")

    def _snapshot_bytes(self, path: str | Path) -> bytes:
        target = Path(os.path.abspath(os.fspath(path)))
        for entry_path, _parent_fd, _fd, _signature, data in self.entries:
            if entry_path == target:
                return data
        raise KeyError(f"path is not pinned serious evidence: {target}")

    def read_pinned_json(self, path: str | Path) -> Mapping[str, Any]:
        payload = json.loads(self.read_pinned_bytes(path).decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("pinned serious JSON must contain an object")
        return payload

    def path_matches_snapshot(self, path: str | Path) -> bool:
        """Check one reviewed path without letting unrelated drift mask its status."""

        target = Path(os.path.abspath(os.fspath(path)))
        if target in self._retired_path_invariants:
            return False
        try:
            entry = next(item for item in self.entries if item[0] == target)
            entry_path, parent_fd, fd, signature, expected = entry
            held = os.fstat(fd)
            current = os.stat(
                entry_path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            parent_fd_held, parent_identity = self.parent_entries[entry_path.parent]
            held_parent = os.fstat(parent_fd_held)
            current_parent = os.stat(entry_path.parent, follow_symlinks=False)
            return (
                stat.S_ISDIR(held_parent.st_mode)
                and stat.S_ISDIR(current_parent.st_mode)
                and (held_parent.st_dev, held_parent.st_ino) == parent_identity
                and (current_parent.st_dev, current_parent.st_ino) == parent_identity
                and self._signature(held) == signature
                and self._signature(current) == signature
                and os.pread(fd, held.st_size, 0) == expected
            )
        except (FileNotFoundError, KeyError, OSError, StopIteration):
            return False

    def absent_path_matches_snapshot(self, path: str | Path) -> bool:
        """Verify an absent entry through an already pinned reviewed parent."""

        target = Path(os.path.abspath(os.fspath(path)))
        try:
            parent_fd, parent_identity = self.parent_entries[target.parent]
            held_parent = os.fstat(parent_fd)
            current_parent = os.stat(target.parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held_parent.st_mode)
                or not stat.S_ISDIR(current_parent.st_mode)
                or (held_parent.st_dev, held_parent.st_ino) != parent_identity
                or (current_parent.st_dev, current_parent.st_ino) != parent_identity
            ):
                return False
            try:
                os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return True
            return False
        except (FileNotFoundError, KeyError, OSError):
            return False

    def retire_replaced_path(
        self,
        path: str | Path,
        *,
        original_fd: int,
        replacement_fd: int,
    ) -> None:
        """Retire one reviewed pathname after its authorized atomic replacement."""

        target = Path(os.path.abspath(os.fspath(path)))
        if target != HISTORICAL_RESULT_PATH:
            raise ValueError("only the historical live-result path may be retired")
        entry = next(item for item in self.entries if item[0] == target)
        _entry_path, parent_fd, held_fd, signature, expected = entry
        held = os.fstat(held_fd)
        original = os.fstat(original_fd)
        if (
            (held.st_dev, held.st_ino) != (original.st_dev, original.st_ino)
            or not self._retired_signature_matches(held, signature)
            or not self._retired_signature_matches(original, signature)
        ):
            raise RuntimeError("historical original descriptor changed before retirement")
        if (
            os.pread(held_fd, held.st_size, 0) != expected
            or os.pread(original_fd, original.st_size, 0) != expected
        ):
            raise RuntimeError("historical original bytes changed before retirement")
        if expected != self._snapshot_bytes(HISTORICAL_ARCHIVE_PATH):
            raise RuntimeError("historical original differs from pinned archive")
        if not self.path_matches_snapshot(
            HISTORICAL_ARCHIVE_PATH
        ) or not self.path_matches_snapshot(HISTORICAL_ARCHIVE_MANIFEST_PATH):
            raise RuntimeError("historical archive changed before retirement")
        parent_fd_held, parent_identity = self.parent_entries[target.parent]
        held_parent = os.fstat(parent_fd_held)
        current_parent = os.stat(target.parent, follow_symlinks=False)
        if (
            parent_fd != parent_fd_held
            or not stat.S_ISDIR(held_parent.st_mode)
            or not stat.S_ISDIR(current_parent.st_mode)
            or (held_parent.st_dev, held_parent.st_ino) != parent_identity
            or (current_parent.st_dev, current_parent.st_ino) != parent_identity
        ):
            raise RuntimeError("historical replacement parent identity changed")
        current = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        replacement = os.fstat(replacement_fd)
        if (
            not stat.S_ISREG(current.st_mode)
            or not stat.S_ISREG(replacement.st_mode)
            or self._signature(current) != self._signature(replacement)
            or replacement.st_nlink != 1
            or replacement.st_uid != os.getuid()
            or replacement.st_gid != os.getgid()
            or stat.S_IMODE(replacement.st_mode) != 0o400
            or replacement.st_size != 0
        ):
            raise RuntimeError("historical replacement path is not the reserved result")
        self._retired_path_invariants.add(target)

    def verify_historical_archive_snapshot(self) -> Mapping[str, Any]:
        manifest = json.loads(
            self._snapshot_bytes(HISTORICAL_ARCHIVE_MANIFEST_PATH).decode("utf-8")
        )
        parse_historical_archive_manifest(manifest)
        archive_data = self._snapshot_bytes(HISTORICAL_ARCHIVE_PATH)
        archive_payload = json.loads(archive_data.decode("utf-8"))
        if manifest["immutable_archive_reference"] != _artifact_reference_from_snapshot(
            path=HISTORICAL_ARCHIVE_PATH,
            payload=archive_payload,
            data=archive_data,
            embedded_hash_rule="stable_without_hash",
        ):
            raise ValueError("historical archive snapshot reference mismatch")
        return manifest

    def verify(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("serious inherited evidence session is closed")
        try:
            self.attempt1.verify()
            report = {}
            for parent, (parent_fd, parent_identity) in self.parent_entries.items():
                held_parent = os.fstat(parent_fd)
                current_parent = os.stat(parent, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(held_parent.st_mode)
                    or not stat.S_ISDIR(current_parent.st_mode)
                    or (held_parent.st_dev, held_parent.st_ino) != parent_identity
                    or (current_parent.st_dev, current_parent.st_ino)
                    != parent_identity
                ):
                    raise RuntimeError(
                        "serious inherited evidence parent identity changed"
                    )
            for path, parent_fd, fd, signature, expected in self.entries:
                held = os.fstat(fd)
                retired = path in self._retired_path_invariants
                signature_matches = (
                    self._retired_signature_matches(held, signature)
                    if retired
                    else self._signature(held) == signature
                )
                if not signature_matches:
                    raise RuntimeError("serious inherited evidence identity changed")
                if not retired:
                    current = os.stat(
                        path.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if self._signature(current) != signature:
                        raise RuntimeError("serious inherited evidence identity changed")
                if stat.S_IMODE(held.st_mode) != self._EXPECTED_MODES[path]:
                    raise RuntimeError("serious inherited evidence mode changed")
                first = os.pread(fd, held.st_size, 0)
                second = os.pread(fd, held.st_size, 0)
                if first != expected or second != expected:
                    raise RuntimeError("serious inherited evidence bytes changed")
                report[path.name] = {
                    "file_sha256": hashlib.sha256(first).hexdigest(),
                    "byte_count": len(first),
                    "file_mode": f"{stat.S_IMODE(held.st_mode):04o}",
                }
            self.attempt1.verify()
            self.verify_historical_archive_snapshot()
            for path, (expected_sha, expected_size, embedded_hash, parser) in (
                PHASE6_TERMINAL_EXPECTATIONS.items()
            ):
                entry = next(item for item in self.entries if item[0] == path)
                data = entry[-1]
                if (
                    len(data) != expected_size
                    or hashlib.sha256(data).hexdigest() != expected_sha
                ):
                    raise RuntimeError("Phase 6 terminal exact-byte closure changed")
                if parser is not None:
                    payload = json.loads(data.decode("utf-8"))
                    parser(payload)
                    if payload.get("artifact_hash") != embedded_hash:
                        raise RuntimeError(
                            "Phase 6 terminal embedded artifact hash changed"
                        )
            attempt1_payloads: dict[str, Mapping[str, Any]] = {}
            for path, (
                expected_sha,
                expected_size,
                _expected_mode,
                embedded_hash,
                role,
            ) in ATTEMPT1_TERMINAL_EXPECTATIONS.items():
                entry = next(item for item in self.entries if item[0] == path)
                data = entry[-1]
                if (
                    len(data) != expected_size
                    or hashlib.sha256(data).hexdigest() != expected_sha
                ):
                    raise RuntimeError(
                        f"Phase 7 attempt-1 terminal exact-byte closure changed: {role}"
                    )
                if embedded_hash is not None:
                    payload = json.loads(data.decode("utf-8"))
                    if payload.get("artifact_hash") != embedded_hash:
                        raise RuntimeError(
                            f"Phase 7 attempt-1 embedded artifact hash changed: {role}"
                        )
                    attempt1_payloads[role] = payload
            _verify_attempt1_terminal_semantics(attempt1_payloads)
            for path, (expected_sha, expected_size) in (
                ATTEMPT1_DOCUMENT_EXPECTATIONS.items()
            ):
                data = self._snapshot_bytes(path)
                if (
                    len(data) != expected_size
                    or hashlib.sha256(data).hexdigest() != expected_sha
                ):
                    raise RuntimeError(
                        "Phase 7 attempt-1 document exact-byte closure changed"
                    )
            for absent in ATTEMPT1_ABSENT_PATHS:
                if not self.absent_path_matches_snapshot(absent):
                    raise RuntimeError(
                        "Phase 7 attempt-1 absent output unexpectedly exists"
                    )
            return report
        except SeriousInheritedEvidenceDriftError:
            raise
        except BaseException as error:
            raise SeriousInheritedEvidenceDriftError(str(error)) from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _path, _parent_fd, fd, _signature, _data in reversed(self.entries):
            try:
                os.close(fd)
            except OSError:
                pass
        self.entries.clear()
        self._retired_path_invariants.clear()
        for parent_fd, _identity in reversed(tuple(self.parent_entries.values())):
            try:
                os.close(parent_fd)
            except OSError:
                pass
        self.parent_entries.clear()
        try:
            self.attempt1.close()
        except AttributeError:
            pass


def _exact(payload: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    expected, observed = frozenset(fields), frozenset(payload)
    if expected != observed:
        raise ValueError(
            f"{label} fields mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a trimmed nonblank string")
    return value


def _sha(value: Any, label: str, *, tagged: bool = True) -> str:
    value = _text(value, label)
    digest = value.removeprefix("sha256:") if tagged else value
    if tagged != value.startswith("sha256:") or len(digest) != 64 or any(
        item not in _HEX for item in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _embed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "artifact_hash" in result:
        raise ValueError("artifact_hash must not be prepopulated")
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _verify_hash(payload: Mapping[str, Any], label: str) -> None:
    observed = _sha(payload.get("artifact_hash"), f"{label} hash")
    expected = canonical_artifact_payload_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if observed != expected:
        raise ValueError(f"{label} embedded hash mismatch")


def _parse_attempt1_proposal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "status", "decision", "phase7_subplan_reference",
            "phase6_result_note_reference", *_ATTEMPT1_PROPOSAL_REFERENCE_SCHEMAS,
            "transition_identity_hash", "serious_execution_identity_hash", "runtime",
            "paths", "command", "implementation_references", "launches_proposed",
            "historical_live_result_replacement", "phase8_authority",
            "neutra_authority", "nonclaims", "artifact_hash",
        ),
        "attempt-1 serious authority proposal",
    )
    if (
        payload.get("schema")
        != ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1
        or payload.get("status") != SERIOUS_AUTHORITY_STATUS_PENDING
        or payload.get("decision") != ATTEMPT1_SERIOUS_AUTHORITY_DECISION
        or payload.get("launches_proposed") != 1
        or payload.get("historical_live_result_replacement")
        != "after_permanent_claim_only"
        or payload.get("phase8_authority") is not False
        or payload.get("neutra_authority") is not False
        or tuple(payload.get("nonclaims", ()))
        != ATTEMPT1_SERIOUS_PROPOSAL_NONCLAIMS
    ):
        raise ValueError("attempt-1 serious proposal boundary mismatch")
    for name, schema in _ATTEMPT1_PROPOSAL_REFERENCE_SCHEMAS.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"attempt-1 serious proposal {name} schema mismatch")
    if (
        payload.get("transition_identity_hash") != TRANSITION_IDENTITY_HASH
        or payload.get("serious_execution_identity_hash")
        != SERIOUS_EXECUTION_IDENTITY_HASH
    ):
        raise ValueError("attempt-1 serious proposal identity mismatch")
    _verify_hash(payload, "attempt-1 serious proposal")
    return payload


def _parse_attempt1_proposal_manifest(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact(
        payload,
        ("schema", "terminal_manifest", "proposal_reference", "artifact_hash"),
        "attempt-1 serious proposal manifest",
    )
    reference = parse_phase5_artifact_reference(payload["proposal_reference"])
    if (
        payload.get("schema")
        != ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1
        or payload.get("terminal_manifest") is not True
        or reference["source_schema"]
        != ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1
    ):
        raise ValueError("attempt-1 serious proposal manifest mismatch")
    _verify_hash(payload, "attempt-1 serious proposal manifest")
    return payload


def _attempt1_expected_approval_statement(manifest_hash: str) -> str:
    _sha(manifest_hash, "attempt-1 serious proposal manifest hash")
    return (
        f"I approve {ATTEMPT1_SERIOUS_AUTHORITY_DECISION} bound to Phase 7 "
        f"authority proposal manifest {manifest_hash}."
    )


def _parse_attempt1_authority(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "status", "decision", "human_approval_statement",
            "human_approval_date", "proposal_manifest_reference",
            "launches_authorized", "mode", "phase8_authority",
            "neutra_authority", "nonclaims", "artifact_hash",
        ),
        "attempt-1 serious authority",
    )
    reference = parse_phase5_artifact_reference(payload["proposal_manifest_reference"])
    if (
        payload.get("schema") != ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1
        or payload.get("status") != SERIOUS_AUTHORITY_STATUS_APPROVED
        or payload.get("decision") != ATTEMPT1_SERIOUS_AUTHORITY_DECISION
        or reference["source_schema"]
        != ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1
        or payload.get("human_approval_statement")
        != _attempt1_expected_approval_statement(reference["embedded_artifact_hash"])
        or payload.get("human_approval_date") != "2026-07-13"
        or payload.get("launches_authorized") != 1
        or payload.get("mode") != "serious"
        or payload.get("phase8_authority") is not False
        or payload.get("neutra_authority") is not False
        or tuple(payload.get("nonclaims", ()))
        != ATTEMPT1_SERIOUS_AUTHORITY_NONCLAIMS
    ):
        raise ValueError("attempt-1 serious authority boundary mismatch")
    _verify_hash(payload, "attempt-1 serious authority")
    return payload


def _parse_attempt1_claim(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "authority_artifact_hash", "proposal_manifest_artifact_hash",
            "historical_archive_manifest_artifact_hash", "command", "paths", "pid",
            "started_at_utc", "file_mode", "permanent_authority_consumption",
            "historical_live_result_replacement_authorized", "artifact_hash",
        ),
        "attempt-1 serious launch claim",
    )
    if (
        payload.get("schema")
        != ATTEMPT1_HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1
        or payload.get("authority_artifact_hash")
        != "sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d"
        or payload.get("proposal_manifest_artifact_hash")
        != "sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330"
        or payload.get("historical_archive_manifest_artifact_hash")
        != "sha256:aa135967aa67eeec3c997ccfc102efca7a221c98e40fa0cf6ab51dff24ea8a8a"
        or payload.get("file_mode") != "0400"
        or payload.get("permanent_authority_consumption") is not True
        or payload.get("historical_live_result_replacement_authorized") is not True
    ):
        raise ValueError("attempt-1 serious claim boundary mismatch")
    _verify_hash(payload, "attempt-1 serious claim")
    return payload


def _parse_attempt1_infrastructure_failure(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "passed", "decision", "serious_authority_artifact_hash",
            "serious_launch_claim_artifact_hash",
            "serious_proposal_manifest_artifact_hash", "stage", "reason",
            "primary_result_artifact_hash", "historical_archive_preserved",
            "phase8_executed", "neutra_executed", "nonclaims", "artifact_hash",
        ),
        "attempt-1 serious infrastructure failure",
    )
    if (
        payload.get("schema")
        != ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1
        or payload.get("passed") is not False
        or payload.get("decision")
        != ATTEMPT1_SERIOUS_INFRASTRUCTURE_BLOCK_DECISION
        or payload.get("serious_authority_artifact_hash")
        != "sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d"
        or payload.get("serious_launch_claim_artifact_hash")
        != "sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c"
        or payload.get("serious_proposal_manifest_artifact_hash")
        != "sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330"
        or payload.get("stage")
        != "secure_output_reservation:historical_result_replacement"
        or payload.get("reason") != "infrastructure_error:RuntimeError"
        or payload.get("primary_result_artifact_hash") is not None
        or payload.get("historical_archive_preserved") is not True
        or payload.get("phase8_executed") is not False
        or payload.get("neutra_executed") is not False
        or tuple(payload.get("nonclaims", ()))
        != ATTEMPT1_SERIOUS_INFRASTRUCTURE_NONCLAIMS
    ):
        raise ValueError("attempt-1 infrastructure failure boundary mismatch")
    _verify_hash(payload, "attempt-1 serious infrastructure failure")
    return payload


def _parse_attempt1_infrastructure_manifest(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "terminal_manifest", "passed",
            "historical_archive_manifest_reference", "authority_reference",
            "claim_reference", "infrastructure_failure_reference",
            "public_result_reference", "public_progress_reference", "log_reference",
            "private_samples_reference", "artifact_hash",
        ),
        "attempt-1 serious infrastructure manifest",
    )
    if (
        payload.get("schema")
        != ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1
        or payload.get("terminal_manifest") is not True
        or payload.get("passed") is not False
        or payload.get("public_progress_reference") is not None
        or payload.get("private_samples_reference") is not None
    ):
        raise ValueError("attempt-1 infrastructure manifest boundary mismatch")
    for name in (
        "historical_archive_manifest_reference",
        "authority_reference",
        "claim_reference",
        "infrastructure_failure_reference",
    ):
        parse_phase5_artifact_reference(payload[name])
    for name in ("public_result_reference", "log_reference"):
        reference = parse_file_reference(payload[name])
        if reference["byte_count"] != 0 or reference["file_sha256"] != hashlib.sha256(b"").hexdigest():
            raise ValueError("attempt-1 infrastructure empty reservation mismatch")
    _verify_hash(payload, "attempt-1 serious infrastructure manifest")
    return payload


def _verify_attempt1_terminal_semantics(
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    proposal = _parse_attempt1_proposal(payloads["proposal"])
    manifest = _parse_attempt1_proposal_manifest(payloads["proposal_manifest"])
    authority = _parse_attempt1_authority(payloads["authority"])
    claim = _parse_attempt1_claim(payloads["claim"])
    failure = _parse_attempt1_infrastructure_failure(
        payloads["infrastructure_failure"]
    )
    infrastructure = _parse_attempt1_infrastructure_manifest(
        payloads["infrastructure_manifest"]
    )
    if (
        manifest["proposal_reference"]["embedded_artifact_hash"]
        != proposal["artifact_hash"]
        or authority["proposal_manifest_reference"]["embedded_artifact_hash"]
        != manifest["artifact_hash"]
        or claim["authority_artifact_hash"] != authority["artifact_hash"]
        or claim["proposal_manifest_artifact_hash"] != manifest["artifact_hash"]
        or failure["serious_launch_claim_artifact_hash"] != claim["artifact_hash"]
        or infrastructure["authority_reference"]["embedded_artifact_hash"]
        != authority["artifact_hash"]
        or infrastructure["claim_reference"]["embedded_artifact_hash"]
        != claim["artifact_hash"]
        or infrastructure["infrastructure_failure_reference"][
            "embedded_artifact_hash"
        ]
        != failure["artifact_hash"]
    ):
        raise ValueError("attempt-1 terminal artifact graph mismatch")


def _read_json(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JSON artifact must contain an object")
    return payload


def _repo_role(path: Path) -> str:
    return "repository_file:" + path.resolve().relative_to(REPO_ROOT).as_posix()


def _artifact_reference_from_snapshot(
    *,
    path: Path,
    payload: Mapping[str, Any],
    data: bytes,
    embedded_hash_rule: str = "canonical_without_hash",
) -> Mapping[str, Any]:
    embedded = payload.get("artifact_hash")
    bare = {key: value for key, value in payload.items() if key != "artifact_hash"}
    if embedded_hash_rule == "canonical_without_hash":
        expected = canonical_artifact_payload_hash(bare)
    elif embedded_hash_rule == "stable_without_hash":
        from bayesfilter.runtime import stable_config_hash

        expected = "sha256:" + stable_config_hash(bare)
    elif embedded_hash_rule == "none":
        expected = None
    else:
        raise ValueError("unsupported snapshot embedded hash rule")
    if embedded != expected:
        raise ValueError("snapshot embedded artifact hash mismatch")
    return {
        "schema": "bayesfilter.hmc_identity_phase5_artifact_reference.v1",
        "source_schema": _text(payload.get("schema"), "snapshot source schema"),
        "embedded_hash_rule": embedded_hash_rule,
        "embedded_artifact_hash": embedded,
        "canonical_payload_hash": canonical_artifact_payload_hash(payload),
        "resolved_path_sha256": hashlib.sha256(
            str(path.resolve()).encode("utf-8")
        ).hexdigest(),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _file_reference_from_snapshot(*, path: Path, data: bytes) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.hmc_phase6_file_reference.v1",
        "resolved_path_sha256": hashlib.sha256(
            str(path.resolve()).encode("utf-8")
        ).hexdigest(),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _absence_reference(
    path: Path, *, evidence: SeriousInheritedEvidenceSession
) -> Mapping[str, Any]:
    if not evidence.absent_path_matches_snapshot(path):
        raise ValueError("attempt-1 path required absent is present")
    return {
        "resolved_path_sha256": hashlib.sha256(
            str(path.resolve(strict=False)).encode("utf-8")
        ).hexdigest(),
        "absent": True,
    }


def _build_attempt1_terminal_evidence(
    evidence: SeriousInheritedEvidenceSession,
) -> Mapping[str, Any]:
    artifact_roles = {
        "proposal_reference": ATTEMPT1_PROPOSAL_PATH,
        "proposal_manifest_reference": ATTEMPT1_PROPOSAL_MANIFEST_PATH,
        "authority_reference": ATTEMPT1_AUTHORITY_PATH,
        "claim_reference": ATTEMPT1_CLAIM_PATH,
        "infrastructure_failure_reference": ATTEMPT1_INFRASTRUCTURE_FAILURE_PATH,
        "infrastructure_manifest_reference": ATTEMPT1_INFRASTRUCTURE_MANIFEST_PATH,
    }
    result: dict[str, Any] = {}
    for role, path in artifact_roles.items():
        data = evidence.read_pinned_bytes(path)
        result[role] = _artifact_reference_from_snapshot(
            path=path,
            payload=json.loads(data.decode("utf-8")),
            data=data,
        )
    for role, path in {
        "output_manifest_reservation_reference": ATTEMPT1_OUTPUT_MANIFEST_PATH,
        "public_result_reservation_reference": ATTEMPT1_PUBLIC_RESULT_PATH,
        "log_reservation_reference": ATTEMPT1_LOG_PATH,
    }.items():
        result[role] = _file_reference_from_snapshot(
            path=path,
            data=evidence.read_pinned_bytes(path),
        )
    archive_data = evidence.read_pinned_bytes(HISTORICAL_ARCHIVE_PATH)
    result["historical_archive_reference"] = _artifact_reference_from_snapshot(
        path=HISTORICAL_ARCHIVE_PATH,
        payload=json.loads(archive_data.decode("utf-8")),
        data=archive_data,
        embedded_hash_rule="stable_without_hash",
    )
    result["public_progress_absence"] = _absence_reference(
        ATTEMPT1_PUBLIC_PROGRESS_PATH,
        evidence=evidence,
    )
    result["private_samples_absence"] = _absence_reference(
        ATTEMPT1_PRIVATE_SAMPLES_PATH,
        evidence=evidence,
    )
    result["result_note_reference"] = _document_reference_from_snapshot(
        ATTEMPT1_RESULT_NOTE_PATH,
        evidence.read_pinned_bytes(ATTEMPT1_RESULT_NOTE_PATH),
    )
    result["terminal_decision"] = ATTEMPT1_SERIOUS_INFRASTRUCTURE_BLOCK_DECISION
    result["runtime_reached"] = False
    result["authority_and_claim_consumed"] = True
    return result


def _parse_attempt1_terminal_evidence(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifact_schemas = {
        "proposal_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1,
        "proposal_manifest_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1,
        "authority_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1,
        "claim_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1,
        "infrastructure_failure_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1,
        "infrastructure_manifest_reference": ATTEMPT1_HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1,
        "historical_archive_reference": "bayesfilter.deterministic_lgssm_hmc_phase7_result.v1",
    }
    file_roles = (
        "output_manifest_reservation_reference",
        "public_result_reservation_reference",
        "log_reservation_reference",
    )
    absence_roles = ("public_progress_absence", "private_samples_absence")
    _exact(
        payload,
        (
            *artifact_schemas,
            *file_roles,
            *absence_roles,
            "result_note_reference",
            "terminal_decision",
            "runtime_reached",
            "authority_and_claim_consumed",
        ),
        "attempt-1 terminal evidence",
    )
    for name, schema in artifact_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"attempt-1 terminal {name} schema mismatch")
    for name in file_roles:
        reference = parse_file_reference(payload[name])
        if (
            reference["byte_count"] != 0
            or reference["file_sha256"] != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError(f"attempt-1 terminal {name} is not empty")
    for name in absence_roles:
        _exact(payload[name], ("resolved_path_sha256", "absent"), name)
        _sha(payload[name]["resolved_path_sha256"], name, tagged=False)
        if payload[name]["absent"] is not True:
            raise ValueError(f"attempt-1 terminal {name} mismatch")
    note = payload["result_note_reference"]
    _exact(note, ("path_sha256", "file_sha256", "byte_count"), "attempt-1 result note")
    _sha(note["path_sha256"], "attempt-1 result note path", tagged=False)
    _sha(note["file_sha256"], "attempt-1 result note bytes", tagged=False)
    if not isinstance(note["byte_count"], int) or note["byte_count"] < 1:
        raise ValueError("attempt-1 result note byte count mismatch")
    if (
        payload.get("terminal_decision")
        != ATTEMPT1_SERIOUS_INFRASTRUCTURE_BLOCK_DECISION
        or payload.get("runtime_reached") is not False
        or payload.get("authority_and_claim_consumed") is not True
    ):
        raise ValueError("attempt-1 terminal decision boundary mismatch")
    return payload


def default_implementation_paths(
    python_executable: str | Path,
) -> Mapping[str, Path]:
    roots = (
        AUTHORITY_MODULE_PATH,
        CONTROLLER_PATH,
        BENCHMARK_DRIVER_PATH,
        LAUNCHER_PATH,
        PROPOSAL_BUILDER_PATH,
        AUTHORITY_BUILDER_PATH,
        ARCHIVE_BUILDER_PATH,
        AUTHORITY_TEST_PATH,
        CONTROLLER_TEST_PATH,
    )
    inventory = {_repo_role(path): path.resolve() for path in roots}
    from bayesfilter.inference.hmc_smoke_authority import _phase6_runtime_source_closure

    for path in _phase6_runtime_source_closure():
        inventory[_repo_role(path)] = path.resolve()
    inventory["python_executable"] = Path(python_executable).resolve()
    return dict(sorted(inventory.items()))


def default_paths() -> Mapping[str, str]:
    return {
        "claim_path": CLAIM_PATH.relative_to(REPO_ROOT).as_posix(),
        "log_path": LOG_PATH.relative_to(REPO_ROOT).as_posix(),
        "public_result_path": PUBLIC_RESULT_PATH.relative_to(REPO_ROOT).as_posix(),
        "public_progress_path": PUBLIC_PROGRESS_PATH.relative_to(REPO_ROOT).as_posix(),
        "output_manifest_path": OUTPUT_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        "infrastructure_failure_path": INFRASTRUCTURE_FAILURE_PATH.relative_to(REPO_ROOT).as_posix(),
        "infrastructure_manifest_path": INFRASTRUCTURE_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        "private_samples_path": PRIVATE_SAMPLES_PATH.relative_to(REPO_ROOT).as_posix(),
    }


def parse_paths(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(payload, _PATH_FIELDS, "serious paths")
    values = tuple(_text(payload[name], name) for name in _PATH_FIELDS)
    if len(set(values)) != len(values):
        raise ValueError("serious paths must be distinct")
    if any(Path(value).is_absolute() for value in values):
        raise ValueError("serious paths must be repository-relative")
    for value in values:
        if "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("serious paths must be normalized")
    if dict(payload) != default_paths():
        raise ValueError("serious paths differ from the reviewed fixed paths")
    return payload


def default_runtime() -> Mapping[str, Any]:
    expected_threads = {
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if {name: os.environ.get(name) for name in expected_threads} != expected_threads:
        raise ValueError("serious parent thread environment mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("serious parent requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    import tensorflow_probability as tfp

    return {
        "mode": "serious",
        "worker_count": 2,
        "chains_per_worker": 2,
        "chain_count": 4,
        "burnin": {"initial": 2000, "window": 1000, "extension": 1000, "cap": 16000},
        "retained": {"initial": 4000, "interval": 2000, "extension": 2000, "cap": 40000},
        "diagnostics": {"rhat_max": 1.01, "bulk_ess_min": 1000.0, "tail_ess_min": 400.0, "all_parameters_required": True},
        "cuda_visible_devices": "-1",
        "dtype": "float64",
        "jit_compile": True,
        "use_xla": True,
        "compile_workers_sequentially": True,
        "root_seed": [20260711, 701],
        "tensorflow_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "python_version": platform.python_version(),
        "thread_environment": expected_threads,
        "wall_time_cap_seconds": 28800,
    }


def parse_runtime(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = default_runtime()
    if payload != expected:
        raise ValueError("serious runtime differs from fixed live contract")
    return payload


def expected_launcher_command(python_executable: str | Path) -> tuple[str, ...]:
    return (
        str(Path(python_executable).resolve()),
        LAUNCHER_PATH.relative_to(REPO_ROOT).as_posix(),
        "--stage",
        "burnin_sampling",
        "--phase7-serious-authority",
        AUTHORITY_PATH.relative_to(REPO_ROOT).as_posix(),
    )


def parse_command(command: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("serious command must be a sequence")
    normalized = tuple(_text(item, "command item") for item in command)
    if normalized != expected_launcher_command(normalized[0]):
        raise ValueError("serious command differs from reviewed command")
    return normalized


def build_historical_archive_manifest() -> Mapping[str, Any]:
    """Return the existing terminal archive manifest without rebuilding it."""

    return verify_historical_archive_bundle()


def write_historical_archive_bundle() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compatibility entry point that only verifies immutable archive evidence."""

    manifest = verify_historical_archive_bundle()
    data = HISTORICAL_ARCHIVE_PATH.read_bytes()
    payload = json.loads(data.decode("utf-8"))
    if (
        len(data) != HISTORICAL_RESULT_BYTE_COUNT
        or hashlib.sha256(data).hexdigest() != HISTORICAL_RESULT_RAW_SHA256
        or payload.get("artifact_hash") != HISTORICAL_RESULT_EMBEDDED_HASH
    ):
        raise ValueError("historical serious archive differs from terminal evidence")
    return payload, manifest


def parse_historical_archive_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "terminal_manifest", "historical_live_reference",
            "immutable_archive_reference",
            "replacement_permitted_only_after_consumed_serious_claim",
            "archive_mutation_permitted", "artifact_hash",
        ),
        "historical archive manifest",
    )
    if payload.get("schema") != HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1 or (
        payload.get("terminal_manifest") is not True
    ):
        raise ValueError("historical archive manifest schema/status mismatch")
    live = parse_phase5_artifact_reference(payload["historical_live_reference"])
    archive = parse_phase5_artifact_reference(payload["immutable_archive_reference"])
    if live["source_schema"] != "bayesfilter.deterministic_lgssm_hmc_phase7_result.v1" or (
        archive["source_schema"] != live["source_schema"]
    ):
        raise ValueError("historical archive schema mismatch")
    if live["file_sha256"] != HISTORICAL_RESULT_RAW_SHA256 or live["byte_count"] != HISTORICAL_RESULT_BYTE_COUNT or (
        live["embedded_artifact_hash"] != HISTORICAL_RESULT_EMBEDDED_HASH
    ):
        raise ValueError("historical live result reference mismatch")
    if archive["file_sha256"] != live["file_sha256"] or archive["byte_count"] != live["byte_count"] or (
        archive["embedded_artifact_hash"] != live["embedded_artifact_hash"]
    ):
        raise ValueError("historical archive does not preserve exact bytes")
    if payload.get("replacement_permitted_only_after_consumed_serious_claim") is not True or (
        payload.get("archive_mutation_permitted") is not False
    ):
        raise ValueError("historical archive boundary mismatch")
    _verify_hash(payload, "historical archive manifest")
    return payload


def verify_historical_archive_bundle() -> Mapping[str, Any]:
    manifest = _read_json(HISTORICAL_ARCHIVE_MANIFEST_PATH)
    parse_historical_archive_manifest(manifest)
    verify_phase5_artifact_reference(
        manifest["immutable_archive_reference"], path=HISTORICAL_ARCHIVE_PATH
    )
    live = ATTEMPT1_PUBLIC_RESULT_PATH
    info = live.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or live.read_bytes() != b""
    ):
        raise ValueError("attempt-1 terminal public result reservation changed")
    return manifest


def _document_reference(path: Path) -> Mapping[str, Any]:
    return _document_reference_from_snapshot(path, path.read_bytes())


def _document_reference_from_snapshot(path: Path, data: bytes) -> Mapping[str, Any]:
    return {
        "path_sha256": hashlib.sha256(str(path.resolve()).encode()).hexdigest(),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def build_default_serious_authority_proposal(
    *,
    python_executable: str | Path,
    evidence_session: SeriousInheritedEvidenceSession | None = None,
) -> Mapping[str, Any]:
    if evidence_session is None:
        with SeriousInheritedEvidenceSession.open() as evidence:
            return build_default_serious_authority_proposal(
                python_executable=python_executable,
                evidence_session=evidence,
            )
    implementation_paths = default_implementation_paths(python_executable)
    document_reference = lambda path: _document_reference_from_snapshot(
        path,
        evidence_session.read_pinned_bytes(path),
    )

    def artifact_reference(path: Path) -> Mapping[str, Any]:
        data = evidence_session.read_pinned_bytes(path)
        return _artifact_reference_from_snapshot(
            path=path,
            payload=json.loads(data.decode("utf-8")),
            data=data,
        )
    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1,
            "status": SERIOUS_AUTHORITY_STATUS_PENDING,
            "decision": SERIOUS_AUTHORITY_DECISION,
            "phase7_subplan_reference": document_reference(PHASE7_SUBPLAN_PATH),
            "phase6_result_note_reference": document_reference(PHASE6_RESULT_NOTE_PATH),
            "v2_config_reference": artifact_reference(V2_CONFIG_PATH),
            "adoption_record_reference": artifact_reference(ADOPTION_RECORD_PATH),
            "preflight_reference": artifact_reference(PREFLIGHT_PATH),
            "phase5_manifest_reference": artifact_reference(PHASE5_MANIFEST_PATH),
            "phase6_result_reference": artifact_reference(PHASE6_RESULT_PATH),
            "phase6_progress_reference": artifact_reference(PHASE6_PROGRESS_PATH),
            "phase6_output_manifest_reference": artifact_reference(PHASE6_OUTPUT_MANIFEST_PATH),
            "historical_archive_manifest_reference": artifact_reference(HISTORICAL_ARCHIVE_MANIFEST_PATH),
            "attempt1_terminal_evidence": _build_attempt1_terminal_evidence(evidence_session),
            "transition_identity_hash": TRANSITION_IDENTITY_HASH,
            "serious_execution_identity_hash": SERIOUS_EXECUTION_IDENTITY_HASH,
            "runtime": default_runtime(),
            "paths": default_paths(),
            "command": expected_launcher_command(python_executable),
            "implementation_references": {
                role: build_file_reference(path)
                for role, path in implementation_paths.items()
            },
            "launches_proposed": 1,
            "attempt2_output_policy": "exclusive_create_all_outputs",
            "attempt1_mutation_permitted": False,
            "phase8_authority": False,
            "neutra_authority": False,
            "nonclaims": SERIOUS_PROPOSAL_NONCLAIMS,
        }
    )


_PROPOSAL_REFERENCE_SCHEMAS = {
    "v2_config_reference": "bayesfilter.deterministic_lgssm_hmc_phase7_config.v2",
    "adoption_record_reference": "bayesfilter.hmc_identity_baseline_adoption_record.v1",
    "preflight_reference": "bayesfilter.hmc_identity_phase5_preflight_report.v1",
    "phase5_manifest_reference": "bayesfilter.hmc_identity_phase5_output_manifest.v1",
    "phase6_result_reference": "bayesfilter.hmc_phase6_smoke_result.v1",
    "phase6_progress_reference": "bayesfilter.hmc_phase6_smoke_progress.v1",
    "phase6_output_manifest_reference": "bayesfilter.hmc_phase6_smoke_output_manifest.v1",
    "historical_archive_manifest_reference": HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1,
}


def parse_serious_authority_proposal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(
        payload,
        (
            "schema", "status", "decision", "phase7_subplan_reference",
            "phase6_result_note_reference", *_PROPOSAL_REFERENCE_SCHEMAS,
            "attempt1_terminal_evidence",
            "transition_identity_hash", "serious_execution_identity_hash", "runtime",
            "paths", "command", "implementation_references", "launches_proposed",
            "attempt2_output_policy", "attempt1_mutation_permitted", "phase8_authority",
            "neutra_authority", "nonclaims", "artifact_hash",
        ),
        "serious authority proposal",
    )
    if payload.get("schema") != HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1 or (
        payload.get("status") != SERIOUS_AUTHORITY_STATUS_PENDING
    ) or payload.get("decision") != SERIOUS_AUTHORITY_DECISION:
        raise ValueError("serious proposal identity/status mismatch")
    for name in ("phase7_subplan_reference", "phase6_result_note_reference"):
        _exact(payload[name], ("path_sha256", "file_sha256", "byte_count"), name)
        _sha(payload[name]["path_sha256"], name, tagged=False)
        _sha(payload[name]["file_sha256"], name, tagged=False)
        if not isinstance(payload[name]["byte_count"], int) or payload[name]["byte_count"] < 1:
            raise ValueError(f"{name} byte count mismatch")
    for name, schema in _PROPOSAL_REFERENCE_SCHEMAS.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"serious proposal {name} schema mismatch")
    _parse_attempt1_terminal_evidence(payload["attempt1_terminal_evidence"])
    if payload.get("transition_identity_hash") != TRANSITION_IDENTITY_HASH or (
        payload.get("serious_execution_identity_hash") != SERIOUS_EXECUTION_IDENTITY_HASH
    ):
        raise ValueError("serious proposal typed identity mismatch")
    parse_runtime(payload["runtime"])
    parse_paths(payload["paths"])
    command = parse_command(payload["command"])
    implementation = payload.get("implementation_references")
    if not isinstance(implementation, Mapping) or "python_executable" not in implementation:
        raise ValueError("serious proposal implementation inventory is incomplete")
    for role, reference in implementation.items():
        parse_file_reference(reference)
        expected = Path(command[0]) if role == "python_executable" else (
            REPO_ROOT / role.removeprefix("repository_file:")
        )
        if role != "python_executable" and not role.startswith("repository_file:"):
            raise ValueError("serious proposal implementation role mismatch")
        if reference["resolved_path_sha256"] != hashlib.sha256(
            str(expected.resolve()).encode()
        ).hexdigest():
            raise ValueError("serious proposal implementation path mismatch")
    if (
        payload.get("launches_proposed") != 1
        or payload.get("attempt2_output_policy") != "exclusive_create_all_outputs"
        or payload.get("attempt1_mutation_permitted") is not False
    ):
        raise ValueError("serious proposal launch/output-isolation mismatch")
    if payload.get("phase8_authority") is not False or payload.get("neutra_authority") is not False:
        raise ValueError("serious proposal crossed a later authority boundary")
    if tuple(payload.get("nonclaims", ())) != SERIOUS_PROPOSAL_NONCLAIMS:
        raise ValueError("serious proposal nonclaims mismatch")
    _verify_hash(payload, "serious authority proposal")
    return payload


def verify_serious_authority_proposal_candidate(
    payload: Mapping[str, Any],
    *,
    python_executable: str | Path,
    evidence_session: SeriousInheritedEvidenceSession | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    owns_evidence = evidence_session is None
    evidence = (
        SeriousInheritedEvidenceSession.open()
        if evidence_session is None
        else evidence_session
    )
    try:
        return _verify_serious_authority_proposal_candidate_from_snapshot(
            payload,
            python_executable=python_executable,
            evidence=evidence,
        )
    finally:
        if owns_evidence:
            evidence.close()


def _verify_serious_authority_proposal_candidate_from_snapshot(
    payload: Mapping[str, Any],
    *,
    python_executable: str | Path,
    evidence: SeriousInheritedEvidenceSession,
) -> tuple[Any, Mapping[str, Any]]:
    parse_serious_authority_proposal(payload)
    if payload["phase7_subplan_reference"] != _document_reference_from_snapshot(
        PHASE7_SUBPLAN_PATH,
        evidence.read_pinned_bytes(PHASE7_SUBPLAN_PATH),
    ) or payload["phase6_result_note_reference"] != _document_reference_from_snapshot(
        PHASE6_RESULT_NOTE_PATH,
        evidence.read_pinned_bytes(PHASE6_RESULT_NOTE_PATH),
    ):
        raise ValueError("serious proposal document reference drift")
    artifact_paths = {
        "v2_config_reference": V2_CONFIG_PATH,
        "adoption_record_reference": ADOPTION_RECORD_PATH,
        "preflight_reference": PREFLIGHT_PATH,
        "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
        "phase6_result_reference": PHASE6_RESULT_PATH,
        "phase6_progress_reference": PHASE6_PROGRESS_PATH,
        "phase6_output_manifest_reference": PHASE6_OUTPUT_MANIFEST_PATH,
        "historical_archive_manifest_reference": HISTORICAL_ARCHIVE_MANIFEST_PATH,
    }
    for name, path in artifact_paths.items():
        snapshot = evidence.read_pinned_bytes(path)
        snapshot_payload = json.loads(snapshot.decode("utf-8"))
        expected_reference = _artifact_reference_from_snapshot(
            path=path,
            payload=snapshot_payload,
            data=snapshot,
            embedded_hash_rule=payload[name]["embedded_hash_rule"],
        )
        if payload[name] != expected_reference:
            raise ValueError(f"serious proposal {name} snapshot mismatch")
    if payload["attempt1_terminal_evidence"] != _build_attempt1_terminal_evidence(
        evidence
    ):
        raise ValueError("serious proposal attempt-1 terminal evidence drift")
    evidence.verify_historical_archive_snapshot()
    evidence.attempt1.verify_semantics()
    paths = default_implementation_paths(python_executable)
    verify_implementation_reference_inventory(
        payload["implementation_references"],
        python_executable=python_executable,
        implementation_paths=paths,
    )
    from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
        DeterministicLGSSMPhase7Config,
        validate_phase7_inputs,
    )
    from bayesfilter.inference.hmc_identity_adoption import (
        parse_phase5_adoption_record,
        parse_phase5_output_manifest,
        parse_phase7_v2_config,
    )

    config_payload = evidence.read_pinned_json(V2_CONFIG_PATH)
    parse_phase7_v2_config(config_payload)
    config = DeterministicLGSSMPhase7Config(
        payload=config_payload,
        path=V2_CONFIG_PATH,
    )
    config.validate()
    if config.runtime_authority is not False:
        raise ValueError("serious proposal requires runtime-inert V2 config")
    adoption = evidence.read_pinned_json(ADOPTION_RECORD_PATH)
    parse_phase5_adoption_record(adoption)
    stored = evidence.read_pinned_json(PREFLIGHT_PATH)
    parse_phase5_preflight_report(stored)
    phase5_manifest = evidence.read_pinned_json(PHASE5_MANIFEST_PATH)
    parse_phase5_output_manifest(phase5_manifest)
    config_reference = _artifact_reference_from_snapshot(
        path=V2_CONFIG_PATH,
        payload=config_payload,
        data=evidence.read_pinned_bytes(V2_CONFIG_PATH),
    )
    adoption_reference = _artifact_reference_from_snapshot(
        path=ADOPTION_RECORD_PATH,
        payload=adoption,
        data=evidence.read_pinned_bytes(ADOPTION_RECORD_PATH),
    )
    if adoption["v2_config_reference"] != config_reference or (
        stored["config_reference"] != config_reference
    ) or stored["adoption_record_reference"] != adoption_reference:
        raise ValueError("Phase 5 snapshot cross-link mismatch")
    phase5_snapshots = (
        ("v2_config", config_payload, evidence.read_pinned_bytes(V2_CONFIG_PATH)),
        (
            "adoption_record",
            adoption,
            evidence.read_pinned_bytes(ADOPTION_RECORD_PATH),
        ),
        ("preflight_report", stored, evidence.read_pinned_bytes(PREFLIGHT_PATH)),
    )
    expected_outputs = tuple(
        {
            "role": role,
            "schema": item["schema"],
            "artifact_hash": item["artifact_hash"],
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
        }
        for role, item, data in phase5_snapshots
    )
    if tuple(phase5_manifest["outputs"]) != expected_outputs:
        raise ValueError("Phase 5 output manifest snapshot mismatch")
    live = validate_phase7_inputs(config)
    if canonical_artifact_payload_hash(live) != canonical_artifact_payload_hash(stored):
        raise ValueError("serious live preflight differs from Phase 5 evidence")
    if live["identity_hashes"]["transition_identity_hash"] != TRANSITION_IDENTITY_HASH or live["identity_hashes"]["serious_execution_contract_hash"] != SERIOUS_EXECUTION_IDENTITY_HASH:
        raise ValueError("serious live typed identity mismatch")
    if payload["runtime"] != default_runtime() or payload["paths"] != default_paths() or tuple(payload["command"]) != expected_launcher_command(python_executable):
        raise ValueError("serious proposal live runtime/path/command mismatch")
    return config, live


def build_serious_authority_proposal_manifest(
    *, proposal_reference: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    reference = (
        build_phase5_artifact_reference(
            PROPOSAL_PATH, embedded_hash_rule="canonical_without_hash"
        )
        if proposal_reference is None
        else dict(proposal_reference)
    )
    parsed = parse_phase5_artifact_reference(reference)
    if parsed["source_schema"] != HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1:
        raise ValueError("serious proposal reference schema mismatch")
    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "proposal_reference": reference,
        }
    )


def parse_serious_authority_proposal_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(payload, ("schema", "terminal_manifest", "proposal_reference", "artifact_hash"), "serious proposal manifest")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 or payload.get("terminal_manifest") is not True:
        raise ValueError("serious proposal manifest identity/status mismatch")
    reference = parse_phase5_artifact_reference(payload["proposal_reference"])
    if reference["source_schema"] != HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1:
        raise ValueError("serious proposal manifest reference mismatch")
    _verify_hash(payload, "serious proposal manifest")
    return payload


def verify_serious_authority_proposal_manifest(
    payload: Mapping[str, Any],
    *,
    proposal_payload: Mapping[str, Any] | None = None,
    proposal_bytes: bytes | None = None,
) -> Mapping[str, Any]:
    parse_serious_authority_proposal_manifest(payload)
    if proposal_payload is None or proposal_bytes is None:
        verify_phase5_artifact_reference(
            payload["proposal_reference"],
            path=PROPOSAL_PATH,
        )
    else:
        expected = _artifact_reference_from_snapshot(
            path=PROPOSAL_PATH,
            payload=proposal_payload,
            data=proposal_bytes,
        )
        if payload["proposal_reference"] != expected:
            raise ValueError("serious proposal manifest pinned snapshot mismatch")
    return payload


def expected_serious_approval_statement(manifest_hash: str) -> str:
    _sha(manifest_hash, "serious proposal manifest hash")
    return (
        f"I approve {SERIOUS_AUTHORITY_DECISION} bound to Phase 7 attempt-2 "
        f"authority proposal manifest {manifest_hash}."
    )


def build_serious_authority(
    *,
    approval_statement: str,
    approval_date: str,
    proposal_manifest: Mapping[str, Any] | None = None,
    proposal_manifest_reference: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    manifest = (
        _read_json(PROPOSAL_MANIFEST_PATH)
        if proposal_manifest is None
        else dict(proposal_manifest)
    )
    parse_serious_authority_proposal_manifest(manifest)
    expected = expected_serious_approval_statement(manifest["artifact_hash"])
    if approval_statement != expected:
        raise ValueError("serious human approval statement mismatch")
    from datetime import date

    try:
        canonical_approval_date = date.fromisoformat(approval_date).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError("serious approval date must use YYYY-MM-DD") from error
    if canonical_approval_date != approval_date:
        raise ValueError("serious approval date must use YYYY-MM-DD")
    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1,
            "status": SERIOUS_AUTHORITY_STATUS_APPROVED,
            "decision": SERIOUS_AUTHORITY_DECISION,
            "human_approval_statement": expected,
            "human_approval_date": approval_date,
            "proposal_manifest_reference": (
                build_phase5_artifact_reference(
                    PROPOSAL_MANIFEST_PATH,
                    embedded_hash_rule="canonical_without_hash",
                )
                if proposal_manifest_reference is None
                else dict(proposal_manifest_reference)
            ),
            "launches_authorized": 1,
            "mode": "serious",
            "phase8_authority": False,
            "neutra_authority": False,
            "nonclaims": SERIOUS_AUTHORITY_NONCLAIMS,
        }
    )


def parse_serious_authority(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(payload, ("schema", "status", "decision", "human_approval_statement", "human_approval_date", "proposal_manifest_reference", "launches_authorized", "mode", "phase8_authority", "neutra_authority", "nonclaims", "artifact_hash"), "serious authority")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1 or payload.get("status") != SERIOUS_AUTHORITY_STATUS_APPROVED or payload.get("decision") != SERIOUS_AUTHORITY_DECISION:
        raise ValueError("serious authority identity/status mismatch")
    reference = parse_phase5_artifact_reference(payload["proposal_manifest_reference"])
    if reference["source_schema"] != HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 or payload.get("human_approval_statement") != expected_serious_approval_statement(reference["embedded_artifact_hash"]):
        raise ValueError("serious authority approval/manifest mismatch")
    if payload.get("launches_authorized") != 1 or payload.get("mode") != "serious" or payload.get("phase8_authority") is not False or payload.get("neutra_authority") is not False:
        raise ValueError("serious authority scope mismatch")
    from datetime import date

    approval_date = _text(payload.get("human_approval_date"), "approval date")
    try:
        canonical_approval_date = date.fromisoformat(approval_date).isoformat()
    except ValueError as error:
        raise ValueError(
            "serious authority approval date must use YYYY-MM-DD"
        ) from error
    if canonical_approval_date != approval_date:
        raise ValueError("serious authority approval date must use YYYY-MM-DD")
    if tuple(payload.get("nonclaims", ())) != SERIOUS_AUTHORITY_NONCLAIMS:
        raise ValueError("serious authority nonclaims mismatch")
    _verify_hash(payload, "serious authority")
    return payload


def build_serious_launch_claim(
    *,
    authority: Mapping[str, Any],
    proposal: Mapping[str, Any],
    manifest: Mapping[str, Any],
    command: Sequence[str],
    paths: Mapping[str, Any],
    pid: int,
) -> Mapping[str, Any]:
    parse_serious_authority(authority)
    parse_serious_authority_proposal(proposal)
    parse_serious_authority_proposal_manifest(manifest)
    parse_command(command)
    parse_paths(paths)
    archive_hash = proposal["historical_archive_manifest_reference"][
        "embedded_artifact_hash"
    ]
    _sha(archive_hash, "historical archive manifest artifact hash")
    if (
        manifest["proposal_reference"]["embedded_artifact_hash"]
        != proposal["artifact_hash"]
        or manifest["proposal_reference"]["canonical_payload_hash"]
        != canonical_artifact_payload_hash(proposal)
    ):
        raise ValueError("serious claim proposal/manifest mismatch")
    if (
        authority["proposal_manifest_reference"]["embedded_artifact_hash"]
        != manifest["artifact_hash"]
        or authority["proposal_manifest_reference"]["canonical_payload_hash"]
        != canonical_artifact_payload_hash(manifest)
    ):
        raise ValueError("serious claim authority/manifest mismatch")
    if tuple(command) != tuple(proposal["command"]) or dict(paths) != dict(
        proposal["paths"]
    ):
        raise ValueError("serious claim command/path differs from proposal")
    from datetime import datetime, timezone

    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1,
            "authority_artifact_hash": authority["artifact_hash"],
            "proposal_manifest_artifact_hash": manifest["artifact_hash"],
            "historical_archive_manifest_artifact_hash": archive_hash,
            "command": tuple(command),
            "paths": dict(paths),
            "pid": int(pid),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_mode": "0400",
            "permanent_authority_consumption": True,
            "exclusive_attempt2_outputs_authorized": True,
            "attempt1_mutation_authorized": False,
        }
    )


def parse_serious_launch_claim(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact(payload, ("schema", "authority_artifact_hash", "proposal_manifest_artifact_hash", "historical_archive_manifest_artifact_hash", "command", "paths", "pid", "started_at_utc", "file_mode", "permanent_authority_consumption", "exclusive_attempt2_outputs_authorized", "attempt1_mutation_authorized", "artifact_hash"), "serious launch claim")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1:
        raise ValueError("serious claim schema mismatch")
    for name in ("authority_artifact_hash", "proposal_manifest_artifact_hash", "historical_archive_manifest_artifact_hash"):
        _sha(payload.get(name), name)
    parse_command(payload["command"])
    parse_paths(payload["paths"])
    if type(payload.get("pid")) is not int or payload["pid"] < 1 or payload.get("file_mode") != "0400" or payload.get("permanent_authority_consumption") is not True or payload.get("exclusive_attempt2_outputs_authorized") is not True or payload.get("attempt1_mutation_authorized") is not False:
        raise ValueError("serious claim scope/mode mismatch")
    from datetime import datetime, timedelta

    started = _text(payload.get("started_at_utc"), "claim start")
    parsed_started = datetime.fromisoformat(started)
    if parsed_started.tzinfo is None or parsed_started.utcoffset() != timedelta(0) or (
        parsed_started.isoformat() != started
    ):
        raise ValueError("serious claim start must be timezone-aware UTC")
    _verify_hash(payload, "serious launch claim")
    return payload


@dataclass
class Phase7SeriousLaunchContext:
    authority_kind: str
    config: Any
    preflight: Mapping[str, Any]
    proposal: Mapping[str, Any]
    proposal_manifest: Mapping[str, Any]
    authority: Mapping[str, Any]
    authority_reference: Mapping[str, Any]
    claim: Mapping[str, Any]
    paths: Mapping[str, Path]
    command: tuple[str, ...]
    implementation_source_bundle: Mapping[str, bytes]
    implementation_paths: Mapping[str, Path]
    output_directories: PinnedSmokeOutputDirectories
    claim_fd: int
    consumed_evidence_session: SeriousInheritedEvidenceSession
    output_session: Any | None = None
    prepared_hash: str | None = None
    _token: Any | None = None


def _context_hash(context: Phase7SeriousLaunchContext) -> str:
    return canonical_artifact_payload_hash(
        {
            "authority_kind": context.authority_kind,
            "config_path": str(context.config.path.resolve()),
            "config_snapshot_hash": canonical_artifact_payload_hash(
                context.config.payload
            ),
            "preflight_snapshot_hash": canonical_artifact_payload_hash(
                context.preflight
            ),
            "proposal_snapshot_hash": canonical_artifact_payload_hash(
                context.proposal
            ),
            "proposal_manifest_snapshot_hash": canonical_artifact_payload_hash(
                context.proposal_manifest
            ),
            "authority_snapshot_hash": canonical_artifact_payload_hash(
                context.authority
            ),
            "authority_reference_snapshot_hash": canonical_artifact_payload_hash(
                context.authority_reference
            ),
            "claim_snapshot_hash": canonical_artifact_payload_hash(context.claim),
            "command": context.command,
            "paths": {
                name: str(path.resolve())
                for name, path in sorted(context.paths.items())
            },
            "source_bundle_hash": implementation_source_bundle_hash(context.implementation_source_bundle),
            "implementation_paths": {
                name: str(path.resolve())
                for name, path in sorted(context.implementation_paths.items())
            },
        }
    )


def _resolve_paths() -> Mapping[str, Path]:
    paths = {name: REPO_ROOT / value for name, value in default_paths().items()}
    for name, path in paths.items():
        if path.resolve() != path or REPO_ROOT not in path.parents or not path.parent.is_dir() or path.parent.is_symlink():
            raise ValueError(f"serious {name} path or parent mismatch")
    for name, path in paths.items():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"serious {name} already exists")
    return paths


def prepare_serious_launch(*, authority_path: str | Path, current_command: Sequence[str]) -> Phase7SeriousLaunchContext:
    if Path(authority_path).resolve() != AUTHORITY_PATH:
        raise ValueError("serious authority path mismatch")
    command = parse_command(current_command)
    evidence = SeriousInheritedEvidenceSession.open(
        extra_paths=(PROPOSAL_PATH, PROPOSAL_MANIFEST_PATH, AUTHORITY_PATH)
    )
    directories = None
    claim_fd = None
    try:
        proposal = evidence.read_pinned_json(PROPOSAL_PATH)
        manifest = evidence.read_pinned_json(PROPOSAL_MANIFEST_PATH)
        parse_serious_authority_proposal(proposal)
        verify_serious_authority_proposal_manifest(
            manifest,
            proposal_payload=proposal,
            proposal_bytes=evidence.read_pinned_bytes(PROPOSAL_PATH),
        )
        config, preflight = verify_serious_authority_proposal_candidate(
            proposal,
            python_executable=command[0],
            evidence_session=evidence,
        )
        authority = evidence.read_pinned_json(AUTHORITY_PATH)
        parse_serious_authority(authority)
        manifest_reference = _artifact_reference_from_snapshot(
            path=PROPOSAL_MANIFEST_PATH,
            payload=manifest,
            data=evidence.read_pinned_bytes(PROPOSAL_MANIFEST_PATH),
        )
        if authority["proposal_manifest_reference"] != manifest_reference:
            raise ValueError("serious authority pinned manifest reference mismatch")
        if tuple(proposal["command"]) != command:
            raise ValueError("serious launch command mismatch")
        implementation_paths = default_implementation_paths(command[0])
        source_bundle = build_verified_implementation_source_bundle(proposal["implementation_references"], python_executable=command[0], implementation_paths=implementation_paths)
        paths = _resolve_paths()
        directories = PinnedSmokeOutputDirectories.open(paths)
        authority_reference = _artifact_reference_from_snapshot(
            path=AUTHORITY_PATH,
            payload=authority,
            data=evidence.read_pinned_bytes(AUTHORITY_PATH),
        )
        if (
            authority_reference["embedded_artifact_hash"]
            != authority["artifact_hash"]
            or authority_reference["canonical_payload_hash"]
            != canonical_artifact_payload_hash(authority)
        ):
            raise ValueError("captured serious authority/reference mismatch")
        evidence.verify()
        claim = build_serious_launch_claim(
            authority=authority,
            proposal=proposal,
            manifest=manifest,
            command=command,
            paths=proposal["paths"],
            pid=os.getpid(),
        )
        token = object()
        context = Phase7SeriousLaunchContext(
            authority_kind="phase7_serious", config=config, preflight=preflight,
            proposal=proposal, proposal_manifest=manifest, authority=authority,
            authority_reference=authority_reference, claim=claim, paths=paths,
            command=command, implementation_source_bundle=source_bundle,
            implementation_paths=implementation_paths,
            output_directories=directories, claim_fd=-1,
            consumed_evidence_session=evidence,
            output_session=None, _token=token,
        )
        context.prepared_hash = _context_hash(context)
        registration = _PreparedSeriousRegistration(
            token=token,
            evidence=evidence,
            directories=directories,
        )
        _PREPARED_TOKENS[id(context)] = registration
        try:
            claim_fd = create_durable_launch_claim_with_consumed_evidence(
                paths["claim_path"],
                claim,
                pinned_directories=directories,
                consumed_evidence_session=evidence,
                parser=parse_serious_launch_claim,
            )
        except BaseException as error:
            try:
                recovered_fd = directories.open_existing_readonly("claim_path")
                encoded_claim = (
                    json.dumps(claim, sort_keys=True, indent=2) + "\n"
                ).encode("utf-8")
                recovered_bytes = os.pread(
                    recovered_fd, os.fstat(recovered_fd).st_size, 0
                )
                if recovered_bytes != encoded_claim:
                    os.close(recovered_fd)
                    raise ValueError("serious claim path contains incomplete bytes")
                directories.assert_fd_matches_path("claim_path", recovered_fd)
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                raise error.with_traceback(error.__traceback__)
            context.claim_fd = recovered_fd
            raise SeriousPostClaimPreparationError(
                context=context,
                cause=error,
            ) from error
        context.claim_fd = claim_fd
        try:
            claim_info = os.fstat(claim_fd)
            registration.claim_fd = claim_fd
            registration.claim_identity = (claim_info.st_dev, claim_info.st_ino)
        except BaseException as error:
            raise SeriousPostClaimPreparationError(
                context=context,
                cause=error,
            ) from error
    except SeriousPostClaimPreparationError:
        raise
    except BaseException:
        if "context" in locals():
            _PREPARED_TOKENS.pop(id(context), None)
        if claim_fd is not None:
            os.close(claim_fd)
        if directories is not None:
            directories.close()
        evidence.close()
        raise
    return context


class SecureSeriousOutputSession(SecureSmokeOutputSession):
    @classmethod
    def reserve_from_context(cls, context: Phase7SeriousLaunchContext) -> "SecureSeriousOutputSession":
        return cls.reserve(
            directories=context.output_directories,
            claim_fd=context.claim_fd,
            consumed_evidence_session=context.consumed_evidence_session,
        )

    @classmethod
    def reserve_emergency_from_context(
        cls, context: Phase7SeriousLaunchContext
    ) -> "SecureSeriousOutputSession":
        claim_fd = context.claim_fd
        try:
            context.output_directories.assert_fd_matches_path("claim_path", claim_fd)
        except (OSError, RuntimeError, ValueError):
            claim_fd = context.output_directories.open_existing_readonly("claim_path")
            context.claim_fd = claim_fd
        encoded_claim = (
            json.dumps(context.claim, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        if os.pread(claim_fd, os.fstat(claim_fd).st_size, 0) != encoded_claim:
            raise RuntimeError("emergency serious claim bytes are not complete")
        session = cls(
            directories=context.output_directories,
            claim_fd=claim_fd,
            consumed_evidence_session=None,
        )
        try:
            for role in (
                "infrastructure_failure_path",
                "infrastructure_manifest_path",
            ):
                session.fds[role] = context.output_directories.open_exclusive(role)
                context.output_directories.assert_fd_matches_path(
                    role, session.fds[role]
                )
                context.output_directories.fsync_parent(role)
        except BaseException:
            session.close()
            raise
        return session

def attach_serious_output_session(context: Phase7SeriousLaunchContext, session: SecureSeriousOutputSession) -> Phase7SeriousLaunchContext:
    verify_prepared_serious_launch_context(context)
    if session.directories is not context.output_directories or session.fds.get("claim_path") != context.claim_fd or session.consumed_evidence_session is not context.consumed_evidence_session:
        raise ValueError("serious output session/context mismatch")
    registration = _PREPARED_TOKENS[id(context)]
    attached = Phase7SeriousLaunchContext(**{**context.__dict__, "output_session": session})
    _PREPARED_TOKENS.pop(id(context), None)
    _PREPARED_TOKENS[id(attached)] = registration
    return attached


def verify_prepared_serious_launch_context(context: Phase7SeriousLaunchContext, *, consume: bool = False) -> Phase7SeriousLaunchContext:
    token = context._token
    registration = _PREPARED_TOKENS.get(id(context))
    if token is None or registration is None or registration.token is not token or context.authority_kind != "phase7_serious":
        raise ValueError("serious launch context was not issued by prepare_serious_launch")
    if registration.evidence is not context.consumed_evidence_session or registration.directories is not context.output_directories:
        raise ValueError("serious launch context retained-object identity mismatch")
    if registration.claim_fd != context.claim_fd:
        raise ValueError("serious launch context descriptor identity mismatch")
    if registration.claim_identity is None:
        raise ValueError("serious launch context claim was not durably installed")
    claim_info = os.fstat(context.claim_fd)
    if (claim_info.st_dev, claim_info.st_ino) != registration.claim_identity:
        raise ValueError("serious launch context retained inode changed")
    context.output_directories.assert_fd_matches_path("claim_path", context.claim_fd)
    context.consumed_evidence_session.verify()
    if context.prepared_hash != _context_hash(context):
        raise ValueError("serious launch context snapshot mismatch")
    if consume:
        _PREPARED_TOKENS.pop(id(context), None)
    return context


def discard_prepared_serious_launch_context(context: Phase7SeriousLaunchContext) -> None:
    _PREPARED_TOKENS.pop(id(context), None)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_diagnostic(
    payload: Mapping[str, Any],
    *,
    require_pass: bool,
    expected_draw_count: int | None = None,
) -> None:
    fields = (
        "schema", "passed", "input_all_finite", "diagnostics_all_finite",
        "draw_count_per_chain", "chain_count", "parameter_count",
        "split_draw_count_per_chain", "split_chain_count", "thresholds",
        "definitions", "max_rhat", "min_bulk_ess", "min_tail_ess",
        "parameter_diagnostics", "hard_vetoes", "nonclaims",
    )
    _exact(payload, fields, "serious diagnostic")
    if payload.get("schema") != "bayesfilter.rank_normalized_hmc_diagnostics.v1":
        raise ValueError("serious diagnostic schema mismatch")
    if payload.get("chain_count") != 4 or payload.get("parameter_count") != 18:
        raise ValueError("serious diagnostic topology mismatch")
    draw_count = payload.get("draw_count_per_chain")
    if not isinstance(draw_count, int) or draw_count < 4:
        raise ValueError("serious diagnostic draw count mismatch")
    if expected_draw_count is not None and draw_count != expected_draw_count:
        raise ValueError("serious diagnostic/result draw count mismatch")
    if payload.get("split_draw_count_per_chain") != draw_count // 2 or (
        payload.get("split_chain_count") != 8
    ):
        raise ValueError("serious diagnostic split topology mismatch")
    if payload.get("thresholds") != {
        "rhat_max": 1.01,
        "bulk_ess_min": 1000.0,
        "tail_ess_min": 400.0,
    }:
        raise ValueError("serious diagnostic threshold mismatch")
    if payload.get("definitions") != SERIOUS_DIAGNOSTIC_DEFINITIONS or tuple(
        payload.get("nonclaims", ())
    ) != SERIOUS_DIAGNOSTIC_NONCLAIMS:
        raise ValueError("serious diagnostic definition/nonclaim mismatch")
    rows = payload.get("parameter_diagnostics")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 18:
        raise ValueError("serious diagnostic parameter rows mismatch")
    parsed_metrics: list[tuple[float, float, float]] = []
    row_fields = (
        "parameter",
        "rank_normalized_split_rhat",
        "folded_rank_normalized_split_rhat",
        "rhat",
        "bulk_ess",
        "tail_ess",
        "lower_tail_ess",
        "upper_tail_ess",
        "passed",
    )
    for index, row in enumerate(rows):
        _exact(row, row_fields, "serious diagnostic row")
        if row.get("parameter") != SERIOUS_PARAMETER_NAMES[index] or not isinstance(
            row.get("passed"), bool
        ):
            raise ValueError("serious diagnostic parameter order/pass mismatch")
        rank_rhat = _finite_number(
            row.get("rank_normalized_split_rhat"), "rank R-hat"
        )
        folded_rhat = _finite_number(
            row.get("folded_rank_normalized_split_rhat"), "folded R-hat"
        )
        rhat = _finite_number(row.get("rhat"), "R-hat")
        bulk = _finite_number(row.get("bulk_ess"), "bulk ESS")
        tail = _finite_number(row.get("tail_ess"), "tail ESS")
        lower = _finite_number(row.get("lower_tail_ess"), "lower-tail ESS")
        upper = _finite_number(row.get("upper_tail_ess"), "upper-tail ESS")
        if rhat != max(rank_rhat, folded_rhat) or tail != min(lower, upper):
            raise ValueError("serious diagnostic derived metric mismatch")
        expected_pass = rhat <= 1.01 and bulk >= 1000.0 and tail >= 400.0
        if row["passed"] is not expected_pass:
            raise ValueError("serious diagnostic row pass flag mismatch")
        parsed_metrics.append((rhat, bulk, tail))
    maximum_rhat = _finite_number(payload.get("max_rhat"), "maximum R-hat")
    minimum_bulk = _finite_number(payload.get("min_bulk_ess"), "minimum bulk ESS")
    minimum_tail = _finite_number(payload.get("min_tail_ess"), "minimum tail ESS")
    if (
        maximum_rhat != max(item[0] for item in parsed_metrics)
        or minimum_bulk != min(item[1] for item in parsed_metrics)
        or minimum_tail != min(item[2] for item in parsed_metrics)
    ):
        raise ValueError("serious diagnostic aggregate mismatch")
    expected_all_pass = all(row["passed"] for row in rows)
    if not isinstance(payload.get("passed"), bool) or payload["passed"] is not (
        expected_all_pass
    ):
        raise ValueError("serious diagnostic overall pass mismatch")
    if require_pass:
        if payload.get("passed") is not True or payload.get("input_all_finite") is not True or payload.get("diagnostics_all_finite") is not True or tuple(payload.get("hard_vetoes", ())) != ():
            raise ValueError("serious passing diagnostic hard-veto mismatch")
    elif (
        payload.get("input_all_finite") is not True
        or payload.get("diagnostics_all_finite") is not True
        or tuple(payload.get("hard_vetoes", ())) != ()
    ):
        raise ValueError("serious finite diagnostic hard-veto mismatch")


def _parse_progress_check(payload: Mapping[str, Any], *, stage: str) -> None:
    _exact(
        payload,
        (
            "stage", "completed_results_per_chain", "passed", "max_rhat",
            "min_bulk_ess", "min_tail_ess", "input_all_finite",
            "diagnostics_all_finite", "hard_vetoes",
        ),
        f"serious {stage} progress check",
    )
    completed = payload.get("completed_results_per_chain")
    if stage == "burnin":
        valid = isinstance(completed, int) and 2000 <= completed <= 16000 and (
            (completed - 2000) % 1000 == 0
        )
    else:
        valid = isinstance(completed, int) and 4000 <= completed <= 40000 and (
            (completed - 4000) % 2000 == 0
        )
    if payload.get("stage") != stage or not valid:
        raise ValueError(f"serious {stage} progress count mismatch")
    if payload.get("input_all_finite") is not True or payload.get("diagnostics_all_finite") is not True or tuple(payload.get("hard_vetoes", ())) != ():
        raise ValueError(f"serious {stage} progress hard-veto mismatch")
    maximum_rhat = _finite_number(payload.get("max_rhat"), "progress R-hat")
    minimum_bulk = _finite_number(payload.get("min_bulk_ess"), "progress bulk ESS")
    minimum_tail = _finite_number(payload.get("min_tail_ess"), "progress tail ESS")
    expected_pass = (
        maximum_rhat <= 1.01
        and minimum_bulk >= 1000.0
        and minimum_tail >= 400.0
    )
    if not isinstance(payload.get("passed"), bool) or payload["passed"] is not (
        expected_pass
    ):
        raise ValueError(f"serious {stage} progress pass mismatch")


def _validate_progress_schedule(
    checks: Sequence[Mapping[str, Any]], *, stage: str
) -> None:
    initial, step = (2000, 1000) if stage == "burnin" else (4000, 2000)
    counts = tuple(item["completed_results_per_chain"] for item in checks)
    if counts != tuple(initial + step * index for index in range(len(counts))):
        raise ValueError(f"serious {stage} progress schedule mismatch")
    if any(item["passed"] is True for item in checks[:-1]):
        raise ValueError(f"serious {stage} progress continued after passing")


def parse_serious_progress(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = ("schema", "status", "config_hash", "smoke", "serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "preflight_before_runtime_artifact_hash", "burnin_checks", "retained_checks", "completed", "passed", "result_artifact_hash", "artifact_hash")
    _exact(payload, fields, "serious progress")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1 or payload.get("smoke") is not False or payload.get("completed") is not True or payload.get("status") not in {"result_written", "blocked_result_written"}:
        raise ValueError("serious progress terminal state mismatch")
    for name in ("config_hash", "serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "preflight_before_runtime_artifact_hash", "result_artifact_hash"):
        _sha(payload.get(name), name)
    if not isinstance(payload.get("burnin_checks"), Sequence) or not isinstance(payload.get("retained_checks"), Sequence):
        raise ValueError("serious progress checks mismatch")
    if payload.get("config_hash") != SERIOUS_CONFIG_HASH:
        raise ValueError("serious progress config mismatch")
    for check in payload["burnin_checks"]:
        _parse_progress_check(check, stage="burnin")
    for check in payload["retained_checks"]:
        _parse_progress_check(check, stage="retained")
    _validate_progress_schedule(payload["burnin_checks"], stage="burnin")
    _validate_progress_schedule(payload["retained_checks"], stage="retained")
    if not isinstance(payload.get("passed"), bool) or (
        payload.get("status") == "result_written"
    ) is not payload.get("passed"):
        raise ValueError("serious progress status/pass mismatch")
    if payload["retained_checks"] and (
        not payload["burnin_checks"]
        or payload["burnin_checks"][-1].get("passed") is not True
    ):
        raise ValueError("serious retained checks began before burn-in passed")
    if payload.get("passed") is True and (
        not payload["burnin_checks"]
        or not payload["retained_checks"]
        or payload["burnin_checks"][-1].get("passed") is not True
        or payload["retained_checks"][-1].get("passed") is not True
    ):
        raise ValueError("passing serious progress lacks passing terminal checks")
    _verify_hash(payload, "serious progress")
    return payload


def parse_serious_terminal_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    success_fields = ("schema", "passed", "decision", "smoke", "serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "preflight_before_runtime_artifact_hash", "config_hash", "preflight_before_runtime", "burnin_results_per_chain", "retained_results_per_chain", "final_diagnostics", "worker_count", "chains_per_worker", "chain_count", "worker_pids", "worker_metadata", "private_retained_sample_reference", "jit_compile", "jit_compile_false_runtime_executed", "cuda_visible_devices", "elapsed_seconds", "serious_runtime_executed", "neutra_executed", "phase8_executed", "nonclaims", "artifact_hash")
    failure_fields = ("schema", "passed", "decision", "smoke", "serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "preflight_before_runtime_artifact_hash", "stage", "reason", "config_hash", "preflight_before_runtime", "worker_pids", "final_diagnostics", "jit_compile_false_runtime_executed", "cuda_visible_devices", "elapsed_seconds", "serious_runtime_executed", "neutra_executed", "phase8_executed", "nonclaims", "artifact_hash")
    schema = payload.get("schema")
    if schema == HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1:
        _exact(payload, success_fields, "serious result")
        burnin_count = payload.get("burnin_results_per_chain")
        retained_count = payload.get("retained_results_per_chain")
        valid_burnin = isinstance(burnin_count, int) and 2000 <= burnin_count <= 16000 and (burnin_count - 2000) % 1000 == 0
        valid_retained = isinstance(retained_count, int) and 4000 <= retained_count <= 40000 and (retained_count - 4000) % 2000 == 0
        if payload.get("passed") is not True or payload.get("decision") != SERIOUS_PASS_DECISION or not valid_burnin or not valid_retained or payload.get("worker_count") != 2 or payload.get("chains_per_worker") != 2 or payload.get("chain_count") != 4:
            raise ValueError("serious passing result contract mismatch")
        _parse_diagnostic(
            payload["final_diagnostics"],
            require_pass=True,
            expected_draw_count=retained_count,
        )
        pids = payload.get("worker_pids")
        if not isinstance(pids, Sequence) or isinstance(pids, (str, bytes)) or len(pids) != 2 or len(set(pids)) != 2 or any(not isinstance(pid, int) or pid < 1 for pid in pids):
            raise ValueError("serious passing result worker PIDs mismatch")
        metadata = payload.get("worker_metadata")
        if not isinstance(metadata, Sequence) or len(metadata) != 2:
            raise ValueError("serious passing result worker metadata mismatch")
        worker_fields = (
            "worker_index", "pid", "child_worker_cache_seal_hash",
            "jit_compile", "use_xla",
            "compile_trace_count", "first_call_s", "warm_call_s",
            "tensorflow_version", "tfp_version", "python_version",
            "cuda_visible_devices", "thread_environment",
            "child_source_references_verified",
            "child_implementation_references_verified",
            "child_loaded_source_bytes_verified",
            "child_implementation_source_bundle_hash",
            "child_transition_identity_verified",
            "child_transition_identity_hash",
        )
        expected_worker_environment = {
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase7-worker",
        }
        bundle_hashes = set()
        for index, worker in enumerate(metadata):
            _exact(worker, worker_fields, "serious worker metadata")
            if worker.get("worker_index") != index or worker.get("pid") != pids[index]:
                raise ValueError("serious worker index/PID provenance mismatch")
            if worker.get("jit_compile") is not True or worker.get("use_xla") is not True or worker.get("cuda_visible_devices") != "-1" or worker.get("compile_trace_count") != 1 or worker.get("child_source_references_verified") is not True or worker.get("child_implementation_references_verified") is not True or worker.get("child_loaded_source_bytes_verified") is not True or worker.get("child_transition_identity_verified") is not True or worker.get("child_transition_identity_hash") != TRANSITION_IDENTITY_HASH:
                raise ValueError("serious passing worker provenance mismatch")
            if worker.get("thread_environment") != expected_worker_environment:
                raise ValueError("serious worker environment mismatch")
            for name in ("tensorflow_version", "tfp_version", "python_version"):
                _text(worker.get(name), f"worker {name}")
            for name in ("first_call_s", "warm_call_s"):
                if _finite_number(worker.get(name), f"worker {name}") < 0.0:
                    raise ValueError("serious worker timing must be nonnegative")
            bundle_hashes.add(
                _sha(
                    worker.get("child_implementation_source_bundle_hash"),
                    "worker source bundle hash",
                )
            )
            _sha(
                worker.get("child_worker_cache_seal_hash"),
                "worker cache seal hash",
            )
        if len(bundle_hashes) != 1:
            raise ValueError("serious workers used different source bundles")
        private = payload.get("private_retained_sample_reference")
        _exact(private, ("file_sha256", "byte_count", "shape_verified", "finite_verified", "provenance_verified", "path_publicized", "raw_samples_publicized"), "serious private sample reference")
        if private.get("shape_verified") is not True or private.get("finite_verified") is not True or private.get("provenance_verified") is not True or private.get("path_publicized") is not False or private.get("raw_samples_publicized") is not False or not isinstance(private.get("byte_count"), int) or private["byte_count"] < 1:
            raise ValueError("serious passing private sample reference mismatch")
        _sha(private.get("file_sha256"), "private sample hash", tagged=False)
        if payload.get("jit_compile") is not True:
            raise ValueError("serious passing result lacks top-level XLA/JIT evidence")
        if tuple(payload.get("nonclaims", ())) != SERIOUS_NONCLAIMS:
            raise ValueError("serious result nonclaims mismatch")
    elif schema == HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1:
        _exact(payload, failure_fields, "serious failure")
        if payload.get("passed") is not False or payload.get("decision") != SERIOUS_BLOCK_DECISION or tuple(payload.get("nonclaims", ())) != SERIOUS_FAILURE_NONCLAIMS:
            raise ValueError("serious failure contract mismatch")
        _text(payload.get("stage"), "serious failure stage")
        _text(payload.get("reason"), "serious failure reason")
        pids = payload.get("worker_pids")
        if not isinstance(pids, Sequence) or isinstance(pids, (str, bytes)) or len(pids) > 2 or len(set(pids)) != len(pids) or any(not isinstance(pid, int) or pid < 1 for pid in pids):
            raise ValueError("serious failure worker PID mismatch")
        if payload.get("final_diagnostics") is not None:
            _parse_diagnostic(payload["final_diagnostics"], require_pass=False)
    else:
        raise ValueError("unsupported serious terminal result schema")
    if payload.get("smoke") is not False or payload.get("serious_runtime_executed") is not True or payload.get("phase8_executed") is not False or payload.get("neutra_executed") is not False or payload.get("cuda_visible_devices") != "-1" or payload.get("jit_compile_false_runtime_executed") is not False:
        raise ValueError("serious terminal execution boundary mismatch")
    for name in ("serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "preflight_before_runtime_artifact_hash", "config_hash"):
        _sha(payload.get(name), name)
    if payload.get("config_hash") != SERIOUS_CONFIG_HASH or payload.get(
        "preflight_before_runtime_artifact_hash"
    ) != PHASE5_PREFLIGHT_ARTIFACT_HASH:
        raise ValueError("serious terminal config/preflight hash mismatch")
    from bayesfilter.inference.hmc_identity_adoption import parse_phase5_preflight_report

    preflight = payload.get("preflight_before_runtime")
    parse_phase5_preflight_report(preflight)
    if preflight.get("artifact_hash") != PHASE5_PREFLIGHT_ARTIFACT_HASH:
        raise ValueError("serious terminal embedded preflight mismatch")
    elapsed = _finite_number(payload.get("elapsed_seconds"), "serious elapsed time")
    if elapsed < 0.0:
        raise ValueError("serious elapsed time must be nonnegative")
    _verify_hash(payload, "serious terminal result")
    return payload


def build_serious_output_manifest(*, context: Phase7SeriousLaunchContext, session: SecureSeriousOutputSession) -> Mapping[str, Any]:
    claim = session.read_json("claim_path")
    progress = session.read_json("public_progress_path")
    result = session.read_json("public_result_path")
    parse_serious_launch_claim(claim)
    parse_serious_progress(progress)
    parse_serious_terminal_result(result)
    if canonical_artifact_payload_hash(claim) != canonical_artifact_payload_hash(
        context.claim
    ):
        raise ValueError("serious durable claim differs from captured claim")
    expected_links = {
        "serious_authority_artifact_hash": context.authority["artifact_hash"],
        "serious_launch_claim_artifact_hash": claim["artifact_hash"],
        "serious_proposal_manifest_artifact_hash": context.proposal_manifest[
            "artifact_hash"
        ],
        "preflight_before_runtime_artifact_hash": context.preflight["artifact_hash"],
        "config_hash": context.config.hash,
    }
    for name, expected in expected_links.items():
        if result[name] != expected or progress[name] != expected:
            raise ValueError(f"serious terminal cross-link mismatch: {name}")
    if (
        progress["result_artifact_hash"] != result["artifact_hash"]
        or progress["passed"] is not result["passed"]
    ):
        raise ValueError("serious progress/result mismatch")
    if result["preflight_before_runtime"] != context.preflight:
        raise ValueError("serious result preflight differs from captured preflight")
    if result["passed"] is True and (
        progress["burnin_checks"][-1]["completed_results_per_chain"]
        != result["burnin_results_per_chain"]
        or progress["retained_checks"][-1]["completed_results_per_chain"]
        != result["retained_results_per_chain"]
    ):
        raise ValueError("serious progress/result count mismatch")
    if result["passed"] is True:
        final_summary = progress["retained_checks"][-1]
        final_diagnostic = result["final_diagnostics"]
        for summary_name, diagnostic_name in (
            ("max_rhat", "max_rhat"),
            ("min_bulk_ess", "min_bulk_ess"),
            ("min_tail_ess", "min_tail_ess"),
        ):
            if final_summary[summary_name] != final_diagnostic[diagnostic_name]:
                raise ValueError("serious progress/final diagnostic metric mismatch")
    private_available = session.nonempty("private_samples_path")
    if result["passed"] is True and not private_available:
        raise ValueError("passing serious result requires private samples")
    private_reference = (
        session.file_reference("private_samples_path")
        if private_available
        else None
    )
    if result["passed"] is True and (
        result["private_retained_sample_reference"]["file_sha256"]
        != private_reference["file_sha256"]
        or result["private_retained_sample_reference"]["byte_count"]
        != private_reference["byte_count"]
    ):
        raise ValueError("serious result/private sample exact-byte mismatch")
    if result["passed"] is True:
        from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
            _secure_worker_cache_seal,
        )

        expected_bundle_hash = implementation_source_bundle_hash(
            context.implementation_source_bundle
        )
        runtime = context.proposal["runtime"]
        for worker_index, worker in enumerate(result["worker_metadata"]):
            if worker["child_implementation_source_bundle_hash"] != (
                expected_bundle_hash
            ):
                raise ValueError("serious worker source bundle differs from authority")
            expected_seal = _secure_worker_cache_seal(
                context.config,
                worker_index=worker_index,
                smoke=False,
                target_scope=context.preflight["target_scope"],
                launch_context=context,
            )
            if worker["child_worker_cache_seal_hash"] != expected_seal[
                "artifact_hash"
            ]:
                raise ValueError("serious worker cache seal differs from authority")
            for name in (
                "tensorflow_version",
                "tfp_version",
                "python_version",
                "cuda_visible_devices",
            ):
                if worker[name] != runtime[name]:
                    raise ValueError(f"serious worker {name} differs from authority")
    for role in ("infrastructure_failure_path", "infrastructure_manifest_path"):
        if session.nonempty(role):
            raise ValueError("ordinary serious manifest cannot include infrastructure bytes")
    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "passed": bool(result["passed"]),
            "historical_archive_manifest_reference": context.proposal[
                "historical_archive_manifest_reference"
            ],
            "authority_reference": context.authority_reference,
            "claim_reference": session.artifact_reference("claim_path"),
            "progress_reference": session.artifact_reference("public_progress_path"),
            "result_reference": session.artifact_reference("public_result_path"),
            "log_reference": session.file_reference("log_path"),
            "private_samples_available": private_available,
            "private_samples_reference": private_reference,
            "infrastructure_failure_written": False,
            "infrastructure_failure_reservation_reference": session.file_reference("infrastructure_failure_path"),
            "infrastructure_manifest_written": False,
            "infrastructure_manifest_reservation_reference": session.file_reference("infrastructure_manifest_path"),
        }
    )


def parse_serious_output_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = ("schema", "terminal_manifest", "passed", "historical_archive_manifest_reference", "authority_reference", "claim_reference", "progress_reference", "result_reference", "log_reference", "private_samples_available", "private_samples_reference", "infrastructure_failure_written", "infrastructure_failure_reservation_reference", "infrastructure_manifest_written", "infrastructure_manifest_reservation_reference", "artifact_hash")
    _exact(payload, fields, "serious output manifest")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_OUTPUT_MANIFEST_SCHEMA_V1 or payload.get("terminal_manifest") is not True:
        raise ValueError("serious output manifest identity/status mismatch")
    expected_schemas = {
        "historical_archive_manifest_reference": HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1,
        "authority_reference": HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1,
        "claim_reference": HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1,
        "progress_reference": HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1,
        "result_reference": (
            HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1
            if payload.get("passed") is True
            else HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1
        ),
    }
    if not isinstance(payload.get("passed"), bool):
        raise ValueError("serious output manifest pass flag mismatch")
    for name, schema in expected_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"serious output {name} schema mismatch")
    for name in ("log_reference", "infrastructure_failure_reservation_reference", "infrastructure_manifest_reservation_reference"):
        parse_file_reference(payload[name])
    if payload.get("private_samples_available") is True:
        parse_file_reference(payload["private_samples_reference"])
    elif payload.get("private_samples_reference") is not None:
        raise ValueError("unavailable private samples cannot have a reference")
    if payload.get("passed") is True and payload.get(
        "private_samples_available"
    ) is not True:
        raise ValueError("passing serious output requires private samples")
    if payload.get("infrastructure_failure_written") is not False or payload.get("infrastructure_manifest_written") is not False or payload["infrastructure_failure_reservation_reference"]["byte_count"] != 0 or payload["infrastructure_manifest_reservation_reference"]["byte_count"] != 0:
        raise ValueError("ordinary serious manifest emergency boundary mismatch")
    _verify_hash(payload, "serious output manifest")
    return payload


def verify_serious_output_manifest(
    payload: Mapping[str, Any],
    *,
    context: Phase7SeriousLaunchContext,
    session: SecureSeriousOutputSession,
) -> Mapping[str, Any]:
    parse_serious_output_manifest(payload)
    expected_paths = {
        "historical_archive_manifest_reference": HISTORICAL_ARCHIVE_MANIFEST_PATH,
        "authority_reference": AUTHORITY_PATH,
        "claim_reference": CLAIM_PATH,
        "progress_reference": PUBLIC_PROGRESS_PATH,
        "result_reference": PUBLIC_RESULT_PATH,
    }
    for name, path in expected_paths.items():
        reference = payload[name]
        if reference["resolved_path_sha256"] != hashlib.sha256(
            str(path.resolve()).encode()
        ).hexdigest():
            raise ValueError(f"serious output {name} path mismatch")
    expected = build_serious_output_manifest(context=context, session=session)
    if canonical_artifact_payload_hash(payload) != canonical_artifact_payload_hash(
        expected
    ):
        raise ValueError("serious output manifest differs from current bytes")
    return payload


def build_serious_infrastructure_failure(*, context: Phase7SeriousLaunchContext, stage: str, error: BaseException, primary_result_hash: str | None) -> Mapping[str, Any]:
    attempt1_preserved = False
    try:
        context.consumed_evidence_session.verify()
        attempt1_preserved = True
    except BaseException:
        attempt1_preserved = False
    return _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1,
            "passed": False,
            "decision": SERIOUS_INFRASTRUCTURE_BLOCK_DECISION,
            "serious_authority_artifact_hash": context.authority["artifact_hash"],
            "serious_launch_claim_artifact_hash": context.claim["artifact_hash"],
            "serious_proposal_manifest_artifact_hash": context.proposal_manifest["artifact_hash"],
            "stage": _text(stage, "infrastructure stage"),
            "reason": f"infrastructure_error:{type(error).__name__}",
            "primary_result_artifact_hash": primary_result_hash,
            "attempt1_terminal_evidence_preserved": attempt1_preserved,
            "phase8_executed": False,
            "neutra_executed": False,
            "nonclaims": SERIOUS_INFRASTRUCTURE_NONCLAIMS,
        }
    )


def parse_serious_infrastructure_failure(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = ("schema", "passed", "decision", "serious_authority_artifact_hash", "serious_launch_claim_artifact_hash", "serious_proposal_manifest_artifact_hash", "stage", "reason", "primary_result_artifact_hash", "attempt1_terminal_evidence_preserved", "phase8_executed", "neutra_executed", "nonclaims", "artifact_hash")
    _exact(payload, fields, "serious infrastructure failure")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1 or payload.get("passed") is not False or payload.get("decision") != SERIOUS_INFRASTRUCTURE_BLOCK_DECISION or not isinstance(payload.get("attempt1_terminal_evidence_preserved"), bool) or payload.get("phase8_executed") is not False or payload.get("neutra_executed") is not False or tuple(payload.get("nonclaims", ())) != SERIOUS_INFRASTRUCTURE_NONCLAIMS:
        raise ValueError("serious infrastructure failure contract mismatch")
    for name in (
        "serious_authority_artifact_hash",
        "serious_launch_claim_artifact_hash",
        "serious_proposal_manifest_artifact_hash",
    ):
        _sha(payload.get(name), name)
    _text(payload.get("stage"), "serious infrastructure stage")
    reason = _text(payload.get("reason"), "serious infrastructure reason")
    if not reason.startswith("infrastructure_error:"):
        raise ValueError("serious infrastructure reason mismatch")
    if payload.get("primary_result_artifact_hash") is not None:
        _sha(payload["primary_result_artifact_hash"], "primary result hash")
    _verify_hash(payload, "serious infrastructure failure")
    return payload


def write_serious_infrastructure_terminal(*, context: Phase7SeriousLaunchContext, session: SecureSeriousOutputSession, stage: str, error: BaseException) -> Mapping[str, Any]:
    primary_hash = None
    if session.available_at_reviewed_path("public_result_path") and session.nonempty("public_result_path"):
        try:
            result = session.read_json("public_result_path")
            parse_serious_terminal_result(result)
            primary_hash = result["artifact_hash"]
        except (OSError, ValueError, json.JSONDecodeError):
            primary_hash = None
    failure = build_serious_infrastructure_failure(context=context, stage=stage, error=error, primary_result_hash=primary_hash)
    session.write_json("infrastructure_failure_path", failure, parser=parse_serious_infrastructure_failure)
    manifest = _embed(
        {
            "schema": HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "passed": False,
            "historical_archive_manifest_reference": context.proposal[
                "historical_archive_manifest_reference"
            ],
            "authority_reference": context.authority_reference,
            "claim_reference": session.artifact_reference("claim_path", require_path_match=False),
            "infrastructure_failure_reference": session.artifact_reference("infrastructure_failure_path", require_path_match=False),
            "public_result_reference": session.file_reference("public_result_path", require_path_match=False) if session.has_role("public_result_path") else None,
            "public_progress_reference": session.file_reference("public_progress_path", require_path_match=False) if session.has_role("public_progress_path") else None,
            "log_reference": session.file_reference("log_path", require_path_match=False) if session.has_role("log_path") else None,
            "private_samples_reference": session.file_reference("private_samples_path", require_path_match=False) if session.has_role("private_samples_path") else None,
        }
    )
    session.write_json(
        "infrastructure_manifest_path",
        manifest,
        parser=parse_serious_infrastructure_manifest,
    )
    return manifest


def parse_serious_infrastructure_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema", "terminal_manifest", "passed",
        "historical_archive_manifest_reference", "authority_reference",
        "claim_reference", "infrastructure_failure_reference",
        "public_result_reference", "public_progress_reference", "log_reference",
        "private_samples_reference", "artifact_hash",
    )
    _exact(payload, fields, "serious infrastructure manifest")
    if payload.get("schema") != HMC_PHASE7_SERIOUS_INFRASTRUCTURE_MANIFEST_SCHEMA_V1 or (
        payload.get("terminal_manifest") is not True or payload.get("passed") is not False
    ):
        raise ValueError("serious infrastructure manifest identity/status mismatch")
    expected_schemas = {
        "historical_archive_manifest_reference": HMC_PHASE7_SERIOUS_ARCHIVE_MANIFEST_SCHEMA_V1,
        "authority_reference": HMC_PHASE7_SERIOUS_AUTHORITY_SCHEMA_V1,
        "claim_reference": HMC_PHASE7_SERIOUS_LAUNCH_CLAIM_SCHEMA_V1,
        "infrastructure_failure_reference": HMC_PHASE7_SERIOUS_INFRASTRUCTURE_FAILURE_SCHEMA_V1,
    }
    for name, schema in expected_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"serious infrastructure {name} schema mismatch")
    for name in (
        "public_result_reference",
        "public_progress_reference",
        "log_reference",
        "private_samples_reference",
    ):
        if payload[name] is not None:
            parse_file_reference(payload[name])
    _verify_hash(payload, "serious infrastructure manifest")
    return payload
