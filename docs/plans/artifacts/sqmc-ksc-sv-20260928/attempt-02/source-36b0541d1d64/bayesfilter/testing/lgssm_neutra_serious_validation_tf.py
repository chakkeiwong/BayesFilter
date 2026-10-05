"""Exact-target LGSSM NeuTra serious-validation campaign utilities.

The module keeps TensorFlow imports inside execution functions so GPU training
and CPU-hidden HMC workers establish their device policy before framework
initialization.  It orchestrates existing BayesFilter target, transport,
training, HMC, and convergence APIs; it does not define a second sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "docs/plans/"
    "bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-"
    "2026-07-13.md"
)
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13"
)
_DEFAULT_PLAN_PATH = PLAN_PATH
_DEFAULT_ARTIFACT_ROOT = ARTIFACT_ROOT
_CAMPAIGN_CONTRACT_OVERRIDE: Mapping[str, Any] | None = None
BASELINE_ROOT = ROOT / (
    "docs/benchmarks/artifacts/"
    "multidim_lgssm_full_estimation_rerun_2026_07_13"
)
MASS_PATH = BASELINE_ROOT / "mass.json"
GEOMETRY_PATH = BASELINE_ROOT / "geometry.json"
COMPARATOR_PATH = BASELINE_ROOT / (
    "phase7_campaign/private/retained_samples.npz"
)

EXPECTED_TARGET_SIGNATURE = (
    "f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30"
)
EXPECTED_ADAPTER_SIGNATURE = (
    "42dc7bad0137fd9c31aa1d618bb4e560f68d1bbe3a7ab4f5ef95e458b2abc985"
)
EXPECTED_MASS_FILE_SHA256 = (
    "54549c9156821536bc4780f0406a7716b0d3fa39a5b5900fa2893cbef2968a95"
)
EXPECTED_MASS_ARTIFACT_HASH = (
    "sha256:2e41adfdebb47e9b949a675671c12ad1261588d6932c27c3c795724abaa355ad"
)
EXPECTED_GEOMETRY_FILE_SHA256 = (
    "bd9b086d60df518b4410c3348b1fd93663fc8dc861428d77b4098aa1e118a87d"
)
EXPECTED_COMPARATOR_FILE_SHA256 = (
    "1b0c05d4ea2981b1be179040d3a52039f05efe6c5b163f9bf7bba64ce2068920"
)

DIMENSION = 18
WORKER_COUNT = 2
CHAINS_PER_WORKER = 2
CHAIN_COUNT = WORKER_COUNT * CHAINS_PER_WORKER
TRAINING_CANDIDATES = {
    "dense_seed1201": (20260713, 1201),
    "dense_seed1202": (20260713, 1202),
}
CANDIDATE_ORDER = ("affine_control", *TRAINING_CANDIDATES)
COMMON_PROBE_SHAPE = (4, DIMENSION)
COMMON_PROBE_TOLERANCES = {
    "transport_max_abs": 1.0e-12,
    "logdet_max_abs": 1.0e-12,
    "value_max_abs": 1.0e-8,
    "score_max_abs": 1.0e-8,
}

PHASE3_STEPS = 8
PHASE3_BATCH_SIZE = 16
PHASE3_SEED = (20260713, 1101)
PHASE4_STEPS = 1000
PHASE4_BATCH_SIZE = 256

TUNING_LEAPFROG_STEPS = 10
TUNING_PROBE_RESULTS = 64
TUNING_PROBE_BURNIN = 128
TUNING_VERIFICATION_RESULTS = 1000
TUNING_VERIFICATION_BURNIN = 1000
TUNING_BASE_STEP_SIZE = 0.1
TUNING_PRIMARY_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
TUNING_REPAIR_SCALES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TUNING_ACCEPTANCE_BAND = (0.60, 0.90)
TUNING_REPAIR_BAND = (0.45, 0.98)

SERIOUS_RESULTS = 4000
SERIOUS_BURNIN = 1000
RHAT_MAX = 1.01
BULK_ESS_MIN = 1000.0
TAIL_ESS_MIN = 400.0
POSTERIOR_AGREEMENT_MAX_Z = 4.0
RECOVERY_MAX_Z = 3.0

TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
CAMPAIGN_NONCLAIMS = (
    "one favorably truth-centered 18D LGSSM fixture only",
    "plain-HMC agreement is not exact-posterior correctness evidence",
    "training loss is explanatory only",
    "acceptance alone cannot admit a kernel",
    "runtime and ESS differences are descriptive only",
    "no sampler superiority, calibration, robustness, or generalization claim",
    "no production or default-readiness claim",
)


class LGSSMNeuTraCampaignError(RuntimeError):
    """Raised when a campaign identity, runtime, or evidence gate fails closed."""


@dataclass(frozen=True)
class StaticCampaignInputs:
    center: np.ndarray
    factor: np.ndarray
    covariance: np.ndarray
    truth: np.ndarray
    parameter_names: tuple[str, ...]
    mass_payload: Mapping[str, Any]
    fixture_payload: Mapping[str, Any]


def campaign_seed_ledger() -> Mapping[str, Any]:
    """Return the immutable seed families for all campaign roles."""

    candidate_rows = {}
    for index, candidate_id in enumerate(CANDIDATE_ORDER):
        offset = 100 * index
        candidate_rows[candidate_id] = {
            "fixed_grid_primary_screen_seed": (20260713, 2101 + offset),
            "fixed_grid_primary_verification_seed": (20260713, 2201 + offset),
            "fixed_grid_repair_screen_seed": (20260713, 2151 + offset),
            "fixed_grid_repair_verification_seed": (20260713, 2251 + offset),
            "serious_seed": (20260713, 3101 + offset),
        }
    return {
        "phase3_training_seed": PHASE3_SEED,
        "phase4_training_seeds": dict(TRAINING_CANDIDATES),
        "candidate_hmc_seeds": candidate_rows,
        "worker_seed_derivation": (
            "(root0, root1 + 1009 * (worker_index + 1))"
        ),
    }


def validate_seed_ledger() -> Mapping[str, Any]:
    ledger = campaign_seed_ledger()
    seeds: list[tuple[int, int]] = [tuple(ledger["phase3_training_seed"])]
    seeds.extend(tuple(value) for value in ledger["phase4_training_seeds"].values())
    for row in ledger["candidate_hmc_seeds"].values():
        for key, value in row.items():
            root = tuple(value)
            if "screen_seed" in key:
                attempt_count = (
                    len(TUNING_REPAIR_SCALES)
                    if "repair" in key
                    else len(TUNING_PRIMARY_SCALES)
                )
                seeds.extend((root[0], root[1] + 10_000 + index) for index in range(attempt_count))
            else:
                seeds.append(root)
    if len(seeds) != len(set(seeds)):
        raise LGSSMNeuTraCampaignError("campaign root seed families must be disjoint")
    derived: list[tuple[int, int]] = []
    for seed in seeds:
        derived.extend(_worker_seeds(seed))
    if len(derived) != len(set(derived)):
        raise LGSSMNeuTraCampaignError("derived worker seed families must be disjoint")
    if set(seeds) & set(derived):
        raise LGSSMNeuTraCampaignError("root and worker-derived seed families overlap")
    return {
        "passed": True,
        "root_seed_count": len(seeds),
        "worker_derived_seed_count": len(derived),
        "ledger": ledger,
    }


def common_probe_points() -> np.ndarray:
    """Return fixed non-random probe points used on GPU and CPU."""

    values = np.arange(np.prod(COMMON_PROBE_SHAPE), dtype=np.float64)
    midpoint = 0.5 * float(values.size - 1)
    return ((values - midpoint) / 97.0).reshape(COMMON_PROBE_SHAPE)


def common_probe_hash() -> str:
    values = np.ascontiguousarray(common_probe_points(), dtype=np.float64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def validate_static_campaign_inputs(
    *, require_comparator: bool = True
) -> StaticCampaignInputs:
    """Validate immutable target geometry and comparator identities without TF."""

    if not PLAN_PATH.is_file():
        raise LGSSMNeuTraCampaignError(f"plan is missing: {PLAN_PATH}")
    if _file_sha256(MASS_PATH) != EXPECTED_MASS_FILE_SHA256:
        raise LGSSMNeuTraCampaignError("immutable mass file SHA-256 mismatch")
    if _file_sha256(GEOMETRY_PATH) != EXPECTED_GEOMETRY_FILE_SHA256:
        raise LGSSMNeuTraCampaignError("immutable geometry file SHA-256 mismatch")
    if require_comparator and _file_sha256(COMPARATOR_PATH) != EXPECTED_COMPARATOR_FILE_SHA256:
        raise LGSSMNeuTraCampaignError("immutable plain-HMC comparator SHA-256 mismatch")

    mass = _read_mapping(MASS_PATH, "mass")
    if mass.get("artifact_hash") != EXPECTED_MASS_ARTIFACT_HASH:
        raise LGSSMNeuTraCampaignError("embedded mass artifact hash mismatch")
    if mass.get("passed") is not True or tuple(mass.get("vetoes", ())) != ():
        raise LGSSMNeuTraCampaignError("mass artifact did not pass its original gate")
    center = np.asarray(mass.get("center"), dtype=np.float64)
    factor = np.asarray(mass.get("factor"), dtype=np.float64)
    covariance = np.asarray(mass.get("mass_covariance"), dtype=np.float64)
    if center.shape != (DIMENSION,):
        raise LGSSMNeuTraCampaignError("mass center shape mismatch")
    if factor.shape != (DIMENSION, DIMENSION):
        raise LGSSMNeuTraCampaignError("mass factor shape mismatch")
    if covariance.shape != (DIMENSION, DIMENSION):
        raise LGSSMNeuTraCampaignError("mass covariance shape mismatch")
    if not all(np.all(np.isfinite(value)) for value in (center, factor, covariance)):
        raise LGSSMNeuTraCampaignError("mass arrays must be finite")
    residual = float(np.max(np.abs(factor @ factor.T - covariance)))
    if residual > 1.0e-12:
        raise LGSSMNeuTraCampaignError("mass factor does not reconstruct covariance")
    recorded = float(mass.get("factor_covariance_max_abs_error", float("inf")))
    if not math.isclose(residual, recorded, rel_tol=0.0, abs_tol=1.0e-16):
        raise LGSSMNeuTraCampaignError("mass reconstruction residual drifted")

    fixture_path = ROOT / str(
        mass.get(
            "config_path",
            "docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json",
        )
    )
    config = _read_mapping(fixture_path, "campaign config")
    fixture = _read_mapping(
        ROOT / str(config["truth_and_data"]["artifact_path"]), "fixture"
    )
    truth = np.asarray(fixture.get("raw_truth"), dtype=np.float64)
    names = tuple(str(item) for item in fixture.get("parameter_names", ()))
    if truth.shape != (DIMENSION,) or not np.all(np.isfinite(truth)):
        raise LGSSMNeuTraCampaignError("fixture raw truth is invalid")
    if names != tuple(str(item) for item in mass.get("parameter_names", ())):
        raise LGSSMNeuTraCampaignError("mass/fixture parameter order mismatch")
    if not np.array_equal(center, truth):
        raise LGSSMNeuTraCampaignError("declared truth-centered affine center drifted")
    validate_seed_ledger()
    return StaticCampaignInputs(
        center=center,
        factor=factor,
        covariance=covariance,
        truth=truth,
        parameter_names=names,
        mass_payload=mass,
        fixture_payload=fixture,
    )


def campaign_contract_payload() -> Mapping[str, Any]:
    """Return fixed Phase 3-6 execution policy before observing results."""

    if _CAMPAIGN_CONTRACT_OVERRIDE is not None:
        return _json_ready(_CAMPAIGN_CONTRACT_OVERRIDE)

    return {
        "schema": "bayesfilter.lgssm_neutra_serious_validation_contract.v1",
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": EXPECTED_ADAPTER_SIGNATURE,
        "dimension": DIMENSION,
        "training": {
            "phase3_steps": PHASE3_STEPS,
            "phase3_batch_size": PHASE3_BATCH_SIZE,
            "phase4_steps": PHASE4_STEPS,
            "phase4_batch_size": PHASE4_BATCH_SIZE,
            "stage_count": 3,
            "hidden_layers": (18, 18),
            "activation": "elu",
            "s_max": 1.0,
            "jit_compile": True,
            "device": "/GPU:0",
        },
        "tuning": {
            "leapfrog_steps": TUNING_LEAPFROG_STEPS,
            "probe_results_per_chain": TUNING_PROBE_RESULTS,
            "probe_burnin": TUNING_PROBE_BURNIN,
            "verification_results_per_chain": TUNING_VERIFICATION_RESULTS,
            "verification_burnin": TUNING_VERIFICATION_BURNIN,
            "base_step_size": TUNING_BASE_STEP_SIZE,
            "primary_scales": TUNING_PRIMARY_SCALES,
            "one_repair_scales": TUNING_REPAIR_SCALES,
            "acceptance_band": TUNING_ACCEPTANCE_BAND,
            "repair_band": TUNING_REPAIR_BAND,
            "modern_rhat_max": RHAT_MAX,
        },
        "serious": {
            "results_per_chain": SERIOUS_RESULTS,
            "burnin": SERIOUS_BURNIN,
            "rhat_max": RHAT_MAX,
            "bulk_ess_min": BULK_ESS_MIN,
            "tail_ess_min": TAIL_ESS_MIN,
            "posterior_agreement_max_combined_mcse": POSTERIOR_AGREEMENT_MAX_Z,
            "recovery_max_posterior_sd": RECOVERY_MAX_Z,
        },
        "runtime": {
            "training": "trusted_gpu_xla_float64",
            "hmc": "cpu_hidden_two_persistent_workers_two_chains_each_xla_float64",
            "worker_count": WORKER_COUNT,
            "chains_per_worker": CHAINS_PER_WORKER,
            "jit_compile": True,
        },
        "seed_ledger": campaign_seed_ledger(),
        "immutable_inputs": {
            "mass_file_sha256": EXPECTED_MASS_FILE_SHA256,
            "geometry_file_sha256": EXPECTED_GEOMETRY_FILE_SHA256,
            "comparator_file_sha256": EXPECTED_COMPARATOR_FILE_SHA256,
        },
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }


def write_campaign_contract() -> Path:
    validate_static_campaign_inputs()
    path = ARTIFACT_ROOT / "campaign_contract.json"
    payload = dict(campaign_contract_payload())
    payload["contract_hash"] = _stable_json_hash(payload)
    _write_new_json(path, payload)
    return path


def configure_execution_context(
    *,
    plan_path: str | Path,
    artifact_root: str | Path,
    contract_payload: Mapping[str, Any],
) -> None:
    """Bind the reusable Phase 5/6 runner to one explicit campaign root."""

    global PLAN_PATH, ARTIFACT_ROOT, _CAMPAIGN_CONTRACT_OVERRIDE
    resolved_plan = Path(plan_path).resolve()
    resolved_root = Path(artifact_root).resolve()
    try:
        resolved_plan.relative_to(ROOT)
        resolved_root.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("campaign context paths must remain inside the repository") from exc
    if not resolved_plan.is_file():
        raise ValueError(f"campaign plan is missing: {resolved_plan}")
    if not isinstance(contract_payload, Mapping):
        raise TypeError("contract_payload must be a mapping")
    PLAN_PATH = resolved_plan
    ARTIFACT_ROOT = resolved_root
    _CAMPAIGN_CONTRACT_OVERRIDE = _json_ready(contract_payload)


def reset_execution_context() -> None:
    """Restore the historical campaign context, primarily for focused tests."""

    global PLAN_PATH, ARTIFACT_ROOT, _CAMPAIGN_CONTRACT_OVERRIDE
    PLAN_PATH = _DEFAULT_PLAN_PATH
    ARTIFACT_ROOT = _DEFAULT_ARTIFACT_ROOT
    _CAMPAIGN_CONTRACT_OVERRIDE = None


def candidate_payload_path(candidate_id: str) -> Path:
    candidate = _validate_candidate_id(candidate_id)
    result_path = ARTIFACT_ROOT / "phase4" / candidate / "result.json"
    if result_path.is_file():
        result = _read_mapping(result_path, f"Phase 4 {candidate} result")
        reference = result.get("payload")
        if not isinstance(reference, Mapping) or not reference.get("path"):
            raise LGSSMNeuTraCampaignError(
                f"Phase 4 {candidate} result lacks a payload reference"
            )
        return (ROOT / str(reference["path"])).resolve()
    if candidate == "affine_control":
        return ARTIFACT_ROOT / "phase4" / candidate / "frozen_transport.json"
    return ARTIFACT_ROOT / "phase4" / candidate / "training" / "frozen_transport.json"


def candidate_phase4_result_path(candidate_id: str) -> Path:
    candidate = _validate_candidate_id(candidate_id)
    return ARTIFACT_ROOT / "phase4" / candidate / "result.json"


def candidate_phase5_result_path(candidate_id: str) -> Path:
    candidate = _validate_candidate_id(candidate_id)
    return ARTIFACT_ROOT / "phase5" / candidate / "result.json"


def candidate_phase6_result_path(candidate_id: str) -> Path:
    candidate = _validate_candidate_id(candidate_id)
    return ARTIFACT_ROOT / "phase6" / candidate / "result.json"


def _validate_candidate_id(candidate_id: str) -> str:
    value = str(candidate_id)
    if value not in CANDIDATE_ORDER:
        raise ValueError(f"candidate_id must be one of {CANDIDATE_ORDER}")
    return value


def _worker_seeds(seed: Sequence[int]) -> tuple[tuple[int, int], ...]:
    root = tuple(int(item) for item in seed)
    if len(root) != 2:
        raise ValueError("seed must contain exactly two integers")
    return tuple((root[0], root[1] + 1009 * (index + 1)) for index in range(WORKER_COUNT))


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise LGSSMNeuTraCampaignError(f"{label} file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise LGSSMNeuTraCampaignError(f"{label} must be a JSON mapping")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _dirty_worktree_summary() -> Mapping[str, Any]:
    result = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = tuple(line for line in result.stdout.splitlines() if line)
    return {
        "dirty": bool(rows),
        "entry_count": len(rows),
        "shared_worktree_disclosure": (
            "The repository contains unrelated concurrent-agent changes; "
            "this campaign preserves them and binds its own artifact hashes."
        ),
    }


def _runtime_base_manifest() -> Mapping[str, Any]:
    return {
        "git_commit": _git_commit(),
        "dirty_worktree": _dirty_worktree_summary(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "command": tuple(sys.argv),
        "working_directory": str(Path.cwd()),
        "platform": platform.platform(),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "campaign_contract_path": str(
            (ARTIFACT_ROOT / "campaign_contract.json").relative_to(ROOT)
        ),
        "campaign_contract_hash": _stable_json_hash(campaign_contract_payload()),
    }


def run_phase3_gpu_canary() -> Mapping[str, Any]:
    """Run one partial step, deterministic resume, freeze, and GPU/XLA probe."""

    inputs = validate_static_campaign_inputs()
    _require_contract()
    phase_root = ARTIFACT_ROOT / "phase3"
    result_path = phase_root / "result.json"
    if result_path.exists():
        raise LGSSMNeuTraCampaignError("Phase 3 result already exists")

    tf, bundle = _trusted_gpu_bundle()
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
        train_plain_dense_iaf,
    )

    training_dir = phase_root / "canary_training"
    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=DIMENSION,
        affine_center=inputs.center,
        affine_factor=inputs.factor,
        output_dir=training_dir,
        seed=PHASE3_SEED,
        hidden_layers=(18, 18),
        stage_count=3,
        activation="elu",
        s_max=1.0,
        steps=PHASE3_STEPS,
        batch_size=PHASE3_BATCH_SIZE,
        learning_rate=1.0e-3,
        final_learning_rate_fraction=0.1,
        clip_norm=10.0,
        checkpoint_every=1,
        heartbeat_every=1,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    start = time.monotonic()
    partial = train_plain_dense_iaf(
        adapter=bundle.adapter,
        config=config,
        stop_after_steps=1,
    )
    resumed = train_plain_dense_iaf(
        adapter=bundle.adapter,
        config=config,
        resume_from=partial.state_path,
        freeze_transport_id="lgssm-neutra-phase3-canary",
    )
    if resumed.completed_steps != PHASE3_STEPS or not resumed.resumed:
        raise LGSSMNeuTraCampaignError("Phase 3 resume did not complete exactly")
    if resumed.frozen_payload_path is None:
        raise LGSSMNeuTraCampaignError("Phase 3 frozen payload is missing")

    flow = restore_plain_dense_iaf_flow(config=config, state_path=resumed.state_path)
    loaded = _load_candidate_artifact(resumed.frozen_payload_path, bundle.target_signature)
    parity = _trainable_frozen_parity(flow, loaded.transport, common_probe_points())
    _assert_parity(parity)
    probe = _gpu_frozen_probe(
        tf=tf,
        bundle=bundle,
        loaded=loaded,
        z=common_probe_points(),
        target_scope="lgssm_neutra_phase3_canary",
    )
    heldout = _gpu_heldout_summary(
        tf=tf,
        bundle=bundle,
        loaded=loaded,
        seed=(20260713, 1111),
        batch_size=64,
        target_scope="lgssm_neutra_phase3_canary_heldout",
    )
    records = tuple(resumed.records)
    if not records or not all(row.get("target_status_all_valid") for row in records):
        raise LGSSMNeuTraCampaignError("Phase 3 training status telemetry did not pass")
    result = {
        "schema": "bayesfilter.lgssm_neutra_phase3_gpu_canary_result.v1",
        "phase": 3,
        "passed": True,
        "decision": "PASS_PHASE3_TRUSTED_EXACT_TARGET_GPU_XLA_CANARY",
        "target_signature": bundle.target_signature,
        "adapter_signature": bundle.adapter.adapter_signature(),
        "config_hash": config.config_hash,
        "seed": PHASE3_SEED,
        "partial_checkpoint": _file_reference(partial.state_path),
        "resumed_checkpoint": _file_reference(resumed.state_path),
        "frozen_payload": _file_reference(resumed.frozen_payload_path),
        "training_state_hash": resumed.state_hash,
        "partial_completed_steps": partial.completed_steps,
        "completed_steps": resumed.completed_steps,
        "resume_verified": resumed.resumed,
        "training_records": records,
        "runtime_metadata": resumed.runtime_metadata,
        "frozen_reload_parity": parity,
        "gpu_fixed_transport_probe": probe,
        "heldout_explanatory_summary": heldout,
        "gpu_manifest": _gpu_manifest(tf),
        "elapsed_seconds": time.monotonic() - start,
        "run_manifest": _runtime_base_manifest(),
        "evidence_role": "engineering_gpu_xla_viability_only",
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(result_path, result)
    return result


def run_phase3_gpu_score_parity_addendum() -> Mapping[str, Any]:
    """Bind frozen explicit-score parity to the unchanged exact 18D canary."""

    inputs = validate_static_campaign_inputs()
    phase3 = _require_phase_pass(ARTIFACT_ROOT / "phase3" / "result.json", phase=3)
    output_path = ARTIFACT_ROOT / "phase3" / "score_parity_addendum.json"
    if output_path.exists():
        raise LGSSMNeuTraCampaignError("Phase 3 score-parity addendum already exists")
    checkpoint_path = _verify_file_reference(
        phase3["resumed_checkpoint"], "Phase 3 resumed checkpoint"
    )
    frozen_path = _verify_file_reference(
        phase3["frozen_payload"], "Phase 3 frozen payload"
    )

    tf, bundle = _trusted_gpu_bundle()
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
        reviewed_value_score_target_fn,
    )
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
    )

    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=DIMENSION,
        affine_center=inputs.center,
        affine_factor=inputs.factor,
        output_dir=checkpoint_path.parent,
        seed=PHASE3_SEED,
        hidden_layers=(18, 18),
        stage_count=3,
        activation="elu",
        s_max=1.0,
        steps=PHASE3_STEPS,
        batch_size=PHASE3_BATCH_SIZE,
        learning_rate=1.0e-3,
        final_learning_rate_fraction=0.1,
        clip_norm=10.0,
        checkpoint_every=1,
        heartbeat_every=1,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    if config.config_hash != phase3["config_hash"]:
        raise LGSSMNeuTraCampaignError("Phase 3 parity config hash mismatch")
    flow = restore_plain_dense_iaf_flow(config=config, state_path=checkpoint_path)
    loaded = _load_candidate_artifact(frozen_path, bundle.target_signature)
    fixed = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope="lgssm_neutra_phase3_exact_artifact_score_parity",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    target_value = reviewed_value_score_target_fn(
        bundle.adapter, dtype=tf.float64, require_batched=True
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def trainable_reference(z_arg):
        with tf.GradientTape() as tape:
            tape.watch(z_arg)
            theta, logdet = flow.forward_and_logdet(z_arg)
            value = target_value(theta) + logdet
            objective = tf.reduce_sum(value)
        score = tape.gradient(objective, z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return theta, logdet, value, score, status

    @tf.function(jit_compile=True, reduce_retracing=True)
    def frozen_explicit(z_arg):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = fixed.log_prob_and_grad_batch(z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return theta, logdet, value, score, status

    generator = np.random.default_rng(20260714)
    heldout = generator.normal(size=(16, DIMENSION)).astype(np.float64)
    probes = np.concatenate((common_probe_points(), heldout), axis=0)
    probe_hash = hashlib.sha256(
        np.ascontiguousarray(probes).tobytes(order="C")
    ).hexdigest()
    with tf.device("/GPU:0"):
        z = tf.constant(probes, dtype=tf.float64)
        reference = trainable_reference(z)
        explicit = frozen_explicit(z)
    reference_tensors = reference[:4]
    explicit_tensors = explicit[:4]
    devices = tuple(
        str(item.device) for item in (*reference_tensors, *explicit_tensors)
    )
    if not all("GPU" in device.upper() for device in devices):
        raise LGSSMNeuTraCampaignError("Phase 3 score parity fell back from GPU")
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(item)).numpy())
        for item in (*reference_tensors, *explicit_tensors)
    ):
        raise LGSSMNeuTraCampaignError("Phase 3 score parity is nonfinite")
    for status in (reference[4], explicit[4]):
        if not bool(
            tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
            and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        ):
            raise LGSSMNeuTraCampaignError("Phase 3 score parity target status failed")
    parity = _score_parity_summary(
        reference={
            "theta": reference[0].numpy(),
            "logdet": reference[1].numpy(),
            "value": reference[2].numpy(),
            "score": reference[3].numpy(),
        },
        explicit={
            "theta": explicit[0].numpy(),
            "logdet": explicit[1].numpy(),
            "value": explicit[2].numpy(),
            "score": explicit[3].numpy(),
        },
    )
    if parity["passed"] is not True:
        raise LGSSMNeuTraCampaignError("Phase 3 frozen explicit-score parity failed")
    payload = {
        "schema": "bayesfilter.lgssm_neutra_phase3_score_parity_addendum.v1",
        "phase": 3,
        "passed": True,
        "decision": "PASS_PHASE3_EXACT_ARTIFACT_FROZEN_SCORE_PARITY",
        "phase3_result": _file_reference(
            ARTIFACT_ROOT / "phase3" / "result.json"
        ),
        "phase3_result_artifact_hash": phase3["artifact_hash"],
        "checkpoint": _file_reference(checkpoint_path),
        "frozen_payload": _file_reference(frozen_path),
        "training_state_hash": phase3["training_state_hash"],
        "config_hash": config.config_hash,
        "target_signature": bundle.target_signature,
        "adapter_signature": bundle.adapter.adapter_signature(),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "fixed_transport_adapter_signature": fixed.adapter_signature(),
        "probe_shape": tuple(int(item) for item in probes.shape),
        "probe_hash": probe_hash,
        "probe_composition": "4 fixed common plus 16 independent heldout points",
        "parity": parity,
        "target_status_all_valid": True,
        "output_devices": devices,
        "all_outputs_on_gpu": True,
        "jit_compile": True,
        "reference_method": (
            "diagnostic GradientTape through restored trainable flow and reviewed "
            "exact-target custom-gradient bridge"
        ),
        "frozen_method": "explicit frozen transport pullback plus logdet score",
        "runtime_gradient_tape_used_by_hmc": False,
        "evidence_role": "exact_artifact_score_correctness_handoff_gate",
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    payload = _with_artifact_hash(payload)
    _write_new_json(output_path, payload)
    return payload


def run_phase4_gpu_candidate(candidate_id: str) -> Mapping[str, Any]:
    """Materialize one immutable affine or trained dense-IAF candidate."""

    candidate = _validate_candidate_id(candidate_id)
    inputs = validate_static_campaign_inputs()
    _require_phase_pass(ARTIFACT_ROOT / "phase3" / "result.json", phase=3)
    result_path = candidate_phase4_result_path(candidate)
    if result_path.exists() or candidate_payload_path(candidate).exists():
        raise LGSSMNeuTraCampaignError(f"Phase 4 candidate already exists: {candidate}")

    tf, bundle = _trusted_gpu_bundle()
    start = time.monotonic()
    repair = None
    if candidate == "affine_control":
        payload_path = candidate_payload_path(candidate)
        payload = _affine_control_payload(inputs, target_signature=bundle.target_signature)
        _write_new_json(payload_path, payload)
        state_hash = str(payload["training_state_hash"])
        training = {
            "executed": False,
            "reason": "declared_fixed_full_affine_control",
            "truth_centered_near_oracle": True,
        }
        loaded = _load_candidate_artifact(payload_path, bundle.target_signature)
        parity = _affine_frozen_parity(
            loaded.transport,
            center=inputs.center,
            factor=inputs.factor,
            z=common_probe_points(),
        )
        heldout_seed = (20260713, 1300)
    else:
        from bayesfilter.inference.neutra_training import (
            NeuTraTrainingError,
            PlainDenseIAFTrainingConfig,
            restore_plain_dense_iaf_flow,
            train_plain_dense_iaf,
        )

        seed = TRAINING_CANDIDATES[candidate]
        training_dir = candidate_payload_path(candidate).parent
        config = PlainDenseIAFTrainingConfig(
            target_signature=bundle.target_signature,
            dimension=DIMENSION,
            affine_center=inputs.center,
            affine_factor=inputs.factor,
            output_dir=training_dir,
            seed=seed,
            hidden_layers=(18, 18),
            stage_count=3,
            activation="elu",
            s_max=1.0,
            steps=PHASE4_STEPS,
            batch_size=PHASE4_BATCH_SIZE,
            learning_rate=1.0e-3,
            final_learning_rate_fraction=0.1,
            clip_norm=10.0,
            checkpoint_every=50,
            heartbeat_every=10,
            jit_compile=True,
            device="/GPU:0",
            require_gpu=True,
        )
        try:
            trained = train_plain_dense_iaf(
                adapter=bundle.adapter,
                config=config,
                freeze_transport_id=f"lgssm-neutra-{candidate}",
            )
        except NeuTraTrainingError as exc:
            if not _training_error_allows_lower_rate_repair(exc):
                raise
            checkpoints = sorted(training_dir.glob("checkpoint_step_*.json"))
            if not checkpoints:
                raise LGSSMNeuTraCampaignError(
                    "nonfinite training occurred before a valid repair checkpoint"
                ) from exc
            repair_dir = training_dir.parent / "training_lower_rate_repair"
            repair_config = PlainDenseIAFTrainingConfig(
                **{
                    **config.__dict__,
                    "output_dir": repair_dir,
                    "learning_rate": 5.0e-4,
                }
            )
            trained = train_plain_dense_iaf(
                adapter=bundle.adapter,
                config=repair_config,
                resume_repair_from=checkpoints[-1],
                freeze_transport_id=f"lgssm-neutra-{candidate}-lower-rate-repair",
            )
            config = repair_config
            repair = {
                "trigger": str(exc),
                "parent_checkpoint": _file_reference(checkpoints[-1]),
                "lower_learning_rate": repair_config.learning_rate,
            }
        if trained.frozen_payload_path is None:
            raise LGSSMNeuTraCampaignError("completed dense training did not freeze")
        payload_path = trained.frozen_payload_path
        loaded = _load_candidate_artifact(payload_path, bundle.target_signature)
        flow = restore_plain_dense_iaf_flow(config=config, state_path=trained.state_path)
        parity = _trainable_frozen_parity(flow, loaded.transport, common_probe_points())
        records = tuple(trained.records)
        if not records or not all(row.get("target_status_all_valid") for row in records):
            raise LGSSMNeuTraCampaignError("Phase 4 target-status telemetry failed")
        state_hash = trained.state_hash
        training = {
            "executed": True,
            "config": config.payload(),
            "config_hash": config.config_hash,
            "seed": seed,
            "completed_steps": trained.completed_steps,
            "checkpoint": _file_reference(trained.state_path),
            "progress": _file_reference(trained.progress_path),
            "latest": _file_reference(trained.latest_path),
            "records": records,
            "runtime_metadata": trained.runtime_metadata,
            "repair": repair,
        }
        heldout_seed = (20260713, 1301 + tuple(TRAINING_CANDIDATES).index(candidate))

    _assert_parity(parity)
    probe = _gpu_frozen_probe(
        tf=tf,
        bundle=bundle,
        loaded=loaded,
        z=common_probe_points(),
        target_scope=f"lgssm_neutra_phase4_{candidate}",
    )
    heldout = _gpu_heldout_summary(
        tf=tf,
        bundle=bundle,
        loaded=loaded,
        seed=heldout_seed,
        batch_size=PHASE4_BATCH_SIZE,
        target_scope=f"lgssm_neutra_phase4_{candidate}_heldout",
    )
    result = {
        "schema": "bayesfilter.lgssm_neutra_phase4_candidate_result.v1",
        "phase": 4,
        "candidate_id": candidate,
        "passed": True,
        "decision": "NOMINATE_ENGINEERING_VALID_FROZEN_CANDIDATE",
        "target_signature": bundle.target_signature,
        "adapter_signature": bundle.adapter.adapter_signature(),
        "payload": _file_reference(payload_path),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "training_state_hash": state_hash,
        "training": training,
        "frozen_reload_parity": parity,
        "gpu_fixed_transport_probe": probe,
        "heldout_explanatory_summary": heldout,
        "gpu_manifest": _gpu_manifest(tf),
        "elapsed_seconds": time.monotonic() - start,
        "run_manifest": _runtime_base_manifest(),
        "evidence_role": "engineering_nomination_only_not_candidate_ranking",
        "truth_centered_affine_geometry": True,
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(result_path, result)
    return result


def finalize_phase4() -> Mapping[str, Any]:
    """Close Phase 4 only after all predeclared candidates have results."""

    rows = []
    for candidate in CANDIDATE_ORDER:
        row = _read_mapping(candidate_phase4_result_path(candidate), candidate)
        if row.get("candidate_id") != candidate:
            raise LGSSMNeuTraCampaignError("Phase 4 candidate identity mismatch")
        rows.append(row)
    viable = tuple(row["candidate_id"] for row in rows if row.get("passed") is True)
    if not viable:
        raise LGSSMNeuTraCampaignError("Phase 4 produced no viable candidate")
    result = {
        "schema": "bayesfilter.lgssm_neutra_phase4_result.v1",
        "phase": 4,
        "passed": True,
        "decision": "PASS_PHASE4_CANDIDATE_FREEZE_AND_HANDOFF",
        "candidate_order": CANDIDATE_ORDER,
        "viable_candidates": viable,
        "candidate_results": tuple(
            {
                "candidate_id": row["candidate_id"],
                "result": _file_reference(candidate_phase4_result_path(row["candidate_id"])),
                "result_artifact_hash": row["artifact_hash"],
                "transport_hash": row["transport_hash"],
            }
            for row in rows
        ),
        "loss_used_for_selection": False,
        "truth_centered_affine_geometry": True,
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(ARTIFACT_ROOT / "phase4" / "result.json", result)
    return result


def _trusted_gpu_bundle():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise LGSSMNeuTraCampaignError("GPU stage cannot run with CUDA hidden")
    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise LGSSMNeuTraCampaignError("trusted TensorFlow GPU is unavailable")
    for device in physical:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
        raise LGSSMNeuTraCampaignError("exact-target adapter signature mismatch")
    return tf, bundle


def _gpu_manifest(tf: Any) -> Mapping[str, Any]:
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "trust_basis": TRUST_BASIS,
        "tensorflow_version": tf.__version__,
        "tensorflow_build_info": tf.sysconfig.get_build_info(),
        "physical_gpus": tuple(str(item) for item in tf.config.list_physical_devices("GPU")),
        "logical_gpus": tuple(str(item) for item in tf.config.list_logical_devices("GPU")),
        "soft_device_placement": tf.config.get_soft_device_placement(),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "nvidia_smi_rows": tuple(
            line.strip() for line in query.stdout.splitlines() if line.strip()
        ),
        "training_dtype": "float64",
        "jit_compile": True,
    }


def _affine_control_payload(
    inputs: StaticCampaignInputs, *, target_signature: str
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_artifacts import (
        finalize_dense_iaf_neutra_artifact_payload,
    )

    return finalize_dense_iaf_neutra_artifact_payload(
        {
            "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
            "transport_id": "lgssm-neutra-affine-control-truth-centered",
            "dimension": DIMENSION,
            "target_signature": target_signature,
            "log_jacobian_available": True,
            "component_order": ("fixed_full_affine",),
            "components": (
                {
                    "component_id": "fixed_full_affine",
                    "kind": "affine",
                    "dim": DIMENSION,
                    "dtype": "float64",
                    "offset": inputs.center.tolist(),
                    "L_np": inputs.factor.tolist(),
                },
            ),
            "training_state_hash": EXPECTED_MASS_ARTIFACT_HASH,
            "truth_centered_near_oracle": True,
            "nonclaims": CAMPAIGN_NONCLAIMS,
        }
    )


def _load_candidate_artifact(path: Path, target_signature: str):
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact

    return load_frozen_neutra_artifact(
        _read_mapping(path, "frozen candidate"),
        expected_target_signature=target_signature,
    )


def _trainable_frozen_parity(flow: Any, frozen: Any, z: np.ndarray) -> Mapping[str, Any]:
    values = np.asarray(z, dtype=np.float64)
    expected, expected_logdet = flow.forward_and_logdet(values)
    actual = frozen.forward_batch(values)
    actual_logdet = frozen.log_abs_det_jacobian_batch(values)
    return {
        "transport_max_abs": float(
            np.max(np.abs(np.asarray(expected.numpy()) - np.asarray(actual.numpy())))
        ),
        "logdet_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(expected_logdet.numpy())
                    - np.asarray(actual_logdet.numpy())
                )
            )
        ),
        "probe_hash": common_probe_hash(),
    }


def _affine_frozen_parity(
    frozen: Any, *, center: np.ndarray, factor: np.ndarray, z: np.ndarray
) -> Mapping[str, Any]:
    values = np.asarray(z, dtype=np.float64)
    expected = center + values @ factor.T
    expected_logdet = np.linalg.slogdet(factor)[1]
    actual = np.asarray(frozen.forward_batch(values).numpy())
    actual_logdet = np.asarray(frozen.log_abs_det_jacobian_batch(values).numpy())
    return {
        "transport_max_abs": float(np.max(np.abs(expected - actual))),
        "logdet_max_abs": float(np.max(np.abs(actual_logdet - expected_logdet))),
        "probe_hash": common_probe_hash(),
    }


def _assert_parity(parity: Mapping[str, Any]) -> None:
    if float(parity["transport_max_abs"]) > COMMON_PROBE_TOLERANCES[
        "transport_max_abs"
    ]:
        raise LGSSMNeuTraCampaignError("frozen transport reload parity failed")
    if float(parity["logdet_max_abs"]) > COMMON_PROBE_TOLERANCES["logdet_max_abs"]:
        raise LGSSMNeuTraCampaignError("frozen logdet reload parity failed")


def _gpu_frozen_probe(
    *, tf: Any, bundle: Any, loaded: Any, z: np.ndarray, target_scope: str
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=target_scope,
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = adapter.log_prob_and_grad_batch(z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return theta, logdet, value, score, status

    with tf.device("/GPU:0"):
        outputs = compiled(tf.constant(z, dtype=tf.float64))
        outputs2 = compiled(tf.constant(z, dtype=tf.float64))
    theta, logdet, value, score, status = outputs
    flat_outputs = (theta, logdet, value, score, *status.values(), *outputs2[:4])
    devices = tuple(str(item.device) for item in flat_outputs)
    if not all("GPU" in device.upper() for device in devices):
        raise LGSSMNeuTraCampaignError("compiled frozen probe fell back from GPU")
    if not all(bool(tf.reduce_all(tf.math.is_finite(item)).numpy()) for item in outputs[:4]):
        raise LGSSMNeuTraCampaignError("compiled frozen probe is nonfinite")
    status_valid = bool(
        tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
    )
    if not status_valid:
        raise LGSSMNeuTraCampaignError("compiled frozen probe target status failed")
    return {
        "probe_hash": common_probe_hash(),
        "probe_shape": tuple(int(item) for item in np.asarray(z).shape),
        "theta": theta.numpy(),
        "logdet": logdet.numpy(),
        "value": value.numpy(),
        "score": score.numpy(),
        "status": status,
        "output_devices": devices,
        "all_outputs_on_gpu": True,
        "jit_compile": True,
        "second_call_exact": bool(
            all(
                np.array_equal(np.asarray(first.numpy()), np.asarray(second.numpy()))
                for first, second in zip(outputs[:4], outputs2[:4])
            )
        ),
        "fixed_transport_adapter_signature": adapter.adapter_signature(),
        "transport_hash": loaded.manifest.transport_hash,
    }


def _gpu_heldout_summary(
    *,
    tf: Any,
    bundle: Any,
    loaded: Any,
    seed: tuple[int, int],
    batch_size: int,
    target_scope: str,
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=target_scope,
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = adapter.log_prob_and_grad_batch(z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return value, score, logdet, status

    with tf.device("/GPU:0"):
        z = tf.random.stateless_normal(
            (int(batch_size), DIMENSION), seed=seed, dtype=tf.float64
        )
        value, score, logdet, status = compiled(z)
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(item)).numpy())
        for item in (value, score, logdet)
    ):
        raise LGSSMNeuTraCampaignError("held-out transport diagnostic is nonfinite")
    status_valid = bool(
        tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
    )
    if not status_valid:
        raise LGSSMNeuTraCampaignError("held-out target status failed")
    force_norm = tf.linalg.norm(score, axis=-1)
    objective = -(value + logdet)
    return {
        "seed": seed,
        "batch_size": int(batch_size),
        "reverse_kl_objective_mean": float(tf.reduce_mean(objective).numpy()),
        "reverse_kl_objective_sd": float(tf.math.reduce_std(objective).numpy()),
        "transformed_force_norm_mean": float(tf.reduce_mean(force_norm).numpy()),
        "transformed_force_norm_max": float(tf.reduce_max(force_norm).numpy()),
        "target_status_all_valid": True,
        "metric_role": "explanatory_only_not_candidate_selection",
        "jit_compile": True,
        "device": str(value.device),
    }


def _file_reference(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "file_sha256": _file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _with_artifact_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = f"sha256:{_stable_json_hash(result)}"
    result["artifact_hash_semantics"] = (
        "stable_json_sha256_excluding_artifact_hash_fields"
    )
    return result


def _require_contract() -> Mapping[str, Any]:
    path = ARTIFACT_ROOT / "campaign_contract.json"
    value = _read_mapping(path, "campaign contract")
    expected = dict(campaign_contract_payload())
    expected["contract_hash"] = _stable_json_hash(expected)
    if _json_ready(value) != _json_ready(expected):
        raise LGSSMNeuTraCampaignError("campaign contract mismatch")
    return value


def _require_phase_pass(path: Path, *, phase: int) -> Mapping[str, Any]:
    value = _read_mapping(path, f"Phase {phase} result")
    if value.get("phase") != phase or value.get("passed") is not True:
        raise LGSSMNeuTraCampaignError(f"Phase {phase} did not pass")
    return value


def _training_error_allows_lower_rate_repair(error: BaseException) -> bool:
    text = str(error)
    return "nonfinite training diagnostic" in text or "nonfinite exact target value" in text


class TwoWorkerFixedTransportHMCRunner:
    """Persistent two-worker CPU-hidden XLA runner for one frozen candidate."""

    def __init__(self, *, candidate_id: str, archive_dir: Path | None = None) -> None:
        self.candidate_id = _validate_candidate_id(candidate_id)
        self.payload_path = candidate_payload_path(self.candidate_id).resolve()
        if not self.payload_path.is_file():
            raise LGSSMNeuTraCampaignError(
                f"candidate payload is missing: {self.payload_path}"
            )
        self.payload_file_sha256 = _file_sha256(self.payload_path)
        self.archive_dir = None if archive_dir is None else archive_dir.resolve()
        if self.archive_dir is not None:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self._workers: list[tuple[Any, mp.Process]] = []
        for worker_index in range(WORKER_COUNT):
            parent_connection, child_connection = context.Pipe(duplex=True)
            process = context.Process(
                target=_hmc_worker_loop,
                args=(
                    child_connection,
                    worker_index,
                    str(self.payload_path),
                    self.payload_file_sha256,
                    str(PLAN_PATH.relative_to(ROOT)),
                ),
                name=f"lgssm-neutra-{self.candidate_id}-worker-{worker_index}",
            )
            process.start()
            child_connection.close()
            self._workers.append((parent_connection, process))
        try:
            ready = [self._receive(index, "initialize") for index in range(WORKER_COUNT)]
        except BaseException:
            self.close(force=True)
            raise
        self.worker_initialization = tuple(ready)
        signatures = {str(row["artifact_signature"]) for row in ready}
        target_signatures = {str(row["target_signature"]) for row in ready}
        adapter_signatures = {str(row["adapter_signature"]) for row in ready}
        if len(signatures) != 1:
            self.close(force=True)
            raise LGSSMNeuTraCampaignError("CPU workers loaded different artifacts")
        if target_signatures != {EXPECTED_TARGET_SIGNATURE}:
            self.close(force=True)
            raise LGSSMNeuTraCampaignError("CPU worker target signature mismatch")
        if adapter_signatures != {EXPECTED_ADAPTER_SIGNATURE}:
            self.close(force=True)
            raise LGSSMNeuTraCampaignError("CPU worker adapter signature mismatch")
        self.artifact_signature = next(iter(signatures))
        self.call_count = 0
        self.archive_references: list[Mapping[str, Any]] = []
        self.closed = False

    def __enter__(self) -> "TwoWorkerFixedTransportHMCRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)

    def close(self, *, force: bool = False) -> None:
        if getattr(self, "closed", False):
            return
        for connection, process in self._workers:
            if process.is_alive() and not force:
                try:
                    connection.send({"command": "shutdown"})
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in self._workers:
            if process.is_alive() and not force:
                try:
                    response = connection.recv()
                    if response.get("status") != "shutdown":
                        force = True
                except (BrokenPipeError, EOFError):
                    force = True
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()
        self.closed = True

    def probe(self) -> Mapping[str, Any]:
        request = {
            "command": "probe",
            "z": common_probe_points(),
            "probe_hash": common_probe_hash(),
        }
        for connection, _process in self._workers:
            connection.send(request)
        rows = tuple(self._receive(index, "probe") for index in range(WORKER_COUNT))
        first = rows[0]
        for row in rows[1:]:
            for key in ("theta", "logdet", "value", "score"):
                if not np.array_equal(np.asarray(row[key]), np.asarray(first[key])):
                    raise LGSSMNeuTraCampaignError(
                        f"CPU worker probe mismatch for {key}"
                    )
        return {
            "probe_hash": common_probe_hash(),
            "workers": rows,
            "theta": first["theta"],
            "logdet": first["logdet"],
            "value": first["value"],
            "score": first["score"],
            "all_workers_exact": True,
        }

    def __call__(self, adapter: Any, initial_state: Any, config: Any):
        from bayesfilter.inference.hmc import FullChainHMCRunResult

        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (CHAIN_COUNT, DIMENSION):
            raise LGSSMNeuTraCampaignError(
                "two-worker initial state must have shape [4, 18]"
            )
        capability = adapter.value_score_capability()
        if not capability.is_accepted_full_chain_xla_diagnostic_authority:
            raise LGSSMNeuTraCampaignError(
                "parent transformed adapter lacks full-chain XLA authority"
            )
        if not bool(config.use_xla) or config.chain_execution_mode != "tf_function":
            raise LGSSMNeuTraCampaignError("two-worker HMC requires tf.function XLA")
        if config.target_status_trace_policy != "per_chain_step":
            raise LGSSMNeuTraCampaignError(
                "campaign HMC requires per-chain-step target status"
            )
        worker_seeds = _worker_seeds(config.seed)
        config_payload = config.signature_payload()
        for worker_index, (connection, _process) in enumerate(self._workers):
            worker_config = dict(config_payload)
            worker_config["seed"] = worker_seeds[worker_index]
            connection.send(
                {
                    "command": "run_hmc",
                    "config": worker_config,
                    "initial_state": state[
                        worker_index * CHAINS_PER_WORKER : (worker_index + 1)
                        * CHAINS_PER_WORKER
                    ],
                }
            )
        rows = tuple(self._receive(index, "run_hmc") for index in range(WORKER_COUNT))
        parent_signature = adapter.adapter_signature()
        if any(
            row["metadata"].get("fixed_transport_adapter_signature")
            != parent_signature
            for row in rows
        ):
            raise LGSSMNeuTraCampaignError(
                "parent/worker transformed adapter signature mismatch"
            )
        samples = np.concatenate(
            tuple(np.asarray(row["samples"], dtype=np.float64) for row in rows),
            axis=1,
        )
        trace = _combine_worker_trace(tuple(row["trace"] for row in rows))
        diagnostics = _combined_hmc_diagnostics(samples=samples, trace=trace)
        self.call_count += 1
        metadata = {
            "runtime": "two_persistent_cpu_hidden_workers_tfp_sample_chain",
            "candidate_id": self.candidate_id,
            "worker_count": WORKER_COUNT,
            "chains_per_worker": CHAINS_PER_WORKER,
            "chain_count": CHAIN_COUNT,
            "worker_seeds": worker_seeds,
            "root_seed": tuple(config.seed),
            "worker_rows": tuple(row["metadata"] for row in rows),
            "jit_compile": True,
            "use_xla": True,
            "chain_execution_mode": "tf_function",
            "cuda_visible_devices": "-1",
            "artifact_signature": self.artifact_signature,
            "payload_file_sha256": self.payload_file_sha256,
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "adapter_signature": EXPECTED_ADAPTER_SIGNATURE,
            "call_count": self.call_count,
        }
        archive_reference = self._maybe_archive_verification(
            adapter=adapter,
            samples=samples,
            trace=trace,
            config=config,
            worker_rows=rows,
        )
        if archive_reference is not None:
            metadata["verification_archive"] = archive_reference
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _receive(self, worker_index: int, command: str) -> Mapping[str, Any]:
        connection, process = self._workers[worker_index]
        try:
            response = connection.recv()
        except EOFError as exc:
            raise LGSSMNeuTraCampaignError(
                f"worker {worker_index} exited during {command} with code "
                f"{process.exitcode}"
            ) from exc
        if response.get("status") != "ok":
            raise LGSSMNeuTraCampaignError(
                f"worker {worker_index} {command} failed: "
                f"{response.get('error_type')}: {response.get('error_message')}"
            )
        return response["payload"]

    def _maybe_archive_verification(
        self,
        *,
        adapter: Any,
        samples: np.ndarray,
        trace: Mapping[str, Any],
        config: Any,
        worker_rows: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        if self.archive_dir is None or int(config.num_results) < TUNING_VERIFICATION_RESULTS:
            return None
        flat = samples.reshape(-1, DIMENSION)
        raw = np.asarray(adapter.transport.forward_batch(flat).numpy(), dtype=np.float64)
        raw = raw.reshape(samples.shape)
        path = self.archive_dir / (
            f"verification_seed_{int(config.seed[0])}_{int(config.seed[1])}.npz"
        )
        if path.exists():
            raise LGSSMNeuTraCampaignError(
                f"verification archive already exists: {path}"
            )
        np.savez_compressed(
            path,
            retained_z_samples=samples,
            retained_raw_samples=raw,
            final_worker_states=np.stack(
                tuple(np.asarray(row["samples"])[-1] for row in worker_rows), axis=0
            ),
            root_seed=np.asarray(config.seed, dtype=np.int64),
            candidate_id=np.asarray(self.candidate_id),
            target_signature=np.asarray(EXPECTED_TARGET_SIGNATURE),
            artifact_signature=np.asarray(self.artifact_signature),
        )
        reference = {
            **_file_reference(path),
            "z_shape": tuple(int(item) for item in samples.shape),
            "raw_shape": tuple(int(item) for item in raw.shape),
            "chain_axes_preserved": True,
            "all_finite": bool(
                np.all(np.isfinite(samples)) and np.all(np.isfinite(raw))
            ),
            "root_seed": tuple(config.seed),
        }
        self.archive_references.append(reference)
        return reference


def _hmc_worker_loop(
    connection: Any,
    worker_index: int,
    payload_path: str,
    expected_payload_sha256: str,
    evidence_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    try:
        from bayesfilter.runtime.device_policy import assert_cpu_only_env

        assert_cpu_only_env()
        import tensorflow as tf

        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.batched_value_score import (
            FixedTransportValueScoreAdapter,
        )
        from bayesfilter.inference.hmc import (
            FullChainHMCConfig,
            build_reusable_full_chain_tfp_hmc_runner,
        )
        from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
        from bayesfilter.inference.neutra_artifacts import (
            load_frozen_neutra_artifact,
        )
        from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
            load_deterministic_lgssm_exact_target,
        )

        artifact_path = Path(payload_path)
        if _file_sha256(artifact_path) != expected_payload_sha256:
            raise LGSSMNeuTraCampaignError("worker payload file hash mismatch")
        bundle = load_deterministic_lgssm_exact_target(
            expected_target_signature=EXPECTED_TARGET_SIGNATURE
        )
        if bundle.adapter.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
            raise LGSSMNeuTraCampaignError("worker adapter signature mismatch")
        loaded = load_frozen_neutra_artifact(
            _read_mapping(artifact_path, "worker frozen payload"),
            expected_target_signature=EXPECTED_TARGET_SIGNATURE,
        )
        runner_cache: dict[str, Any] = {}
        connection.send(
            {
                "status": "ok",
                "payload": {
                    "worker_index": worker_index,
                    "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tensorflow_version": tf.__version__,
                    "physical_gpus": tuple(
                        str(item) for item in tf.config.list_physical_devices("GPU")
                    ),
                    "target_signature": bundle.target_signature,
                    "adapter_signature": bundle.adapter.adapter_signature(),
                    "artifact_signature": loaded.artifact_signature,
                    "transport_hash": loaded.manifest.transport_hash,
                    "payload_file_sha256": expected_payload_sha256,
                    "jit_compile": True,
                },
            }
        )
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "shutdown":
                connection.send({"status": "shutdown"})
                break
            if command == "probe":
                payload = _worker_fixed_probe(
                    tf=tf,
                    bundle=bundle,
                    loaded=loaded,
                    z=np.asarray(request["z"], dtype=np.float64),
                    probe_hash=str(request["probe_hash"]),
                    worker_index=worker_index,
                    evidence_path=evidence_path,
                )
                connection.send({"status": "ok", "payload": payload})
                continue
            if command != "run_hmc":
                raise LGSSMNeuTraCampaignError(f"unknown worker command: {command}")
            payload = _worker_run_hmc(
                tf=tf,
                bundle=bundle,
                loaded=loaded,
                worker_index=worker_index,
                config_payload=request["config"],
                initial_state=np.asarray(request["initial_state"], dtype=np.float64),
                runner_cache=runner_cache,
                FullChainHMCConfig=FullChainHMCConfig,
                HMCTuningPolicy=HMCTuningPolicy,
                build_runner=build_reusable_full_chain_tfp_hmc_runner,
                FixedTransportValueScoreAdapter=FixedTransportValueScoreAdapter,
                evidence_path=evidence_path,
            )
            connection.send({"status": "ok", "payload": payload})
    except EOFError:
        pass
    except BaseException as exc:  # noqa: BLE001 - child must report fail-closed.
        try:
            connection.send(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "worker_index": worker_index,
                }
            )
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _worker_fixed_probe(
    *,
    tf: Any,
    bundle: Any,
    loaded: Any,
    z: np.ndarray,
    probe_hash: str,
    worker_index: int,
    evidence_path: str,
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    if probe_hash != common_probe_hash() or not np.array_equal(z, common_probe_points()):
        raise LGSSMNeuTraCampaignError("worker fixed probe identity mismatch")
    scope = f"lgssm_neutra_cpu_probe_{loaded.manifest.transport_hash}"
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=scope,
        evidence_path=evidence_path,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = adapter.log_prob_and_grad_batch(z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return theta, logdet, value, score, status

    outputs = compiled(tf.constant(z, dtype=tf.float64))
    theta, logdet, value, score, status = outputs
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(item)).numpy()) for item in outputs[:4]
    ):
        raise LGSSMNeuTraCampaignError("CPU fixed probe is nonfinite")
    status_valid = bool(
        tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
    )
    if not status_valid:
        raise LGSSMNeuTraCampaignError("CPU fixed probe target status failed")
    return {
        "worker_index": worker_index,
        "pid": os.getpid(),
        "probe_hash": probe_hash,
        "theta": theta.numpy(),
        "logdet": logdet.numpy(),
        "value": value.numpy(),
        "score": score.numpy(),
        "status": status,
        "jit_compile": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": tuple(str(item) for item in tf.config.list_physical_devices("GPU")),
        "fixed_transport_adapter_signature": adapter.adapter_signature(),
    }


def _worker_run_hmc(
    *,
    tf: Any,
    bundle: Any,
    loaded: Any,
    worker_index: int,
    config_payload: Mapping[str, Any],
    initial_state: np.ndarray,
    runner_cache: dict[str, Any],
    FullChainHMCConfig: Any,
    HMCTuningPolicy: Any,
    build_runner: Any,
    FixedTransportValueScoreAdapter: Any,
    evidence_path: str,
) -> Mapping[str, Any]:
    if initial_state.shape != (CHAINS_PER_WORKER, DIMENSION):
        raise LGSSMNeuTraCampaignError("worker initial state shape mismatch")
    policy_payload = dict(config_payload["tuning_policy"])
    policy = HMCTuningPolicy(
        label=str(policy_payload["label"]),
        adaptation_policy=str(policy_payload["adaptation_policy"]),
        num_adaptation_steps=int(policy_payload["num_adaptation_steps"]),
        target_accept_prob=policy_payload.get("target_accept_prob"),
        source=str(policy_payload["source"]),
        enabled=bool(policy_payload["enabled"]),
        implemented=bool(policy_payload["implemented"]),
        diagnostic_role=str(policy_payload["diagnostic_role"]),
        nonclaims=tuple(str(item) for item in policy_payload["nonclaims"]),
    )
    config = FullChainHMCConfig(
        num_results=int(config_payload["num_results"]),
        num_burnin_steps=int(config_payload["num_burnin_steps"]),
        step_size=float(config_payload["step_size"]),
        num_leapfrog_steps=int(config_payload["num_leapfrog_steps"]),
        seed=tuple(config_payload["seed"]),
        use_xla=bool(config_payload["use_xla"]),
        trace_policy=str(config_payload["trace_policy"]),
        target_status_trace_policy=str(config_payload["target_status_trace_policy"]),
        tuning_policy=policy,
        target_scope=str(config_payload["target_scope"]),
        chain_execution_mode=str(config_payload["chain_execution_mode"]),
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=config.target_scope,
        evidence_path=evidence_path,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    static_payload = dict(config.signature_payload())
    static_payload.pop("seed", None)
    static_payload.pop("step_size", None)
    cache_key = _stable_json_hash(
        {
            "config": static_payload,
            "initial_state_shape": initial_state.shape,
            "artifact_signature": loaded.artifact_signature,
            "target_signature": bundle.target_signature,
        }
    )
    runner = runner_cache.get(cache_key)
    reused = runner is not None
    if runner is None:
        runner = build_runner(adapter, initial_state, config)
        runner_cache[cache_key] = runner
    start = time.monotonic()
    result = runner.run(
        current_state=initial_state,
        seed=config.seed,
        step_size=config.step_size,
    )
    elapsed = time.monotonic() - start
    return {
        "worker_index": worker_index,
        "pid": os.getpid(),
        "samples": result.samples.numpy(),
        "trace": _json_ready(result.trace),
        "diagnostics": _json_ready(result.diagnostics),
        "metadata": {
            **_json_ready(result.metadata),
            "worker_index": worker_index,
            "pid": os.getpid(),
            "worker_seed": config.seed,
            "runner_cache_key": cache_key,
            "runner_reused": reused,
            "runner_cache_size": len(runner_cache),
            "elapsed_seconds": elapsed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": tuple(
                str(item) for item in tf.config.list_physical_devices("GPU")
            ),
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
        },
    }


def _combine_worker_trace(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise LGSSMNeuTraCampaignError("worker trace rows are empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise LGSSMNeuTraCampaignError("worker trace keys differ")
    result = {}
    for key in sorted(keys):
        values = tuple(row[key] for row in rows)
        if all(isinstance(value, Mapping) for value in values):
            result[key] = _combine_worker_trace(values)
            continue
        arrays = tuple(np.asarray(value) for value in values)
        if any(array.ndim < 2 for array in arrays):
            if not all(np.array_equal(array, arrays[0]) for array in arrays[1:]):
                raise LGSSMNeuTraCampaignError(f"worker scalar trace mismatch: {key}")
            result[key] = arrays[0]
        else:
            result[key] = np.concatenate(arrays, axis=1)
    return result


def _combined_hmc_diagnostics(
    *, samples: np.ndarray, trace: Mapping[str, Any]
) -> Mapping[str, Any]:
    finite_by_sample = np.all(np.isfinite(samples), axis=-1)
    accepted = np.asarray(trace.get("is_accepted"), dtype=bool)
    log_accept = np.asarray(trace.get("log_accept_ratio"), dtype=np.float64)
    target = np.asarray(trace.get("target_log_prob"), dtype=np.float64)
    telemetry = trace.get("target_status_telemetry")
    telemetry_summary = None
    if isinstance(telemetry, Mapping):
        status = np.asarray(telemetry["status_code"], dtype=np.int32)
        valid = np.asarray(telemetry["valid_pre_regularized_score"], dtype=bool)
        nonvalid = (status != 0) | ~valid
        telemetry_summary = {
            "trace_entry_count": int(status.size),
            "status_nonvalid_count": int(np.sum(nonvalid)),
            "all_status_valid": bool(not np.any(nonvalid)),
            "floor_count_total": int(
                np.sum(np.asarray(telemetry["floor_count_value"], dtype=np.int64))
            ),
            "max_floor_count_value": int(
                np.max(np.asarray(telemetry["floor_count_value"], dtype=np.int64))
            ),
            "min_min_innovation_eigenvalue": float(
                np.min(
                    np.asarray(
                        telemetry["min_innovation_eigenvalue"], dtype=np.float64
                    )
                )
            ),
            "max_innovation_condition_estimate": float(
                np.max(
                    np.asarray(
                        telemetry["innovation_condition_estimate"], dtype=np.float64
                    )
                )
            ),
            "telemetry_failure_veto": bool(np.any(nonvalid)),
        }
    divergence = trace.get("divergence")
    return {
        "acceptance_rate": float(np.mean(accepted)),
        "finite_sample_count": int(np.sum(finite_by_sample)),
        "nonfinite_sample_count": int(np.sum(~finite_by_sample)),
        "sample_shape": tuple(int(item) for item in samples.shape),
        "trace_policy": "standard",
        "divergence_status": "available" if divergence is not None else "not_exposed_by_kernel",
        "divergence_count": (
            int(np.sum(np.asarray(divergence, dtype=bool)))
            if divergence is not None
            else None
        ),
        "target_status_telemetry": telemetry_summary,
        "log_accept_ratio_finite_count": int(np.sum(np.isfinite(log_accept))),
        "log_accept_ratio_nonfinite_count": int(np.sum(~np.isfinite(log_accept))),
        "target_log_prob_finite_count": int(np.sum(np.isfinite(target))),
        "target_log_prob_nonfinite_count": int(np.sum(~np.isfinite(target))),
        "all_samples_finite": bool(np.all(finite_by_sample)),
    }


def run_phase5_candidate(candidate_id: str) -> Mapping[str, Any]:
    """Tune and admit one frozen candidate on the CPU-hidden worker route."""

    candidate = _validate_candidate_id(candidate_id)
    validate_static_campaign_inputs()
    _require_contract()
    phase4 = _require_phase_pass(ARTIFACT_ROOT / "phase4" / "result.json", phase=4)
    if candidate not in tuple(phase4["viable_candidates"]):
        raise LGSSMNeuTraCampaignError(f"candidate is not Phase 4 viable: {candidate}")
    result_path = candidate_phase5_result_path(candidate)
    if result_path.parent.exists():
        raise LGSSMNeuTraCampaignError(
            f"Phase 5 candidate directory already exists: {candidate}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraCampaignError(
            "Phase 5 parent must set CUDA_VISIBLE_DEVICES=-1 before import"
        )
    root = result_path.parent
    start = time.monotonic()
    seed_row = campaign_seed_ledger()["candidate_hmc_seeds"][candidate]

    with TwoWorkerFixedTransportHMCRunner(
        candidate_id=candidate,
        archive_dir=root / "private",
    ) as runner:
        cpu_probe = runner.probe()
        gpu_probe = _read_mapping(
            candidate_phase4_result_path(candidate), f"Phase 4 {candidate} result"
        )["gpu_fixed_transport_probe"]
        cross_device_parity = _cross_device_probe_parity(gpu_probe, cpu_probe)
        _assert_cross_device_probe_parity(cross_device_parity)

        bundle, loaded = _cpu_parent_bundle_and_candidate(candidate)
        from bayesfilter.inference.fixed_transport_hmc_tuning import (
            FixedTransportHMCKernelTuningConfig,
            tune_fixed_transport_hmc_kernel,
        )

        primary_config = _phase5_tuning_config(
            candidate_id=candidate,
            screen_seed=tuple(seed_row["fixed_grid_primary_screen_seed"]),
            verification_seed=tuple(
                seed_row["fixed_grid_primary_verification_seed"]
            ),
            scales=TUNING_PRIMARY_SCALES,
            output_filename="tuning_result.json",
            source_suffix="primary",
        )
        primary = tune_fixed_transport_hmc_kernel(
            base_adapter=bundle.adapter,
            fixed_transport=loaded.transport,
            initial_position=np.zeros(DIMENSION, dtype=np.float64),
            config=primary_config,
            output_dir=root / "primary",
            run_full_chain=runner,
        )
        selected = primary
        repair = None
        if not primary.passed:
            if not _phase5_repair_allowed(primary):
                selected = primary
            else:
                repair_config = _phase5_tuning_config(
                    candidate_id=candidate,
                    screen_seed=tuple(seed_row["fixed_grid_repair_screen_seed"]),
                    verification_seed=tuple(
                        seed_row["fixed_grid_repair_verification_seed"]
                    ),
                    scales=TUNING_REPAIR_SCALES,
                    output_filename="tuning_result.json",
                    source_suffix="single_declared_grid_repair",
                )
                repair = tune_fixed_transport_hmc_kernel(
                    base_adapter=bundle.adapter,
                    fixed_transport=loaded.transport,
                    initial_position=np.zeros(DIMENSION, dtype=np.float64),
                    config=repair_config,
                    output_dir=root / "repair",
                    run_full_chain=runner,
                )
                selected = repair

        selected_payload = selected.payload()
        final_kernel = selected.final_kernel_payload
        admitted = bool(selected.passed and isinstance(final_kernel, Mapping))
        verification_archive = _select_verification_archive(
            runner.archive_references,
            verification_seed=(
                tuple(seed_row["fixed_grid_repair_verification_seed"])
                if repair is not None
                else tuple(seed_row["fixed_grid_primary_verification_seed"])
            ),
            required=admitted,
        )
        result = {
            "schema": "bayesfilter.lgssm_neutra_phase5_candidate_result.v1",
            "phase": 5,
            "candidate_id": candidate,
            "completed": True,
            "passed": admitted,
            "admitted": admitted,
            "decision": (
                "ADMIT_FIXED_TRANSPORT_KERNEL_FOR_CONFIRMATORY_PHASE6"
                if admitted
                else "REJECT_CANDIDATE_AT_PHASE5_ADMISSION"
            ),
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "payload": _file_reference(candidate_payload_path(candidate)),
            "gpu_cpu_fixed_probe_parity": cross_device_parity,
            "cpu_worker_initialization": runner.worker_initialization,
            "cpu_probe": cpu_probe,
            "primary_tuning": {
                "result": _file_reference(Path(primary.artifact_path)),
                "artifact_hash": primary.artifact_hash,
                "passed": primary.passed,
                "final_status": primary.final_status,
                "hard_vetoes": primary.hard_vetoes,
                "repair_triggers": primary.repair_triggers,
            },
            "repair_tuning": (
                None
                if repair is None
                else {
                    "result": _file_reference(Path(repair.artifact_path)),
                    "artifact_hash": repair.artifact_hash,
                    "passed": repair.passed,
                    "final_status": repair.final_status,
                    "hard_vetoes": repair.hard_vetoes,
                    "repair_triggers": repair.repair_triggers,
                }
            ),
            "selected_tuning_artifact_hash": selected.artifact_hash,
            "selected_tuning_payload": selected_payload,
            "final_kernel_payload": final_kernel,
            "final_kernel_hash": selected.final_kernel_hash,
            "verification_archive": verification_archive,
            "all_verification_archives": tuple(runner.archive_references),
            "seed_ledger": seed_row,
            "runtime_identity": {
                "worker_count": WORKER_COUNT,
                "chains_per_worker": CHAINS_PER_WORKER,
                "cuda_visible_devices": "-1",
                "jit_compile": True,
                "dtype": "float64",
                "tuning_and_serious_route_same": True,
            },
            "elapsed_seconds": time.monotonic() - start,
            "run_manifest": _runtime_base_manifest(),
            "evidence_role": "fixed_kernel_admission_not_posterior_convergence",
            "nonclaims": CAMPAIGN_NONCLAIMS,
        }
    result = _with_artifact_hash(result)
    _write_new_json(result_path, result)
    return result


def finalize_phase5() -> Mapping[str, Any]:
    """Freeze all admission decisions before any confirmatory sampling."""

    phase4 = _require_phase_pass(ARTIFACT_ROOT / "phase4" / "result.json", phase=4)
    rows = []
    for candidate in tuple(phase4["viable_candidates"]):
        row = _read_mapping(candidate_phase5_result_path(candidate), candidate)
        if row.get("candidate_id") != candidate or row.get("completed") is not True:
            raise LGSSMNeuTraCampaignError("Phase 5 candidate result is incomplete")
        rows.append(row)
    admitted = tuple(row["candidate_id"] for row in rows if row.get("admitted") is True)
    result = {
        "schema": "bayesfilter.lgssm_neutra_phase5_result.v1",
        "phase": 5,
        "passed": bool(admitted),
        "decision": (
            "PASS_PHASE5_ALL_ADMISSION_DECISIONS_FROZEN"
            if admitted
            else "BLOCK_PHASE5_NO_ADMITTED_CANDIDATE_AFTER_DECLARED_REPAIR"
        ),
        "all_phase4_candidates_processed": True,
        "candidate_order": tuple(phase4["viable_candidates"]),
        "admitted_candidates": admitted,
        "rejected_candidates": tuple(
            row["candidate_id"] for row in rows if row.get("admitted") is not True
        ),
        "candidate_results": tuple(
            {
                "candidate_id": row["candidate_id"],
                "admitted": row["admitted"],
                "result": _file_reference(candidate_phase5_result_path(row["candidate_id"])),
                "result_artifact_hash": row["artifact_hash"],
                "final_kernel_hash": row.get("final_kernel_hash"),
            }
            for row in rows
        ),
        "serious_sampling_executed": False,
        "post_admission_retuning_allowed": False,
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(ARTIFACT_ROOT / "phase5" / "result.json", result)
    return result


def run_phase6_candidate(candidate_id: str) -> Mapping[str, Any]:
    """Run one immutable confirmatory four-chain candidate/kernel pair."""

    candidate = _validate_candidate_id(candidate_id)
    inputs = validate_static_campaign_inputs()
    phase5 = _require_phase_pass(ARTIFACT_ROOT / "phase5" / "result.json", phase=5)
    if candidate not in tuple(phase5["admitted_candidates"]):
        raise LGSSMNeuTraCampaignError(f"candidate was not admitted: {candidate}")
    result_path = candidate_phase6_result_path(candidate)
    if result_path.parent.exists():
        raise LGSSMNeuTraCampaignError(
            "Phase 6 is confirmatory and cannot be rerun for this candidate"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraCampaignError(
            "Phase 6 parent must set CUDA_VISIBLE_DEVICES=-1 before import"
        )
    phase5_candidate = _read_mapping(
        candidate_phase5_result_path(candidate), f"Phase 5 {candidate} result"
    )
    if phase5_candidate.get("admitted") is not True:
        raise LGSSMNeuTraCampaignError("Phase 5 candidate admission mismatch")
    kernel = phase5_candidate.get("final_kernel_payload")
    if not isinstance(kernel, Mapping):
        raise LGSSMNeuTraCampaignError("admitted candidate lacks a fixed kernel")
    seed = tuple(
        campaign_seed_ledger()["candidate_hmc_seeds"][candidate]["serious_seed"]
    )
    root = result_path.parent
    start = time.monotonic()

    with TwoWorkerFixedTransportHMCRunner(candidate_id=candidate) as runner:
        cpu_probe = runner.probe()
        phase5_cpu_probe = phase5_candidate["cpu_probe"]
        _assert_cpu_probe_identity(phase5_cpu_probe, cpu_probe)
        bundle, loaded = _cpu_parent_bundle_and_candidate(candidate)
        from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
        from bayesfilter.inference.hmc import FullChainHMCConfig

        target_scope = str(kernel["transformed_target_scope"])
        adapter = FixedTransportValueScoreAdapter(
            base_adapter=bundle.adapter,
            transport=loaded.transport,
            target_scope=target_scope,
            evidence_path=str(PLAN_PATH.relative_to(ROOT)),
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
        )
        if adapter.adapter_signature() != kernel["transformed_adapter_signature"]:
            raise LGSSMNeuTraCampaignError("Phase 5/6 transformed adapter drifted")
        config = FullChainHMCConfig(
            num_results=SERIOUS_RESULTS,
            num_burnin_steps=SERIOUS_BURNIN,
            step_size=float(kernel["step_size"]),
            num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
            seed=seed,
            use_xla=True,
            trace_policy="standard",
            target_status_trace_policy="per_chain_step",
            target_scope=target_scope,
            chain_execution_mode="tf_function",
        )
        run = runner(
            adapter,
            np.zeros((CHAIN_COUNT, DIMENSION), dtype=np.float64),
            config,
        )
        z_samples = np.asarray(run.samples, dtype=np.float64)
        raw_samples = np.asarray(
            loaded.transport.forward_batch(z_samples.reshape(-1, DIMENSION)).numpy(),
            dtype=np.float64,
        ).reshape(z_samples.shape)
        archive_path = root / "private" / "retained_samples.npz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            raise LGSSMNeuTraCampaignError("Phase 6 private archive already exists")
        np.savez_compressed(
            archive_path,
            retained_z_samples=z_samples,
            retained_raw_samples=raw_samples,
            final_worker_states=z_samples[-1].reshape(
                WORKER_COUNT, CHAINS_PER_WORKER, DIMENSION
            ),
            candidate_id=np.asarray(candidate),
            root_seed=np.asarray(seed, dtype=np.int64),
            target_signature=np.asarray(bundle.target_signature),
            artifact_signature=np.asarray(loaded.artifact_signature),
            final_kernel_hash=np.asarray(str(phase5_candidate["final_kernel_hash"])),
            phase5_result_artifact_hash=np.asarray(
                str(phase5_candidate["artifact_hash"])
            ),
        )

        from bayesfilter.inference.hmc_convergence import (
            RankNormalizedHMCThresholds,
            rank_normalized_hmc_diagnostics,
        )

        convergence = rank_normalized_hmc_diagnostics(
            raw_samples,
            parameter_names=inputs.parameter_names,
            thresholds=RankNormalizedHMCThresholds(
                rhat_max=RHAT_MAX,
                bulk_ess_min=BULK_ESS_MIN,
                tail_ess_min=TAIL_ESS_MIN,
            ),
        )
        comparator = _load_comparator_samples()
        summaries = _serious_posterior_summaries(
            candidate_samples=raw_samples,
            comparator_samples=comparator,
            truth=inputs.truth,
            parameter_names=inputs.parameter_names,
        )
        health = _serious_health_screen(
            samples=z_samples,
            raw_samples=raw_samples,
            diagnostics=run.diagnostics,
            trace=run.trace,
        )
        passed = bool(
            convergence["passed"]
            and not convergence.get("hard_vetoes")
            and health["passed"]
            and summaries["posterior_agreement_passed"]
            and summaries["recovery_passed"]
        )
        result = {
            "schema": "bayesfilter.lgssm_neutra_phase6_candidate_result.v1",
            "phase": 6,
            "candidate_id": candidate,
            "passed": passed,
            "viable": passed,
            "decision": (
                "PASS_FIXED_18D_LGSSM_TUNED_NEUTRA_HMC_CAMPAIGN"
                if passed
                else "REJECT_CANDIDATE_AFTER_CONFIRMATORY_PHASE6"
            ),
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "phase5_result": _file_reference(candidate_phase5_result_path(candidate)),
            "phase5_result_artifact_hash": phase5_candidate["artifact_hash"],
            "final_kernel_hash": phase5_candidate["final_kernel_hash"],
            "fixed_kernel": kernel,
            "serious_config": config.signature_payload(),
            "serious_seed": seed,
            "private_retained_archive": {
                **_file_reference(archive_path),
                "z_shape": tuple(int(item) for item in z_samples.shape),
                "raw_shape": tuple(int(item) for item in raw_samples.shape),
                "chain_axes_preserved": True,
                "all_finite": bool(
                    np.all(np.isfinite(z_samples))
                    and np.all(np.isfinite(raw_samples))
                ),
            },
            "health_screen": health,
            "convergence_diagnostics": convergence,
            "posterior_summaries": summaries,
            "acceptance_rate": float(run.diagnostics["acceptance_rate"]),
            "runtime_metadata": run.metadata,
            "cpu_probe": cpu_probe,
            "elapsed_seconds": time.monotonic() - start,
            "run_manifest": {
                **_runtime_base_manifest(),
                "command_role": "confirmatory_no_same_campaign_rerun",
                "environment": "tf-gpu conda environment with CUDA hidden",
                "cpu_gpu_status": "CPU-hidden HMC; GPU intentionally unavailable",
                "data_version": _file_sha256(bundle.fixture_path),
                "random_seeds": {
                    "root": seed,
                    "workers": _worker_seeds(seed),
                },
                "wall_time_seconds": time.monotonic() - start,
                "output_artifacts": {
                    "result": str(result_path.relative_to(ROOT)),
                    "retained_archive": str(archive_path.relative_to(ROOT)),
                },
            },
            "evidence_role": (
                "confirmatory_single_fixture_convergence_and_supporting_screens"
            ),
            "same_campaign_rerun_allowed": False,
            "nonclaims": CAMPAIGN_NONCLAIMS,
        }
    result = _with_artifact_hash(result)
    _write_new_json(result_path, result)
    return result


def finalize_phase6() -> Mapping[str, Any]:
    """Close confirmatory sampling after every pre-admitted pair completes."""

    phase5 = _require_phase_pass(ARTIFACT_ROOT / "phase5" / "result.json", phase=5)
    rows = []
    for candidate in tuple(phase5["admitted_candidates"]):
        row = _read_mapping(candidate_phase6_result_path(candidate), candidate)
        if row.get("candidate_id") != candidate:
            raise LGSSMNeuTraCampaignError("Phase 6 candidate identity mismatch")
        rows.append(row)
    viable = tuple(row["candidate_id"] for row in rows if row.get("passed") is True)
    result = {
        "schema": "bayesfilter.lgssm_neutra_phase6_result.v1",
        "phase": 6,
        "passed": bool(viable),
        "decision": (
            "PASS_PHASE6_AT_LEAST_ONE_CONFIRMATORY_PAIR_VIABLE"
            if viable
            else "BLOCK_PHASE6_NO_CONFIRMATORY_PAIR_PASSED"
        ),
        "all_pre_admitted_pairs_completed": True,
        "admitted_candidates": tuple(phase5["admitted_candidates"]),
        "viable_candidates": viable,
        "rejected_candidates": tuple(
            row["candidate_id"] for row in rows if row.get("passed") is not True
        ),
        "candidate_results": tuple(
            {
                "candidate_id": row["candidate_id"],
                "passed": row["passed"],
                "result": _file_reference(candidate_phase6_result_path(row["candidate_id"])),
                "result_artifact_hash": row["artifact_hash"],
                "max_rhat": row["convergence_diagnostics"]["max_rhat"],
                "min_bulk_ess": row["convergence_diagnostics"]["min_bulk_ess"],
                "min_tail_ess": row["convergence_diagnostics"]["min_tail_ess"],
                "max_posterior_agreement_z": row["posterior_summaries"][
                    "max_posterior_agreement_combined_mcse"
                ],
                "max_recovery_z": row["posterior_summaries"][
                    "max_abs_mean_minus_truth_over_sd"
                ],
            }
            for row in rows
        ),
        "same_campaign_retuning_retraining_reseeding_or_rerun_used": False,
        "nonclaims": CAMPAIGN_NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(ARTIFACT_ROOT / "phase6" / "result.json", result)
    return result


def _cpu_parent_bundle_and_candidate(candidate_id: str):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraCampaignError("CPU parent must hide CUDA before TF import")
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
        raise LGSSMNeuTraCampaignError("CPU parent adapter signature mismatch")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(candidate_payload_path(candidate_id), "candidate payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    return bundle, loaded


def _phase5_tuning_config(
    *,
    candidate_id: str,
    screen_seed: tuple[int, int],
    verification_seed: tuple[int, int],
    scales: Sequence[float],
    output_filename: str,
    source_suffix: str,
):
    from bayesfilter.inference.fixed_transport_hmc_tuning import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
    )

    return FixedTransportHMCKernelTuningConfig(
        initial_step_size=TUNING_BASE_STEP_SIZE,
        # This July campaign predates the measured joint-grid contract.  Keep
        # its one-L directional ladder available only as an explicit
        # diagnostic record; it cannot issue a claim-bearing handoff.
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        selection_policy="acceptance_target_distance",
        leapfrog_grid=(TUNING_LEAPFROG_STEPS,),
        chain_count=CHAIN_COUNT,
        target_accept_prob=0.70,
        acceptance_band=TUNING_ACCEPTANCE_BAND,
        repair_band=TUNING_REPAIR_BAND,
        budget_schedule=(8,),
        tune_num_results=TUNING_PROBE_RESULTS,
        screen_num_results=TUNING_PROBE_RESULTS,
        screen_num_burnin_steps=TUNING_PROBE_BURNIN,
        verification_num_results=TUNING_VERIFICATION_RESULTS,
        verification_num_burnin_steps=TUNING_VERIFICATION_BURNIN,
        require_modern_rank_normalized_verification=True,
        verification_min_retained_results_per_chain=(
            TUNING_VERIFICATION_RESULTS
        ),
        verification_rhat_max=RHAT_MAX,
        tune_seed_base=(screen_seed[0], screen_seed[1] - 500_000),
        screen_seed_base=screen_seed,
        verification_seed_base=verification_seed,
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=f"lgssm_neutra_{candidate_id}_fixed_transport",
        target_status_trace_policy="per_chain_step",
        fixed_grid_base_step_size_candidates=(TUNING_BASE_STEP_SIZE,),
        fixed_grid_scale_candidates=tuple(float(item) for item in scales),
        fixed_grid_num_leapfrog_steps=TUNING_LEAPFROG_STEPS,
        fixed_grid_max_attempts=len(tuple(scales)),
        fixed_grid_fallback_acceptance_max=0.95,
        output_filename=output_filename,
        source=(
            "bayesfilter.lgssm_neutra_serious_validation."
            f"{candidate_id}.{source_suffix}"
        ),
    )


def _phase5_repair_allowed(result: Any) -> bool:
    """Allow only the predeclared finite acceptance/R-hat tuning repair."""

    allowed_vetoes = {
        "verification_acceptance_outside_repair_band",
        "verification_modern_rank_folded_rhat_failed",
    }
    vetoes = set(str(item) for item in result.hard_vetoes)
    if vetoes - allowed_vetoes:
        return False
    scale = result.fixed_grid_scale_selection_payload
    if scale is None:
        return False
    attempts = tuple(scale.get("attempts", ()))
    if not attempts:
        return False
    for attempt in attempts:
        diagnostics = attempt.get("probe_diagnostics")
        if not isinstance(diagnostics, Mapping):
            return False
        if diagnostics.get("samples_all_finite") is not True:
            return False
        if diagnostics.get("log_accept_ratio_finite") is not True:
            return False
        if diagnostics.get("target_log_prob_finite") is False:
            return False
        telemetry = diagnostics.get("target_status_telemetry")
        if not isinstance(telemetry, Mapping):
            return False
        if telemetry.get("telemetry_failure_veto") is not False:
            return False
        divergence_count = diagnostics.get("divergence_count")
        if divergence_count is not None and int(divergence_count) > 0:
            return False
    return True


def _cross_device_probe_parity(
    gpu_probe: Mapping[str, Any], cpu_probe: Mapping[str, Any]
) -> Mapping[str, Any]:
    if gpu_probe.get("probe_hash") != common_probe_hash():
        raise LGSSMNeuTraCampaignError("GPU probe hash mismatch")
    if cpu_probe.get("probe_hash") != common_probe_hash():
        raise LGSSMNeuTraCampaignError("CPU probe hash mismatch")
    result = {"probe_hash": common_probe_hash()}
    for key in ("theta", "logdet", "value", "score"):
        gpu = np.asarray(gpu_probe[key], dtype=np.float64)
        cpu = np.asarray(cpu_probe[key], dtype=np.float64)
        if gpu.shape != cpu.shape:
            raise LGSSMNeuTraCampaignError(f"GPU/CPU probe shape mismatch: {key}")
        result[f"{key}_max_abs"] = float(np.max(np.abs(gpu - cpu)))
    result["tolerances"] = COMMON_PROBE_TOLERANCES
    result["passed"] = bool(
        result["theta_max_abs"] <= COMMON_PROBE_TOLERANCES["transport_max_abs"]
        and result["logdet_max_abs"] <= COMMON_PROBE_TOLERANCES["logdet_max_abs"]
        and result["value_max_abs"] <= COMMON_PROBE_TOLERANCES["value_max_abs"]
        and result["score_max_abs"] <= COMMON_PROBE_TOLERANCES["score_max_abs"]
    )
    return result


def _score_parity_summary(
    *, reference: Mapping[str, Any], explicit: Mapping[str, Any]
) -> Mapping[str, Any]:
    differences = {}
    for key in ("theta", "logdet", "value", "score"):
        left = np.asarray(reference[key], dtype=np.float64)
        right = np.asarray(explicit[key], dtype=np.float64)
        if left.shape != right.shape:
            raise LGSSMNeuTraCampaignError(f"score parity shape mismatch: {key}")
        differences[f"{key}_max_abs"] = float(np.max(np.abs(left - right)))
    differences["tolerances"] = COMMON_PROBE_TOLERANCES
    differences["passed"] = bool(
        differences["theta_max_abs"]
        <= COMMON_PROBE_TOLERANCES["transport_max_abs"]
        and differences["logdet_max_abs"]
        <= COMMON_PROBE_TOLERANCES["logdet_max_abs"]
        and differences["value_max_abs"] <= COMMON_PROBE_TOLERANCES["value_max_abs"]
        and differences["score_max_abs"] <= COMMON_PROBE_TOLERANCES["score_max_abs"]
    )
    return differences


def _assert_cross_device_probe_parity(value: Mapping[str, Any]) -> None:
    if value.get("passed") is not True:
        raise LGSSMNeuTraCampaignError("GPU/CPU fixed-objective parity failed")


def _assert_cpu_probe_identity(
    earlier: Mapping[str, Any], later: Mapping[str, Any]
) -> None:
    if earlier.get("probe_hash") != common_probe_hash() or later.get(
        "probe_hash"
    ) != common_probe_hash():
        raise LGSSMNeuTraCampaignError("Phase 5/6 CPU probe hash mismatch")
    for key in ("theta", "logdet", "value", "score"):
        if not np.array_equal(np.asarray(earlier[key]), np.asarray(later[key])):
            raise LGSSMNeuTraCampaignError(
                f"Phase 5/6 CPU fixed-objective probe drifted: {key}"
            )


def _select_verification_archive(
    references: Sequence[Mapping[str, Any]],
    *,
    verification_seed: tuple[int, int],
    required: bool,
) -> Mapping[str, Any] | None:
    matches = tuple(
        row for row in references if tuple(row.get("root_seed", ())) == verification_seed
    )
    if required and len(matches) != 1:
        raise LGSSMNeuTraCampaignError(
            "admitted candidate requires exactly one fresh verification archive"
        )
    return matches[0] if matches else None


def _verify_file_reference(reference: Any, label: str) -> Path:
    if not isinstance(reference, Mapping):
        raise LGSSMNeuTraCampaignError(f"{label} reference must be a mapping")
    path = (ROOT / str(reference.get("path", ""))).resolve()
    if not path.is_file():
        raise LGSSMNeuTraCampaignError(f"{label} file is missing")
    if _file_sha256(path) != reference.get("file_sha256"):
        raise LGSSMNeuTraCampaignError(f"{label} file hash mismatch")
    if path.stat().st_size != int(reference.get("byte_count", -1)):
        raise LGSSMNeuTraCampaignError(f"{label} byte count mismatch")
    return path


def _load_comparator_samples() -> np.ndarray:
    if _file_sha256(COMPARATOR_PATH) != EXPECTED_COMPARATOR_FILE_SHA256:
        raise LGSSMNeuTraCampaignError("plain-HMC comparator hash drifted")
    with np.load(COMPARATOR_PATH, allow_pickle=False) as archive:
        samples = np.asarray(archive["retained_raw_samples"], dtype=np.float64)
    if samples.shape != (4000, CHAIN_COUNT, DIMENSION):
        raise LGSSMNeuTraCampaignError("plain-HMC comparator shape mismatch")
    if not np.all(np.isfinite(samples)):
        raise LGSSMNeuTraCampaignError("plain-HMC comparator is nonfinite")
    return samples


def _raw_mean_ess_and_mcse(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (CHAIN_COUNT, DIMENSION):
        raise LGSSMNeuTraCampaignError("mean ESS samples must have shape [draw,4,18]")
    half = values.shape[0] // 2
    split = np.reshape(
        np.stack((values[:half], values[-half:]), axis=2),
        (half, 2 * CHAIN_COUNT, DIMENSION),
    )
    ess = np.asarray(
        tfp.mcmc.effective_sample_size(
            tf.constant(split, dtype=tf.float64),
            filter_beyond_positive_pairs=True,
            cross_chain_dims=1,
        ).numpy(),
        dtype=np.float64,
    )
    pooled = values.reshape(-1, DIMENSION)
    sd = np.std(pooled, axis=0, ddof=1)
    if ess.shape != (DIMENSION,) or np.any(~np.isfinite(ess)) or np.any(ess <= 0.0):
        raise LGSSMNeuTraCampaignError("mean ESS is invalid")
    mcse = sd / np.sqrt(ess)
    if np.any(~np.isfinite(mcse)) or np.any(mcse <= 0.0):
        raise LGSSMNeuTraCampaignError("mean MCSE is invalid")
    return ess, mcse


def _serious_posterior_summaries(
    *,
    candidate_samples: np.ndarray,
    comparator_samples: np.ndarray,
    truth: np.ndarray,
    parameter_names: Sequence[str],
) -> Mapping[str, Any]:
    candidate = np.asarray(candidate_samples, dtype=np.float64)
    comparator = np.asarray(comparator_samples, dtype=np.float64)
    candidate_pooled = candidate.reshape(-1, DIMENSION)
    comparator_pooled = comparator.reshape(-1, DIMENSION)
    candidate_mean = np.mean(candidate_pooled, axis=0)
    comparator_mean = np.mean(comparator_pooled, axis=0)
    candidate_sd = np.std(candidate_pooled, axis=0, ddof=1)
    comparator_sd = np.std(comparator_pooled, axis=0, ddof=1)
    candidate_ess, candidate_mcse = _raw_mean_ess_and_mcse(candidate)
    comparator_ess, comparator_mcse = _raw_mean_ess_and_mcse(comparator)
    agreement_z = np.abs(candidate_mean - comparator_mean) / np.sqrt(
        np.square(candidate_mcse) + np.square(comparator_mcse)
    )
    recovery_z = np.abs(candidate_mean - np.asarray(truth, dtype=np.float64)) / candidate_sd
    quantiles = np.quantile(candidate_pooled, (0.05, 0.5, 0.95), axis=0, method="linear")
    rows = []
    for index, name in enumerate(parameter_names):
        rows.append(
            {
                "parameter": str(name),
                "truth": float(truth[index]),
                "neutra_mean": float(candidate_mean[index]),
                "neutra_sd": float(candidate_sd[index]),
                "neutra_mean_ess": float(candidate_ess[index]),
                "neutra_mean_mcse": float(candidate_mcse[index]),
                "plain_hmc_mean": float(comparator_mean[index]),
                "plain_hmc_sd": float(comparator_sd[index]),
                "plain_hmc_mean_ess": float(comparator_ess[index]),
                "plain_hmc_mean_mcse": float(comparator_mcse[index]),
                "abs_mean_difference_over_combined_mcse": float(agreement_z[index]),
                "posterior_agreement_passed": bool(
                    np.isfinite(agreement_z[index])
                    and agreement_z[index] <= POSTERIOR_AGREEMENT_MAX_Z
                ),
                "abs_mean_minus_truth_over_sd": float(recovery_z[index]),
                "recovery_passed": bool(
                    np.isfinite(recovery_z[index])
                    and candidate_sd[index] > 0.0
                    and recovery_z[index] <= RECOVERY_MAX_Z
                ),
                "q05": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q95": float(quantiles[2, index]),
            }
        )
    return {
        "comparator": {
            "path": str(COMPARATOR_PATH.relative_to(ROOT)),
            "file_sha256": EXPECTED_COMPARATOR_FILE_SHA256,
            "shape": tuple(int(item) for item in comparator.shape),
            "chain_axes_preserved": True,
        },
        "mean_mcse_definition": (
            "posterior_sd / sqrt(split-chain cross-chain ESS of raw draws)"
        ),
        "posterior_agreement_definition": (
            "abs(mean_neutra-mean_plain_hmc)/sqrt(mcse_neutra^2+mcse_plain_hmc^2)"
        ),
        "posterior_agreement_threshold": POSTERIOR_AGREEMENT_MAX_Z,
        "max_posterior_agreement_combined_mcse": float(np.max(agreement_z)),
        "posterior_agreement_passed": bool(
            np.all(np.isfinite(agreement_z))
            and np.all(agreement_z <= POSTERIOR_AGREEMENT_MAX_Z)
        ),
        "recovery_threshold_posterior_sd": RECOVERY_MAX_Z,
        "max_abs_mean_minus_truth_over_sd": float(np.max(recovery_z)),
        "recovery_passed": bool(
            np.all(np.isfinite(recovery_z))
            and np.all(candidate_sd > 0.0)
            and np.all(recovery_z <= RECOVERY_MAX_Z)
        ),
        "parameter_rows": tuple(rows),
        "metric_roles": {
            "posterior_agreement": "same_target_material_disagreement_veto",
            "recovery": "single_fixture_supporting_veto_screen",
            "means_sds_quantiles_mcse": "descriptive_and_uncertainty_diagnostics",
        },
        "nonclaims": (
            "comparator agreement is not exact-posterior correctness evidence",
            "single-fixture recovery is not calibration or robustness evidence",
        ),
    }


def _serious_health_screen(
    *,
    samples: np.ndarray,
    raw_samples: np.ndarray,
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    log_accept = np.asarray(trace.get("log_accept_ratio"), dtype=np.float64)
    target = np.asarray(trace.get("target_log_prob"), dtype=np.float64)
    telemetry = diagnostics.get("target_status_telemetry")
    divergence_count = diagnostics.get("divergence_count")
    checks = {
        "z_samples_finite": bool(np.all(np.isfinite(samples))),
        "raw_samples_finite": bool(np.all(np.isfinite(raw_samples))),
        "log_accept_ratio_finite": bool(np.all(np.isfinite(log_accept))),
        "target_log_prob_finite": bool(np.all(np.isfinite(target))),
        "target_status_available": isinstance(telemetry, Mapping),
        "target_status_valid": bool(
            isinstance(telemetry, Mapping)
            and telemetry.get("telemetry_failure_veto") is False
            and telemetry.get("all_status_valid") is True
        ),
        "native_divergence_veto_clear": bool(
            divergence_count is None or int(divergence_count) == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "acceptance_rate": float(diagnostics["acceptance_rate"]),
        "divergence_status": diagnostics.get("divergence_status"),
        "divergence_count": divergence_count,
        "native_divergence_interpretation": (
            "not exposed by this TFP HMC result; not claimed as zero"
            if divergence_count is None
            else "native boolean divergence field counted"
        ),
        "target_status_telemetry": telemetry,
    }
