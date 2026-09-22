"""Target-specific training protocol for serious LGSSM NeuTra validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


class _LazyParentCampaign:
    def __getattr__(self, name: str) -> Any:
        module = import_module(
            "bayesfilter.testing.lgssm_neutra_serious_validation_tf"
        )
        globals()["parent"] = module
        return getattr(module, name)


parent: Any = _LazyParentCampaign()


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "docs/plans/"
    "bayesfilter-lgssm-neutra-target-specific-training-protocol-amendment-"
    "2026-07-14.md"
)
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/lgssm_neutra_target_specific_protocol_2026_07_14"
)
HISTORICAL_ROOT = ROOT / (
    "docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13"
)
EXPECTED_HISTORICAL_AFFINE_RESULT_SHA256 = (
    "b18f83dc062ce9cc1284d18cb55aded0cfb9f9c45cd0c1c39dba7df72c4a7f22"
)
EXPECTED_HISTORICAL_AFFINE_PAYLOAD_SHA256 = (
    "b113e8a63de4992465a1faf1bf369091ca6c2b29f99b01fcc6c0ea5e3f945af4"
)
EXPECTED_HISTORICAL_AFFINE_ARTIFACT_HASH = (
    "sha256:3996ea7d3e5e0377ce381cf092abf7168c1b76156c8d959a817f533b4ce55494"
)

SCREEN_STEPS = 500
SMOKE_STEPS = 5
FINAL_STEPS = 5000
BATCH_SIZE = 128
CHECKPOINT_EVERY = 50
HEARTBEAT_EVERY = 10
HELDOUT_BATCH_COUNT = 8
HELDOUT_BATCH_SIZE = 128
SCREEN_SEED = (20260714, 1401)
FINAL_SEEDS = {
    "dense_seed1201": (20260713, 1201),
    "dense_seed1202": (20260713, 1202),
}
HELDOUT_SEEDS = tuple((20260714, 1501 + index) for index in range(HELDOUT_BATCH_COUNT))
FINAL_CANDIDATE_ORDER = ("affine_control", *FINAL_SEEDS)

SOURCE_ANCHOR = "source_anchor_lr5e3"
SCREEN_RECIPE_ORDER = (
    SOURCE_ANCHOR,
    "lower_lr1e3",
    "shallow_2stage_lr5e3",
    "wide_2x_lr5e3",
)
SMOKE_SEEDS = {
    recipe_id: (20260714, 1411 + index)
    for index, recipe_id in enumerate(SCREEN_RECIPE_ORDER)
}

NONCLAIMS = (
    "screen held-out reverse KL is nomination-only",
    "four complete recipes do not isolate causal capacity or learning-rate effects",
    "500-step behavior is not serious training evidence",
    "one favorably truth-centered 18D LGSSM fixture only",
    "plain-HMC agreement is not exact-posterior correctness evidence",
    "no sampler superiority, calibration, robustness, or generalization claim",
    "no production or default-readiness claim",
)


class TargetSpecificProtocolError(RuntimeError):
    """Raised when the reviewed target-specific campaign fails closed."""


@dataclass(frozen=True)
class TrainingRecipe:
    recipe_id: str
    stage_count: int
    hidden_layers: tuple[int, ...]
    learning_rate: float

    def payload(self) -> Mapping[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage_count": self.stage_count,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": 1.0,
            "batch_size": BATCH_SIZE,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_constant_learning_rate",
            "composition": "T_phi(z)=affine(dense_iaf_stack(z))",
        }


SCREEN_RECIPES = {
    SOURCE_ANCHOR: TrainingRecipe(SOURCE_ANCHOR, 3, (18, 18), 5.0e-3),
    "lower_lr1e3": TrainingRecipe("lower_lr1e3", 3, (18, 18), 1.0e-3),
    "shallow_2stage_lr5e3": TrainingRecipe(
        "shallow_2stage_lr5e3", 2, (18, 18), 5.0e-3
    ),
    "wide_2x_lr5e3": TrainingRecipe(
        "wide_2x_lr5e3", 3, (36, 36), 5.0e-3
    ),
}


def campaign_seed_ledger() -> Mapping[str, Any]:
    return {
        "screen_seed": SCREEN_SEED,
        "smoke_seeds": dict(SMOKE_SEEDS),
        "heldout_seeds": HELDOUT_SEEDS,
        "final_training_seeds": dict(FINAL_SEEDS),
        "candidate_hmc_seeds": parent.campaign_seed_ledger()["candidate_hmc_seeds"],
    }


def validate_seed_ledger() -> Mapping[str, Any]:
    ledger = campaign_seed_ledger()
    parent_check = parent.validate_seed_ledger()
    new_roots: list[tuple[int, int]] = [tuple(ledger["screen_seed"])]
    new_roots.extend(tuple(value) for value in ledger["smoke_seeds"].values())
    new_roots.extend(tuple(value) for value in ledger["heldout_seeds"])
    new_roots.extend(tuple(value) for value in ledger["final_training_seeds"].values())
    if len(new_roots) != len(set(new_roots)):
        raise TargetSpecificProtocolError("target-specific root seeds must be disjoint")
    raw_parent_roots = {
        tuple(value)
        for row in ledger["candidate_hmc_seeds"].values()
        for value in row.values()
    }
    allowed_shared_final_training = set(FINAL_SEEDS.values())
    overlap = (set(new_roots) - allowed_shared_final_training) & raw_parent_roots
    if overlap:
        raise TargetSpecificProtocolError(
            f"training/screen and HMC root seed families overlap: {sorted(overlap)}"
        )
    return {
        "passed": True,
        "root_seed_count": len(new_roots),
        "parent_hmc_seed_validation": parent_check,
        "ledger": ledger,
    }


def campaign_contract_payload() -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.lgssm_neutra_target_specific_contract.v1",
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
        "dimension": parent.DIMENSION,
        "screen": {
            "recipe_order": SCREEN_RECIPE_ORDER,
            "recipes": tuple(SCREEN_RECIPES[item].payload() for item in SCREEN_RECIPE_ORDER),
            "smoke_steps": SMOKE_STEPS,
            "screen_steps": SCREEN_STEPS,
            "heldout_batch_count": HELDOUT_BATCH_COUNT,
            "heldout_batch_size": HELDOUT_BATCH_SIZE,
            "selection_rule": "lowest_common_heldout_mean_with_source_anchor_paired_mcse_preference",
            "zero_survivor_action": "terminal_no_recipe_result",
        },
        "final_training": {
            "candidate_order": tuple(FINAL_SEEDS),
            "steps": FINAL_STEPS,
            "batch_size": BATCH_SIZE,
            "checkpoint_every": CHECKPOINT_EVERY,
            "heartbeat_every": HEARTBEAT_EVERY,
            "screen_weights_reused": False,
            "max_infrastructure_resumes_per_job": 1,
        },
        "tuning": parent.campaign_contract_payload()["tuning"],
        "serious": parent.campaign_contract_payload()["serious"],
        "runtime": parent.campaign_contract_payload()["runtime"],
        "seed_ledger": campaign_seed_ledger(),
        "immutable_inputs": parent.campaign_contract_payload()["immutable_inputs"],
        "historical_parent_root": str(HISTORICAL_ROOT.relative_to(ROOT)),
        "nonclaims": NONCLAIMS,
    }


def configure_parent_context() -> None:
    parent.configure_execution_context(
        plan_path=PLAN_PATH,
        artifact_root=ARTIFACT_ROOT,
        contract_payload=campaign_contract_payload(),
    )


def write_campaign_contract() -> Mapping[str, Any]:
    parent.validate_static_campaign_inputs()
    validate_seed_ledger()
    payload = dict(campaign_contract_payload())
    payload["contract_hash"] = _stable_json_hash(payload)
    path = ARTIFACT_ROOT / "campaign_contract.json"
    _write_new_json(path, payload)
    return {"path": str(path.relative_to(ROOT)), "contract_hash": payload["contract_hash"]}


def validate_contract() -> Mapping[str, Any]:
    path = ARTIFACT_ROOT / "campaign_contract.json"
    value = _read_mapping(path, "target-specific campaign contract")
    expected = dict(campaign_contract_payload())
    expected["contract_hash"] = _stable_json_hash(expected)
    if _json_ready(value) != _json_ready(expected):
        raise TargetSpecificProtocolError("target-specific campaign contract mismatch")
    return value


def run_gpu_training_job(
    *,
    job_kind: str,
    job_id: str,
    resume_checkpoint: str | Path | None = None,
) -> Mapping[str, Any]:
    """Delegate every training call to the strict graph-native harness."""

    if resume_checkpoint is not None:
        raise TargetSpecificProtocolError(
            "graph-native training has terminal-only checkpoints; implicit "
            "infrastructure resume is unavailable"
        )
    from bayesfilter.testing import lgssm_neutra_strict_training_tf as strict

    return strict.run_gpu_training_job(job_kind=job_kind, job_id=job_id)


def finalize_screen() -> Mapping[str, Any]:
    """Freeze one recipe from complete immutable screen results."""

    validate_contract()
    _require_all_smokes_pass()
    rows = tuple(_resolve_training_job_result("screen", item) for item in SCREEN_RECIPE_ORDER)
    selection = select_screen_recipe(rows)
    result = {
        "schema": "bayesfilter.lgssm_neutra_target_specific_screen_result.v1",
        "phase": "screen",
        "passed": selection["selected_recipe_id"] is not None,
        "decision": (
            "NOMINATE_ONE_RECIPE_FOR_FRESH_LONG_BUDGET_TRAINING"
            if selection["selected_recipe_id"] is not None
            else "STOP_NO_SURVIVING_SCREEN_RECIPE"
        ),
        "all_predeclared_recipes_processed": True,
        "recipe_order": SCREEN_RECIPE_ORDER,
        "candidate_rows": tuple(_screen_result_reference(row) for row in rows),
        "selection": selection,
        "selected_recipe": (
            None
            if selection["selected_recipe_id"] is None
            else SCREEN_RECIPES[str(selection["selected_recipe_id"])].payload()
        ),
        "screen_weights_reused_by_final": False,
        "evidence_role": "proxy_nomination_only_not_transport_promotion",
        "nonclaims": NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(ARTIFACT_ROOT / "screen" / "result.json", result)
    if result["passed"]:
        spec = {
            "schema": "bayesfilter.lgssm_neutra_selected_training_recipe.v1",
            "selected_recipe": result["selected_recipe"],
            "selection_result_artifact_hash": result["artifact_hash"],
            "selection_result": _file_reference(ARTIFACT_ROOT / "screen" / "result.json"),
            "final_steps": FINAL_STEPS,
            "final_seeds": dict(FINAL_SEEDS),
            "screen_weights_reused": False,
            "nonclaims": NONCLAIMS,
        }
        spec = _with_artifact_hash(spec)
        _write_new_json(ARTIFACT_ROOT / "selected_recipe.json", spec)
    return result


def select_screen_recipe(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    row_ids = tuple(str(row.get("job_id")) for row in rows)
    if row_ids != SCREEN_RECIPE_ORDER:
        raise TargetSpecificProtocolError("screen rows must match the exact recipe order")
    by_id = {str(row.get("job_id")): row for row in rows}
    survivors = tuple(item for item in SCREEN_RECIPE_ORDER if by_id[item].get("passed") is True)
    if not survivors:
        return {
            "selected_recipe_id": None,
            "status": "zero_surviving_recipes_terminal",
            "ranking_statistically_supported": False,
            "candidate_failure_table": tuple(
                {"recipe_id": item, "passed": False, "vetoes": by_id[item].get("vetoes", ())}
                for item in SCREEN_RECIPE_ORDER
            ),
        }
    summaries = {item: _heldout_vector(by_id[item]) for item in survivors}
    means = {item: _finite_mean(values) for item, values in summaries.items()}
    provisional = min(survivors, key=lambda item: (means[item], SCREEN_RECIPE_ORDER.index(item)))
    reference = SOURCE_ANCHOR if SOURCE_ANCHOR in survivors else provisional
    paired_rows = {}
    for item in survivors:
        delta = tuple(
            candidate - reference_value
            for candidate, reference_value in zip(
                summaries[item], summaries[reference], strict=True
            )
        )
        mean_delta = _finite_mean(delta)
        mcse = _sample_mean_mcse(delta)
        paired_rows[item] = {
            "reference_recipe_id": reference,
            "mean_difference_candidate_minus_reference": mean_delta,
            "paired_difference_mcse": mcse,
            "within_one_paired_mcse_of_zero": bool(abs(mean_delta) <= mcse),
        }
    selected = provisional
    if reference == SOURCE_ANCHOR and paired_rows[provisional]["within_one_paired_mcse_of_zero"]:
        selected = SOURCE_ANCHOR
    return {
        "selected_recipe_id": selected,
        "status": "one_recipe_nominated",
        "heldout_objective_means": means,
        "paired_comparisons": paired_rows,
        "provisional_lowest_mean_recipe_id": provisional,
        "source_anchor_preference_applied": bool(
            selected == SOURCE_ANCHOR and provisional != SOURCE_ANCHOR
        ),
        "ranking_statistically_supported": False,
        "selection_interpretation": "deterministic_proxy_nomination_only",
    }


def finalize_phase4() -> Mapping[str, Any]:
    """Bind the affine control and both final long-budget candidates."""

    validate_contract()
    selected = _read_mapping(ARTIFACT_ROOT / "selected_recipe.json", "selected recipe")
    final_rows = tuple(_resolve_training_job_result("final", item) for item in FINAL_SEEDS)
    for row in final_rows:
        if row.get("passed") is True and int(row.get("steps", -1)) != FINAL_STEPS:
            raise TargetSpecificProtocolError("passed final candidate is not a 5,000-step freeze")
        if row.get("recipe") != selected.get("selected_recipe"):
            raise TargetSpecificProtocolError("final candidate recipe drifted")

    affine_source = HISTORICAL_ROOT / "phase4" / "affine_control" / "result.json"
    if _file_sha256(affine_source) != EXPECTED_HISTORICAL_AFFINE_RESULT_SHA256:
        raise TargetSpecificProtocolError("historical affine result file hash mismatch")
    affine = _read_mapping(affine_source, "historical affine control")
    _validate_historical_affine_result(affine)
    historical_affine_hash = affine.get("artifact_hash")
    affine_result = {
        **{
            key: value
            for key, value in affine.items()
            if key not in {"artifact_hash", "artifact_hash_semantics"}
        },
        "result_reused_from_historical_parent": _file_reference(affine_source),
        "historical_result_artifact_hash": historical_affine_hash,
        "reused_without_recomputation": True,
        "evidence_role": "unchanged_target_bound_affine_control_reuse",
    }
    affine_result = _with_artifact_hash(affine_result)
    _write_new_json(ARTIFACT_ROOT / "phase4" / "affine_control" / "result.json", affine_result)

    candidate_rows = [affine_result]
    for row in final_rows:
        candidate_id = str(row["job_id"])
        passed = row.get("passed") is True
        result = {
            "schema": "bayesfilter.lgssm_neutra_phase4_candidate_result.v2",
            "phase": 4,
            "candidate_id": candidate_id,
            "passed": passed,
            "decision": (
                "NOMINATE_ENGINEERING_VALID_5000_STEP_FROZEN_CANDIDATE"
                if passed
                else "REJECT_FINAL_TRAINING_CANDIDATE"
            ),
            "target_signature": row["target_signature"],
            "adapter_signature": row["adapter_signature"],
            "payload": row.get("payload"),
            "artifact_signature": row.get("artifact_signature"),
            "transport_hash": row.get("transport_hash"),
            "training_state_hash": row.get("training_state_hash"),
            "training": {
                "executed": True,
                "recipe": row["recipe"],
                "seed": row["seed"],
                "completed_steps": row["steps"],
                "checkpoint": row.get("checkpoint"),
                "progress": row.get("progress"),
                "resumed": row["resumed"],
                "source_training_job_artifact_hash": row["artifact_hash"],
                "vetoes": row.get("vetoes", ()),
                "error": row.get("error"),
            },
            "frozen_reload_parity": row.get("frozen_reload_parity"),
            "frozen_score_parity": row.get("frozen_score_parity"),
            "gpu_fixed_transport_probe": row.get("gpu_fixed_transport_probe"),
            "gpu_manifest": row.get("gpu_manifest"),
            "selected_recipe": selected["selected_recipe"],
            "evidence_role": "engineering_nomination_only_not_candidate_ranking",
            "nonclaims": NONCLAIMS,
        }
        result = _with_artifact_hash(result)
        _write_new_json(ARTIFACT_ROOT / "phase4" / candidate_id / "result.json", result)
        candidate_rows.append(result)

    learned_viable = tuple(
        row["candidate_id"]
        for row in candidate_rows
        if row["candidate_id"] in FINAL_SEEDS and row.get("passed") is True
    )
    phase4_passed = bool(learned_viable)
    aggregate = {
        "schema": "bayesfilter.lgssm_neutra_phase4_result.v2",
        "phase": 4,
        "passed": phase4_passed,
        "decision": (
            "PASS_TARGET_SPECIFIC_LONG_BUDGET_CANDIDATE_FREEZE"
            if phase4_passed
            else "STOP_NO_SURVIVING_TARGET_SPECIFIC_FINAL_CANDIDATE"
        ),
        "candidate_order": FINAL_CANDIDATE_ORDER,
        "viable_candidates": tuple(
            row["candidate_id"] for row in candidate_rows if row.get("passed") is True
        ),
        "rejected_candidates": tuple(
            row["candidate_id"] for row in candidate_rows if row.get("passed") is not True
        ),
        "learned_viable_candidates": learned_viable,
        "affine_control_alone_cannot_pass_target_specific_phase4": True,
        "candidate_results": tuple(
            {
                "candidate_id": row["candidate_id"],
                "result": _file_reference(ARTIFACT_ROOT / "phase4" / row["candidate_id"] / "result.json"),
                "result_artifact_hash": row["artifact_hash"],
                "transport_hash": row.get("transport_hash"),
            }
            for row in candidate_rows
        ),
        "selected_recipe": selected,
        "loss_used_for_selection": True,
        "loss_selection_role": "proxy_nomination_only",
        "truth_centered_affine_geometry": True,
        "nonclaims": NONCLAIMS,
    }
    aggregate = _with_artifact_hash(aggregate)
    _write_new_json(ARTIFACT_ROOT / "phase4" / "result.json", aggregate)
    return aggregate


def run_phase5_candidate(candidate_id: str) -> Mapping[str, Any]:
    configure_parent_context()
    _require_phase4_candidate_for_downstream(candidate_id)
    return parent.run_phase5_candidate(candidate_id)


def finalize_phase5() -> Mapping[str, Any]:
    configure_parent_context()
    _require_target_specific_phase4_survivor()
    return parent.finalize_phase5()


def run_phase6_candidate(candidate_id: str) -> Mapping[str, Any]:
    configure_parent_context()
    _require_phase4_candidate_for_downstream(candidate_id)
    return parent.run_phase6_candidate(candidate_id)


def finalize_phase6() -> Mapping[str, Any]:
    configure_parent_context()
    _require_target_specific_phase4_survivor()
    return parent.finalize_phase6()


def _job_spec(*, job_kind: str, job_id: str) -> tuple[TrainingRecipe, tuple[int, int], int, Path]:
    if job_kind in {"smoke", "screen"}:
        if job_id not in SCREEN_RECIPES:
            raise ValueError(f"unknown screen recipe: {job_id}")
        recipe = SCREEN_RECIPES[job_id]
        if job_kind == "smoke":
            seed = tuple(campaign_seed_ledger()["smoke_seeds"][job_id])
            steps = SMOKE_STEPS
        else:
            seed = SCREEN_SEED
            steps = SCREEN_STEPS
        root = ARTIFACT_ROOT / job_kind / "candidates" / job_id / "attempt_1"
        return recipe, seed, steps, root
    if job_kind == "final":
        if job_id not in FINAL_SEEDS:
            raise ValueError(f"unknown final candidate: {job_id}")
        selected = _read_mapping(ARTIFACT_ROOT / "selected_recipe.json", "selected recipe")
        recipe_id = str(selected["selected_recipe"]["recipe_id"])
        recipe = SCREEN_RECIPES[recipe_id]
        root = ARTIFACT_ROOT / "phase4" / "training_jobs" / job_id / "attempt_1"
        return recipe, FINAL_SEEDS[job_id], FINAL_STEPS, root
    raise ValueError(f"unknown job kind: {job_kind}")


def _fresh_resume_root(first_attempt_root: Path) -> Path:
    return first_attempt_root.with_name("attempt_2_infrastructure_resume")


def _validate_resume_checkpoint(
    *, first_attempt_root: Path, checkpoint: Path, planned_steps: int
) -> None:
    resolved_root = first_attempt_root.resolve()
    resolved_checkpoint = checkpoint.resolve()
    expected_training_root = resolved_root / "training"
    try:
        resolved_checkpoint.relative_to(expected_training_root)
    except ValueError as exc:
        raise TargetSpecificProtocolError(
            "resume checkpoint must belong to the first attempt training directory"
        ) from exc
    if not resolved_checkpoint.is_file():
        raise TargetSpecificProtocolError("resume checkpoint is missing")
    if (resolved_root / "result.json").exists():
        raise TargetSpecificProtocolError("a terminal first-attempt result cannot resume")
    if (expected_training_root / "frozen_transport.json").exists():
        raise TargetSpecificProtocolError("a frozen first attempt cannot resume")
    checkpoints = tuple(sorted(expected_training_root.glob("checkpoint_step_*.json")))
    if not checkpoints or checkpoints[-1].resolve() != resolved_checkpoint:
        raise TargetSpecificProtocolError("resume must use the latest immutable checkpoint")
    state = _read_mapping(resolved_checkpoint, "resume checkpoint")
    completed = int(state.get("completed_steps", -1))
    if completed <= 0 or completed >= int(planned_steps):
        raise TargetSpecificProtocolError("resume checkpoint step is not resumable")
    if completed % CHECKPOINT_EVERY != 0:
        raise TargetSpecificProtocolError("resume checkpoint violates fixed cadence")


def _require_all_smokes_pass() -> None:
    for recipe_id in SCREEN_RECIPE_ORDER:
        row = _resolve_training_job_result("smoke", recipe_id)
        if row.get("passed") is not True or int(row.get("steps", -1)) != SMOKE_STEPS:
            raise TargetSpecificProtocolError(
                f"wiring smoke did not pass for recipe: {recipe_id}"
            )


def _write_failed_training_job(
    *,
    root: Path,
    job_kind: str,
    job_id: str,
    recipe: TrainingRecipe,
    seed: tuple[int, int],
    steps: int,
    resume_checkpoint: str | Path | None,
    error: BaseException,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    training_dir = root / "training"
    progress = training_dir / "training_progress.jsonl"
    checkpoints = tuple(sorted(training_dir.glob("checkpoint_step_*.json")))
    result = {
        "schema": "bayesfilter.lgssm_neutra_target_specific_training_job.v1",
        "job_kind": job_kind,
        "job_id": job_id,
        "passed": False,
        "decision": "REJECT_TRAINING_JOB_AT_ENGINEERING_VETO",
        "recipe": recipe.payload(),
        "seed": seed,
        "steps": steps,
        "screen_weights_reused_by_final": False,
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
        "payload": None,
        "checkpoint": (
            None if not checkpoints else _file_reference(checkpoints[-1])
        ),
        "progress": None if not progress.is_file() else _file_reference(progress),
        "training_state_hash": None,
        "resumed": resume_checkpoint is not None,
        "resume_parent": (
            None if resume_checkpoint is None else _file_reference(Path(resume_checkpoint))
        ),
        "records": (),
        "frozen_reload_parity": None,
        "frozen_score_parity": None,
        "gpu_fixed_transport_probe": None,
        "heldout_common_batches": None,
        "vetoes": ("training_job_engineering_or_numerical_failure",),
        "error": {"type": type(error).__name__, "message": str(error)},
        "elapsed_seconds": elapsed_seconds,
        "evidence_role": "candidate_rejection_not_research_direction_rejection",
        "nonclaims": NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(root / "result.json", result)
    return result


def _resolve_training_job_result(job_kind: str, job_id: str) -> Mapping[str, Any]:
    recipe, seed, steps, first = _job_spec(job_kind=job_kind, job_id=job_id)
    candidates = (
        first / "result.json",
        first.with_name("attempt_1_graph_native") / "result.json",
        _fresh_resume_root(first) / "result.json",
    )
    existing = tuple(path for path in candidates if path.is_file())
    if len(existing) != 1:
        raise TargetSpecificProtocolError(
            f"expected exactly one terminal {job_kind} result for {job_id}, found {len(existing)}"
        )
    row = _read_mapping(existing[0], f"{job_kind} {job_id} result")
    if row.get("job_kind") != job_kind or row.get("job_id") != job_id:
        raise TargetSpecificProtocolError("training job result identity mismatch")
    if _json_ready(row.get("recipe")) != _json_ready(recipe.payload()):
        raise TargetSpecificProtocolError("training job recipe identity mismatch")
    if tuple(row.get("seed", ())) != tuple(seed):
        raise TargetSpecificProtocolError("training job seed identity mismatch")
    if int(row.get("steps", -1)) != int(steps):
        raise TargetSpecificProtocolError("training job step-budget identity mismatch")
    if row.get("target_signature") != parent.EXPECTED_TARGET_SIGNATURE:
        raise TargetSpecificProtocolError("training job target signature mismatch")
    if row.get("adapter_signature") != parent.EXPECTED_ADAPTER_SIGNATURE:
        raise TargetSpecificProtocolError("training job adapter signature mismatch")
    return row


def _require_target_specific_phase4_survivor() -> Mapping[str, Any]:
    result = _read_mapping(ARTIFACT_ROOT / "phase4" / "result.json", "Phase 4 result")
    learned = tuple(result.get("learned_viable_candidates", ()))
    if result.get("passed") is not True or not learned:
        raise TargetSpecificProtocolError(
            "Phase 4 has no surviving target-specific long-budget candidate"
        )
    if any(candidate not in FINAL_SEEDS for candidate in learned):
        raise TargetSpecificProtocolError("Phase 4 learned candidate identity mismatch")
    viable = tuple(result.get("viable_candidates", ()))
    if any(candidate not in viable for candidate in learned):
        raise TargetSpecificProtocolError("Phase 4 learned/viable candidate mismatch")
    candidate_refs = {
        str(row.get("candidate_id")): row
        for row in tuple(result.get("candidate_results", ()))
        if isinstance(row, Mapping)
    }
    for candidate in learned:
        phase4_path = ARTIFACT_ROOT / "phase4" / candidate / "result.json"
        phase4_row = _read_mapping(phase4_path, f"Phase 4 {candidate} result")
        reference = candidate_refs.get(candidate)
        if not isinstance(reference, Mapping):
            raise TargetSpecificProtocolError("Phase 4 candidate reference is missing")
        _verify_file_reference(reference.get("result"), phase4_path, "Phase 4 candidate")
        if reference.get("result_artifact_hash") != phase4_row.get("artifact_hash"):
            raise TargetSpecificProtocolError("Phase 4 candidate artifact hash mismatch")
        if (
            phase4_row.get("phase") != 4
            or phase4_row.get("candidate_id") != candidate
            or phase4_row.get("passed") is not True
            or phase4_row.get("target_signature") != parent.EXPECTED_TARGET_SIGNATURE
            or phase4_row.get("adapter_signature") != parent.EXPECTED_ADAPTER_SIGNATURE
        ):
            raise TargetSpecificProtocolError("Phase 4 learned candidate result is invalid")
        training_job = _resolve_training_job_result("final", candidate)
        training = phase4_row.get("training")
        if not isinstance(training, Mapping):
            raise TargetSpecificProtocolError("Phase 4 learned candidate training is missing")
        if training.get("source_training_job_artifact_hash") != training_job.get(
            "artifact_hash"
        ):
            raise TargetSpecificProtocolError("Phase 4/training-job artifact mismatch")
        for key in ("payload", "checkpoint", "progress"):
            if _json_ready(phase4_row.get(key) if key == "payload" else training.get(key)) != _json_ready(
                training_job.get(key)
            ):
                raise TargetSpecificProtocolError(
                    f"Phase 4/training-job {key} reference mismatch"
                )
            _verify_file_reference(
                training_job.get(key),
                _expected_training_job_artifact_path(
                    job_kind="final", job_id=candidate, artifact=key
                ),
                f"{candidate} {key}",
            )
    return result


def _require_phase4_candidate_for_downstream(candidate_id: str) -> Mapping[str, Any]:
    candidate = str(candidate_id)
    if candidate not in FINAL_CANDIDATE_ORDER:
        raise TargetSpecificProtocolError(f"unknown downstream candidate: {candidate}")
    result = _require_target_specific_phase4_survivor()
    viable = tuple(result.get("viable_candidates", ()))
    if candidate not in viable:
        raise TargetSpecificProtocolError(
            f"requested candidate is not Phase 4 viable: {candidate}"
        )
    if candidate in FINAL_SEEDS:
        learned = tuple(result.get("learned_viable_candidates", ()))
        if candidate not in learned:
            raise TargetSpecificProtocolError(
                f"requested learned candidate was not revalidated: {candidate}"
            )
        return result

    affine_path = ARTIFACT_ROOT / "phase4" / "affine_control" / "result.json"
    affine = _read_mapping(affine_path, "rebound affine control")
    if (
        affine.get("phase") != 4
        or affine.get("candidate_id") != "affine_control"
        or affine.get("passed") is not True
        or affine.get("target_signature") != parent.EXPECTED_TARGET_SIGNATURE
        or affine.get("adapter_signature") != parent.EXPECTED_ADAPTER_SIGNATURE
        or affine.get("historical_result_artifact_hash")
        != EXPECTED_HISTORICAL_AFFINE_ARTIFACT_HASH
    ):
        raise TargetSpecificProtocolError("rebound affine control identity mismatch")
    source = affine.get("result_reused_from_historical_parent")
    _verify_file_reference(
        source,
        HISTORICAL_ROOT / "phase4" / "affine_control" / "result.json",
        "rebound affine historical source",
    )
    payload = affine.get("payload")
    _verify_file_reference(
        payload,
        HISTORICAL_ROOT / "phase4" / "affine_control" / "frozen_transport.json",
        "rebound affine payload",
    )
    return result


def _validate_historical_affine_result(affine: Mapping[str, Any]) -> None:
    if (
        affine.get("phase") != 4
        or affine.get("candidate_id") != "affine_control"
        or affine.get("passed") is not True
        or affine.get("target_signature") != parent.EXPECTED_TARGET_SIGNATURE
        or affine.get("adapter_signature") != parent.EXPECTED_ADAPTER_SIGNATURE
        or affine.get("artifact_hash") != EXPECTED_HISTORICAL_AFFINE_ARTIFACT_HASH
    ):
        raise TargetSpecificProtocolError("historical affine control identity mismatch")
    payload = affine.get("payload")
    if not isinstance(payload, Mapping):
        raise TargetSpecificProtocolError("historical affine payload reference is missing")
    expected_path = HISTORICAL_ROOT / "phase4" / "affine_control" / "frozen_transport.json"
    _verify_file_reference(payload, expected_path, "historical affine payload")
    if payload.get("file_sha256") != EXPECTED_HISTORICAL_AFFINE_PAYLOAD_SHA256:
        raise TargetSpecificProtocolError("historical affine payload hash mismatch")


def _verify_reference_payload(reference: Any, label: str) -> Path:
    if not isinstance(reference, Mapping) or not reference.get("path"):
        raise TargetSpecificProtocolError(f"{label} reference is invalid")
    path = (ROOT / str(reference["path"])).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise TargetSpecificProtocolError(f"{label} path leaves the repository") from exc
    if not path.is_file():
        raise TargetSpecificProtocolError(f"{label} file is missing")
    if reference.get("file_sha256") != _file_sha256(path):
        raise TargetSpecificProtocolError(f"{label} file hash mismatch")
    if int(reference.get("byte_count", -1)) != path.stat().st_size:
        raise TargetSpecificProtocolError(f"{label} byte count mismatch")
    return path


def _verify_file_reference(
    reference: Any, expected_path: Path, label: str
) -> Path:
    path = _verify_reference_payload(reference, label)
    if path != expected_path.resolve():
        raise TargetSpecificProtocolError(f"{label} path mismatch")
    return path


def _expected_training_job_artifact_path(
    *, job_kind: str, job_id: str, artifact: str
) -> Path:
    result_path = _resolve_result_path(job_kind, job_id)
    training_root = result_path.parent / "training"
    _recipe, _seed, steps, _first = _job_spec(job_kind=job_kind, job_id=job_id)
    filenames = {
        "payload": "frozen_transport.json",
        "checkpoint": f"checkpoint_step_{steps:06d}.json",
        "progress": "training_progress.jsonl",
    }
    try:
        filename = filenames[artifact]
    except KeyError as exc:
        raise ValueError(f"unknown training-job artifact: {artifact}") from exc
    return training_root / filename


def _common_heldout_summary(*, tf: Any, bundle: Any, loaded: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_target_specific_heldout_{loaded.manifest.transport_hash}",
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

    rows = []
    for seed in HELDOUT_SEEDS:
        with tf.device("/GPU:0"):
            z = tf.random.stateless_normal(
                (HELDOUT_BATCH_SIZE, parent.DIMENSION), seed=seed, dtype=tf.float64
            )
            value, score, logdet, status = compiled(z)
        if not all(bool(tf.reduce_all(tf.math.is_finite(item)).numpy()) for item in (value, score, logdet)):
            raise TargetSpecificProtocolError("held-out diagnostic is nonfinite")
        valid = bool(
            tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
            and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        )
        if not valid:
            raise TargetSpecificProtocolError("held-out target status failed")
        objective = -(value + logdet)
        force_norm = tf.linalg.norm(score, axis=-1)
        rows.append(
            {
                "seed": seed,
                "reverse_kl_objective_mean": float(tf.reduce_mean(objective).numpy()),
                "reverse_kl_objective_sd": float(tf.math.reduce_std(objective).numpy()),
                "transformed_force_norm_mean": float(tf.reduce_mean(force_norm).numpy()),
                "transformed_force_norm_max": float(tf.reduce_max(force_norm).numpy()),
                "target_status_all_valid": True,
                "device": str(value.device),
            }
        )
    objectives = tuple(float(row["reverse_kl_objective_mean"]) for row in rows)
    return {
        "batch_count": HELDOUT_BATCH_COUNT,
        "batch_size": HELDOUT_BATCH_SIZE,
        "common_seed_policy": "identical_stateless_base_draws_for_every_recipe",
        "rows": tuple(rows),
        "mean_reverse_kl_objective": _finite_mean(objectives),
        "mcse_across_batches": _sample_mean_mcse(objectives),
        "target_status_all_valid": True,
        "metric_role": "proxy_nomination_only_not_transport_promotion",
    }


def _trainable_frozen_score_parity(
    *, tf: Any, bundle: Any, flow: Any, loaded: Any, job_kind: str, job_id: str
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
        reviewed_value_score_target_fn,
    )

    target_value = reviewed_value_score_target_fn(
        bundle.adapter, dtype=tf.float64, require_batched=True
    )
    fixed = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_target_specific_score_parity_{job_kind}_{job_id}",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def trainable_reference(z_arg):
        with tf.GradientTape() as tape:
            tape.watch(z_arg)
            theta, logdet = flow.forward_and_logdet(z_arg)
            value = target_value(theta) + logdet
            objective = tf.reduce_sum(value)
        score = tape.gradient(objective, z_arg)
        return theta, logdet, value, score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def frozen_explicit(z_arg):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = fixed.log_prob_and_grad_batch(z_arg)
        return theta, logdet, value, score

    with tf.device("/GPU:0"):
        z = tf.constant(parent.common_probe_points(), dtype=tf.float64)
        reference = trainable_reference(z)
        explicit = frozen_explicit(z)
    devices = tuple(str(item.device) for item in (*reference, *explicit))
    if not all("GPU" in device.upper() for device in devices):
        raise TargetSpecificProtocolError("frozen score parity fell back from GPU")
    parity = parent._score_parity_summary(
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
        raise TargetSpecificProtocolError("frozen explicit-score parity failed")
    return {**parity, "all_outputs_on_gpu": True, "output_devices": devices}


def _heldout_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    heldout = row.get("heldout_common_batches")
    if not isinstance(heldout, Mapping):
        raise TargetSpecificProtocolError("surviving screen row lacks held-out summary")
    rows = tuple(heldout.get("rows", ()))
    if len(rows) != HELDOUT_BATCH_COUNT:
        raise TargetSpecificProtocolError("held-out batch count mismatch")
    seeds = tuple(tuple(item["seed"]) for item in rows)
    if seeds != HELDOUT_SEEDS:
        raise TargetSpecificProtocolError("held-out common seed order mismatch")
    values = tuple(float(item["reverse_kl_objective_mean"]) for item in rows)
    if len(values) != HELDOUT_BATCH_COUNT or not all(math.isfinite(value) for value in values):
        raise TargetSpecificProtocolError("held-out objective vector is invalid")
    return values


def _finite_mean(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise TargetSpecificProtocolError("mean inputs must be finite and nonempty")
    return math.fsum(numeric) / len(numeric)


def _sample_mean_mcse(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    if len(numeric) <= 1:
        return math.inf
    mean = _finite_mean(numeric)
    sample_variance = math.fsum((value - mean) ** 2 for value in numeric) / (
        len(numeric) - 1
    )
    return math.sqrt(sample_variance / len(numeric))


def _screen_result_reference(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _resolve_result_path(str(row["job_kind"]), str(row["job_id"]))
    heldout = row.get("heldout_common_batches")
    return {
        "recipe_id": row["job_id"],
        "passed": row["passed"],
        "result": _file_reference(result),
        "result_artifact_hash": row["artifact_hash"],
        "vetoes": row.get("vetoes", ()),
        "heldout_mean": (
            None if not isinstance(heldout, Mapping) else heldout["mean_reverse_kl_objective"]
        ),
        "heldout_mcse": (
            None if not isinstance(heldout, Mapping) else heldout["mcse_across_batches"]
        ),
    }


def _resolve_result_path(job_kind: str, job_id: str) -> Path:
    _recipe, _seed, _steps, first = _job_spec(job_kind=job_kind, job_id=job_id)
    paths = (
        first / "result.json",
        first.with_name("attempt_1_graph_native") / "result.json",
        _fresh_resume_root(first) / "result.json",
    )
    existing = tuple(path for path in paths if path.is_file())
    if len(existing) != 1:
        raise TargetSpecificProtocolError("training job result path is ambiguous")
    return existing[0]


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetSpecificProtocolError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise TargetSpecificProtocolError(f"{label} must be a JSON object")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "file_sha256": _file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _with_artifact_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = f"sha256:{_stable_json_hash(result)}"
    result["artifact_hash_semantics"] = "stable_json_sha256_excluding_artifact_hash_fields"
    return result


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _json_ready(materialized.tolist())
        if hasattr(materialized, "item"):
            return _json_ready(materialized.item())
        return _json_ready(materialized)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
