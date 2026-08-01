"""Target-specific GPU/XLA NeuTra training for the F1 new LGSSM fixture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
F1_ROOT = ROOT / (
    "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-"
    "2026-07-15/f1"
)
F0_ROOT = F1_ROOT.parent / "f0"
CONFIG_PATH = F0_ROOT / "config.json"
FIXTURE_PATH = F0_ROOT / "plain-hmc/fixture_T120_seed20260715_701.json"
MASS_PATH = F0_ROOT / "plain-hmc/mass.json"
COMPARATOR_PATH = F0_ROOT / "plain-hmc/comparator-repair-attempt-02/result.json"
EXPECTED_TARGET_SIGNATURE = (
    "312d2f4ceb5d65bf18251fa53ae1276781c62fd2daefaba0bda8dc3d46a5d283"
)
BATCH_SIZE = 128
SCREEN_STEPS = 500
FINAL_STEPS = 5000
SCREEN_SEED = (20260715, 8101)
FINAL_SEED = (20260715, 8201)
HELDOUT_SEEDS = tuple((20260715, 8301 + index) for index in range(8))
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


class F1TrainingError(RuntimeError):
    """Raised when F1 training identity, runtime, or evidence fails closed."""


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    hidden_layers: tuple[int, ...]
    learning_rate: float

    def payload(self) -> Mapping[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage_count": 3,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": 1.0,
            "batch_size": BATCH_SIZE,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_constant_learning_rate",
        }


RECIPE_ORDER = (
    "inherited_wide_lr5e3",
    "source_width_lr5e3",
    "wide_lower_lr1e3",
)
RECIPES = {
    "inherited_wide_lr5e3": Recipe("inherited_wide_lr5e3", (36, 36), 5e-3),
    "source_width_lr5e3": Recipe("source_width_lr5e3", (18, 18), 5e-3),
    "wide_lower_lr1e3": Recipe("wide_lower_lr1e3", (36, 36), 1e-3),
}

NONCLAIMS = (
    "training and heldout reverse KL are nomination-only",
    "screen differences are descriptive without a supported method ranking",
    "no HMC convergence, posterior correctness, robustness, or superiority claim",
    "no production or default-readiness claim",
)


def run_training_job(*, job_kind: str, recipe_id: str) -> Mapping[str, Any]:
    """Run one fresh graph-native screen or final training job."""

    if job_kind not in {"screen", "final"}:
        raise F1TrainingError("job_kind must be screen or final")
    try:
        recipe = RECIPES[recipe_id]
    except KeyError as exc:
        raise F1TrainingError(f"unknown F1 recipe: {recipe_id}") from exc
    if job_kind == "final":
        selection = _read_mapping(F1_ROOT / "screen/selection.json")
        if (
            selection.get("passed") is not True
            or selection.get("selected_recipe_id") != recipe_id
            or not _artifact_hash_matches(selection)
        ):
            raise F1TrainingError("final recipe is not bound to the screen selection")
    steps = SCREEN_STEPS if job_kind == "screen" else FINAL_STEPS
    seed = SCREEN_SEED if job_kind == "screen" else FINAL_SEED
    root = (
        F1_ROOT / "screen/candidates" / recipe_id / "attempt-01"
        if job_kind == "screen"
        else F1_ROOT / "final" / recipe_id / "attempt-01"
    )
    if root.exists():
        raise FileExistsError(f"F1 training root exists: {root}")
    tf, memory_policy = _trusted_tensorflow()
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
        train_plain_dense_iaf,
    )
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )
    from bayesfilter.testing.lgssm_neutra_strict_training_tf import (
        _compiled_parity,
        _gpu_manifest,
        audit_imported_bayesfilter_closure,
    )

    bundle = load_deterministic_lgssm_exact_target(
        config_path=CONFIG_PATH,
        fixture_path=FIXTURE_PATH,
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    center, factor, geometry = _training_geometry(tf, bundle)
    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=18,
        affine_center=center,
        affine_factor=factor,
        output_dir=root / "training",
        seed=seed,
        hidden_layers=recipe.hidden_layers,
        stage_count=3,
        activation="elu",
        s_max=1.0,
        init_scale=0.02,
        steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=recipe.learning_rate,
        final_learning_rate_fraction=1.0,
        clip_norm=10.0,
        checkpoint_every=steps,
        heartbeat_every=10,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    started = time.monotonic()
    trained = train_plain_dense_iaf(
        adapter=bundle.adapter,
        config=config,
        freeze_transport_id=f"lgssm-f1-{job_kind}-{recipe_id}-{steps}steps",
    )
    if trained.frozen_payload_path is None or trained.completed_steps != steps:
        raise F1TrainingError("F1 training did not freeze at its terminal step")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(trained.frozen_payload_path),
        expected_target_signature=bundle.target_signature,
    )
    flow = restore_plain_dense_iaf_flow(config=config, state_path=trained.state_path)
    parity = _compiled_parity(tf, flow, loaded)
    heldout = _heldout(tf, bundle, loaded)
    closure = audit_imported_bayesfilter_closure()
    if closure["passed"] is not True:
        raise F1TrainingError("F1 repository import closure failed")
    result = _with_hash(
        {
            "schema": "bayesfilter.neutra_robustness_f1_training_job.v1",
            "job_kind": job_kind,
            "recipe_id": recipe_id,
            "passed": True,
            "decision": "PASS_F1_GRAPH_NATIVE_TRAINING_JOB",
            "recipe": recipe.payload(),
            "seed": seed,
            "steps": steps,
            "screen_weights_reused": False,
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "training_state_hash": trained.state_hash,
            "geometry": geometry,
            "payload": _file_reference(trained.frozen_payload_path),
            "checkpoint": _file_reference(trained.state_path),
            "progress": _file_reference(trained.progress_path),
            "records": trained.records,
            "runtime_metadata": trained.runtime_metadata,
            "frozen_reload_and_score_parity": parity,
            "heldout_common_batches": heldout,
            "repository_import_closure": closure,
            "gpu_manifest": _gpu_manifest(tf, gpu_memory_policy=memory_policy),
            "elapsed_seconds": time.monotonic() - started,
            "evidence_role": (
                "proxy_nomination_only" if job_kind == "screen" else
                "engineering_candidate_for_f2_downstream_hmc"
            ),
            "nonclaims": NONCLAIMS,
        }
    )
    _write_new(root / "result.json", result)
    return result


def finalize_screen() -> Mapping[str, Any]:
    rows = []
    for recipe_id in RECIPE_ORDER:
        path = F1_ROOT / "screen/candidates" / recipe_id / "attempt-01/result.json"
        row = _read_mapping(path)
        if (
            row.get("schema") != "bayesfilter.neutra_robustness_f1_training_job.v1"
            or row.get("job_kind") != "screen"
            or row.get("recipe_id") != recipe_id
            or row.get("passed") is not True
            or not _artifact_hash_matches(row)
        ):
            raise F1TrainingError(f"invalid F1 screen row: {recipe_id}")
        rows.append((path, row))
    means = {
        row["recipe_id"]: float(row["heldout_common_batches"]["mean_reverse_kl"])
        for _path, row in rows
    }
    nominal = min(RECIPE_ORDER, key=lambda item: means[item])
    nominal_values = _heldout_values(dict(rows)[
        F1_ROOT / "screen/candidates" / nominal / "attempt-01/result.json"
    ])
    viable = []
    comparisons = []
    for path, row in rows:
        recipe_id = row["recipe_id"]
        values = _heldout_values(row)
        differences = tuple(left - right for left, right in zip(values, nominal_values))
        mean_difference = _mean(differences)
        mcse = _mcse(differences)
        within_two_mcse = bool(mean_difference <= 2.0 * mcse)
        if within_two_mcse:
            viable.append(recipe_id)
        comparisons.append(
            {
                "recipe_id": recipe_id,
                "mean_reverse_kl": means[recipe_id],
                "paired_mean_difference_from_nominal": mean_difference,
                "paired_difference_mcse": mcse,
                "within_two_paired_mcse": within_two_mcse,
                "result": _file_reference(path),
            }
        )
    selected = min(
        viable,
        key=lambda item: (
            _parameter_count(RECIPES[item]),
            RECIPES[item].learning_rate,
            RECIPE_ORDER.index(item),
        ),
    )
    result = _with_hash(
        {
            "schema": "bayesfilter.neutra_robustness_f1_screen_selection.v1",
            "passed": True,
            "decision": "NOMINATE_F1_RECIPE_FOR_FRESH_5000_STEP_TRAINING",
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "nominal_lowest_mean_recipe": nominal,
            "selected_recipe_id": selected,
            "selected_recipe": RECIPES[selected].payload(),
            "comparison_rows": tuple(comparisons),
            "selection_rule": (
                "within_two_paired_mcse_of_lowest_mean_then_parameter_count_"
                "then_learning_rate_then_declared_order"
            ),
            "screen_weights_reused": False,
            "statistically_supported_ranking": False,
            "evidence_role": "proxy_nomination_only_not_transport_promotion",
            "nonclaims": NONCLAIMS,
        }
    )
    _write_new(F1_ROOT / "screen/selection.json", result)
    return result


def _training_geometry(tf: Any, bundle: Any):
    mass = _read_mapping(MASS_PATH)
    comparator = _read_mapping(COMPARATOR_PATH)
    if mass.get("passed") is not True or comparator.get("passed") is not True:
        raise F1TrainingError("F1 requires passing F0 mass and comparator")
    scale = tf.constant(mass["scale"], tf.float64)
    signs = tf.where(
        tf.equal(tf.math.floormod(tf.range(18), 2), 0),
        tf.ones((18,), tf.float64),
        -tf.ones((18,), tf.float64),
    )
    truth_center = tf.constant(mass["center"], tf.float64)
    center = truth_center + tf.constant(0.25, tf.float64) * scale * signs
    factor = tf.constant(mass["factor"], tf.float64)
    return center, factor, {
        "source_mass": _file_reference(MASS_PATH),
        "source_mass_artifact_hash": mass["artifact_hash"],
        "factor_role": "new_fixture_target_specific_mass_factor",
        "center_policy": "mass_center_plus_quarter_prior_scale_alternating_sign",
        "center_equals_truth_or_prior_center": False,
        "center_offset": tuple(float(item) for item in (center - truth_center).numpy()),
        "center_hash": _tensor_hash(tf, center),
        "factor_hash": _tensor_hash(tf, factor),
    }


def _heldout(tf: Any, bundle: Any, loaded: Any) -> Mapping[str, Any]:
    seeds = tf.constant(HELDOUT_SEEDS, tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(seed_tensor):
        z = tf.map_fn(
            lambda seed: tf.random.stateless_normal(
                (BATCH_SIZE, 18), seed=seed, dtype=tf.float64
            ),
            seed_tensor,
            fn_output_signature=tf.TensorSpec((BATCH_SIZE, 18), tf.float64),
        )
        flat = tf.reshape(z, (-1, 18))
        theta = loaded.transport.forward_batch(flat)
        logdet = loaded.transport.log_abs_det_jacobian_batch(flat)
        value, _score, status = bundle.adapter.neutra_batch_log_prob_and_grad_status(theta)
        objective = tf.reshape(-(value + logdet), (len(HELDOUT_SEEDS), BATCH_SIZE))
        return (
            tf.reduce_mean(objective, axis=1),
            tf.math.reduce_std(objective, axis=1),
            tf.reduce_all(tf.equal(status["status_code"], 0)),
            tf.reduce_all(status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(seeds)
    if not bool(outputs[2].numpy()) or not bool(outputs[3].numpy()):
        raise F1TrainingError("F1 heldout target status failed")
    means = tuple(float(item) for item in outputs[0].numpy().tolist())
    return {
        "batch_count": len(HELDOUT_SEEDS),
        "batch_size": BATCH_SIZE,
        "seeds": HELDOUT_SEEDS,
        "reverse_kl_means": means,
        "reverse_kl_sds": tuple(float(item) for item in outputs[1].numpy().tolist()),
        "mean_reverse_kl": _mean(means),
        "mcse_across_batches": _mcse(means),
        "target_status_all_valid": True,
        "single_compiled_heldout_invocation": True,
        "metric_role": "proxy_nomination_only_not_transport_promotion",
    }


def _trusted_tensorflow():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise F1TrainingError("F1 training cannot run with CUDA hidden")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    return tf, policy


def _parameter_count(recipe: Recipe) -> int:
    sizes = (18, *recipe.hidden_layers, 36)
    per_stage = sum(left * right + right for left, right in zip(sizes[:-1], sizes[1:]))
    return 3 * per_stage


def _heldout_values(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(item) for item in row["heldout_common_batches"]["reverse_kl_means"])


def _mean(values: Sequence[float]) -> float:
    return math.fsum(float(item) for item in values) / len(values)


def _mcse(values: Sequence[float]) -> float:
    numeric = tuple(float(item) for item in values)
    mean = _mean(numeric)
    return math.sqrt(math.fsum((item - mean) ** 2 for item in numeric) / ((len(numeric) - 1) * len(numeric)))


def _tensor_hash(tf: Any, value: Any) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise F1TrainingError(f"artifact must be a mapping: {path}")
    return value


def _file_reference(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": _file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = "sha256:" + _stable_hash(result)
    result["artifact_hash_semantics"] = "stable_json_sha256_excluding_artifact_hash_fields"
    return result


def _artifact_hash_matches(payload: Mapping[str, Any]) -> bool:
    clean = {key: value for key, value in payload.items() if key not in {"artifact_hash", "artifact_hash_semantics"}}
    return payload.get("artifact_hash") == "sha256:" + _stable_hash(clean)


def _stable_hash(value: Any) -> str:
    blob = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy().tolist())
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
