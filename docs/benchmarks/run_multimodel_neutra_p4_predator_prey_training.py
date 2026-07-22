"""Run and select P4 target-specific predator-prey dense-IAF training jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base


PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-target-specific-training-subplan-2026-07-16.md"
)
CELLS = ("PP-UKF", "PP-SGQF")
SCREEN_STEPS = 500
FINAL_STEPS = 5000
BATCH_SIZE = 128
HELDOUT_BATCH_COUNT = 8
HELDOUT_BATCH_SIZE = 128
RECIPE_ORDER = (
    "source_width_lr1e3",
    "source_width_lr5e3",
    "wide_lr1e3",
    "wide_lr5e3",
)
SCREEN_SEEDS = {
    "PP-UKF": {
        recipe: (20260716, 10001 + index)
        for index, recipe in enumerate(RECIPE_ORDER)
    },
    "PP-SGQF": {
        recipe: (20260716, 11001 + index)
        for index, recipe in enumerate(RECIPE_ORDER)
    },
}
HELDOUT_SEEDS = {"PP-UKF": (20260716, 10100), "PP-SGQF": (20260716, 11100)}
FINAL_SEEDS = {"PP-UKF": (20260716, 10201), "PP-SGQF": (20260716, 11201)}
COMPARATOR_ROOTS = {
    "PP-UKF": base.PHASE_ROOT
    / "PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z",
    "PP-SGQF": base.PHASE_ROOT
    / "PP-SGQF/plain-hmc-laplace/attempt-01-20260715T170000Z",
}
EXPECTED_COMPARATOR_HASHES = {
    "PP-UKF": "4c7e001b181033f4191acf5a6dd841c2dc507c4b25c015ce69817976eec345d5",
    "PP-SGQF": "015348e162d35cb062be274eb4b420ee881eb364473b5b7ce5acfdca7c0192ec",
}
GEOMETRY_ROOTS = {
    "PP-UKF": COMPARATOR_ROOTS["PP-UKF"],
    "PP-SGQF": base.PHASE_ROOT
    / "PP-SGQF/laplace-geometry/attempt-01-20260715T165000Z",
}
EXPECTED_GEOMETRY_HASHES = {
    "PP-UKF": EXPECTED_COMPARATOR_HASHES["PP-UKF"],
    "PP-SGQF": "b54343fdee59c3f86ffb8f8ac69ba0ea31b7a0c780a4f2eb290374df060cabc3",
}
NONCLAIMS = (
    "training and common-heldout reverse KL are nomination-only",
    "screen weights are never reused by final training",
    "no transported HMC convergence or plain-HMC agreement claim",
    "no filter exactness, superiority, calibration, robustness, or readiness claim",
)


class P4TrainingError(RuntimeError):
    """Raised when a frozen P4 training boundary fails closed."""


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


RECIPES = {
    "source_width_lr1e3": Recipe("source_width_lr1e3", (18, 18), 1.0e-3),
    "source_width_lr5e3": Recipe("source_width_lr5e3", (18, 18), 5.0e-3),
    "wide_lr1e3": Recipe("wide_lr1e3", (36, 36), 1.0e-3),
    "wide_lr5e3": Recipe("wide_lr5e3", (36, 36), 5.0e-3),
}


def run_training_job(
    *, cell_id: str, job_kind: str, recipe_id: str, output_root: Path
) -> Mapping[str, Any]:
    cell = _cell(cell_id)
    if job_kind not in {"screen", "final"}:
        raise P4TrainingError("job_kind must be screen or final")
    try:
        recipe = RECIPES[recipe_id]
    except KeyError as exc:
        raise P4TrainingError(f"unknown training recipe: {recipe_id}") from exc
    if output_root.exists():
        raise FileExistsError(f"training output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_campaign import (
        load_campaign_neutra_transport,
        train_campaign_neutra,
    )
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
    )

    adapter, identity, identity_reference = _reconstruct_identity(tf, cell)
    center, factor, geometry_reference = _load_geometry(tf, cell)
    comparator_reference = _verify_result_root(
        COMPARATOR_ROOTS[cell], EXPECTED_COMPARATOR_HASHES[cell], require_passed=True
    )
    if job_kind == "final":
        selection = _load_selection(cell)
        if selection["selected_recipe_id"] != recipe_id:
            raise P4TrainingError("final recipe does not match frozen screen selection")
        seed = FINAL_SEEDS[cell]
        steps = FINAL_STEPS
    else:
        selection = None
        seed = SCREEN_SEEDS[cell][recipe_id]
        steps = SCREEN_STEPS

    config = PlainDenseIAFTrainingConfig(
        target_signature=identity.target_signature,
        dimension=6,
        affine_center=center,
        affine_factor=factor,
        output_dir=output_root / "training",
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
    trained = train_campaign_neutra(
        identity=identity,
        adapter=adapter,
        config=config,
        freeze_transport_id=f"p4-{cell.lower()}-{job_kind}-{recipe_id}-{steps}",
        gpu_memory_policy=memory_policy,
    )
    if trained.completed_steps != steps or trained.frozen_payload_path is None:
        raise P4TrainingError("training did not freeze at the terminal step")
    payload = _read_mapping(trained.frozen_payload_path)
    loaded = load_campaign_neutra_transport(
        identity=identity, adapter=adapter, payload=payload
    )
    flow = restore_plain_dense_iaf_flow(config=config, state_path=trained.state_path)
    parity = _compiled_parity(tf, flow, loaded, cell=cell, recipe_id=recipe_id)
    heldout = _heldout(tf, adapter, loaded, cell=cell)
    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_training_job.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": cell,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "passed": True,
        "decision": (
            "PASS_P4_TRAINING_SCREEN_JOB"
            if job_kind == "screen"
            else "ADMIT_P4_FRESH_5000_STEP_FROZEN_TRAINING"
        ),
        "recipe": recipe.payload(),
        "seed": seed,
        "steps": steps,
        "screen_weights_reused_by_final": False,
        "target_identity": identity.payload(),
        "identity_reference": identity_reference,
        "comparator_reference": comparator_reference,
        "geometry_reference": geometry_reference,
        "selection_reference": selection,
        "training_state_hash": trained.state_hash,
        "transport_artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "payload": _file_reference(trained.frozen_payload_path),
        "checkpoint": _file_reference(trained.state_path),
        "progress": _file_reference(trained.progress_path),
        "records": trained.records,
        "runtime_metadata": trained.runtime_metadata,
        "frozen_trainable_parity": parity,
        "heldout_common_batches": heldout,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_role": (
            "proxy_nomination_only"
            if job_kind == "screen"
            else "engineering_candidate_for_r4_neutra_hmc"
        ),
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_root / "result.json", result)
    _write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            cell=cell,
            job_kind=job_kind,
            recipe_id=recipe_id,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            wall_time=time.monotonic() - started,
        ),
    )
    _write_recursive_hashes(output_root)
    return result


def finalize_screen(*, cell_id: str, output_path: Path) -> Mapping[str, Any]:
    cell = _cell(cell_id)
    if output_path.exists():
        raise FileExistsError(f"screen selection already exists: {output_path}")
    rows = []
    for recipe_id in RECIPE_ORDER:
        root = _default_job_root(cell, "screen", recipe_id)
        _verify_result_root(root, _file_sha256(root / "result.json"), require_passed=True)
        row = _read_mapping(root / "result.json")
        if (
            row.get("cell_id") != cell
            or row.get("job_kind") != "screen"
            or row.get("recipe_id") != recipe_id
            or _json_ready(row.get("recipe")) != _json_ready(RECIPES[recipe_id].payload())
        ):
            raise P4TrainingError(f"invalid screen row: {recipe_id}")
        rows.append((root, row))
    means = {
        row["recipe_id"]: float(row["heldout_common_batches"]["mean_reverse_kl"])
        for _root, row in rows
    }
    nominal = min(RECIPE_ORDER, key=lambda item: means[item])
    nominal_values = _heldout_values(dict(rows)[_default_job_root(cell, "screen", nominal)])
    viable = []
    comparisons = []
    for root, row in rows:
        recipe_id = row["recipe_id"]
        values = _heldout_values(row)
        differences = tuple(left - right for left, right in zip(values, nominal_values))
        mean_difference = _mean(differences)
        paired_mcse = _mcse(differences)
        within = bool(mean_difference <= 2.0 * paired_mcse)
        if within:
            viable.append(recipe_id)
        comparisons.append(
            {
                "recipe_id": recipe_id,
                "mean_reverse_kl": means[recipe_id],
                "paired_mean_difference_from_nominal": mean_difference,
                "paired_difference_mcse": paired_mcse,
                "within_two_paired_mcse": within,
                "result": _file_reference(root / "result.json"),
                "artifact_hashes": _file_reference(root / "artifact_hashes.json"),
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
    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_training_selection.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": cell,
        "passed": True,
        "decision": "NOMINATE_P4_RECIPE_FOR_FRESH_5000_STEP_TRAINING",
        "recipe_order": RECIPE_ORDER,
        "nominal_lowest_mean_recipe": nominal,
        "selected_recipe_id": selected,
        "selected_recipe": RECIPES[selected].payload(),
        "comparison_rows": comparisons,
        "selection_rule": "within_two_paired_mcse_of_lowest_mean_then_parameter_count_then_learning_rate_then_declared_order",
        "statistically_supported_ranking": False,
        "screen_weights_reused_by_final": False,
        "evidence_role": "proxy_nomination_only_not_transport_promotion",
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_path, result)
    return result


def _reconstruct_identity(tf: Any, cell: str) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_campaign import (
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        generate_frozen_predator_prey_dataset_tf,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    root = base.IDENTITY_ROOTS[cell]
    reference = base._verify_source_root(root)
    source = _read_mapping(root / "result.json")
    expected = _read_mapping(root / "target_identity.json")
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    if cell == "PP-UKF":
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            PredatorPreyUKFLikelihoodRecomposer,
            make_predator_prey_ukf_neutra_adapter,
        )

        adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
        recomposer = PredatorPreyUKFLikelihoodRecomposer(adapter)
    else:
        from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
            PredatorPreySGQFLikelihoodRecomposer,
            make_predator_prey_sgqf_neutra_adapter,
        )

        if source.get("selected_level") != 2:
            raise P4TrainingError("PP-SGQF selected level drift")
        adapter = make_predator_prey_sgqf_neutra_adapter(
            sparse_level=2, observations=observations
        )
        recomposer = PredatorPreySGQFLikelihoodRecomposer(adapter)
    admission = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=tf.constant(source["audit_points"], tf.float64),
        prior_value_score_fn=source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=source_six_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    registry = _read_mapping(root / "repaired_registry.json")
    identity = issue_typed_neutra_target_identity(
        program_id=base.PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=cell,
        adapter=adapter,
        recomposition=admission,
        registry_row=registry,
        registry_artifact_sha256=_file_sha256(root / "repaired_registry.json"),
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if _json_ready(identity.payload()) != expected:
        raise P4TrainingError("training target identity drift")
    return adapter, identity, reference


def _load_geometry(tf: Any, cell: str) -> tuple[Any, Any, Mapping[str, Any]]:
    root = GEOMETRY_ROOTS[cell]
    reference = _verify_result_root(
        root, EXPECTED_GEOMETRY_HASHES[cell], require_passed=True
    )
    result = _read_mapping(root / "result.json")
    if cell == "PP-UKF":
        mass_path = root / "affine_mass.json"
        mass = _read_mapping(mass_path)
        center = tf.constant(mass["center"], tf.float64)
        factor = tf.constant(mass["cholesky_factor"], tf.float64)
        reference = {**reference, "mass": _file_reference(mass_path)}
    else:
        center = tf.constant(result["final_geometry"]["center"], tf.float64)
        factor = tf.constant(
            result["final_geometry"]["cholesky_factor"], tf.float64
        )
    return center, factor, reference


def _compiled_parity(
    tf: Any, flow: Any, loaded: Any, *, cell: str, recipe_id: str
) -> Mapping[str, Any]:
    probes = tf.constant(
        [[0.0] * 6, [0.1, -0.1, 0.08, -0.08, 0.06, -0.06]], tf.float64
    )
    score_seed = SCREEN_SEEDS[cell][recipe_id]
    theta_score = tf.random.stateless_normal(
        tf.shape(probes), seed=(score_seed[0], score_seed[1] + 700), dtype=tf.float64
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg, theta_score_arg):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_arg)
            theta_train, logdet_train = flow.forward_and_logdet(z_arg)
            pullback_objective = tf.reduce_sum(theta_train * theta_score_arg)
            logdet_objective = tf.reduce_sum(logdet_train)
        pullback_train = tape.gradient(pullback_objective, z_arg)
        logdet_score_train = tape.gradient(logdet_objective, z_arg)
        theta_frozen = loaded.transport.forward_batch(z_arg)
        logdet_frozen = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        pullback_frozen = loaded.transport.pullback_score_batch(z_arg, theta_score_arg)
        logdet_score_frozen = loaded.transport.log_abs_det_jacobian_score_batch(z_arg)
        return (
            tf.reduce_max(tf.abs(theta_train - theta_frozen)),
            tf.reduce_max(tf.abs(logdet_train - logdet_frozen)),
            tf.reduce_max(tf.abs(pullback_train - pullback_frozen)),
            tf.reduce_max(tf.abs(logdet_score_train - logdet_score_frozen)),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(probes, theta_score)
    gaps = tuple(float(item.numpy()) for item in outputs)
    passed = all(item <= 1.0e-10 for item in gaps) and all(
        "GPU" in str(item.device).upper() for item in outputs
    )
    if not passed:
        raise P4TrainingError("frozen/trainable transport parity failed")
    return {
        "passed": True,
        "transport_max_abs": gaps[0],
        "logdet_max_abs": gaps[1],
        "pullback_score_max_abs": gaps[2],
        "logdet_score_max_abs": gaps[3],
        "output_devices": tuple(str(item.device) for item in outputs),
        "jit_compile": True,
    }


def _heldout(tf: Any, adapter: Any, loaded: Any, *, cell: str) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    fixed = FixedTransportValueScoreAdapter(
        base_adapter=adapter,
        transport=loaded.transport,
        target_scope=f"p4_{cell.lower()}_training_heldout",
        evidence_path=str(PLAN_PATH),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(seed):
        z = tf.random.stateless_normal(
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE, 6),
            seed=seed,
            dtype=tf.float64,
        )
        flat = tf.reshape(z, (-1, 6))
        value, score = fixed.log_prob_and_grad_batch(flat)
        status = fixed.target_status_telemetry(flat)
        objective = tf.reshape(-value, (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE))
        force = tf.reshape(
            tf.linalg.norm(score, axis=-1),
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE),
        )
        return (
            tf.reduce_mean(objective, axis=1),
            tf.math.reduce_std(objective, axis=1),
            tf.reduce_mean(force, axis=1),
            tf.reduce_max(force, axis=1),
            tf.reduce_all(tf.equal(status["status_code"], 0)),
            tf.reduce_all(status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(tf.constant(HELDOUT_SEEDS[cell], tf.int32))
    if not bool(outputs[4].numpy()) or not bool(outputs[5].numpy()):
        raise P4TrainingError("common-heldout target status failed")
    means = tuple(float(item) for item in outputs[0].numpy().tolist())
    if not all(math.isfinite(item) for item in means):
        raise P4TrainingError("common-heldout objective is nonfinite")
    return {
        "batch_count": HELDOUT_BATCH_COUNT,
        "batch_size": HELDOUT_BATCH_SIZE,
        "root_seed": HELDOUT_SEEDS[cell],
        "common_seed_policy": "one_identical_stateless_8x128x6_base_tensor_per_cell",
        "reverse_kl_means": means,
        "reverse_kl_sds": tuple(float(item) for item in outputs[1].numpy().tolist()),
        "transformed_force_means": tuple(
            float(item) for item in outputs[2].numpy().tolist()
        ),
        "transformed_force_maxima": tuple(
            float(item) for item in outputs[3].numpy().tolist()
        ),
        "mean_reverse_kl": _mean(means),
        "mcse_across_batches": _mcse(means),
        "target_status_all_valid": True,
        "single_compiled_heldout_invocation": True,
        "output_devices": tuple(str(item.device) for item in outputs),
        "metric_role": "proxy_nomination_only_not_transport_promotion",
    }


def _load_selection(cell: str) -> Mapping[str, Any]:
    path = _selection_path(cell)
    selection = _read_mapping(path)
    if (
        selection.get("schema") != "bayesfilter.multimodel_neutra_p4_training_selection.v1"
        or selection.get("cell_id") != cell
        or selection.get("passed") is not True
        or selection.get("selected_recipe_id") not in RECIPES
    ):
        raise P4TrainingError("invalid frozen training selection")
    return {**selection, "path": str(path), "sha256": _file_sha256(path)}


def _verify_result_root(
    root: Path, expected_result_hash: str, *, require_passed: bool
) -> Mapping[str, Any]:
    result_path = root / "result.json"
    if _file_sha256(result_path) != expected_result_hash:
        raise P4TrainingError(f"result hash mismatch: {root}")
    hashes = _read_mapping(root / "artifact_hashes.json")["artifacts"]
    for relative_path, expected in hashes.items():
        if _file_sha256(root / relative_path) != expected:
            raise P4TrainingError(f"artifact hash mismatch: {root / relative_path}")
    result = _read_mapping(result_path)
    if require_passed and result.get("passed") is not True:
        raise P4TrainingError(f"required passing result is blocked: {root}")
    return {
        "root": str(root),
        "result_sha256": expected_result_hash,
        "artifact_hashes_sha256": _file_sha256(root / "artifact_hashes.json"),
    }


def _run_manifest(
    *,
    output_root: Path,
    cell: str,
    job_kind: str,
    recipe_id: str,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    wall_time: float,
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p4_training_manifest.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": cell,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_training.py "
            f"--action run --cell {cell} --job-kind {job_kind} --recipe {recipe_id} "
            f"--output-root {output_root}"
        ),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "random_seed": (
            SCREEN_SEEDS[cell][recipe_id] if job_kind == "screen" else FINAL_SEEDS[cell]
        ),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _default_job_root(cell: str, job_kind: str, recipe_id: str) -> Path:
    stage = "screen/candidates" if job_kind == "screen" else "final"
    return base.PHASE_ROOT / cell / "training" / stage / recipe_id / "attempt-01"


def _selection_path(cell: str) -> Path:
    return base.PHASE_ROOT / cell / "training/screen/selection.json"


def _cell(value: str) -> str:
    cell = str(value)
    if cell not in CELLS:
        raise P4TrainingError(f"unsupported training cell: {cell}")
    return cell


def _parameter_count(recipe: Recipe) -> int:
    sizes = (6, *recipe.hidden_layers, 12)
    per_stage = sum(left * right + right for left, right in zip(sizes[:-1], sizes[1:]))
    return 3 * per_stage


def _heldout_values(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(item) for item in row["heldout_common_batches"]["reverse_kl_means"])


def _mean(values: Sequence[float]) -> float:
    return math.fsum(float(item) for item in values) / len(values)


def _mcse(values: Sequence[float]) -> float:
    numeric = tuple(float(item) for item in values)
    if len(numeric) <= 1:
        return 0.0
    mean = _mean(numeric)
    return math.sqrt(
        math.fsum((item - mean) ** 2 for item in numeric)
        / ((len(numeric) - 1) * len(numeric))
    )


def _file_reference(path: Path) -> Mapping[str, Any]:
    return {"path": str(path), "file_sha256": _file_sha256(path), "byte_count": path.stat().st_size}


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise P4TrainingError(f"artifact is not a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_recursive_hashes(root: Path) -> None:
    hashes = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        root / "artifact_hashes.json",
        {"schema": "bayesfilter.multimodel_neutra_p4_training_hashes.v1", "artifacts": hashes},
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("run", "finalize"), required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--job-kind", choices=("screen", "final"))
    parser.add_argument("--recipe", choices=RECIPE_ORDER)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.action == "finalize":
        result = finalize_screen(cell_id=args.cell, output_path=_selection_path(args.cell))
    else:
        if args.job_kind is None or args.recipe is None:
            parser.error("run requires --job-kind and --recipe")
        output_root = args.output_root or _default_job_root(
            args.cell, args.job_kind, args.recipe
        )
        result = run_training_job(
            cell_id=args.cell,
            job_kind=args.job_kind,
            recipe_id=args.recipe,
            output_root=output_root,
        )
    print(
        json.dumps(
            {
                "cell_id": result["cell_id"],
                "passed": result["passed"],
                "decision": result["decision"],
                "recipe_id": result.get("recipe_id", result.get("selected_recipe_id")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
