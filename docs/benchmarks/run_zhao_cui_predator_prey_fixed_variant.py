#!/usr/bin/env python3
"""Bounded predator-prey Zhao-Cui fixed-variant tuning and claim harness.

The default command performs a small offline tuning ladder on three independent
synthetic data splits and writes a repository-issued tuning artifact. Claim
preparation is opt-in and requires that artifact plus the sealed seed-81104
observations. This harness does not modify leaderboard registries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping


if "--cpu-reference" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf

# Configure the shared-GPU allocator before importing any BayesFilter module.
# Several model modules construct TensorFlow constants at import time, which
# can otherwise create a logical GPU before the repository policy is applied.
_INITIAL_GPU_MEMORY_POLICY = None
if "--cpu-reference" not in sys.argv:
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    _INITIAL_GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=True
    )
    tf.config.experimental.enable_tensor_float_32_execution(True)

from bayesfilter.highdim.models import PredatorPreySSM
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    EVENT_ORDER,
    TARGET_ID,
    TARGET_OBSERVATION_SHA256,
    TARGET_STATE_SHA256,
    compile_source_order_ttsirt_proposal_branch,
    prepare_predator_prey_fixed_variant_program,
    prepare_source_order_frozen_apf_program,
)
from bayesfilter.highdim.zhao_cui_predator_prey_proposal_tf import (
    PredatorPreyProposalCandidate,
    PredatorPreyProposalSpec,
    evaluate_predator_prey_proposal_candidate,
    PredatorPreyTuningArtifact,
    fit_predator_prey_proposal_candidate,
    make_tuning_artifact,
    select_l1_candidate,
    select_structural_candidate,
)
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    generate_source_order_predator_prey_dataset_tf,
)


SCHEMA = "bayesfilter.zhao_cui_predator_prey_fixed_variant_campaign.v1"
PROPOSAL_MAX_RESIDUAL_RMS_GATE = 0.75
PROPOSAL_MAX_RESIDUAL_ABS_GATE = 40.0
PLAN_PATH = (
    "docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-active-plan-2026-07-23.md"
)
TUNING_SCHEMA = "bayesfilter.predator_prey_ttsirt_tuning.v1"
CLAIM_SCHEMA = "bayesfilter.zhao_cui_predator_prey_fixed_variant_claim.v1"
CLAIM_PARTICLE_COUNT = 1002
CLAIM_REFERENCE_SEEDS = {
    "initial_reference": (81104, 9101),
    "transition_reference": (81104, 9102),
    "ancestor_uniform": (81104, 9103),
}
CLAIM_FIT_SEEDS = {
    "calibration": 8112101,
    "validation": 8112102,
}
CLAIM_FD_POINTS = (
    ("truth", (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)),
    ("prior_box_midpoint", (0.6, 120.0, 25.0, 0.6, 0.5, 0.5)),
)
CLAIM_FD_STEPS = (2e-6, 2e-4, 2e-5, 2e-6, 2e-6, 2e-6)
CLAIM_GPU_VALUE_ABS_TOLERANCE = 1.0
CLAIM_GPU_SCORE_ABS_TOLERANCE = 2.0
CLAIM_MINIMUM_ESS_FRACTION = 0.1
AUXILIARY_LAW_ID = "frozen_reference_theta_filtering_weight_categorical_v1"
AUXILIARY_TUNING_PARTICLE_COUNT = CLAIM_PARTICLE_COUNT


def _device_policy(cpu_reference: bool) -> Mapping[str, object]:
    if cpu_reference:
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("CPU reference requires CUDA_VISIBLE_DEVICES=-1 before import")
        return {
            "execution_class": "explicit_cpu_reference",
            "trust_basis": "explicit_cpu_reference",
            "physical_gpus": (),
            "logical_gpus": (),
            "online_device": "/CPU:0",
            "tf32_enabled": False,
            "memory_policy": "N/A: GPU hidden before TensorFlow import",
        }
    policy = _INITIAL_GPU_MEMORY_POLICY
    if policy is None:
        raise RuntimeError("GPU memory policy was not configured before imports")
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("claim harness requires a visible GPU")
    return {
        "execution_class": "trusted_visible_gpu",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "physical_gpus": tuple(row["device"] for row in policy["physical_devices"]),
        "logical_gpus": tuple(device.name for device in logical),
        "online_device": "/GPU:0",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "memory_policy": policy,
    }


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_line_count": len(dirty)}


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        data = value.numpy()
        return data.item() if value.shape.rank == 0 else data.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _finalize_run_manifest(
    payload: Mapping[str, object],
    *,
    args: argparse.Namespace,
    started_at: datetime,
    elapsed_seconds: float,
) -> Mapping[str, object]:
    """Add outer-run metadata without discarding claim-specific provenance.

    Claim preparation records a precise fit/branch seed map inside its own
    manifest.  The outer CLI wrapper must augment that record, not replace it
    with the tuning ladder's legacy seed list.
    """

    existing = payload.get("run_manifest")
    manifest = dict(existing) if isinstance(existing, Mapping) else {}
    manifest.setdefault("plan_file", PLAN_PATH)
    manifest.setdefault("command", " ".join(sys.argv))
    manifest.setdefault("environment", sys.executable)
    manifest.setdefault("python_version", platform.python_version())
    manifest.setdefault("tensorflow_version", tf.__version__)
    manifest.setdefault("fit_dtype", "float64")
    manifest.setdefault("online_dtype", "float32")
    manifest.setdefault("jit_compile", True)
    manifest.setdefault(
        "random_seeds",
        [
            81108,
            81109,
            81110,
            8110801,
            8110802,
            8110803,
            8110804,
            8110901,
            8110902,
            8110903,
            8110904,
            8111801,
            8111901,
            8112001,
            8112002,
            8112003,
        ],
    )
    manifest.setdefault("started_at_utc", started_at.isoformat())
    manifest.setdefault("wall_time_seconds", float(elapsed_seconds))
    manifest.setdefault("output_root", str(args.output_root))
    manifest.setdefault(
        "trust_basis",
        (
            "explicit_cpu_reference"
            if args.cpu_reference
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
    )
    return manifest


def _split_observations(seed: int) -> tf.Tensor:
    model = PredatorPreySSM(dtype=tf.float64)
    _states, observations = model.simulate(model.true_parameters(), 20, int(seed))
    return observations[1:]


def _candidate_payload(candidate: PredatorPreyProposalCandidate) -> Mapping[str, object]:
    return {
        "scope_id": candidate.scope_id,
        "data_role": candidate.data_role,
        "observation_hash": candidate.observation_hash,
        "validation_observation_hash": candidate.validation_observation_hash,
        "spec": candidate.spec.payload(),
        "validation_max_residual_rms": max(
            float(row["residual_rms"]) for row in candidate.validation_diagnostics
        ),
        "validation_mean_residual_rms": sum(
            float(row["residual_rms"]) for row in candidate.validation_diagnostics
        )
        / len(candidate.validation_diagnostics),
        "all_validation_finite": all(
            bool(row["finite"]) for row in candidate.validation_diagnostics
        ),
        "fit_manifest": candidate.fit_manifest,
    }


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_tuning_artifact(path: Path) -> tuple[PredatorPreyTuningArtifact, Mapping[str, object]]:
    """Load and reissue one immutable tuning identity from its JSON manifest."""

    tuning_path = path.resolve()
    try:
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read tuning artifact: {tuning_path}") from error
    root = _require_mapping(payload, "tuning artifact")
    if root.get("schema") != SCHEMA:
        raise ValueError("tuning artifact campaign schema mismatch")
    if root.get("status") != "PASS_TUNING_ARTIFACT_ISSUED":
        raise ValueError("tuning artifact did not pass the proposal-quality gate")

    target = _require_mapping(root.get("target"), "tuning target")
    expected_target = {
        "target_id": TARGET_ID,
        "sealed_state_sha256": TARGET_STATE_SHA256,
        "sealed_observation_sha256": TARGET_OBSERVATION_SHA256,
    }
    target_mismatches = [
        name for name, expected in expected_target.items() if target.get(name) != expected
    ]
    if target_mismatches:
        raise ValueError(
            "tuning artifact target mismatch: " + ", ".join(target_mismatches)
        )
    quality = _require_mapping(root.get("proposal_quality_gate"), "proposal quality gate")
    if quality.get("pass") is not True:
        raise ValueError("tuning artifact proposal-quality veto failed")
    auxiliary = _require_mapping(
        root.get("auxiliary_law_selection"), "auxiliary-law selection"
    )
    if auxiliary.get("pass") is not True:
        raise ValueError("tuning artifact auxiliary-law ESS veto failed")
    if auxiliary.get("selected_auxiliary_law") != AUXILIARY_LAW_ID:
        raise ValueError("tuning artifact selected an unsupported auxiliary law")
    if auxiliary.get("reference_theta_source") != "predator_prey_true_parameters":
        raise ValueError("tuning artifact reference-theta selection mismatch")
    if float(auxiliary.get("selected_reference_minimum_ess_fraction", -1.0)) < float(
        auxiliary.get("minimum_ess_fraction_threshold", CLAIM_MINIMUM_ESS_FRACTION)
    ):
        raise ValueError("tuning artifact reference-point ESS gate failed")

    manifest = _require_mapping(root.get("tuning_artifact"), "tuning manifest")
    if manifest.get("schema") != TUNING_SCHEMA:
        raise ValueError("tuning manifest schema mismatch")
    if manifest.get("sealed_claim_observation_hash") != TARGET_OBSERVATION_SHA256:
        raise ValueError("tuning manifest sealed observation mismatch")
    if manifest.get("audit_used_for_selection") is not False:
        raise ValueError("tuning manifest used audit data for selection")
    if [float(value) for value in manifest.get("reference_theta", ())] != [
        0.6,
        114.0,
        25.0,
        0.3,
        0.5,
        0.5,
    ]:
        raise ValueError("tuning manifest reference theta is not the selected target hypothesis")
    selected = _require_mapping(root.get("selected_candidate"), "selected candidate")
    if selected.get("scope_id") != manifest.get("validation_candidate_scope_id"):
        raise ValueError("selected candidate does not match tuning manifest")
    if selected.get("spec") != manifest.get("selected_spec"):
        raise ValueError("selected candidate controls do not match tuning manifest")
    selected_fit = _require_mapping(selected.get("fit_manifest"), "selected fit manifest")
    recorded_dependencies = _require_mapping(
        selected_fit.get("source_dependency_sha256"), "source dependency closure"
    )
    current_dependencies = {
        "model": hashlib.sha256((ROOT / "bayesfilter/highdim/models.py").read_bytes()).hexdigest(),
        "proposal": hashlib.sha256(
            (ROOT / "bayesfilter/highdim/zhao_cui_predator_prey_proposal_tf.py").read_bytes()
        ).hexdigest(),
        "training": hashlib.sha256(
            (ROOT / "bayesfilter/highdim/stochastic_density_training.py").read_bytes()
        ).hexdigest(),
        "transport": hashlib.sha256(
            (ROOT / "bayesfilter/highdim/transport.py").read_bytes()
        ).hexdigest(),
    }
    if dict(recorded_dependencies) != current_dependencies:
        raise ValueError("tuning artifact source dependency closure is stale")

    try:
        spec = PredatorPreyProposalSpec(
            **dict(_require_mapping(manifest.get("selected_spec"), "selected spec"))
        )
        artifact = PredatorPreyTuningArtifact(
            selected_spec=spec,
            reference_theta=tf.constant(manifest["reference_theta"], tf.float64),
            calibration_observation_hash=str(manifest["calibration_observation_hash"]),
            validation_observation_hash=str(manifest["validation_observation_hash"]),
            audit_observation_hash=str(manifest["audit_observation_hash"]),
            calibration_candidate_scope_id=str(
                manifest["calibration_candidate_scope_id"]
            ),
            validation_candidate_scope_id=str(
                manifest["validation_candidate_scope_id"]
            ),
            audit_candidate_scope_id=str(manifest["audit_candidate_scope_id"]),
            audit_id=str(manifest["audit_id"]),
            audit_design_order=int(manifest["audit_design_order"]),
            audit_design_seed=int(manifest["audit_design_seed"]),
            audit_diagnostics=tuple(manifest["audit_diagnostics"]),
            selection_diagnostics=dict(
                _require_mapping(
                    manifest.get("selection_diagnostics"), "selection diagnostics"
                )
            ),
            tuning_scope_id=str(manifest["tuning_scope_id"]),
            artifact_id=str(manifest["artifact_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tuning manifest failed repository identity validation") from error
    if artifact.manifest_payload()["artifact_id"] != manifest.get("artifact_id"):
        raise ValueError("tuning manifest identity mismatch")
    if not artifact.proposal_quality_pass:
        raise ValueError("tuning artifact failed the proposal-quality promotion veto")
    return artifact, root


def _claim_randomness(particle_count: int) -> Mapping[str, tf.Tensor]:
    count = int(particle_count)
    if count <= 1000:
        raise ValueError("claim particle count must be greater than 1000")
    initial = tf.random.stateless_uniform(
        [2, count], CLAIM_REFERENCE_SEEDS["initial_reference"], dtype=tf.float64
    )
    transition = tf.random.stateless_uniform(
        [20, 2, count],
        CLAIM_REFERENCE_SEEDS["transition_reference"],
        dtype=tf.float64,
    )
    ancestor = tf.random.stateless_uniform(
        [20, count], CLAIM_REFERENCE_SEEDS["ancestor_uniform"], dtype=tf.float64
    )
    auxiliary_log = tf.fill(
        [20, count], -tf.math.log(tf.cast(count, tf.float64))
    )
    return {
        "initial_reference_points": initial,
        "transition_reference_points": transition,
        "ancestor_uniforms": ancestor,
        "uniform_auxiliary_log_probabilities": auxiliary_log,
    }


def _make_reference_auxiliary_law(
    *,
    candidate: PredatorPreyProposalCandidate,
    observations: tf.Tensor,
    randomness: Mapping[str, tf.Tensor],
    reference_theta: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Freeze reference-point filtering weights as auxiliary categorical laws.

    Zhao-Cui's author route uses weighted sequential correction and adaptive
    reapproximation (`full_sol.m:21-42,46-124`).  This deterministic reference-
    point freeze is a fixed-HMC adaptation of that weighting operation.  It is
    deliberately compiled offline and is not a source-faithful claim about the
    adaptive MATLAB algorithm.
    """

    model = PredatorPreySSM(dtype=tf.float64)
    theta = tf.cast(reference_theta, tf.float64)
    initial_transport = candidate.transports[0]
    initial_map = candidate.previous_maps[0]
    initial_local = initial_transport.inverse_transport(
        randomness["initial_reference_points"]
    )
    initial_physical, initial_log_det = initial_map.forward(
        tf.transpose(initial_local)
    )
    initial_log_q = (
        tf.math.log(initial_transport.eval_pdf(initial_local)) - initial_log_det
    )
    initial_log_unnormalized = model.initial_log_density(theta, initial_physical) - initial_log_q
    previous_log_weights = initial_log_unnormalized - tf.reduce_logsumexp(
        initial_log_unnormalized
    )
    rows = []
    states = [initial_physical]
    for time_index, transport in enumerate(candidate.transports[1:]):
        # The previous normalized weights are the frozen auxiliary law for the
        # next categorical ancestor draw.
        rows.append(previous_log_weights)
        cdf = tf.concat(
            [tf.math.cumsum(tf.exp(previous_log_weights))[:-1], tf.ones([1], tf.float64)],
            axis=0,
        )
        ancestor = tf.searchsorted(
            cdf,
            randomness["ancestor_uniforms"][time_index],
            side="right",
            out_type=tf.int32,
        )
        parent_physical = tf.gather(states[-1], ancestor)
        previous_local, _ = candidate.previous_maps[time_index].inverse(parent_physical)
        current_local = transport.conditional_inverse_transport(
            tf.transpose(previous_local),
            randomness["transition_reference_points"][time_index],
        )
        current_physical, current_log_det = candidate.current_maps[time_index].forward(
            tf.transpose(current_local)
        )
        transition_log_q = transport.conditional_proposal_log_density(
            conditioning_points=tf.transpose(previous_local),
            generated_points=current_local,
        ) - current_log_det
        parent_log_weights = tf.gather(previous_log_weights, ancestor)
        target_log_density = (
            model.transition_log_density(
                theta, parent_physical, current_physical, time_index + 1
            )
            + model.observation_log_density(
                theta,
                current_physical,
                observations[time_index],
                time_index + 1,
            )
        )
        current_log_unnormalized = (
            parent_log_weights
            + target_log_density
            - tf.gather(previous_log_weights, ancestor)
            - transition_log_q
        )
        previous_log_weights = current_log_unnormalized - tf.reduce_logsumexp(
            current_log_unnormalized
        )
        states.append(current_physical)
    auxiliary = tf.stack(rows)
    normalization_error = tf.reduce_max(tf.abs(tf.reduce_logsumexp(auxiliary, axis=1)))
    if float(normalization_error.numpy()) > 1.0e-10:
        raise ValueError("reference auxiliary categorical laws are not normalized")
    if not bool(tf.reduce_all(tf.math.is_finite(auxiliary)).numpy()):
        raise ValueError("reference auxiliary categorical laws are non-finite")
    return {
        "auxiliary_log_probabilities": auxiliary,
        "reference_states": tf.stack(states),
    }


def _reference_auxiliary_selection_diagnostic(
    *,
    candidate: PredatorPreyProposalCandidate,
    observations: tf.Tensor,
    evaluation_thetas: Mapping[str, tf.Tensor],
    particle_count: int = AUXILIARY_TUNING_PARTICLE_COUNT,
) -> Mapping[str, object]:
    """Score reference-theta/auxiliary hypotheses on nonsealed validation data."""

    randomness = dict(_claim_randomness(int(particle_count)))
    reference_auxiliary = _make_reference_auxiliary_law(
        candidate=candidate,
        observations=observations,
        randomness=randomness,
        reference_theta=candidate.reference_theta,
    )
    laws = {
        "uniform": tf.fill(
            [20, int(particle_count)],
            -tf.math.log(tf.cast(particle_count, tf.float64)),
        ),
        AUXILIARY_LAW_ID: reference_auxiliary["auxiliary_log_probabilities"],
    }
    rows = {}
    model = PredatorPreySSM(dtype=tf.float64)
    for law_name, auxiliary in laws.items():
        compilation = compile_source_order_ttsirt_proposal_branch(
            observations=tf.cast(observations, tf.float64),
            initial_transport=candidate.transports[0],
            transition_transports=candidate.transports[1:],
            previous_coordinate_maps=candidate.previous_maps,
            current_coordinate_maps=candidate.current_maps,
            initial_reference_points=randomness["initial_reference_points"],
            ancestor_uniforms=randomness["ancestor_uniforms"],
            auxiliary_log_probabilities=auxiliary,
            transition_reference_points=randomness["transition_reference_points"],
            target_id="zhao_cui_predator_prey_validation_auxiliary_selection_v1",
            event_order=EVENT_ORDER,
            target_seed=81109,
            target_state_sha256="0" * 64,
            target_observation_sha256="1" * 64,
            tuning_artifact_id="0" * 64,
            online_dtype=tf.float64,
        )
        program = prepare_source_order_frozen_apf_program(model, compilation.branch)
        point_rows = {}
        for point_name, theta in evaluation_thetas.items():
            result = program.evaluate(tf.cast(theta, tf.float64))
            point_rows[point_name] = {
                "minimum_ess": float(result["minimum_ess"].numpy()),
                "minimum_ess_fraction": float(
                    result["minimum_ess"].numpy() / float(particle_count)
                ),
                "finite": bool(result["finite"].numpy()),
                "value": float(result["log_likelihood"].numpy()),
            }
        rows[law_name] = {
            "points": point_rows,
            "minimum_ess": min(
                row["minimum_ess"] for row in point_rows.values()
            ),
            "minimum_ess_fraction": min(
                row["minimum_ess_fraction"] for row in point_rows.values()
            ),
            "all_finite": all(row["finite"] for row in point_rows.values()),
            "branch_id": compilation.branch.branch_id,
        }
    selected = rows[AUXILIARY_LAW_ID]
    selected_reference = selected["points"].get("reference_theta")
    if selected_reference is None:
        raise ValueError("auxiliary selection requires a reference_theta diagnostic")
    return {
        "candidate_scope_id": candidate.scope_id,
        "reference_theta": [float(value) for value in candidate.reference_theta.numpy()],
        "laws": rows,
        "selected_auxiliary_law": AUXILIARY_LAW_ID,
        "selected_reference_minimum_ess_fraction": selected_reference[
            "minimum_ess_fraction"
        ],
        "selected_all_points_minimum_ess_fraction": selected[
            "minimum_ess_fraction"
        ],
        "selection_role": "validation_only_reference_and_auxiliary_hypothesis",
    }


def _fixed_branch_hashes(
    randomness: Mapping[str, tf.Tensor], branch: object
) -> Mapping[str, str]:
    return {
        "initial_reference_points": _tensor_hash(
            randomness["initial_reference_points"]
        ),
        "transition_reference_points": _tensor_hash(
            randomness["transition_reference_points"]
        ),
        "ancestor_uniforms": _tensor_hash(randomness["ancestor_uniforms"]),
        "auxiliary_log_probabilities": _tensor_hash(
            randomness["auxiliary_log_probabilities"]
        ),
        "reference_auxiliary_states": _tensor_hash(
            randomness["reference_auxiliary_states"]
        ),
        "states": _tensor_hash(branch.states),
        "ancestors": _tensor_hash(branch.ancestors),
        "initial_log_proposal_density": _tensor_hash(
            branch.initial_log_proposal_density
        ),
        "transition_log_proposal_density": _tensor_hash(
            branch.transition_log_proposal_density
        ),
    }


def _claim_fd_diagnostic(program: object) -> Mapping[str, object]:
    """Compare the analytical score with central FD of the same scalar only."""

    evaluator = program.compiled(jit_compile=False)
    rows = []
    for point_name, point_values in CLAIM_FD_POINTS:
        theta = tf.constant(point_values, tf.float64)
        base = evaluator(tf.cast(theta, program.branch.dtype))
        finite_difference = []
        for parameter_index, step_value in enumerate(CLAIM_FD_STEPS):
            direction = tf.one_hot(
                parameter_index, int(theta.shape[0]), dtype=theta.dtype
            )
            plus = evaluator(
                tf.cast(
                    theta + tf.cast(step_value, theta.dtype) * direction,
                    program.branch.dtype,
                )
            )["log_likelihood"]
            minus = evaluator(
                tf.cast(
                    theta - tf.cast(step_value, theta.dtype) * direction,
                    program.branch.dtype,
                )
            )["log_likelihood"]
            finite_difference.append(
                (tf.cast(plus, theta.dtype) - tf.cast(minus, theta.dtype))
                / tf.cast(2.0 * step_value, theta.dtype)
            )
        fd_score = tf.stack(finite_difference)
        analytical = tf.cast(base["score"], theta.dtype)
        absolute = tf.abs(analytical - fd_score)
        relative = absolute / tf.maximum(
            tf.abs(fd_score), tf.constant(1.0e-8, theta.dtype)
        )
        rows.append(
            {
                "point": point_name,
                "theta": list(point_values),
                "score": [float(value) for value in analytical.numpy()],
                "finite_difference_score": [float(value) for value in fd_score.numpy()],
                "absolute_error": [float(value) for value in absolute.numpy()],
                "relative_error": [float(value) for value in relative.numpy()],
                "maximum_absolute_error": float(tf.reduce_max(absolute).numpy()),
                "maximum_relative_error": float(tf.reduce_max(relative).numpy()),
                "pass": bool(
                    tf.reduce_all(
                        absolute <= tf.constant(3.0e-2, theta.dtype)
                    ).numpy()
                ),
            }
        )
    return {
        "points": rows,
        "steps": list(CLAIM_FD_STEPS),
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
        "maximum_relative_error": max(row["maximum_relative_error"] for row in rows),
        "pass": all(row["pass"] for row in rows),
        "role": "same_program_diagnostic_only",
        "runtime_autodiff": False,
        "runtime_finite_difference": False,
    }


def _run_program(
    program: object,
    theta: tf.Tensor,
    *,
    jit_compile: bool,
    device: str,
) -> tuple[Mapping[str, tf.Tensor], float]:
    evaluator = program.compiled(jit_compile=jit_compile)
    started = time.perf_counter()
    with tf.device(device):
        result = evaluator(tf.cast(theta, program.branch.dtype))
        # Force completion before measuring elapsed time on an asynchronous GPU.
        for value in result.values():
            if isinstance(value, tf.Tensor):
                _ = value.numpy()
    return result, time.perf_counter() - started


def _result_payload(result: Mapping[str, tf.Tensor]) -> Mapping[str, object]:
    return {
        "value": float(result["log_likelihood"].numpy()),
        "score": [float(value) for value in result["score"].numpy()],
        "log_increments": [float(value) for value in result["log_increments"].numpy()],
        "increment_scores": [
            [float(value) for value in row]
            for row in result["increment_scores"].numpy()
        ],
        "ess_by_time": [float(value) for value in result["ess_by_time"].numpy()],
        "log_weight_spread_by_time": [
            float(value) for value in result["log_weight_spread_by_time"].numpy()
        ],
        "minimum_ess": float(result["minimum_ess"].numpy()),
        "maximum_log_weight_spread": float(
            result["maximum_log_weight_spread"].numpy()
        ),
        "finite": bool(result["finite"].numpy()),
        "particle_count": int(result["particle_count"].numpy()),
        "transition_count": int(result["transition_count"].numpy()),
        "increment_sum_residual": float(
            tf.abs(
                result["log_likelihood"]
                - tf.reduce_sum(result["log_increments"])
            ).numpy()
        ),
        "score_sum_residual": float(
            tf.reduce_max(
                tf.abs(
                    result["score"]
                    - tf.reduce_sum(result["increment_scores"], axis=0)
                )
            ).numpy()
        ),
    }


def _comparators(
    *,
    theta: tf.Tensor,
    observations: tf.Tensor,
    particle_count: int,
    gpu: bool,
) -> Mapping[str, object]:
    """Return same-target SGQF and descriptive positive-GenUT rows."""

    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        make_predator_prey_source_sgqf_route,
    )

    route = make_predator_prey_source_sgqf_route()
    sgqf_started = time.perf_counter()
    with tf.device("/CPU:0"):
        sgqf_value, sgqf_score, sgqf_status = route.physical_value_score_status(
            theta[tf.newaxis, :]
        )
        _ = sgqf_value.numpy(), sgqf_score.numpy()
    sgqf_ok = bool(
        tf.reduce_all(tf.math.is_finite(sgqf_value)).numpy()
        and tf.reduce_all(tf.math.is_finite(sgqf_score)).numpy()
        and tf.reduce_all(tf.equal(sgqf_status["status_code"], 0)).numpy()
    )
    rows: dict[str, object] = {
        "sgqf": {
            "method": "fixed_sgqf",
            "status": "executed" if sgqf_ok else "vetoed",
            "value": float(sgqf_value[0].numpy()) if sgqf_ok else None,
            "score": [float(value) for value in sgqf_score[0].numpy()] if sgqf_ok else None,
            "route_id": route.manifest["route_id"],
            "dtype": "float64",
            "device": "/CPU:0",
            "time_order": route.manifest["time_order"],
            "elapsed_seconds": time.perf_counter() - sgqf_started,
            "role": "descriptive_same_target_comparator",
        }
    }

    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    from bayesfilter.highdim.cubature_genut_adapters import (
        predator_prey_candidate_adapter,
    )
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    design = replicate_positive_genut(
        gaussian_genut_design(dim=2, dtype=tf.float32),
        num_particles=int(particle_count),
    )
    initial_noise = tf.random.stateless_normal(
        [int(particle_count), 2], [81104, 9201], dtype=tf.float32
    )
    process_noise = tf.random.stateless_normal(
        [20, int(particle_count), 2], [81104, 9202], dtype=tf.float32
    )
    adapter = predator_prey_candidate_adapter()

    @tf.function(jit_compile=bool(gpu), autograph=False)
    def genut_evaluate(
        theta_value: tf.Tensor,
        observations_value: tf.Tensor,
        initial_noise_value: tf.Tensor,
        process_noise_value: tf.Tensor,
        design_value: tf.Tensor,
    ):
        return finite_value_score(
            adapter,
            theta_value,
            observations_value,
            initial_noise_value,
            process_noise_value,
            design_value,
            transition_before_first_observation=True,
        )

    genut_started = time.perf_counter()
    with tf.device("/GPU:0" if gpu else "/CPU:0"):
        genut_value, genut_score, genut_diag = genut_evaluate(
            tf.cast(theta, tf.float32),
            tf.cast(observations, tf.float32),
            initial_noise,
            process_noise,
            design,
        )
        _ = genut_value.numpy(), genut_score.numpy()
    genut_ok = bool(
        tf.math.is_finite(genut_value).numpy()
        and tf.reduce_all(tf.math.is_finite(genut_score)).numpy()
        and bool(genut_diag["program_valid"].numpy())
    )
    rows["genut"] = {
        "method": "gaussian_genut",
        "status": "executed" if genut_ok else "vetoed",
        "value": float(genut_value.numpy()) if genut_ok else None,
        "score": [float(value) for value in genut_score.numpy()] if genut_ok else None,
        "route_id": "cubature_genut_nonfused_positive_ot_row_quotient_candidate_v2",
        "dtype": "float32",
        "device": "/GPU:0" if gpu else "/CPU:0",
        "tf32": bool(gpu),
        "jit_compile": bool(gpu),
        "particle_count": int(particle_count),
        "elapsed_seconds": time.perf_counter() - genut_started,
        "program_valid": bool(genut_diag["program_valid"].numpy()),
        "maximum_mean_residual": float(genut_diag["max_mean_residual"].numpy()),
        "maximum_row_residual": float(genut_diag["max_row_residual"].numpy()),
        "maximum_column_residual": float(genut_diag["max_col_residual"].numpy()),
        "role": "descriptive_same_target_comparator",
    }
    return rows


def _tune(
    *,
    output_root: Path,
    debug_smoke: bool,
    device: Mapping[str, object],
) -> Mapping[str, object]:
    calibration_data_seed = 81108
    validation_data_seed = 81109
    audit_data_seed = 81110
    calibration_order = 2 if debug_smoke else 5
    validation_order = 3 if debug_smoke else 6
    audit_order = 4 if debug_smoke else 7
    calibration = _split_observations(calibration_data_seed)
    validation = _split_observations(validation_data_seed)
    audit = _split_observations(audit_data_seed)
    # The model's declared truth is a predeclared proposal-reference
    # hypothesis. It is not inferred from, or selected using, sealed claim
    # observations.
    reference_theta = PredatorPreySSM(dtype=tf.float64).true_parameters()
    hashes = {
        "calibration": _tensor_hash(calibration),
        "validation": _tensor_hash(validation),
        "audit": _tensor_hash(audit),
    }
    if len(set(hashes.values())) != 3:
        raise RuntimeError("tuning split hashes are not disjoint")

    structural_specs = (
        PredatorPreyProposalSpec(
            degree=0,
            rank=1,
            coordinate_scale=8.0,
            defensive_tau=1e-3,
            l1_weight=0.0,
            ridge=1e-6,
            prefit_steps=0,
            train_steps=0,
            cdf_grid_size=17,
            cdf_bisection_steps=8,
        ),
        PredatorPreyProposalSpec(
            degree=2,
            rank=2,
            coordinate_map_family="gaussian_quantile",
            coordinate_scale=1.0,
            defensive_tau=1e-6,
            l1_weight=0.0,
            ridge=1e-8,
            prefit_steps=2 if debug_smoke else 8,
            train_steps=2 if debug_smoke else 8,
            cdf_grid_size=33,
            cdf_bisection_steps=12,
        ),
        # The original handoff ladder omitted the first capacity arm that
        # resolves the narrow predator-prey transition ridge.  This is a
        # target-specific hypothesis, not a transferred default.
        PredatorPreyProposalSpec(
            degree=4,
            rank=4,
            coordinate_map_family="gaussian_quantile",
            coordinate_scale=2.0,
            defensive_tau=1e-6,
            l1_weight=0.0,
            ridge=1e-8,
            prefit_steps=2 if debug_smoke else 8,
            train_steps=2 if debug_smoke else 8,
            cdf_grid_size=33,
            cdf_bisection_steps=12,
        ),
        PredatorPreyProposalSpec(
            degree=4,
            rank=8,
            coordinate_map_family="gaussian_quantile",
            coordinate_scale=2.0,
            defensive_tau=1e-6,
            l1_weight=0.0,
            ridge=1e-8,
            prefit_steps=2 if debug_smoke else 8,
            train_steps=2 if debug_smoke else 8,
            cdf_grid_size=33,
            cdf_bisection_steps=12,
        ),
    )
    structural_candidates = []
    structural_failures = []
    for index, spec in enumerate(structural_specs):
        try:
            candidate = fit_predator_prey_proposal_candidate(
                observations=calibration,
                spec=spec,
                calibration_order=calibration_order,
                validation_order=validation_order,
                calibration_seed=8110801 + index,
                validation_seed=8110901 + index,
                reference_theta=reference_theta,
                data_role="calibration",
            )
            structural_candidates.append(candidate)
        except Exception as error:  # preserve candidate failure evidence
            structural_failures.append(
                {
                    "candidate_index": index,
                    "spec": spec.payload(),
                    "failure_classification": "candidate_or_fit_failure",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            )
    if not structural_candidates:
        raise RuntimeError("all bounded structural candidates failed")
    structural_selected, structural_selection = select_structural_candidate(
        structural_candidates
    )
    base = structural_selected.spec
    l1_specs = tuple(
        PredatorPreyProposalSpec(
            degree=base.degree,
            rank=base.rank,
            coordinate_map_family=base.coordinate_map_family,
            coordinate_scale=base.coordinate_scale,
            defensive_tau=base.defensive_tau,
            l1_weight=l1,
            ridge=base.ridge,
            prefit_steps=base.prefit_steps,
            train_steps=base.train_steps,
            cdf_grid_size=base.cdf_grid_size,
            cdf_bisection_steps=base.cdf_bisection_steps,
        )
        for l1 in (0.0, 1e-6)
    )
    candidates = []
    failures = []
    for index, spec in enumerate(l1_specs):
        try:
            candidate = fit_predator_prey_proposal_candidate(
                observations=validation,
                spec=spec,
                calibration_order=calibration_order,
                validation_order=validation_order,
                calibration_seed=8111801,
                validation_seed=8111901,
                reference_theta=reference_theta,
                data_role="validation",
            )
            candidates.append(candidate)
        except Exception as error:
            failures.append(
                {
                    "candidate_index": index,
                    "spec": spec.payload(),
                    "failure_classification": "l1_candidate_or_fit_failure",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            )
    if not candidates:
        raise RuntimeError("all bounded L1 candidates failed")

    selected, l1_selection = select_l1_candidate(candidates)
    # Fit the audit-specific proposal only after structural and L1 controls
    # are frozen.  The source scope binding proves that audit data cannot
    # select those controls; the distinct quadrature order remains a holdout
    # for the audit candidate's fitted cores.
    audit_candidate = fit_predator_prey_proposal_candidate(
        observations=audit,
        spec=selected.spec,
        calibration_order=calibration_order,
        validation_order=validation_order,
        calibration_seed=8112001,
        validation_seed=8112002,
        reference_theta=reference_theta,
        data_role="audit",
        frozen_control_source_scope_id=selected.scope_id,
    )
    audit_result = evaluate_predator_prey_proposal_candidate(
        audit_candidate,
        observations=audit,
        # Use a quadrature order not used by either the audit fit or its
        # validation diagnostic.  This is a genuine design holdout.
        design_order=audit_order,
        design_seed=8112003,
    )
    auxiliary_selection = _reference_auxiliary_selection_diagnostic(
        candidate=selected,
        observations=validation,
        evaluation_thetas={
            "reference_theta": selected.reference_theta,
            "interior_alternative": tf.constant(
                [0.55, 116.0, 24.0, 0.45, 0.4, 0.6], tf.float64
            ),
        },
        particle_count=AUXILIARY_TUNING_PARTICLE_COUNT,
    )
    auxiliary_audit = _reference_auxiliary_selection_diagnostic(
        candidate=audit_candidate,
        observations=audit,
        evaluation_thetas={"reference_theta": selected.reference_theta},
        particle_count=AUXILIARY_TUNING_PARTICLE_COUNT,
    )
    # The selected target truth is a predeclared model hypothesis, not sealed
    # observation data. It is used only to choose a proposal reference point;
    # the untouched claim observations remain unavailable until claim time.
    target_truth = reference_theta
    validation_reference_row = auxiliary_selection["laws"][AUXILIARY_LAW_ID][
        "points"
    ]["reference_theta"]
    audit_reference_row = auxiliary_audit["laws"][AUXILIARY_LAW_ID]["points"][
        "reference_theta"
    ]
    auxiliary_quality_pass = bool(
        validation_reference_row["minimum_ess_fraction"]
        >= CLAIM_MINIMUM_ESS_FRACTION
        and audit_reference_row["minimum_ess_fraction"]
        >= CLAIM_MINIMUM_ESS_FRACTION
        and validation_reference_row["finite"]
        and audit_reference_row["finite"]
    )
    tuning = make_tuning_artifact(
        calibration_candidate=structural_selected,
        selected_candidate=selected,
        audit_candidate=audit_candidate,
        calibration_observation_hash=hashes["calibration"],
        validation_observation_hash=hashes["validation"],
        audit=audit_result,
        selection_diagnostics={
            "structural_selection": structural_selection,
            "l1_selection": l1_selection,
            "candidate_count": len(candidates),
            "structural_candidate_count": len(structural_candidates),
            "audit_evaluated_for_selection": False,
            "auxiliary_selection": auxiliary_selection,
            "protocol": {
                "calibration_order": calibration_order,
                "validation_order": validation_order,
                "audit_order": audit_order,
                "calibration_data_seed": calibration_data_seed,
                "validation_data_seed": validation_data_seed,
                "audit_data_seed": audit_data_seed,
                "calibration_fit_seed_base": 8110801,
                "calibration_validation_seed_base": 8110901,
                "l1_fit_seed": 8111801,
                "l1_validation_seed": 8111901,
                "audit_design_seed": 8112003,
                "fit_semantics": "controls_frozen_then_observation_specific_audit_fit_and_quadrature_holdout",
                "calibration_path": "calibration_candidate_fit_and_quadrature_holdout",
                "selection_path": "validation_candidate_fit_and_quadrature_holdout",
                "audit_path": "audit_candidate_fit_with_validation_controls_then_frozen_candidate_evaluation",
            },
        },
    )
    audit_max_rms = max(
        float(row["residual_rms"]) for row in audit_result.diagnostics
    )
    audit_max_abs = max(
        float(row["residual_abs_max"]) for row in audit_result.diagnostics
    )
    proposal_quality_pass = (
        audit_max_rms <= PROPOSAL_MAX_RESIDUAL_RMS_GATE
        and audit_max_abs <= PROPOSAL_MAX_RESIDUAL_ABS_GATE
    )
    proposal_quality_pass = proposal_quality_pass and auxiliary_quality_pass
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": (
            "PASS_TUNING_ARTIFACT_ISSUED"
            if proposal_quality_pass
            else "REJECT_PROPOSAL_QUALITY_REPAIR_REQUIRED"
        ),
        "execution_role": "debug_smoke" if debug_smoke else "offline_tuning",
        "device": device,
        "target": {
            "target_id": TARGET_ID,
            "sealed_state_sha256": TARGET_STATE_SHA256,
            "sealed_observation_sha256": TARGET_OBSERVATION_SHA256,
            "tuning_split_hashes": hashes,
            "reference_theta": [float(value) for value in reference_theta.numpy()],
        },
        "candidates": [_candidate_payload(item) for item in candidates],
        "structural_candidates": [_candidate_payload(item) for item in structural_candidates],
        "structural_candidate_failures": structural_failures,
        "candidate_failures": failures,
        "audit": audit_result.manifest_payload(),
        "audit_candidate": _candidate_payload(audit_candidate),
        "proposal_quality_gate": {
            "max_residual_rms_threshold": PROPOSAL_MAX_RESIDUAL_RMS_GATE,
            "observed_max_residual_rms": audit_max_rms,
            "max_residual_abs_threshold": PROPOSAL_MAX_RESIDUAL_ABS_GATE,
            "observed_max_residual_abs": audit_max_abs,
            "pass": proposal_quality_pass,
            "role": "promotion_veto_for_sealed_claim_preparation",
        },
        "auxiliary_law_selection": {
            **auxiliary_selection,
            "audit_veto": auxiliary_audit,
            "reference_theta_source": "predator_prey_true_parameters",
            "selected_reference_theta": [float(value) for value in target_truth.numpy()],
            "minimum_ess_fraction_threshold": CLAIM_MINIMUM_ESS_FRACTION,
            "off_reference_parameter_diagnostics_explanatory_only": True,
            "pass": auxiliary_quality_pass,
            "role": "promotion_veto_for_sealed_claim_preparation",
        },
        "selected_candidate": _candidate_payload(selected),
        "l1_selection": l1_selection,
        "tuning_artifact": tuning.manifest_payload(),
        "nonclaims": [
            "no exact likelihood claim",
            "no source-faithful assembled-route claim",
            "no posterior or HMC claim",
            "no leaderboard admission",
            "no superiority or default-readiness claim",
        ],
    }
    (output_root / "tuning.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _claim(
    *,
    output_root: Path,
    tuning_artifact: PredatorPreyTuningArtifact,
    tuning_payload: Mapping[str, object],
    device: Mapping[str, object],
) -> Mapping[str, object]:
    if tuning_payload.get("status") != "PASS_TUNING_ARTIFACT_ISSUED":
        raise ValueError("claim requires a passing repository-issued tuning artifact")
    # The frozen branch is an offline artifact.  Build it on CPU so its fitted
    # FP64 cores and stateless draws do not depend on whether a claim process
    # happens to have a visible GPU.  Only the online FP32/XLA evaluation below
    # is allowed to use the shared GPU.
    with tf.device("/CPU:0"):
        sealed_states, sealed_observations = (
            generate_source_order_predator_prey_dataset_tf()
        )
        observations = tf.cast(sealed_observations, tf.float64)
        if _tensor_hash(sealed_states) != TARGET_STATE_SHA256:
            raise ValueError("sealed state hash mismatch before claim preparation")
        if _tensor_hash(observations) != TARGET_OBSERVATION_SHA256:
            raise ValueError("sealed observation hash mismatch before claim preparation")

        selected_payload = _require_mapping(
            tuning_payload.get("selected_candidate"), "selected candidate"
        )
        selected_fit_manifest = _require_mapping(
            selected_payload.get("fit_manifest"), "selected fit manifest"
        )
        calibration_order = int(selected_fit_manifest.get("calibration_order", 5))
        validation_order = int(selected_fit_manifest.get("validation_order", 6))
        claim_candidate = fit_predator_prey_proposal_candidate(
            observations=observations,
            spec=tuning_artifact.selected_spec,
            calibration_order=calibration_order,
            validation_order=validation_order,
            calibration_seed=CLAIM_FIT_SEEDS["calibration"],
            validation_seed=CLAIM_FIT_SEEDS["validation"],
            reference_theta=tuning_artifact.reference_theta,
            data_role="sealed_claim_preparation",
            tuning_artifact=tuning_artifact,
        )
        tuning_artifact.require_claim_candidate(claim_candidate)

        randomness = dict(_claim_randomness(CLAIM_PARTICLE_COUNT))
        reference_auxiliary = _make_reference_auxiliary_law(
            candidate=claim_candidate,
            observations=observations,
            randomness=randomness,
            reference_theta=tuning_artifact.reference_theta,
        )
        randomness["auxiliary_log_probabilities"] = reference_auxiliary[
            "auxiliary_log_probabilities"
        ]
        randomness["reference_auxiliary_states"] = reference_auxiliary[
            "reference_states"
        ]
        # Compile a FP64 branch for the independent reference/FD audit and a
        # separately typed FP32 branch for the required GPU/XLA execution.
        compilation64 = claim_candidate.compile_branch(
            observations=observations,
            initial_reference_points=randomness["initial_reference_points"],
            ancestor_uniforms=randomness["ancestor_uniforms"],
            auxiliary_log_probabilities=randomness["auxiliary_log_probabilities"],
            transition_reference_points=randomness["transition_reference_points"],
            online_dtype=tf.float64,
            tuning_artifact=tuning_artifact,
        )
        compilation32 = claim_candidate.compile_branch(
            observations=observations,
            initial_reference_points=randomness["initial_reference_points"],
            ancestor_uniforms=randomness["ancestor_uniforms"],
            auxiliary_log_probabilities=randomness["auxiliary_log_probabilities"],
            transition_reference_points=randomness["transition_reference_points"],
            online_dtype=tf.float32,
            tuning_artifact=tuning_artifact,
        )
        model64 = PredatorPreySSM(dtype=tf.float64)
        model32 = PredatorPreySSM(dtype=tf.float32)
        program64 = prepare_predator_prey_fixed_variant_program(
            model64, compilation64.branch
        )
        program32 = prepare_predator_prey_fixed_variant_program(
            model32, compilation32.branch
        )

    is_gpu = device.get("execution_class") == "trusted_visible_gpu"
    reference64, reference_elapsed = _run_program(
        program64,
        tf.cast(model64.true_parameters(), tf.float64),
        jit_compile=False,
        device="/CPU:0",
    )
    online32, online_elapsed = _run_program(
        program32,
        tf.cast(model32.true_parameters(), tf.float32),
        jit_compile=True,
        device="/GPU:0" if is_gpu else "/CPU:0",
    )
    reference_payload = _result_payload(reference64)
    online_payload = _result_payload(online32)
    value_delta = float(online_payload["value"] - reference_payload["value"])
    score_delta = [
        float(online - reference)
        for online, reference in zip(online_payload["score"], reference_payload["score"])
    ]
    same_program_fd = _claim_fd_diagnostic(program64)
    # Comparators are explanatory only.  Keep them on CPU so the shared GPU is
    # used only for the claim-bearing Zhao-Cui FP32/XLA evaluator.
    comparator_payload = _comparators(
        theta=tf.cast(model64.true_parameters(), tf.float64),
        observations=observations,
        particle_count=CLAIM_PARTICLE_COUNT,
        gpu=False,
    )
    comparator_payload["zhao_cui"] = {
        "method": "zhao_cui_fixed_variant",
        "status": "executed" if reference_payload["finite"] else "vetoed",
        "value": reference_payload["value"],
        "score": reference_payload["score"],
        "route_id": program64.manifest_payload()["route_id"],
        "route_classification": "extension_or_invention",
        "dtype": "float64_reference",
        "role": "candidate_result_not_source_faithful",
    }
    for name, row in tuple(comparator_payload.items()):
        if not isinstance(row, Mapping) or row.get("value") is None:
            continue
        row = dict(row)
        row["value_difference_vs_zhao_cui"] = float(
            row["value"] - reference_payload["value"]
        )
        if row.get("score") is not None:
            row["score_difference_vs_zhao_cui"] = [
                float(left - right)
                for left, right in zip(row["score"], reference_payload["score"])
            ]
        comparator_payload[name] = row

    allocator = {}
    if is_gpu:
        allocator = {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    branch_hashes = _fixed_branch_hashes(randomness, compilation32.branch)
    memory_terms = {
        "state_clouds_fp32_bytes": (20 + 1)
        * CLAIM_PARTICLE_COUNT
        * 2
        * 4,
        "transition_log_proposal_fp32_bytes": 20 * CLAIM_PARTICLE_COUNT * 4,
        "auxiliary_log_probabilities_fp32_bytes": 20 * CLAIM_PARTICLE_COUNT * 4,
        "ancestor_indices_int32_bytes": 20 * CLAIM_PARTICLE_COUNT * 4,
        "score_working_set_order": "O(N * parameter_dim * bytes)",
        "retained_tensor_product_grid": False,
    }
    proposal_noncollapsed = bool(
        reference_payload["minimum_ess"]
        >= CLAIM_MINIMUM_ESS_FRACTION * CLAIM_PARTICLE_COUNT
    )
    online_tie_out = bool(
        abs(value_delta) <= CLAIM_GPU_VALUE_ABS_TOLERANCE
        and max(abs(value) for value in score_delta)
        <= CLAIM_GPU_SCORE_ABS_TOLERANCE
    )
    claim_pass = bool(
        reference_payload["finite"]
        and online_payload["finite"]
        and same_program_fd["pass"]
        and reference_payload["increment_sum_residual"] <= 1.0e-8
        and reference_payload["score_sum_residual"] <= 1.0e-8
        and proposal_noncollapsed
        and online_tie_out
        and (is_gpu or device.get("execution_class") == "explicit_cpu_reference")
    )
    payload: dict[str, object] = {
        "schema": CLAIM_SCHEMA,
        "status": (
            "PASS_SEALED_CLAIM_IMPLEMENTATION_GATES"
            if claim_pass
            else "VETO_SEALED_CLAIM_IMPLEMENTATION_GATE"
        ),
        "target": {
            "target_id": TARGET_ID,
            "event_order": "x0 -> transition_1..20 -> y1..y20",
            "target_seed": 81104,
            "state_sha256": TARGET_STATE_SHA256,
            "observation_sha256": TARGET_OBSERVATION_SHA256,
            "particle_count": CLAIM_PARTICLE_COUNT,
            "parameter_order": ["r", "K", "a", "s", "u", "v"],
        },
        "tuning": {
            "artifact_id": tuning_artifact.artifact_id,
            "tuning_scope_id": tuning_artifact.tuning_scope_id,
            "selected_spec": tuning_artifact.selected_spec.payload(),
            "reference_theta": [
                float(value) for value in tuning_artifact.reference_theta.numpy()
            ],
            "claim_candidate_scope_id": claim_candidate.scope_id,
            "claim_fit_manifest": claim_candidate.fit_manifest,
            "controls_frozen_before_sealed_fit": True,
            "sealed_observations_used_for_control_selection": False,
        },
        "randomness": {
            "seeds": {name: list(seed) for name, seed in CLAIM_REFERENCE_SEEDS.items()},
            "preparation_device": "/CPU:0",
            "device_invariant_preparation_required": True,
            "auxiliary_law": "frozen_reference_theta_filtering_weight_categorical_v1",
            "auxiliary_law_classification": "fixed_hmc_adaptation_with_extension_assembly",
            "auxiliary_reference_theta": [
                float(value) for value in tuning_artifact.reference_theta.numpy()
            ],
            "auxiliary_source_anchor": "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-42,46-124",
            "hashes": branch_hashes,
        },
        "branch": {
            "fp64": {
                "branch_id": compilation64.branch.branch_id,
                "compiler_id": compilation64.compiler_id,
                "program_id": program64.program_id,
                "manifest": compilation64.manifest,
            },
            "fp32": {
                "branch_id": compilation32.branch.branch_id,
                "compiler_id": compilation32.compiler_id,
                "program_id": program32.program_id,
                "manifest": compilation32.manifest,
            },
            "hashes": branch_hashes,
        },
        "reference_cpu_fp64": {
            **reference_payload,
            "elapsed_seconds": reference_elapsed,
            "jit_compile": False,
            "device": "/CPU:0",
        },
        "online_execution": {
            **online_payload,
            "elapsed_seconds": online_elapsed,
            "jit_compile": True,
            "dtype": "float32",
            "tf32_enabled": bool(device.get("tf32_enabled", False)),
            "device": device.get("online_device"),
            "value_difference_vs_cpu_fp64": value_delta,
            "score_difference_vs_cpu_fp64": score_delta,
            "tie_out": online_tie_out,
            "tie_out_tolerances": {
                "value_absolute": CLAIM_GPU_VALUE_ABS_TOLERANCE,
                "score_maximum_absolute": CLAIM_GPU_SCORE_ABS_TOLERANCE,
            },
        },
        "same_program_score_audit": same_program_fd,
        "proposal_quality": {
            "minimum_ess_fraction_threshold": CLAIM_MINIMUM_ESS_FRACTION,
            "observed_minimum_ess_fraction": reference_payload["minimum_ess"]
            / CLAIM_PARTICLE_COUNT,
            "noncollapsed": proposal_noncollapsed,
            "role": "claim_promotion_veto",
        },
        "comparators": comparator_payload,
        "memory": memory_terms,
        "gpu_allocator": allocator,
        "device": device,
        "source_classification": {
            "assembled_route": "extension_or_invention",
            "squared_tt_defensive_density": "source_faithful_operation_only",
            "paired_core_conditional": "source_faithful_operation_only",
            "frozen_randomness_and_settings": "fixed_hmc_adaptation",
            "frozen_reference_weight_auxiliary_law": "fixed_hmc_adaptation_with_extension_assembly",
            "finite_grid_inverse_and_source_order_apf": "extension_or_invention",
            "paper_anchor": ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539-670,890-924",
            "author_source_anchor": "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-135; deep-tensor.dev/src/SIRT.m:51-85",
        },
        "inference_status": {
            "hard_veto_screen": "finite value/score, identity, increment, same-program FD, ESS, and CPU/online tie-out screens recorded",
            "statistically_supported_ranking": "none; one sealed seed and no uncertainty interval",
            "descriptive_only_differences": "all SGQF/GenUT/Zhao-Cui value and score gaps",
            "default_readiness": "not assessed",
            "next_evidence_needed": "multi-seed or larger-N uncertainty study and reviewed route-admission decision",
        },
        "nonclaims": [
            "not the exact observed-data likelihood",
            "not a source-faithful assembled Zhao-Cui filter",
            "not unbiased pseudo-marginal evidence",
            "not posterior or HMC readiness",
            "not statistical superiority or default readiness",
            "not high-dimensional scalability evidence beyond this predator-prey target",
            "the frozen reference-weight auxiliary law was selected as a repair trigger after the uniform claim arm collapsed",
        ],
        "run_manifest": {
            "plan_file": PLAN_PATH,
            "tuning_artifact_path": str(tuning_payload.get("_artifact_path", "")),
            "offline_preparation_device": "/CPU:0",
            "online_execution_device": device.get("online_device"),
            "descriptive_comparator_device": "/CPU:0",
            "command": " ".join(sys.argv),
            "environment": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "fit_dtype": "float64",
            "online_dtype": "float32",
            "jit_compile": True,
            "random_seeds": {
                "fit": CLAIM_FIT_SEEDS,
                "branch": CLAIM_REFERENCE_SEEDS,
            },
            "git": _git_payload(),
            "output_root": str(output_root),
            "trust_basis": device.get("trust_basis"),
        },
    }
    (output_root / "claim.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "claim.md").write_text(
        "# Zhao-Cui Predator-Prey Fixed-Variant Claim\n\n"
        f"Status: `{payload['status']}`\n\n"
        "The assembled route is classified as `extension_or_invention`; comparator differences are descriptive only.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--debug-smoke", action="store_true")
    parser.add_argument(
        "--claim",
        action="store_true",
        help="run the sealed claim phase using an existing tuning artifact",
    )
    parser.add_argument(
        "--tuning-artifact",
        type=Path,
        help="tuning.json issued by a prior offline tuning run",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    device = _device_policy(args.cpu_reference)
    if args.claim:
        if args.debug_smoke:
            raise ValueError("--debug-smoke cannot be combined with --claim")
        if args.tuning_artifact is None:
            raise ValueError("--claim requires --tuning-artifact")
        tuning_artifact, payload = _load_tuning_artifact(args.tuning_artifact)
        payload = dict(payload)
        payload["_artifact_path"] = str(args.tuning_artifact.resolve())
        payload = _claim(
            output_root=args.output_root,
            tuning_artifact=tuning_artifact,
            tuning_payload=payload,
            device=device,
        )
    else:
        if args.tuning_artifact is not None:
            raise ValueError("--tuning-artifact is only valid with --claim")
        payload = _tune(
            output_root=args.output_root,
            debug_smoke=args.debug_smoke,
            device=device,
        )
    payload = {
        **payload,
        "run_manifest": {
            **_finalize_run_manifest(
                payload,
                args=args,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
            ),
            "git": _git_payload(),
        },
        "decision": {
            "candidate_rejected": (
                payload.get("status") not in {
                    "PASS_TUNING_ARTIFACT_ISSUED",
                    "PASS_SEALED_CLAIM_IMPLEMENTATION_GATES",
                }
            ),
            "research_direction_rejected": False,
            "primary_criterion_status": (
                "sealed_claim_implementation_gates_assessed"
                if args.claim
                else "not_assessed_tuning_only"
            ),
            "next_justified_action": (
                "review claim evidence and decide whether to authorize a larger-N/multi-seed study"
                if args.claim and payload.get("status") == "PASS_SEALED_CLAIM_IMPLEMENTATION_GATES"
                else (
                    "prepare the sealed claim branch"
                    if payload.get("proposal_quality_gate", {}).get("pass")
                    else "repair the proposal ladder on fresh calibration data; do not prepare the sealed claim branch"
                )
            ),
            "not_concluded": "no likelihood, posterior, HMC, leaderboard, superiority, or default-readiness claim",
        },
    }
    (args.output_root / "result.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "result.md").write_text(
        "# Zhao-Cui Predator-Prey Fixed-Variant Run\n\n"
        f"Status: `{payload['status']}`\n\n"
        + (
            "This artifact records sealed claim implementation evidence; leaderboard admission is not executed.\n"
            if args.claim
            else "This artifact records offline scope-specific tuning only. The sealed claim run and leaderboard admission are not executed.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), sort_keys=True))


if __name__ == "__main__":
    main()
