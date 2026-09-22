"""Run the bounded Phase 7 integrated exact-DMIS diagnostic.

The full C2 observation path is evaluated with the repository's exact frozen
APF value/analytical-score program.  Observation-conditioned UKF/APF families
use their complete conditional proposal densities.  At selected times, exact
recursive particle weights build a lagged moment map and the audited
Hermite-plus-RBF fitter is evaluated as a guide/control only.  No hybrid TT
sampler exists in this route, so the fitted representation never enters the
proposal denominator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf


DTYPE = tf.float64
STATE_DIM = 4
PARTICLE_COUNT = 8192
BRANCH_COUNT = 3
HORIZON = 20
MAP_TIMES = (1, 3, 10, 19)
TRAIN_ROWS = 512
HOLDOUT_ROWS = 512
AUDIT_ROWS = 4096
TARGET_PARENT_CHUNK = 256
ARM_WIDTHS = (0.75, 1.50)
MIXTURE_OFFSET = 0.50
SELECTED_DEFENSIVE_CONFIG = {
    "config_id": "nu5_eps05_20",
    "nu": 5.0,
    "epsilon_min": 0.05,
    "epsilon_max": 0.20,
    "gate_center": 4.0,
    "gate_temperature": 8.0,
    "offset": MIXTURE_OFFSET,
    "local_component_counts": (1, 2, 4),
}
CONTROL_FAMILIES = ("ukf_apf_k1_defensive", "ukf_apf_k1")
HEURISTIC_FAMILIES = (
    "bootstrap_conditional",
    "transformed_student_nu8",
    "gaussian_hint_marginal",
    "stationary_independence",
)
COMPLEX_FAMILIES = (
    "ukf_apf_k1_defensive",
    "ukf_apf_k2_defensive",
    "ukf_apf_k4_defensive",
    "ukf_apf_k1",
    "ukf_apf_k2",
    "ukf_apf_k4",
    "retained_tt",
)

P2_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py"
P5A_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5a_hermite_20260904.py"
P5B_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5b_rbf_20260904.py"
P5C_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5c_hybrid_20260904.py"
P5CR_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5c_replication_20260904.py"
PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase7-integrated-20260904.md"
MASTER_PLAN_PATH = ROOT / "docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md"
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
ADAPTER_PATH = ROOT / "bayesfilter/highdim/c2_mixture_ukf_apf_c2_adapter.py"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
EXACT_PATH = ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py"
MOMENT_MAP_PATH = ROOT / "bayesfilter/highdim/recursive_moment_map_tf.py"
HYBRID_PATH = ROOT / "bayesfilter/highdim/hybrid_basis_tf.py"

PHASE_ID = "c2_mixture_ukf_apf_phase7_integrated_exact_dmis_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase7_integrated_result_v1"
MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase7_integrated_manifest_v1"
ROUTE_ID = "c2_exact_dmis_recursive_map_hybrid_control_diagnostic_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _workspace_manifest() -> Mapping[str, object]:
    status = _git("status", "--porcelain=v1")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": status,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value!r}")
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(value: object) -> bool:
    return bool(
        tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value, DTYPE))).numpy()
    )


def _scalar(value: object) -> float:
    tensor = tf.reshape(tf.convert_to_tensor(value, DTYPE), [])
    if not _finite(tensor):
        raise ValueError("non-finite scalar")
    return float(tensor.numpy())


def _max_abs(value: object) -> float:
    return _scalar(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))))


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _make_output_root(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _derived_seed(branch_index: int, family_index: int, time_index: int, role: int) -> tuple[int, int]:
    return (
        20260904,
        700000 + 10000 * int(branch_index) + 1000 * int(family_index) + 10 * int(time_index) + int(role),
    )


def _make_chunked_target_kernel(
    *,
    particle_count: int,
    state_dim: int,
    row_count: int,
    transition: tf.Tensor,
    theta: tf.Tensor,
    sigma: float,
    parent_chunk: int,
    jit_compile: bool,
):
    """Build an exact predictive-mixture target without an R x N allocation."""

    particle_count = int(particle_count)
    state_dim = int(state_dim)
    row_count = int(row_count)
    parent_chunk = int(parent_chunk)
    if particle_count % parent_chunk != 0:
        raise ValueError("particle_count must be divisible by parent_chunk")
    transition = tf.ensure_shape(
        tf.convert_to_tensor(transition, DTYPE), [state_dim, state_dim]
    )
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, DTYPE), [2])
    log_two_pi_sigma2 = tf.constant(
        math.log(2.0 * math.pi * float(sigma) ** 2), DTYPE
    )
    negative_infinity = tf.constant(float("-inf"), DTYPE)

    @tf.function(
        input_signature=[
            tf.TensorSpec([particle_count, state_dim], DTYPE),
            tf.TensorSpec([particle_count], DTYPE),
            tf.TensorSpec([state_dim], DTYPE),
            tf.TensorSpec([row_count, state_dim], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        parent_states: tf.Tensor,
        parent_weights: tf.Tensor,
        observation: tf.Tensor,
        physical_rows: tf.Tensor,
    ) -> tf.Tensor:
        log_parent_weights = tf.math.log(parent_weights)

        def condition(index: tf.Tensor, _accumulator: tf.Tensor) -> tf.Tensor:
            return index < tf.constant(particle_count, tf.int32)

        def body(index: tf.Tensor, accumulator: tf.Tensor):
            parents = tf.slice(
                parent_states, [index, 0], [parent_chunk, state_dim]
            )
            log_weights = tf.slice(log_parent_weights, [index], [parent_chunk])
            means = tf.linalg.matmul(parents, transition, transpose_b=True)
            residual = physical_rows[:, None, :] - means[None, :, :]
            log_transition = -0.5 * (
                tf.cast(state_dim, DTYPE) * log_two_pi_sigma2
                + tf.reduce_sum(tf.square(residual), axis=2)
            )
            chunk_value = tf.reduce_logsumexp(
                log_transition + log_weights[None, :], axis=1
            )
            combined = tf.reduce_logsumexp(
                tf.stack([accumulator, chunk_value], axis=1), axis=1
            )
            return index + tf.constant(parent_chunk, tf.int32), combined

        _, log_predictive = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                tf.fill([row_count], negative_infinity),
            ),
            parallel_iterations=1,
        )
        log_observation = tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), DTYPE)
            - theta[1]
            - 0.5 * physical_rows
            - 0.5
            * tf.square(observation)[None, :]
            * tf.exp(-physical_rows - 2.0 * theta[1]),
            axis=1,
        )
        return log_predictive + log_observation

    return kernel


def _exact_log_weight_path(
    p2: Any,
    model: Any,
    branch: Any,
    theta: tf.Tensor,
) -> Mapping[str, object]:
    """Reconstruct the exact normalized filtering weights at every state time."""

    initial_target = model.initial_log_density(theta, branch.states[0]) + model.observation_log_density(
        theta, branch.states[0], branch.observations[0], 0
    )
    log_unnormalized = (
        branch.initial_log_base_mass
        + initial_target
        - branch.initial_log_proposal_density
    )
    log_weights = log_unnormalized - tf.reduce_logsumexp(log_unnormalized)
    paths = [log_weights]
    identity_backward_max = tf.constant(0.0, DTYPE)
    for time_index in range(1, branch.time_steps):
        ancestor = branch.ancestors[time_index - 1]
        previous = tf.gather(branch.states[time_index - 1], ancestor)
        current = branch.states[time_index]
        selected_previous = tf.gather(log_weights, ancestor)
        selected_auxiliary = tf.gather(
            branch.auxiliary_log_probabilities[time_index - 1], ancestor
        )
        transition_log_density = model.transition_log_density(
            theta, previous, current, time_index
        )
        observation_log_density = model.observation_log_density(
            theta, current, branch.observations[time_index], time_index
        )
        transition_log_q = branch.transition_log_proposal_density[time_index - 1]
        exact_conditional_target = (
            selected_previous + transition_log_density + observation_log_density
        )
        corrected = exact_conditional_target - selected_auxiliary - transition_log_q
        residual = selected_auxiliary + transition_log_q + corrected - exact_conditional_target
        _, _, relative = p2._scaled_backward_error(
            residual,
            (
                selected_auxiliary,
                transition_log_q,
                corrected,
                exact_conditional_target,
            ),
        )
        identity_backward_max = tf.maximum(
            identity_backward_max, tf.reduce_max(relative)
        )
        log_unnormalized = branch.transition_log_base_mass[time_index - 1] + corrected
        log_weights = log_unnormalized - tf.reduce_logsumexp(log_unnormalized)
        paths.append(log_weights)
    exact = p2.prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
    parity = _max_abs(log_weights - exact["final_log_weights"])
    return {
        "log_weights": tuple(paths),
        "final_parity_max_abs": parity,
        "identity_backward_error_max": _scalar(identity_backward_max),
        "finite": all(_finite(row) for row in paths),
        "normalized_max_abs_error": max(
            _scalar(tf.abs(tf.reduce_logsumexp(row))) for row in paths
        ),
    }


def _target_values(
    *,
    p5a: Any,
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    transition: tf.Tensor,
    process: tf.Tensor,
    sigma: float,
    parent_states: tf.Tensor,
    parent_weights: tf.Tensor,
    observation_time: int,
    coordinate_map: Any,
    banks: Mapping[str, tf.Tensor],
    target_kernels: Mapping[str, Any],
    predictive_seed: tuple[int, int],
) -> Mapping[str, object]:
    wrappers = {
        name: (
            lambda physical, name=name: target_kernels[name](
                parent_states,
                parent_weights,
                observations[observation_time],
                physical,
            )
        )
        for name in banks
    }
    train_physical, train_logdet = coordinate_map.forward(banks["train"])
    train_log_gamma = wrappers["train"](train_physical)
    train_log_eta = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(banks["train"]), axis=1)
    )
    shift = tf.reduce_logsumexp(
        train_log_gamma + train_logdet - train_log_eta
    ) - tf.math.log(tf.cast(TRAIN_ROWS, DTYPE))
    log_targets: dict[str, tf.Tensor] = {}
    sqrt_targets: dict[str, tf.Tensor] = {}
    for name, rows in banks.items():
        log_targets[name], sqrt_targets[name], _ = p5a._reference_target_values(
            rows,
            coordinate_map=coordinate_map,
            target_kernel=wrappers[name],
            shift=shift,
        )
        if not _finite(log_targets[name]) or not _finite(sqrt_targets[name]):
            raise ValueError(f"non-finite exact target values in {name}")
    predictive_payload = {
        "states": parent_states,
        "weights": parent_weights,
        "transition": transition,
        "process": process,
        "theta": theta,
        "observations": tf.stack([observations[0], observations[observation_time]]),
        "model": model,
    }
    predictive = p5a._prior_predictive_target_audit(
        predictive_payload, row_count=AUDIT_ROWS, seed=predictive_seed
    )
    return {
        "shift": shift,
        "log_targets": log_targets,
        "sqrt_targets": sqrt_targets,
        "predictive": predictive,
        "finite": _finite(predictive["likelihood_values"]),
    }


def _control_records_for_branch(
    *,
    branch_index: int,
    family_index: int,
    family: str,
    compilation: Any,
    p2: Any,
    p5a: Any,
    p5b: Any,
    p5c: Any,
    p5cr: Any,
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    transition: tf.Tensor,
    process: tf.Tensor,
    sigma: float,
    target_kernels: Mapping[str, Any],
    moment_kernel: Any,
) -> tuple[list[Mapping[str, object]], Mapping[str, object], Mapping[str, str]]:
    path = _exact_log_weight_path(p2, model, compilation.branch, theta)
    if not bool(path["finite"]):
        raise ValueError("exact recursive branch weights are non-finite")
    if float(path["final_parity_max_abs"]) > 2.0e-10:
        raise ValueError("recursive branch weights differ from exact evaluator")
    records: list[Mapping[str, object]] = []
    stream_hashes: dict[str, str] = {}
    cell_validity: dict[str, bool] = {}
    for map_time in MAP_TIMES:
        parent_states = compilation.branch.states[map_time - 1]
        parent_log_weights = path["log_weights"][map_time - 1]
        parent_weights = tf.exp(parent_log_weights)
        conditional_means = tf.linalg.matmul(parent_states, transition, transpose_b=True)
        conditional_covariances = tf.broadcast_to(
            process[None, :, :], [PARTICLE_COUNT, STATE_DIM, STATE_DIM]
        )
        built = p5a.build_lagged_moment_map(
            parent_weights,
            conditional_means,
            conditional_covariances,
            jit_compile=True,
            kernel=moment_kernel,
        )
        coordinate_map = built["coordinate_map"]
        map_probe = p5a._standard_normal_bank(
            128, _derived_seed(branch_index, family_index, map_time, 0)
        )
        map_metrics = p5cr._map_metrics(coordinate_map, map_probe)
        banks = {
            "train": p5a._standard_normal_bank(
                TRAIN_ROWS, _derived_seed(branch_index, family_index, map_time, 1)
            ),
            "holdout": p5a._standard_normal_bank(
                HOLDOUT_ROWS, _derived_seed(branch_index, family_index, map_time, 2)
            ),
            "audit": p5a._standard_normal_bank(
                AUDIT_ROWS, _derived_seed(branch_index, family_index, map_time, 3)
            ),
        }
        for role, rows in banks.items():
            stream_hashes[
                f"branch{branch_index}_{family}_t{map_time}_{role}"
            ] = _tensor_hash(rows)
        target = _target_values(
            p5a=p5a,
            model=model,
            theta=theta,
            observations=observations,
            transition=transition,
            process=process,
            sigma=sigma,
            parent_states=parent_states,
            parent_weights=parent_weights,
            observation_time=map_time,
            coordinate_map=coordinate_map,
            banks=banks,
            target_kernels=target_kernels,
            predictive_seed=_derived_seed(branch_index, family_index, map_time, 4),
        )
        shared = {
            "map_condition_number": built["condition_number"],
            "map_minimum_eigenvalue": built["minimum_eigenvalue"],
        }
        arms: list[Mapping[str, object]] = []
        for width in ARM_WIDTHS:
            arm = p5c._fit_arm(
                f"hybrid_d6_w{int(round(width * 100)):03d}",
                width,
                p5a=p5a,
                p5b=p5b,
                shared=shared,
                banks=banks,
                sqrt_targets=target["sqrt_targets"],
                log_targets=target["log_targets"],
                direct_target={
                    "z_t": target["predictive"]["z_t"],
                    "standard_error": target["predictive"]["standard_error"],
                },
                shift=target["shift"],
            )
            arms.append(
                {
                    "arm": arm["arm"],
                    "width": arm["width"],
                    "basis_family": arm["basis_family"],
                    "basis_dim_per_axis": arm["basis_dim_per_axis"],
                    "holdout_rms": arm["holdout_rms"],
                    "holdout_central_rms": arm["holdout_central_rms"],
                    "holdout_shell_rms": arm["holdout_shell_rms"],
                    "log_z_h_minus_log_z_t": arm["log_z_h_minus_log_z_t"],
                    "z_h": arm["z_h"],
                    "z_t": arm["z_t_direct"],
                    "z_t_standard_error": arm["z_t_direct_standard_error"],
                    "mass_condition_number": arm["contractions"]["mass_condition_number"],
                    "mass_quadrature_pass": arm["contractions"]["quadrature_pass"],
                    "fit_condition_max": arm["condition"]["scaled_augmented_condition_max"],
                    "fit_status": arm["fit_status"],
                    "finite": arm["finite"],
                    "hard_valid": arm["hard_valid"],
                }
            )
        map_valid = bool(
            built["valid"].numpy()
            and map_metrics["physical_finite"]
            and map_metrics["reference_finite"]
            and float(map_metrics["roundtrip_max_abs"]) <= 2.0e-12
            and float(map_metrics["minimum_cholesky_diagonal"]) > 0.0
        )
        target_valid = bool(target["finite"])
        quadrature_and_target_valid = bool(
            target_valid
            and all(bool(arm["mass_quadrature_pass"]) and bool(arm["finite"]) for arm in arms)
        )
        at_least_one_arm_valid = any(bool(arm["hard_valid"]) for arm in arms)
        cell_valid = map_valid and quadrature_and_target_valid and at_least_one_arm_valid
        cell_validity[f"{family}_t{map_time}"] = cell_valid
        records.append(
            {
                "branch_index": branch_index,
                "family": family,
                "map_time": map_time,
                "parent_time": map_time - 1,
                "parent_ess": _scalar(
                    tf.math.reciprocal(tf.reduce_sum(tf.square(parent_weights)))
                ),
                "parent_weight_normalization_abs_error": _scalar(
                    tf.abs(tf.reduce_sum(parent_weights) - 1.0)
                ),
                "map_condition_number": _scalar(built["condition_number"]),
                "map_minimum_eigenvalue": _scalar(built["minimum_eigenvalue"]),
                "map_roundtrip_max_abs": map_metrics["roundtrip_max_abs"],
                "map_valid": map_valid,
                "target_valid": target_valid,
                "z_t": _scalar(target["predictive"]["z_t"]),
                "z_t_standard_error": _scalar(target["predictive"]["standard_error"]),
                "z_t_relative_standard_error": _scalar(
                    target["predictive"]["relative_standard_error"]
                ),
                "at_least_one_arm_valid": at_least_one_arm_valid,
                "quadrature_and_target_valid": quadrature_and_target_valid,
                "cell_valid": cell_valid,
                "arms": arms,
            }
        )
    return records, {
        "final_log_weight_parity_max_abs": path["final_parity_max_abs"],
        "identity_backward_error_max": path["identity_backward_error_max"],
        "normalized_log_weight_max_abs_error": path["normalized_max_abs_error"],
        "cell_validity": cell_validity,
    }, stream_hashes


def _family_summary(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        grouped.setdefault(str(row["label"]), []).append(row)
    summary: dict[str, object] = {}
    for label, rows in sorted(grouped.items()):
        evaluated = [row for row in rows if "error" not in row]
        if not evaluated:
            summary[label] = {
                "record_count": len(rows),
                "evaluated_count": 0,
                "valid_count": 0,
            }
            continue
        minima = [float(row["minimum_ess"]) for row in evaluated]
        log_values = [float(row["log_likelihood"]) for row in evaluated]
        time_values = [
            [float(value) for value in row["ess_by_time"]] for row in evaluated
        ]
        summary[label] = {
            "record_count": len(rows),
            "evaluated_count": len(evaluated),
            "valid_count": sum(bool(row.get("all_checks_pass", False)) for row in rows),
            "minimum_ess_mean": statistics.mean(minima),
            "minimum_ess_min": min(minima),
            "minimum_ess_max": max(minima),
            "log_likelihood_mean": statistics.mean(log_values),
            "log_likelihood_min": min(log_values),
            "log_likelihood_max": max(log_values),
            "conditional_ess_mean": {
                str(time_index): statistics.mean(values[time_index] for values in time_values)
                for time_index in MAP_TIMES
            },
        }
        spreads = [
            float(row["proposal"].get("per_ancestor_posterior_mean_spread_max", 0.0))
            for row in evaluated
            if "proposal" in row
        ]
        lookaheads = [
            float(row["proposal"].get("lookahead_spread_max", 0.0))
            for row in evaluated
            if "proposal" in row
        ]
        if spreads:
            summary[label]["posterior_mean_spread_mean"] = statistics.mean(spreads)
        if lookaheads:
            summary[label]["lookahead_spread_mean"] = statistics.mean(lookaheads)
    return summary


def _heuristic_dominance(summary: Mapping[str, object]) -> Mapping[str, object]:
    verdicts: dict[str, object] = {}
    for candidate in COMPLEX_FAMILIES:
        comparisons: list[Mapping[str, object]] = []
        candidate_row = summary.get(candidate)
        if not isinstance(candidate_row, Mapping) or int(candidate_row.get("evaluated_count", 0)) == 0:
            verdicts[candidate] = {
                "verdict": "NOT_EVALUABLE",
                "comparisons": comparisons,
            }
            continue
        for heuristic in HEURISTIC_FAMILIES:
            heuristic_row = summary.get(heuristic)
            if not isinstance(heuristic_row, Mapping) or int(heuristic_row.get("evaluated_count", 0)) == 0:
                continue
            for time_index in MAP_TIMES:
                candidate_ess = float(candidate_row["conditional_ess_mean"][str(time_index)])
                heuristic_ess = float(heuristic_row["conditional_ess_mean"][str(time_index)])
                comparisons.append(
                    {
                        "heuristic": heuristic,
                        "time_index": time_index,
                        "candidate_mean_ess": candidate_ess,
                        "heuristic_mean_ess": heuristic_ess,
                        "candidate_minus_heuristic": candidate_ess - heuristic_ess,
                        "candidate_loses": candidate_ess < heuristic_ess,
                    }
                )
        loses = any(bool(row["candidate_loses"]) for row in comparisons)
        verdicts[candidate] = {
            "verdict": (
                "PROMOTION_VETO_HEURISTIC_LOSS"
                if loses
                else "PASSES_WEAK_HEURISTIC_SCREEN_ONLY"
            ),
            "comparisons": comparisons,
        }
    return verdicts


def _result_markdown(payload: Mapping[str, object]) -> str:
    checks = payload["checks"]
    losses = [
        family
        for family, row in payload["heuristic_dominance"].items()
        if row["verdict"] == "PROMOTION_VETO_HEURISTIC_LOSS"
    ]
    lines = [
        "# C2 Mixture-UKF/APF Phase 7 Integrated Result",
        "",
        f"Status: `{payload['status']}`  ",
        f"Continuation: `{payload['continuation']}`  ",
        f"Failure class: `{payload['failure_class']}`",
        "",
        "The exact observation-conditioned APF/DMIS call chain "
        + ("passed" if checks["hard_vetoes_pass"] else "failed")
        + " its finite-program, proposal-law, GPU/XLA, recursive-map, and hybrid-control validity checks.",
        "",
        "Heuristic-dominance headline: "
        + (", ".join(f"`{name}`" for name in losses) + " lost to at least one cheap proposal at a salient time and are promotion-vetoed."
           if losses else "no complex family lost the predeclared weak screen; this does not certify quality."),
        "",
        "## Decision",
        "",
        "| Decision | Status | Interpretation |",
        "| --- | --- | --- |",
        f"| Exact frozen APF/DMIS program | {'PASS' if checks['proposal_records_valid'] else 'VETO'} | {'all 33 paired records use exact target and complete denominators' if checks['proposal_records_valid'] else 'repair before interpretation'} |",
        f"| Recursive map and hybrid control | {'PASS' if checks['control_records_valid'] else 'VETO'} | {'all control cells are valid; hybrid remains outside the sampler' if checks['control_records_valid'] else 'repair map/target/control'} |",
        "| Family or width promotion | NOT TESTED | three branches are descriptive and this phase has no promotion criterion |",
        "| Hybrid integrated sampler | NOT IMPLEMENTED | fitted TT does not generate particles and is absent from q |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        f"| Hard veto screen | {'passed' if checks['hard_vetoes_pass'] else 'failed'} |",
        "| Statistically supported ranking | none; three paired branches are underpowered |",
        "| Descriptive-only differences | per-family ESS, exact values, and control residuals below |",
        "| Default readiness | not assessed |",
        "| Next evidence | implement a proposal sampler that can actually consume the recursive fitted guide, or run the separately planned reference/decision study |",
        "",
        "## Proposal families",
        "",
        "| family | records valid | min ESS mean | min ESS range | ESS t1 | ESS t3 | ESS t10 | ESS t19 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in payload["family_summary"].items():
        if int(row.get("evaluated_count", 0)) == 0:
            lines.append(f"| {family} | 0/{row['record_count']} | error | error | error | error | error | error |")
            continue
        conditional = row["conditional_ess_mean"]
        lines.append(
            f"| {family} | {row['valid_count']}/{row['record_count']} | {row['minimum_ess_mean']:.5g} | "
            f"[{row['minimum_ess_min']:.5g}, {row['minimum_ess_max']:.5g}] | "
            f"{conditional['1']:.5g} | {conditional['3']:.5g} | {conditional['10']:.5g} | {conditional['19']:.5g} |"
        )
    lines += [
        "",
        "These are paired descriptive summaries. ESS is an efficiency diagnostic, not a correctness proof and not a tuning target.",
        "",
        "## Recursive hybrid controls",
        "",
        "| branch | family | target time | parent ESS | map cond. | min eig. | ZT rel. SE | width | shell RMS | log ZH-ZT | fit cond. | valid |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for cell in payload["control_records"]:
        for arm in cell["arms"]:
            lines.append(
                f"| {cell['branch_index']} | {cell['family']} | {cell['map_time']} | {cell['parent_ess']:.5g} | "
                f"{cell['map_condition_number']:.5g} | {cell['map_minimum_eigenvalue']:.5g} | {cell['z_t_relative_standard_error']:.4g} | "
                f"{arm['width']:.3g} | {arm['holdout_shell_rms']:.5g} | {arm['log_z_h_minus_log_z_t']:.5g} | "
                f"{arm['fit_condition_max']:.5g} | {arm['hard_valid']} |"
            )
    lines += [
        "",
        "## Computed target",
        "",
        "The claimed finite target is the frozen APF likelihood estimator with exact C2 transition and observation densities, the realized ancestor law, and the complete conditional proposal density. The program computes that target. The recursive hybrid representation is a separate diagnostic and is not equal to, or substituted for, the proposal law.",
        "",
        "## Repair history",
        "",
        "Before this run, the call-chain audit found that the working-tree exact evaluator lacked the generalized base-mass fields already required by the deterministic-mixture compiler and Phase 4 diagnostics. The preserved implementation was restored exactly and a nonuniform-base-mass regression was added. The initial test invocation selected the system pytest entry point; the unchanged tests passed when invoked through the declared conda Python. GPU attempt 01 then stopped before proposal construction because helper imports initialized TensorFlow before memory growth was configured. Attempt 02 moves repository memory-policy configuration ahead of every dynamic scientific-module import; the target, data, methods, and evidence criteria are unchanged.",
        "",
        "## Post-run red team",
        "",
        "The strongest alternative explanation for any ESS contrast is finite-branch variation on one C2 observation path. Exact denominator and score checks establish finite-program validity, not posterior accuracy or general-model performance. The hybrid controls can reveal representation mismatch but cannot demonstrate proposal improvement because they do not sample or appear in the denominator.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--jit-compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not bool(args.jit_compile):
        raise ValueError("Phase 7 serious execution requires --jit-compile")
    output = _make_output_root(args.output_root)
    (output / "plan-at-launch.md").write_text(
        PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    started = time.perf_counter()

    # Memory growth must be established before helper imports create any
    # TensorFlow constants or logical devices.
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("Phase 7 requires a logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], DTYPE))
    if "GPU" not in str(placement_probe.device).upper():
        raise RuntimeError(f"Phase 7 placement probe ran on {placement_probe.device}")
    runtime = {
        "execution_lane": "owner_designated_managed_session_visible_gpu_trusted",
        "memory_policy": memory_policy,
        "physical_devices": [str(device.name) for device in physical_gpus],
        "logical_devices": [str(device.name) for device in logical_gpus],
        "placement_probe_device": str(placement_probe.device),
        "placement_probe_value": _scalar(placement_probe),
    }

    p2 = _load_module("c2_phase7_p2_helpers", P2_DRIVER_PATH)
    p5c = _load_module("c2_phase7_p5c_helpers", P5C_DRIVER_PATH)
    p5b = p5c._load_phase5b()
    p5a = p5b._load_phase5a()
    p5cr = _load_module("c2_phase7_p5cr_helpers", P5CR_DRIVER_PATH)
    p2._load_algorithm_modules()
    p5a._load_modules()
    p5c._load_hybrid_basis()

    fixture = p2._load_c2_fixture()
    model, theta, observations = p2._c2_model_and_inputs(fixture)
    if int(observations.shape[0]) != HORIZON or int(observations.shape[1]) != STATE_DIM:
        raise ValueError("Phase 7 requires the frozen T=20, d=4 C2 fixture")
    prepared = p2._prepare_phase2_proposals(
        fixture=fixture,
        model=model,
        theta=theta,
        observations=observations,
        output_root=output,
    )
    proposal_sets = {
        "retained_tt": prepared["tt_proposals"],
        "gaussian_hint_marginal": prepared["hint_proposals"],
        "stationary_independence": prepared["stationary_proposals"],
        "transformed_student_nu8": prepared["defensive_proposals"],
    }
    transition = model.transition_matrix(theta)
    process = tf.eye(STATE_DIM, dtype=DTYPE) * float(fixture["sigma"]) ** 2
    target_kernels = {
        "train": _make_chunked_target_kernel(
            particle_count=PARTICLE_COUNT,
            state_dim=STATE_DIM,
            row_count=TRAIN_ROWS,
            transition=transition,
            theta=theta,
            sigma=float(fixture["sigma"]),
            parent_chunk=TARGET_PARENT_CHUNK,
            jit_compile=True,
        ),
        "holdout": _make_chunked_target_kernel(
            particle_count=PARTICLE_COUNT,
            state_dim=STATE_DIM,
            row_count=HOLDOUT_ROWS,
            transition=transition,
            theta=theta,
            sigma=float(fixture["sigma"]),
            parent_chunk=TARGET_PARENT_CHUNK,
            jit_compile=True,
        ),
        "audit": _make_chunked_target_kernel(
            particle_count=PARTICLE_COUNT,
            state_dim=STATE_DIM,
            row_count=AUDIT_ROWS,
            transition=transition,
            theta=theta,
            sigma=float(fixture["sigma"]),
            parent_chunk=TARGET_PARENT_CHUNK,
            jit_compile=True,
        ),
    }
    moment_kernel = p5a.make_weighted_transition_moment_kernel(
        particle_count=PARTICLE_COUNT,
        state_dim=STATE_DIM,
        jit_compile=True,
    )

    proposal_records: list[Mapping[str, object]] = []
    control_records: list[Mapping[str, object]] = []
    control_path_checks: dict[str, object] = {}
    stream_hashes: dict[str, str] = {}
    branch_seeds: list[int] = []
    for branch_index in range(BRANCH_COUNT):
        seed = p2.PHASE4_CLAIM_SEED_BASE + PARTICLE_COUNT + 1009 * branch_index
        branch_seeds.append(seed)
        specs = p2._phase4_candidate_specs(
            model=model,
            observations=observations,
            theta=theta,
            proposal_sets=proposal_sets,
            particle_count=PARTICLE_COUNT,
            seed=seed,
            mixture_offset=MIXTURE_OFFSET,
            defensive_config=SELECTED_DEFENSIVE_CONFIG,
        )
        selected_compilations: dict[str, Any] = {}
        for label, compiler, _declared_check_xla in specs:
            branch_started = time.perf_counter()
            record: dict[str, object] = {
                "particle_count": PARTICLE_COUNT,
                "branch_index": branch_index,
                "seed": seed,
                "label": label,
                "mixture_offset": MIXTURE_OFFSET,
                "defensive_config": SELECTED_DEFENSIVE_CONFIG,
                "requested_jit_compile": True,
            }
            try:
                compilation = compiler()
                evaluated = dict(
                    p2._evaluate_c2_candidate(
                        label=label,
                        compilation=compilation,
                        model=model,
                        theta=theta,
                        check_xla=True,
                    )
                )
                gpu_output = "GPU" in str(evaluated.get("output_device", "")).upper()
                evaluated["gpu_output"] = gpu_output
                evaluated["branch_wall_seconds"] = time.perf_counter() - branch_started
                evaluated["checks"] = dict(evaluated["checks"])
                evaluated["checks"]["gpu_output"] = gpu_output
                evaluated["checks"]["branch_identity_wired"] = bool(
                    evaluated["branch_id"] == compilation.branch.branch_id
                )
                evaluated["checks"]["base_mass_rows_normalized"] = bool(
                    float(evaluated["apf"]["max_base_mass_normalization_abs_error"])
                    <= p2.ABS_TOL
                )
                evaluated["all_checks_pass"] = all(
                    bool(value) for value in evaluated["checks"].values()
                )
                evaluated["compiler_manifest"] = compilation.manifest
                record.update(_jsonable(evaluated))
                if label in CONTROL_FAMILIES:
                    selected_compilations[label] = compilation
            except Exception as exc:
                record.update(
                    {
                        "all_checks_pass": False,
                        "failure_class": "candidate_or_comparator_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "branch_wall_seconds": time.perf_counter() - branch_started,
                        "gpu_output": False,
                    }
                )
            proposal_records.append(record)
            _write_json(output / "proposal_records.partial.json", proposal_records)

        for family_index, family in enumerate(CONTROL_FAMILIES):
            compilation = selected_compilations.get(family)
            if compilation is None:
                control_path_checks[f"branch{branch_index}_{family}"] = {
                    "valid": False,
                    "error": "proposal compilation unavailable",
                }
                continue
            try:
                rows, path_checks, hashes = _control_records_for_branch(
                    branch_index=branch_index,
                    family_index=family_index,
                    family=family,
                    compilation=compilation,
                    p2=p2,
                    p5a=p5a,
                    p5b=p5b,
                    p5c=p5c,
                    p5cr=p5cr,
                    model=model,
                    theta=theta,
                    observations=observations,
                    transition=transition,
                    process=process,
                    sigma=float(fixture["sigma"]),
                    target_kernels=target_kernels,
                    moment_kernel=moment_kernel,
                )
                control_records.extend(rows)
                stream_hashes.update(hashes)
                control_path_checks[f"branch{branch_index}_{family}"] = {
                    "valid": all(bool(row["cell_valid"]) for row in rows),
                    **path_checks,
                }
            except Exception as exc:
                control_path_checks[f"branch{branch_index}_{family}"] = {
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _write_json(output / "control_records.partial.json", control_records)

    family_summary = _family_summary(proposal_records)
    heuristic = _heuristic_dominance(family_summary)
    expected_proposals = BRANCH_COUNT * len(p2.PHASE4_FAMILIES)
    expected_controls = BRANCH_COUNT * len(CONTROL_FAMILIES) * len(MAP_TIMES)
    proposal_records_valid = bool(
        len(proposal_records) == expected_proposals
        and all(bool(row.get("all_checks_pass", False)) for row in proposal_records)
    )
    control_records_valid = bool(
        len(control_records) == expected_controls
        and all(bool(row.get("cell_valid", False)) for row in control_records)
        and len(control_path_checks) == BRANCH_COUNT * len(CONTROL_FAMILIES)
        and all(bool(row.get("valid", False)) for row in control_path_checks.values())
    )
    checks = {
        "source_files_present": all(
            path.is_file()
            for path in (
                PLAN_PATH,
                MASTER_PLAN_PATH,
                P2_DRIVER_PATH,
                P5A_DRIVER_PATH,
                P5B_DRIVER_PATH,
                P5C_DRIVER_PATH,
                P5CR_DRIVER_PATH,
                FIXTURE_PATH,
                ADAPTER_PATH,
                MODEL_PATH,
                EXACT_PATH,
                MOMENT_MAP_PATH,
                HYBRID_PATH,
            )
        ),
        "proposal_records_complete": len(proposal_records) == expected_proposals,
        "proposal_records_valid": proposal_records_valid,
        "control_records_complete": len(control_records) == expected_controls,
        "control_records_valid": control_records_valid,
        "hybrid_absent_from_proposal_denominator": True,
        "gpu_xla_requested": bool(args.jit_compile),
        "memory_growth_verified": bool(
            runtime["memory_policy"]["all_physical_devices_memory_growth"]
        ),
    }
    checks["hard_vetoes_pass"] = all(bool(value) for value in checks.values())
    any_heuristic_loss = any(
        row["verdict"] == "PROMOTION_VETO_HEURISTIC_LOSS"
        for row in heuristic.values()
    )
    any_width_failure = any(
        not bool(arm["hard_valid"])
        for cell in control_records
        for arm in cell["arms"]
    )
    failure_class = (
        "implementation_or_numerical_validity"
        if not checks["hard_vetoes_pass"]
        else ("candidate_failure" if any_heuristic_loss or any_width_failure else "none")
    )
    status = (
        "PASS_PHASE7_INTEGRATED_VALIDITY_WITH_PROMOTION_VETO"
        if checks["hard_vetoes_pass"] and (any_heuristic_loss or any_width_failure)
        else (
            "PASS_PHASE7_INTEGRATED_DIAGNOSTIC"
            if checks["hard_vetoes_pass"]
            else "VETO_PHASE7_INTEGRATED_VALIDITY"
        )
    )
    continuation = (
        "CONTINUE_REFERENCE_OR_HYBRID_SAMPLER_REPAIR"
        if checks["hard_vetoes_pass"]
        else "CONTINUATION_VETO_REPAIR_PHASE7_VALIDITY"
    )
    elapsed = time.perf_counter() - started
    source_paths = {
        "plan": PLAN_PATH,
        "master_plan": MASTER_PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "phase2_driver": P2_DRIVER_PATH,
        "phase5a_driver": P5A_DRIVER_PATH,
        "phase5b_driver": P5B_DRIVER_PATH,
        "phase5c_driver": P5C_DRIVER_PATH,
        "phase5cr_driver": P5CR_DRIVER_PATH,
        "fixture": FIXTURE_PATH,
        "adapter": ADAPTER_PATH,
        "model": MODEL_PATH,
        "exact_evaluator": EXACT_PATH,
        "moment_map": MOMENT_MAP_PATH,
        "hybrid_basis": HYBRID_PATH,
    }
    sources = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    nonclaims = [
        "no posterior-correctness or unbiased-likelihood claim",
        "no statistically supported family or width ranking",
        "no claim that the hybrid TT representation generated proposals",
        "no adaptive total-gradient or HMC claim",
        "no default, production, or general-model claim",
    ]
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH",
                _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH or "unset",
            ),
            "jit_compile": bool(args.jit_compile),
            "runtime": runtime,
        },
        "particle_count": PARTICLE_COUNT,
        "branch_count": BRANCH_COUNT,
        "branch_seeds": branch_seeds,
        "horizon": HORIZON,
        "proposal_families": p2.PHASE4_FAMILIES,
        "control_families": CONTROL_FAMILIES,
        "map_times": MAP_TIMES,
        "control_rows": {
            "train": TRAIN_ROWS,
            "holdout": HOLDOUT_ROWS,
            "audit": AUDIT_ROWS,
        },
        "target_parent_chunk": TARGET_PARENT_CHUNK,
        "hybrid_widths": ARM_WIDTHS,
        "selected_defensive_config": SELECTED_DEFENSIVE_CONFIG,
        "fresh_retained_tt_fit_seconds": prepared["fit_seconds"],
        "fresh_retained_tt_run_identity": prepared["run_identity"],
        "sources": sources,
        "workspace": _workspace_manifest(),
        "checks": checks,
        "proposal_record_count": len(proposal_records),
        "control_record_count": len(control_records),
        "proposal_records": proposal_records,
        "family_summary": family_summary,
        "heuristic_dominance": heuristic,
        "control_records": control_records,
        "control_path_checks": control_path_checks,
        "stream_hashes": stream_hashes,
        "repair_history": [
            {
                "attempt": "preflight",
                "classification": "call_chain_regression",
                "repair": "restore generalized normalized base-mass contract and add direct regression",
            },
            {
                "attempt": "phase7-integrated-attempt01",
                "classification": "harness_initialization_order",
                "repair": "configure and verify GPU memory growth before dynamic scientific-module imports",
                "scientific_output": False,
            },
        ],
        "nonclaims": nonclaims,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": sources["plan"]["sha256"],
        "master_plan_sha256": sources["master_plan"]["sha256"],
        "sources": sources,
        "workspace": payload["workspace"],
        "environment": payload["environment"],
        "particle_count": PARTICLE_COUNT,
        "branch_count": BRANCH_COUNT,
        "branch_seeds": branch_seeds,
        "horizon": HORIZON,
        "proposal_families": p2.PHASE4_FAMILIES,
        "control_families": CONTROL_FAMILIES,
        "map_times": MAP_TIMES,
        "control_rows": payload["control_rows"],
        "target_parent_chunk": TARGET_PARENT_CHUNK,
        "hybrid_widths": ARM_WIDTHS,
        "selected_defensive_config": SELECTED_DEFENSIVE_CONFIG,
        "stream_hashes": stream_hashes,
        "evidence_contract": {
            "question": "exact-DMIS validity and observation localization under a full-horizon branch with recursive hybrid controls",
            "primary": "all 33 exact proposal records and all 24 recursive control cells pass declared validity checks",
            "promotion_veto": "any complex family loses to any cheap heuristic at a salient time",
            "continuation_veto": "target/denominator mismatch, incomplete or nonfinite records, invalid map/control, or GPU/XLA/memory provenance failure",
            "descriptive": "ESS, log likelihood, residuals, normalizer gaps, and three-branch contrasts",
            "nonclaims": nonclaims,
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "proposal_records.json", proposal_records)
    _write_json(output / "control_records.json", control_records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    output = run(args)
    print(json.dumps({"phase": PHASE_ID, "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
