"""Run the bounded Phase 5C-R replicated/recursive hybrid diagnostic.

Three independent seed families fit the two surviving Phase 5C hybrid arms at
two recursively rebuilt lagged moment maps.  The exact C2 transition-mixture
target and predictive-mixture normalizer are retained at every step.  This is
diagnostic finite recursion, not a claim-bearing particle filter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf


DTYPE = tf.float64
STATE_DIM = 4
CARRIED_ROWS = 128
TRAIN_ROWS = 512
HOLDOUT_ROWS = 512
AUDIT_ROWS = 4096
HORIZON = 2
TT_RANK = 2
ALS_SWEEPS = 2
RIDGE = 1.0e-8
CONDITION_VETO = 1.0e14
SHELL_RADIUS = 2.0
QUADRATURE_ORDERS = (80, 100)
HERMITE_DEGREE = 6
CENTERS = (-2.0, -1.0, 0.0, 1.0, 2.0)
ARM_WIDTHS = (0.75, 1.50)
SEED_FAMILIES = ((20260904, 701), (20260904, 1701), (20260904, 2701))
MAP_SEED_FAMILIES = ((20260904, 501), (20260904, 1501), (20260904, 2501))

PHASE5C_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5c_hybrid_20260904.py"
PHASE5A_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5a_hermite_20260904.py"
PHASE5B_DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5b_rbf_20260904.py"
HYBRID_MODULE_PATH = ROOT / "bayesfilter/highdim/hybrid_basis_tf.py"
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
MOMENT_MAP_PATH = ROOT / "bayesfilter/highdim/recursive_moment_map_tf.py"
PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase5c-replication-20260904.md"

PHASE_ID = "c2_mixture_ukf_apf_phase5c_replication_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase5c_replication_result_v1"
MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase5c_replication_manifest_v1"
ROUTE_ID = "c2_fixed_map_hermite_rbf_recursive_replication_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

_PHASE5C: Any | None = None


def _load_phase5c() -> Any:
    global _PHASE5C
    if _PHASE5C is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "c2_phase5c_shared_helpers", PHASE5C_DRIVER_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load Phase 5C shared helper module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PHASE5C = module
    return _PHASE5C


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
            raise ValueError("non-finite artifact value")
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


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


def _configure_runtime() -> Mapping[str, object]:
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    p5c = _load_phase5c()
    p5b = p5c._load_phase5b()
    runtime = dict(p5b._configure_runtime())
    runtime["execution_lane"] = "trusted_gpu_xla_phase5c_replication"
    return runtime


def _make_output_root(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _derived_seed(base: tuple[int, int], step: int, offset: int) -> tuple[int, int]:
    # Keep each stream disjoint while making its role visible in the manifest.
    return (int(base[0]), int(base[1]) + int(step) * 100 + int(offset))


def _map_metrics(coordinate_map: Any, reference_bank: tf.Tensor) -> Mapping[str, object]:
    physical, _ = coordinate_map.forward(reference_bank)
    recovered, _ = coordinate_map.inverse(physical)
    singular = tf.linalg.svd(coordinate_map.matrix, compute_uv=False)
    return {
        "roundtrip_max_abs": _max_abs(recovered - reference_bank),
        "physical_finite": _finite(physical),
        "reference_finite": _finite(recovered),
        "condition_number": _scalar(tf.reduce_max(singular) / tf.reduce_min(singular)),
        "minimum_cholesky_diagonal": _scalar(tf.reduce_min(tf.linalg.diag_part(coordinate_map.matrix))),
    }


def _weighted_moments(weights: tf.Tensor, points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.einsum("n,nd->d", weights, points)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, 0.5 * (covariance + tf.transpose(covariance))


def _fit_step(
    *,
    p5a: Any,
    p5b: Any,
    p5c: Any,
    name: str,
    width: float,
    shared: Mapping[str, object],
    banks: Mapping[str, tf.Tensor],
    shift: tf.Tensor,
    log_targets: Mapping[str, tf.Tensor],
    sqrt_targets: Mapping[str, tf.Tensor],
    predictive: Mapping[str, tf.Tensor],
) -> Mapping[str, object]:
    # The audited Phase 5C endpoint is the only fitting call in this driver.
    return p5c._fit_arm(
        name,
        width,
        p5a=p5a,
        p5b=p5b,
        shared=shared,
        banks=banks,
        sqrt_targets=sqrt_targets,
        log_targets=log_targets,
        direct_target={"z_t": predictive["z_t"], "standard_error": predictive["standard_error"]},
        shift=shift,
    )


def _initial_payload(p5a: Any, fixture: Mapping[str, object], map_seed: tuple[int, int], *, jit_compile: bool) -> Mapping[str, object]:
    old_map_seed = p5a.MAP_SEED
    p5a.MAP_SEED = map_seed
    try:
        payload = p5a._model_and_map(fixture, jit_compile=bool(jit_compile))
    finally:
        p5a.MAP_SEED = old_map_seed
    return payload


def _make_recursive_target_kernel(
    transition: tf.Tensor,
    theta: tf.Tensor,
    sigma: float,
    row_count: int,
    *,
    jit_compile: bool,
):
    """Build one fixed-signature target kernel and pass the recursive cloud in."""

    transition = tf.ensure_shape(tf.convert_to_tensor(transition, DTYPE), [STATE_DIM, STATE_DIM])
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, DTYPE), [2])
    log_two_pi_sigma2 = tf.constant(math.log(2.0 * math.pi * float(sigma) ** 2), DTYPE)

    @tf.function(
        input_signature=[
            tf.TensorSpec([CARRIED_ROWS, STATE_DIM], DTYPE),
            tf.TensorSpec([CARRIED_ROWS], DTYPE),
            tf.TensorSpec([STATE_DIM], DTYPE),
            tf.TensorSpec([int(row_count), STATE_DIM], DTYPE),
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
        means = tf.linalg.matmul(parent_states, transition, transpose_b=True)
        residual = physical_rows[:, None, :] - means[None, :, :]
        log_transition = -0.5 * (
            tf.cast(STATE_DIM, DTYPE) * log_two_pi_sigma2
            + tf.reduce_sum(tf.square(residual), axis=2)
        )
        log_gamma = tf.reduce_logsumexp(
            log_transition + tf.math.log(parent_weights)[None, :], axis=1
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
        return log_gamma + log_observation

    return kernel


def _step_target(
    *,
    p5a: Any,
    fixture: Mapping[str, object],
    shared: Mapping[str, object],
    parent_states: tf.Tensor,
    parent_weights: tf.Tensor,
    observation: tf.Tensor,
    banks: Mapping[str, tf.Tensor],
    coordinate_map: Any,
    target_kernels: Mapping[str, Any],
    predictive_seed: tuple[int, int],
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor], Mapping[str, tf.Tensor]]:
    del fixture
    kernels = {
        name: (lambda physical, kernel=target_kernels[name]: kernel(
            parent_states, parent_weights, observation, physical
        ))
        for name in banks
    }
    train_physical, train_logdet = coordinate_map.forward(banks["train"])
    train_log_gamma = kernels["train"](train_physical)
    train_log_eta = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(banks["train"]), axis=1)
    )
    shift = tf.reduce_logsumexp(train_log_gamma + train_logdet - train_log_eta) - tf.math.log(
        tf.cast(TRAIN_ROWS, DTYPE)
    )
    log_targets: dict[str, tf.Tensor] = {}
    sqrt_targets: dict[str, tf.Tensor] = {}
    for name, rows in banks.items():
        log_targets[name], sqrt_targets[name], _ = p5a._reference_target_values(
            rows,
            coordinate_map=coordinate_map,
            target_kernel=kernels[name],
            shift=shift,
        )
        if not _finite(log_targets[name]) or not _finite(sqrt_targets[name]):
            raise ValueError(f"non-finite exact target values in {name}")
    target_payload = dict(shared)
    target_payload["states"] = parent_states
    target_payload["weights"] = parent_weights
    target_payload["observations"] = tf.stack([shared["observations"][0], observation])
    predictive = p5a._prior_predictive_target_audit(
        target_payload, row_count=AUDIT_ROWS, seed=predictive_seed
    )
    return shift, log_targets, sqrt_targets, predictive


def _summarize_uncertainty(records: list[Mapping[str, object]]) -> Mapping[str, object]:
    by_arm_seed: dict[str, dict[int, list[Mapping[str, object]]]] = {}
    for row in records:
        by_arm_seed.setdefault(str(row["arm"]), {}).setdefault(int(row["seed_index"]), []).append(row)
    summary: dict[str, object] = {}
    t_975_df2 = 4.303  # predeclared two-sided 95% t multiplier for n=3
    for arm, seed_rows in by_arm_seed.items():
        seed_means: list[Mapping[str, object]] = []
        for seed_index, rows in sorted(seed_rows.items()):
            shell = statistics.mean(float(r["holdout_shell_rms"]) for r in rows)
            central = statistics.mean(float(r["holdout_central_rms"]) for r in rows)
            gap = statistics.mean(float(r["log_z_h_minus_log_z_t"]) for r in rows)
            seed_means.append(
                {
                    "seed_index": seed_index,
                    "holdout_shell_rms_mean": shell,
                    "holdout_central_rms_mean": central,
                    "log_gap_mean": gap,
                    "all_steps_valid": all(bool(r["hard_valid"]) for r in rows),
                }
            )
        shell_values = [float(r["holdout_shell_rms_mean"]) for r in seed_means]
        central_values = [float(r["holdout_central_rms_mean"]) for r in seed_means]
        gap_values = [float(r["log_gap_mean"]) for r in seed_means]
        def stats(values: list[float]) -> Mapping[str, float]:
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values))
            half = t_975_df2 * se
            return {
                "mean": mean,
                "sd": sd,
                "standard_error": se,
                "t95_low": mean - half,
                "t95_high": mean + half,
            }
        summary[arm] = {
            "seed_means": seed_means,
            "holdout_shell_rms": stats(shell_values),
            "holdout_central_rms": stats(central_values),
            "log_gap": stats(gap_values),
            "seed_count": len(seed_means),
            "t_multiplier": t_975_df2,
        }
    return summary


def _result_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# C2 Mixture-UKF/APF Phase 5C-R Replication Result",
        "",
        f"Status: `{payload['status']}`  ",
        f"Continuation: `{payload['continuation']}`  ",
        f"Failure class: `{payload['failure_class']}`",
        "",
        "## Decision",
        "",
        "| Decision | Criterion | Status | Interpretation |",
        "| --- | --- | --- | --- |",
        f"| Exact recursive target/map | all seed/step fixture, map, and target checks | {'PASS' if payload['checks']['global_validity'] else 'VETO'} | {'records interpretable' if payload['checks']['global_validity'] else 'repair target/map'} |",
        f"| Hybrid endpoint | 12-channel wiring and at least one valid arm per seed/step | {'PASS' if payload['checks']['hard_vetoes_pass'] else 'VETO'} | {'candidate arms remain eligible for longer validation' if payload['checks']['hard_vetoes_pass'] else 'continuation blocked'} |",
        "| Ranking/promotion | no promotion criterion in this phase | NOT TESTED | intervals are descriptive and small-sample |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        f"| Hard veto screen | {'passed' if payload['checks']['hard_vetoes_pass'] else 'failed'} |",
        "| Statistically supported ranking | none; three seed clusters and two steps are underpowered for a width ranking |",
        "| Descriptive differences | per-seed/per-step records and t intervals below |",
        "| Default readiness | not assessed |",
        "| Next evidence | longer replicated recursive run or integrated proposal test under a new plan |",
        "",
        "## Per-seed and per-step records",
        "",
        "| seed | step | arm | holdout RMS | shell RMS | log ZH-ZT | map cond. | map min eig. | observation ESS | valid |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in payload["records"]:
        lines.append(
            f"| {row['seed_index']} | {row['step']} | {row['arm']} | {row['holdout_rms']:.5g} | {row['holdout_shell_rms']:.5g} | {row['log_z_h_minus_log_z_t']:.5g} | {row['map_condition_number']:.5g} | {row['map_minimum_eigenvalue']:.5g} | {row['observation_ess']:.5g} | {row['hard_valid']} |"
        )
    lines += ["", "## Seed-cluster uncertainty (descriptive)", "", "| arm | metric | mean | SD | SE | t95 low | t95 high |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, summary in payload["uncertainty"].items():
        for metric, values in (("shell RMS", summary["holdout_shell_rms"]), ("central RMS", summary["holdout_central_rms"]), ("log gap", summary["log_gap"])):
            lines.append(f"| {arm} | {metric} | {values['mean']:.5g} | {values['sd']:.5g} | {values['standard_error']:.5g} | {values['t95_low']:.5g} | {values['t95_high']:.5g} |")
    lines += [
        "",
        "The t intervals use three seed-cluster means and are uncertainty summaries, not evidence of superiority. The independent predictive-mixture normalizer is the target comparator at every step.",
        "",
        "## Red team",
        "",
        "The strongest alternative explanation is finite-cloud noise or a map-induced representation effect shared across arms. The exact target wiring, disjoint streams, and per-step map records address wiring risk, but the short horizon and three seed clusters cannot establish recursive posterior quality or a general-model result.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if int(args.horizon) < 1 or int(args.horizon) > 4:
        raise ValueError("--horizon must lie in [1, 4]")
    output = _make_output_root(args.output_root)
    started = time.perf_counter()
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    p5c = _load_phase5c()
    p5b = p5c._load_phase5b()
    runtime = dict(p5b._configure_runtime())
    runtime["execution_lane"] = "trusted_gpu_xla_phase5c_replication"
    p5c._load_hybrid_basis()
    p5a = p5b._load_phase5a()
    fixture = p5a._load_fixture()
    all_records: list[Mapping[str, object]] = []
    stream_hashes: dict[str, str] = {}
    global_validity = True
    endpoint_wired = True
    per_seed_step_valid: dict[str, bool] = {}
    # Both kernels have fixed signatures.  Parent clouds and observations are
    # inputs, so recursive calls do not create a new tf.function per step.
    reference_transition = tf.convert_to_tensor(fixture["transition_matrix"], DTYPE)
    reference_theta = tf.convert_to_tensor(
        [float(fixture["gamma"]), math.log(float(fixture["beta"]))], DTYPE
    )
    target_kernels_by_rows = {
        rows: _make_recursive_target_kernel(
            reference_transition,
            reference_theta,
            float(fixture["sigma"]),
            rows,
            jit_compile=bool(args.jit_compile),
        )
        for rows in (TRAIN_ROWS, HOLDOUT_ROWS, AUDIT_ROWS)
    }
    target_kernels = {
        "train": target_kernels_by_rows[TRAIN_ROWS],
        "holdout": target_kernels_by_rows[HOLDOUT_ROWS],
        "audit": target_kernels_by_rows[AUDIT_ROWS],
    }
    moment_kernel = p5a.make_weighted_transition_moment_kernel(
        particle_count=CARRIED_ROWS,
        state_dim=STATE_DIM,
        jit_compile=bool(args.jit_compile),
    )

    for seed_index, (seed, map_seed) in enumerate(zip(SEED_FAMILIES, MAP_SEED_FAMILIES)):
        initial = _initial_payload(p5a, fixture, map_seed, jit_compile=bool(args.jit_compile))
        global_validity = global_validity and bool(
            initial["transition_fixture_error"] <= 1.0e-12
            and initial["process_fixture_error"] <= 1.0e-12
            and initial["stationary_fixture_error"] <= 1.0e-12
            and _finite(initial["predicted_mean"])
            and _finite(initial["predicted_covariance"])
            and _scalar(initial["map_minimum_eigenvalue"]) > 0.0
        )
        model = initial["model"]
        theta = initial["theta"]
        transition = initial["transition"]
        process = initial["process"]
        observations = initial["observations"]
        states = initial["states"]
        weights = initial["weights"]
        recursive_bank = p5a._standard_normal_bank(
            CARRIED_ROWS, _derived_seed(seed, 0, 15)
        )
        stream_hashes[f"seed{seed_index}_recursive_cloud"] = hashlib.sha256(
            tf.io.serialize_tensor(recursive_bank).numpy()
        ).hexdigest()
        previous_map = None
        for step in range(1, int(args.horizon) + 1):
            conditional_means = tf.linalg.matmul(states, transition, transpose_b=True)
            conditional_covariances = tf.broadcast_to(
                process[None, :, :], [CARRIED_ROWS, STATE_DIM, STATE_DIM]
            )
            built = p5a.build_lagged_moment_map(
                weights,
                conditional_means,
                conditional_covariances,
                jit_compile=bool(args.jit_compile),
                kernel=moment_kernel,
            )
            coordinate_map = built["coordinate_map"]
            global_validity = global_validity and bool(built["valid"].numpy())
            map_metrics = _map_metrics(coordinate_map, recursive_bank)
            global_validity = global_validity and bool(
                map_metrics["physical_finite"]
                and map_metrics["reference_finite"]
                and map_metrics["roundtrip_max_abs"] <= 2.0e-12
                and math.isfinite(float(map_metrics["condition_number"]))
                and float(map_metrics["minimum_cholesky_diagonal"]) > 0.0
            )
            map_shift = {"offset_max_abs": 0.0, "matrix_max_abs": 0.0, "max_abs": 0.0}
            if previous_map is not None:
                map_shift = {
                    "offset_max_abs": _max_abs(coordinate_map.offset - previous_map.offset),
                    "matrix_max_abs": _max_abs(coordinate_map.matrix - previous_map.matrix),
                    "max_abs": max(
                        _max_abs(coordinate_map.offset - previous_map.offset),
                        _max_abs(coordinate_map.matrix - previous_map.matrix),
                    ),
                }
            banks = {
                "train": p5a._standard_normal_bank(TRAIN_ROWS, _derived_seed(seed, step, 11)),
                "holdout": p5a._standard_normal_bank(HOLDOUT_ROWS, _derived_seed(seed, step, 12)),
                "audit": p5a._standard_normal_bank(AUDIT_ROWS, _derived_seed(seed, step, 13)),
            }
            for name, rows in banks.items():
                stream_hashes[f"seed{seed_index}_step{step}_{name}"] = hashlib.sha256(
                    tf.io.serialize_tensor(rows).numpy()
                ).hexdigest()
            shared = dict(initial)
            shared.update(
                {
                    "states": states,
                    "weights": weights,
                    "coordinate_map": coordinate_map,
                    "transition": transition,
                    "process": process,
                    "theta": theta,
                    "model": model,
                    "observations": observations,
                    "seed": seed,
                    "map_seed": map_seed,
                    "map_condition_number": built["condition_number"],
                    "map_minimum_eigenvalue": built["minimum_eigenvalue"],
                }
            )
            shift, log_targets, sqrt_targets, predictive = _step_target(
                p5a=p5a,
                fixture=fixture,
                shared=shared,
                parent_states=states,
                parent_weights=weights,
                observation=observations[step],
                banks=banks,
                coordinate_map=coordinate_map,
                target_kernels=target_kernels,
                predictive_seed=_derived_seed(seed, step, 14),
            )
            predictive_finite = _finite(predictive["likelihood_values"])
            global_validity = global_validity and predictive_finite
            predicted_states, _ = coordinate_map.forward(recursive_bank)
            log_observation = model.observation_log_density(
                theta, predicted_states, observations[step], step
            )
            posterior_weights = tf.exp(log_observation - tf.reduce_logsumexp(log_observation))
            observation_ess = _scalar(
                tf.math.reciprocal(tf.reduce_sum(tf.square(posterior_weights)))
            )
            posterior_finite = _finite(predicted_states) and _finite(posterior_weights)
            global_validity = global_validity and posterior_finite
            step_valid = False
            for width in ARM_WIDTHS:
                arm_name = f"hybrid_d6_w{int(round(width * 100)):03d}"
                arm = _fit_step(
                    p5a=p5a,
                    p5b=p5b,
                    p5c=p5c,
                    name=arm_name,
                    width=width,
                    shared=shared,
                    banks=banks,
                    shift=shift,
                    log_targets=log_targets,
                    sqrt_targets=sqrt_targets,
                    predictive=predictive,
                )
                endpoint_wired = endpoint_wired and bool(
                    arm["basis_family"] == "gaussian_hermite_rbf_reference"
                    and int(arm["basis_dim_per_axis"]) == HERMITE_DEGREE + 1 + len(CENTERS)
                    and bool(arm["hermite_constant"])
                    and not bool(arm["rbf_constant"])
                )
                step_valid = step_valid or bool(arm["hard_valid"])
                all_records.append(
                    {
                        "seed_index": seed_index,
                        "seed": seed,
                        "map_seed": map_seed,
                        "step": step,
                        "arm": arm["arm"],
                        "width": arm["width"],
                        "holdout_rms": arm["holdout_rms"],
                        "holdout_central_rms": arm["holdout_central_rms"],
                        "holdout_shell_rms": arm["holdout_shell_rms"],
                        "log_z_h_minus_log_z_t": arm["log_z_h_minus_log_z_t"],
                        "z_h": arm["z_h"],
                        "z_t": arm["z_t_direct"],
                        "z_t_standard_error": arm["z_t_direct_standard_error"],
                        "mass_condition_number": arm["contractions"]["mass_condition_number"],
                        "fit_condition_max": arm["condition"]["scaled_augmented_condition_max"],
                        "hard_valid": arm["hard_valid"],
                        "fit_status": arm["fit_status"],
                        "map_condition_number": _scalar(built["condition_number"]),
                        "map_minimum_eigenvalue": _scalar(built["minimum_eigenvalue"]),
                        "map_roundtrip_max_abs": map_metrics["roundtrip_max_abs"],
                        "map_shift_max_abs": map_shift["max_abs"],
                        "observation_ess": observation_ess,
                        "predictive_finite": predictive_finite,
                    }
                )
            per_seed_step_valid[f"seed{seed_index}_step{step}"] = step_valid
            global_validity = global_validity and step_valid
            weights = posterior_weights
            states = predicted_states
            previous_map = coordinate_map
            global_validity = global_validity and _finite(weights) and _finite(states)

    checks = {
        "global_validity": bool(global_validity),
        "endpoint_wired": bool(endpoint_wired),
        "all_seed_step_have_valid_arm": all(per_seed_step_valid.values()),
        "records_complete": len(all_records) == len(SEED_FAMILIES) * int(args.horizon) * len(ARM_WIDTHS),
        "source_files_present": all(
            path.is_file()
            for path in (
                PLAN_PATH,
                PHASE5C_DRIVER_PATH,
                PHASE5A_DRIVER_PATH,
                PHASE5B_DRIVER_PATH,
                HYBRID_MODULE_PATH,
                FIXTURE_PATH,
                MODEL_PATH,
                MOMENT_MAP_PATH,
            )
        ),
    }
    checks["hard_vetoes_pass"] = all(
        bool(checks[key])
        for key in (
            "global_validity",
            "endpoint_wired",
            "all_seed_step_have_valid_arm",
            "records_complete",
            "source_files_present",
        )
    )
    uncertainty = _summarize_uncertainty(all_records)
    elapsed = time.perf_counter() - started
    source_paths = {
        "plan": PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "phase5c_driver": PHASE5C_DRIVER_PATH,
        "phase5a_driver": PHASE5A_DRIVER_PATH,
        "phase5b_driver": PHASE5B_DRIVER_PATH,
        "hybrid_basis": HYBRID_MODULE_PATH,
        "fixture": FIXTURE_PATH,
        "model": MODEL_PATH,
        "moment_map": MOMENT_MAP_PATH,
    }
    sources = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    failure_class = "candidate_failure" if checks["hard_vetoes_pass"] and not all(
        bool(row["hard_valid"]) for row in all_records
    ) else ("none" if checks["hard_vetoes_pass"] else "implementation_or_numerical_validity")
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": "PASS_PHASE5C_REPLICATION" if checks["hard_vetoes_pass"] else "VETO_PHASE5C_REPLICATION",
        "continuation": "CONTINUE_NEXT_INTEGRATED_PHASE" if checks["hard_vetoes_pass"] else "CONTINUATION_VETO_PHASE5C_REPLICATION",
        "failure_class": failure_class,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH or "unset"),
            "jit_compile": bool(args.jit_compile),
            "runtime": runtime,
        },
        "seed_families": SEED_FAMILIES,
        "map_seed_families": MAP_SEED_FAMILIES,
        "horizon": int(args.horizon),
        "carried_rows": CARRIED_ROWS,
        "train_rows": TRAIN_ROWS,
        "holdout_rows": HOLDOUT_ROWS,
        "audit_rows": AUDIT_ROWS,
        "hermite_degree": HERMITE_DEGREE,
        "centers": CENTERS,
        "widths": ARM_WIDTHS,
        "quadrature_orders": QUADRATURE_ORDERS,
        "tt_rank": TT_RANK,
        "als_sweeps": ALS_SWEEPS,
        "ridge": RIDGE,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "sources": sources,
        "workspace": _workspace_manifest(),
        "checks": checks,
        "per_seed_step_valid": per_seed_step_valid,
        "stream_hashes": stream_hashes,
        "records": all_records,
        "uncertainty": uncertainty,
        "nonclaims": [
            "no proposal-efficiency or ESS promotion claim",
            "no posterior-correctness or pseudo-marginal claim",
            "no statistical arm ranking or width promotion claim",
            "no analytical total-gradient or HMC claim",
            "no production/default-readiness claim",
        ],
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": sources["plan"]["sha256"],
        "sources": sources,
        "seed_families": SEED_FAMILIES,
        "map_seed_families": MAP_SEED_FAMILIES,
        "horizon": int(args.horizon),
        "stream_hashes": stream_hashes,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "environment": payload["environment"],
        "workspace": payload["workspace"],
        "evidence_contract": {
            "primary": "three-seed two-step exact-target hybrid representation validation",
            "hard_vetoes": "target/map/bank mismatch, no valid arm at a seed/step, nonfinite/corrupt records, endpoint wiring mismatch",
            "descriptive": "per-step residuals, direct normalizer gaps, conditions, ESS, and seed-cluster t intervals",
            "nonclaims": payload["nonclaims"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "result.json", payload)
    _write_json(output / "records.json", {"records": all_records, "uncertainty": uncertainty})
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    return output


def _parse_args_and_run() -> int:
    args = _parse_args()
    output = run(args)
    print(json.dumps({"phase": PHASE_ID, "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_parse_args_and_run())
