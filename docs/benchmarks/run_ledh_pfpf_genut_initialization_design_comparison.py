#!/usr/bin/env python3
"""CPU/XLA comparison of Gaussian cloud designs for LEDH-PFPF-GenUT.

Only the standardized Gaussian clouds change across arms. The LEDH flow, exact
PFPF correction, GenUT/Contract-E primal reset, and repository standard
backward filtering-score recursion are shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.gaussian_cloud_designs_tf import (
    SUPPORTED_DESIGNS,
    cloud_diagnostics,
    standard_normal_cloud,
)
from bayesfilter.highdim.ledh_pfpf_genut_initialization_tf import (
    InitializationModelSpec,
    diagonal_lgssm_spec,
    finite_value_standard_score_ledh_pfpf_genut,
    reduced_sir_spec,
)


SCHEMA_VERSION = "bayesfilter.ledh_pfpf_genut.initialization_design_comparison.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ledh-pfpf-genut-initialization-design-comparison-plan-2026-08-03.md"
)
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/ledh_pfpf_genut_initialization_20260803"
)
PARTICLE_COUNT = 48
HORIZON = 6
RANDOMIZATION_SEEDS = tuple(range(93801, 93809))
SCOPES = ("initial_only", "all_innovations")
BASELINE = "iid_gaussian"
EPSILON = 2.0
SINKHORN_STEPS = 8
BALANCE_STEPS = 8
RIDGE = 1.0e-5
BOOTSTRAP_REPLICATES = 4096
LGSSM_DGP_SEED = 81100
SIR_DGP_SEED = 97001
LGSSM_THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
SIR_THETA = (0.0, 0.0, 0.0)
PARAMETER_NAMES = {
    "diagonal_lgssm": ("phi1", "phi2", "phi3", "q_scale", "r_scale"),
    "reduced_sir": (
        "log_kappa_scale",
        "log_nu_scale",
        "log_observation_noise_scale",
    ),
}
SOURCE_PATHS = (
    PLAN,
    Path("bayesfilter/highdim/gaussian_cloud_designs_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_initialization_tf.py"),
    Path("bayesfilter/highdim/genut_guided_proposal_tf.py"),
    Path("bayesfilter/highdim/sir_online_score_teacher_tf.py"),
    Path("bayesfilter/highdim/sir_latent_preclip_reference_tf.py"),
    Path("bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py"),
    Path("docs/benchmarks/run_lgssm_cubature_genut_fp32.py"),
    Path("tests/highdim/test_ledh_pfpf_genut_initialization_designs.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_initialization_design_comparison.py"),
)
ERROR_METRICS = (
    "absolute_value_error",
    "squared_value_error",
    "score_l2_error",
    "squared_score_l2_error",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(serialized).hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_cpu_reference() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES=-1 before importing TensorFlow for this CPU run"
        )
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-reference run found a visible physical GPU")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tf.config.list_logical_devices("CPU")
    if not logical:
        raise RuntimeError("CPU-reference run found no logical CPU")
    return {
        "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        "logical_devices": [device.name for device in logical],
        "cuda_visible_devices": "-1",
        "device": "/device:CPU:0",
        "dtype": "float32",
        "jit_compile": True,
        "xla_platform": "Host",
        "tf32_execution_enabled": False,
        "trust_basis": "cpu_reference_gpu_intentionally_hidden",
        "production_target_status": "explicit_cpu_xla_reference_exception",
    }


def _next_output_directory(*, smoke: bool) -> Path:
    root = ROOT / ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_attempt" if smoke else "attempt"
    for attempt in range(1, 100):
        candidate = root / f"{prefix}{attempt:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("no unused output directory remains")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _lgssm_observations() -> tf.Tensor:
    theta = tf.constant(LGSSM_THETA, tf.float32)
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    observation_matrix = tf.constant(
        ((1.0, 0.25, -0.15), (0.2, 1.1, 0.3), (-0.1, 0.35, 0.9)),
        tf.float32,
    )
    state = (
        q_scale
        * tf.random.stateless_normal(
            [3], [LGSSM_DGP_SEED, 1], dtype=tf.float32
        )
        / tf.sqrt(1.0 - tf.square(phi))
    )
    process = tf.random.stateless_normal(
        [HORIZON, 3], [LGSSM_DGP_SEED, 2], dtype=tf.float32
    )
    observation_noise = tf.random.stateless_normal(
        [HORIZON, 3], [LGSSM_DGP_SEED, 3], dtype=tf.float32
    )
    values = []
    for time_index in range(HORIZON):
        state = phi * state + q_scale * process[time_index]
        values.append(
            tf.linalg.matvec(observation_matrix, state)
            + r_scale * observation_noise[time_index]
        )
    return tf.stack(values)


def _sir_context() -> tuple[Any, Any, tf.Tensor]:
    from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
        reduced_latent_preclip_sir_model,
    )
    from bayesfilter.highdim.sir_online_score_teacher_tf import static_spec_from_model

    model = reduced_latent_preclip_sir_model()
    theta = tf.constant(SIR_THETA, tf.float64)
    simulation = model.simulate_from_standard_normals(
        theta,
        tf.random.stateless_normal(
            [2], [SIR_DGP_SEED, 1], dtype=tf.float64
        ),
        tf.random.stateless_normal(
            [HORIZON - 1, 2], [SIR_DGP_SEED, 2], dtype=tf.float64
        ),
        tf.random.stateless_normal(
            [HORIZON, 1], [SIR_DGP_SEED, 3], dtype=tf.float64
        ),
    )
    return model, static_spec_from_model(model), tf.cast(
        simulation["observations"], tf.float32
    )


def _lgssm_reference(observations: tf.Tensor) -> dict[str, Any]:
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    started = time.perf_counter()
    value, score = _kalman_value_score(
        tf.constant(LGSSM_THETA, tf.float32), observations
    )
    return {
        "reference_id": "exact_affine_kalman_analytical_physical_parameter_score",
        "role": "exact_same_target_accuracy_reference",
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "finite": bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "dtype": "float64",
        "elapsed_seconds": time.perf_counter() - started,
        "boundary_mass_veto": None,
    }


def _sir_reference(model: Any, observations: tf.Tensor) -> dict[str, Any]:
    from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
        REFERENCE_ID,
        dense_latent_sir_value_and_manual_score,
        prepare_reduced_dense_grids,
    )

    started = time.perf_counter()
    theta = tf.constant(SIR_THETA, tf.float64)
    grids = prepare_reduced_dense_grids(
        model,
        theta,
        time_steps=HORIZON - 1,
        order=29,
        radius=7.0,
        integration_rule="split_gauss_legendre",
    )
    result = dense_latent_sir_value_and_manual_score(
        model, theta, tf.cast(observations, tf.float64), grids
    )
    boundary_mass = float(tf.reduce_max(result["boundary_mass_history"]).numpy())
    value = tf.convert_to_tensor(result["objective"])
    score = tf.convert_to_tensor(result["score"])
    return {
        "reference_id": REFERENCE_ID,
        "role": "independent_deterministic_approximate_same_target_accuracy_anchor",
        "score_definition": "standard_normalized_filtering_score",
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "finite": bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "dtype": "float64",
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            "order": 29,
            "radius": 7.0,
            "integration_rule": "split_gauss_legendre",
        },
        "maximum_boundary_mass": boundary_mass,
        "boundary_mass_veto": boundary_mass > 1.0e-8,
        "exact": False,
    }


def _make_evaluator(
    spec: InitializationModelSpec, *, sir_static_spec: Any = None
) -> Callable[..., Any]:
    process_steps = HORIZON if spec.transition_before_first_observation else HORIZON - 1

    @tf.function(
        input_signature=(
            tf.TensorSpec([spec.parameter_count], tf.float32),
            tf.TensorSpec([HORIZON, spec.observation_dimension], tf.float32),
            tf.TensorSpec([PARTICLE_COUNT, spec.state_dimension], tf.float32),
            tf.TensorSpec(
                [process_steps, PARTICLE_COUNT, spec.state_dimension], tf.float32
            ),
            tf.TensorSpec([PARTICLE_COUNT, spec.state_dimension], tf.float32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def evaluate(theta, observations, initial_noise, process_noise, design):
        with tf.device("/CPU:0"):
            return finite_value_standard_score_ledh_pfpf_genut(
                spec,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                sir_static_spec=sir_static_spec,
                epsilon=EPSILON,
                sinkhorn_steps=SINKHORN_STEPS,
                balance_steps=BALANCE_STEPS,
                ridge=RIDGE,
            )

    return evaluate


def _xla_contract(evaluator: Callable[..., Any]) -> dict[str, Any]:
    concrete = evaluator.get_concrete_function()
    must_compile = concrete.function_def.attr.get("_XlaMustCompile")
    return {
        "jit_compile_requested": True,
        "xla_must_compile_attribute": bool(must_compile.b) if must_compile else None,
        "tracing_count": int(evaluator.experimental_get_tracing_count()),
    }


def _clouds(
    spec: InitializationModelSpec, design_id: str, scope: str, seed: int
) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope}")
    dimension = spec.state_dimension
    process_steps = HORIZON if spec.transition_before_first_observation else HORIZON - 1
    model_salt = 10_000 if spec.family == "diagonal_lgssm" else 20_000
    started = time.perf_counter()
    initial = standard_normal_cloud(
        design_id,
        num_particles=PARTICLE_COUNT,
        dimension=dimension,
        seed=seed,
        salt=model_salt + 101,
    )
    process_design = design_id if scope == "all_innovations" else BASELINE
    process_rows = [
        standard_normal_cloud(
            process_design,
            num_particles=PARTICLE_COUNT,
            dimension=dimension,
            seed=seed,
            salt=model_salt + 1000 + time_index,
        )
        for time_index in range(process_steps)
    ]
    process = tf.stack(process_rows)
    initial_diagnostics = cloud_diagnostics(initial)
    process_diagnostics = [cloud_diagnostics(cloud) for cloud in process_rows]
    diagnostics = {
        "initial": {
            key: float(value.numpy()) for key, value in initial_diagnostics.items()
        },
        "process_design": process_design,
        "maximum_process_absolute_mean": max(
            float(row["maximum_absolute_mean"].numpy())
            for row in process_diagnostics
        ),
        "maximum_process_covariance_frobenius_error": max(
            float(row["covariance_frobenius_error"].numpy())
            for row in process_diagnostics
        ),
        "maximum_process_absolute_value": max(
            float(row["maximum_absolute_value"].numpy())
            for row in process_diagnostics
        ),
        "generation_seconds": time.perf_counter() - started,
    }
    return initial, process, diagnostics


def _error_fields(
    value: tf.Tensor,
    score: tf.Tensor,
    reference: dict[str, Any],
) -> dict[str, Any]:
    value64 = tf.cast(value, tf.float64)
    score64 = tf.cast(score, tf.float64)
    reference_value = tf.constant(reference["value"], tf.float64)
    reference_score = tf.constant(reference["score"], tf.float64)
    value_error = value64 - reference_value
    score_error = score64 - reference_score
    score_l2 = tf.linalg.norm(score_error)
    return {
        "value_error": float(value_error.numpy()),
        "absolute_value_error": float(tf.abs(value_error).numpy()),
        "squared_value_error": float(tf.square(value_error).numpy()),
        "score_error": [float(item) for item in score_error.numpy()],
        "score_l2_error": float(score_l2.numpy()),
        "squared_score_l2_error": float(tf.square(score_l2).numpy()),
    }


def _run_row(
    *,
    spec: InitializationModelSpec,
    theta: tf.Tensor,
    observations: tf.Tensor,
    evaluator: Callable[..., Any],
    reset_design: tf.Tensor,
    reference: dict[str, Any],
    design_id: str,
    scope: str,
    seed: int,
) -> dict[str, Any]:
    initial, process, cloud_info = _clouds(spec, design_id, scope, seed)
    started = time.perf_counter()
    value, score, diagnostics = evaluator(
        theta, observations, initial, process, reset_design
    )
    elapsed = time.perf_counter() - started
    valid = bool(diagnostics["program_valid"].numpy())
    finite = bool(tf.math.is_finite(value).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    if not valid or not finite:
        raise RuntimeError(
            f"candidate invalid for {spec.family}/{scope}/{design_id}/seed={seed}"
        )
    row = {
        "model": spec.family,
        "model_id": spec.model_id,
        "scope": scope,
        "design": design_id,
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "program_valid": valid,
        "finite": finite,
        "filter_seconds": elapsed,
        "value_device": value.device,
        "score_device": score.device,
        "minimum_ess": float(tf.reduce_min(diagnostics["ess"]).numpy()),
        "maximum_normalized_weight": float(
            tf.reduce_max(diagnostics["maximum_normalized_weight"]).numpy()
        ),
        "maximum_reset_mean_residual": float(
            tf.reduce_max(diagnostics["reset_mean_residual"]).numpy()
        ),
        "cloud": cloud_info,
    }
    row.update(_error_fields(value, score, reference))
    return row


def _summary(values: list[float]) -> dict[str, Any]:
    tensor = tf.constant(values, tf.float64)
    count = len(values)
    mean = tf.reduce_mean(tensor)
    standard_deviation = (
        tf.math.reduce_std(tensor) * tf.sqrt(tf.cast(count, tf.float64) / (count - 1))
        if count > 1
        else tf.zeros([], tf.float64)
    )
    return {
        "count": count,
        "mean": float(mean.numpy()),
        "sample_standard_deviation": float(standard_deviation.numpy()),
        "minimum": float(tf.reduce_min(tensor).numpy()),
        "maximum": float(tf.reduce_max(tensor).numpy()),
    }


def _bootstrap_seed(label: str) -> tf.Tensor:
    digest = hashlib.sha256(label.encode("ascii")).digest()
    first = int.from_bytes(digest[:4], "little") % 2_147_483_647
    second = int.from_bytes(digest[4:8], "little") % 2_147_483_647
    return tf.constant([first, second], tf.int32)


def _paired_bootstrap(differences: list[float], *, label: str) -> dict[str, Any]:
    values = tf.constant(differences, tf.float64)
    count = len(differences)
    indices = tf.random.stateless_uniform(
        [BOOTSTRAP_REPLICATES, count],
        _bootstrap_seed(label),
        minval=0,
        maxval=count,
        dtype=tf.int32,
    )
    bootstrap_means = tf.sort(
        tf.reduce_mean(tf.gather(values, indices), axis=1)
    )
    lower_index = int(math.floor(0.025 * (BOOTSTRAP_REPLICATES - 1)))
    upper_index = int(math.ceil(0.975 * (BOOTSTRAP_REPLICATES - 1)))
    lower = float(bootstrap_means[lower_index].numpy())
    upper = float(bootstrap_means[upper_index].numpy())
    if upper < 0.0:
        signal = "candidate_lower_error_pilot_signal"
    elif lower > 0.0:
        signal = "candidate_higher_error_pilot_signal"
    else:
        signal = "inconclusive_interval_contains_zero"
    return {
        "paired_count": count,
        "mean_candidate_minus_comparator": float(tf.reduce_mean(values).numpy()),
        "bootstrap_percentile_95_interval": [lower, upper],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "signal": signal,
        "inference_limit": (
            "pilot_pairwise_signal_only; eight pairs and unadjusted multiple "
            "comparisons do not establish a ranking"
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cell_summaries = []
    paired = []
    scope_contrasts = []
    for model in sorted({row["model"] for row in rows}):
        for scope in SCOPES:
            for design_id in SUPPORTED_DESIGNS:
                cell = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["scope"] == scope
                    and row["design"] == design_id
                ]
                if not cell:
                    continue
                cell_summaries.append(
                    {
                        "model": model,
                        "scope": scope,
                        "design": design_id,
                        "metrics": {
                            metric: _summary([float(row[metric]) for row in cell])
                            for metric in ERROR_METRICS
                        },
                        "filter_seconds": _summary(
                            [float(row["filter_seconds"]) for row in cell]
                        ),
                        "minimum_ess": _summary(
                            [float(row["minimum_ess"]) for row in cell]
                        ),
                    }
                )
                if design_id == BASELINE:
                    continue
                candidate = {int(row["seed"]): row for row in cell}
                baseline = {
                    int(row["seed"]): row
                    for row in rows
                    if row["model"] == model
                    and row["scope"] == scope
                    and row["design"] == BASELINE
                }
                if set(candidate) != set(baseline):
                    raise RuntimeError("candidate/IID pairing is incomplete")
                paired.append(
                    {
                        "model": model,
                        "scope": scope,
                        "candidate": design_id,
                        "comparator": BASELINE,
                        "metrics": {
                            metric: _paired_bootstrap(
                                [
                                    float(candidate[seed][metric])
                                    - float(baseline[seed][metric])
                                    for seed in sorted(candidate)
                                ],
                                label=f"{model}/{scope}/{design_id}/{metric}/iid",
                            )
                            for metric in ERROR_METRICS
                        },
                    }
                )

        for design_id in SUPPORTED_DESIGNS:
            initial = {
                int(row["seed"]): row
                for row in rows
                if row["model"] == model
                and row["scope"] == "initial_only"
                and row["design"] == design_id
            }
            all_innovations = {
                int(row["seed"]): row
                for row in rows
                if row["model"] == model
                and row["scope"] == "all_innovations"
                and row["design"] == design_id
            }
            if not initial and not all_innovations:
                continue
            if set(initial) != set(all_innovations):
                raise RuntimeError("initial/all-innovations pairing is incomplete")
            scope_contrasts.append(
                {
                    "model": model,
                    "design": design_id,
                    "candidate": "all_innovations",
                    "comparator": "initial_only",
                    "metrics": {
                        metric: _paired_bootstrap(
                            [
                                float(all_innovations[seed][metric])
                                - float(initial[seed][metric])
                                for seed in sorted(initial)
                            ],
                            label=f"{model}/{design_id}/{metric}/scope",
                        )
                        for metric in ERROR_METRICS
                    },
                }
            )
    return cell_summaries, paired, scope_contrasts


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LEDH-PFPF-GenUT Initialization Design Comparison",
        "",
        f"Status: `{result['status']}`.",
        "",
        (
            "All arm differences are a bounded CPU/XLA pilot. Unadjusted paired "
            "bootstrap intervals are signals, not a statistically supported universal ranking."
        ),
        "",
        "| Model | Scope | Design | Mean abs value error | Mean score L2 error | Mean minimum ESS |",
        "|---|---|---|---:|---:|---:|",
    ]
    for cell in result["cell_summaries"]:
        lines.append(
            f"| {cell['model']} | {cell['scope']} | {cell['design']} | "
            f"{cell['metrics']['absolute_value_error']['mean']:.6g} | "
            f"{cell['metrics']['score_l2_error']['mean']:.6g} | "
            f"{cell['minimum_ess']['mean']:.6g} |"
        )
    lines += [
        "",
        "## Inference Status",
        "",
        "| Question | Verdict |",
        "|---|---|",
        f"| Hard veto screen | `{result['inference_status']['hard_veto_screen']}` |",
        "| Statistically supported ranking | None; eight paired randomizations and unadjusted multiple comparisons |",
        "| Descriptive-only differences | Cell means, tails, runtime, ESS, and individual errors |",
        "| Default readiness | Not evaluated; reset controls were frozen rather than scope-tuned |",
        "| Next evidence needed | Model-specific tuning followed by more independent paired replications and multiplicity-aware confirmatory intervals |",
        "",
        "## Decision",
        "",
        f"- Primary criterion: `{result['decision']['primary_criterion_status']}`",
        f"- Veto diagnostics: `{result['decision']['veto_diagnostic_status']}`",
        f"- Main uncertainty: {result['decision']['main_uncertainty']}",
        f"- Next justified action: {result['decision']['next_justified_action']}",
        f"- Not concluded: {result['decision']['not_concluded']}",
        "",
        "Raw rows and paired intervals are preserved in `raw.json` and `result.json`.",
    ]
    return "\n".join(lines) + "\n"


def _manifest(
    *,
    output_directory: Path,
    device_policy: dict[str, Any],
    started_at: float,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "device_policy": device_policy,
        "smoke": smoke,
        "particle_count": PARTICLE_COUNT,
        "horizon": HORIZON,
        "randomization_seeds": list(RANDOMIZATION_SEEDS[:1] if smoke else RANDOMIZATION_SEEDS),
        "data": {
            "lgssm": {
                "dataset_id": "fixed_simulated_diagonal_lgssm_transition_first_v1",
                "dgp_seed": LGSSM_DGP_SEED,
            },
            "reduced_sir": {
                "dataset_id": "reduced_continuous_preclip_sir_j1_fixed_simulation_v1",
                "dgp_seed": SIR_DGP_SEED,
            },
        },
        "designs": list(SUPPORTED_DESIGNS),
        "scopes": list(SCOPES),
        "filter_controls": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": BALANCE_STEPS,
            "ridge": RIDGE,
            "reset_status": "experimental_contract_e_chol_primal_not_canonical_identity",
        },
        "score_backend": {
            "definition": "repository_standard_pairwise_backward_filtering_score",
            "autodiff": False,
            "handwritten_in_experiment": False,
            "lgssm_local_score_dtype": "float32",
            "sir_existing_local_score_provider_internal_dtype": "float64",
            "reported_candidate_score_dtype": "float32",
        },
        "plan": str(PLAN),
        "output_directory": str(output_directory.relative_to(ROOT)),
        "source_sha256": {str(path): _sha256(path) for path in SOURCE_PATHS},
        "started_unix_seconds": started_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one LGSSM seed across every design and scope",
    )
    args = parser.parse_args()
    started_at = time.time()
    campaign_started = time.perf_counter()
    device_policy = _configure_cpu_reference()
    output_directory = _next_output_directory(smoke=args.smoke)
    manifest = _manifest(
        output_directory=output_directory,
        device_policy=device_policy,
        started_at=started_at,
        smoke=args.smoke,
    )
    _write_json(output_directory / "run_manifest.json", manifest)

    lgssm_observations = _lgssm_observations()
    sir_model, sir_static_spec, sir_observations = _sir_context()
    manifest["data"]["lgssm"]["observation_sha256"] = _tensor_sha256(
        lgssm_observations
    )
    manifest["data"]["reduced_sir"]["observation_sha256"] = _tensor_sha256(
        sir_observations
    )
    _write_json(output_directory / "run_manifest.json", manifest)
    contexts = [
        {
            "spec": diagonal_lgssm_spec(),
            "theta": tf.constant(LGSSM_THETA, tf.float32),
            "observations": lgssm_observations,
            "static_spec": None,
            "reference": _lgssm_reference(lgssm_observations),
        }
    ]
    if not args.smoke:
        contexts.append(
            {
                "spec": reduced_sir_spec(),
                "theta": tf.constant(SIR_THETA, tf.float32),
                "observations": sir_observations,
                "static_spec": sir_static_spec,
                "reference": _sir_reference(sir_model, sir_observations),
            }
        )

    rows = []
    compile_records = []
    seeds = RANDOMIZATION_SEEDS[:1] if args.smoke else RANDOMIZATION_SEEDS
    references: dict[str, Any] = {}
    for context in contexts:
        spec = context["spec"]
        reference = context["reference"]
        references[spec.family] = reference
        if not reference["finite"] or bool(reference.get("boundary_mass_veto", False)):
            raise RuntimeError(f"reference veto for {spec.family}")
        evaluator = _make_evaluator(spec, sir_static_spec=context["static_spec"])
        reset_design = replicate_positive_genut(
            gaussian_genut_design(dim=spec.state_dimension),
            num_particles=PARTICLE_COUNT,
        )
        warm_initial, warm_process, _ = _clouds(
            spec, BASELINE, "all_innovations", 93001
        )
        compile_started = time.perf_counter()
        warm_value, warm_score, warm_diagnostics = evaluator(
            context["theta"],
            context["observations"],
            warm_initial,
            warm_process,
            reset_design,
        )
        compile_elapsed = time.perf_counter() - compile_started
        if not bool(warm_diagnostics["program_valid"].numpy()):
            raise RuntimeError(f"compiled warm-up invalid for {spec.family}")
        compile_records.append(
            {
                "model": spec.family,
                "compile_and_first_call_seconds": compile_elapsed,
                "warm_value_finite": bool(tf.math.is_finite(warm_value).numpy()),
                "warm_score_finite": bool(
                    tf.reduce_all(tf.math.is_finite(warm_score)).numpy()
                ),
                "value_device": warm_value.device,
                "score_device": warm_score.device,
                **_xla_contract(evaluator),
            }
        )
        for scope in SCOPES:
            for design_id in SUPPORTED_DESIGNS:
                for seed in seeds:
                    rows.append(
                        _run_row(
                            spec=spec,
                            theta=context["theta"],
                            observations=context["observations"],
                            evaluator=evaluator,
                            reset_design=reset_design,
                            reference=reference,
                            design_id=design_id,
                            scope=scope,
                            seed=seed,
                        )
                    )
        if evaluator.experimental_get_tracing_count() != 1:
            raise RuntimeError(f"unexpected retracing for {spec.family}")

    expected_rows = len(contexts) * len(SCOPES) * len(SUPPORTED_DESIGNS) * len(seeds)
    if len(rows) != expected_rows:
        raise RuntimeError("campaign cell count mismatch")
    if any("CPU:0" not in row["value_device"] for row in rows):
        raise RuntimeError("a result tensor was not placed on CPU")
    cell_summaries, paired, scope_contrasts = _aggregate(rows)
    raw = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "expected_row_count": expected_rows,
        "references": references,
        "compile_records": compile_records,
    }
    _write_json(output_directory / "raw.json", raw)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "mechanics_pilot_pass" if not args.smoke else "smoke_pass",
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "references": references,
        "compile_records": compile_records,
        "cell_summaries": cell_summaries,
        "paired_candidate_minus_iid": paired,
        "paired_all_innovations_minus_initial_only": scope_contrasts,
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "cell means, runtime, ESS, tail values, and individual errors"
            ),
            "pilot_pairwise_signals": (
                "unadjusted bootstrap intervals are preserved but do not rank methods"
            ),
            "default_readiness": "not_evaluated",
            "next_evidence_needed": (
                "scope-specific tuning, more paired replications, multiplicity-aware "
                "confirmatory intervals, and GPU/XLA target validation"
            ),
        },
        "decision": {
            "primary_criterion_status": "mechanics_and_execution_passed",
            "veto_diagnostic_status": "no_cloud_flow_reset_reference_or_execution_veto",
            "main_uncertainty": (
                "eight randomizations, two short models, frozen reset controls, and "
                "an approximate reduced-SIR reference"
            ),
            "next_justified_action": (
                "use pilot signals only to nominate designs for a tuned confirmatory campaign"
            ),
            "not_concluded": (
                "no universal winner, default initialization, unbiased estimator, "
                "GPU performance, HMC readiness, or Austria-SIR result"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "observed differences may be seed noise or interaction with untuned reset controls"
            ),
            "result_that_would_overturn": (
                "multiplicity-aware confirmatory paired intervals on tuned scopes that "
                "reverse or erase the pilot signals"
            ),
            "weakest_evidence": (
                "tail behavior and cross-model generalization from eight short replicates"
            ),
        },
        "elapsed_seconds": time.perf_counter() - campaign_started,
        "artifact_paths": {
            "raw": str((output_directory / "raw.json").relative_to(ROOT)),
            "result": str((output_directory / "result.json").relative_to(ROOT)),
            "markdown": str((output_directory / "result.md").relative_to(ROOT)),
            "manifest": str((output_directory / "run_manifest.json").relative_to(ROOT)),
        },
    }
    _write_json(output_directory / "result.json", result)
    (output_directory / "result.md").write_text(_markdown(result), encoding="utf-8")
    manifest.update(
        {
            "status": "complete",
            "completed_unix_seconds": time.time(),
            "wall_time_seconds": time.perf_counter() - campaign_started,
            "result_file": result["artifact_paths"]["result"],
            "artifact_paths": result["artifact_paths"],
            "artifact_sha256": {
                name: _sha256(Path(path))
                for name, path in result["artifact_paths"].items()
                if name != "manifest"
            },
            "reference_ids": {
                name: reference["reference_id"]
                for name, reference in references.items()
            },
            "row_count": len(rows),
            "expected_row_count": expected_rows,
        }
    )
    _write_json(output_directory / "run_manifest.json", manifest)
    print(json.dumps({"status": result["status"], **result["artifact_paths"]}, indent=2))


if __name__ == "__main__":
    main()
