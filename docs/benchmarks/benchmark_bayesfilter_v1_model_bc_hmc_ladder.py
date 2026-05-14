"""CPU-only Model B/C HMC readiness ladder for BayesFilter V1.

This benchmark is diagnostic infrastructure.  It uses the analytic
sigma-point score as the gradient supplied to TFP HMC via ``tf.custom_gradient``.
It does not certify convergence or posterior recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from bayesfilter.testing import (  # noqa: E402
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_score_branch_summary,
    tf_nonlinear_sigma_point_score,
)


tfm = tfp.mcmc
BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")
MODELS = ("model_b_nonlinear_accumulation", "model_c_autonomous_nonlinear_growth")
DEFAULT_SEEDS = ((20260515, 11), (20260515, 23), (20260515, 37))


@dataclass(frozen=True)
class TargetSpec:
    model: str
    backend: str
    parameter_names: tuple[str, ...]
    initial_parameters: tuple[float, ...]
    prior_mean: tuple[float, ...]
    prior_scale: tuple[float, ...]
    parameter_box: tuple[tuple[float, float], ...]
    observations: tuple[tuple[float, ...], ...]
    allow_fixed_null_support: bool
    step_size: float
    num_leapfrog_steps: int
    note: str


@dataclass(frozen=True)
class TargetResult:
    target: str
    model: str
    backend: str
    classification: str
    claim_scope: str
    parameter_names: tuple[str, ...]
    initial_parameters: tuple[float, ...]
    prior_mean: tuple[float, ...]
    prior_scale: tuple[float, ...]
    parameter_box: tuple[tuple[float, float], ...]
    allow_fixed_null_support: bool
    branch_ok_count: int
    branch_total_count: int
    branch_failure_labels: tuple[str, ...]
    compiled_eager_value_abs_residual: float | None
    compiled_eager_gradient_max_abs_residual: float | None
    compiled_eager_parity_ok: bool
    initial_target_log_prob: float | None
    initial_gradient: tuple[float, ...] | None
    initial_gradient_finite: bool
    chains: int
    draws_per_chain: int
    burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seeds: tuple[tuple[int, int], ...]
    finite_sample_count: int
    nonfinite_sample_count: int
    acceptance_rate_by_chain: tuple[float, ...]
    min_acceptance_rate: float | None
    max_acceptance_rate: float | None
    max_abs_log_accept_ratio: float | None
    rhat: tuple[float, ...] | None
    max_rhat: float | None
    sample_mean: tuple[float, ...] | None
    sample_stddev: tuple[float, ...] | None
    naive_mcse_sd_ratio: tuple[float, ...] | None
    divergence_count: int | None
    runtime_seconds: float
    error: str | None


class AnalyticScoreHMCTarget:
    """Nonlinear target whose HMC gradient is the analytic score plus prior."""

    def __init__(self, spec: TargetSpec):
        self.spec = spec
        self.observations = tf.constant(spec.observations, dtype=tf.float64)
        self.initial_parameters = tf.constant(spec.initial_parameters, dtype=tf.float64)
        self.prior_mean = tf.constant(spec.prior_mean, dtype=tf.float64)
        self.prior_scale = tf.constant(spec.prior_scale, dtype=tf.float64)

    def model_and_derivatives(self, parameters: tf.Tensor):
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        if self.spec.model == "model_b_nonlinear_accumulation":
            return (
                make_nonlinear_accumulation_model_tf(
                    rho=params[0],
                    sigma=params[1],
                    beta=params[2],
                ),
                make_nonlinear_accumulation_first_derivatives_tf(
                    rho=params[0],
                    sigma=params[1],
                    beta=params[2],
                ),
            )
        if self.spec.model == "model_c_autonomous_nonlinear_growth":
            return (
                make_univariate_nonlinear_growth_model_tf(
                    process_sigma=params[0],
                    observation_sigma=params[1],
                    initial_variance=params[2],
                ),
                make_univariate_nonlinear_growth_first_derivatives_tf(
                    process_sigma=params[0],
                    observation_sigma=params[1],
                ),
            )
        raise ValueError(f"unknown model: {self.spec.model}")

    def likelihood_and_score(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        model, derivatives = self.model_and_derivatives(parameters)
        result = tf_nonlinear_sigma_point_score(
            self.observations,
            model,
            derivatives,
            backend=self.spec.backend,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
            allow_fixed_null_support=self.spec.allow_fixed_null_support,
        )
        return result.log_likelihood, result.score

    def target_log_prob_and_grad(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        lower = tf.constant([lo for lo, _hi in self.spec.parameter_box], dtype=tf.float64)
        upper = tf.constant([hi for _lo, hi in self.spec.parameter_box], dtype=tf.float64)
        inside = tf.reduce_all(tf.logical_and(params >= lower, params <= upper))
        value, score = self.likelihood_and_score(params)
        centered = params - self.prior_mean
        prior_quadratic = tf.reduce_sum(tf.square(centered / self.prior_scale))
        prior_score = -(centered / tf.square(self.prior_scale))
        target_value = value - 0.5 * prior_quadratic
        target_score = score + prior_score
        return (
            tf.where(inside, target_value, tf.constant(-1.0e100, dtype=tf.float64)),
            tf.where(inside, target_score, tf.zeros_like(target_score)),
        )

    def target_log_prob(self, parameters: tf.Tensor) -> tf.Tensor:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)

        @tf.custom_gradient
        def _with_analytic_gradient(theta: tf.Tensor) -> tuple[tf.Tensor, Any]:
            value, gradient = self.target_log_prob_and_grad(theta)

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                return upstream * gradient

            return value, grad

        return _with_analytic_gradient(params)

    def branch_summary(self) -> dict[str, Any]:
        grid = _branch_grid(self.spec.model)
        summary = nonlinear_sigma_point_score_branch_summary(
            self.observations,
            tf.constant(grid, dtype=tf.float64),
            lambda params: self.model_and_derivatives(params)[0],
            lambda params: self.model_and_derivatives(params)[1],
            backend=self.spec.backend,
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
            allow_fixed_null_support=self.spec.allow_fixed_null_support,
        )
        return {
            "ok_count": summary.ok_count,
            "total_count": summary.total_count,
            "failure_labels": summary.failure_labels,
        }


def _branch_grid(model: str) -> tuple[tuple[float, ...], ...]:
    if model == "model_b_nonlinear_accumulation":
        return (
            (0.62, 0.20, 0.70),
            (0.70, 0.25, 0.80),
            (0.78, 0.30, 0.90),
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return (
            (0.90, 1.00, 0.20),
            (1.00, 1.00, 0.20),
            (1.10, 1.10, 0.25),
        )
    raise ValueError(f"unknown model: {model}")


def _target_specs(backends: tuple[str, ...], models: tuple[str, ...]) -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = []
    for model in models:
        for backend in backends:
            if model == "model_b_nonlinear_accumulation":
                specs.append(
                    TargetSpec(
                        model=model,
                        backend=backend,
                        parameter_names=("rho", "sigma", "beta"),
                        initial_parameters=(0.70, 0.25, 0.80),
                        prior_mean=(0.70, 0.25, 0.80),
                        prior_scale=(0.25, 0.15, 0.25),
                        parameter_box=((0.55, 0.85), (0.15, 0.40), (0.45, 1.10)),
                        observations=_as_tuple(model_b_observations_tf()),
                        allow_fixed_null_support=False,
                        step_size=0.005,
                        num_leapfrog_steps=2,
                        note="BC1-BC3 full envelope passed for Model B.",
                    )
                )
            elif model == "model_c_autonomous_nonlinear_growth":
                specs.append(
                    TargetSpec(
                        model=model,
                        backend=backend,
                        parameter_names=("sigma_u", "sigma_y", "P0x"),
                        initial_parameters=(1.00, 1.00, 0.20),
                        prior_mean=(1.00, 1.00, 0.20),
                        prior_scale=(0.25, 0.25, 0.12),
                        parameter_box=((0.60, 1.40), (0.60, 1.40), (0.10, 0.50)),
                        observations=_as_tuple(model_c_observations_tf()),
                        allow_fixed_null_support=True,
                        step_size=0.0001,
                        num_leapfrog_steps=2,
                        note=(
                            "Default short Model C target uses structural fixed support. "
                            "BC3 blocks selected SVD-UKF T=32 rows but not this short target."
                        ),
                    )
                )
            else:
                raise ValueError(f"unknown model: {model}")
    return tuple(specs)


def _as_tuple(tensor: tf.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in tensor.numpy())


def _run_target(
    spec: TargetSpec,
    *,
    seeds: tuple[tuple[int, int], ...],
    num_results: int,
    num_burnin_steps: int,
) -> TargetResult:
    target = AnalyticScoreHMCTarget(spec)
    start = time.perf_counter()
    try:
        branch = target.branch_summary()
        initial_value, initial_gradient = target.target_log_prob_and_grad(
            target.initial_parameters
        )
        initial_gradient_finite = bool(
            tf.reduce_all(tf.math.is_finite(initial_gradient)).numpy()
        )
        parity = _compiled_eager_parity(target, target.initial_parameters)
        if branch["ok_count"] != branch["total_count"] or not initial_gradient_finite:
            return _blocked_result(
                spec,
                branch=branch,
                parity=parity,
                initial_value=initial_value,
                initial_gradient=initial_gradient,
                initial_gradient_finite=initial_gradient_finite,
                seeds=seeds,
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                runtime_seconds=time.perf_counter() - start,
                error="branch_or_initial_gradient_gate_failed",
            )

        samples_by_chain = []
        acceptance_rates = []
        max_log_accept = 0.0
        finite_count = 0
        nonfinite_count = 0
        for seed in seeds:
            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target.target_log_prob,
                step_size=tf.constant(spec.step_size, dtype=tf.float64),
                num_leapfrog_steps=spec.num_leapfrog_steps,
            )
            samples, trace = tfm.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=target.initial_parameters,
                kernel=kernel,
                trace_fn=lambda _state, kernel_results: {
                    "is_accepted": kernel_results.is_accepted,
                    "log_accept_ratio": kernel_results.log_accept_ratio,
                    "target_log_prob": kernel_results.accepted_results.target_log_prob,
                },
                seed=tf.constant(seed, dtype=tf.int32),
            )
            samples_by_chain.append(samples.numpy())
            accepted = trace["is_accepted"].numpy()
            acceptance_rates.append(float(np.mean(accepted)))
            max_log_accept = max(
                max_log_accept,
                float(np.max(np.abs(trace["log_accept_ratio"].numpy()))),
            )
            finite_mask = np.all(np.isfinite(samples.numpy()), axis=-1)
            finite_count += int(np.sum(finite_mask))
            nonfinite_count += int(np.sum(~finite_mask))

        samples_np = np.stack(samples_by_chain, axis=0)
        rhat = _rhat(samples_np)
        sample_mean = np.mean(samples_np.reshape((-1, samples_np.shape[-1])), axis=0)
        sample_stddev = np.std(samples_np.reshape((-1, samples_np.shape[-1])), axis=0)
        total_draws = samples_np.shape[0] * samples_np.shape[1]
        naive_ratio = np.full(samples_np.shape[-1], 1.0 / math.sqrt(total_draws))
        min_acceptance = min(acceptance_rates) if acceptance_rates else None
        max_acceptance = max(acceptance_rates) if acceptance_rates else None
        finite_ok = nonfinite_count == 0 and initial_gradient_finite
        acceptance_ok = (
            min_acceptance is not None
            and max_acceptance is not None
            and min_acceptance >= 0.05
            and max_acceptance <= 1.0
        )
        classification = "candidate" if finite_ok and acceptance_ok else "blocked"
        return TargetResult(
            target=f"{spec.model}_{spec.backend}",
            model=spec.model,
            backend=spec.backend,
            classification=classification,
            claim_scope="hmc_readiness_candidate_not_convergence",
            parameter_names=spec.parameter_names,
            initial_parameters=spec.initial_parameters,
            prior_mean=spec.prior_mean,
            prior_scale=spec.prior_scale,
            parameter_box=spec.parameter_box,
            allow_fixed_null_support=spec.allow_fixed_null_support,
            branch_ok_count=int(branch["ok_count"]),
            branch_total_count=int(branch["total_count"]),
            branch_failure_labels=tuple(branch["failure_labels"]),
            compiled_eager_value_abs_residual=parity["value_abs_residual"],
            compiled_eager_gradient_max_abs_residual=parity["gradient_max_abs_residual"],
            compiled_eager_parity_ok=parity["ok"],
            initial_target_log_prob=float(initial_value.numpy()),
            initial_gradient=tuple(float(x) for x in initial_gradient.numpy()),
            initial_gradient_finite=initial_gradient_finite,
            chains=len(seeds),
            draws_per_chain=num_results,
            burnin_steps=num_burnin_steps,
            step_size=spec.step_size,
            num_leapfrog_steps=spec.num_leapfrog_steps,
            seeds=seeds,
            finite_sample_count=finite_count,
            nonfinite_sample_count=nonfinite_count,
            acceptance_rate_by_chain=tuple(acceptance_rates),
            min_acceptance_rate=min_acceptance,
            max_acceptance_rate=max_acceptance,
            max_abs_log_accept_ratio=max_log_accept,
            rhat=tuple(float(x) for x in rhat),
            max_rhat=float(np.max(rhat)),
            sample_mean=tuple(float(x) for x in sample_mean),
            sample_stddev=tuple(float(x) for x in sample_stddev),
            naive_mcse_sd_ratio=tuple(float(x) for x in naive_ratio),
            divergence_count=None,
            runtime_seconds=time.perf_counter() - start,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - diagnostic artifact path.
        return _blocked_result(
            spec,
            branch={"ok_count": 0, "total_count": 0, "failure_labels": ()},
            parity={
                "value_abs_residual": None,
                "gradient_max_abs_residual": None,
                "ok": False,
            },
            initial_value=None,
            initial_gradient=None,
            initial_gradient_finite=False,
            seeds=seeds,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            runtime_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def _blocked_result(
    spec: TargetSpec,
    *,
    branch: dict[str, Any],
    parity: dict[str, float | bool | None],
    initial_value: tf.Tensor | None,
    initial_gradient: tf.Tensor | None,
    initial_gradient_finite: bool,
    seeds: tuple[tuple[int, int], ...],
    num_results: int,
    num_burnin_steps: int,
    runtime_seconds: float,
    error: str,
) -> TargetResult:
    return TargetResult(
        target=f"{spec.model}_{spec.backend}",
        model=spec.model,
        backend=spec.backend,
        classification="blocked",
        claim_scope="hmc_readiness_blocked",
        parameter_names=spec.parameter_names,
        initial_parameters=spec.initial_parameters,
        prior_mean=spec.prior_mean,
        prior_scale=spec.prior_scale,
        parameter_box=spec.parameter_box,
        allow_fixed_null_support=spec.allow_fixed_null_support,
        branch_ok_count=int(branch["ok_count"]),
        branch_total_count=int(branch["total_count"]),
        branch_failure_labels=tuple(branch["failure_labels"]),
        compiled_eager_value_abs_residual=parity["value_abs_residual"],
        compiled_eager_gradient_max_abs_residual=parity["gradient_max_abs_residual"],
        compiled_eager_parity_ok=bool(parity["ok"]),
        initial_target_log_prob=None if initial_value is None else float(initial_value.numpy()),
        initial_gradient=None
        if initial_gradient is None
        else tuple(float(x) for x in initial_gradient.numpy()),
        initial_gradient_finite=initial_gradient_finite,
        chains=len(seeds),
        draws_per_chain=num_results,
        burnin_steps=num_burnin_steps,
        step_size=spec.step_size,
        num_leapfrog_steps=spec.num_leapfrog_steps,
        seeds=seeds,
        finite_sample_count=0,
        nonfinite_sample_count=0,
        acceptance_rate_by_chain=(),
        min_acceptance_rate=None,
        max_acceptance_rate=None,
        max_abs_log_accept_ratio=None,
        rhat=None,
        max_rhat=None,
        sample_mean=None,
        sample_stddev=None,
        naive_mcse_sd_ratio=None,
        divergence_count=None,
        runtime_seconds=runtime_seconds,
        error=error,
    )


def _compiled_eager_parity(
    target: AnalyticScoreHMCTarget,
    parameters: tf.Tensor,
) -> dict[str, float | bool]:
    eager_value, eager_gradient = target.target_log_prob_and_grad(parameters)

    @tf.function(reduce_retracing=True)
    def compiled(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return target.target_log_prob_and_grad(values)

    compiled_value, compiled_gradient = compiled(parameters)
    value_abs_residual = float(tf.abs(compiled_value - eager_value).numpy())
    gradient_max_abs_residual = float(
        tf.reduce_max(tf.abs(compiled_gradient - eager_gradient)).numpy()
    )
    return {
        "value_abs_residual": value_abs_residual,
        "gradient_max_abs_residual": gradient_max_abs_residual,
        "ok": value_abs_residual <= 1e-10 and gradient_max_abs_residual <= 1e-8,
    }


def _rhat(samples: np.ndarray) -> np.ndarray:
    chains, draws, _dim = samples.shape
    if chains < 2 or draws < 2:
        return np.full(samples.shape[-1], np.nan)
    chain_means = np.mean(samples, axis=1)
    chain_vars = np.var(samples, axis=1, ddof=1)
    between = draws * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(chain_vars, axis=0)
    var_hat = ((draws - 1.0) / draws) * within + between / draws
    return np.sqrt(var_hat / np.maximum(within, np.finfo(float).tiny))


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
        "policy": (
            "CPU-only opt-in HMC readiness diagnostic.  Classification is "
            "candidate/blocked, not convergence."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# BayesFilter V1 Model B/C HMC Ladder",
        "",
        f"The JSON file is authoritative: `{json_path}`.",
        "",
        "## Claim Scope",
        "",
        str(payload["claim_scope"]),
        "",
        "## Rows",
        "",
        "| Target | Classification | Chains | Draws | Acceptance | Max R-hat | Nonfinite |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in payload["rows"]:
        acceptance = row["acceptance_rate_by_chain"]
        max_rhat = row["max_rhat"]
        lines.append(
            "| {target} | `{classification}` | {chains} | {draws} | {acceptance} | {rhat} | {nonfinite} |".format(
                target=row["target"],
                classification=row["classification"],
                chains=row["chains"],
                draws=row["draws_per_chain"],
                acceptance="n/a" if not acceptance else ", ".join(f"{x:.3f}" for x in acceptance),
                rhat="n/a" if max_rhat is None else f"{max_rhat:.3f}",
                nonfinite=row["nonfinite_sample_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Candidate means finite branch-gated CPU HMC diagnostics suitable for a future convergence ladder.  It is not a convergence or posterior-recovery claim.",
        ]
    )
    return "\n".join(lines)


def _parse_csv(raw: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown --{name} value(s): {', '.join(unknown)}")
    if not values:
        raise ValueError(f"--{name} must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--num-results", type=int, default=16)
    parser.add_argument("--num-burnin-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    models = _parse_csv(args.models, MODELS, "models")
    backends = _parse_csv(args.backends, BACKENDS, "backends")
    specs = _target_specs(backends, models)
    rows = [
        _run_target(
            spec,
            seeds=DEFAULT_SEEDS,
            num_results=args.num_results,
            num_burnin_steps=args.num_burnin_steps,
        )
        for spec in specs
    ]
    payload = _json_safe(
        {
            "benchmark": "bayesfilter_v1_model_bc_hmc_ladder",
            "claim_scope": "cpu_hmc_readiness_candidate_not_convergence",
            "config": {
                "models": models,
                "backends": backends,
                "num_results": args.num_results,
                "num_burnin_steps": args.num_burnin_steps,
                "seeds": DEFAULT_SEEDS,
                "gradient_path": "tf_custom_gradient_wrapping_analytic_sigma_point_score",
                "divergence_count_policy": "not available from this fixed-step HMC diagnostic",
            },
            "environment": _environment(),
            "rows": [asdict(row) for row in rows],
        }
    )
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    summary = {
        row["target"]: row["classification"]
        for row in payload["rows"]
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
