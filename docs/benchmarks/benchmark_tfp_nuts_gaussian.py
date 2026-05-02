"""Benchmark TFP HMC and NUTS on a fixed Gaussian target.

This intentionally boring benchmark exists to test one narrow implementation
claim: whether TFP NUTS should be proposed as the default BayesFilter fix for
HMC pathologies.  The target is a standard multivariate Gaussian with static
shape, so any overhead observed here is sampler/framework overhead rather than
filtering-model complexity.

The script reports first-call time, second-call time, and effective steady
iteration time for eager, graph, and XLA modes.  First-call time includes
tracing/compilation when a compiled mode is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402


tfd = tfp.distributions
tfm = tfp.mcmc


@dataclass
class BenchmarkConfig:
    dim: int
    chains: int
    num_results: int
    num_burnin_steps: int
    repeats: int
    hmc_leapfrog_steps: int
    nuts_max_tree_depth: int
    step_size: float
    seed: int


@dataclass
class BenchmarkResult:
    sampler: str
    mode: str
    first_call_seconds: float | None
    second_call_seconds: float | None
    mean_repeat_seconds: float | None
    seconds_per_draw: float | None
    acceptance_rate: float | None
    sample_mean: float | None
    status: str
    error: str | None


def _target_log_prob(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return -0.5 * tf.reduce_sum(tf.square(x), axis=-1)


def _make_kernel(sampler: str, config: BenchmarkConfig) -> tfm.TransitionKernel:
    step_size = tf.constant(config.step_size, dtype=tf.float32)
    if sampler == "hmc":
        return tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=_target_log_prob,
            step_size=step_size,
            num_leapfrog_steps=config.hmc_leapfrog_steps,
        )
    if sampler == "nuts":
        return tfm.NoUTurnSampler(
            target_log_prob_fn=_target_log_prob,
            step_size=step_size,
            max_tree_depth=config.nuts_max_tree_depth,
        )
    raise ValueError(f"unknown sampler: {sampler}")


def _make_runner(
    sampler: str,
    mode: str,
    config: BenchmarkConfig,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    initial_state = tf.zeros([config.chains, config.dim], dtype=tf.float32)
    kernel = _make_kernel(sampler, config)

    def body(seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        samples, trace = tfm.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _state, kernel_results: kernel_results.is_accepted,
            seed=seed,
        )
        acceptance_rate = tf.reduce_mean(tf.cast(trace, tf.float32))
        return tf.reduce_mean(samples), acceptance_rate

    if mode == "eager":
        return body
    if mode == "graph":
        return tf.function(body)
    if mode == "xla":
        return tf.function(body, jit_compile=True)
    raise ValueError(f"unknown mode: {mode}")


def _time_call(
    runner: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    seed: int,
) -> tuple[float, float, float]:
    start = time.perf_counter()
    sample_mean, acceptance_rate = runner(tf.constant([seed, seed + 1], tf.int32))
    sample_mean_value = float(sample_mean.numpy())
    acceptance_rate_value = float(acceptance_rate.numpy())
    return time.perf_counter() - start, sample_mean_value, acceptance_rate_value


def run_one(sampler: str, mode: str, config: BenchmarkConfig) -> BenchmarkResult:
    runner = _make_runner(sampler, mode, config)
    timings: list[float] = []
    means: list[float] = []
    acceptance_rates: list[float] = []

    try:
        for repeat in range(config.repeats):
            elapsed, mean, acceptance_rate = _time_call(
                runner, config.seed + 1009 * repeat
            )
            timings.append(elapsed)
            means.append(mean)
            acceptance_rates.append(acceptance_rate)
    except Exception as exc:  # pragma: no cover - benchmark records environment failures.
        return BenchmarkResult(
            sampler=sampler,
            mode=mode,
            first_call_seconds=timings[0] if timings else None,
            second_call_seconds=timings[1] if len(timings) > 1 else None,
            mean_repeat_seconds=None,
            seconds_per_draw=None,
            acceptance_rate=None,
            sample_mean=None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    steady_timings = timings[1:] if len(timings) > 1 else timings
    mean_repeat_seconds = sum(steady_timings) / len(steady_timings)
    draws_per_repeat = config.num_results * config.chains
    return BenchmarkResult(
        sampler=sampler,
        mode=mode,
        first_call_seconds=timings[0],
        second_call_seconds=timings[1] if len(timings) > 1 else None,
        mean_repeat_seconds=mean_repeat_seconds,
        seconds_per_draw=mean_repeat_seconds / draws_per_repeat,
        acceptance_rate=sum(acceptance_rates) / len(acceptance_rates),
        sample_mean=sum(means) / len(means),
        status="ok",
        error=None,
    )


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
        "xla_note": (
            "First XLA call includes tracing and compilation; later calls reuse "
            "the compiled executable for the same static shape."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--num-results", type=int, default=32)
    parser.add_argument("--num-burnin-steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=3)
    parser.add_argument("--nuts-max-tree-depth", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig(
        dim=args.dim,
        chains=args.chains,
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        repeats=args.repeats,
        hmc_leapfrog_steps=args.hmc_leapfrog_steps,
        nuts_max_tree_depth=args.nuts_max_tree_depth,
        step_size=args.step_size,
        seed=args.seed,
    )

    rows = [
        run_one(sampler, mode, config)
        for sampler in ("hmc", "nuts")
        for mode in ("eager", "graph", "xla")
    ]
    payload = {
        "benchmark": "tfp_hmc_nuts_gaussian",
        "config": asdict(config),
        "environment": _environment(),
        "results": [asdict(row) for row in rows],
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
