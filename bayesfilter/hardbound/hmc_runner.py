"""Thin seeded NUTS runner (master program pre-approved kernel choice).

Deliberately independent of the NeuTra/route-ledger stack. Sets GPU memory
growth before any device initialization when GPUs are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp


def configure_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass  # already initialized; acceptable for CPU-approved runs


@dataclass(frozen=True)
class NutsConfig:
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    initial_step_size: float = 0.05
    target_accept: float = 0.9
    seed: int = 20260821
    max_tree_depth: int = 10
    jit_compile: bool = True


def run_nuts(log_prob_fn, initial_states, config: NutsConfig):
    """Run multi-chain NUTS with dual-averaging warmup.

    ``initial_states``: list of tensors, each [num_chains, ...].
    Returns dict with samples, divergence count, step size, and
    per-variable potential-scale-reduction.
    """
    configure_memory_growth()
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=tf.constant(config.initial_step_size, tf.float64),
        max_tree_depth=config.max_tree_depth,
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * config.num_warmup),
        target_accept_prob=tf.constant(config.target_accept, tf.float64),
    )

    @tf.function(jit_compile=config.jit_compile)
    def _run():
        return tfp.mcmc.sample_chain(
            num_results=config.num_samples,
            num_burnin_steps=config.num_warmup,
            current_state=initial_states,
            kernel=adaptive,
            seed=tf.constant([config.seed, config.seed + 1], tf.int32),
            trace_fn=lambda _, pkr: {
                "diverged": pkr.inner_results.has_divergence,
                "step_size": pkr.inner_results.step_size,
                "accept": pkr.inner_results.log_accept_ratio,
            },
        )

    samples, trace = _run()
    rhat = [tfp.mcmc.potential_scale_reduction(s) for s in samples]
    if config.num_chains > 1:
        ess = [tfp.mcmc.effective_sample_size(s, cross_chain_dims=1)
               for s in samples]
    else:
        ess = [tfp.mcmc.effective_sample_size(s) for s in samples]
    return {
        "samples": samples,
        "divergences": int(tf.reduce_sum(
            tf.cast(trace["diverged"], tf.int32))),
        "rhat": rhat,
        "ess": ess,
        "final_step_size": trace["step_size"][-1],
    }
