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
    # Diagonal mass matrix adaptation during warmup. Default False keeps the
    # master-program kernel row (NoUTurnSampler + DualAveraging step size)
    # exactly as approved; opt in for badly conditioned targets where one
    # scalar step size cannot serve every coordinate. See master program
    # Amendment A2: G2.3 spans 1.9e4 in per-coordinate posterior sd, so an
    # identity mass matrix cannot mix theta_bar.
    diagonal_mass_matrix: bool = False


def run_nuts(log_prob_fn, initial_states, config: NutsConfig):
    """Run multi-chain NUTS with dual-averaging warmup.

    ``initial_states``: list of tensors, each [num_chains, ...].
    Returns dict with samples, divergence count, step size, and
    per-variable potential-scale-reduction.
    """
    configure_memory_growth()
    step = tf.constant(config.initial_step_size, tf.float64)
    if config.diagonal_mass_matrix:
        # Plain NoUTurnSampler exposes no momentum_distribution slot in TFP
        # 0.25.0, so diagonal preconditioning needs the Preconditioned
        # variant. Same NUTS algorithm. Adaptation is warmup-only.
        inner = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=log_prob_fn,
            step_size=step,
            max_tree_depth=config.max_tree_depth,
        )
        kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
            inner_kernel=inner,
            initial_running_variance=[
                tfp.experimental.stats.RunningVariance.from_shape(
                    s.shape[1:], dtype=s.dtype)
                for s in initial_states
            ],
            num_estimation_steps=int(0.8 * config.num_warmup),
        )
    else:
        kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=log_prob_fn,
            step_size=step,
            max_tree_depth=config.max_tree_depth,
        )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * config.num_warmup),
        target_accept_prob=tf.constant(config.target_accept, tf.float64),
    )

    def _nuts_results(pkr):
        """Descend to NUTS kernel results through any adaptation wrappers.

        Wrapper depth depends on config: one (step size only) or two (step
        size over mass matrix adaptation).
        """
        while not hasattr(pkr, "has_divergence"):
            pkr = pkr.inner_results
        return pkr

    @tf.function(jit_compile=config.jit_compile)
    def _run():
        return tfp.mcmc.sample_chain(
            num_results=config.num_samples,
            num_burnin_steps=config.num_warmup,
            current_state=initial_states,
            kernel=adaptive,
            seed=tf.constant([config.seed, config.seed + 1], tf.int32),
            trace_fn=lambda _, pkr: {
                "diverged": _nuts_results(pkr).has_divergence,
                "step_size": _nuts_results(pkr).step_size,
                "accept": _nuts_results(pkr).log_accept_ratio,
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
