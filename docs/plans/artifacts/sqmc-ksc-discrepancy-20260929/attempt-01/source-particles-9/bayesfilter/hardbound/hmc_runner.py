"""Thin seeded fixed-trajectory HMC runner (master program Amendment A3).

Deliberately independent of the NeuTra/route-ledger stack. Sets GPU memory
growth before any device initialization when GPUs are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.hardbound.dense_mass_matrix_adaptation import DenseMassMatrixAdaptation
from bayesfilter.hardbound.windowed_dense_mass_adaptation import run_windowed_dense_mass_adaptation


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
    target_accept: float = 0.70  # Amendment A3: repository standard, band (0.65, 0.75)
    seed: int = 20260821
    num_leapfrog_steps: int = 50  # Amendment A3: explicit trajectory length for fixed-trajectory HMC
    jit_compile: bool = True
    # Diagonal mass matrix adaptation during warmup. Default False keeps the
    # master-program kernel row (HamiltonianMonteCarlo + DualAveraging step size)
    # exactly as approved; opt in for badly conditioned targets where one
    # scalar step size cannot serve every coordinate. See master program
    # Amendment A2: G2.3 spans 1.9e4 in per-coordinate posterior sd, so an
    # identity mass matrix cannot mix theta_bar.
    diagonal_mass_matrix: bool = False
    # Dense (full covariance) mass matrix adaptation. Estimates the full
    # posterior covariance and uses its Cholesky factor as the momentum
    # precision. Mutually exclusive with diagonal_mass_matrix. Required when
    # diagonal fails to capture strong posterior correlation. G2.3 Amendment
    # A3: diagonal reduced max R-hat from 60.5 to 1.048 but stalled; 6 of 9
    # theta coordinates have low ESS and R-hat > 1.02, indicating off-diagonal
    # structure the diagonal preconditioner cannot capture. The Cholesky factor
    # is wrapped in LinearOperatorLowerTriangular so the log-abs-det is
    # closed-form O(N), not the O(N^3) dense LU that LinearOperatorFullMatrix
    # would use, making the route XLA-compatible.
    dense_mass_matrix: bool = False
    # Windowed dense mass matrix adaptation. Uses Stan-style windowed schedule
    # (initial buffer, progressive doubling windows, final buffer) with
    # covariance shrinkage toward identity to handle n≪p regimes. Amendment A4:
    # dense_mass_matrix single-freeze scheme gave max R-hat 1.083 (theta_2),
    # indicating tuning-candidate failure. Windowed route applies shrinkage
    # (default λ=0.1) and refines covariance progressively across doubling
    # windows. Mutually exclusive with diagonal_mass_matrix and dense_mass_matrix.
    dense_mass_windowed: bool = False


def run_nuts(log_prob_fn, initial_states, config: NutsConfig):
    """Run multi-chain NUTS with dual-averaging warmup.

    ``initial_states``: list of tensors, each [num_chains, ...].
    Returns dict with samples, divergence count, step size, and
    per-variable potential-scale-reduction.
    """
    configure_memory_growth()
    mass_flags = sum([
        config.diagonal_mass_matrix,
        config.dense_mass_matrix,
        config.dense_mass_windowed,
    ])
    if mass_flags > 1:
        raise ValueError(
            "diagonal_mass_matrix, dense_mass_matrix, and dense_mass_windowed "
            "are mutually exclusive"
        )
    step = tf.constant(config.initial_step_size, tf.float64)
    if config.dense_mass_windowed:
        # Windowed dense mass adaptation with covariance shrinkage. Implements
        # Stan-style windowed warmup schedule (initial buffer, progressive
        # doubling slow windows, final buffer) and handles n≪p regime with
        # shrinkage toward identity. Returns final samples and diagnostics
        # directly; does not use TFP kernel wrappers.
        return run_windowed_dense_mass_adaptation(
            target_log_prob_fn=log_prob_fn,
            initial_states=initial_states,
            num_warmup_steps=config.num_warmup,
            num_samples=config.num_samples,
            initial_step_size=config.initial_step_size,
            target_accept_prob=config.target_accept,
            seed=config.seed,
            num_leapfrog_steps=config.num_leapfrog_steps,
        )
    elif config.dense_mass_matrix:
        # Full-covariance preconditioning. Same PreconditionedHamiltonianMonteCarlo
        # base as the diagonal route; the adaptation kernel differs only in
        # tracking the full covariance instead of its diagonal. Warmup-only.
        inner = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            step_size=step,
            num_leapfrog_steps=config.num_leapfrog_steps,
        )
        kernel = DenseMassMatrixAdaptation(
            inner_kernel=inner,
            initial_running_covariance=[
                tfp.experimental.stats.RunningCovariance.from_shape(
                    s.shape[1:], dtype=s.dtype)
                for s in initial_states
            ],
            num_estimation_steps=int(0.8 * config.num_warmup),
        )
    elif config.diagonal_mass_matrix:
        # Plain HamiltonianMonteCarlo exposes no momentum_distribution slot in
        # TFP 0.25.0, so diagonal preconditioning needs the Preconditioned
        # variant. Same fixed-trajectory HMC algorithm. Adaptation is
        # warmup-only.
        inner = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            step_size=step,
            num_leapfrog_steps=config.num_leapfrog_steps,
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
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            step_size=step,
            num_leapfrog_steps=config.num_leapfrog_steps,
        )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * config.num_warmup),
        target_accept_prob=tf.constant(config.target_accept, tf.float64),
    )

    def _hmc_results(pkr):
        """Descend to the Metropolis-Hastings results through adaptation wrappers.

        Wrapper depth depends on config: one (step size only) or two (step
        size over mass matrix adaptation).  `log_accept_ratio` is the sentinel
        because fixed-trajectory HMC has no `has_divergence` field; it lives on
        `MetropolisHastingsKernelResults` alongside `is_accepted`, while
        `step_size` sits one level deeper on `accepted_results`.
        """
        while not hasattr(pkr, "log_accept_ratio"):
            pkr = pkr.inner_results
        return pkr

    @tf.function(jit_compile=config.jit_compile)
    def _run():
        def _trace(_state, pkr):
            mh = _hmc_results(pkr)
            return {
                # Amendment A3: HMC divergence detection is a nonfinite
                # log-accept ratio, per fixed_trajectory_hmc_tuning_v2.py line
                # 209.  There is no NUTS-style `has_divergence` field.
                "diverged": tf.math.logical_not(
                    tf.math.is_finite(mh.log_accept_ratio)),
                "is_accepted": mh.is_accepted,
                # The DualAveragingStepSizeAdaptation wrapper is present for the
                # whole chain here (it merely stops updating after
                # `num_adaptation_steps`), so it keeps this field populated.
                "step_size": mh.accepted_results.step_size,
                "accept": mh.log_accept_ratio,
            }

        return tfp.mcmc.sample_chain(
            num_results=config.num_samples,
            num_burnin_steps=config.num_warmup,
            current_state=initial_states,
            kernel=adaptive,
            seed=tf.constant([config.seed, config.seed + 1], tf.int32),
            trace_fn=_trace,
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
