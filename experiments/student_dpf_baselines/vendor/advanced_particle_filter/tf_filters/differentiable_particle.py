"""
Differentiable TensorFlow Particle Filter with pluggable resampler.

Supports two differentiable resamplers:
  - "soft":     straight-through soft resampling (Jonschkowski 2018,
                Corenflos 2021). Low memory, fast. Default.
  - "sinkhorn": entropy-regularised OT resampling (Corenflos 2021).
                Theoretically cleaner but O(N_s * B * N^2) backward
                memory — OOMs on consumer GPUs at realistic T.

See docs/memory_compute_analysis.pdf for the tradeoff analysis.

Structure mirrors tf_filters/particle.py (TFBootstrapParticleFilter):
  - class with __init__ settings and filter() entry point
  - heavy work in @tf.function-decorated _filter_impl
  - tf.while_loop over time (no Python for-loops)
  - tf.random.Generator passed in explicitly (no global state)

Deviations from TFBootstrapParticleFilter:
  - Particle shape [B, N, D] rather than [N, nx] to support batched
    parallel filtering (HMC chains x MC replicas).
  - Every step resamples unconditionally. Conditional resampling would
    require tf.cond inside the loop, complicating gradient flow.
  - Parameters flow through the filter as traced tensors (not
    tf.constants), so gradients propagate to them. This is the key
    enabler for HMC.
"""

import tensorflow as tf
from typing import NamedTuple

from ..tf_models.svssm import (
    SVSSMParams,
    svssm_sample_initial,
    svssm_sample_dynamics,
    svssm_observation_log_prob,
)
from ..tf_utils.sinkhorn import (
    sinkhorn_resample,
    batched_normalize_log_weights,
)
from ..tf_utils.soft_resampler import (
    soft_resample,
)


class DPFResult(NamedTuple):
    """
    Differentiable particle filter output.

    Attributes:
        log_evidence:    [B]         total log marginal likelihood per batch
        final_particles: [B, N, D]   particles after the last step
        final_log_w:     [B, N]      log-weights after the last step
    """
    log_evidence: tf.Tensor
    final_particles: tf.Tensor
    final_log_w: tf.Tensor


class TFDifferentiableParticleFilter:
    """
    Differentiable bootstrap-style particle filter. Batched over a
    leading dimension B for GPU parallelism and multi-chain HMC.

    Example:
        dpf = TFDifferentiableParticleFilter(
            n_particles=100, resampler="soft", alpha=0.5, dtype=tf.float64,
        )
        result = dpf.filter(params, observations, rng)
        log_ev = result.log_evidence  # [B]
    """

    def __init__(
        self,
        n_particles: int = 100,
        resampler: str = "soft",
        # soft-resampler hyperparameters
        alpha: float = 0.5,
        # sinkhorn hyperparameters
        epsilon: float = 0.1,
        sinkhorn_iters: int = 50,
        # amortized OT hyperparameters
        amortized_ckpt_dir: str = None,
        amortized_eps: float = 0.5,
        amortized_d: int = None,
        obs_log_prob_fn=None,
        dtype: tf.DType = tf.float64,
    ):
        """
        Args:
            n_particles:    Particles per batch element.
            resampler:      "soft" (default), "sinkhorn", or "amortized".
            alpha:          Soft-resampler mixing coefficient in (0, 1].
                            1 = pure categorical (no gradient smoothness);
                            0 = pure uniform (ignores weights).
            epsilon:        Sinkhorn entropic regularisation.
            sinkhorn_iters: Sinkhorn iteration count.
            amortized_ckpt_dir: Path to the trained CouplingOperator's
                            checkpoint root. Required iff resampler='amortized'.
            amortized_eps:  Eps fed to the amortized operator's eps-conditioning.
                            Default 0.5 (Corenflos default; matches training).
                            Note: this is NOT the Sinkhorn epsilon.
            amortized_d:    State dimension for the amortized operator.
                            Required iff resampler='amortized'. Must match
                            the trained operator (currently d=2 only).
            obs_log_prob_fn: Optional callable (y_t, particles) -> [B, N].
                            If None, uses svssm_observation_log_prob (default).
                            Pass a different function for linear-Gaussian
                            observation model or other SSM variants.
            dtype:          tf.float64 recommended for HMC numerical
                            stability.
        """
        if resampler not in ("soft", "sinkhorn", "amortized"):
            raise ValueError(
                f"resampler must be 'soft', 'sinkhorn', or 'amortized', "
                f"got {resampler!r}"
            )
        self.n_particles = n_particles
        self.resampler = resampler
        self.alpha = alpha
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.amortized_eps = amortized_eps
        self.obs_log_prob_fn = obs_log_prob_fn if obs_log_prob_fn is not None \
            else svssm_observation_log_prob
        self.dtype = dtype

        # Construct the amortized resampler at __init__ time so the
        # checkpoint load (heavy, one-time) happens here, not on every
        # filter call.
        self.amortized_resampler = None
        if resampler == "amortized":
            if amortized_d is None:
                raise ValueError(
                    "resampler='amortized' requires amortized_d "
                    "(state dimension)."
                )
            from ..tf_utils.amortized_resampler import AmortizedOTResampler
            self.amortized_resampler = AmortizedOTResampler(
                ckpt_dir=amortized_ckpt_dir,    # None -> use vendored default
                d=amortized_d,
                N=n_particles,
                eps=amortized_eps,
                dtype=dtype,
            )

    def filter(
        self,
        params: SVSSMParams,
        observations: tf.Tensor,
        rng: tf.random.Generator,
    ) -> DPFResult:
        """
        Run the differentiable PF over an observation sequence.

        Args:
            params:       SVSSMParams with leading batch dim B.
            observations: [T, d] observation sequence (shared across batch).
            rng:          tf.random.Generator.

        Returns:
            DPFResult
        """
        if self.resampler == "soft":
            alpha_t = tf.constant(self.alpha, dtype=self.dtype)
            return self._filter_impl_soft(
                params.mu, params.Phi, params.Sigma_eta_chol,
                observations, rng, alpha_t,
            )
        elif self.resampler == "sinkhorn":
            eps = tf.constant(self.epsilon, dtype=self.dtype)
            # sinkhorn_iters stays Python int so sinkhorn_resample can
            # unroll its inner loop for XLA fusion.  self.sinkhorn_iters
            # is captured by closure in _filter_impl_sinkhorn's body.
            return self._filter_impl_sinkhorn(
                params.mu, params.Phi, params.Sigma_eta_chol,
                observations, rng, eps,
            )
        else:  # amortized
            return self._filter_impl_amortized(
                params.mu, params.Phi, params.Sigma_eta_chol,
                observations, rng,
            )

    # ------------------------------------------------------------------
    # Soft resampling implementation (XLA-fused inner step)
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _filter_impl_soft(
        self,
        mu: tf.Tensor,
        Phi: tf.Tensor,
        Sigma_eta_chol: tf.Tensor,
        observations: tf.Tensor,
        rng: tf.random.Generator,
        alpha: tf.Tensor,
    ) -> DPFResult:
        """
        Fully-traced forward pass using soft resampling.

        Memory-efficient noise generation: instead of pre-allocating
        [T, B, N, N] Gumbel noise and [T, B, N, D] dynamics noise (which
        would be O(T) memory — prohibitive at T > ~150 at large N), we
        derive a root seed from `rng` once, then use stateless RNG inside
        each step, seeded by (root_seed, t). This gives O(1) memory for
        noise (only the current step's noise lives on the GPU) while
        preserving determinism given the root seed.

        The outer tf.while_loop stays in plain @tf.function; only the
        per-step body runs under jit_compile=True. Gradients flow through
        the ST-soft-resample and reweight as before — stateless_normal /
        stateless_uniform are non-differentiable w.r.t. the seed, which
        is correct: the noise is a constant at each forward pass from
        autodiff's perspective.
        """
        params = SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)

        dtype = self.dtype
        T = tf.shape(observations)[0]
        B_static = mu.shape[0]
        B = tf.shape(mu)[0]
        D = mu.shape[1]
        N = self.n_particles
        N_f = tf.cast(N, dtype)

        # --- Derive a single root seed from the stateful RNG ---
        # `make_seeds` returns a [2, k] int32 tensor; we want one [2] seed.
        # Using make_seeds(3)[:, 0] gives a deterministic seed that advances
        # the rng state by one "draw" regardless of T.
        root_seeds = rng.make_seeds(3)                  # [2, 3] int32
        seed_init    = root_seeds[:, 0]                 # [2] for initial particles
        seed_gumbel  = root_seeds[:, 1]                 # [2] for per-step Gumbel
        seed_dyn     = root_seeds[:, 2]                 # [2] for per-step dynamics

        # --- Initial particles (stateless, reproducible) ---
        init_noise = tf.random.stateless_normal(
            shape=[B, N, D], seed=seed_init, dtype=dtype,
        )
        noise_scaled = tf.einsum('bij,bnj->bni', Sigma_eta_chol, init_noise)
        particles_init = mu[:, tf.newaxis, :] + noise_scaled

        log_w_init = tf.fill([B, N], -tf.math.log(N_f))
        log_ev_init = tf.zeros([B], dtype=dtype)
        t_init = tf.constant(0, dtype=tf.int32)

        # --- XLA-compiled per-step kernel ---
        obs_fn = self.obs_log_prob_fn

        @tf.function(jit_compile=True)
        def _xla_step(particles, log_w, y_t, gumbel_t, dyn_noise_t,
                      mu_, Phi_, L_chol_, alpha_):
            """One PF step: resample -> propagate -> reweight. Pure tensors."""
            B_ = tf.shape(particles)[0]
            N_ = tf.shape(particles)[1]
            N_f_ = tf.cast(N_, dtype)

            # 1. Soft resample (deterministic given gumbel_t)
            w_norm = tf.exp(log_w)
            w_uniform = tf.ones_like(w_norm) / N_f_
            w_soft = alpha_ * w_norm + (1.0 - alpha_) * w_uniform
            log_w_soft = tf.math.log(w_soft + 1e-30)

            # Gumbel-max sampling: gumbel_t is [B, N, N]
            scored = log_w_soft[:, tf.newaxis, :] + gumbel_t  # [B, N, N]
            ancestors = tf.cast(tf.argmax(scored, axis=-1), tf.int32)  # [B, N]

            # Gather
            batch_idx = tf.broadcast_to(
                tf.range(B_, dtype=tf.int32)[:, tf.newaxis], tf.shape(ancestors)
            )
            gather_idx = tf.stack([batch_idx, ancestors], axis=-1)
            particles_hard = tf.gather_nd(particles, gather_idx)

            # ST surrogate
            x_interp = tf.reduce_sum(
                w_soft[:, :, tf.newaxis] * particles, axis=1, keepdims=True
            )
            x_interp = tf.broadcast_to(x_interp, tf.shape(particles))
            particles_r = tf.stop_gradient(particles_hard - x_interp) + x_interp

            # Weight correction
            log_w_ancestors = tf.gather_nd(log_w, gather_idx)
            log_w_soft_ancestors = tf.gather_nd(log_w_soft, gather_idx)
            log_w_r = log_w_ancestors - log_w_soft_ancestors
            log_w_r = log_w_r - tf.math.reduce_logsumexp(
                log_w_r, axis=-1, keepdims=True
            )

            # 2. Propagate (deterministic given dyn_noise_t)
            centered = particles_r - mu_[:, tf.newaxis, :]
            propagated = tf.einsum('bij,bnj->bni', Phi_, centered)
            mean = mu_[:, tf.newaxis, :] + propagated
            noise = tf.einsum('bij,bnj->bni', L_chol_, dyn_noise_t)
            particles_next = mean + noise

            # 3. Reweight
            log_lik = obs_fn(y_t, particles_next)
            log_w_unnorm = log_w_r + log_lik

            # Normalize
            max_lw = tf.reduce_max(log_w_unnorm, axis=-1, keepdims=True)
            shifted = log_w_unnorm - max_lw
            log_sum = tf.math.log(tf.reduce_sum(tf.exp(shifted), axis=-1,
                                                keepdims=True))
            log_ev_t = tf.squeeze(max_lw + log_sum, axis=-1)  # [B]
            log_w_new = shifted - log_sum

            return particles_next, log_w_new, log_ev_t

        # --- Outer while_loop (plain tf.function, no XLA) ---
        # Each step generates its own noise via stateless RNG seeded by
        # (root_seed, t). Memory footprint per step: [B, N, N] for Gumbel
        # and [B, N, D] for dynamics — independent of T.
        def body(t, particles, log_w, log_ev):
            y_t = observations[t]

            # Per-step seeds: fold t into the root seed deterministically.
            # stateless_fold_in maps (seed, int) -> fresh int32[2], suitable
            # for stateless_* RNGs. Different root seeds give independent
            # RNG streams; the same root_seed + same t always gives the
            # same noise.
            gumbel_seed_t = tf.random.experimental.stateless_fold_in(
                seed_gumbel, t
            )
            dyn_seed_t = tf.random.experimental.stateless_fold_in(
                seed_dyn, t
            )

            # Per-step Gumbel noise [B, N, N]
            gumbel_u = tf.random.stateless_uniform(
                shape=[B, N, N],
                seed=gumbel_seed_t,
                minval=tf.constant(1e-20, dtype=dtype),
                maxval=tf.constant(1.0, dtype=dtype),
                dtype=dtype,
            )
            gumbel_t = -tf.math.log(-tf.math.log(gumbel_u))

            # Per-step dynamics noise [B, N, D]
            dyn_noise_t = tf.random.stateless_normal(
                shape=[B, N, D], seed=dyn_seed_t, dtype=dtype,
            )

            particles_next, log_w_new, log_ev_t = _xla_step(
                particles, log_w, y_t,
                gumbel_t, dyn_noise_t,
                mu, Phi, Sigma_eta_chol, alpha,
            )
            return t + 1, particles_next, log_w_new, log_ev + log_ev_t

        _, final_particles, final_log_w, total_log_ev = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(t_init, particles_init, log_w_init, log_ev_init),
            shape_invariants=(
                t_init.shape,
                tf.TensorShape([B_static, N, D]),
                tf.TensorShape([B_static, N]),
                tf.TensorShape([B_static]),
            ),
            parallel_iterations=1,
            maximum_iterations=T,
        )

        return DPFResult(
            log_evidence=total_log_ev,
            final_particles=final_particles,
            final_log_w=final_log_w,
        )

    # ------------------------------------------------------------------
    # Sinkhorn-OT resampling implementation
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _filter_impl_sinkhorn(
        self,
        mu: tf.Tensor,
        Phi: tf.Tensor,
        Sigma_eta_chol: tf.Tensor,
        observations: tf.Tensor,
        rng: tf.random.Generator,
        epsilon: tf.Tensor,
    ) -> DPFResult:
        """Fully-traced forward pass using Sinkhorn-OT resampling.

        NOTE ON PERFORMANCE
        -------------------
        sinkhorn_iters is read from self (Python int) rather than passed
        as a tf.Tensor. sinkhorn_resample itself is @tf.function with
        jit_compile=True and unrolls its inner iteration loop in Python
        at trace time so XLA can fuse all rounds into a single kernel
        (see tf_utils/sinkhorn.py). This is the main compute speedup path.

        The outer time loop below stays as plain tf.while_loop (no JIT)
        because backward through a JIT'd while_loop requires passing
        TensorList across the XLA/TF boundary, which is not supported.

        WARNING: peak backward-activation memory is O(T * N_s * B * N^2);
        OOMs on Colab T4 at realistic T, N. See docs/memory_compute_analysis.
        Prefer "soft" unless OT properties are strictly needed.
        """
        params = SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)

        dtype = self.dtype
        T = tf.shape(observations)[0]
        B_static = mu.shape[0]
        B = tf.shape(mu)[0]
        D = mu.shape[1]
        N = self.n_particles
        N_f = tf.cast(N, dtype)

        # Capture sinkhorn_iters as Python int for XLA-unrolled inner loop
        n_iters_py = int(self.sinkhorn_iters)

        particles_init = svssm_sample_initial(params, N, rng)
        log_w_init = tf.fill([B, N], -tf.math.log(N_f))
        log_ev_init = tf.zeros([B], dtype=dtype)
        t_init = tf.constant(0, dtype=tf.int32)

        def body(t, particles, log_w, log_ev):
            particles_r, log_w_r = sinkhorn_resample(
                particles, log_w, epsilon, n_iters_py
            )
            particles_next = svssm_sample_dynamics(particles_r, params, rng)
            y_t = observations[t]
            log_lik = self.obs_log_prob_fn(y_t, particles_next)
            log_w_unnorm = log_w_r + log_lik
            log_w_new, log_ev_t = batched_normalize_log_weights(log_w_unnorm)
            return t + 1, particles_next, log_w_new, log_ev + log_ev_t

        _, final_particles, final_log_w, total_log_ev = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(t_init, particles_init, log_w_init, log_ev_init),
            shape_invariants=(
                t_init.shape,
                tf.TensorShape([B_static, N, D]),
                tf.TensorShape([B_static, N]),
                tf.TensorShape([B_static]),
            ),
            parallel_iterations=1,
            maximum_iterations=T,
        )

        return DPFResult(
            log_evidence=total_log_ev,
            final_particles=final_particles,
            final_log_w=final_log_w,
        )

    # ------------------------------------------------------------------
    # Amortized OT resampling implementation
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _filter_impl_amortized(
        self,
        mu: tf.Tensor,
        Phi: tf.Tensor,
        Sigma_eta_chol: tf.Tensor,
        observations: tf.Tensor,
        rng: tf.random.Generator,
    ) -> DPFResult:
        """
        Forward pass using the amortized OT resampler.

        The amortized resampler is an attention-based neural operator
        trained to mimic Sinkhorn's barycentric projection. Its forward
        pass is one network evaluation per resampling call (~3 ms on A100
        for B=1, N=1000), as opposed to ~300 ms for full Sinkhorn at the
        same scale.

        Memory: O(B * N^2) for the attention coupling matrix per step.
        Same order as Sinkhorn but only one round (not 1000 iterations'
        worth of activations), so wall-clock and peak memory both improve.

        Gradient flow: gradients backprop through self.amortized_resampler,
        which internally backprops through the Set Transformer, hypernet,
        and coupling head. The resampler does NOT support gradients w.r.t.
        the model parameters at HMC time (those are frozen post-training);
        gradients only flow w.r.t. the input particles.
        """
        params = SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)

        dtype = self.dtype
        T = tf.shape(observations)[0]
        B_static = mu.shape[0]
        B = tf.shape(mu)[0]
        D = mu.shape[1]
        N = self.n_particles
        N_f = tf.cast(N, dtype)

        particles_init = svssm_sample_initial(params, N, rng)
        log_w_init = tf.fill([B, N], -tf.math.log(N_f))
        log_ev_init = tf.zeros([B], dtype=dtype)
        t_init = tf.constant(0, dtype=tf.int32)

        # Capture the resampler in closure (cannot pass non-tensor through
        # tf.while_loop loop_vars).
        resampler = self.amortized_resampler
        obs_log_prob_fn = self.obs_log_prob_fn

        def body(t, particles, log_w, log_ev):
            # 1. Amortized resample (deterministic — no RNG inside the network).
            particles_r, log_w_r = resampler(particles, log_w)

            # 2. Propagate.
            particles_next = svssm_sample_dynamics(particles_r, params, rng)

            # 3. Reweight.
            y_t = observations[t]
            log_lik = obs_log_prob_fn(y_t, particles_next)
            log_w_unnorm = log_w_r + log_lik
            log_w_new, log_ev_t = batched_normalize_log_weights(log_w_unnorm)
            return t + 1, particles_next, log_w_new, log_ev + log_ev_t

        _, final_particles, final_log_w, total_log_ev = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(t_init, particles_init, log_w_init, log_ev_init),
            shape_invariants=(
                t_init.shape,
                tf.TensorShape([B_static, N, D]),
                tf.TensorShape([B_static, N]),
                tf.TensorShape([B_static]),
            ),
            parallel_iterations=1,
            maximum_iterations=T,
        )

        return DPFResult(
            log_evidence=total_log_ev,
            final_particles=final_particles,
            final_log_w=final_log_w,
        )
