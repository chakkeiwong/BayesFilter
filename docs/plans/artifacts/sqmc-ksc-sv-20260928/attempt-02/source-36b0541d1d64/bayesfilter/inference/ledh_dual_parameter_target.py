"""Dual-parameter LEDH HMC target for Corollary 5.2 surrogate-force HMC.

Implements the variance note Corollary 5.2 (lines 953-963): HMC with exact
executed potential but biased (cheaper) force. Value computed with exact
parameters; score computed with larger damping parameters.

Authority: bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex
Phase: Phase 3 Task 3.1a of ledh-surrogate-hmc-unified-program-2026-09-06.md
Date: 2026-09-11
"""

import tensorflow as tf
from typing import Callable

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,
    PerPointScoreModel,
)


class DualParameterLEDHTarget:
    """Corollary 5.2 surrogate-force HMC target.

    Computes value with exact parameters and score with biased (larger damping)
    parameters. The HMC chain targets exp(-U) where U is the exact executed
    potential, while using a cheaper (more damped) force for mixing.

    Corollary 5.2 guarantees invariance: the θ-marginal π(θ) ∝ exp(-U(θ)) is
    preserved regardless of force quality. Force affects only acceptance rate
    and mixing efficiency.

    Parameters
    ----------
    model : PerPointScoreModel
        The LEDH model (transition, observation functions)
    initial_states : Tensor
        Initial state particles [N, d]
    initial_covariances : Tensor
        Initial covariance particles [N, d, d]
    noises : Tensor
        Frozen process noises [T, N, d] - MUST be deterministic (Corollary 5.2)
    observations : Tensor
        Observations [T, obs_dim]
    exact_params : dict
        Parameters for exact value computation
    biased_params : dict
        Parameters for biased score computation (larger damping)
    """

    def __init__(
        self,
        model: PerPointScoreModel,
        initial_states: tf.Tensor,
        initial_covariances: tf.Tensor,
        noises: tf.Tensor,
        observations: tf.Tensor,
        *,
        exact_params: dict,
        biased_params: dict,
    ):
        self.model = model
        self.initial_states = initial_states
        self.initial_covariances = initial_covariances
        self.noises = noises
        self.observations = observations
        self.exact_params = exact_params
        self.biased_params = biased_params
        self._graph_callable = None

        # Verify noises are frozen (deterministic)
        if not isinstance(noises, tf.Tensor):
            raise TypeError("noises must be a Tensor (frozen, not generated)")

        # Verify parameter keys match
        exact_keys = set(exact_params.keys())
        biased_keys = set(biased_params.keys())
        if exact_keys != biased_keys:
            raise ValueError(
                f"Parameter key mismatch: exact={exact_keys}, biased={biased_keys}"
            )

    def __call__(self, theta: tf.Tensor) -> tf.Tensor:
        """Evaluate HMC target with custom gradient (Corollary 5.2).

        Returns exact value, but autodiff will get biased (surrogate) gradient.
        This is correct per Corollary 5.2: HMC targets exp(-U_exact) while
        using cheaper (damped) force for mixing.

        Parameters
        ----------
        theta : Tensor, shape [P]
            Parameter vector (HMC always passes rank-1)

        Returns
        -------
        value : Tensor, shape []
            Exact log-likelihood (from exact_params)
        """
        @tf.custom_gradient
        def target_with_surrogate_gradient(theta_inner):
            # Ensure rank-2 for batch API
            theta_batch = theta_inner[None, :]  # [P] -> [1, P]
            param_dim = theta_inner.shape[0]
            full_directions = tf.eye(param_dim, dtype=theta_inner.dtype)[None, :, :]

            # Compute exact value (for HMC acceptance).
            #
            # The engine evaluates directional derivatives ONE AT A TIME
            # (tf.map_fn over K with parallel_iterations=1), so each extra
            # direction costs a full T x substeps filter pass. This call needs
            # only the primal value, and the engine asserts the primal value is
            # direction-invariant to rtol=1e-12, so a single direction returns
            # the identical value at 1/P the cost.
            value_direction = full_directions[:, :1, :]  # [1, 1, P]
            exact_value, _, _ = canonical_batch_fused_value_score(
                self.model,
                theta_batch,
                value_direction,
                self.initial_states,
                self.initial_covariances,
                self.noises,
                self.observations,
                **self.exact_params,
            )
            exact_value = exact_value[0]  # [1] -> []

            # Compute biased score (for HMC force). This needs every coordinate
            # direction, so K=P passes are required here.
            _, biased_score, _ = canonical_batch_fused_value_score(
                self.model,
                theta_batch,
                full_directions,
                self.initial_states,
                self.initial_covariances,
                self.noises,
                self.observations,
                **self.biased_params,
            )
            biased_score = biased_score[0]  # [1, P] -> [P]

            # Check for -inf (graceful failure from numerical instability)
            # When exact_value is -inf, return -inf with zero gradient
            is_invalid = tf.math.is_inf(exact_value) & (exact_value < 0.0)

            def grad_fn(dy):
                # Return biased gradient (surrogate force) when valid
                # Return zero gradient when invalid (-inf)
                # dy is upstream gradient (scalar for log-prob)
                zero_grad = tf.zeros_like(biased_score)
                return tf.where(is_invalid, zero_grad, dy * biased_score)

            return exact_value, grad_fn

        return target_with_surrogate_gradient(theta)

    def as_graph_callable(self, param_dim: int) -> Callable[[tf.Tensor], tf.Tensor]:
        """Return this target compiled with a stable ``input_signature``.

        The repo TensorFlow Graph policy requires repeated numerical kernels to
        execute through ``tf.function`` with an explicit, stable signature.
        Eager evaluation of this target dispatches every one of the
        ``T x substeps`` sequential stages from Python, which measured 8.65x
        slower than the compiled path at T=5 (see
        ``docs/benchmarks/ledh_cost_structure_probe.py``).

        Compile at THIS granularity - one value+gradient evaluation - not around
        ``tfp.mcmc.sample_chain``. Wrapping the chain unrolls every HMC step into
        a single graph and exhausts host memory; that failure is what led to the
        eager workaround being adopted, and it was a granularity error, not
        evidence against graph mode. Here the signature is fixed (one ``[P]``
        vector), so the graph is traced once and reused for every leapfrog step.

        The returned callable is cached, so repeated calls reuse one trace.

        Parameters
        ----------
        param_dim : int
            Static parameter dimension P. Required for the input signature;
            passing the wrong value raises at call time rather than retracing.

        Returns
        -------
        callable
            ``theta[P] -> value[]``, carrying the same surrogate gradient as
            ``__call__``. Verified against the eager path by
            ``docs/benchmarks/ledh_graph_mode_gradient_verification.py``.
        """
        if self._graph_callable is None:
            dtype = self.observations.dtype

            @tf.function(input_signature=[tf.TensorSpec([param_dim], dtype)])
            def graph_target(theta: tf.Tensor) -> tf.Tensor:
                return self(theta)

            self._graph_callable = graph_target
        return self._graph_callable

    def value_only(self, theta: tf.Tensor) -> tf.Tensor:
        """Compute only exact value (for diagnostics)."""
        theta = tf.convert_to_tensor(theta)
        if theta.shape.rank == 1:
            theta = theta[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = tf.shape(theta)[0]
        param_dim = theta.shape[1]
        # Value only: one direction suffices (the engine asserts the primal
        # value is direction-invariant), and each extra direction costs a full
        # filter pass. See the note in __call__.
        directions = tf.eye(param_dim, dtype=theta.dtype)[None, :1, :]
        directions = tf.broadcast_to(directions, [batch_size, 1, param_dim])

        value, _, _ = canonical_batch_fused_value_score(
            self.model,
            theta,
            directions,
            self.initial_states,
            self.initial_covariances,
            self.noises,
            self.observations,
            **self.exact_params,
        )

        if squeeze_output:
            value = tf.squeeze(value, axis=0)

        return value

    def score_only(self, theta: tf.Tensor, *, exact: bool = False) -> tf.Tensor:
        """Compute score (for diagnostics).

        Parameters
        ----------
        theta : Tensor
            Parameter vector(s)
        exact : bool, default False
            If True, use exact_params; if False, use biased_params

        Returns
        -------
        score : Tensor
            Gradient vector(s)
        """
        theta = tf.convert_to_tensor(theta)
        if theta.shape.rank == 1:
            theta = theta[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = tf.shape(theta)[0]
        param_dim = theta.shape[1]
        directions = tf.eye(param_dim, dtype=theta.dtype)[None, :, :]
        directions = tf.broadcast_to(directions, [batch_size, param_dim, param_dim])

        params = self.exact_params if exact else self.biased_params

        _, score, _ = canonical_batch_fused_value_score(
            self.model,
            theta,
            directions,
            self.initial_states,
            self.initial_covariances,
            self.noises,
            self.observations,
            **params,
        )

        if squeeze_output:
            score = tf.squeeze(score, axis=0)

        return score


def make_dual_parameter_target(
    model: PerPointScoreModel,
    initial_states: tf.Tensor,
    initial_covariances: tf.Tensor,
    noises: tf.Tensor,
    observations: tf.Tensor,
    *,
    damping_ratio: float = 100.0,
    base_reset_ridge: float = 1e-5,
    base_lm_damping: float = 1e-2,
    **shared_params,
) -> DualParameterLEDHTarget:
    """Construct dual-parameter target with damping ratio.

    Parameters
    ----------
    damping_ratio : float, default 100.0
        Biased parameters = base parameters × damping_ratio
    base_reset_ridge : float, default 1e-5
        Exact reset_ridge (biased = base × damping_ratio)
    base_lm_damping : float, default 1e-2
        Exact correction_lm_damping (biased = base × damping_ratio)
    **shared_params
        Other parameters (substeps, reset_policy, etc.) shared by both calls

    Returns
    -------
    target : DualParameterLEDHTarget
        HMC target with exact value, biased score
    """
    exact_params = {
        **shared_params,
        'reset_ridge': base_reset_ridge,
        'correction_lm_damping': base_lm_damping,
    }

    biased_params = {
        **shared_params,
        'reset_ridge': base_reset_ridge * damping_ratio,
        'correction_lm_damping': base_lm_damping * damping_ratio,
    }

    return DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )
