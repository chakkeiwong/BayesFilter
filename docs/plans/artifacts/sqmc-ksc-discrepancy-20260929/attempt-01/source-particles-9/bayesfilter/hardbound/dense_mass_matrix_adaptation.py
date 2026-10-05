"""Dense (full covariance) mass matrix adaptation for HMC.

Custom TransitionKernel that wraps PreconditionedHamiltonianMonteCarlo and adapts
the momentum distribution using the full estimated posterior covariance.
Follows the pattern of TFP's DiagonalMassMatrixAdaptation but tracks
RunningCovariance instead of RunningVariance.

Required for targets where diagonal preconditioning cannot capture strong
posterior correlation structure (e.g. G2.3 where diagonal stalls at R-hat
~1.048 despite improving from 60.5). Amendment A3: applies to fixed-trajectory
HMC; NUTS replaced suite-wide.
"""

import collections

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distributions import (
    mvn_precision_factor_linop as mvn_pfl)
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

# Relative ridge added to the estimated covariance diagonal before the Cholesky
# factorization: `RIDGE_REL * trace(cov) / n`. Scale- and dimension-aware so it
# does not silently expire as the posterior scale or dimension changes. The
# dense estimate is rank-deficient until the sample count exceeds the part
# dimension (337 total here, 320 in the eta block), so the guard fires early in
# warmup and must not be removed.
RIDGE_REL = 1e-6


def hmc_like_momentum_distribution_setter_fn(kernel_results, new_distribution):
    """Setter for `momentum_distribution` so it can be adapted."""
    # Note that unnest.replace_innermost has a special path for going into
    # `accepted_results` preferentially, so this will set
    # `accepted_results.momentum_distribution`.
    return unnest.replace_innermost(
        kernel_results, momentum_distribution=new_distribution)


def hmc_like_momentum_distribution_getter_fn(kernel_results):
    """Getter for `momentum_distribution` so it can be updated."""
    return unnest.get_innermost(kernel_results, 'momentum_distribution')


class DenseMassMatrixAdaptation(tfp.mcmc.TransitionKernel):
    """Dense mass matrix adaptation during warmup.

    Estimates the full posterior covariance from samples during the warmup
    phase, then uses the Cholesky factor of the precision (inverse covariance)
    as the momentum distribution for PreconditionedHamiltonianMonteCarlo.

    Args:
        inner_kernel: PreconditionedHamiltonianMonteCarlo instance.
        initial_running_covariance: List of RunningCovariance objects, one per
            state part. Use RunningCovariance.from_shape(state_part.shape[1:])
            where state_part has shape [num_chains, ...].
        num_estimation_steps: Number of warmup steps to use for covariance
            estimation. Typically 0.8 * num_warmup.
    """

    def __init__(
        self,
        inner_kernel,
        initial_running_covariance,
        num_estimation_steps,
        name=None,
    ):
        self._parameters = dict(
            inner_kernel=inner_kernel,
            initial_running_covariance=initial_running_covariance,
            num_estimation_steps=num_estimation_steps,
            name=name or 'dense_mass_matrix_adaptation',
        )

    @property
    def inner_kernel(self):
        return self._parameters['inner_kernel']

    @property
    def initial_running_covariance(self):
        return self._parameters['initial_running_covariance']

    @property
    def num_estimation_steps(self):
        return self._parameters['num_estimation_steps']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        return self._parameters

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Execute one step of the kernel."""
        with tf.name_scope(self.name + '.one_step'):
            # Extract adaptation state from previous results
            inner_results = previous_kernel_results.inner_results
            step = previous_kernel_results.step
            covariance_parts = previous_kernel_results.covariance_parts

            # Run inner kernel
            new_state, new_inner_results = self.inner_kernel.one_step(
                current_state, inner_results, seed=seed)

            # Update covariance estimate if still in warmup
            new_step = step + 1
            is_adapting = new_step <= self.num_estimation_steps

            def update_covariance():
                """Update running covariance with new state sample."""
                new_covariance_parts = []
                for cov_part, state_part in zip(covariance_parts, new_state):
                    # state_part has shape [num_chains, ...], we want to update
                    # covariance across chains (axis=0)
                    new_cov = cov_part.update(state_part, axis=0)
                    new_covariance_parts.append(new_cov)
                return new_covariance_parts

            def keep_covariance():
                return covariance_parts

            new_covariance_parts = mcmc_util.choose(
                is_adapting,
                update_covariance(),
                covariance_parts,
            )

            # Update momentum distribution if we just finished warmup
            is_final_warmup_step = tf.equal(new_step, self.num_estimation_steps)

            def update_momentum():
                """Replace momentum distribution with full covariance precision."""
                covariances = [cov.covariance() for cov in new_covariance_parts]
                prev_momentum = hmc_like_momentum_distribution_getter_fn(
                    new_inner_results)
                new_momentum = update_dense_momentum_distribution(
                    prev_momentum, covariances)
                updated_inner = hmc_like_momentum_distribution_setter_fn(
                    new_inner_results, new_momentum)
                return updated_inner

            def keep_momentum():
                return new_inner_results

            final_inner_results = mcmc_util.choose(
                is_final_warmup_step,
                update_momentum(),
                new_inner_results,
            )

            new_kernel_results = previous_kernel_results._replace(
                inner_results=final_inner_results,
                step=new_step,
                covariance_parts=new_covariance_parts,
            )

            return new_state, new_kernel_results

    def bootstrap_results(self, init_state):
        """Initialize kernel results with a dense-structured initial momentum."""
        with tf.name_scope(self.name + '.bootstrap_results'):
            inner_results = self.inner_kernel.bootstrap_results(init_state)

            # Set the dense momentum distribution structure at the start,
            # so mcmc_util.choose can swap values later without structure change
            batch_shape = ps.shape(
                unnest.get_innermost(inner_results, 'target_log_prob'))
            init_state_parts = tf.nest.flatten(init_state)
            momentum_distribution = make_dense_momentum_distribution(
                init_state_parts, batch_shape, covariance_parts=None)
            inner_results = hmc_like_momentum_distribution_setter_fn(
                inner_results, momentum_distribution)

            kernel_results = DenseMassMatrixAdaptationResults(
                inner_results=inner_results,
                step=tf.constant(0, dtype=tf.int32),
                covariance_parts=self.initial_running_covariance,
            )

            return kernel_results

    @property
    def is_calibrated(self):
        return self.inner_kernel.is_calibrated


class DenseMassMatrixAdaptationResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('DenseMassMatrixAdaptationResults', [
        'inner_results',
        'step',
        'covariance_parts',
    ])):
    """Results of the DenseMassMatrixAdaptation TransitionKernel."""
    __slots__ = ()


def _flat_event_size(state_part, batch_ndims):
    """Event shape and flattened event size of a state part.

    `nevt` is returned as a Python `int` whenever the event shape is statically
    known, because `tf.eye` and `tf.linalg.LinearOperator` static-shape checks
    reject numpy integer scalars.
    """
    event_shape = state_part.shape[batch_ndims:]
    if not tensorshape_util.is_fully_defined(event_shape):
        event_shape = ps.shape(state_part)[batch_ndims:]
    nevt = ps.cast(ps.reduce_prod(event_shape), tf.int32)
    nevt_static = tf.get_static_value(nevt)
    if nevt_static is not None:
        nevt = int(nevt_static)
    return event_shape, nevt


def _ridged_cholesky(cov_matrix):
    """Cholesky factor of `cov_matrix` after relative ridge stabilization.

    The ridge is scale-aware: `RIDGE_REL * trace(cov) / n` on the diagonal, so
    it stays proportional to the average marginal variance rather than
    expiring silently as the posterior scale or dimension changes.
    """
    n = ps.shape(cov_matrix)[-1]
    trace = tf.linalg.trace(cov_matrix)
    ridge = RIDGE_REL * trace / tf.cast(n, cov_matrix.dtype)
    cov_ridged = tf.linalg.set_diag(
        cov_matrix, tf.linalg.diag_part(cov_matrix) + ridge[..., tf.newaxis])
    return tf.linalg.cholesky(cov_ridged)


def make_dense_momentum_distribution(state_parts, batch_shape,
                                     covariance_parts=None):
    """Construct a block-dense momentum distribution.

    Mirrors `preconditioning_utils.make_momentum_distribution` but uses
    `LinearOperatorFullMatrix` so the full within-part covariance structure is
    representable. Following TFP's convention, the momentum *precision* is the
    state covariance `Sigma` (mass matrix `M = Sigma^{-1}`, momentum
    `~ N(0, M)`), so `precision_factor = chol(Sigma)`.

    Args:
        state_parts: List of state `Tensor`s shaped `batch_shape + event_shape`.
        batch_shape: Batch shape of the chains.
        covariance_parts: Optional list of covariance `Tensor`s shaped
            `event_shape + event_shape` per part. Defaults to identity.

    Returns:
        `JointDistributionSequential` matching the structure consumed by
        `PreconditionedHamiltonianMonteCarlo` (Amendment A3; the same structure
        `PreconditionedNoUTurnSampler` took before NUTS was retired, which is why
        this builder needed no functional change).
    """
    batch_ndims = ps.rank_from_shape(batch_shape)
    model = []
    for i, state_part in enumerate(state_parts):
        event_shape, nevt = _flat_event_size(state_part, batch_ndims)
        if covariance_parts is None:
            chol = tf.linalg.eye(nevt, dtype=state_part.dtype)
        else:
            chol = _ridged_cholesky(
                tf.reshape(covariance_parts[i], ps.stack([nevt, nevt])))

        mvnpfl = mvn_pfl.MultivariateNormalPrecisionFactorLinearOperator(
            precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                chol, is_non_singular=True),
            precision=tf.linalg.LinearOperatorFullMatrix(
                tf.matmul(chol, chol, transpose_b=True),
                is_non_singular=True, is_self_adjoint=True,
                is_positive_definite=True))
        td = tfp.distributions.TransformedDistribution(
            distribution=mvnpfl,
            bijector=tfp.bijectors.Reshape(
                event_shape_out=event_shape, name='reshape_mvnpfl'))
        model.append(tfp.distributions.BatchBroadcast(
            td, with_shape=batch_shape))
    return tfp.distributions.JointDistributionSequential(model)


def update_dense_momentum_distribution(momentum_distribution,
                                       covariance_parts):
    """Swap new covariances into an existing block-dense momentum distribution.

    Uses `.copy()` on each nested distribution so the returned object has the
    identical composite-tensor structure as the input. That is required because
    `mcmc_util.choose` selects between the pre- and post-adaptation results
    leafwise inside a `tf.while_loop` body, which rejects a structure change.
    """
    if len(covariance_parts) != len(momentum_distribution.model):
        raise ValueError(
            'State size mismatch: '
            f'{len(covariance_parts)} vs {len(momentum_distribution.model)}')
    model = []
    for cov, bb in zip(covariance_parts, momentum_distribution.model):
        if not isinstance(bb, tfp.distributions.BatchBroadcast):
            raise ValueError(f'Part dist is not a BatchBroadcast: {bb}')
        td = bb.distribution
        if not isinstance(td, tfp.distributions.TransformedDistribution):
            raise ValueError(
                f'Inner dist is not a TransformedDistribution: {td}')
        mvnpfl = td.distribution
        if not isinstance(
                mvnpfl,
                mvn_pfl.MultivariateNormalPrecisionFactorLinearOperator):
            raise ValueError(
                'Inner dist is not a '
                f'MultivariateNormalPrecisionFactorLinearOperator: {mvnpfl}')
        nevt = ps.shape(mvnpfl.precision.to_dense())[-1]
        chol = _ridged_cholesky(tf.reshape(cov, [nevt, nevt]))
        mvnpfl = mvnpfl.copy(
            precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                chol, is_non_singular=True),
            precision=tf.linalg.LinearOperatorFullMatrix(
                tf.matmul(chol, chol, transpose_b=True),
                is_non_singular=True, is_self_adjoint=True,
                is_positive_definite=True))
        model.append(bb.copy(distribution=td.copy(distribution=mvnpfl)))
    return tfp.distributions.JointDistributionSequential(model)
