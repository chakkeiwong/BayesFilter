"""LEDH dual-force adapter for surrogate-force HMC validation (Phase 3).

Provides exact-force and damped-force evaluation on a single frozen particle-noise
realization ω, satisfying the Corollary 5.2 premise that F is a deterministic
function of θ alone.

This is a Phase 3 test fixture, not production code. It uses LGSSM d=3 T=10 N=100
as a minimal test case for seed-policy verification (V1 determinism, V2 reversibility,
V3 no-call-count dependence).
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


class LEDHDualForceAdapter:
    """
    LEDH dual-adapter for Phase 3 seed-policy tests.

    Freezes one particle-noise realization ω at construction, then provides
    exact-force and damped-force evaluation functions that both use that same ω.

    This is NOT production code — it's a Phase 3 test fixture for verifying
    Corollary 5.2 premises (determinism, reversibility, no call-count dependence).
    """

    def __init__(
        self,
        observations: tf.Tensor,  # [T, obs_dim]
        num_particles: int = 100,
        master_seed: int = 42,
        damping_epsilon: float = 0.01,
        sinkhorn_steps: int = 2,
        balance_steps: int = 2,
        transport_epsilon: float = 0.5,
        scaling: float = 0.9,
        prepared_ridge: float = 1.0e-6,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        Initialize with frozen master omega.

        Args:
            observations: [T, obs_dim] observed data
            num_particles: Particle count N
            master_seed: Seed for master omega, frozen for all evaluations
            damping_epsilon: Damping coefficient for the damped force
            sinkhorn_steps: Sinkhorn iterations in the reset
            balance_steps: Balance iterations in the reset
            transport_epsilon: Entropic regularization for the transport
            scaling: Cost scaling for the transport
            prepared_ridge: Contract-E ridge
            dtype: float32 or float64
        """
        self._dtype = tf.dtypes.as_dtype(dtype)
        self.observations = tf.convert_to_tensor(observations, dtype=self._dtype)
        self.num_particles = int(num_particles)
        self.master_seed = int(master_seed)
        self.damping_epsilon = float(damping_epsilon)
        self.sinkhorn_steps = int(sinkhorn_steps)
        self.balance_steps = int(balance_steps)

        horizon = int(self.observations.shape[0])
        self.steps = horizon

        # Transport chunks are a dataclass, not a mapping.
        chunks = select_transport_chunks(self.num_particles)
        self.row_chunk_size = chunks.row_chunk_size
        self.col_chunk_size = chunks.col_chunk_size

        # Reset is active at every step, ridged uniformly. Shapes are
        # [num_seeds, horizon] because preparation is seed-major.
        fixed_reset_mask = [[True] * horizon]
        ridge_rows = [[float(prepared_ridge)] * horizon]

        prepared_result = preparation.prepare_contract_e_lgssm_inputs(
            observations=self.observations,
            estimator_seeds=(self.master_seed,),
            num_particles=self.num_particles,
            fixed_reset_mask=fixed_reset_mask,
            prepared_ridge=ridge_rows,
            epsilon=float(transport_epsilon),
            scaling=float(scaling),
            sinkhorn_steps=self.sinkhorn_steps,
            balance_steps=self.balance_steps,
            row_chunk_size=self.row_chunk_size,
            col_chunk_size=self.col_chunk_size,
            dtype=self._dtype,
        )
        # The frozen omega lives under the "prepared" key.
        self._prepared = prepared_result["prepared"]

        self._value_and_score_fn = canonical.make_canonical_value_and_score_tf(
            self._prepared,
            steps=self.steps,
            balance_steps=self.balance_steps,
            row_chunk_size=self.row_chunk_size,
            col_chunk_size=self.col_chunk_size,
            jit_compile=False,
            dtype=self._dtype,
            cache_same_cloud_geometry=False,
        )

        self._call_count = 0

    def exact_force(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Exact LEDH force: gradient of the log-likelihood at frozen omega.

        Args:
            theta: [PARAMETER_COUNT] or [num_chains, PARAMETER_COUNT]

        Returns:
            force: same shape as theta
        """
        theta = tf.convert_to_tensor(theta, dtype=self._dtype)

        if theta.shape.rank == 2:
            forces = []
            for chain_theta in tf.unstack(theta, axis=0):
                self._call_count += 1
                forces.append(self._value_and_score_fn(chain_theta)["score"])
            return tf.stack(forces, axis=0)
        if theta.shape.rank == 1:
            self._call_count += 1
            return self._value_and_score_fn(theta)["score"]
        raise ValueError(
            f"theta must have rank 1 or 2, got rank {theta.shape.rank}"
        )

    def damped_force(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Damped LEDH force: the exact force scaled by 1/(1 + damping_epsilon).

        Args:
            theta: [PARAMETER_COUNT] or [num_chains, PARAMETER_COUNT]

        Returns:
            force: same shape as theta
        """
        exact = self.exact_force(theta)
        return exact / tf.cast(1.0 + self.damping_epsilon, exact.dtype)

    def exact_value(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Exact LEDH log-likelihood at frozen omega.

        This is the scalar the Metropolis step must use, per Corollary 5.2.

        Args:
            theta: [PARAMETER_COUNT] or [num_chains, PARAMETER_COUNT]

        Returns:
            log-likelihood: scalar, or [num_chains]
        """
        theta = tf.convert_to_tensor(theta, dtype=self._dtype)

        if theta.shape.rank == 2:
            values = []
            for chain_theta in tf.unstack(theta, axis=0):
                values.append(self._value_and_score_fn(chain_theta)["objective"])
            return tf.stack(values, axis=0)
        if theta.shape.rank == 1:
            return self._value_and_score_fn(theta)["objective"]
        raise ValueError(
            f"theta must have rank 1 or 2, got rank {theta.shape.rank}"
        )

    def reset_call_count(self):
        """Reset call count for V3 test."""
        self._call_count = 0

    def get_call_count(self) -> int:
        """Get current call count."""
        return self._call_count


def create_simple_lgssm_fixture(T: int = 10, seed: int = 999):
    """
    Create a minimal LGSSM fixture for Phase 3 tests.

    Returns observations [T, 3] matching the Phase 5 test fixture format.

    Returns:
        observations: [T, 3] tensor (float32)
    """
    rng = tf.random.Generator.from_seed(seed)
    # Simple random walk observation for testing
    observations = rng.normal([T, 3], dtype=tf.float32)
    return observations


__all__ = ["LEDHDualForceAdapter", "create_simple_lgssm_fixture"]
