"""Phase 0: Smoke test for SQMC oracle comparison oracles.

Verifies that both LGSSM and KSC-SV oracles work before starting the full campaign.
"""

import tensorflow as tf
import numpy as np

from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_oracle_value_and_score
from bayesfilter.highdim.sv_mixture_cut4 import independent_panel_sv_mixture_kalman_filter


def test_lgssm_oracle():
    """Test LGSSM Kalman oracle with simple 1D system."""
    print("=" * 80)
    print("Testing LGSSM Kalman Oracle")
    print("=" * 80)

    # Simple 1D random walk + noise
    T = 20
    state_dim = 1
    obs_dim = 1

    # True parameters
    F = tf.constant([[0.9]], dtype=tf.float32)  # transition
    Q = tf.constant([[0.1]], dtype=tf.float32)  # process noise
    C = tf.constant([[1.0]], dtype=tf.float32)  # observation
    R = tf.constant([[0.2]], dtype=tf.float32)  # observation noise
    mu0 = tf.constant([0.0], dtype=tf.float32)
    P0 = tf.constant([[1.0]], dtype=tf.float32)

    # Generate synthetic observations
    np.random.seed(42)
    states = [np.array([0.0])]
    obs = []
    for t in range(T):
        if t > 0:
            states.append(0.9 * states[-1] + np.random.normal(0, np.sqrt(0.1), size=1))
        obs.append(states[-1] + np.random.normal(0, np.sqrt(0.2), size=1))

    observations = tf.constant(np.array(obs), dtype=tf.float32)

    # Test oracle - theta must be connected to params for gradient to work
    def theta_to_params(theta):
        # Make F depend on theta for gradient tracking
        # theta[0] will scale the transition
        scale = theta[0]
        return {
            'transition_matrix': F * scale,  # Now depends on theta
            'process_covariance': Q,
            'observation_matrix': C,
            'observation_covariance': R,
            'initial_mean': mu0,
            'initial_covariance': P0,
        }

    theta = tf.constant([1.0], dtype=tf.float32)  # parameter to differentiate w.r.t.
    result = kalman_oracle_value_and_score(observations, theta, theta_to_params, dtype=tf.float32)

    print(f"  Value (log-likelihood): {result['value'].numpy():.4f}")
    print(f"  Score (gradient): {result['score'].numpy()}")
    print(f"  Finite? {tf.math.is_finite(result['value']).numpy()}")
    print(f"  Score finite? {tf.reduce_all(tf.math.is_finite(result['score'])).numpy()}")

    if not tf.math.is_finite(result['value']).numpy():
        raise ValueError("LGSSM oracle returned non-finite value!")
    if not tf.reduce_all(tf.math.is_finite(result['score'])).numpy():
        raise ValueError("LGSSM oracle returned non-finite score!")

    print("✓ LGSSM oracle test passed\n")
    return result


def test_ksc_sv_oracle():
    """Test KSC-SV dense Kalman oracle with 1D SV."""
    print("=" * 80)
    print("Testing KSC-SV Dense Kalman Oracle")
    print("=" * 80)

    # Simple SV parameters
    T = 10
    gamma = 0.97  # persistence
    beta = 0.5    # volatility of volatility
    sigma = 0.15  # observation noise scale

    # Generate synthetic observations
    np.random.seed(43)
    obs = np.random.normal(0, 1, size=(T, 1))  # [T, 1] for single series
    observations = tf.constant(obs, dtype=tf.float64)

    # Test oracle
    result = independent_panel_sv_mixture_kalman_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    )

    print(f"  Log-likelihood: {result.log_likelihood.numpy():.4f}")
    print(f"  Final mean: {result.mean_path[-1].numpy()}")
    print(f"  Finite? {tf.math.is_finite(result.log_likelihood).numpy()}")

    if not tf.math.is_finite(result.log_likelihood).numpy():
        raise ValueError("KSC-SV oracle returned non-finite log-likelihood!")

    print("✓ KSC-SV oracle test passed\n")
    return result


def test_all_horizons():
    """Test both oracles at multiple horizons to verify feasibility."""
    print("=" * 80)
    print("Testing All Campaign Horizons")
    print("=" * 80)

    horizons_lgssm = [20, 50, 360]
    horizons_sv = [10, 20, 50, 120]

    print("\nLGSSM horizons:")
    for T in horizons_lgssm:
        # Simple 1D system
        F = tf.constant([[0.9]], dtype=tf.float32)
        Q = tf.constant([[0.1]], dtype=tf.float32)
        C = tf.constant([[1.0]], dtype=tf.float32)
        R = tf.constant([[0.2]], dtype=tf.float32)
        mu0 = tf.constant([0.0], dtype=tf.float32)
        P0 = tf.constant([[1.0]], dtype=tf.float32)

        np.random.seed(42)
        obs = np.random.normal(0, 1, size=(T, 1))
        observations = tf.constant(obs, dtype=tf.float32)

        def theta_to_params(theta):
            # Make parameters depend on theta for gradient tracking
            return {
                'transition_matrix': F * theta[0],
                'process_covariance': Q,
                'observation_matrix': C,
                'observation_covariance': R,
                'initial_mean': mu0,
                'initial_covariance': P0,
            }

        theta = tf.constant([1.0], dtype=tf.float32)
        result = kalman_oracle_value_and_score(observations, theta, theta_to_params, dtype=tf.float32)
        print(f"  T={T:3d}: loglik={result['value'].numpy():8.2f}, finite={tf.math.is_finite(result['value']).numpy()}")

    print("\nKSC-SV horizons:")
    for T in horizons_sv:
        np.random.seed(43)
        obs = np.random.normal(0, 1, size=(T, 1))
        observations = tf.constant(obs, dtype=tf.float64)

        result = independent_panel_sv_mixture_kalman_filter(
            observations,
            gamma=0.97,
            beta=0.5,
            sigma=0.15,
        )
        print(f"  T={T:3d}: loglik={result.log_likelihood.numpy():8.2f}, finite={tf.math.is_finite(result.log_likelihood).numpy()}")

    print("\n✓ All horizons feasible\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SQMC Oracle Comparison - Phase 0 Verification")
    print("=" * 80 + "\n")

    try:
        test_lgssm_oracle()
        test_ksc_sv_oracle()
        test_all_horizons()

        print("=" * 80)
        print("✓ Phase 0 verification PASSED")
        print("=" * 80)
        print("\nReady to proceed to Phase 1 execution.\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Phase 0 verification FAILED")
        print("=" * 80)
        print(f"\nError: {e}\n")
        raise
