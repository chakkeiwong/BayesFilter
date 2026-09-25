"""Phase 3.5.2 Measurement: Substep while_loop performance.

Measures trace time, steady state, and graph size after substep loop conversion.

Target metrics (from master program Phase 3.5):
- Trace time: <50s (baseline: 485s)
- Steady state: ≤80s (baseline: 39.3s)
- Graph size: O(10³) nodes (baseline: ~400 flow stages unrolled)
"""

import os
import time
import tensorflow as tf
import numpy as np
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.ledh_canonical_models_tf import austria_sir_canonical_model

# CPU-only for measurement reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Test configuration - plan scale (matches baseline measurement)
N_PARTICLES = 252
TIME_HORIZON = 50
SUBSTEPS = 8
ANNEALED_STAGES = 1
STATE_DIM = 18  # Austria SIR state dimension
OBS_DIM = 9     # Austria SIR observation dimension
SEED = 42


def create_test_model():
    """Create Austria SIR model for measurement.

    Austria SIR requires theta at runtime, but tangent direction is baked in.
    The direction vector must be set via set_direction before tf.function tracing.
    """
    # Use a fixed theta for Austria SIR
    theta_fixed = tf.constant([0.5, 0.3, 0.1], dtype=tf.float32)
    model, set_direction = austria_sir_canonical_model(theta_fixed, dtype=tf.float32)
    # Set direction to [1, 0, 0] for the first parameter
    d_theta = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
    set_direction(d_theta)  # Pass the direction tensor, not an index
    return model, theta_fixed, d_theta


def measure_trace_and_steady(n: int, t: int, substeps: int, state_dim: int, obs_dim: int) -> dict:
    """Measure trace time and steady-state time."""
    model, theta, d_theta = create_test_model()

    # Generate test data
    np.random.seed(SEED)
    initial_states = tf.constant(np.random.randn(n, state_dim), dtype=tf.float32)
    initial_covs = tf.eye(state_dim, batch_shape=[n], dtype=tf.float32) * 0.1
    noises = tf.constant(np.random.randn(t, n, state_dim), dtype=tf.float32) * 0.1
    observations = tf.constant(np.random.randn(t, obs_dim), dtype=tf.float32)

    # Create tf.function with fixed input_signature for trace measurement
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[3], dtype=tf.float32),  # theta
            tf.TensorSpec(shape=[n, state_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[n, state_dim, state_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[t, n, state_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[t, obs_dim], dtype=tf.float32),
        ]
    )
    def evaluate(th, init, covs, nz, obs):
        return canonical_value_and_analytical_score(
            model=model,
            theta=th,  # Austria SIR requires runtime theta parameter
            initial_states=init,
            initial_covariances=covs,
            noises=nz,
            observations=obs,
            flow_substeps=substeps,
            annealed_stages=ANNEALED_STAGES,
            correction_steps=0,
            pairwise_steps=0,
            with_score=True,
        )

    # First call measures trace time
    t0 = time.perf_counter()
    val1, score1 = evaluate(theta, initial_states, initial_covs, noises, observations)
    t1 = time.perf_counter()
    trace_time = t1 - t0

    # Subsequent calls measure steady state (average of 5)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        val, score = evaluate(theta, initial_states, initial_covs, noises, observations)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    steady_time = np.mean(times)

    # Verify consistency
    val_diff = tf.reduce_max(tf.abs(val - val1)).numpy()
    score_diff = tf.reduce_max(tf.abs(score - score1)).numpy()

    return {
        "trace_time": trace_time,
        "steady_time": steady_time,
        "val_consistency": val_diff,
        "score_consistency": score_diff,
        "concrete_fn": evaluate,
    }


def main():
    print("=" * 80)
    print("Phase 3.5.2 Measurement: Substep while_loop Performance")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  N = {N_PARTICLES} particles")
    print(f"  T = {TIME_HORIZON} timesteps")
    print(f"  substeps = {SUBSTEPS}")
    print(f"  d = {STATE_DIM}")
    print(f"  annealed_stages = {ANNEALED_STAGES}")
    print()
    print("Target metrics (from master program):")
    print("  - Trace time: <50s (baseline: 485s)")
    print("  - Steady state: ≤80s (baseline: 39.3s)")
    print("  - Graph size: O(10³) nodes")
    print()
    print("-" * 80)
    print()

    # Small scale measurement
    print("Small scale (N=24, T=5, substeps=2):")
    result_small = measure_trace_and_steady(24, 5, 2, STATE_DIM, OBS_DIM)
    print(f"  Trace time: {result_small['trace_time']:.2f}s")
    print(f"  Steady time: {result_small['steady_time']:.3f}s")
    print(f"  Value consistency: {result_small['val_consistency']:.2e}")
    print(f"  Score consistency: {result_small['score_consistency']:.2e}")
    print()

    # Plan scale measurement
    print(f"Plan scale (N={N_PARTICLES}, T={TIME_HORIZON}, substeps={SUBSTEPS}):")
    result_plan = measure_trace_and_steady(N_PARTICLES, TIME_HORIZON, SUBSTEPS, STATE_DIM, OBS_DIM)
    print(f"  Trace time: {result_plan['trace_time']:.2f}s")
    print(f"  Steady time: {result_plan['steady_time']:.3f}s")
    print(f"  Value consistency: {result_plan['val_consistency']:.2e}")
    print(f"  Score consistency: {result_plan['score_consistency']:.2e}")
    print()

    # Success assessment
    print("-" * 80)
    print()
    print("Success Criteria Assessment:")
    print()

    trace_target = 50.0
    steady_target = 80.0
    trace_baseline = 485.0
    steady_baseline = 39.3

    trace_pass = result_plan['trace_time'] < trace_target
    steady_pass = result_plan['steady_time'] <= steady_target

    trace_speedup = trace_baseline / result_plan['trace_time']
    steady_speedup = steady_baseline / result_plan['steady_time']

    print(f"1. Trace time: {result_plan['trace_time']:.2f}s < {trace_target}s? {'✓ PASS' if trace_pass else '✗ FAIL'}")
    print(f"   Speedup: {trace_speedup:.2f}× vs baseline ({trace_baseline}s)")
    print()
    print(f"2. Steady state: {result_plan['steady_time']:.3f}s ≤ {steady_target}s? {'✓ PASS' if steady_pass else '✗ FAIL'}")
    print(f"   Speedup: {steady_speedup:.2f}× vs baseline ({steady_baseline}s)")
    print()
    print(f"3. Graph size: O(10³) nodes? (manual inspection required)")
    print(f"   Expected: T={TIME_HORIZON} copies of 1 substep body")
    print(f"   Baseline: T={TIME_HORIZON} × substeps={SUBSTEPS} = {TIME_HORIZON * SUBSTEPS} unrolled stages")
    print(f"   Reduction: {SUBSTEPS}× fewer graph copies")
    print()

    overall_pass = trace_pass and steady_pass
    print("=" * 80)
    print(f"Phase 3.5.2 Performance: {'✓ SUCCESS' if overall_pass else '✗ INCOMPLETE'}")
    print("=" * 80)

    return result_plan


if __name__ == "__main__":
    result = main()
