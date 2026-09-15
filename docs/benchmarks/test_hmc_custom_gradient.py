"""Test tf.custom_gradient with HMC (no LEDH)."""

import tensorflow as tf
import tensorflow_probability as tfp

print("Testing custom_gradient with TFP HMC...")

# Simple test target: custom_gradient with fake surrogate
@tf.custom_gradient
def surrogate_target(theta):
    # Exact value
    value = tf.reduce_sum(theta**2)

    def grad_fn(dy):
        # Surrogate gradient (damped by factor of 100)
        exact_grad = 2.0 * theta
        surrogate_grad = exact_grad / 100.0
        return dy * surrogate_grad

    return value, grad_fn

print("\nTest 1: Call returns scalar")
theta = tf.constant([1.0, 2.0, 3.0])
val = surrogate_target(theta)
print(f"  Value: {val.numpy()}")

print("\nTest 2: GradientTape gets surrogate gradient")
with tf.GradientTape() as tape:
    tape.watch(theta)
    val = surrogate_target(theta)
grad = tape.gradient(val, theta)
print(f"  Gradient: {grad.numpy()}")
print(f"  Expected: [0.02, 0.04, 0.06] (exact/100)")

print("\nTest 3: Run short HMC chain")
kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=lambda x: -surrogate_target(x),  # Negative for log-prob
    step_size=0.1,
    num_leapfrog_steps=3,
)

init_theta = tf.constant([1.0, 2.0, 3.0])
samples, trace = tfp.mcmc.sample_chain(
    num_results=10,
    num_burnin_steps=5,
    current_state=init_theta,
    kernel=kernel,
    trace_fn=lambda _, pkr: pkr.is_accepted,
    seed=42,
)

print(f"  Samples shape: {samples.shape}")
print(f"  Acceptance: {tf.reduce_mean(tf.cast(trace, tf.float32)).numpy():.2f}")

print("\n✓ custom_gradient works with TFP HMC!")
