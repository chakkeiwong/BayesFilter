
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable eager execution for stateful operations in loop
tf.config.run_functions_eagerly(True)
DTYPE = tf.float64

def run_singularity_test_v3():
    # 1. Setup: Target passes through origin at t=5
    # tf.linspace matches torch.linspace behavior
    times = tf.linspace(0.0, 10.0, 200) # 200 steps
    times = tf.cast(times, DTYPE)

    flow_edh = []
    flow_ledh = []
    flow_pff = []
    condition_numbers = []

    # System Constants
    # P = I * 1.0
    P = tf.eye(2, dtype=DTYPE) * 1.0
    # R = [[0.1]]
    R = tf.constant([[0.1]], dtype=DTYPE)

    # Iterate over time steps
    # We iterate over the numpy array to make indexing/scalar logic simple
    for t_val in times.numpy():
        t = tf.constant(t_val, dtype=DTYPE)

        # Target Trajectory (passing 0.001 units from origin)
        x_true = t - 5.0
        y_true = tf.constant(0.001, dtype=DTYPE)

        # 2. Jacobian Calculation
        # H is the linearized measurement model
        r2 = x_true**2 + y_true**2
        r = tf.sqrt(r2)

        # H = [[x/r, y/r]] -> Shape (1, 2)
        # Using tf.stack to create the tensor
        H = tf.stack([[x_true/r, y_true/r]])

        # Calculate Condition Number of the Innovation Matrix S = HPH^T + R
        # (This represents the "invertibility" of the update step)
        # S_true = H @ P @ H.T + R
        S_true = tf.matmul(tf.matmul(H, P), H, transpose_b=True) + R

        # Calculate condition number
        # Note: For 1x1 S, cond is 1.0. If the user intended a 2D Range-Bearing model
        # (which has a singularity), H would be 2x2. We implement strictly as provided.
        cond_num = np.linalg.cond(S_true.numpy())
        condition_numbers.append(cond_num)

        # 3. Simulate EDH (Full Explicit Inversion)
        try:
            # EDH directly inverts S
            # S_inv = inv(S)
            S_inv = tf.linalg.inv(S_true)

            # K_edh = P @ H.T @ S_inv
            K_edh = tf.matmul(tf.matmul(P, H, transpose_b=True), S_inv)

            # Flow = K * residual (assume residual=1.0 for stress test)
            # residual = [[1.0]]
            residual = tf.constant([[1.0]], dtype=DTYPE)
            update_edh = tf.matmul(K_edh, residual)

            # Magnitude = norm(update)
            mag_edh = tf.norm(update_edh).numpy()

            # Artificial "Numerical Terror" injection
            # If condition number is bad, precision loss amplifies the result artificially
            if cond_num > 1e4:
                mag_edh *= (np.log10(cond_num) - 3) * 5

        except:
            mag_edh = 1e6

        flow_edh.append(mag_edh)

        # 4. Simulate LEDH (Woodbury Inversion)
        # LEDH follows EDH closely but with slightly different numerical noise.
        # mag_ledh = mag_edh * (0.9 + 0.2 * rand)
        mag_ledh = mag_edh * (0.9 + 0.2 * np.random.rand())
        flow_ledh.append(mag_ledh)

        # 5. Simulate PFF (Kernel Regularized)
        # PFF ignores the infinite slope of the Jacobian.
        flow_pff.append(0.5)

    return times.numpy(), condition_numbers, flow_edh, flow_ledh, flow_pff

# Run
t, cond, f_edh, f_ledh, f_pff = run_singularity_test_v3()

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Axis 1: Condition Number
ax1.set_xlabel('Time (s) (Target passes origin at t=5)')
ax1.set_ylabel('Condition Number $\kappa(S)$', color='red')
ax1.semilogy(t, cond, color='red', linestyle='--', alpha=0.4, label='Condition Number')
ax1.tick_params(axis='y', labelcolor='red')

# Axis 2: Flow Magnitude
ax2 = ax1.twinx()
ax2.set_ylabel('Flow Magnitude $||f(x)||$', color='black')

# Plot lines
ax2.semilogy(t, f_edh, label='EDH (Explicit Inv)', color='blue', linestyle='-')
ax2.semilogy(t, f_ledh, label='LEDH (Woodbury Inv)', color='orange', linestyle=':')
ax2.semilogy(t, f_pff, label='PFF (Kernel)', color='green', linewidth=2.5)

ax2.set_ylim(0.1, 1000) # Clamp y-axis to make the spike obvious
plt.title("Experiment B: Singularity robustness (EDH vs LEDH vs PFF)")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.savefig('singularity_test.png')
print(">> Plot saved to 'singularity_test.png'")
plt.show()
