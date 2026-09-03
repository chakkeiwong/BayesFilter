"""Valid warm-time measurement for swept vs batched direction targets.

Replaces the time.time() readings in diagnose_graph_size_20260830.py, which
produced a negative interval (time.time() is not monotonic; the WSL2 clock
stepped backwards mid-run). Uses time.perf_counter().

Also separates trace time from steady-state evaluation time, and reports the
per-call FLOP-equivalence caveat: batching directions tiles theta to B=5, so
the pointwise arithmetic is NOT reduced -- only graph size, trace time, and
kernel-launch count are.

CPU-only by deliberate choice (CUDA_VISIBLE_DEVICES=-1 before TF import).
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
import time
from pathlib import Path

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)

DTYPE = tf.float64
N_PARTICLES = 252
PARAM_DIM = 5
REPEATS = 5

target = make_canonical_neutra_target(
    "lgssm", particle_count=N_PARTICLES, noise_seed=140000, substeps=12
)
exact_model = target.fused_model
damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=exact_model.process_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
)
SUBSTEPS = int(target.substeps)


def one_call(model, theta_b, directions):
    return canonical_batch_fused_value_score(
        model, theta_b, directions,
        target.initial_states, target.initial_covariances,
        target.noises, target.observations, substeps=SUBSTEPS,
    )


def build(batched: bool):
    @tf.function(autograph=False, input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)])
    def fn(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            theta_batch = tf.reshape(theta, [1, -1])

            @tf.custom_gradient
            def value_with_grad(theta_b):
                vals, _, _ = one_call(exact_model, theta_b, tf.zeros_like(theta_b))

                def grad_fn(dy):
                    if batched:
                        _, scores, _ = one_call(
                            damped_model,
                            tf.tile(theta_b, [PARAM_DIM, 1]),
                            tf.eye(PARAM_DIM, dtype=DTYPE),
                        )
                        grad_batch = tf.reshape(scores, [1, PARAM_DIM])
                    else:
                        cols = []
                        for p in range(PARAM_DIM):
                            _, scores, _ = one_call(
                                damped_model, theta_b,
                                tf.one_hot([p], PARAM_DIM, dtype=DTYPE),
                            )
                            cols.append(scores)
                        grad_batch = tf.stack(cols, axis=1)
                    return dy[:, None] * grad_batch

                return vals, grad_fn

            value = -value_with_grad(theta_batch)[0]
        return value, tape.gradient(value, theta)

    return fn


theta0 = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
print("=" * 74)
print("Warm value+gradient evaluation time (monotonic clock)")
print(f"N={N_PARTICLES}, dim=3, horizon={int(target.observations.shape[0])}, substeps={SUBSTEPS}")
print("=" * 74)

for label, batched in (("swept (6 LEDH calls)", False), ("batched (2 LEDH calls)", True)):
    fn = build(batched)
    t0 = time.perf_counter()
    value, grad = fn(theta0)
    trace_and_first = time.perf_counter() - t0

    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(theta0)
        samples.append(time.perf_counter() - t0)
    samples.sort()

    print(f"\n{label}")
    print(f"  trace + first eval : {trace_and_first:8.3f} s")
    print(f"  warm min / med / max: {samples[0]:.3f} / {samples[REPEATS // 2]:.3f} / {samples[-1]:.3f} s")
    print(f"  value    = {value.numpy():+.12e}")
    print(f"  gradient = {grad.numpy()}")

print("\nCaveat: batching directions tiles theta to B=5, so m=5*N points.")
print("Pointwise arithmetic is NOT reduced; graph size, trace time and kernel")
print("launch count are. Primal recursion is still recomputed once per direction")
print("in BOTH variants -- removing that redundancy needs a multi-direction")
print("tangent recursion inside the kernel, which is a kernel change.")
print("Device policy: CPU-only, CUDA_VISIBLE_DEVICES=-1 set before TF import.")
