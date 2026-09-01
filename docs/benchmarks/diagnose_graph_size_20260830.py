"""Diagnostic: where does the surrogate-force HMC graph/memory actually go?

Questions (graph-size/op-count diagnostic, not a research-decision run):
  1. Is the LEDH score autodiff or analytical forward-mode recursion?
  2. Why does one HMC gradient issue 5 LEDH calls?
  3. If the recursion is O(1) in horizon, where does the memory go?

Method: count nodes in the concrete TF graph (top-level + nested FuncGraph
library) and record host RSS for
  (A) one canonical_batch_fused_value_score call, B=1
  (B) value + 5 swept single-direction score calls (the step1-3 target)
  (C) value + one batched-direction score call (theta tiled to B=5, tf.eye(5))
then sweep horizon and substeps to test whether node count scales with the
Python unroll length.

CPU-only by deliberate choice (CUDA_VISIBLE_DEVICES=-1 set before TF import)
so the numbers measure host graph construction, not device allocation.
The score carries no autodiff tape; see the module docstring of
ledh_canonical_batch_fused_tf ("NO autodiff (C-9): score is the analytical
recursion, fused").
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
import resource
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


def rss_now_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def count_nodes(concrete) -> tuple[int, int, float]:
    """Return (total nodes incl. nested functions, top-level nodes, MB)."""
    graph_def = concrete.graph.as_graph_def()
    top = len(graph_def.node)
    nested = sum(len(fn.node_def) for fn in graph_def.library.function)
    return top + nested, top, graph_def.ByteSize() / 1e6


print("=" * 78)
print("Graph-size diagnostic for surrogate-force LEDH HMC target")
print("=" * 78)
print(f"RSS at start: {rss_now_mb():.0f} MB")

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
    process_covariance=exact_model.process_covariance
    + 1e-3 * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance
    + 1e-3 * tf.eye(3, dtype=DTYPE),
)

HORIZON = int(target.observations.shape[0])
SUBSTEPS = int(target.substeps)
print(f"\nTarget scope: N={N_PARTICLES}, dim=3, horizon={HORIZON}, substeps={SUBSTEPS}")
print(f"Static unrolled bodies per LEDH call: {HORIZON} UKF + {HORIZON * SUBSTEPS} flow substeps")
print(f"RSS after target build: {rss_now_mb():.0f} MB\n")


def one_call(model, theta_b, directions, horizon=None, substeps=None):
    horizon = HORIZON if horizon is None else horizon
    substeps = SUBSTEPS if substeps is None else substeps
    return canonical_batch_fused_value_score(
        model,
        theta_b,
        directions,
        target.initial_states,
        target.initial_covariances,
        target.noises[:horizon],
        target.observations[:horizon],
        substeps=substeps,
    )


# --------------------------------------------------------------- case A
@tf.function(
    autograph=False,
    input_signature=[
        tf.TensorSpec([1, PARAM_DIM], DTYPE),
        tf.TensorSpec([1, PARAM_DIM], DTYPE),
    ],
)
def single_call(theta_b, directions):
    value, score, _ = one_call(exact_model, theta_b, directions)
    return value, score


def report(tag, concrete, elapsed, note=""):
    total, top, mbytes = count_nodes(concrete)
    print(
        f"[{tag}] nodes={total:>9,} (top={top:>9,})  graphdef={mbytes:>7.2f} MB"
        f"  trace={elapsed:>6.1f}s  RSS={rss_now_mb():>6.0f} MB  peak={rss_peak_mb():>6.0f} MB  {note}"
    )
    return total


t0 = time.time()
cf_a = single_call.get_concrete_function()
nodes_a = report("A", cf_a, time.time() - t0, "one LEDH call (value+score), B=1")


# --------------------------------------------------------------- case B
@tf.function(autograph=False, input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)])
def target_sweep(theta):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        theta_batch = tf.reshape(theta, [1, -1])

        @tf.custom_gradient
        def value_with_grad(theta_b):
            vals, _, _ = one_call(exact_model, theta_b, tf.zeros_like(theta_b))

            def grad_fn(dy):
                gradients = []
                for p in range(PARAM_DIM):
                    direction = tf.one_hot([p], PARAM_DIM, dtype=DTYPE)
                    _, scores, _ = one_call(damped_model, theta_b, direction)
                    gradients.append(scores)
                grad_batch = tf.stack(gradients, axis=1)
                return dy[:, None] * grad_batch

            return vals, grad_fn

        value = -value_with_grad(theta_batch)[0]
    return value, tape.gradient(value, theta)


t0 = time.time()
cf_b = target_sweep.get_concrete_function()
nodes_b = report("B", cf_b, time.time() - t0, "value + 5 swept directions")


# --------------------------------------------------------------- case C
@tf.function(autograph=False, input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)])
def target_batched(theta):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        theta_batch = tf.reshape(theta, [1, -1])

        @tf.custom_gradient
        def value_with_grad(theta_b):
            vals, _, _ = one_call(exact_model, theta_b, tf.zeros_like(theta_b))

            def grad_fn(dy):
                theta_tiled = tf.tile(theta_b, [PARAM_DIM, 1])
                directions = tf.eye(PARAM_DIM, dtype=DTYPE)
                _, scores, _ = one_call(damped_model, theta_tiled, directions)
                grad_batch = tf.reshape(scores, [1, PARAM_DIM])
                return dy[:, None] * grad_batch

            return vals, grad_fn

        value = -value_with_grad(theta_batch)[0]
    return value, tape.gradient(value, theta)


t0 = time.time()
cf_c = target_batched.get_concrete_function()
nodes_c = report("C", cf_c, time.time() - t0, "value + 1 batched-direction call")

print("\n" + "-" * 78)
print(f"B/A node ratio = {nodes_b / nodes_a:.2f}   (6 LEDH calls expected: 1 value + 5 score)")
print(f"C/A node ratio = {nodes_c / nodes_a:.2f}   (2 LEDH calls expected: 1 value + 1 score)")
print(f"B/C node ratio = {nodes_b / nodes_c:.2f}   graph shrink from batching directions")

# ------------------------------------------- unroll scaling: horizon, substeps
print("\n" + "-" * 78)
print("Unroll scaling of a single LEDH call (does node count track the Python loops?)")
for horizon, substeps in ((5, SUBSTEPS), (10, SUBSTEPS), (25, SUBSTEPS), (10, 3), (10, 6)):

    @tf.function(
        autograph=False,
        input_signature=[
            tf.TensorSpec([1, PARAM_DIM], DTYPE),
            tf.TensorSpec([1, PARAM_DIM], DTYPE),
        ],
    )
    def probe(theta_b, directions, _h=horizon, _s=substeps):
        value, score, _ = one_call(exact_model, theta_b, directions, _h, _s)
        return value, score

    t0 = time.time()
    concrete = probe.get_concrete_function()
    total, _, mbytes = count_nodes(concrete)
    bodies = horizon * substeps
    print(
        f"  horizon={horizon:>3}  substeps={substeps:>3}  bodies={bodies:>5}"
        f"  nodes={total:>9,}  nodes/body={total/bodies:>7.1f}"
        f"  graphdef={mbytes:>6.2f} MB  trace={time.time()-t0:>5.1f}s"
    )

# ------------------------------------------ parity + wall time, swept vs batched
theta0 = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
v_b, g_b = target_sweep(theta0)
v_c, g_c = target_batched(theta0)
print("\n" + "-" * 78)
print("Parity check, swept vs batched directions:")
print(f"  value    swept  = {v_b.numpy():+.12e}")
print(f"  value    batched= {v_c.numpy():+.12e}")
print(f"  gradient swept  = {g_b.numpy()}")
print(f"  gradient batched= {g_c.numpy()}")
print(f"  max abs gradient difference = {tf.reduce_max(tf.abs(g_b - g_c)).numpy():.3e}")

print("\nWarm wall time per value+gradient evaluation:")
for label, fn in (("swept (6 calls)", target_sweep), ("batched (2 calls)", target_batched)):
    fn(theta0)
    t0 = time.time()
    for _ in range(3):
        fn(theta0)
    print(f"  {label:<20} {(time.time() - t0) / 3:.3f} s/eval")

print(f"\nFinal RSS: {rss_now_mb():.0f} MB   peak RSS: {rss_peak_mb():.0f} MB")
print("Device policy: CPU-only, CUDA_VISIBLE_DEVICES=-1 set before TF import.")
