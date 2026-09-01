"""Why is the batched-direction target 2.6x slower per warm eval than the
Python-swept target, despite a 3x smaller graph?

Hypothesis (H1): the swept variant issues D score calls that share a bitwise
identical primal recursion (UKF predict, flow, weights) and differ only in the
tangent, so graph optimization (Grappler CSE) can dedupe D-1 of the D primals.
Tiling theta to B=D makes the rows distinct data, so the batched variant cannot
dedupe and pays D full primals.

Discriminating prediction:
  H1 true  -> swept warm time grows slowly in D (tangent-only increment),
              batched warm time grows ~linearly in D.
  H1 false -> both grow ~linearly in D with similar slope.

Run at horizon=10 to keep trace cost affordable; the CSE mechanism does not
depend on horizon. These are COST probes: for D<5 the assembled vector is a
partial gradient, not a valid 5-parameter gradient. Cost is the only quantity
being measured here.

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
HORIZON = 10
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
        target.noises[:HORIZON], target.observations[:HORIZON],
        substeps=SUBSTEPS,
    )


def graph_nodes(concrete) -> int:
    graph_def = concrete.graph.as_graph_def()
    return len(graph_def.node) + sum(len(f.node_def) for f in graph_def.library.function)


def timed(fn, theta0):
    t0 = time.perf_counter()
    fn(theta0)
    trace_first = time.perf_counter() - t0
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(theta0)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return trace_first, samples[REPEATS // 2]


theta0 = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)

print("=" * 80)
print("Direction-count cost scaling: swept vs batched")
print(f"N={N_PARTICLES}, dim=3, horizon={HORIZON} (truncated), substeps={SUBSTEPS}, CPU-only")
print("=" * 80)

# ---- reference: one score call, one direction, no value call
@tf.function(
    autograph=False,
    input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)],
)
def score_only(theta):
    theta_b = tf.reshape(theta, [1, -1])
    _, scores, _ = one_call(damped_model, theta_b, tf.one_hot([0], PARAM_DIM, dtype=DTYPE))
    return scores


cf = score_only.get_concrete_function()
trace_first, warm = timed(score_only, theta0)
base_nodes, base_warm = graph_nodes(cf), warm
print(f"\nreference: 1 score call (1 primal + 1 tangent)")
print(f"  nodes={base_nodes:>8,}  trace+first={trace_first:>7.2f}s  warm={warm:.4f}s")

print(f"\n{'variant':<10} {'D':>2} {'nodes':>9} {'trace+1st':>10} {'warm s':>8} "
      f"{'warm/base':>10} {'nodes/base':>11}")
print("-" * 80)

for variant in ("swept", "batched"):
    for directions in (1, 2, 3, 5):

        if variant == "swept":
            @tf.function(autograph=False, input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)])
            def probe(theta, _d=directions):
                theta_b = tf.reshape(theta, [1, -1])
                cols = []
                for p in range(_d):
                    _, scores, _ = one_call(
                        damped_model, theta_b, tf.one_hot([p], PARAM_DIM, dtype=DTYPE)
                    )
                    cols.append(scores)
                return tf.stack(cols, axis=1)
        else:
            @tf.function(autograph=False, input_signature=[tf.TensorSpec([PARAM_DIM], DTYPE)])
            def probe(theta, _d=directions):
                theta_b = tf.reshape(theta, [1, -1])
                theta_tiled = tf.tile(theta_b, [_d, 1])
                dirs = tf.eye(_d, PARAM_DIM, dtype=DTYPE)
                _, scores, _ = one_call(damped_model, theta_tiled, dirs)
                return tf.reshape(scores, [1, _d])

        cf = probe.get_concrete_function()
        nodes = graph_nodes(cf)
        trace_first, warm = timed(probe, theta0)
        print(f"{variant:<10} {directions:>2} {nodes:>9,} {trace_first:>9.2f}s "
              f"{warm:>8.4f} {warm/base_warm:>10.2f} {nodes/base_nodes:>11.2f}")

print("\n" + "-" * 80)
print("Reading: if swept warm/base stays near 1 while batched warm/base tracks D,")
print("then graph optimization is deduping the shared primal in the swept variant")
print("and tiling defeats it (H1 supported). If both track D, H1 is refuted.")
print("Device policy: CPU-only, CUDA_VISIBLE_DEVICES=-1 set before TF import.")
