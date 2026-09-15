"""Does graph mode preserve the Corollary 5.2 surrogate-gradient contract?

Graph compilation is a performance change, but it touches the one mechanism the
whole Corollary 5.2 setup depends on: `tf.custom_gradient`. If `tf.function`
tracing were to bypass the registered gradient and differentiate through the
LEDH filter instead, HMC would still run, still accept, and still produce
plausible samples - while using the WRONG force. That is a silent wrong-science
failure, not a crash, so it is checked explicitly here.

Three properties, in increasing strength:

  V1  value agreement    graph value == eager value
  V2  gradient identity   graph gradient == the ANALYTICAL BIASED SCORE
                          (score_only(exact=False)), not the exact score and not
                          an autodiff result
  V3  gradient is biased  the returned gradient differs from the EXACT score,
                          confirming the surrogate force is actually in use

V2 is the load-bearing check. V3 guards the opposite failure: a gradient that
silently matches the exact score would mean the damping had no effect, making
every damping arm identical and the whole sweep vacuous.

Diagnostic only. No scientific claim about damping.
"""

import json
import os
import sys
import time
from pathlib import Path

import tensorflow as tf

_GPUS = tf.config.list_physical_devices("GPU")
for _g in _GPUS:
    tf.config.experimental.set_memory_growth(_g, True)
    if not tf.config.experimental.get_memory_growth(_g):
        raise RuntimeError(f"set_memory_growth failed on {_g.name}")

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

OBS_FULL = _lgssm_frozen_observations()
MODEL = _diagonal_lgssm_fused_model()
D = 3
DTYPE = OBS_FULL.dtype
THETA = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=DTYPE)
P = int(THETA.shape[0])


def build_target(N, T, substeps, sinkhorn, damping_ratio):
    observations = OBS_FULL[:T]
    gen = tf.random.Generator.from_seed(81100)
    initial_states = gen.normal([N, D], dtype=DTYPE) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(D, dtype=DTYPE)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = gen.normal([T, N, D], dtype=DTYPE) * 0.1
    basis = tf.concat([tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0)
    reps = (N + 2 * D - 1) // (2 * D)
    reset_design = tf.tile(basis, [reps, 1])[:N]

    shared = dict(
        substeps=substeps,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )
    base_ridge, base_lm = 1e-5, 1e-2
    return DualParameterLEDHTarget(
        model=MODEL,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params={**shared, "reset_ridge": base_ridge,
                      "correction_lm_damping": base_lm},
        biased_params={**shared, "reset_ridge": base_ridge * damping_ratio,
                       "correction_lm_damping": base_lm * damping_ratio},
    )


def rel(a, b):
    denom = max(float(tf.norm(a)), 1e-300)
    return float(tf.norm(a - b)) / denom


def main():
    N = int(os.environ.get("VERIFY_N", 24))
    T = int(os.environ.get("VERIFY_T", 5))
    substeps = int(os.environ.get("VERIFY_SUBSTEPS", 2))
    sinkhorn = int(os.environ.get("VERIFY_SINKHORN", 2))
    ratio = float(os.environ.get("VERIFY_DAMPING", 100.0))

    print("Graph-mode surrogate-gradient verification (diagnostic only)")
    print(f"  gpus={[g.name for g in _GPUS]} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")
    print(f"  N={N} T={T} substeps={substeps} sinkhorn={sinkhorn} "
          f"damping_ratio={ratio:g}\n", flush=True)

    target = build_target(N, T, substeps, sinkhorn, ratio)
    out = {"N": N, "T": T, "substeps": substeps, "sinkhorn": sinkhorn,
           "damping_ratio": ratio}
    failures = []

    # Reference: the analytical scores, computed directly.
    print("  computing analytical reference scores ...", flush=True)
    biased_score = target.score_only(THETA, exact=False)
    exact_score = target.score_only(THETA, exact=True)
    print(f"    biased score: {biased_score.numpy()}")
    print(f"    exact  score: {exact_score.numpy()}")
    score_sep = rel(exact_score, biased_score)
    print(f"    rel separation exact vs biased: {score_sep:.6e}", flush=True)
    out["score_separation"] = score_sep
    if score_sep < 1e-9:
        failures.append(
            "exact and biased scores are numerically indistinguishable: the "
            "damping ratio has no effect, so every damping arm would be "
            "identical and the sweep vacuous"
        )

    # Eager path.
    print("\n  eager value+gradient ...", flush=True)
    t0 = time.time()
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        v_eager = target(THETA)
    g_eager = tape.gradient(v_eager, THETA)
    eager_s = time.time() - t0
    print(f"    value={float(v_eager)!r}  ({eager_s:.1f}s)")
    print(f"    grad ={g_eager.numpy()}", flush=True)

    # Graph path, via the class helper.
    print("\n  graph value+gradient (traced once) ...", flush=True)
    graph_target = target.as_graph_callable(P)
    t1 = time.time()
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        v_graph = graph_target(THETA)
    g_graph = tape.gradient(v_graph, THETA)
    graph_first_s = time.time() - t1
    print(f"    value={float(v_graph)!r}  ({graph_first_s:.1f}s incl. trace)")
    print(f"    grad ={g_graph.numpy()}", flush=True)

    t2 = time.time()
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        v2 = graph_target(THETA)
    _ = tape.gradient(v2, THETA)
    graph_steady_s = time.time() - t2
    out.update({"eager_s": eager_s, "graph_first_s": graph_first_s,
                "graph_steady_s": graph_steady_s})

    # V1: value agreement.
    v_rel = abs(float(v_graph) - float(v_eager)) / max(abs(float(v_eager)), 1e-300)
    ok_v1 = v_rel <= 1e-12
    out["v1_value_rel_diff"] = v_rel
    if not ok_v1:
        failures.append(f"V1 value mismatch: rel {v_rel:.3e}")

    # V2: graph gradient IS the analytical biased score.
    g_rel_biased = rel(biased_score, g_graph)
    ok_v2 = g_rel_biased <= 1e-12
    out["v2_grad_vs_biased_rel"] = g_rel_biased
    if not ok_v2:
        failures.append(
            f"V2 graph gradient is NOT the analytical biased score "
            f"(rel {g_rel_biased:.3e}): custom_gradient may have been bypassed"
        )

    # V3: that gradient is not the exact score.
    g_rel_exact = rel(exact_score, g_graph)
    ok_v3 = g_rel_exact > 1e-9
    out["v3_grad_vs_exact_rel"] = g_rel_exact
    if not ok_v3:
        failures.append(
            f"V3 graph gradient equals the EXACT score (rel {g_rel_exact:.3e}): "
            f"the surrogate force is not in use"
        )

    # Eager must satisfy the same contract, as a control.
    e_rel_biased = rel(biased_score, g_eager)
    out["eager_grad_vs_biased_rel"] = e_rel_biased
    ok_ctrl = e_rel_biased <= 1e-12
    if not ok_ctrl:
        failures.append(
            f"control: EAGER gradient is not the analytical biased score "
            f"(rel {e_rel_biased:.3e})"
        )

    print("\n  --- results ---")
    print(f"  V1 graph value == eager value        : "
          f"{'PASS' if ok_v1 else 'FAIL'}  (rel {v_rel:.3e})")
    print(f"  V2 graph grad == analytical biased   : "
          f"{'PASS' if ok_v2 else 'FAIL'}  (rel {g_rel_biased:.3e})")
    print(f"  V3 graph grad != exact score         : "
          f"{'PASS' if ok_v3 else 'FAIL'}  (rel {g_rel_exact:.3e})")
    print(f"  control: eager grad == biased        : "
          f"{'PASS' if ok_ctrl else 'FAIL'}  (rel {e_rel_biased:.3e})")
    print(f"\n  timing: eager {eager_s:.1f}s | graph first {graph_first_s:.1f}s "
          f"| graph steady {graph_steady_s:.1f}s")
    if graph_steady_s > 0:
        print(f"  steady-state speedup: x{eager_s/graph_steady_s:.2f}")
        out["speedup"] = eager_s / graph_steady_s

    out["failures"] = failures
    p = Path(os.environ.get("VERIFY_OUT", "/tmp/ledh_graph_gradient_verify.json"))
    p.write_text(json.dumps(out, indent=2))

    if failures:
        print("\n  RESULT: FAIL")
        for f in failures:
            print(f"    - {f}")
        print(f"\n  saved: {p}")
        return 1
    print("\n  RESULT: PASS - graph mode preserves the Corollary 5.2 contract:")
    print("    same exact value for the acceptance ratio, and the analytical")
    print("    biased score as the force.")
    print(f"\n  saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
