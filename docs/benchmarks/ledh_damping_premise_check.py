"""Is the damped force actually CHEAPER, and does theta have flat directions?

Two checks on the premise of the damping campaign itself. Both are cheap and
both can invalidate the campaign's design, so they belong before any long run.

C1  Is the biased (damped) score cheaper to compute than the exact score?

    Corollary 5.2's value proposition is an exact value for the acceptance ratio
    plus a CHEAPER biased force for mixing. The damping knobs here are
    `reset_ridge` and `correction_lm_damping` - numerical magnitudes, not
    iteration counts. Changing 1e-5 -> 1e-3 alters the arithmetic performed but
    not the AMOUNT of arithmetic. If the biased score costs the same as the
    exact score, then the dual-parameter target performs 2x the work of plain
    exact-gradient HMC for no compute saving, and the campaign measures an
    optimization that saves nothing.

C2  How many theta coordinates actually receive score?

    A coordinate with identically zero score gets no HMC restoring force and
    random-walks. Such coordinates corrupt ESS (measuring a random walk) and
    corrupt any arm-vs-arm distributional comparison that pools coordinates.

Diagnostic only.
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


def build_target(N, T, substeps, sinkhorn, ratio):
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
        substeps=substeps, reset_policy="contract_e", reset_design=reset_design,
        reset_epsilon=2.0, reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn, correction_steps=4,
        correction_strength=0.2, correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5, pairwise_steps=4, pairwise_strength=0.02,
        pairwise_rms_cap=2.0, coordinate_cap=0.0, annealed_stages=1,
        annealed_seed=0,
    )
    return DualParameterLEDHTarget(
        model=MODEL, initial_states=initial_states,
        initial_covariances=initial_covariances, noises=noises,
        observations=observations,
        exact_params={**shared, "reset_ridge": 1e-5,
                      "correction_lm_damping": 1e-2},
        biased_params={**shared, "reset_ridge": 1e-5 * ratio,
                       "correction_lm_damping": 1e-2 * ratio},
    )


def main():
    N = int(os.environ.get("PREMISE_N", 24))
    T = int(os.environ.get("PREMISE_T", 5))
    substeps = int(os.environ.get("PREMISE_SUBSTEPS", 2))
    sinkhorn = int(os.environ.get("PREMISE_SINKHORN", 2))
    repeats = int(os.environ.get("PREMISE_REPEATS", 3))
    ratios = [float(x) for x in
              os.environ.get("PREMISE_RATIOS", "1,10,100,1000").split(",")]

    print("Damping premise check (diagnostic only)")
    print(f"  N={N} T={T} substeps={substeps} sinkhorn={sinkhorn} "
          f"repeats={repeats}\n", flush=True)

    theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=DTYPE)
    P = int(theta.shape[0])
    out = {"N": N, "T": T, "substeps": substeps, "repeats": repeats,
           "ratios": ratios}

    # ---- C1: cost of exact vs biased score, per damping ratio ----
    print("C1  cost of one score evaluation: exact params vs biased params")
    print("    (same K=P directions in both; only the damping VALUES differ)\n")
    rows = []
    for ratio in ratios:
        tgt = build_target(N, T, substeps, sinkhorn, ratio)
        tgt.score_only(theta, exact=True)   # warm up
        te = []
        for _ in range(repeats):
            t0 = time.time()
            s = tgt.score_only(theta, exact=True)
            _ = float(tf.norm(s))
            te.append(time.time() - t0)
        tb = []
        for _ in range(repeats):
            t0 = time.time()
            s = tgt.score_only(theta, exact=False)
            _ = float(tf.norm(s))
            tb.append(time.time() - t0)
        be, bb = min(te), min(tb)
        rows.append({"ratio": ratio, "exact_s": be, "biased_s": bb,
                     "ratio_biased_over_exact": bb / be if be > 0 else None})
        print(f"    damping {ratio:7.0f}x   exact {be:7.3f}s   "
              f"biased {bb:7.3f}s   biased/exact = {bb/be:5.3f}", flush=True)
    out["c1"] = rows

    worst = max(abs(r["ratio_biased_over_exact"] - 1.0) for r in rows)
    print(f"\n    max deviation of biased/exact from 1.0: {worst:.3f}")
    if worst < 0.10:
        print("    => the damped force is NOT cheaper (within 10%).")
        print("       The dual-parameter target therefore costs ~2x a plain")
        print("       exact-gradient HMC step and saves no compute. Corollary")
        print("       5.2's cheap-force benefit is not instantiated by this")
        print("       damping parameterization.")
        out["c1_verdict"] = "biased force is not cheaper"
    else:
        print("    => biased force cost differs materially from exact.")
        out["c1_verdict"] = "biased force cost differs"

    # ---- C2: which theta coordinates receive score ----
    print("\nC2  score support across theta coordinates")
    tgt = build_target(N, T, substeps, sinkhorn, 100.0)
    s_exact = tgt.score_only(theta, exact=True).numpy()
    s_biased = tgt.score_only(theta, exact=False).numpy()
    print(f"    exact  score: {s_exact}")
    print(f"    biased score: {s_biased}")
    dead = [i for i in range(P)
            if abs(s_exact[i]) == 0.0 and abs(s_biased[i]) == 0.0]
    live = [i for i in range(P) if i not in dead]
    print(f"    coordinates WITH score   : {live}")
    print(f"    coordinates with NO score: {dead}")
    out["c2_live"] = live
    out["c2_dead"] = dead

    if dead:
        print(f"\n    => {len(dead)} of {P} coordinates have identically zero")
        print("       score. Under HMC they receive no restoring force and")
        print("       random-walk. Consequences:")
        print("         - ESS in those coordinates measures a random walk;")
        print("         - any arm-vs-arm distributional comparison that POOLS")
        print("           coordinates is contaminated by them;")
        print("         - sampling them at all wastes work.")
        print(f"       Restricting theta to {live} would also cut the score")
        print(f"       from K={P} to K={len(live)} filter passes per call")
        print(f"       ({(1+P)} -> {(1+len(live))} passes total, "
              f"-{100*(P-len(live))/(1+P):.0f}%).")
        print("       NOTE: this is a property of THIS fixture model, not of")
        print("       the adapter. Another model may identify all P.")

    p = Path(os.environ.get("PREMISE_OUT", "/tmp/ledh_damping_premise.json"))
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
