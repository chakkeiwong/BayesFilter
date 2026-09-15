"""Which damping knob actually produces a CHEAPER force?

The magnitude knobs (`reset_ridge`, `correction_lm_damping`) were measured to
give no compute saving: biased/exact cost ratio within 7% of 1.0 at every
damping ratio, and only 3e-05 relative force separation at plan scale. They
change what the arithmetic computes, not how much of it runs.

Corollary 5.2 needs a biased force that is genuinely cheaper. The candidates are
the knobs that control ITERATION COUNTS rather than magnitudes:

  reset_sinkhorn_steps / reset_balance_steps   transport solve iterations
  correction_steps                             correction sweeps
  pairwise_steps                               pairwise interaction sweeps
  substeps                                     flow substeps per timestep

For each candidate this measures BOTH quantities a usable knob needs:

  cost ratio    biased_cost / exact_cost   -- must be < 1 to be worth anything
  force change  rel ||biased - exact||     -- must be large enough to matter,
                                              but not so large that HMC
                                              acceptance collapses

A knob is viable only if it moves cost AND force together. The magnitude knobs
move neither. Note `substeps` was measured to leave cost unchanged (x1.01), so
it is expected to fail the cost test despite being an iteration count - included
precisely to confirm the measurement discriminates.

Diagnostic only. Does not select a knob; produces the table an owner needs.
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


def base_shared(N, substeps, sinkhorn):
    basis = tf.concat([tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0)
    reps = (N + 2 * D - 1) // (2 * D)
    reset_design = tf.tile(basis, [reps, 1])[:N]
    return dict(
        substeps=substeps, reset_policy="contract_e", reset_design=reset_design,
        reset_epsilon=2.0, reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn, correction_steps=4,
        correction_strength=0.2, correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5, pairwise_steps=4, pairwise_strength=0.02,
        pairwise_rms_cap=2.0, coordinate_cap=0.0, annealed_stages=1,
        annealed_seed=0, reset_ridge=1e-5, correction_lm_damping=1e-2,
    )


def build(N, T, shared_exact, shared_biased):
    observations = OBS_FULL[:T]
    gen = tf.random.Generator.from_seed(81100)
    initial_states = gen.normal([N, D], dtype=DTYPE) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(D, dtype=DTYPE)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = gen.normal([T, N, D], dtype=DTYPE) * 0.1
    return DualParameterLEDHTarget(
        model=MODEL, initial_states=initial_states,
        initial_covariances=initial_covariances, noises=noises,
        observations=observations,
        exact_params=shared_exact, biased_params=shared_biased,
    )


def timed_score(target, exact, repeats):
    target.score_only(THETA, exact=exact)  # warm up
    ts = []
    for _ in range(repeats):
        t0 = time.time()
        s = target.score_only(THETA, exact=exact)
        _ = float(tf.norm(s))
        ts.append(time.time() - t0)
    return min(ts), s


def main():
    N = int(os.environ.get("KNOB_N", 24))
    T = int(os.environ.get("KNOB_T", 5))
    substeps = int(os.environ.get("KNOB_SUBSTEPS", 2))
    sinkhorn = int(os.environ.get("KNOB_SINKHORN", 8))
    repeats = int(os.environ.get("KNOB_REPEATS", 3))

    print("Cheap-force knob search (diagnostic only)")
    print(f"  N={N} T={T} base substeps={substeps} base sinkhorn={sinkhorn} "
          f"repeats={repeats}")
    print("  exact params are the SAME in every row; only the biased params "
          "change.\n", flush=True)

    # Candidate biased-parameter overrides. Each halves or quarters an
    # iteration count relative to the exact parameters.
    candidates = [
        ("baseline (identical params)", {}),
        ("reset_ridge x100 (magnitude)", {"reset_ridge": 1e-3}),
        ("correction_lm_damping x100 (magnitude)",
         {"correction_lm_damping": 1.0}),
        ("reset_sinkhorn_steps 8->4", {"reset_sinkhorn_steps": 4,
                                       "reset_balance_steps": 4}),
        ("reset_sinkhorn_steps 8->2", {"reset_sinkhorn_steps": 2,
                                       "reset_balance_steps": 2}),
        ("correction_steps 4->2", {"correction_steps": 2}),
        ("correction_steps 4->0", {"correction_steps": 0}),
        ("pairwise_steps 4->0", {"pairwise_steps": 0}),
    ]

    exact_shared = base_shared(N, substeps, sinkhorn)
    rows = []
    exact_cost = None
    exact_score_vec = None

    print(f"  {'knob':42s} {'cost':>8s} {'cost/exact':>11s} "
          f"{'force rel':>11s}  viable")
    print(f"  {'-'*42} {'-'*8} {'-'*11} {'-'*11}  ------")

    for label, override in candidates:
        biased_shared = {**exact_shared, **override}
        tgt = build(N, T, exact_shared, biased_shared)

        if exact_cost is None:
            exact_cost, exact_score_vec = timed_score(tgt, True, repeats)
            print(f"  {'(exact reference)':42s} {exact_cost:7.3f}s "
                  f"{1.0:11.3f} {0.0:11.3e}", flush=True)

        bcost, bscore = timed_score(tgt, False, repeats)
        cost_ratio = bcost / exact_cost
        denom = max(float(tf.norm(exact_score_vec)), 1e-300)
        force_rel = float(tf.norm(bscore - exact_score_vec)) / denom

        cheaper = cost_ratio < 0.90
        moves_force = force_rel > 1e-3
        viable = "YES" if (cheaper and moves_force) else "no"
        rows.append({"knob": label, "override": {k: v for k, v in override.items()},
                     "cost_s": bcost, "cost_ratio": cost_ratio,
                     "force_rel": force_rel, "cheaper": cheaper,
                     "moves_force": moves_force, "viable": viable == "YES"})
        print(f"  {label:42s} {bcost:7.3f}s {cost_ratio:11.3f} "
              f"{force_rel:11.3e}  {viable}", flush=True)

    print("\n  viable = cost/exact < 0.90 AND force rel > 1e-3")
    print("  (a knob must actually save compute AND actually change the force)")

    viable = [r for r in rows if r["viable"]]
    print(f"\n  viable knobs: {len(viable)}")
    for r in viable:
        saving = 100 * (1 - r["cost_ratio"])
        print(f"    - {r['knob']}: {saving:.0f}% cheaper, "
              f"force moves {r['force_rel']:.2e}")
    if not viable:
        print("    NONE. No tested knob produces a cheaper force at this scale.")
        print("    Corollary 5.2's cheap-force benefit would not be realizable")
        print("    with these controls, and the sweep has nothing to measure.")

    print("\n  NOT concluded: that a viable knob is SAFE. A cheaper force that")
    print("  moves the force a lot may collapse HMC acceptance. That is what a")
    print("  properly re-parameterized sweep would measure - this table only")
    print("  says which knobs are candidates worth sweeping.")

    p = Path(os.environ.get("KNOB_OUT", "/tmp/ledh_knob_search.json"))
    p.write_text(json.dumps({"N": N, "T": T, "substeps": substeps,
                             "sinkhorn": sinkhorn, "repeats": repeats,
                             "exact_cost_s": exact_cost, "rows": rows}, indent=2))
    print(f"\n  saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
