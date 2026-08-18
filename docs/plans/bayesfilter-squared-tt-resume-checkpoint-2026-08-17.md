# Resume checkpoint 2026-08-17 (classifier outage during P2 conditioning repair)

## State at pause

The conditioning repair (Addendum 4's identified fix) is IMPLEMENTED but
NOT YET VERIFIED — the Bash permission classifier became unavailable
mid-run; no test has executed since the edits below.

## Edits landed (working tree, uncommitted)

1. `bayesfilter/highdim/squared_tt_adjoint_tf.py`:
   - new `scaled_normal_solve(design, weights, ridge, rhs)`: solves
     (A'WA + rho I)x = rhs via the VALUE path's scaled augmented QR
     (column scales from `_weighted_column_scales`, augmented [sqrt(W) A S^-1;
     sqrt(rho) S^-1], R'R backsubstitution, unscale by S). Derivative
     solves now inherit value-path conditioning instead of squaring it.
   - `solve_node_adjoint`: lambda solve rerouted through
     `scaled_normal_solve` (was raw `tf.linalg.solve` on N).
   - new `forward_jvp_replay_scaled(updates, cores0, dots0, dot_target)`:
     ordered forward JVP over the TRACED value updates (value cores =
     traced solutions bit-identical to the program; tangent solves through
     `scaled_normal_solve`; dot_A via `differentiate_design_matrix` with
     current dots).
2. `tests/highdim/test_p2_adjoint_vs_forward_jvp.py`:
   - instrument switched from the donor `fixed_als_value_jvp` (unscaled)
     to `_fixed_als_fit_traced` + `forward_jvp_replay_scaled`;
   - strict-xfail REMOVED — the test now asserts the tightened 1e-9 gate.

## First command on resume

CUDA_VISIBLE_DEVICES=-1 <tf-gpu python> -m pytest \
  tests/highdim/test_p2_adjoint_nodes.py \
  tests/highdim/test_p2_adjoint_vs_forward_jvp.py \
  tests/highdim/test_p2_adjoint_engine_fd.py -q

Expected outcomes and dispositions:
- ALL PASS -> conditioning repair confirmed; extend the FD-gate file's n=2
  test comment (FD remains resolution-limited; I-P2-4 is the decisive
  gate); update Addendum 4 with "repair verified"; proceed to queue items:
  v0.3 smoke reruns -> P1B ladder rerun (v0.3) -> T=120 adjoint stress.
- I-P2-4 still >1e-9 but improved -> the residual is in the adjoint side's
  remaining unscaled pieces (check: dot_N application inside
  `forward_jvp_replay_scaled` uses raw matvecs — correct; then inspect
  solve_node_adjoint's residual term precision); bisect per-update by
  comparing bar/dot inner products update-by-update (the pairing identity
  holds per node, so the first diverging update localizes the defect).
- U-ADJ-SOLVE-1 fails -> `scaled_normal_solve` has a bug; verify against
  `_solve_scaled_augmented_ridge` solution on the same system first
  (they must agree to ~1e-15 on well-conditioned fixtures).

## Queue after verification (unchanged from Addendum 4)

(2) v0.3 smoke-gate reruns (n=1 already green; n=2 ~40-min cell);
(3) P1B ladder rerun under v0.3 — attempt01 is program-defect evidence,
    not rank evidence;
(4) T=120 adjoint-state stress (P2A full-horizon obligation);
then P3 (XLA), P4 (adapters + reproduction gates), P5 (tuning v1.1),
P2S after UB-3 review, P6 leaderboard + NAWM-representative gate.
