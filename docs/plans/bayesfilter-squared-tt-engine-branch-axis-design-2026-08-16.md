# Design note: Branch-Axis Target Assembly for P1B (no V3 amendment needed)

Date: 2026-08-16
Status: `DESIGN_FOR_IMPLEMENTATION`
Repairs: P1B smoke rejection in
`bayesfilter-squared-tt-engine-p0-p1a-p1b-smoke-result-2026-08-16.md`.

## Problem

Naive assembly fits sqrt(f) where f contains h_prev^2 -> |h_prev| kinks.
Branch decomposition (f = sum_g (u_g)^2 G + tau G, u_g = H_L L[:,g],
L L' = E) gives smooth branch targets u_g sqrt(G), but fitting each branch
as its own TT makes the retained boundary rank grow geometrically
(R_{t+1} = (R_t + 1) r), forcing a truncation that would touch veto V3.

## Resolution: branch index as a TT axis

Fit ONE functional TT over the extended block `(g, z_curr, z_prev)` where
`g` is a DISCRETE axis of cardinality B = r_c + 1 (branches + tau branch)
with indicator basis and counting-measure mass matrix (identity):

    F(g, z_c, z_p) ~= u_g(z_p) * sqrt(G(z_c, z_p)),   u_{B} = sqrt(tau)

Then `sum_g F(g,.)^2 = f` exactly at target level, and both the
normalizer and the suffix marginalization are the SAME Gram contractions
as before — the branch axis is contracted with identity mass exactly like
any other axis:

    Z_h  = <F, F>_{counting x mu}          (Gram chain, exact)
    E_new = int F-suffix Gram over (z_p)   -> retained boundary rank <= r

Boundary rank at the (z_c | z_p) split is capped by the FIT rank r for
every step: **no growth, no truncation, no V3 amendment**. The branch
count next step is again B = r + 1: constant across time.

## Smoothness guard (replaces compression policy)

u_g needs a factor L with L L' = E. Cholesky is smooth where E is PD;
non-smooth only at singular E. Guard: declared eigenvalue-ratio floor on E
(lambda_min/lambda_max below threshold -> hard status veto, V13-style).
No truncation is ever performed; degenerate E is a veto, not a repair.
(tau > 0 keeps the full retained density bounded away from zero, which in
practice keeps E away from exact singularity; the veto covers the rest.)

## Implementation deltas (engine v0 -> v0.1)

1. `DiscreteIndicatorBasis1D`: basis_dim B, evaluate = one-hot at integer-
   coded points, mass_matrix = identity for both measures, domain [0, B-1].
   Conforms to Basis1DProtocol for FixedTTFitter reuse.
2. Step assembly (t >= 1): rows = frozen z-rows x all B branch codes
   (N_z * B rows); target per row `u_g(z_p) sqrt(G)` with signed u_g (no
   abs); tau branch constant sqrt(tau) * sqrt(G).
3. Max-shift: applied to log f = log(sum_g u_g^2 G + tau G) as before for
   the increment; branch targets scaled by exp(-s/2) consistently.
4. Retention: split after (g, z_c...) prefix? NO — split so that retained
   axes are z_c only: axis order (z_c..., g, z_p...): put the branch axis
   in the SUFFIX so it is integrated (summed) out with z_p. Retained
   object stays the standard RetainedQuadraticForm over z_c.
5. Sign note: u_g are signed smooth functions; no |.| appears anywhere.
6. Score path (P2, unchanged in structure): the branch axis adds one more
   frozen axis; tangents flow through L(theta) = chol(E(theta)) — the
   Cholesky JVP is closed-form (existing `_cholesky_jvp` in
   cubature_genut_batch_tf.py is the donor pattern); recorded as a UB-1
   addendum obligation before P2.

## Gate

The strict-xfail smoke tests flip to expected-pass at their declared
tolerances (5e-3 at n=1, 2e-2 at n=2) with the SAME declared tolerances —
no gate relaxation. Then the P1B ladder proceeds per plan.
