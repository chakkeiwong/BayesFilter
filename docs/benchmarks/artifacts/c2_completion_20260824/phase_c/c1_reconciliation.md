# C1: Cohen-Migliorati (2017) read + half/beta-mixture reconciliation (2026-08-25)

Sections read: Theorem 1 (unweighted condition (10)), Theorem 2
(weighted stability under K_{m,w} <= kappa n/ln n, tail bound (15)),
Corollary 1 (optimal w: K = m, n >~ m ln n), the unbounded-domain
remarks (Gaussian measure on R explicitly covered; uniform-bound items
(ii)/(iii) fail there, the conditioned-estimator item (iv) survives),
and the dimension-independence remark.

Reconciliation with the engine's row law:

1. PER-AXIS: our beta-mixture has K_axis = sup w*k = ell/beta
   (ell=13: K = 26 at beta=0.5, 130 at beta=0.10) vs the paper's
   optimal K = ell = 13. Theorem 2's condition at n_rows = 8192:
   K <= kappa*8192/ln(8192) ~ 300+ for modest r — satisfied with
   margin at both betas. The per-axis design is THEORY-COVERED.
2. TENSOR/ALS TRANSFER: the ALS subproblem dictionary is
   interface-augmented (width r*ell*r), spanning cross-axis products;
   its K_{m,w} under the per-axis product mixture is NOT controlled by
   the per-axis analysis (corner regions). The transfer is
   MEASURED-ADEQUATE within the validated envelope (oracle-exact at
   deg 6 / 9 axes; Gram cond ~1.3 per axis) and measurably breaks at
   ell=13 / 9 axes (the A3 +9.5-nat unseen-mass boundary). This is the
   honest classification: per-axis guaranteed, tensor transfer
   evidence-bounded, envelope guarded by the D1 re-evidence rule.
3. Their unbounded-domain caveat (nonuniformly bounded targets ->
   conditioned estimator needed) maps onto the engine's fail-closed
   guards (non-finite retention / increment checks), which play the
   truncation-operator role.
4. Delta from the cited law: ours is the defensive beta-mixture
   variant (bounded weights), paying factor 1/beta in K for bounded
   product weights — the trade the A3 calibration measured directly.

Status: C1 CLOSED. No change to the engine required; the plan's audit
row for the row law keeps its "measured repair, theory-covered
per-axis" status.
