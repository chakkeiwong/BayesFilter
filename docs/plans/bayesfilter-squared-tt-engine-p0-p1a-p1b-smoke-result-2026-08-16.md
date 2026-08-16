# Result: P0/P1A/P1B-smoke Execution, Generic Squared-TT Engine (2026-08-16)

Plan: `bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
(rev 3; P1A content gate unblocked per Codex 2026-08-16, metadata repair done).

## Run manifest

- Commit `18cfe609` (dirty tree, program work uncommitted); TF 2.19.1,
  float64, CPU-only (GPUs hidden, intentional); deterministic seeds.
- New library code: `bayesfilter/highdim/retained_quadratic_form_tf.py`,
  `bayesfilter/highdim/squared_tt_engine_v0_tf.py` (value-only engine v0).
- New tests: `tests/highdim/test_p1a_retained_quadratic_form.py` (5 tests),
  `tests/highdim/test_p1b_engine_v0_lgssm_smoke.py` (declared-tolerance
  smoke; see verdict).

## Plan consistency review (pre-execution)

One inconsistency found and fixed: the plan's retained-type sketch still
said "tau=0 default", contradicting owner decision D1; replaced with the
per-scope-tuned field and the dual-evaluator/`Zc_ref` payload matching
UB-1. UB-1 metadata normalized per Codex (revision/status/Z_0,ref alias).
No other contradictions found; A1-A18, V1-V13, phase ordering coherent.

## P1A: PASSED (claim-bearing gate)

All five tests green:
- U-MARG-TYPE-1: retained evaluator == brute-force suffix integration of
  h^2 (<=1e-10 rel), suffix Gram genuinely rank 3 (>1), quadratic-form
  typed; Zc at tau=0 == full-block Gram integral (<=1e-10 rel).
- U-MARG-DERIV-1: (dot_E, dot_Zh, dot log p_ret_ref) vs centered FD
  (<=1e-5..1e-6).
- U-MEASURE-1: reference/physical conversion identity (<=1e-12), mass 1
  under BOTH measures (<=1e-10), defensive component included.
- U-TAU-1: Zc(tau) - Zc(0) == tau exactly; complete-normalizer evaluator
  identity; mass 1 at tau>0; dot_Zc == dot_Zh vs FD.
- Symmetry assert-not-symmetrize enforced.
- U-ALS-REPLAY-1 requirement: covered by the donor's existing
  `test_zhao_cui_moment_teacher_als.py` (6/6 green, includes FD-checked
  ordered JVP replay with nonzero initial dot cores driving moving
  environments).

## P1B smoke: DECLARED TOLERANCE NOT MET — with a diagnosed mechanism

Setup: LGSSM n=1, T=8, exact Kalman truth; declared smoke tolerance 5e-3
(written before execution). Engine v0 runs the UB-1 V1-V5 recursion with
scattered frozen rows (no dense grid anywhere; V2 held).

| Config | tau | gap to Kalman | max fit rms |
|---|---|---|---|
| b=9, r=3, hw=8 | 0 | 4.50 | (unresolved basis) |
| b=15, r=3, hw=4 | 0 | 2.7e-1 | 2.8e-2 |
| b=19..23, r=6..8, hw=4 | 0 | 2.0e-1..2.2e-1 | plateau 2.2e-2 |
| b=19, r=6 | 1e-4 | 2.0e-1 | 2.7e-2 |
| **b=19, r=6** | **1e-3** | **1.0e-2** | 3.9e-2 |
| b=19, r=6 | 1e-2 | 1.62 | 8.2e-2 |
| b=21..25, r=6..8 | 3e-4..5e-4 | 7.5e-2..1.2e-1 | - |

Diagnosis (mechanism, verified by the step-resolved rms):
1. Step 0 fits to rms ~1e-5 — the machinery is exact where the target is
   smooth. Every transition step plateaus at rms ~2.2e-2 REGARDLESS of
   degree/rank — the signature of a non-polynomial target feature, not
   under-resolution.
2. The feature is identified: at tau=0 the step target contains
   sqrt(h_prev(z)^2) = |h_prev(z)|, which has absolute-value KINKS along
   the zero set of the previous fitted sqrt-density. A global polynomial
   L2 fit cannot converge through those kinks.
3. tau > 0 smooths the kinks at scale sqrt(tau): gap improves 20x at
   tau=1e-3. Too-large tau over-mixes (defensive mass inflates the
   likelihood: gap 1.62 at 1e-2). The tau landscape is sharp and
   non-monotone at fixed resolution.

Interpretation:
- **Owner decision D1 is empirically vindicated by the engine's own
  mathematics**: tau is not merely starvation insurance; it is what makes
  the sqrt-target fittable at all in this route. tau=0 is not viable for
  the L2-fit engine; per-scope tuning with a mixing-bias control (Codex
  F4's exact concern) is mandatory.
- Candidate rejection, not direction rejection: the naive
  single-sqrt-refit target assembly is the failing candidate. The
  Prop-2-consistent repair is the **sum-of-squares branch decomposition**:
  with E = L L', the step target factorizes exactly as
  f = sum_g (u_g(z_prev))^2 G + tau G, u_g = H_L L[:,g], and each branch
  u_g * sqrt(G) is SMOOTH (no kinks; G > 0 for Gaussian kernels). This is
  precisely the author structure ("still need rk functions",
  `marginalise.m:35-37,63-65`): the recursion should carry multiple
  smooth branch functions, never |h|.
- The branch route has a known cost: boundary rank grows additively per
  step (sum over branches), so it REQUIRES a declared branch-compression
  policy. Frozen eigen-truncation of the PSD Gram E touches veto V3
  (runtime SVD-truncation non-smoothness) and therefore needs a plan
  amendment with a smoothness analysis (eigen-truncation of a PD matrix
  with a spectral-gap guard, or a fixed branch budget with Gram-weighted
  grouping) BEFORE implementation.

## Decision table

| Item | Status |
|---|---|
| Decision | P1A promoted (all gates passed). P1B smoke: candidate target-assembly REJECTED at declared tolerance; planned repair identified (branch decomposition + compression policy) |
| Primary criterion | P1A: passed. P1B smoke: failed (best gap 1.0e-2 vs declared 5e-3) |
| Veto status | No hard vetoes (finite values, no condition vetoes, no ties). The failure is a promotion veto on the candidate, NOT a continuation veto on the program |
| Main uncertainty | Whether branch compression can hold boundary rank bounded over T=120 without violating score smoothness (V3) |
| Next justified action | (1) Plan amendment: branch-decomposed target assembly + declared compression policy (V3 analysis required); (2) implement; (3) rerun this smoke, then the full ladder |
| Not concluded | No rank-scaling claim (ladder not run); no tau-accuracy claim (viability screen only, per D1/F4); no cross-model claim |

## What must not be concluded

The 5e-3 smoke failure does not invalidate P1A (exact-marginal machinery
is proven), the retained-type design, the score derivation, or the
program direction — the failing component is the step-target assembly
route, and the repair is anchored in the source's own retained-structure
(Prop. 2). The tau findings are viability evidence only.

## Post-run red team

- Strongest alternative explanation for the plateau: Monte-Carlo row noise
  rather than kinks. Rejected: the plateau is invariant to row count
  (512->4096) and the step-0 fit reaches 1e-5 with the same rows.
- What would overturn the branch-repair recommendation: a demonstration
  that tau-tuning alone reaches ladder tolerances at higher resolution —
  worth one cheap probe (tau grid x degree grid at n=1) before investing
  in branch machinery.
- Weakest evidence: single seed, n=1, T=8; the sharp tau landscape may
  differ materially at higher n.

## Addendum (same session): branch-axis repair implemented and validated

Design note: `bayesfilter-squared-tt-engine-branch-axis-design-2026-08-16.md`.
The branch-axis formulation (branch index as a discrete TT axis with
counting-measure mass; smooth signed branch targets u_g*sqrt(G); retention
split keeps the branch axis in the integrated suffix) ELIMINATES both the
|h| kinks and the boundary-rank growth — no V3 amendment needed. The
score-path smoothness guard (PD-Gram Cholesky + conditioning veto) is
deferred to P2 as a claim gate; the value path uses an eigh-based factor
(any L with LL'=E is exact for values).

Implementation: `run_value_filter_branch_axis` + `DiscreteIndicatorBasis1D`
in `squared_tt_engine_v0_tf.py`. Two implementation defects found and
fixed in-session: (1) hard Gram-conditioning veto misapplied to the value
path (near-singular E is expected for near-Gaussian laws and benign for
values); (2) indicator-axis zero-initialization starved branches g>0 in
ALS environments — replaced by frozen stateless-random init.

Probe results (n=1, T=8, LGSSM vs exact Kalman, N=768):
| config | gap | max fit rms |
|---|---|---|
| naive route best (tau=1e-3) | 1.0e-2 | 3.9e-2 (kink plateau) |
| branch-axis tau=1e-6 deg=14 r=3 s=3 | 2.4e-2 | 2.9e-3 |
| branch-axis tau=1e-4 deg=14 r=4 s=3 | **7.4e-3** | 2.8e-3 |

The fit-rms plateau is GONE (2.2e-2 -> 2.8e-3, responsive to resolution):
the kink mechanism is confirmed repaired. tau is no longer needed for
smoothing (1e-6 works), reverting tau to its intended starvation-insurance
role — consistent with D1. Remaining 7.4e-3 vs the declared 5e-3 smoke
gate is resolution-limited; next action is one resolution notch
(deg 16, r 4, N 1024, sweeps 3), flip the xfail gates, then the ladder.
