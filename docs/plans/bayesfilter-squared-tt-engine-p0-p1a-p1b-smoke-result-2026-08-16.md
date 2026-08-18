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

## Addendum 2 (2026-08-17): both smoke gates closed at declared tolerances

Engine deltas since Addendum 1: (a) optional tensor-GL quadrature rows for
small-n diagnostic runs (`quadrature_order`; scattered rows remain the
ladder default per V2); (b) branch rows now carry mu-weighted z-row weights
(counting measure over branches); (c) scale-relative symmetry assert in
RetainedQuadraticForm (absolute 1e-10 tripped on benign float noise at
n=2 Gram scale).

Gate results (declared tolerances unchanged):
- n=1: gap 1.2e-3 vs 5e-3 -> PASS (deg 16, r 4, q 24, tau 1e-6, hw 4)
- n=2: gap 1.2e-2 vs 2e-2 -> PASS (deg 12, r 6, q 14, tau 1e-6, hw 3)
- naive route retained as strict-xfail historical record.
- Suite: 7 passed, 1 xfailed (42 min CPU).

Rank evidence at n=2: r=4 -> 5.1e-2 (fail), r=6 -> 1.2e-2 (pass): the
rank requirement is real and measurable — exactly what the P1B ladder
now quantifies. tau=1e-6 suffices post-branch-repair (starvation
insurance only), consistent with D1.

Next: P1B ladder per
`bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md` (n in {2,4,8},
r in {2,4,6,8}, 3 seeds + adversarial arm, declared per-step tolerance
2.5e-3, resource stop 45 min/cell).

## Addendum 3 (2026-08-17): P2 adjoint engine, program v0.2/v0.3, ladder attempt01 classification

P2A DECISION (result note bayesfilter-p2a-cost-prototype-result-2026-08-17.md):
forward tangent replay measured at ~326x value cost at p=300 (gate <=6x)
-> ADJOINT selected; solver-reuse checks passed (8e-17, 4.7e-11).

P2 IMPLEMENTED: UB-1 Addendum A (manual adjoint derivation, incl. corrected
solve-node adjoint bar_A = W(g-Ac)lambda' - (WAlambda)c'); node primitives
squared_tt_adjoint_tf.py with 7/7 pairing/FD tests green
(tests/highdim/test_p2_adjoint_nodes.py; two bugs found+fixed: suffix-list
off-by-one, Cholesky-VJP solve ordering); full-path engine
squared_tt_adjoint_engine_tf.py (forward trace + reverse sweep).

PROGRAM REPAIRS while validating (each FD-quality-first diagnosed):
- v0.2 RELATIVE defensive mass: tau_abs = tau*Z_h_prev (defensive FRACTION
  shift-invariant). The v0/v0.1 absolute tau made the VALUE jump ~2e-5
  under 1e-6 theta-perturbations at shift-branch switches (C0 violation).
- v0.3 SMOOTH SHIFT: s = logsumexp(log_f) - log N replaces argmax max-shift
  in BOTH engines. Diagnosis: at n=2 fit resolution, argmax switches are
  generic along theta and each causes an O(fit-error) value jump -> FD
  invalid almost everywhere; the smooth shift removes the branch structure
  entirely (F6 strengthened: no tie machinery needed in these engines).
- Same-scalar unification: value engine now uses the SAME Cholesky branch
  factor as the score engine with a DECLARED relative Gram floor
  (branch_gram_floor=1e-12, config field) - eigh factor removed (V5).

ADJOINT VALIDATION STATE:
- n=1 (T=4): value-eq 0.0; adjoint vs FD rel 4.7e-10 -> EXACT.
- n=1 self-FD of the score engine: 2.8e-10 (machinery proven).
- n=2 (T=4): value-eq 0.0; FD ladder still shows ~1e-6-scale value wiggles
  (step 1e-3 -> agree ~1%; smaller steps diverge). OPEN: source of the
  residual n=2 wiggle not yet identified (eigvalsh telemetry is
  value-inert; Cholesky+floor smooth). Adjoint-vs-FD at n=2 is therefore
  INCONCLUSIVE pending either the wiggle diagnosis or I-P2-4 (full-path
  forward-JVP cross-check, FD-independent) - next work item.

P1B LADDER attempt01 (docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/):
ran under the OLD v0.1 program (absolute tau, max-shift, scattered rows);
r_star null at n in {2,4}; resource stop at n=4 r=8 (5175s > 45min/cell).
CLASSIFICATION: NOT rank evidence - the v0.1 program's shift-jump noise
floor (diagnosed above) sits at/above the declared 2.5e-3/step tolerance
for scattered-row fits. Ladder MUST be rerun under v0.3. attempt01 is
preserved as the program-defect discovery artifact.

NEXT (in order): (1) diagnose the residual n=2 value wiggle (suspects:
near-singular-E Cholesky numerics at 3x3, LAPACK-level; or fit
conditioning); (2) I-P2-4 forward-JVP full-path cross-check; (3) update
tests/plan for v0.3 semantics (U-TAU relative-mass form; smoke gates
rerun); (4) P1B ladder rerun under v0.3; (5) T=120 adjoint-state stress.

## Addendum 4 (2026-08-17, continuation): wiggle root cause, I-P2-4 instrument, conditioning repair identified

WIGGLE ROOT CAUSE (item 1 of the recorded next actions): the residual n=2
value roughness is the RANK-DEGENERATE retained Gram, not the floor: at
rank 3 the third Gram eigenvalue sits at numerical zero and its Cholesky
column rotates erratically with theta (floor-independent, +-2e-5 value
wiggles); at rank 2 the value is smooth (clean linear consecutive diffs).
CONSEQUENCE (tuning-procedure gate, recorded): claim-bearing scopes must
select rank so the retained Gram passes the conditioning bound - rank vs
Gram-conditioning is a smoothness gate, matching the score engine's
existing veto. FD-gate test updated to the well-conditioned regime
(rank 2 at n=2) with the diagnosis documented in the test.

I-P2-4 BUILT (item 2): full-path forward-JVP instrument
(tests/highdim/test_p2_adjoint_vs_forward_jvp.py) chaining the donor
ordered ALS value+JVP with the P1A retained tangents - FD-independent
cross-check of the adjoint.

FINDINGS:
- Instruments run the IDENTICAL value program (value-diff exactly 0.0).
- n=1 adjoint remains exact (FD rel 4.7e-10; suite 13 passed + 1 xfailed).
- n>=2: adjoint vs forward-JVP gap is CONDITIONING-DRIVEN (3.4e-3 at the
  fd-config; 6.3e-4 at deg6/q10, i.e. 5x better with better conditioning);
  FD sits between them and is inconsistent with itself below ~1e-3.
  DIAGNOSIS: BOTH derivative chains route their per-update solves through
  UNSCALED normal equations (donor fixed_design_lsq_derivative and
  solve_node_adjoint's lambda solve) while the VALUE program solves via
  the scaled augmented path - at ill-conditioned fits each derivative
  chain loses digits independently. This is exactly P2A obligation (b)
  surfacing at implementation level.

IDENTIFIED REPAIR (next work item, bounded): route the derivative solves
(donor dot_c solve; adjoint lambda solve) through the same scaled
augmented factorization as the value solves; then I-P2-4 (currently
strict-xfail with the full diagnosis) tightens to 1e-9 and the FD gate
extends to n>=2 in well-conditioned scopes.

REMAINING QUEUE (unchanged): v0.3 suite sync (smoke gates rerun under
smooth-shift semantics - n=1 gate already re-verified green this session),
P1B ladder rerun under v0.3, T=120 adjoint-state stress.

## Addendum 5 (2026-08-17, fresh session): scaled-solve repair VERIFIED; two residual test defects diagnosed and fixed

Verification run (Section 6 of the relaunch memo, first execution of the
scaled-solve repair): initial suite result 8 passed / 2 FAILED
(I-P2-4 rel 3.4e-3; I-P2-1 n=2 FD rel 4.2e-4). Both failures were
TEST-SIDE defects, not adjoint-engine defects; the engine code was not
modified in this session.

Defect A (I-P2-4 instrument, `test_p2_adjoint_vs_forward_jvp.py`): the
forward-JVP filter's inline Cholesky JVP used `dot_L = L @ Phi^T`;
the correct formula is `dot_L = L @ Phi` with
`Phi = tril(L^-1 dM L^-T)` minus half its diagonal. Verified standalone
against centered FD of `tf.linalg.cholesky` (wrong form rel ~1.3;
correct form rel 7.5e-11). Fix applied to the test; engine's
`cholesky_vjp` (the transpose map) was already consistent with the
correct JVP. After the fix the residual gap dropped to 4.4e-6 —
conditioning-shaped, leading to:

Defect B (shared test config `_config(n=2)`): `quadrature_order=8` with
`basis_degree=10` gives 8 Gauss nodes per axis against 11 basis
functions per axis — the per-core ALS design is column-rank-deficient
and the fit solve is ridge-dominated (measured max cond(A'WA + rho I)
= 3.2e14 at qo=8 vs 9.8e4 at qo=12). In that near-null space the two
independent derivative chains (scaled or not) legitimately differ at
~1e-5. This is a test-fixture defect (under-resolved quadrature), not a
failure of the scaled-solve repair. Fix: qo 8 -> 12 for n=2 (n=1 stays
14), with the constraint documented in `_config`.

Bisect evidence: T-horizon sweep localized onset at the first
transition step (T=2); qo sweep {8: 2.6e-5, 11: 6.7e-16, 12: 9.2e-15}
cleanly separates conditioning from analytic error; pairing tests of
`_prefix_gram_adjoint`, `gram_chain_adjoint`, `prefix_rows_adjoint`
against their forward tangents on random mixed-basis cores all paired
at <= 1.1e-14.

RESULT after the two test fixes (all at qo=12 for n=2):
- I-P2-4 adjoint vs forward-JVP, n=2, T=4: rel 2.4e-12 / 1.1e-12
  (assert 1e-9) — the scaled-solve repair is CONFIRMED as the decisive
  FD-independent gate.
- I-P2-1 FD, n=2: rel 1.0e-9 (assert 1e-6) — FD gate now extends to
  n=2 in the well-conditioned regime as predicted by Addendum 4.
- Full suite `test_p2_adjoint_nodes.py + test_p2_adjoint_vs_forward_jvp.py
  + test_p2_adjoint_engine_fd.py + test_p1a_retained_quadratic_form.py`:
  15 passed, 0 failed, 70 s. The previous strict-xfail on I-P2-4 is
  gone (assert active).

Verdict ledger: scaled_normal_solve / forward_jvp_replay_scaled /
adjoint lambda routing — correct (verified via I-P2-4 at 1e-12 and
U-ADJ-SOLVE-1 green). Pre-fix I-P2-4 instrument — wrong relative to the
Cholesky JVP it claimed to implement. qo=8 n=2 fixture — wrong relative
to the resolved-fit regime the gate declares. Not concluded: anything
about n>2, T>4, ladder ranks, or ill-conditioned-fit gradient accuracy
(the conditioning gate remains a tuning-procedure obligation).

Run manifest: branch main @ dae37183 (working tree, uncommitted);
`/home/chakwong/anaconda3/envs/tf-gpu/bin/python` (TF 2.19.1), CPU-only
via CUDA_VISIBLE_DEVICES=-1 (GPU intentionally hidden); pytest wall
70 s; no artifacts beyond this note and the two test-file edits
(`test_p2_adjoint_vs_forward_jvp.py` dot_chol fix,
`test_p2_adjoint_engine_fd.py` qo=12 + comment).

Note: the harness Bash-permission-classifier outage reported in the
relaunch memo did not reproduce; all commands ran normally.

REMAINING QUEUE (unchanged from Addendum 4, item 1 now closed):
v0.3 n=2 smoke gate rerun (~40 min), P1B ladder rerun under v0.3
(attempt02), T=120 adjoint-state stress.

## Addendum 6 (2026-08-18): v0.3 smoke gates closed, ladder attempt02 classified, T=120 stress gate failed and repaired, phase review

QUEUE ITEM 2 (v0.3 smoke reruns): both branch-axis gates GREEN at the
declared tolerances under v0.3 (n=1 fast; n=2 in 3285 s). Naive-route
strict-xfail record intact. Suite state: adjoint gates 15/15 plus both
smoke gates.

QUEUE ITEM 3 (P1B ladder attempt02): ran to the declared resource stop
(25 cells, 16622 s). ALL cells fail 2.5e-3/step; r_star null everywhere.
Full classification, diagnostics, and decision/inference tables in
`bayesfilter-p1b-lgssm-value-ladder-attempt02-result-2026-08-18.md`.
Verdicts: attempt02 is INVALID AS RANK EVIDENCE (fixture row budgets
under-resolved >= 16x at n=2, measured by a row-scaling diagnostic
5.18e-2 -> 9.7e-3 per-step over 4096 -> 65536 rows); the memo's
shift-jump-noise explanation of attempt01 is REFUTED for the ladder
(v0.1/v0.3 gaps agree cell-by-cell); the r=4 floor ~9e-3 is
rank-limited (same-fixture quadrature arm matches scattered-65536).
Promotion veto on the fixture design only — NOT a continuation veto;
harness and engine not invalidated (quadrature smoke passes at
1.5e-3/step, r=6).

QUEUE ITEM 4 (T=120 adjoint-state stress; plan
`bayesfilter-p2-t120-adjoint-stress-plan-2026-08-17.md`, artifacts
`docs/benchmarks/artifacts/p2_t120_adjoint_stress_20260817/`):
- attempt01: value arm 214 s / 0.81 GiB; adjoint arm 511 s / 19.53 GiB
  peak RSS -> the declared 16 GiB feasibility gate FAILED. Log-lik
  bit-identical between engines at T=120 (same-program at full horizon).
  Mechanism: stored per-update design matrices dominate the trace
  (linear in T) — the pre-mortem's named failure mode.
- Store-vs-recompute decision (taken here, the phase review): designs
  are a deterministic function of (basis, rows, weights, target,
  pre-update cores, core_index), and a core's own design does not
  contain that core, so the reverse-sweep walk state suffices for a
  BIT-IDENTICAL rebuild. Implemented: drop designs from the trace,
  rebuild per update in the reverse sweep
  (`squared_tt_adjoint_engine_tf.py`).
- Verification: I-P2-4 + FD gates re-run GREEN post-change; attempt02
  adjoint arm: 653 s / 1.69 GiB (gate PASSES with 9.4x margin), value
  and grad_norm BIT-IDENTICAL to attempt01. Wall overhead vs attempt01
  +28% (descriptive; both arms ran with a concurrent diagnostic,
  load averages recorded in the artifacts).

QUEUE ITEM 5 (phase review): master plan updated to REVISION 4 —
v0.2 relative-tau and v0.3 smooth-shift semantics folded into Sections
3.2/3.4 (argmax shift and A15 tie machinery retired for these engines);
scaled-derivative-solve requirement recorded in 3.4; rank-conditioning
+ quadrature-resolution gate added to the Section 8 tuning procedure.
`sqrt_target_adjoint` documented as pre-v0.3 node-test-only (not on the
active score path).

POSITION: P2 obligations are closed (score verified n<=2 at 1e-12 by
I-P2-4; T=120 resource gate passed after repair). P1B's scientific
product r*(n) remains OPEN pending an attempt03 under amended row
budgets/design (measured scaling now in hand; QMC/importance rows are
the identified V2-compliant lever). Next phases per plan: P3 (XLA
port), P4 (adapters), P5 (tuning), P2S (after UB-3 focused review), P6.

Honest limits: all score evidence is n<=2, T<=4 well-conditioned
regimes (T=120 measured for RESOURCES only, no gradient-accuracy oracle
there); ladder mechanism (normalizer error-mass inflation) is a
supported hypothesis, not isolated; no ranking claims anywhere (seeds
too few; descriptive language only).

## Addendum 7 (2026-08-18): P1B n=2 rung closed — r*(2) = 6

attempt03 (Sobol 32768 rows per Amendment A1, no resource stop, 7468 s):
r=6 passes the declared 2.5e-3/step tolerance on all three standard
seeds (margins +8/+37/+59%); r=4 fails on all three. First valid r*
data point of the program:  r*(2) = 6. The +8% seed sits at the
measured resolution floor — scope-marginal per the pre-declared honesty
bound; no robustness or Sobol-default claim. Full tables and red-team
note: `bayesfilter-p1b-lgssm-value-ladder-attempt03-result-2026-08-18.md`.

SESSION QUEUE COMPLETE (relaunch memo Section 6, items 1-5): scaled-solve
repair verified (Add. 5); v0.3 smoke gates green; ladder classified and
repaired through attempt03; T=120 stress measured, gate repaired and
re-passed (Add. 6); plan at revision 4. NEXT: n>=4 row-design/
feasibility note, then P3 (XLA). Working tree remains uncommitted
pending owner checkpoint approval.
