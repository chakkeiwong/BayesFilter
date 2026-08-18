# Experiment plan: P1B LGSSM value ladder (rank-sufficiency curve)

Date: 2026-08-17
Program plan: `bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
(rev 3, P1B section incl. branch-axis amendment).
Design: `bayesfilter-squared-tt-engine-branch-axis-design-2026-08-16.md`.
Prerequisite: P1A passed; both P1B smoke gates flipped to pass at declared
tolerances (n=1: 1.2e-3 vs 5e-3; n=2: 1.2e-2 vs 2e-2).

## Question

How does the frozen rank r required for a declared same-target value
tolerance grow with state dimension n for the branch-axis squared-TT
engine on LGSSM (exact Kalman truth), and how do retained-Gram boundary
rank/conditioning behave along the recursion?

## Evidence contract

- DGP/truth: the smoke module's LGSSM family (A = 0.7 I + 0.1 off-diag
  perturbation, Q = 0.4 I, H = I, R = 0.5 I, m0 = 0, P0 = I), exact
  Kalman log-likelihood as EXACT_ORACLE per (n, seed).
- Rungs: n in {2, 4, 8}; T = 8; ranks r in {2, 4, 6, 8}; 3 seeds per
  (n, r) — seeds 42, 142, 242 (+n offsets). n in {16, 32, 64} deferred to
  a follow-up rung after the cost profile of n=8 is known (stop condition
  below). Scattered frozen rows ONLY (V2; quadrature rows are diagnostic
  and n<=2).
- Fixed controls: basis_degree 12, sweeps 3, ridge 1e-10, tau 1e-6,
  half-width 3.0, N rows = 4096 (n=2), 8192 (n=4), 16384 (n=8).
- PRIMARY promotion criterion (declared now): mean |gap|/T <= 2.5e-3 per
  step (i.e. |total gap| <= 2e-2 at T=8) at some r within budget defines
  r*(n).
- Hard vetoes: nonfinite value; condition-number veto; symmetry assert.
- Explanatory only: fit rms; suffix-Gram condition; wall time; per-step
  drift profile.
- Stop condition: if a single (n=8, r=8) run exceeds 45 min CPU, record
  and stop the ladder there (resource bound; larger n needs the batched/
  XLA engine of P3, not eager loops).
- Not concluded: no claim beyond LGSSM-family fixtures; no T=120 claim
  (T-scaling is a separate rung after P2A per audit Finding 5); no
  cross-model transfer; descriptive statistics only at 3 seeds
  (mean/range; no ranking language).

## Command

`python docs/benchmarks/run_p1b_lgssm_value_ladder_20260817.py --output
docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt01/result.json`

## Pre-mortem

- Pass-for-wrong-reason: near-Gaussian LGSSM may be too easy for rank —
  that is exactly what the ladder measures; the adversarial-structure arm
  (correlated A) is included at n in {4, 8} via one extra seed with
  A = 0.7 I + 0.25 dense perturbation.
- Fail-for-wrong-reason: under-resolved N at n=8 could masquerade as rank
  insufficiency; N doubles once (32768) on the failing r before the
  failure is attributed to rank.

## Amendment A1 (2026-08-18, declared before attempt03 execution)

Evidence basis: attempt02 result note
(`bayesfilter-p1b-lgssm-value-ladder-attempt02-result-2026-08-18.md`)
and its focused diagnostics: MC row-sampling bias dominated attempt01/02
(row budgets under-resolved >= 16x at n=2); the r=4 floor ~9e-3 is
rank-limited; quadrature r=6 clears the declared tolerance on the ladder
fixture (2.314e-3 vs 2.5e-3); rotated-Sobol rows reach the quadrature
floor at 16384 rows (2.550e-3) where MC needs >= 65536.

Amended design for attempt03 (all other contract fields unchanged:
oracle, per-step tolerance 2.5e-3, horizon 8, seeds, 45-min/cell stop):
- row_design = "sobol" (frozen randomized-QMC scattered rows:
  Sobol + Cranley-Patterson rotation, seed-frozen; V2-compliant
  scattered design, engine option `EngineConfig.row_design`);
- scope reduced to n=2 with rows=32768, ranks {4, 6}; r=8 runs only as
  a contingency if r=6 fails a standard seed (its projected cell wall
  exceeds the declared stop otherwise);
- n in {4, 8} are EXCLUDED from attempt03: brute-force row scaling is
  not affordable there and a row-design/feasibility decision belongs to
  a separate planned artifact after r*(2) exists.

Declared reading: r*(2) = smallest rank whose standard-arm seeds all
pass, as before. HONESTY BOUND declared in advance: the r=6
resolution floor on this fixture (2.31e-3, quadrature single run) sits
8% below the tolerance, so a pass at r=6 is scope-marginal; the note
must report per-seed margins and must not convert a marginal pass into
a robustness claim. A failure of r=6 under Sobol at 32768 rows is a
fixture-tolerance boundary finding, not a rank-structure verdict.
