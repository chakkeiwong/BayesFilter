# Row-Design Contract v2: Proposal-Weighted Frozen Rows (Design Note) — 2026-08-19

Status: `DRAFT_FOR_REVIEW` (program-contract change: touches the declared
row design and mu-weight bookkeeping; per the plan's design-note
discipline this lands BEFORE engine implementation. A step-level
validation probe is included as evidence; it changes no engine code.)

Motivating evidence: note Section 6 of
`bayesfilter-p1b-n4-row-design-note-2026-08-18.md` — uniform-box rows
collapse against the concentrated transition-step target (ESS 11/8192
at n=4; per-step -2..-3 nat normalizer under-count at every step).

## 1. Contract derivation

The declared program evaluates all step integrals with respect to the
normalized reference measure mu(dz) = prod_i dz_i / 2 on the box
B = [-1,1]^d (d = 2n at transition steps). The current design draws
z_i uniform-mu (MC or rotated Sobol) with weights w_i = 1/N, so that
for any f:  sum_i w_i f(z_i) -> int_B f dmu.

v2 replaces ONLY the (rows, weights) pair. Let q be a PROPOSAL density
with respect to mu on B (q > 0 mu-a.e., known pointwise). Draw
z_i ~ q (frozen by seed), and set

    w_i = 1 / (N * q(z_i)).

Then E_q[ sum_i w_i f(z_i) ] = int_B (f/q) q dmu = int_B f dmu: every
downstream quantity — the weighted ALS objective
sum_i w_i (h(z_i) - g(z_i))^2 ~ ||h - g||^2_{L2(mu)}, the branch-row
replication (weights copied per branch, counting measure unchanged),
and the column scaling — is UNCHANGED IN MEANING. The exact Gram
normalizer Z_h = int h^2 dmu never used rows and is untouched. The
declared program remains "fit in L2(mu), normalize exactly"; only the
quadrature-design measure changes. V2 compliance: rows are still a
frozen scattered design fully determined by (count, dimension, seed,
declared proposal family + its frozen parameters); no runtime
adaptation inside a filter pass (V1) — the proposal for step t may
depend only on ALREADY-COMPUTED state (retained object from t-1, y_t),
exactly like the frozen seeds today.

Bookkeeping in engine coordinates: mu has Lebesgue density 2^{-d} on B.
For a proposal built in PHYSICAL coordinates x = half * z with Lebesgue
density p_x, the mu-density is q(z) = p_x(half*z) * half^d * 2^d, and
the weight is w_i = 2^{-d} * half^{-d} / (N * p_x(x_i)) after
cancellation — implemented once, unit-tested against int 1 dmu = 1.

Defensive mixture (required): q = (1-alpha) * q_prop + alpha * 1
(alpha uniform-mu mass, alpha ~ 0.1-0.5 declared per scope). This
bounds w_i <= 1/(N*alpha) (no weight blowup where the proposal
underweights), guarantees q > 0, and preserves box-wide coverage for
the defensive-tau machinery. Proposal samples falling outside B are
rejected-and-redrawn; the truncation correction is absorbed into a
declared normalization check (the int 1 dmu ~ 1 diagnostic must pass
at 1e-2 or the scope's proposal is rejected).

## 2. Proposal families (per adapter mode)

- F1 (generic, model-free): x_p-block from the RETAINED law of step
  t-1 — the squared-TT retained object is exactly sampleable by the
  Zhao-Cui marginal chain (Prop. 2 structure; author route
  `@TTSIRT/eval_irt`), frozen by seed; x_c-block from a declared
  spread family around the sampled x_p (scope-tuned covariance).
  No adapter change. This is the v2 DEFAULT candidate.
- F2 (adapter-assisted): optional adapter field
  `transition_proposal_sample(x_p, y_t, seed)` + matching log-density;
  exact conjugate proposals for Gaussian-kernel models (used by the
  probe below). Adapter contract addition; requires the F2 adapter to
  supply BOTH sampler and density (fail-closed if either missing).
- F3 (pilot-fit): one uniform pilot fit, then rows from the pilot h^2
  by TT sampling. Two fits per step (cost x2); fallback if F1 spread
  tuning proves fragile.

Selection, spread/alpha values, and per-scope validation (ESS floor,
normalization check, n=2 regression vs the known-good uniform route)
belong to tuning procedure v1.1 as new scope-identity fields.

## 3. Validation probe (step-level, no engine change)

`docs/benchmarks/run_rowdesign_v2_probe_20260819.py`: rebuilds the
t=1 transition-step increment OUTSIDE the engine (same math as the
engine step; pattern of the I-P2-4 instrument) on the n=4 fixture,
with (a) the current uniform-Sobol design and (b) an F2 conjugate
proposal (LGSSM: x_p ~ N(m0_filt, P0_filt), x_c | x_p ~ N(C(Q^{-1}A x_p
+ R^{-1}y), C), C = (Q^{-1}+R^{-1})^{-1}, alpha = 0.25), both at
N = 8192, r = 6, deg 12, hw 3.0. Reported: per-step increment error vs
exact Kalman, achieved ESS, normalization diagnostic.

RESULTS: pending (filled by the probe run below).

Success criterion for the DESIGN DIRECTION (not promotion): the
importance arm's t=1 increment error improves by >= 10x over uniform
at equal N; ESS >= 500. Failure sends the note back to F3/pilot-fit
before any engine work.

## 4. Non-claims

No engine behavior changes with this note. F1's spread-family tuning
is undesigned (open). No cross-model transfer of alpha/spread; no
claim that ESS alone certifies fit quality (it is a necessary-coverage
diagnostic). The probe is single-seed, single-step, one fixture.

## 3a. Probe RESULTS (2026-08-19/20): Section 1 design REFUTED

`/tmp/rowdesign_v2_probe.log`, n=4 fixture, t=1 step, N=8192, r=6:

| arm | incr err (nats) | fit rms | ESS |
|---|---|---|---|
| uniform Sobol (current) | -1.71 | 4.7e-3 | 11 |
| proposal ISLS (alpha .25) | +7.28 | 2.5e-1 | 15 |
| proposal q-fit | +13.6 | 6.8 | 15 |
| q-fit deg 16 / 20 / 24 | +26.8 / +43.5 / +62.0 | 8.9..51 | 15 |
| ISLS alpha .90 / .75 / .50 | +11.1 / +13.8 / +23.2 | 0.38..0.58 | 13/14/6 |

VERDICT: wrong relative to the drafted claim. Importance rows flip the
error from mild under-count to catastrophic OVER-count, degree makes it
monotonically WORSE, and even 90%-uniform mixtures are far worse than
pure uniform. Mechanism (consistent across all arms): once rows include
near-manifold points, the sqrt-target the fit must match carries the
target's full dynamic range (amplitudes ~e^{10+} relative to the bulk
after the smooth shift); a rank-6/deg-12..24 polynomial TT cannot
represent that spike over the fixed box, the fit oscillates, and the
EXACT normalizer Z_h = int h^2 integrates the oscillation into spurious
mass. Uniform rows "work better" only because they hide the spike.
ESS also stays ~15 for all mixtures — the defensive-mixture IS design
cannot concentrate where BOTH constraints (peak + box control) bind.

REDIRECT (root-root cause): the binding constraint at n>=4 is the
FIXED coordinate box: the step target concentrates on a correlated
O(sd/hw)^{2n} fraction of [-hw,hw]^{2n}, so its reference-coordinate
representation has unbounded dynamic range as n grows. Row design
cannot fix a representation problem. The program-level repair direction
is per-step ADAPTED coordinate maps (offset/scale — and for the
correlation ridge, the source literature's layered/preconditioned
composition: DIRT-style deep inverse Rosenblatt / temperature layering,
which is the authors' own answer to concentration). The engine
architecture already reserves this (plan Section 5: engine owns
coordinate maps + support/scale hints; AffineCoordinateMap exists;
per-step frozen maps from ALREADY-COMPUTED state are V1/V2-compliant
like frozen seeds).

NEXT ARTIFACT: a paper-grounded design note for adapted/composed maps
(read the Zhao-Cui/TT-DIRT concentration treatment sections + author
code first, per the literature discipline) — NOT more row-design
sweeps. Status of THIS note: `REFUTED_REDIRECTED` (kept as the
evidence record for why importance rows alone are a dead end).
