# P1B n>=4 Row-Design / Feasibility Note (draft under execution) — 2026-08-18

Status: `IN_PROGRESS` — Section 3 awaits the running/queued measured
diagnostic; everything else is evidence-in-hand.
Follows: `bayesfilter-p1b-lgssm-value-ladder-attempt03-result-2026-08-18.md`
(r*(2)=6 under Sobol 32768 rows).

## 1. Question

Under what row design and budget can the branch-axis value ladder
produce a VALID r*(n) at n in {4, 8} within the declared per-cell
resource bound (45 min), given that attempt02 established row-sampling
bias as the dominant error at the original MC budgets?

## 2. Evidence in hand (measured, n=2 unless noted)

- MC row bias decays slowly: 4096 -> 65536 rows gives per-step
  5.2e-2 -> 9.7e-3 at r=4 (rank floor ~9e-3 reached only at 16x the
  declared budget).
- Rotated Sobol reaches the r=6 resolution floor at 16384 rows
  (2.55e-3 vs quadrature 2.31e-3); MC needs >= 65536.
- attempt03 (Sobol 32768): r=6 passes all seeds; walls ~2000-2300 s per
  r=6 cell at n=2 (fit variables 2n+1 = 5 axes).
- attempt02 n=4 walls (MC 8192 rows): r=6 ~1470 s, r=8 ~5330 s (over
  the 45-min stop). Fit variables at n=4: 9 axes; rows enter the ALS
  design assembly linearly, ranks quadratically in the environment
  contraction.
- Dimension scaling caveat: QMC advantage degrades with effective
  dimension; the n=2 gain (4x rows) is NOT assumed to transfer to
  2n=8/16-dimensional row spaces (plan discipline: cross-task transfer
  is a hypothesis, not a default).

## 3. Discriminating diagnostic (single seed, descriptive, scoping-only)

n=4, r=6 warm-start hypothesis, Sobol rows in {8192, 16384, 32768}:
per-step gap and wall per cell. Decision rule declared in advance:
- If per-step approaches a flat floor above 2.5e-3 across the row
  sweep, the binding constraint at n=4 is rank/resolution, not rows ->
  next arm varies rank at the best affordable row budget.
- If per-step is still falling at 32768 rows, rows remain binding ->
  the ladder needs either a larger budget with a longer per-cell stop
  (owner resource decision) or a better row design (importance rows).
- If walls exceed ~40 min at the row budget needed, the n=4 ladder is
  NOT feasible pre-P3; defer the n>=4 ladder until the XLA port lands
  and re-measure (V12: no feasibility language without measured walls).

RESULTS (2026-08-18, `/tmp/p1b_n4_rowdesign_diag.log`, script
`docs/benchmarks/run_p1b_n4_rowdesign_diag_20260818.py`, single seed 42,
n=4, r=6, deg 12, hw 3.0):

| rows (Sobol) | per-step gap | max fit rms | wall |
|---|---|---|---|
| 8192  | 2.291 | 7.1e-2 | 1386 s |
| 16384 | 2.007 | 7.9e-2 | 2745 s |
| 32768 | 1.800 | 8.7e-2 | 5508 s |

Reading (complete, single seed, descriptive): FLAT-FLOOR branch of the
declared decision rule. 4x rows buys 21% total where the same lever at
n=2 bought an order of magnitude; Sobol-8192 is descriptively equal to
attempt02's MC cells (2.05-2.46) at the same fixture. The n=4 error at
this fixture is NOT row-dominated; the ~2/step magnitude (~1000x
tolerance) indicates a rank/resolution/fixture constraint. Candidate
mechanisms for the next discriminating arm (one variable at a time):
r=6 far below r*(4); basis degree 12 under-resolving the 4-D marginal
at half-width 3.0; retained boundary-rank/branch-count growth outpacing
the fit. Walls independently settle feasibility: 16384-row cells cost
~46 min and 32768-row cells ~92 min against the ladder's declared
45-min stop, so ANY n=4 resolution/rank upgrade requires the P3 XLA
port first (V12: measured statement for this CPU route only).

DECISION (per the pre-declared rule, third branch): defer the n>=4
ladder until after P3; run the mechanism-discriminating arms (rank,
degree, half-width) at XLA speed. P3 is now the program's critical
path.

## 4. Non-claims

No r*(4) or r*(8) statement exists yet; r=6 at n=4 is a warm-start
hypothesis only; single-seed diagnostics nominate, they do not promote.

## 5. Post-P3.3 feasibility update and mechanism arms (2026-08-19)

XLA value engine (P3.3, parity-gated 2.8e-14): the n=4 Sobol-8192 r=6
cell runs in 603 s including compile, 522 s warm, vs 1386 s eager
(`/tmp/p33_n4_xla_wall.log`; XLA reruns bit-identical). Projected from
cost scaling (~rows * cols^2, cols ~ rank^2 * basis_dim): r=8 cells
~27 min, deg-16 cells ~15 min — INSIDE the 45-min budget. The deferral
decision of Section 3 is lifted for BOUNDED single-seed mechanism arms.

Declared arms (one variable at a time vs the Section 3 baseline
per_step 2.209 at r=6/deg12/hw3.0/sobol8192; single seed, descriptive,
XLA engine):
- ARM-R: rank 8 (r*(2)=6 may not transfer; tests rank-boundness).
- ARM-D: basis degree 16 (tests marginal resolution).
- ARM-H: half-width 4.0 (tests support truncation at n=4 excursions).
Decision rule: an arm that moves per_step by >= 5x nominates its
variable as the binding constraint and justifies a follow-up ladder
plan; all arms flat (< 2x) escalates to a structural suspects review
(branch/boundary-rank growth, target assembly at n=4) before any
further compute. No arm result is promotion evidence.

### Section 5 results (2026-08-19, XLA engine, single seed, descriptive)

| arm | per-step gap | vs baseline 2.209 | max fit rms | wall |
|---|---|---|---|---|
| ARM-R rank 8 | 2.480 | +12% (flat) | 6.1e-2 | 2325 s |
| ARM-D deg 16 | 1.952 | -12% (flat) | 5.6e-2 | 941 s |
| ARM-H hw 4.0 | 4.656 | +110% (worse) | 2.9e-2 | 512 s |

(ARM-H's first attempt died in XLA's LLVM compile after two prior
large compilations in the same process — infrastructure failure,
rerun clean in a fresh process; not a veto.)

DECISION (per the declared rule): no arm improved >= 5x; ARM-R and
ARM-D are flat; ARM-H made things WORSE while its in-sample rms
IMPROVED — support truncation is not the binding constraint, and
in-sample rms is confirmed (again) as a misleading proxy for value
error at n=4. The rule's all-flat branch fires: STOP compute, run the
structural-suspects review before any further n=4 cells. Suspects,
in priority order:
1. branch/boundary-rank growth: at n=4 the retained boundary rank
   (hence branch count and fit column count) is larger, and the
   branch-axis target must be jointly fit across all branches — the
   per-branch effective resolution shrinks with n even at fixed
   rank/degree;
2. target-assembly conditioning at 2n+1 = 9 axes (frozen-core
   environments at random init may be poorly scaled at this depth);
3. normalizer error-mass mechanism (Section: Mechanism hypothesis of
   the attempt02 note) — its holdout-mass diagnostic instrument is
   still unbuilt and would now discriminate cheaply at XLA speed.
The review is analysis-first (read the diagnostics telemetry per step,
build the holdout-mass instrument), not another sweep.

## 6. Structural review result: ROW COVERAGE COLLAPSE (2026-08-19)

Instrument 1 — per-step localization
(`docs/benchmarks/run_n4_step_localization_20260819.py`,
`/tmp/n4_step_localization.log`): at n=4 the error enters at EVERY
transition step (t=0 marginal is fine: -0.003) as a per-step bias of
-1.7..-3.4 nats, roughly constant, not compounding; n=2 shows the same
shape at -0.001..-0.013. TT increments are consistently BELOW Kalman
(mass lost, not gained). Gram conditioning is benign at both n
(<= 5.7e3), and fit rms is misleadingly GOOD at n=4 — killing the
boundary-rank-growth and assembly-conditioning suspects.

Instrument 2 — row-coverage ESS at the t=1 target (self-contained
probe, log in session record): with uniform-box Sobol rows at hw 3.0,
the transition-step target (concentrated near the x_c ~ A x_p,
x_c ~ y_t manifold) has

    n=2: ESS 219 / 8192 rows  (2.7%)
    n=4: ESS 11 / 8192 rows   (0.14%)

VERDICT: the n=4 plateau is a ROW-DESIGN COVERAGE failure, not a
rank/resolution/engine defect. The uniform-box design's overlap with
the concentrated 2n-dim step target shrinks exponentially with n; at
n=4 the ALS fit determines ~208 columns from ~11 effective rows, so
the fitted h^2 misses target mass and every step's normalizer
under-counts (matching the uniform negative bias). This RETROACTIVELY
EXPLAINS all Section 3/5 arms: more rows raises ESS only linearly
(flat), rank/degree cannot add mass the rows never see (flat), and
hw 4.0 DILUTES coverage further (worse, with better in-sample rms on
the easy off-manifold bulk). The n=2 ladder passed because ESS ~200 is
marginally sufficient there.

REPAIR DIRECTION (next design artifact, before any further n>=4
compute): importance/target-adapted frozen row design — e.g. rows
drawn once from a declared proposal concentrated near the transition
manifold (prior-predictive or Kalman-proposal family), frozen by seed
(V2-compliant: still a fixed scattered design), with the mu-weights
carrying the proposal correction. This must be a reviewed design note
(it touches the declared program's row-design contract and the
measure/weight bookkeeping), not an ad-hoc patch.

Non-claims: single seed, one fixture family; ESS is a coverage
diagnostic, not a validity proof of the proposed repair.
