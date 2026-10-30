# KSC reset and protection repair experiment

Status: reviewed for execution, 2026-09-29. Owner requested a plan, thorough
review and execution. This is an experimental FP64 GPU/XLA comparison, not a
new default, canonical admission, HMC run or publication. Starting commit:
`f72cbfe3`. Literature and source anchors are in
`sqmc-ksc-reset-repair-literature-20260929.md`.

## Question and evidence contract

Does replacing the repeated two-point residual cloud and/or the final
shape-distorting cap reduce KSC likelihood and analytical score errors?
The target is the seven-component observation-mixture KSC model, theta
`(gamma, log_beta)=(1.5,0)`, variance-one transition, initial `N(0,1)`, with
prediction before the first observation. Keep all four existing ancestry
routes and both score coordinates. Pairwise state-coordinate correction has
no off-diagonal work at dimension one; this is not particle-pair correction.

Baseline: unchanged Contract-E reset, original residual design, original
dual-cap controls. Oracle: full seven-mixture quadrature, checked against a
separate density-grid recurrence and refinement; tolerance `1e-7` on value and
each score coordinate. Gaussian Kalman approximation, zero score and the
first-observation-only score form the constructed cheap adversary set. The
Gaussian comparator ignores mixture shape, zero tests whether an inaccurate
score is useful at all, and the first-only score tests the value of recursive
information. Report each fresh data case and horizon separately.

Primary repair evidence: paired changes in absolute log-likelihood error,
absolute gamma/log-beta score errors and score-vector L2 error against the
seven-mixture oracle on untouched data. A repair nomination requires lower
mean score L2 error in every tested case/horizon, no more than 10% increase in
mean absolute value error (or an absolute increase of 0.01), and no numerical
veto. A statistically supported improvement additionally requires a paired
95% interval wholly below zero for score L2 differences in every cell. These
are exploratory repair criteria, not canonical admission or default promotion.
Losing to a heuristic in any cell vetoes promotion irrespective of improvement
against the old baseline. Four methods remain retained regardless of ranking.

| Diagnostic | Role |
|---|---|
| Oracle mismatch, missing provenance, budget/deadline exhausted | Continuation veto |
| Nonfinite particles/scores, covariance invalidity, failed total-derivative check | Candidate veto; implementation failure is a repair trigger |
| Untuned scope or absent repository-issued admission artifact | Canonical/default promotion veto; diagnostics may continue |
| Healthy protection non-harm and bounded stress behavior | Safety-candidate acceptance criterion |
| Shape moments, parity clustering, cap activation, local prediction changes, transport residuals | Explanatory only |
| Paired oracle errors and uncertainty on untouched data | Primary repair criterion |

Do not infer posterior/HMC correctness, broad superiority, literal GenUT
moment matching, or exclusive causation of the previous error from this work.
Finite differences check the implemented finite scalar, not target accuracy.

## Candidates and analytical definitions

All implementations use the shared general correction and full analytical
canonical executor. No reduced scalar algorithm or lane-specific fork.
Existing defaults remain unchanged.

* R0: repeated signed coordinate-axis residual design (existing baseline).
* R1: fixed normal midpoint quantiles, centered and whitened. In dimension one
  interleave negative/positive magnitudes so both parity groups have a range
  of magnitudes. For general dimensions use fixed deterministic permutations
  of midpoint quantiles before whitening. Rank failure rejects the design.
  A reversed ordering diagnostic tests sensitivity to correspondence with the
  transported cloud. This is a coverage hypothesis, not a Gaussian posterior
  assumption. Design is theta-independent; its parameter derivative is zero.
* C0: existing smooth standardized absolute cap followed by covariance restore.
* C1: an identity-core, bounded-tail standardized cap, followed by the same
  affine restore. For `a=abs(z)`, core radius `r>0`, existing route cap `c`,
  tail width `w=r*c`, use `f(z)=z` for `a<=r`, otherwise
  `sign(z)*(r+w*tanh((a-r)/w))`. Its derivative is one in the core and
  `1-tanh((a-r)/w)^2` outside. The map is C2 at the boundary; all upstream
  standardization and downstream restore derivatives remain included.
  The pre-recolor bound is `r*(1+c)`; recoloring does NOT guarantee the same
  final support bound. No claim of a hard final particle/domain bound.

C1 is an extension, not the published GenUT constraint algorithm. Retain the
existing diagonal trust radius 0.5, LM regularization and pairwise RMS cap.
Calibrate `r` from `{2,4,8}` using fixed healthy normal, moderate skew and
mixture-shaped standardized fixtures. Select the smallest radius enclosing
all healthy fixture coordinates, with identical mapped outputs/tangents there.
Stress outliers outside the core must be finite, bounded and flagged. The
fixtures define the protected healthy region; safety beyond them is unproved.
Never select the safety radius using KSC value or score accuracy.

## Sequence, partitions and limits

1. Implement optional shared cap, fixed residual design and stage observability.
   CPU-only tests check derivatives, exact core identity, stress bound, batch
   versus general parity and unchanged default behavior. CPU is an explicit
   diagnostic exception with GPU hidden. No new package or environment.
2. Start a new immutable prior-budget link and bounded GPU supervisor. Include
   escalated device/framework probes in charged time; verify memory growth,
   FP64, TF32 off, stable signatures and XLA. N=1008 uses chunk=1008.
   Check complete finite-program scores against branch-matched finite
   differences on T=2 and T=10 before accepting candidate comparisons.
3. Stage audit at T=10 on the first fresh calibration case (241001): weighted source, raw
   reset, pre-cap correction, post-cap/pre-recolor, final output. Record
   moments and exact next-observation predictive values and directional scores.
   Run the R0/C0, R1/C0, R0/C1, R1/C1 factorial with identical random inputs.
4. Fresh calibration data seeds 241001 and 241002; validation 242001;
   untouched comparison 243001 and 243002. Generate 120 observations once per
   seed with the existing simulator; T=10 is the prefix. Particle designs:
   calibration 251001,251002; validation 252001,252002; untouched
   253001..253008. Never select controls on untouched data. Use N=1008,
   both T=10 and T=120, four ancestry routes, all coordinates.
5. On calibration only, test factorial arms, then R1/C1 with correction steps
   4 and 8, and terminal epsilon 25.6 (original 102.4). Terminal epsilon
   is distinct from solver iteration count. Retain transport residual gates;
   lack of convergence rejects an arm rather than fabricating a result.
   Record actual transport cost scale on the stage trace. Use the same
   calibration work allowance for each route. Pick by minimum mean score L2
   among valid arms within each route/horizon, with the value-error guard.
   Freeze that experimental selection before validation and untouched data.
   Validation failure is preserved and vetoes nomination; still report the
   already specified factorial repair question. No admission artifact is
   issued: every changed scope remains a named diagnostic variant.
6. Untouched comparison: baseline and frozen candidate, paired designs. Begin
   with four designs/cell; extend to eight only if measured runtime and the
   remaining allowance permit the entire balanced extension. Save actual
   values/scores, signed and absolute errors, mean vector error, replicate SD,
   component SE and paired bootstrap intervals (10,000 fixed-seed resamples).
   Conditional intervals describe design randomness for these fixed datasets,
   not a population of financial datasets. Report uncertainty of the mean
   error vector separately from SD of replicate error norms.
7. Terminal evidence review, decision/inference tables, checkpoint and master
   summary. Commit local work and evidence. No merge or push in this task.

If runtime is prohibitive, preserve balanced four-design results before adding
particles or seeds. N=4032 is not planned: changing geometry at fixed N is the
question. A candidate failure triggers the subsequent planned repair, not
abandonment of the shared research direction.

## Default/assumption audit and pre-mortem

| Choice/provenance | Why use it; failure mode; early check; status |
|---|---|
| N1008, T10/120, four routes from completed campaign | Controlled comparison; scope transfer is untuned; per-scope fresh calibration; diagnostic baseline |
| Original flow8, epsilon102.4, Sinkhorn24/balance12, correction1/.12, damping.01, floor.0001, trust.5, pairwise1/.03/cap2, coordinate.98 (.97 ablation), power8, ridge1e-5 | Preserve baseline; inherited controls can explain failure; isolate geometry then steps/epsilon; frozen baseline, not reviewed universal defaults |
| Normal quantiles and fixed ordering | Coverage at low cost; can mismatch tails/transport coupling; stage moments, reverse ordering and oracle errors; hypothesis |
| Identity-core cap/radius calibration | Avoid changing declared healthy points; empirical healthy region may be inadequate; non-harm curve and stress/activation diagnostics; safety hypothesis |
| theta=(1.5,0), synthetic seeds | Same target to isolate repair; limited regimes and parameter locality; conditional reporting and explicit non-generalization |
| FP64, TF32 off | Diagnose method error; cannot establish FP32/TF32 production performance; explicit comparison arm |
| Four/eight designs, paired bootstrap | Bounded budget; low-power intervals and selection risk; freeze before untouched seeds, report sample count; exploratory inference |

The experiment could pass a moment test while retaining wrong likelihood
scores, pass a derivative test for an inaccurate finite approximation, or
improve only selected seeds. Original-target scores, fresh partitions and
heuristic comparisons address these failure modes. A cap can improve shape
while causing covariance failure: validity remains a hard candidate veto.

## Budget, artifacts and exact environment

Prior charged use is 31,543.211557s of 43,200s, imported once from
`artifacts/sqmc-ksc-discrepancy-20260929/budget.json` with checksum. Remaining
11,656.788443s, including every new GPU probe, compile, check and failed retry.
Deadline `2026-09-30T16:15:40.010888+00:00`. Two infrastructure retries per unit.
Provisional allocations: checks/stage 1800s, calibration 4200s, validation and
untouched 4100s, repair reserve 1556.788443s. Allocations may be redistributed
within the unchanged total; record measured estimates before a larger phase.

Runner: `docs/benchmarks/run_sqmc_ksc_reset_repair.py`; execute with
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python` and escalated GPU access.
New root `docs/plans/artifacts/sqmc-ksc-reset-repair-20260929/attempt-01/`.
Supervisor records per-attempt logs, exact argv, git/source hashes, device and
growth metadata, seeds/data hashes, start/end/wall time and charged remaining
budget. No overwrite of previous evidence. Result note:
`docs/benchmarks/sqmc-ksc-reset-repair-results-20260929.md`.

## Skeptical review before implementation/execution

Reviewed baseline, target, score coordinates, scope transfer, default controls,
selection leakage, guard calibration, runtime, and interpretation on 2026-09-29.
Material issue repaired: the earlier evidence measured reset+correction+cap
together; it did not identify which operation caused shape loss. Stage traces
now separate them. A literal GenUT replacement would change weights and the
algorithm; this plan tests explicitly named shared extensions instead.
Safety calibration is independent of primary-score selection. Approximate
Gaussian Kalman is a heuristic, never the seven-mixture oracle. Numerical
validity, finite-program derivative correctness and target accuracy are separate.
Optional controls/default-preserving shared plumbing avoid lane forks.
Budget and balanced stop conditions prevent a partial convenient leaderboard.
Remaining risks: small synthetic data set, exploratory selection, narrow safety
fixtures, nonlinear recoloring and low replication. These prevent promotion,
not the authorized diagnostic experiment. Self-review passes for this scope;
no independent reviewer has inspected this new plan.
