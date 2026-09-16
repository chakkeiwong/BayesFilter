# A09: improve the standalone SGQF-initialized pair TT

Status: complete. See the [result](observation-aware-tt-warm-improvement-20260916-result.md)
and artifacts/observation-tt-warm-improvement-20260916-01/result-review.md.
The executed source/plan snapshots are retained in each attempt manifest.
Scalar improvements passed the exploratory comparison; four-dimensional
guide validity and tail coverage require a separate amendment before execution.

Owner instruction: create a plan for the proposed regression improvements,
review it thoroughly, and execute it. This amends the
[master](observation-aware-tt-repair-complete-program-20260913.md) within H11's
existing budget. Active start: 2026-09-16 04:05:18 UTC; commit
`509871fd28e21862250e8f094e8d466b004af7e3`, branch `surrogate-hmc`.

## Question and boundaries

Can higher polynomial degree, more regression rows, or regularization toward
the SGQF initializer improve a standalone pair TT used as the conditional
proposal in the exact-weight particle filter? A08's hybrid selected generic
TT, SGQF-initialized TT or analytic SGQF at each time. Its repaired-pair
comparator was generic-start. Neither establishes standalone warm-start quality.

At the initial observation use the same SGQF Gaussian for every warm-TT arm.
At each later observation fit and consume a TT, carrying that arm's own
retained TT into its next target. No TT-versus-Gaussian selector is allowed.
Select controls, including L1, offline by dimension and freeze them before
confirmation. Sampling and importance weights must use the same defended q.

Classification: `extension_or_invention` regression diagnostic, not a claim
of Zhao-Cui source faithfulness. The unchanged exact-weight particle consumer
is the downstream computation. The dense Gaussian coefficient initializer is
bounded to d<=4; this is not a scalable production algorithm. Fixed controls
remove one discrete selection, but L1 active sets, SVD degeneracies,
guide-level selection and resampling still prevent a claim of a globally
smooth filter. No total analytical gradient or HMC claim is planned.

## Mathematical variants

For a fixed core with weighted design D, weighted target b and scaled SGQF
initial core c0, minimize

    F(c) = ||D c - b||^2/(2 N) + lambda ||c - a||_1.

Baseline a=0; preservation hypothesis a=c0. The update is

    c_next = a + soft_threshold(c - (G c - r)/L - a, lambda/L),
    G = D' D/N, r = D' b/N, L = largest eigenvalue of G.

Anchors stay fixed throughout a fit. Both penalties use the same scaled
TT-SVD gauge, with no in-fit gauge changes. This penalty is gauge dependent
and does not assert closeness of densities. Evaluate density and filtering
effects directly. Lambda=0 must reproduce the existing solver. A large lambda
must preserve a known one-core anchor. Check objective descent and the shifted
KKT subgradient at c-a rather than c.

## Stages and controls

1. **Implementation/mechanics.** Add optional initial-centered L1 without
   changing defaults. Build a diagnostic standalone driver using the existing
   TF fitter, projection recurrence, pair sampler and exact-weight consumer.
   Verify the call chain with focused tests and GPU/XLA smoke. Fix rank 3,
   four sweeps, 128 proximal steps and defensive mass 1e-5. No rank/solver search.
2. **Offline calibration.** Three fresh T=20 sequences per d in {1,4}.
   Run all degree {3,4}, training rows {1024,4096}, zero-centered L1
   {0,1e-5,1e-3} combinations, each with its own recursion. Disjoint 4096-row
   validation panels diagnose regression. Four N=512 particle replicates and
   independent references evaluate filtering. Choose baseline L1 among the
   degree-3/1024-row arms and the capacity/sample nominee among all twelve.
   Select by mean sequence-normalized filtering MSE, deterministic ties by
   listed order. Heuristics and audit panels cannot select controls.
   Use the same guide-valid sequences for every arm, with at least two per
   dimension. A candidate missing one of those sequences is ineligible;
   a failed independent reference stops scientific selection. Do not reward
   a candidate for failing a difficult sequence.
3. **Initializer preservation.** At the selected degree/row size, run
   initial-centered L1 {1e-5,1e-3}. The lambda=0 member is shared with the
   zero-centered grid. Select among those three using the same calibration
   MSE. Report duplicate nominees; do not force a more complex winner.
4. **Controlled regression.** Hold the selected baseline's incoming retained
   density fixed at times 1,10,19 of every calibration sequence. Refit the
   three nominees to this identical target with common random streams and
   evaluate on 8192 fresh rows. Store initial/fitted amplitude RMS, defended
   Hellinger discrepancy, row ESS, target-weight ESS/max weight, conversion
   truncation/compression, KKT, conditioning and time. These diagnostics cannot
   revise frozen choices or establish filtering superiority.
5. **Fresh confirmation.** Freeze/hash choices before generating twelve new
   T=20 sequences per dimension. Run the three warm nominees, generic-start
   repaired pair TT, and four constructed cheap adversaries: transition
   (uses dynamics), stationary prior (ignores dynamics), SGQF marginal (uses
   observations), SGQF joint (also uses dependence). Four N=512 replicates.
   Report proposal-build time, particle time and amortized total separately.
   Duplicate nominees may share computations, with aliases recorded. Their
   timing is the underlying computation's timing, never zero. Timing includes
   compilation/setup and is descriptive, not a speed ranking.

Historical revision-1 data seeds (superseded by revision 2 below):
18626000 + 10000*partition + 100*d + sequence; calibration
partition=0, confirmation=1. Fit seed=data seed+2000000; particle seed=data
seed+3000000+10*replicate; reference seed=data seed+4000000. Panel offsets
0/100000/200000 separate train/validation/audit. Share streams across arms;
different row counts need not be nested because mixture allocation can change.
Smoke uses separate seed 929999. Record actual designs and seed formulas.

## Research intent and evidence contract

| Role | Decision |
|---|---|
| Question | Does a revised standalone warm TT improve exact-weight filtering over the standalone degree-3/1024-row warm TT? |
| Failure hypothesis | Polynomial truncation, noisy weighted regression, shrinkage away from the initializer, or solver conditioning; recursive errors can accumulate. |
| Primary improvement criterion | Fresh sequence-level paired normalized filtering-mean MSE, candidate minus baseline. A simultaneous 95% bootstrap upper bound <0 supports improvement for that dimension/candidate contrast only. |
| Promotion veto | Nonfinite output, mismatched sampling/density, failed reference, missing guide/candidate coverage, or evidence disagreement under the inherited screen. Conditional heuristic losses veto default claims, not measurement or TT repair. |
| Continuation veto | Wrong target/weight calculation, corrupted/overwritten evidence, absent required diagnostics, exhausted budget, or unavailable trusted execution. |
| Repair trigger | Candidate fitting failure, high residual/KKT, poor ESS or guide failure. Preserve it and continue unaffected comparisons; no silent covariance clipping or replacement sequences. |
| Explanatory only | Fit loss including loss to SGQF, truncation/compression, actual pre-resampling ESS, weight concentration, resampling, conditioning and runtime. |
| Not concluded | Default readiness, scalability, superiority from calibration, total analytical derivatives, raw TT normalizer/moment validity, or source faithfulness. |

Divide squared mean errors by coordinate stationary variances. Independent
sequences are the inferential units, not time points or particle replicates.
Use 9999 paired sequence bootstrap resamples, seed 927777, with maximum
standardized absolute deviation across four primary contrasts (two nominees
versus baseline, two dimensions). Zero-SE and incomplete contrasts are
ineligible for ranking. The n=12 bootstrap is exploratory uncertainty evidence,
not a guarantee of nominal coverage.
Resample independently across dimensions and jointly across arms within a
dimension. An identical nominee has an exact zero difference but supplies no
evidence of improvement.

Report all/near-zero/ordinary/large regimes using inherited |y|/beta thresholds
<=.5, (.5,2), >=2. Heuristic comparisons are descriptive falsification, not
tuning objectives. Enumerate missing sequences and matched denominators.
Conditional intervals need eight sequences and 24 coordinate-times; otherwise
report descriptive values only. No unconditional success claim after guide loss.

References reuse A08's independent scalar physical grids (801/1201 nodes,
resolution agreement <=1e-6) and d=4 particle references (four seeds, 32768
then 65536 then optionally 131072 particles), with their existing resolution
screen. References must pass before ranking. The inherited evidence screen
is |bias| <= .15 + 3.182446 * combined candidate/reference MCSE. This is an
exploratory agreement screen, not simultaneous confidence. Report failed arms.

## Default/assumption audit

| Choice/status | Provenance/rationale | Failure and early diagnostic |
|---|---|---|
| Degree 3/rank 3 baseline; degree 4 hypothesis | A04 conversion was dominated by truncation; hold rank fixed | Overfitting; compare row ladder and fixed-target audit. No degree-5 expansion. |
| 1024/4096 train, 4096 validation | Existing design plus fourfold ladder | Missed tails; emit row and target-weight ESS/max weight. |
| SGQF/product mixture, product weight .2 | Existing exact rho/proposal weighting, weights <=5 | Wrong measure; retain analytic weight and normalization checks. |
| L1 {0,1e-5,1e-3} | Lane requires tuning; earlier settings are hypotheses | Gauge/units; record scale/anchors and lambda-zero parity. |
| Four sweeps/128 steps | Bounded existing solver; more iterations previously mixed | Inexact solves; emit KKT/descent/condition; no minimizer claim. |
| Defensive mass 1e-5 | A05 safeguard, fixed across warm arms | Tail mismatch; record CDF, ESS/max weight and exact correction. |
| Highest valid SGQF levels 2--5 | Existing frozen guide | Invalid signed covariance; record diagnostics/failures, no ridge. Guide repair stays pending. |
| d=1,4; T=20; N=512; four replicates | A08 scope on fresh data | Cannot establish high-dimensional usefulness. |
| Dense TF Gaussian projection | Checked A04 recurrence/TT-SVD | Exponential storage; degree-4/d=4 has 390625 coefficients; diagnostic only. |

Pre-mortem: calibration can reward luck; fresh sequences/paired uncertainty
address that. Importance correction can hide poor raw TT quality; report both
without transferring validity. Poor results can reflect guide or solver failure;
keep these explanations separate from evidence against TT.

## Execution and budget

Ceiling: six active hours (21600 s) including engineering/review/closeout,
within starting conservative H11 balance 63678.892609 s. Numerical allowance
14400 s across at most four smoke, three calibration and three confirmation
attempts. Per-attempt limits: 600/5400/5400 s. No hardware/paid-compute expansion.
Run wall time is included in active time and charged once. Local infrastructure
repairs may retry in fresh directories within caps. Scientific changes need a
recorded amendment/review before launch, not a new approval token.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted RTX
5080, CUDA_VISIBLE_DEVICES=1 after probe, TF float64/XLA, verified memory
growth, TF_FORCE_GPU_ALLOW_GROWTH=true, two intra-op/one inter-op threads.
TF-SVD and reference/statistical setup are diagnostic CPU exceptions. CPU
tests explicitly hide GPUs. Manifests record hashes, commit/dirty state,
controls, seeds, environment/device, wall time and allocator peak.

Driver: `docs/benchmarks/run_observation_tt_warm_improvement.py`, arguments
`--stage smoke|calibration|confirmation`, `--output-root`,
`--wall-budget-seconds`, confirmation `--calibration-root`.
Unique attempts: `docs/benchmarks/artifacts/observation_tt_warm_improvement_20260916/`.
Review/execution notes: `docs/plans/artifacts/observation-tt-warm-improvement-20260916-01/`.
Result: `docs/plans/observation-aware-tt-warm-improvement-20260916-result.md`.
Finish with decision/inference tables, alternative explanation, master,
checkpoint and budget updates.

## Pre-execution review requirement

Revision 2, 2026-09-16 09:32 UTC, before confirmation replacement: the final
call-chain audit found that particle_filter adds t to each scalar seed. The
revision-1 spacing of one across sequences and ten across repetitions reused
streams over T=20. Calibration remains an exploratory nomination and its
choices stay frozen; confirmation-01 is descriptive only and its intervals
are ineligible. Neither is evidence of independent algorithm repetitions.
The reference helper now accepts an explicit repetition stride (historical
default unchanged). A09 uses 1000 for both particle and reference repetitions
and ten-million seed blocks per sequence, separate fit/panel/particle/reference
offsets, and fresh data partition 2. This repairs the independence contract
without changing the method, frozen settings, metrics, vetoes or attempt budget.
An exhaustive finite schedule check must pass before replacement execution.
Review: PASS for this localized harness repair; retain both earlier attempts
and report their limitation, even if replacement confirmation passes.

Revision 1, before replacement calibration: the initial seed root 926000
collided with A07. Attempt-calibration-01 is invalidated and preserved, and
none of its results may select controls. The new root 18626000 and confirmation
root 18636000 have no matches in the observation-TT plans/runners checked.
This restores the intended fresh-data contract; it does not change methods,
selection criteria or budget. See execution.md for review and charged attempt.

Before execution, record a skeptical review of baseline identity, same-target
versus recursive comparisons, holdout separation, heuristic roles, derivative
limitations, exclusions, reference/weight correctness, gauge/units, compute
bounds and continuation rules. Review is scientific, not a token ceremony.

Completed review: [plan-review.md](artifacts/observation-tt-warm-improvement-20260916-01/plan-review.md).
