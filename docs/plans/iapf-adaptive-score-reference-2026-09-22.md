# Adaptive R reference comparison after the floor diagnosis

Status: COMPLETE. All400 evaluations complete;729 terminal checks pass. The
floor distortion disappears by the second score fit in all d80 runs. Score with
later doubling fails the descriptive d40 heuristic screen; no statistical
ranking follows. Result: `artifacts/iapf-adaptive-score-reference-20260922-01/result.md`.
Skeptical review PASS; authorized continuation of the renewed
48 CPU-hour/48 GPU-hour campaign. Phase14 completed with119 terminal checks,
400 final evaluations and no numerical rejection. The diagonal score/tail8 arm
cleared the descriptive cheap-baseline screen; its floor perturbs the first
bootstrap-cloud fit at d80, so iterative behavior remains an open question.
Root: `docs/plans/artifacts/iapf-adaptive-score-reference-20260922-01/`.
Budget transferred:45.756654 CPU /47.745145 GPU hours. CPU-only independent R
reference phase, at most three CPU worker hours, twelve launches of <=600s each,
at most two concurrent single-threaded workers; reserve every attempt in advance.

## Question and method boundary

Does fitting on successively guided clouds remove the initial floor distortion,
and does the diagonal score extension produce reliable final likelihoods under
the paper's adaptive controller? Compare score/tail8 against frozen QR/tail8,
crossed with the two existing doubling-start hypotheses: `after_k` and
`first_full_window`. Keep N0=1000,T100,k=5,tau=.5,kappa=.5, six likelihoods in
the CV window, sample-SD convention, cap12 iterations and N<=4000. Preserve any
iteration/particle/fit rejection as failure, not a missing observation.

Source: GJL(2017) Algorithm4, local text lines670–704, and §5.1–5.2 checked in
phase14. The stopping test explicitly requires l>k; the doubling step uses the
l-k:l history but does not state how to treat early negative indices. The two
frozen local interpretations first allow doubling at l=k or at l>k. Neither
is silently selected as author identity. At l=k, `first_full_window` may double
before the first permitted stopping check, changing final particle count and
cost. Report this difference explicitly. `iapf_iterate` is the controller
under test; do not copy it into a new loop. Both fitters change Eq15, and the
score fitter uses analytic coordinate gradients. No original-paper replication,
general nonlinear applicability, TF runtime or model-score claim follows.

## Frozen design and evidence contract

One fresh observation data set at each d in5,10,20,40,80, from seed `961000+d`.
Ten complete independent learner replicates, r=1..10. Learner RNG seed
`962000+100*d+r` is reset for each of the four arms, creating paired labels
without assuming equal random-number consumption. The controller's final
filter is fresh after stopping; never report its stopping-history likelihood
as the final estimate. All fit and controller histories are retained.

Two batches per dimension, r1:5 and6:10, ten CPU launches total. Each batch
also runs the constructed cheap heuristic set for those five labels with seed
`963000+100*d+r`: BPF N10000, FA-APF N5000, current-observation guide N1000,
and full Gaussian oracle N1000. These are the paper-count classical comparators
and same-particle myopic/exact controls, respectively. Conditional comparisons
are per dimension and per doubling convention, not pooled across dimensions.
No equal-runtime claim; particle counts and end-to-end wall times are reported.

Primary numerical pass: prior reference hashes; all48 phase13 R/TF recursion
checks and target-gradient/rejection preflight pass after adding observability;
actual-controller wiring is exercised; exact oracle matches Kalman <=1e-8;
completed candidates produce finite states and log likelihoods. Harness,
reference, source or missing-artifact failure is a continuation veto followed
by localized repair. Candidate caps/rejections and heuristic losses are
promotion vetoes and triggers for the next scientific diagnosis.

Primary downstream quantities: completion proportion, final ratio Zhat/Z,
relative RMSE, ratio mean/SD, and log-error MSE. Underflowed ratios are failures
of estimation, not low-variance successes. Preserve all per-replicate records;
do not calculate a whole-arm success statistic by discarding capped replicates.
With ten replicates, continuous rankings are descriptive only. No superiority
or Table1 replication claim; larger fixed-data replication is the next evidence
needed if these arms remain viable. Report machine-readable conditional
heuristic losses; exact-oracle gaps remain visible and block default promotion.

Explanatory fitting diagnostics: target Gaussian responsibility min/mean at each
time, cloud/precision margins, score residual, and maximum coefficient difference
between the actual tail8 fit and a separate negligible-floor fit on the identical
cloud. Record every score fitting call. Comparing the first and last call tests
the proposed cloud mechanism without tuning on final likelihoods. Final proposal
floor probabilities alone cannot establish negligible fitting distortion.
The negligible-floor refit is diagnostic only and never replaces the candidate.

## Assumptions and skeptical review

| Choice | Provenance/status | Justification and failure mode | Early check |
|---|---|---|---|
| Paper model/timing | Published §5.2 baseline | Removes earlier local mismatch | Exact APF oracle/Kalman equality |
| score vs QR | Explicit reconstructed methods | Tests new objective while preserving old comparator | Frozen source hashes and old parity fixture |
| tail8 exponent8 | Existing hypothesis, not author default | Gaussian-contour derivation; may distort remote clouds | Per-fit target probabilities and no-floor coefficient difference |
| two doubling conventions | Existing local hypotheses | Early-index ambiguity; materially different N/cost | Actual count/history and convention recorded |
| k5,tau.5,ESS.5,sampleSD | Paper settings plus explicit SD hypothesis | CV can stop noisy or biased-looking finite samples | Fresh final run and exact likelihood errors |
| Ncap4000,iterationcap12 | Bounded diagnostic convenience | May cap valid but slower adaptation | Keep and classify every cap; no selective retries |
| Ten replicates/one data set per d | Bounded reference ladder | Conditional stochastic evidence only | No population or paper-table promotion |
| No ridge/clipping | Derived score regression | Exact full-rank solution; invalid precision must reject | Relative guards, no replacement with oracle |
| R CPU exception | User-authorized independent reference | Helpful debugger, not production backend | GPUs hidden, pinned BLAS, run provenance |

Review PASS: the controller is reused through its real consumer endpoint; the
two consequential early-doubling defaults are tested, not hidden. The positive
floor remains fixed after phase14. New diagnostics do not change the fit.
Fresh final likelihoods and strong cheap baselines address false stopping
success. Caps consume the campaign budget and preserve failed candidates.
Full covariance is only the exact oracle here; no extension is mislabeled Eq15.
No nontrivial unexamined default remains beyond the documented source gaps.

## Execution and stop/repair rules

Add observability to the new diagnostic R fitter only, leaving the three frozen
references untouched. Run preflight, then the ten bounded case batches. Snapshot
code/plan, git commit, R version, environment, seeds, data and complete logs for
every launch. One spare launch slot remains for a localized harness repair, not
new scientific arms. A numerical veto stops pending launches; candidate
rejection is recorded and the next predeclared arm runs. Renewed owner approval
is unnecessary for this unchanged-scope work under the remaining campaign.

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_adaptive_score_reference.py --mode preflight
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_adaptive_score_reference.py --mode cases
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_adaptive_score_reference.py --mode results
```

Pre-mortem: a controller convention can appear more accurate only because it
doubles N; a successful stopping CV can coexist with poor final accuracy; a
tiny final floor probability can hide earlier fitting distortion. Counts, fresh
final errors and per-fit responsibility diagnostics separate these explanations.
Terminal review will state viable candidates, hard vetoes, descriptive-only
differences, default status and the exact next evidence required. If the
extension survives, scale conditional replication before pursuing a new default;
if it fails, use the saved histories to discriminate fitting from stopping.
