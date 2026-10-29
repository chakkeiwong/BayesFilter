# Controlled-SMC downstream comparison, 9 October 2026

**Planning-error correction, 11 October 2026:** the fixed 1% likelihood and
5% variance margins used below were wrong planning for controlled SMC: the
rule was set before verifying source tail/moment stability and achievable
interval precision. The frozen Q2 computations and original verdict remain
historical decisions under that rule; they are neither a current-rule port
failure nor a retroactive pass. See the
[author-relative acceptance reset](controlled-smc-author-relative-acceptance-20261011.md).

The final 13,120-pair qualification finished at 23:05 HKT on 9 October.
Execution and integrity checks passed; the frozen statistical verdict is
**NOT_QUALIFIED**. Zero additional TF failures were observed, but only five of
41 cells meet both adequacy and numerical-margin criteria. See
[the terminal result](#completed-untouched-run). Earlier sections retain the
pilot and calibration history.

## Historical pilot

The complete 96-pair pilot has finished. There are **no additional TensorFlow failures**: source and port each fail only seed 261009010 at F=2.5. This is development/calibration evidence, not untouched qualification. Exact source-output and final-RNG controls pass on the first trial of all three strata; all 95 source-valid initial APF and all controller stopping checks pass.

Plan: [downstream qualification and conditional repair](controlled-smc-downstream-qualification-20261009.md). Evidence: [completed pilot](artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-165753/summary.json).

| Forcing | Valid pairs / attempted | Mean likelihood difference | Var(log Z) ratio, TF/source | 95% upper extra-failure probability |
|---|---:|---:|---:|---:|
| 2.5 | 31/32 | +3.590% | 0.0998 | 9.21% |
| 4.8801 | 32/32 | -0.156% | 0.0992 | 8.94% |
| 8.5 | 32/32 | +0.264% | 1.0516 | 8.94% |

Continuous low-forcing results condition on both implementations being valid (31 pairs); the other strata use all 32. The matched failure is retained, not imputed or counted as a TF translation defect. The source reports a covariance eigenvalue of -8.19981e-5; TF rejects refit three. This demonstrates matched failure on this fresh seed, not identical failing coefficients or proof for all source failures.

The +3.59% low-forcing estimate is not evidence of equivalence and is not yet a demonstrated population offset: its paired influence standard error is 0.02855 in log-ratio units, and source log-likelihood kurtosis is 15.92. The central and high estimates are descriptive as well. The high-forcing sample variance ratio is 1.0516; the log-ratio standard error is 0.2834, so the accepted 1.05 upper-confidence-limit criterion is unresolved. Lower observed variance in the other strata does not establish statistical superiority.

Source-only equal-population, independent-arm 90%-power approximations give approximately 107353, 20275 and 23966 pairs per stratum, driven by variance precision. Those approximations are unstable with 32 runs and can be conservative under pairing or substantially lower candidate variance. They are a warning about precision, not a revised acceptance rule or a reason to stop replication. Next is the prescribed source-only full-grid calibration and a bounded fixed qualification design; an unaffordable power target will be reported explicitly.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue calibration | Untouched qualification not run | No extra pilot failures; source/RNG/APF/stopping controls pass | Tail moments, full-grid scope and attainable CI width | Source-only original 40-grid calibration in isolated chunks, then freeze final design | Full cSMC completion |
| Retain centered-QR candidate | Population mean/variance still unqualified | No demonstrated additional filter failure | Low-forcing location and high-forcing variance | Diagnose only after a supported failure or failed qualification | Bitwise source equivalence |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Two harness-only failures repaired; no extra TF filter failure in completed pilot |
| Statistically supported ranking | None; Q1 is calibration only |
| Descriptive-only differences | Mean likelihood and variance ratios above |
| Default readiness | Not evaluated; FP64 reference-precision GPU candidate |
| Next evidence needed | Full declared grid, frozen inference/sample sizes/seeds and untouched paired execution |

The actual consumer is LorenzRefinement.run, with native TensorFlow time/RK4/backward-refinement loops and one explicit stable signature per forcing. GPU placement and memory growth pass; TF32 is off. There is one trace per forcing and peak TF live allocator usage is 103507712 bytes. No NumPy computation enters the TF worker. Python loops schedule independent repetitions and file I/O only.

Execution and repairs:

- Attempt 163409: Octave lacks jsonencode; source-output serialization failed after 5.460939 CPU seconds, zero GPU. Repaired with the existing independent-reference MAT decoder.
- Attempt 163637: all source runs and ten TF runs completed; saving the matched failed trajectory exposed CPU-only SerializeTensor under strict GPU placement. Charges 193.443289 CPU / 72.169415 GPU seconds. Explicit CPU artifact placement passes a focused regression.
- Attempt 165753: completed 96/96 pairs, 190.148364 CPU / 432.827912 GPU seconds. Frozen algorithm hashes are identical across retries. Source output/seed order remained unchanged; failed attempts are not extra independent samples.

Campaign balance after these attempts: {"cpu": 242522.23250880867, "gpu": 159444.7585781109} seconds. Focused engineering checks will be charged separately with the phase close.

Post-run red-team: the strongest alternative explanation for the low-forcing offset is a rare source fitting tail rather than systematic TF bias. The weakest evidence is the small tail sample and missing interior grid points. Fresh full-grid source calibration and untouched paired CIs could overturn the pilot impression. Successful controls validate the harness, not the statistical claim.


## Full-grid calibration and Q2 execution readiness

Source-only calibration completed all 41 cells, 1,312 runs, with 1,309 valid
outcomes. Two intrinsic failures occurred at F=2.5 and one at F=7.423076923076923.
All 41 pristine-output/final-RNG controls passed. CPU charge: 2,074.901118
seconds; zero GPU. Evidence: csmc-qualification-20261009-grid-185030/summary.json.
Independent-arm variance sample-size approximations span 9,685–198,508/cell;
they are unstable planning diagnostics, not demands to spend that compute.
Heavy source tails and a narrow 5% variance margin explain the precision risk.

The reviewed Q2 design retains every grid cell and freezes 320 independent
pairs per cell, 13,120 total, with 32-run source processes and a GPU consumer
per cell. It is a bounded full-grid comparison, without a promise of
sufficient precision. The initial draft seed namespace overlapped source
calibration; review caught and corrected it before execution. The final
namespace starts at 271100001 and is checked against all calibration seeds.
CPU/XLA inference controls passed identity, known likelihood and variance
shifts, shared-normalization invariance, paired sampling, zero variance,
tail veto and stable trace checks. No NumPy enters inference.

One Q1b prelaunch command failed because an Octave newline was escaped
incorrectly; it launched no research worker. Space-separated grid export
repaired the command and exact 40-point endpoints were checked. Engineering
checks, including this prelaunch failure, have a reserved charge of 300 CPU /
60 GPU seconds at phase close. The numerical candidate remains unchanged.

The new source checks inspect every returned stage for real/finite fields
and valid ESS; native covariance sampling remains the source authority.
The TF worker now saves all computed result diagnostics on failure, and
terminates qualification on an author-valid/TF-invalid pair. This changes
observability and execution control, not filter arithmetic.

Executable pre-launch review completed: PASS. CPU/XLA estimator controls,
exact pristine source/RNG plus known-valid/known-failed outcome regression,
GPU/XLA parity and complete failure-diagnostic serialization, one trace per
forcing, verified memory growth, AST checks and live process-group cancellation
all passed. Focused source runtime was 7.747 s; GPU regression 19.650 s; cleanup
1.003 s; estimator checks 2.599 s. A conservative 300 CPU / 60 GPU second
engineering allowance is now charged, including prior short checks. Fresh
seeds and every frozen source/candidate hash are checked again before launch.
The active allocation, rather than the larger historical balance, is now
reported by the master command.


## Completed untouched run

Run: [frozen specification](artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-final-013640/qualification-spec.json).
The supervisor completed at 2026-10-09 23:05 HKT. The run attempted all
13,120 pairs across 41 cells and charged 19,262.993 CPU seconds and
41,127.248 GPU seconds. The frozen source and candidate snapshots, plan hash,
data hash, row hashes, and runtime controls passed the terminal review:
[terminal-review-20261009.json](artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-final-013640/terminal-review-20261009.json).

The source produced 13,106 valid outcomes and 14 intrinsic failures. The TF
candidate produced 13,115 valid outcomes and five rejections; all five were on
source-failed pairs, so there were **zero additional TF failures on an
author-valid pair**. All 410 source pristine controls, APF/stopping controls,
GPU memory-growth checks, FP64/XLA settings, and one-trace checks passed.
The implementation and harness therefore survived the engineering and failure
screen.

The frozen statistical decision is **NOT_QUALIFIED** for the required
intersection-union claim. Only 5 of 41 cells meet both interval-adequacy and
margin criteria. The 1% likelihood interval lies inside its margin in 39 of the 40 author-grid
cells (the F=2.5 interval is wider), and only 23 cells pass likelihood
adequacy. Only 7 cells pass variance adequacy and 5 also pass the 5% upper
variance-ratio margin. The generating-parameter cell F=4.8801 passes both
pointwise criteria (mean ratio exp(estimate)=1.00105, 90% mean-likelihood ratio interval
[1.00039, 1.00172], variance ratio 0.10776, one-sided 95% upper bound 0.13030).
The full-grid claim still fails because every fixed cell must pass.

No cell establishes a mean discrepancy beyond the accepted 1% margin or
variance degradation beyond the accepted 5% margin under the frozen adequacy
checks. No additional-failure repair is triggered. The main limitation is statistical
precision and tail adequacy, not a demonstrated extra candidate failure:
18 cells fail the likelihood influence-share adequacy check, 34 fail the
variance influence-share check, and 13 fail the residual-effective-count
check. These vetoes make the corresponding intervals inconclusive; they do
not establish equivalence.

| Decision | Primary criterion | Veto / control status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Frozen controlled-SMC replication | NOT_QUALIFIED: 5/41 pointwise cells pass both; global criterion is not established | Zero extra TF failure; all harness controls pass | Heavy-tailed log-likelihood and variance precision | Execute the already planned precision/budget assessment before designing further sampling | No established numerical degradation, superiority, or full replication |
| Q3 conditional follow-up | Precision/budget assessment triggered; numerical repair not yet implicated | No author-valid/TF-invalid pair and no supported out-of-margin discrepancy | Tail influence and attainable variance precision | Assess unresolved cells using preserved final diagnostics and remaining phase budget | A solver/controller repair has not been justified |
| Engineering integrity | PASS | Hashes, pairing, seeds, controls, runtime and trace checks pass | None found | Preserve artifacts and close Q2 | No bitwise coefficient claim |

| Inference status | Verdict |
|---|---|
| Hard veto screen | PASS for source/RNG/APF/stopping/runtime controls; intrinsic source failures retained |
| Statistically supported ranking | None |
| Descriptive differences | Cell-level estimates and ratios are descriptive unless adequacy and the stated interval criterion pass |
| Default readiness | Not evaluated |
| Replication status | Controlled-SMC remains incomplete under the agreed no-worse acceptance rule; KDM/IWSG remains separate and incomplete |
| Next evidence | Q3 precision/budget assessment, followed by a frozen fresh sample if feasible within the existing scope and budget |

At the six cells with intrinsic source failures, continuous estimates condition
on paired valid outcomes. The run does not establish an unconditional
likelihood claim there. Per-cell exact 95% upper bounds for an additional TF
failure are approximately 0.93–0.96%; zero observed extra failures is not proof
of zero population failure probability.

Terminal red-team review: passing controls establishes the recorded execution,
not population equivalence. Rare solver-sensitive outcomes could hide either
degradation or a source tail that makes TF look favorable; the inadequate
variance intervals cannot discriminate these explanations. Adequate fresh
intervals outside the margins would justify localization; intervals entirely
inside the margins across every declared cell would support qualification.
The weakest evidence is tail-moment precision. Preserve the frozen Q2 result
and thresholds; further sampling requires a prospectively specified design.

First complete-cell monitoring review, 2026-10-09 02:14 UTC: all 320 pairs at F=2.5 completed. The source has 312 valid outcomes and eight intrinsic failures; TF has 315 valid outcomes and five rejections, all on source-failed seeds. There is no additional TF failure and all applicable APF/stopping controls pass. The GPU/XLA controller traced exactly once; verified memory growth was enabled, with allocator peak 103,725,312 bytes. All frozen code/source hashes and the plan hash still match. The supervisor automatically advanced to the next forcing cell. These are failure/control counts, not a likelihood or variance acceptance decision. No interim confidence interval was evaluated. [Monitoring record](artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-final-013640/monitor-20261009T021407Z.json).
