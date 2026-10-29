# R4c result: finite iAPF implementation and remaining replication gaps

28 September 2026, Hong Kong. Both full scientific replications remain
**INCOMPLETE**. The fitter and adaptive controller are implemented and pass
the small reference checks. Strict parameter parity fails at the paper
horizon. The actual GPU/XLA filter completes that horizon with the same
controller counts as R. These statements refer to different evidence and
must not be collapsed into either “replicated” or “the filter cannot work.”

The governing [master](../../iapf-kdm-replication-master-program-2026-09-24.md),
[whole repair plan](../../iapf-mdpf-whole-repair-plan-20260928.md) and
[finite-port contract](finite-port-spec.md) preserve scope. Earlier
iAPF/MDPF/source/legacy-graph findings remain in [result.md](result.md).
This note covers attempts 17–28. Each attempt manifest holds its exact command,
environment, seeds, source snapshot/hashes, Git commit, wall time and output
hashes. Base checkout: surrogate-hmc,
89065bc6354801cb368d5e163ba49fd9c4372d10; relevant uncommitted sources were
snapshotted and unrelated work preserved.

## What changed

[iapf_finite_tf.py](../../../../bayesfilter/score_study/iapf_finite_tf.py)
implements the frozen joint Eq. (15) objective and analytic gradient, the declared
full-rank initializer, finite BFGS, backward Gaussian-plus-floor fits and the
adaptive controller. The actual consumer calls the shared
[iapf_paper_tf.py](../../../../bayesfilter/score_study/iapf_paper_tf.py)
value kernel; this is executed call-chain evidence, not an unused helper.

The source family is the publisher article Algorithms 1–4/Section 5.1/Eq. (15) and
the thesis Section 7.3 finite fitting discussion. The finite solver comparison
uses R 4.1.2 and the local matching optim.c::vmmin source. It is an explicit
reconstruction, not recovered author code. Covariances are diagonal fitted
twists. Unpenalized A_raw and scaled B_scaled remain distinct frozen hypotheses.
No regularizer, damping or clipped objective was silently introduced.

Repeated numerical time, particle, fitting and controller work runs through
stable-signature TensorFlow graphs/native loops, with XLA default on. Python
orchestrates a bounded number of adaptive passes and artifact writes.
There is no NumPy numerical path, pfor or per-particle Python loop.
GPU runs use FP64, TF32 off, escalated visibility and verified memory growth;
CPU reference lanes deliberately hide GPUs. FP32/TF32 equivalence is not tested.

The exporter now writes 17 significant digits and verifies exact float64
roundtrip for every matrix. A no-change BFGS line-search edge was also fixed:
when the candidate rounds unchanged but objective change remains large, R
retains the old Fmin. This localized repair is not established as the cause
of the observed 100-iteration failures.

## Executed evidence

| Attempts | Result | Meaning |
| --- | --- | --- |
|17 | R small fixture complete | Eight isolated fits and full small adaptive trace |
|18,19 | Numerical checks pass; dispatcher FAILED | Required decision.csv omitted;19 was a premature retry of the same harness defect |
|20 | CPU129/129 PASS | Reporting fixed; full small fitter/backward/controller comparisons pass |
|21 | GPU graph129/129 PASS | Diagnostic graph mode; one trace per signature |
|22 | GPU/XLA129/129 PASS | Same small scope; one trace per signature |
|23 | R paper reference complete | d=5, T=100,N0=1000; stop6, finalN=2000 |
|24 | Paper fixed-cloud backward parity FAIL | Call2 center/covariance mismatch; actual adaptive GPU consumer withheld |
|25 | Failure replay complete | Gradients agree closely; late finite trajectories diverge in GPU graph and XLA |
|26 | Full-precision R fixture complete | Same seed/model; 3,142 exact matrix roundtrips; all eight isolated controls included |
|27 | Repaired-fixture parity FAIL |95 checks; call 2 centers/covariance/floor fail; other five backward calls and all eight isolated fits pass |
|28 | Sensitivity and actual-consumer diagnostic complete | All eight actual filter calls and six backward passes complete; strict parity failure retained |

The small tests use d=2, T=8, N0=64; both R and TF stop at iteration 6 and use
N128 for the final draw. The isolated fit cases cover Gaussian, mixture,
nonconcave and paper-cloud targets with both objective scales. Objective/
gradient tolerance is1e-8*(1+abs(reference)); coefficient/backward tolerance
is 1e-5*(1+abs(reference)); classifications/counts are exact. These are
engineering checks, not 129 independent scientific repetitions.

The small CPU/GPU graph/GPU XLA runs predate the localized no-change edge
repair; their source snapshots identify that fact. The corrected version was
tested again on all eight fit controls and six paper backward traces, then
on the actual paper-horizon consumer. Small controller parity is not silently
relabelled as a new rerun of the changed source.

### Numerical failure and sensitivity

Attempt 24 maximum center/covariance discrepancies were.0002534552/.0002552702.
Attempt 25 replay at fixed R inputs found CPU/GPU graph/GPU XLA endpoint
discrepancies around 0.00004184/.0063999/.0003160 at backward call 2/time 54.
Analytic gradient discrepancies along reference accepted iterates were at
most4.1e-17. The first substantial optimizer-path departures appear late,
around iterations 82–86; the fits reach the 100-iteration limit.

A direct control found that R write.table serialized pi as 3.14159265358979
despite options(digits=17), losing about 3.1e-15. Explicit 17-digit serialization
repairs this real artifact defect. Attempt 27 still fails: maximum backward2
center/covariance/log-floor discrepancies are respectively
**.002951481398/.002978247755/.001212122163**. Fixing serialization did not
close numerical parity and did not even uniformly reduce the discrepancy.

Attempt 28 then holds the GPU/XLA backend fixed and changes only exact
time 54 log targets by a deterministic alternating zero-mean pattern:

| Log-target perturbation | Realized maximum change | Maximum coefficient change from unperturbed TF | Maximum density change / baseline maximum density |
| --- | ---: | ---: | ---: |
|+1e-14 |1.066e-14 |.00284495 |.00377842 |
|-1e-14 |1.066e-14 |.00330664 |.00439262 |
|+1e-12 |1.002e-12 |.00149543 |.00198478 |
|-1e-12 |1.002e-12 |.00310616 |.00412588 |

All remain finite and stop at 100 iterations. This establishes sensitivity
of this finite reconstruction on this case. It does not establish that every
discrepancy is floating-point error or that the final likelihood is unaffected.
Full fits/gradients/initializers are in
[sensitivity-summary.json](attempt28_iapf_finite_impact/sensitivity-summary.json)
and adjacent per-case files.

The source audit also leaves two precise questions for R4d. The TF line
search aborts at an invalid trial; R can backtrack from a nonfinite function
value. This is a behavior/domain difference to repair before general
R-solver equivalence is claimed. R stores one inverse-Hessian triangle and
reads it symmetrically; TF updates a full matrix. Possible roundoff asymmetry
needs a differential test. Neither is shown to cause the observed healthy
paper mismatch, so neither is presented as the root cause.

### Actual filtering impact

The attempted claim is full finite-parameter/path agreement with the
reconstructed R procedure. The computed GPU result is not equal under that
criterion. It is a complete, finite downstream filtering result under the
same model, entropy arrays and declared reconstruction settings.

| Quantity | R reference | Actual GPU/XLA |
| --- | ---: | ---: |
| Filter calls |8 |8 |
| Backward passes |6 |6 |
| Stopping iteration |6 |6 |
| Fresh final particles |2000 |2000 |
| Fresh final log-likelihood | -894.014159908217 | -894.0119586545842 |
| Exact Kalman log-likelihood | -894.071542920976 | Same target/oracle |

GPU-minus-R final log-likelihood is **.002201253633**; GPU-minus-Kalman is
**.059584266392**. All filter outputs are finite and all backward fits pass
their declared validity checks. Particle clouds differ substantially after
earlier fit differences; the final cloud maximum difference is about 3.0143,
while maximum final-pass prefix-likelihood difference is 0.00598925.
A close final likelihood does not erase the changed particle paths.

The exact shared consumer executed; six numerical signatures each traced
once. Attempt 28 charged 134.3284 GPU seconds including startup, sensitivity,
compilation and output work. Peak TF GPU allocator was 63,824,384 bytes;
peak host RSS was 1,454,376 KiB. These are one-run resource observations, not
a backend speed ranking. [Impact details](attempt28_iapf_finite_impact/impact.json)
and [controller trace](attempt28_iapf_finite_impact/actual-controller.json)
preserve the result.

This answers the earlier concern precisely: we can run the reconstructed
filter on the paper horizon. We have not yet shown stable equivalence across
new datasets/dimensions, unbiasedness/variance replication, or the original
authors' exact fitting behavior.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept small implementation evidence |129 checks per lane pass on recorded snapshots |No fixture numerical veto |Scope/version/finite-domain limits |Retain controls; audit source-semantic edges |Full empirical replication |
| Reject paper parameter-parity claim |Frozen tolerance fails |Call2 discrepancy persists after export repair |Finite optimizer conditioning versus remaining semantics |R4d discriminating semantic/arithmetic tests |XLA alone is broken |
| Accept downstream execution finding |Actual consumer completes, controller counts match |No finite-probability veto in this run |One dataset/entropy path only |Predeclare fresh downstream/statistical checks after repair |Harmlessness, superiority or default readiness |
| Keep MDPF replication open |One source NLL trial/screen only |Missing density/learning/repetition evidence |Unrecovered Appendix B settings and resource shortfall |R3b/R3c/R4e/R5b |A complete training replication |
| Keep both source reproductions qualified |Inspected papers/code support reconstruction scope |Exact original choices remain missing |Unknown original finite recipe/details |Recover source or retain explicit reconstruction label |A fundamental error in either paper |

| Inference-status item | Verdict |
| --- | --- |
| Hard veto screen |Paper finite-parameter parity FAIL; previous B_scaled has three amplitude failures; no nonfinite actual-filter veto in attempt 28 |
| Viable candidates |A_raw remains a finite-filter reconstruction candidate; completed B_scaled runs do not erase its endpoint failures |
| Statistically supported ranking |None |
| Descriptive-only differences |Pilot completion/likelihood summaries, single-checkpoint MDPF metrics, all timing/peak-memory and the 0.0022 final logZ difference |
| Default-readiness |Not established; no default/HMC promotion |
| Next evidence needed |Semantic differential controls, frozen downstream criterion, fresh multi-stream/dataset uncertainty, all paper cells, matched normalized heuristics and independent MDPF training seeds |

Heuristic status remains scoped: R2 constructed BPF/fully adapted/observation
twist comparators; R3a constructed persistence/motion/marginal point baselines
conditionally. R4c is an implementation diagnostic using the exact Kalman
oracle, not a quality promotion. Full conditional iAPF and normalized MDPF
heuristic gates remain open; no single-run closeness substitutes for them.

## What remains and resource feasibility

The next iAPF repair is numerical/source-semantic validation, not writing a
missing fitter. Its successful remedy is not known yet. Original code/settings
remain unrecovered; more compute alone cannot settle their identity.
Full d5/10/20/40/80,T=100,1000-repetition empirical work also remains.
MDPF still needs normalized classical-density comparators, actual Appendix B
learning, target-specific training details, TF/TFP GPU consumers, refinement
and independent repeats. Those engineering routes are specified in the
[refreshed plan](../../iapf-mdpf-whole-repair-plan-20260928.md), but not
implemented or falsely advertised as executable.

R4c attempts 17–28 consumed 166.006598 CPU seconds and 453.417178 GPU seconds.
The full campaign has 28 attempts:19 COMPLETE,9 FAILED, all preserved.
Total charged CPU/GPU seconds: 1017.654854/42530.694701. Remaining:
782.345146 CPU seconds and 36.435918 GPU hours. No process is running.
All 28 tranche slots are used; a new bounded tranche can be added within the
existing compute authorization after its actual command/specification exists.

At observed 8.589h/source training, ten more MDPF runs cost about 85.89h before
comparators/refinement. The full repeated suite is under-budgeted; lowering
the count cannot retain the same completion claim. iAPF full-suite runtime
still needs a steady-state cost ladder.

## Post-run red-team note

The strongest alternative explanation is a remaining finite-solver semantic
difference plus conditioning, not purely backend roundoff. A matched-input,
matched-initializer, step-level differential check could overturn the
roundoff-dominant interpretation. Conversely, persistent sensitivity after
semantic alignment would support a separate distributional contract, never
a retroactive parameter-tolerance change.

The weakest positive evidence is the single paper-horizon run. A fresh
multi-dataset failure, systematic likelihood error or conditional heuristic
loss would overturn the suggestion that this reconstruction is useful.
The known density/amplitude escape and unknown published regularizer remain
scientific limitations even though finite filtering now executes.

The master now reports FAIL for failed paper parity and keeps diagnostic
completion separate. Fourteen dispatcher tests pass, including a guard that
later diagnostic success cannot erase the failed parity or complete either
replication. [Regression record](regression-r4c.json).
