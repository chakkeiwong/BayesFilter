# R4d finite-solver repair and unchanged paper-horizon check

28 September 2026. **Source-semantic controls PASS; repaired small filter
regression PASS; strict paper-horizon coefficient comparison still FAIL.
Both full scientific replications remain INCOMPLETE.**

The repaired shared fitter now backtracks from a nonfinite trial function
value, as R does, rather than aborting immediately. Its inverse Hessian is
reconstructed from the updated lower triangle, matching R's symmetric storage
meaning. The optimizer loop is shared by deterministic tests, isolated fits,
backward passes and the adaptive filter. No objective, maxit=100, tolerance,
seed, regularizer or damping parameter was retuned to obtain a pass.

## What actually ran

All commands, environments, source snapshots, seeds, timeouts, wall times and
output hashes are in each attempt's manifest. The pre-run contract is
[r4d-finite-semantics-spec.md](r4d-finite-semantics-spec.md). GPU runs used
escalated CUDA access, verified memory growth, FP64, TF32 off and recorded XLA
status. CPU runs explicitly hid GPU devices. Graph-only runs are diagnostic
exceptions; they are not replacements for the GPU/XLA route.

| Attempt | Question | Result |
| --- | --- | --- |
| 29 | Exact R reference for seven deterministic cases | Exported; installed R version recorded |
| 30 | Shared solver CPU semantics | 37 checks PASS |
| 31 | Shared solver GPU graph semantics | 37 checks PASS |
| 32 | Shared solver GPU/XLA semantics | 37 checks PASS |
| 33 | Actual small backward/controller consumer | FAILED before numerical execution: new diagnostic state keys absent from loop initializer |
| 34 | GPU/XLA semantics after local state/counter repair | 43 checks PASS |
| 35 | Actual repaired small CPU consumer | 129 checks PASS |
| 36 | Actual repaired small GPU/XLA consumer | 129 checks PASS |
| 37 | Unchanged full-precision d=5, T=100 fixed-cloud paper comparison | 92/95 checks PASS; three backward-2 coefficient checks FAIL |

Attempt 33 is a local implementation regression introduced in this repair,
preserved and charged. Updating the empty loop-state structure repaired it.
Inspection also found a diagnostic counter could report minus one rejected
trial on a no-change exit. Excluding a stale acceptance from that counter
does not change the numerical trajectory; attempt 34 tests the correction.
The original 37 checks concern solver behavior; the six added checks verify
nonnegative rejection counts. No result from an earlier source snapshot is
silently attributed to the later source.

The controls cover an analytic quadratic, an exponential objective with
nonfinite rejected trials, Rosenbrock, a stationary objective, floating-point
no-change, negative curvature/restarts, and an invalid initial objective.
R and TF agree on declared endpoints, termination and logical function/gradient
counts. TF computes trial loss and derivative jointly; logical gradient counts
mean the reference's accepted-gradient cadence, not a claim about identical
physical evaluation cost. Invalid initial values are rejected. The negative
curvature control demonstrates restarts; the overflow control demonstrates
recovery. These are finite-program checks, not convergence or optimization
quality certification.

## What remains different

The unchanged paper-horizon comparison still fails:

| Quantity at backward pass 2 | Maximum absolute difference |
| --- | ---: |
| Gaussian centers | 0.001754728174329756 |
| Gaussian covariances | 0.0017694376137442624 |
| Log floors | 0.000720729853076918 |

The prior 1e-5 absolute-plus-relative criterion is retained. The other five
backward passes and the eight isolated fitting controls pass. Five compiled
signatures each trace once. Attempt 37 took 146.290708 charged GPU seconds;
its internal diagnostic wall time was 140.178950 seconds, host peak
1,351,948 KiB, and TensorFlow device allocator peak 8,463,104 bytes.
The strict harness withheld its subsequent adaptive run after these failures.
The earlier successful full adaptive run in attempt 28 remains evidence for its
own R4c source snapshot; it does not validate the repaired full-horizon consumer.

Using exactly the saved R initializer does not resolve the sensitive isolated
time-54 fit:

| Backend | Endpoint error with local initializer | Endpoint error with exact R initializer |
| --- | ---: | ---: |
| CPU graph | 0.00177120 | 0.000184638 |
| GPU graph | 0.000673514 | 0.00295567 |
| GPU XLA | 0.00483055 | 0.00314469 |

All these fits are finite and reach the declared 100-iteration limit. The
accepted-iterate discrepancy first exceeds 1e-8 around iteration 83 or 84,
depending on the arm. The saved JSON key is historically named
first_iteration_error_above_1e8; its implemented threshold is **1e-8**, as shown
in the snapshotted source. These magnitudes are descriptive numerical
comparisons; they do not rank backends. Initialization differences alone and
XLA alone are ruled out as complete explanations.

The R4c same-backend perturbation experiment remains relevant: changes near
1e-14 in the log targets moved fitted coefficients by about 0.003.
The R4d source repairs address real finite-program differences but do not
remove endpoint sensitivity. The individual primitive/reduction responsible
for the initial arithmetic discrepancy is not isolated. Matching the sampled
edge cases does not prove general R vmmin equivalence.

## Scientific interpretation and the next question

The claimed finite target is the frozen R/BFGS reconstruction on saved inputs.
The TF computation follows the repaired finite structure on the tested
controls but does not meet the paper-horizon numerical criterion. It is wrong
to call that strict port validated. Neither result identifies the original
authors' exact fitting implementation.

The paper's Proposition 1 gives likelihood-target invariance for bounded,
continuous positive twists; Equations (5)–(6) cancel the twist factors in the
path density. A finite positive-covariance Gaussian plus a positive floor has
that functional form. Section 5.1, Equations (15)–(16), describes regularized
fitting and the positive floor. Thus, subject to correctly implemented proposal
densities and importance weights, coefficient equality is not the theorem's
condition. It tests this finite port, while scientific replication also needs
variance, cost and downstream filtering evidence. This distinction does not
waive the failed port criterion or establish harmlessness of the numerical
differences. Source: [publisher paper](../../../../.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf),
technical text at publisher-reading.txt lines276–324 and905–957.

Next work must define a **separate** paired downstream equivalence protocol:
freeze the recipe and numerical/error criteria using calibration data, then
evaluate untouched datasets and random streams against R and exact Kalman
likelihoods, with BPF/fully adapted comparators, uncertainty and full adaptation
cost. Coefficient/loss closeness cannot substitute for that evidence. No
scientifically justified numerical-equivalence margin is validated yet. An
identified source-semantic defect should still be repaired; another arbitrary
retry of the same 1e-5 test would not answer a new question.

The original article's executable initializer, coordinates, regularizer,
floor controls and seeds remain unrecovered. The thesis supports a finite
unpenalized reconstruction family, not proof of the article's exact numerical
recipe. More compute cannot identify undocumented choices. Recover settings
or continue explicitly as reconstruction; do not interpret this mismatch as
evidence of a fundamental defect in the paper.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept local source-semantic repairs in reconstruction | Targeted finite controls pass; actual small consumer passes | Initial invalid states rejected; state/counter regressions repaired | General finite arithmetic equivalence not proved | Preserve source snapshots and controls | Original author recipe recovered |
| Reject strict longer-horizon coefficient parity | Three unchanged checks fail | Numerical parity veto supported | Downstream effect across fresh data | Predeclare paired downstream protocol; retain FAIL | iAPF is invalid or XLA is sole cause |
| Keep both replications incomplete | Full empirical protocols absent | Source/configuration and statistical evidence gaps remain | Missing settings and total costs | Follow refreshed master R3b/R3c/R4e/R5 | Paper results replicated |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Strict paper-horizon parity fails; invalid initial control rejected; no accepted nonfinite fit in retained tested cases |
| Statistically supported ranking | None |
| Descriptive-only differences | Backend coefficient errors; previous single-seed logZ closeness; timings |
| Default-readiness | Not established by these reconstruction diagnostics |
| Next evidence needed | Predeclared paired downstream uncertainty analysis and full scoped empirical comparisons |

Post-run red team: the strongest alternative explanation is an untested
finite-source arithmetic/semantic discrepancy rather than intrinsic conditioning
alone. A repair closing the unchanged full gate would overturn the claim that
another equivalence protocol is necessary for this port. The weakest evidence
is transfer from one difficult cloud and one actual long adaptive run to other
datasets/dimensions. No such transfer is promoted.

## Budget and master state

Attempts 29–37 consumed 29.178210 CPU seconds and 218.004948 GPU seconds, within the 240 CPU/1800 GPU-second R4d ceilings.
Campaign totals: 1046.833064 CPU seconds and 42748.699649 GPU seconds.
Remaining: 753.166936 CPU seconds and 36.375361 GPU hours.
37/38 launch slots accounted: 26 COMPLETE, 11 FAILED across the campaign. No live job.
The next implementation tranche may be specified under the remaining authorized
budget; the unused slot is not authority to repeat an unchanged failed test.

The source-semantic repair tranche is complete as a bounded diagnostic.
Unchanged full parity is failed, not silently upgraded to PASS. No job is
running. Remaining phases require their declared implementations/contracts.
See [review](r4d-review.md), [live saved status](status-after-r4d.json), and
[master](../../iapf-kdm-replication-master-program-2026-09-24.md).
