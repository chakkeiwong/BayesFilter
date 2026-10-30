# Controlled-SMC / KDM repair execution, 29 September 2026

Later RBF work continues in
[the current RBF result](controlled-smc-rbf-fitting-execution-result-20260929.md).
The numerical and resource totals below describe the completed stage through32.


The guided neuroscience comparison and the Lorenz solver repair are complete.
Both full replications remain INCOMPLETE; iAPF remains incomplete and deferred.
This result closes attempts20–32, not the whole master program.

## What was wrong, and what was repaired

The original controlled intermediate weights contain future-observation
information. They are wrong for the claimed ordinary filtering distribution
unless the remaining look-ahead integral is removed. The repaired reporting
consumer divides by that integral and also corrects prefix normalizers.
Normalized EKF and positive three-point quadrature proposals retain exact
importance corrections. Attempt20 passed43 analytic, numerical and replay
checks. Attempt21 rejected insufficient quadrature support before collecting
comparisons; attempt22 widened the support at unchanged resolution/tolerances
and completed all10000 rows across20 source variance cells.

The conditional comparison exposes a real limitation: in the extreme-observation
regime, controlled SMC has larger filtering-mean squared error than all four
comparators in every variance cell. Each pointwise95% interval excludes zero.
The comparators are same-count BPF, source larger-count BPF, EKF-proposal PF and
quadrature-proposal PF. Other observations give mixed comparisons; controlled SMC
improves over the same-count BPF but loses to the larger-count BPF in all20 cells.
The central regime has no observations and is unevaluated. There are2871 extreme
and129 other observations. These are100 repetitions on one source data series,
not independent data-series replications. Intervals are pointwise, not a
simultaneous claim across all contrasts. Training/adaptation and GPU cost are
not matched across arms.

For the paper's normalizer statistic, log10 relative variance of log-likelihood
estimates, controlled SMC is lower than all four comparators at every cell,
with pointwise intervals below zero. This supports the scoped likelihood result;
it does not erase conditional filtering underperformance. Candidate superiority
and default promotion fail the heuristic screen. The replication direction and
the paper's likelihood claim are not invalidated by that separate finding.

The Lorenz source quadratic design is ill conditioned in physical coordinates.
Raw Gram reciprocal condition can be about1.5e-17. On identical targets, raw
normal equations give total squared residual849.09 versus2.7012 for centered
QR. All forward source-policy checks pass, so this is a fitting numerics issue,
not an unknown fitting procedure. Centering the design alone left a coefficient
discrepancy, despite downstream agreement. Replacing the difference of two large
Gaussian log integrals by an exact integral of their quadratic correction under
the already twisted Gaussian removes that cancellation. Response centering
preserves the intercept and the same full quadratic least-squares objective.

Attempt31 found that the installed XLA GPU backend cannot compile slogdet.
The positive-definite log determinant now uses Cholesky diagonals. Attempt32
passes all28 checks, with XLA on and one trace for each fitting, regression and
forward signature. There is no pfor, runtime NumPy, scalar sample loop or Python
time loop in this consumer.

## Numerical evidence

| Check | Observed result | Interpretation |
|---|---|---|
| Batched exact Gaussian-ratio identity | Maximum error3.33e-16 | Same mathematical integral; invalid precision rejected |
| Healthy exact quadratic coefficients | Errors at most2.22e-16 | Repair does not disturb this well-conditioned control |
| Author-policy forward replay | Particle error9.77e-15; likelihood error6.97e-10; exact ancestry | Unchanged forward source target |
| Independent centered SVD coefficients | A1.89e-10, b1.66e-9, c6.29e-9 maximum errors | All pass unchanged1e-7 absolute +1e-9 relative tolerance |
| Independent complete filter replay | Particle error1.25e-14; likelihood error7.29e-10; exact ancestry | Actual fit-to-forward call chain checked |
| Fresh forward streams | Both valid; minimum fitted precision margin0.0139 | Two-seed scoped validity, not empirical performance |
| GPU execution | TF2.20 development build, FP64, TF32 off, XLA on, memory growth verified | Peak allocator65,595,136 bytes in attempt32 |

The validated scope is d8, H101, N512, forcing4.8801 and one refinement.
The two fitting references use the author seed clouds; fresh forward seeds are
20261001 and20261002. Exact raw-author coefficient parity remains numerically
unresolved and is not claimed. The port is paper-derived; the author snapshot
is an independent reference, not copied runtime code.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | What is not concluded |
|---|---|---|---|---|---|
| Accept corrected filtering comparison | Exact untwisting and converged quadrature pass | Conditional heuristic underperformance blocks superiority/default claims | One source series and absent central observations | Preserve result; keep normalizer and filtering claims distinct | General filter superiority |
| Accept scoped Lorenz equivalent-solver repair | Independent coefficient and full-consumer parity pass | No validity veto; prior failures retained | Multiple refinements, forcing/dimension changes and author raw numerics | Wider source consumer validation before Figure6 | Full Lorenz replication |
| Continue multimodal RBF work | Source operation known, implementation pending | No numerical experiment yet | NNLS cost, source horizon discrepancy and top16 support | Cost-bounded source reference then separate TF port | Quadratic port covers RBF |
| Continue KDM learning work | Primitive and sequential controls passed earlier | Exact training settings unresolved | Parameters, schedule and affordable repeated learning | Provenance or labeled reconstruction, target-specific protocol | Empirical learning replication |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Invalid narrow reference rejected; raw normal-equation portability unresolved; conditional filtering underperformance vetoes superiority/default claims |
| Statistically supported ranking | Scoped pointwise95% contrasts support likelihood-variance reduction and extreme-regime filtering losses; no simultaneous/global ranking |
| Descriptive-only differences | Lorenz two-seed likelihoods, coefficient errors, timings and memory; other unqualified metric differences |
| Default readiness | Not established; controlled-SMC filtering superiority is vetoed on this source series |
| Next evidence needed | Remaining controlled-SMC empirical matrices, repeated learning for KDM, regime coverage and cost-qualified comparisons |

## Execution and terminal review

Artifacts are preserved under
artifacts/controlled-smc-kdm-replication-20260928/attemptNN_PHASE.
Attempts20 and22 contain guide validation and the full comparison;24 contains
the Lorenz author fixture;25 the conditioning diagnosis;27 and29 the first SVD/QR
comparison;30 and32 the exact-ratio reference and successful validation.
Attempts21,23,26,28,29 and31 preserve rejected reference, serialization,
metadata, numerical and XLA failures. No threshold was relaxed to obtain a pass.

Commands were the dispatcher run command with phases c3_guided_validation,
c3_guided_matrix, c1_lorenz_consumer, c2_lorenz_consumer,
c1_lorenz_solver_reference, c2_lorenz_solver_validation,
c1_lorenz_centered_reference and c2_lorenz_centered_validation.
Each manifest contains the actual child command, source snapshots, git state,
environment, input hashes, seed, wall time and artifact paths. CPU references
explicitly hide GPUs; numerical GPU commands use trusted access.

Through attempt32, charged CPU time is102.426504s and GPU time is
10816.088075s. Remaining inherited resources are650.740431 CPU
seconds and120135.212275 GPU seconds (33.370892h).
No numerical job remains running at this stage boundary. The comparison plot
was generated; visual inspection through the image tool was blocked by its
filesystem sandbox, so no visual-review claim is made.

Skeptical terminal review: implementation, numerical validity and scientific
interpretation are separate. Shared math between SVD and QR could conceal an
algebra error; the independent author forward replay and Gaussian identity
reduce that risk. The weakest scientific evidence is extrapolation from one
data series and a single Lorenz scope. Wider-refinement disagreement would
overturn solver generalization, while additional independent series could
change filtering-performance conclusions. Nothing inspected establishes a
fundamental defect in either paper. Known implementation repairs are complete
for this stage; broader replication gaps remain concrete work in the master.
