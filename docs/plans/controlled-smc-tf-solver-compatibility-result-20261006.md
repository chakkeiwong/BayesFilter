# TensorFlow solver compatibility: completed diagnostic, replication incomplete

Supplying the author's solver branches does not reproduce the author's
regressions. GPU/XLA and CPU graph TensorFlow each pass the existing strict
prediction screen on72 of229 saved systems. All229 answers are finite in both
lanes. The failure therefore cannot be repaired merely by adding the missing
Cholesky/SVD branch or disabling XLA. It does not establish that either complete
filter has worse statistical performance: that remains the separate P1 question.

The [plan](controlled-smc-tf-solver-compatibility-20261006.md) was reviewed before
execution. No production numerical core, author method, rank cutoff, ridge,
damping or acceptance tolerance was changed. The only code repairs were in the
new diagnostic: explicit float64 SVD convergence settings and CPU artifact
serialization. Controlled-SMC and KDM replication both remain incomplete.

## Comparator, operations and results

The exact comparator is Octave's recorded finite-arithmetic answer, not an
ideal least-squares solution. Its linked system LAPACK reproduces all229
coefficient vectors bit for bit:128 earlier crossover systems and101 source
regressions from F=4.8801, seed20260930. These are reused diagnostic cases, not
229 independent repetitions. There are155 Cholesky and74 SVD source branches.
Source SVD retains rank45 in every fallback.

The TensorFlow diagnostic receives the exact source matrix and right-hand side
and the source branch as an oracle. Cholesky uses `cholesky_solve`; SVD uses
`V diag(1/s) U' h` with the source relative rank cutoff2^-53. This isolates
arithmetic compatibility and deliberately omits runtime branch estimation.
The primary screen compares both ordinary and centered polynomial predictions
at the original source cloud using1e-7+1e-9*abs(reference). It is an inherited
localization screen, not a calibrated no-worse or final acceptance criterion.

| Saved systems | GPU/XLA passing | CPU graph passing | Maximum GPU polynomial difference | Maximum CPU polynomial difference |
|---|---:|---:|---:|---:|
| All229 |72|72|0.579438|0.577898|
| Earlier128 crossover systems |68|68|0.258259|0.256828|
|101 offset regressions |4|4|0.579438|0.577898|
|155 Cholesky branches |36|36|6.66e-5|1.31e-4|
|74 SVD branches |36|36|0.579438|0.577898|

The centered-prediction errors are essentially the same. These differences are
in the fitted log-policy polynomial, not likelihood percentages. CPU graph and
GPU/XLA produce different coefficients even where the aggregate pass counts
coincide; equal counts do not mean equal answers. No performance ranking follows.

GPU/XLA loses one retained rank at times20–26 of the offset case:7 source
rank45 systems become rank44. CPU graph does so at times16,18–26:10 systems.
This happens under the same configured cutoff because the implementations
return different smallest singular values. The relative cutoff was not tuned.

| Offset time | Source DGELSD smallest/largest singular value | GPU/XLA | CPU graph |
|---|---:|---:|---:|
|20|3.94589e-16|7.18499e-17|0|
|25|2.32922e-16|7.86039e-17|0|
|28|1.14074e-15|1.91832e-16|1.92629e-16|

The cutoff is1.11022e-16. These are the algorithms' reported spectra, not a
claim that any one is the exact mathematical spectrum. At time28 all methods
retain rank45, yet the GPU and CPU polynomial differences from source are
0.450768 and0.452118. Rank matching alone is also insufficient.

## The concrete XLA issue

Attempt157 stopped before the real cases: default GPU/XLA `tf.linalg.svd`
missed the solution of a known, well-conditioned dense SPD system by
3.915788125e-7. Cholesky/LU and the diagonal controls did not show this problem.
The local build source `tensorflow/compiler/tf2xla/kernels/xla_svd_op.cc`
sets public SVD convergence tolerance1e-6 and
`2*Log2Ceiling(max(m,n))` sweeps, which is12 for45x45. The actual compiled
control HLO independently contains constant12 and
9.9999999747524271e-7. Float64 dtype alone does not request float64 convergence
accuracy from this implementation.

The diagnostic retry uses explicit XlaSvd with2^-52 convergence tolerance,
100 sweeps and HIGHEST dot precision. It passes all seven known-answer controls;
the dense SVD error falls to1.01030e-14. This combined settings change was not
ablated to assign improvement separately to tolerance and iteration count.
The source rank cutoff remains2^-53. Every exercised graph traced once, so
retracing is not the explanation for these runs.

This newly discovered XLA SVD limitation is an obstacle to a proposed repair.
It is not the cause of the earlier1.1% filter gap by itself: those earlier
TensorFlow candidates used LU or QR, not this SVD endpoint. The corrected SVD
still fails saved-system compatibility, and CPU TensorFlow also fails it.

## Why small residuals do not settle the comparison

The post-run reference audit calculates the normwise backward error

`||G beta-h||inf / (||G||inf ||beta||inf + ||h||inf)`.

Its maximum over229 systems is2.14e-15 for the source,1.72e-15 for GPU/XLA,
and2.47e-15 for CPU graph. Thus all three can have very small relative
linear-system residuals while producing materially different policy values.
These float64 backward errors are explanatory, not precision-certified bounds.

This is consistent with the checked localization result: forming `G=D'D`
squares the exact-arithmetic two-norm condition number. The time28 design
condition number is about7.20e7, giving a normal-equation condition scale
about5.19e15 before finite-precision perturbations. The measured source
reciprocal one-norm condition is7.81e-17. Factorization roundoff therefore
matters to the coefficient and policy values. Source DGELSD, XLA SVD, and CPU
TensorFlow SVD are different finite-arithmetic computations of the same
full-rank formula, and rank truncation additionally changes some TF solves.

The evidence identifies an implementation/backend compatibility problem and
ill-conditioning. It does not establish a fundamental failure of the paper's
controlled-SMC method. It also does not establish that more accurate original
design least squares yields a statistically better filter.

## Attempts, resources and verification

Artifacts are under
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/`.

| Artifact | Outcome | Charged wall seconds |
|---|---|---:|
|`solver-compatibility-reference-20261006-01`|229 exact source answers; independent CPU reference|2.266400 separately recorded|
|`attempt157_p0_tf_solver_compatibility`|Stopped on default-XLA SVD control|5.829373 GPU|
|`attempt158_p0_tf_solver_compatibility`|All systems evaluated; final GPU SerializeTensor failed|69.867636 GPU|
|`attempt159_p0_tf_solver_compatibility`|Complete preserved GPU/XLA comparison|70.764695 GPU|
|`attempt160_p0_tf_solver_cpu_reference`|Complete CPU graph comparison; CUDA hidden|4.575467 CPU|
|`solver-compatibility-backward-error-20261006-01`|Source spectra and saved-solution residual audit|2.953988 separately recorded CPU|

Attempt158's CPU-only serialization operation was mistakenly placed inside an
explicit GPU context. The repair moved artifact serialization to CPU and added
incremental row/control preservation. Both failed attempts remain intact.
The attempt ceiling rose to160 to permit this localized retry; the360-second
GPU phase cap and total campaign grant were unchanged.

Successful GPU evidence records RTX4080SUPER, float64, TF32 off, XLA on,
verified memory growth, allocator use and trace counts. The CPU comparison is
an explicit non-default diagnostic with CUDA hidden. TensorFlow version is
2.20.0-dev0+selfbuilt. Failed attempts did not reach the terminal memory-policy
summary; their code/logs are retained and they provide no qualification evidence.
All actual commands, seeds, hashes, source snapshots and wall times are in the
individual manifests. Compiler-source snapshots are in
`solver-compatibility-source-audit-20261006-01`.

Remaining charged allocation:702.756496 GPU seconds (11.71minutes) and
151423.790188 CPU seconds. The5.220388-second reference/post-analysis total and
routine software checks are recorded separately. There are no running or queued
numerical jobs. Focused software checks passed17 tests before execution and
again after the serialization repair. The terminal self-review passed17 tests,
4 compilation checks and143 checksum checks, verified both72/229 verdicts,
and confirmed the current checkpoint and absence of active jobs. Evidence:
`solver-compatibility-final-review-20261006-01/summary.json`. No independent
agent review was requested or run.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Do not promote forced-branch TF candidate|72/229 pass in each lane|Rank mismatches; prediction mismatches|Compatibility tolerance is not P1 calibration|Inspect source DGELSD arithmetic on times20,25,28 before another solver repair|Complete filter is statistically worse|
|Reject default XLA SVD for this float64 diagnostic|Known SPD control failed|3.92e-7 error|Tolerance and sweep effects not separated|Retain explicit accuracy controls for any future XLA SVD|Existing LU/QR gap was caused by SVD|
|Continue replication work|P0/P1 and KDM unfinished|No budget or research-direction veto|Downstream non-inferiority and remaining source cases|Keep mathematical/source fidelity and downstream qualification separate|Paper invalidity, default readiness, HMC readiness|

| Inference category | Status |
|---|---|
|Hard veto screen|Default-SVD control failed; tightened and CPU controls pass; both preserved-system compatibility candidates fail|
|Statistically supported ranking|None; these are deterministic saved-system diagnostics|
|Descriptive differences|Policy errors, backward errors, residuals, pass counts, ranks and timing|
|Default readiness|Not established; no production core changed|
|Next evidence needed|Source-operation compatibility or a justified equivalent-algebra P0 argument, then fresh independently calibrated downstream P1 comparison|

Post-run red-team: the strict prediction screen could reject a filter with
acceptable or better downstream behavior. It is therefore a veto on claiming
finite-arithmetic source compatibility, not proof of filter inferiority. The
same failures in CPU and GPU, including full-rank time28, defeat an explanation
based only on XLA or truncation. A source-operation implementation that passes
the preserved systems would reopen the direct-compatibility route. Fresh P1
evidence could support the separate equivalent-algebra route, but cannot be
replaced by these residuals or fitted-policy substitutions. No arbitrary rank
floor, ridge, author repair, or tolerance relaxation is justified by this result.
