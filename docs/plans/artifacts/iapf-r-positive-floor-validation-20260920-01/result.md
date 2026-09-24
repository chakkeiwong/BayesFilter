# Frozen positive-floor R iAPF validation result

2026-09-20. The optional log-quadratic R iAPF with floor power8 passed the
predeclared conditional likelihood-accuracy screen on two untouched d80 datasets
and one fresh d40 control, all at T100. All96 independent filter repetitions
completed. This supports the repaired reference on the checked linear-Gaussian
scope; it does not reproduce the paper's Eq.15 fitting objective or establish
production/default readiness.

The [reviewed plan](../../iapf-r-positive-floor-validation-2026-09-20.md) froze
the fitter, floor, controller, data seeds, repetition IDs, heuristic comparisons,
tail check and accuracy threshold before launch. Each dataset required32 repeats
and a 2,000-resample percentile bootstrap95 interval for the mean likelihood
ratio wholly within[.9,1.1]. No pooling, retuning or early accuracy stopping was
used. Kalman likelihoods are exact for the same generated observations.

## Accuracy and numerical evidence

| Dataset | Data seed | Mean likelihood ratio | Bootstrap95 interval | Ratio range | Accuracy |
|---|---:|---:|---|---|---|
| d80 A |80000080|1.020141|[.975004,1.073665]|[.816966,1.479541]|Pass|
| d80 B |81000080|.981628|[.927681,1.036752]|[.726764,1.580688]|Pass|
| d40 control |80000040|1.041181|[1.005136,1.078378]|[.788784,1.238939]|Pass|

Both d80 batches finish at N2000; adaptation uses8/9 passes in A and7/8 in B.
The d40 control uses7/8 passes and N1000/2000. No run hits the unchanged
20-pass/16000-particle caps. All64100 QR fits have full design rank, finite
required diagnostics, zero optimizer failure and no variance-boundary flag.
All9600 Gaussian-limit second-moment checks pass; the minimum relative matrix
margin is.301642, far above100*d*machine_epsilon. Maximum recorded final-filter
floor probability is9.48457e-20. The checked final filters therefore do not
reproduce the previously demonstrated near-unit floor probability.

The tail diagnostic checks M=Q^-1+2H-V^-1 on the saved final guides, with
H=C'R^-1 C+A'(Q+V_next)^-1 A and no future term at T. It establishes the
checked Gaussian-limit integrability condition for each fixed ancestor, not
a uniform particle-system variance bound or all-adaptation-path guarantee.

The d40 interval excludes1 while remaining inside the declared10% tolerance.
That is a nominal conditional discrepancy requiring honest follow-up; it is
not silently relabeled exact agreement. Finite repetitions, percentile-bootstrap
coverage and multiple datasets limit interpretation. The controller takes a
fresh final filter draw after stopping, so direct reuse of a stopping-selected
likelihood is not the explanation. This study neither proves bias nor proves
unbiasedness, and rare weight tails remain incompletely measured.

## Constructed heuristic comparisons

Entries are mean squared prefix log-likelihood errors. These are descriptive;
the predeclared veto would fire if iAPF exceeded any heuristic in either regime.
Large innovations exceed the dimension-specific Kalman chi-square90th percentile.
No veto fires. Particle/computing budgets differ, so this table is not an
efficiency ranking or evidence of general superiority.

| Dataset / situation | iAPF | Fully adapted N5000 | Bootstrap N10000 | SIS N10000 |
|---|---:|---:|---:|---:|
| d80 A ordinary |.103576|1.838973|2150280.480|74195691.987|
| d80 A large innovation |.081603|1.833250|2468059.733|93705265.412|
| d80 B ordinary |.117385|1.909028|2053284.385|42849174.090|
| d80 B large innovation |.141114|2.601371|2913314.364|64235913.598|
| d40 ordinary |.025900|.141007|89072.749|6220706.508|
| d40 large innovation |.016892|.175190|126786.789|9037307.646|

At d80, bootstrap/SIS likelihood ratios underflow to zero when their finite
log ratios are exponentiated. Their log-likelihoods and prefix diagnostics
remain finite; the artifact does not mean the mathematical likelihood is zero.
Kalman, rather than these weak high-dimensional particles, remains the primary
accuracy authority. The fully adapted comparison supplies a stronger stochastic
baseline, with its observed variability preserved in each inspection.json.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain optional power8/log-quadratic R reference for further validation |All three32-repeat Kalman intervals inside[.9,1.1]|No observed fit, tail, controller or conditional heuristic veto|Rare tails, d40 upward discrepancy, dataset dependence|Plan additional independent control and larger reference replication; resolve author-fitting gap before calling it paper replication|Eq.15 reproduction, original-author settings, general correctness or TF admission|
| Preserve existing numerical defaults |No default-change criterion tested|No policy change attempted|Alternative objective differs from paper|Keep power8/log fit explicit|New default, score/HMC/KDM/LEDH promotion|

| Inference status | Finding |
|---|---|
| Hard veto screen |No hard candidate veto in96 completed repetitions; tail9600/9600 and fit64100/64100 checks pass|
| Statistically supported ranking |None declared or established; accuracy intervals concern each method's ratio to exact Kalman|
| Descriptive-only differences |Heuristic MSE, ranges, SD, runtime, ESS and floor probabilities; no stochastic-method ranking|
| Default readiness |Not tested; this is an explicitly optional independent CPU/R reference|
| Next evidence needed |Additional untouched data/repetitions for d40 discrepancy and rare tails; paper objective/solver/floor/source reconciliation before author-replication claims|

## Execution, repairs and checks

Commit6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef, branch surrogate-hmc; dirty
source closures captured per attempt. Numerical core SHA256
c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0 is unchanged.
Base R4.1.2 runs with CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1 and
OMP_NUM_THREADS=1. CPU reference use and intentionally hidden GPUs are recorded.
The driver is Python3.11.15 in the tftwogpu environment; no TensorFlow kernel
or GPU was used. At most two R workers ran concurrently.

Each attempt/manifest.json contains the exact captured command, environment,
seeds, hashes, CPU status and time. Repetition IDs are501:532,601:632,701:732;
method seeds use53000000+dimension*100000+replication*10+method_id. Bootstrap
seeds are data_seed+900. Every attempt's diagnostic-manifest.json records the
post-run R command and input/result hashes. report-sources.json and
report-run-*.json preserve the separately captured reporting implementation.

All three launches completed, without a scientific retry. Replication consumed
3085.985672 worker-seconds; saved-guide checks/reporting consumed60.040942,
including a conservative1-second charge for one pre-launch report-wrapper
failure. Total3146.026614/4000, remaining853.973386. All planned launches are
complete; remaining time is not an allocated extra experiment. The minor
wrapper failure read a still-running diagnostic manifest as though it had a
final wall time; reserving its timeout fixed it before report execution. See
report-repair-note.md. No result or numerical setting changed.

Ten focused pytest cases passed in3.36sec, including88 R conformance checks,
33 alternative-fit checks, source mutations, captured-source execution,
child-process timeout cleanup, phase logging and an analytic finite/infinite
second-moment fixture. Isolated saved-evidence mutation checks accepted the
complete run and rejected missing fits, duplicate prefixes, inconsistent
likelihoods, false tail flags, wrong floor identity and changed observations.
These are engineering checks; scientific conclusions use the runs above.

## Terminal self-review and next stage

Self-review: PASS for the stated conditional screen and reference-only decision.
The exact comparator, seeds, numerical core and source closure agree with the
plan. No successful subset is substituted for a failed run; there are no failed
filter runs. The strongest alternative explanation of success is that these
linear-Gaussian datasets are especially suitable for the log-quadratic fit.
The weakest evidence is tail coverage from32 repetitions per dataset. A fresh
accuracy/heuristic failure or observed floor domination would overturn an
attempt to generalize this result and trigger saved-state diagnosis.

The master program must preserve two separate remaining tasks: qualify the
optional R reference with additional independent control and bounded scaling,
and establish which fitting objective/solver/floor reproduces the original
paper. Simply increasing this alternative objective to1000 repetitions cannot
close the second task. TensorFlow debugging can later use it as a qualified
alternative reference, but must not label the comparison original-paper
replication. No new filter campaign is launched from this result alone.
