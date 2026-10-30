# M13 score measurement before and after main synchronization — 2026-10-04

The question is whether synchronizing main changed M13's analytical proposal
score accuracy, either as a derivative of its own finite LEDH likelihood or
as a force for the distinct frozen importance-likelihood acceptance target.
This is a diagnostic within the surrogate-HMC campaign. It does not resume
sampling, change controls, issue tuning authority, or establish convergence.

## Evidence contract and research intent

Compare the exact preserved pre-sync working tree (stash 2ca7b4f plus
hash-checked original untracked source files) with a frozen copy of the current
post-sync working tree (HEAD a55c6be plus its local modifications). Preserve all
source hashes. Use M13 data1704, T20, four-dimensional states, two observations,
two log standard-deviation multipliers, N4096, design seed20260923/partition9,
and the previously selected materiality controls. No renewed tuning is involved:
these are frozen historical settings under diagnostic comparison, not newly
admitted controls.

Primary quantities are paired, pointwise analytical-score errors against
step-refined central differences of each version's own FP64 finite likelihood,
and the error of the actual FP32 proposal force against central differences of
the frozen FP64 acceptance log posterior. The FP64 LEDH diagnostic promotes the
same once-rounded inputs/model coefficients; it approximates the real-valued
finite algorithm, not a derivative of discontinuous floating-point rounding.
Report raw and frozen-whitened coordinates. The common Gaussian prior has
sd log(2) in both coordinates and is included exactly in the posterior force
comparison.

Choose eight fixed probes before evaluating any scores: actual archived draw
rows nearest empirical process-scale percentiles 1,5,25,50,75,95,99, plus the
pooled mean. These are visited-state probes, not independent posterior draws;
the old retained chain has unresolved mixing. Round the probe coordinates once
to FP32 and use these identical coordinates in every arm. The low-percentile
points test the sticky region without interpreting empirical frequency as
posterior truth.

Use central differences at h=1e-3, 3e-4, 1e-4 and save every value. Stability
means the two finest derivatives differ by <=1e-4*(1+abs(finest derivative))
per component. Failure is a reference-resolution repair trigger, not automatic
evidence of a wrong score. Add a bounded larger/smaller step ladder if needed.
An error reduction is a measured improvement at a fixed probe only if it
exceeds three times the sum of both finite-difference refinement discrepancies.
No statistical ranking over the posterior is claimed from eight fixed probes.

Nonfinite values, invalid program status, changed fixed inputs, import of the
wrong source tree, or unresolved finite-difference instability veto the affected
comparison. An invalid source reconstruction vetoes the whole before/after
comparison. These are continuation vetoes for that measurement until repaired.
A large but resolved score error is a result to report, not a reason to discard
a probe or stop the diagnostic.

Explanatory quantities: value shifts, score shifts, derivative step curves,
relative L2 errors, angles, runtime, and frozen-coordinate force discrepancies.
Constructed cheap force adversaries are zero force (flat potential), prior-only
force (ignore data), and the Gaussian force from the saved center/covariance.
Compare each pointwise to the same finite acceptance-target derivative. These
are sanity checks, not tuning objectives. The finite acceptance target is the
oracle for this comparison; it is not the exact integrated model likelihood.

## Default and assumption audit

| Choice | Provenance / purpose | Failure mode / early check | Status |
|---|---|---|---|
| M13/T20/N4096/selected controls | Actual saved chain scope | Wrong scope: compare archived data and input hashes | Frozen comparator |
| Pre-sync stash plus untracked source | Main-sync preservation record | Missing/changed original source: check inventory hashes and imports | Exact reconstruction |
| Post-sync sources | Current checkout | Changing code during run: frozen source copy and hashes | Candidate under measurement |
| GPU1, TF32 FP32, XLA, batch4 | Actual force arithmetic; idle local GPU | Compiler/device or memory issue: one batch pilot, memory growth, record placement | Runtime lane diagnostic |
| FP64 finite differences | Independent numerical derivative check | Truncation/roundoff/branch change: three steps and validity | Reference hypothesis |
| Eight visited-state probes | Bounded stratified coverage | Sparse/nonconverged archive: conditional conclusions only | Diagnostic design |
| Saved whitening | Actual frozen HMC factor | Wrong coordinate transform: g_z=L^T g_theta, covariance reconstruction | Explanatory |
| Fixed endpoint seed / 8x4096 trajectories | Actual acceptance target | Different random design: source and fixed-seed checks | Frozen comparator |
| Existing ridges, caps, damping | Previously selected controls | Altered scientific object: no retuning or new protection in this test | Historical diagnostic settings |

## Skeptical audit and pre-mortem

The main KSC observation-tangent fix is not called by M13. Shared numerical
changes could still matter. A comparison only of two score vectors cannot
establish accuracy; the plan includes independent scalar derivatives. Comparing
the LEDH score with the acceptance target alone would conflate an implementation
error with two different finite likelihoods, so both comparisons are required.
Fixed points and unchanged inputs prevent data/design differences masquerading
as a repair. FP64 reference arithmetic is labeled separately from the actual
FP32 runtime force. The finite endpoint is explicitly distinguished from the
true likelihood. Old source-bound tuning artifacts are provenance, not authority
for the new code. The audit passes for this bounded diagnostic.

## Execution and limits

Environment /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
Driver: docs/plans/artifacts/surrogate-hmc-m13-score-before-after-20261004-01/measure.py
Output root: the same new versioned directory. Prepare uses CPU with GPUs
intentionally hidden; numerical workers use escalated GPU access,
CUDA_VISIBLE_DEVICES=1, TF_FORCE_GPU_ALLOW_GROWTH=true, verified memory growth,
TF32 enabled for FP32, stable tf.function signatures and XLA. No pfor.
Record executed commands, versions, GPU, source/input hashes, seeds and times.

Budget: 60 minutes aggregate numerical-worker wall time, no worker over
20 minutes; at most two localized infrastructure retries within that cap.
Begin with four-point batches. A reference-resolution repair may use remaining
budget without altering inputs or controls. Preserve partial results on failure.
Stop when all fixed-probe measurements and their checks are complete or this
budget is exhausted. Persist report.json, result.md, run-manifest.json and full
worker logs. The active checkpoint points to this plan until completion.

## Localized packaging repair

Attempt 1: pre-sync FP32 passed in 54.60 seconds; post-sync import failed in
3.08 seconds because the isolated Python-source copy omitted the existing
_symmetric_sylvester_ops.so. Copy the unchanged library, record its SHA-256,
and continue the three remaining arms under attempt-02. No numerical source
or scope changes. CUDA_VISIBLE_DEVICES=1 maps to the RTX5080 on this machine;
the verified device name, memory-growth configuration, and allocator metrics
are recorded, rather than inferred from nvidia-smi display order.

## Focused result-audit supplement

Within the same60-minute budget, repeat the eight FP32 base probes in each
source version, capture the full returned base status and endpoint diagnostics,
and check actual runtime-force parity at all eight points (the first worker
checked four). This checks whether the observed FP32 differences reproduce and
retains computed observability. No extra random seeds or HMC transitions.
observe.py writes a fresh observability-01 directory, <=300seconds per arm.
Run alongside the remaining FP64 diagnostic on the same recorded GPU; aggregate
worker wall time, not concurrent elapsed time, counts against the budget.

## Completed measurement and interpretation

All four primary workers and both FP32 observability workers completed. Results:
[measured comparison](artifacts/surrogate-hmc-m13-score-before-after-20261004-01/result.md).
The pre/post FP64 scores differ by at most 2.94e-12 at the eight fixed probes;
both agree with their own finite-likelihood numerical derivative within 2.57e-6
per component. Agreement with the separate acceptance-likelihood score remains
poor in the low-Q region. The finite acceptance target itself is unchanged.

The FP32 repeat supplement shows cross-process score variation. Therefore the
primary-run per-probe FP32 difference labels describe those evaluations only:
finite-difference resolution does not account for FP32 execution variation and
cannot establish a reproducible source-change benefit. No stochastic/global
ranking, posterior validity, or new default follows.

One postprocessing assertion expected correctly rounded scalar division, whereas
the actual XLA wrapper uses the numerically matching rounded-reciprocal form.
The mismatch was one FP32 ULP (2.98e-8) at one component. The reconstruction was
corrected and checked exactly against all eight actual runtime forces in both
versions. No likelihood, derivative implementation, controls, or data changed.
This is a diagnostic arithmetic repair, separate from the logged initial
snapshot packaging repair. The completed measurement uses 32.03 minutes of
aggregate numerical-worker time including CPU preparation, within the 60-minute
cap. No HMC transitions were run.
