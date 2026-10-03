# Remaining iAPF replication gaps: executed audit

The independent R reference now has stronger evidence for its filter mechanics,
but it still differs from the paper's fitting procedure and numerical results.
The main empirical veto is a conditional loss to FA-APF on one d20 dataset:
large-innovation prefix MSE is0.0410 for iAPF versus 0.0145 for FA-APF. The paired
95% interval for the difference is[-0.0060,0.0861], so the observed loss blocks
promotion under the conservative screen without establishing true inferiority.

This stage completed the [reviewed plan](../../iapf-r-replication-gap-audit-2026-09-21.md),
including the d40 timeout repair and a fixed independent check of an apparent
d20 mean deficit. Ten worker launches consumed 1850.801430 of 2400 reserved
worker seconds;549.198570 seconds remain unspent, with the ten-launch allowance
fully used. No worker remains active. Detailed source anchors, derivations and
the complete remaining-gap register are in [the code and mathematical audit](source-and-math-audit.md).

## What was repaired and checked

The previous local Equation 15 arm rejected any active optimizer bound. That
established sensitivity to an imposed box, not an invalid constrained filter.
An explicit diagnostic option now accepts a converged constrained fit while
recording active coordinates and projected gradients. The former rejecting
behavior remains available and unchanged by default. This is an optional R
reference path, not a production default change.

The first box-width experiment accidentally changed optimizer parameter scaling
with the box width. It is preserved as confounded evidence. The corrected
attempt freezes optimizer scaling and varies only the bounds; all eight fits
(QR plus three box widths, at d10 and d20) converge. See [repair note](repair-note.md).

Executable checks pass for the exact future-guide/Kalman identity, actual
fully-adapted consumer equivalence, finite-positive-floor proposal/weight
telescoping, original-prefix likelihood correction, and constrained-consumer
wiring. Twelve existing regression tests also pass. The common R numerical core
was unchanged; both captured source trees pass hash verification. The d40
completion uses the same data, seeds, worker and core, with exact observation
and Kalman-file equality before merging complete records.

## The fitting problem is substantive

At particles x_i, let b_i be the positive backward target and p_i a normalized
Gaussian density. Equation 15 minimizes sum_i(p_i-lambda*b_i)^2. Eliminating its
free scale gives

\[
 L_* = \|p\|^2 - (p^T b)^2/\|b\|^2.
\]

For covariance V=s^2 V0, p_i tends to zero as s grows, so L* tends to zero even
for a poor shape. This proves an escape in the unrestricted objective; it does
not prove that the authors' particular local optimizer failed. Compact bounds
remove that escape but introduce numerical choices absent from the manuscript.
The currently usable QR reconstruction instead fits a quadratic to log b and
therefore does not minimize the printed objective.

The density loss also weights relative errors by b_i^2. In the fresh d10 and
d20 t99 training clouds, those weights have effective counts 3.05 and2.35 out
of 1000. The corrected experiment demonstrates why a small training loss cannot
select a guide:

| Case | Fit | Scaled training Equation 15 loss | Heldout smoothing relative density residual | Heldout centered log RMSE |
|---|---|---:|---:|---:|
| d10 | QR |7.08e-5|0.0611|0.322|
| d10 | box0.5 |4.91e-8|0.1173|0.433|
| d10 | box1 |2.06e-9|0.4951|1.195|
| d10 | box2 |1.30e-9|0.6219|1.955|
| d20 | QR |1.85e-4|0.1084|0.432|
| d20 | each tested box |3.22e-10|0.0970|0.795|

Loss scaling is fixed within each dimension; the d10/d20 losses are not
cross-dimension comparisons. Each diagnostic uses one fresh training cloud and
independent 2000-point predictive and smoothing clouds. All predictive results
are retained in the structured report. These explain mechanisms rather than
rank full algorithms. The d20 disagreement between density and log residuals
also shows that the diagnostic metric matters.

The constrained Equation 15 d5 pilot completed: ratio 1.04176, finalN2000,
31 active-bound fits, and all 100 Gaussian-limit tail checks passing. This
rescues the route from categorical rejection. One pilot is insufficient to
establish paper replication or competitiveness.

## The controller measures learning history as well as filter noise

The first four prior d80 validation histories were reconstructed exactly:
both likelihood histories and final outputs agree with their saved originals
to the recorded precision (maximum differences zero). Their guides immediately
before the first eligible doubling were saved and each replayed16 times at
N 1000 using fresh randomness.

| Learning history | Controller six-estimate CV | Last-five CV, explanatory only | Fresh pre-doubling guide CV | Fresh mean ratio,95% interval |
|---|---:|---:|---:|---|
|1401|0.687|0.450|0.233|1.034 [0.926,1.155]|
|1402|0.517|0.155|0.192|1.046 [0.953,1.147]|
|1403|0.556|0.245|0.214|0.990 [0.888,1.085]|
|1404|0.656|0.407|0.205|0.903 [0.823,0.995]|

The observed fixed-guide CVs are well below the controller threshold0.5 even
before doubling. This supports learning transients as an explanation for excess
particle use. It does not establish a replacement stopping policy. In particular,
history1404's small-sample mean interval excludes1; fixed-guide accuracy and
tail coverage are not settled by 16 runs. The four histories were selected by
their first IDs, not their outcomes, and are conditional examples rather than
a random sample supporting a universal controller claim.

For the final guides, the observed ratio SDs atN1000 range0.247--0.328 and at
N 2000 range0.129--0.224. Guides and floors were held fixed when changing N.
These differences are descriptive. The exact future-guide control agrees with
Kalman atN32 in all four runs. The observation-only control atN1000 has mean
ratio 0.0809 over only four runs; that is a poor finite-sample outcome, not proof
of a biased or incorrectly implemented fully adapted filter.

The six-estimate window is printed in Algorithm 4. Shortening it or using a
fixed-guide variance check would be an explicit algorithm extension. The
positive floor, meanwhile, bounds fixed-guide weights at finite horizon in
ideal arithmetic; the Gaussian-limit tail test is a conservative quality
diagnostic, not proof about infinite variance of the positive-floor algorithm.

## New full-filter evidence

All cells below use the frozen delayed/QR/floor8 reconstruction. The first three
include BPF10000, FA-APF5000 and SIS10000, with exact Kalman likelihoods. The
confirmation is iAPF-only because its question is the apparent mean deficit.

| Dataset | Repeats | Mean ratio,95% interval | Ratio SD | Mean final N | Practical screen including heuristics |
|---|---:|---|---:|---:|---|
| d20-A,seed87000020 |32|0.9412 [0.9136,0.9684]|0.0831|1062.5|Pass; mean deficit required follow-up|
| d20-B,seed88000020 |32|1.0032 [0.9675,1.0411]|0.1053|1000|Veto: observed large-innovation loss to FA|
| d40,seed87000040 |32|0.9735 [0.9300,1.0183]|0.1312|1156.25|Pass|
| d20-A,new fixed batch |64|1.0163 [0.9902,1.0412]|0.1048|1000|Mean check only; no new heuristic comparison|

The initial d20-A deficit did not recur. Pooling96 repeats gives 0.9913
[0.9710,1.0125], reported as exploratory because the first 32 triggered the
follow-up. The confirmation used exactly 64 new seeds and stopped there. Neither
interval coverage nor a practical-screen pass proves unbiasedness; fresh-final
conditional APF unbiasedness is the separate mathematical argument.

The d40 timeout left27 complete four-method replicates. A separate attempt
completed the remaining five seeds. The derived32-repeat dataset contains no
duplicates or incomplete records and retains links/hashes for both source
attempts. All160 fresh full iAPF replicates have finite outputs,96700 converged
backward-fit records and16000 passing tail checks; minimum relative tail margin
is0.32027. The conditional guide replays add800 passing tail checks; the
constrained pilot adds100.

Literal table-pattern agreement remains absent. Mean final resampling counts
are 9.31 and9.47 at d20 versus the paper's27.61, and19.53 at d40 versus 42.41.
All three cells fail the separately declared variability/resampling-agreement
screen. Lower empirical variance or fewer resamplings cannot establish better
performance: the data realization and guide-fitting objective differ, and
total learning cost has not been matched between methods.

## Decisions and next justified actions

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep QR R reference as a debugging instrument | Mechanics/accuracy checks pass in tested cells | d20-B heuristic promotion veto | Generality across data and dimensions | Preserve exact identity tests and conditional result tables | Paper-identical iAPF or production readiness |
| Retain constrained Equation 15 as viable | Fits and one full pilot complete | No numerical veto in this pilot | Bound dependence, heldout shape, small sample | Compare full filters on fresh d5/d20 calibration/validation partitions, with explicit bound choices | Superiority or original-author numerical identity |
| Diagnose controller repair separately | Pre-doubling guide noise lower than history CV in four examples | No direction veto; one16-run mean warning | Four learned guides only | Compare the printed controller with a labeled extension using fresh data and total-cost accounting | A new default controller |
| Complete dimension-table replication after freezing choices | New d40 and d20 evidence complete | Literal agreement and d20 heuristic screen remain open | Unknown original data/settings;1000-repeat scale missing | Freeze one fully specified reconstruction, then execute the published five-dimension scale | That matching a few table numbers identifies the method |
| Extend study and TF coverage separately | Not yet checked | TF consumer rejects multivariate models | Whole-filter method/randomness matching | Add fixed-data alpha sweep, then multidimensional R/TF consumer parity | PMMH, nonlinear, GPU, gradient, HMC or LEDH correctness |

The next numerical question is whether an explicitly constrained density fit
can retain the full-filter accuracy of QR across fresh data. Compare the fixed
QR baseline and bounds motivated by the heldout diagnostic; use calibration
only to nominate, freeze before untouched validation, and keep density/log
residuals explanatory. Test controller changes as a separate factor so that a
fit change and a stopping change cannot conceal one another. Compare total
learning plus final-filter work with the classical baselines. These steps must
precede spending on 1000 repeats at every dimension.

The remaining source gaps are the original optimizer configuration, positive
floor rule, early-doubling convention, realized observations/RNG and verified
author implementation. The bounded sensitivity study narrows their practical
consequences; it cannot recover their historical values. The publisher final
technical text was not retrieved; the available primary v1/v2 manuscripts
and cached paper underpin the audited claims. The alpha sweep and later PMMH/SV
studies remain coverage gaps. The TF endpoint's current one-dimensional limit
and its different full-quadratic fitting family prevent treating component
parity as whole-filter parity.

## Inference status and terminal review

| Evidence class | Result |
|---|---|
| Hard veto screen | No observed numeric/fit/tail failure in completed validation; resource timeout repaired. d20-B has a conservative observed heuristic-loss veto. |
| Statistically supported ranking | No overall, simultaneous, equal-cost, or bound-choice ranking. Pointwise conditional comparisons are preserved in JSON. |
| Descriptive-only differences | Fit-box residuals, guide-replay variance, resampling and particle-count differences, and the one constrained pilot. |
| Default-readiness | Not established; all new numerical work is an independent CPU R reference. |
| Next evidence needed | Fresh full-filter fit/controller comparisons, exact method freeze, multivariate TF parity, published replication scale and missing study coverage. |

Terminal skeptical self-review: the training-loss minimum is not a promotion
metric; active bounds are not mistaken for invalid weights; final-guide replay
is supplemented by pre-doubling replay; the apparent d20 deficit receives a
fixed independent confirmation; the original resource failure and confounded
attempt remain preserved. The paper's resampling definition was rechecked and
removed from the list of unresolved algorithmic definitions. No external
reviewer was used for this bounded stage.

Strongest alternative explanation: discrepancies with the published tables may
reflect the unknown realized observations and numerical conventions rather than
a fundamental failure of iAPF. Wider fresh full-filter validation could overturn
the current practical assessment. The weakest evidence is the small conditional
guide sample and the single constrained-fit pilot. Snapshot and record-count
checks, focused regression and git diff --check pass.

Reproducibility: [manifest](manifest.json), [structured terminal report](terminal-analysis/summary.json),
[d40 merge provenance](terminal-analysis/d40-complete/provenance.json), captured
source trees and per-attempt launch.json/run.log/settings files. R 4.1.2,
single-thread CPU reference with CUDA_VISIBLE_DEVICES=-1, at most two workers;
Git 6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef plus recorded source hashes.
The prior 72.641587-second balance was not transferred or spent.
