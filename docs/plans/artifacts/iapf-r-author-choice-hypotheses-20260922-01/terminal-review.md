# Author-choice campaign: result and final review

The bounded series is complete. None of the tested combinations qualifies for
validation across dimensions 5, 20, and 80. Target-weighted log fitting is computationally
viable at dimensions 5 and 20 and passes their small calibration screens, but fails
at dimension 80. The alternative floors and SD convention do not remove that
failure. This narrows the next repair to high-dimensional fitting stability.
It does not establish that the original iAPF fails or replicate the authors' code.

The authoritative numerical report is
[report-v3](runs/report-v3/results/result.md), with
[structured results](runs/report-v3/results/summary.json). The original report and
all unsuccessful attempts remain preserved. The pre-execution and final reviews
were Codex self-reviews; no independent reviewer approval is claimed.

## What ran

The paper's first linear-Gaussian model used T=100, alpha=.42, identity initial,
process and observation covariances, and initial N=1000 for adaptive filters.
There were two fixed calibration observation datasets per dimension, disjoint
from the six fixed-cloud datasets. Each calibration cell scheduled four filter
replications. Exact Kalman likelihoods supplied the reference; QR, bootstrap PF,
fully adapted APF and SIS supplied the declared comparator ladder.

All 126 fixed-cloud fits completed. Of 296 scheduled full-filter replications,
288 have final records: 189 complete and 99 failed under the numerical or resource
checks. Eight lack final records because three workers reached the 180-second
limit. These are resource-censored, not numerical failures. Each affected fitting
arm also has recorded failures, so filling its censored rows cannot restore the
all-probes-complete admission condition. No untouched-validation data were drawn.

| Fitting choice | Completed / scheduled d5,d20 probes | Failed records | Resource-censored |
|---|---:|---:|---:|
| Equation15, QR start, strict termination | 6/16 | 8 | 2 |
| Equation15, QR start, loose termination | 6/16 | 4 | 6 |
| Equation15, previous-fit start, strict | 0/16 | 16 | 0 |
| Equation15, previous-fit start, loose | 0/16 | 16 | 0 |
| Log fit weighted by backward target | 16/16 | 0 | 0 |
| Log fit weighted by squared target | 9/16 | 7 | 0 |

F1 failures include nonconvergence, density underflow, configured log-variance boundaries
and the outer iteration cap. F2 usually reached later iterations: 13 strict and
14 loose failures have recorded iteration > 0. Three strict and one loose run
failed during iteration 0; one loose optimizer error lacks iteration metadata.
Thus warm starts were exercised, although not in every run.

The weighted log fits change Equation15's objective. Completing them is evidence
about an explicitly alternative reconstruction, not evidence of paper identity.

| Weighted-target fit and floor | SD convention | Calibration datasets passing / six |
|---|---|---:|
| Chi-square tail8 | Sample | 4/6 |
| Chi-square tail8 | Population | 4/6 |
| Peak/N^8 | Sample | 4/6 |
| Peak/N^4 | Sample | 4/6 |
| Peak/N^2 | Sample | 2/6 |

Tail8, peak/N^8 and peak/N^4 pass both datasets at d5 and d20. Peak/N^2 passes
only d5; d20 has six iteration-cap failures and two nonconcave fits. All d80
weighted-target runs fail. Population SD therefore ran on the prescribed tail8
diagnostic fallback. On all 16 completed d5/d20 paired replications it leaves
final N, stopping iteration and likelihood exactly unchanged.

For tail8, the d5 likelihood-ratio means are1.016 and1.011, with SDs 0.062 and 0.048;
d20 means are 1.036 and 0.964, with SDs 0.130 and 0.073. Mean final N is 1000 in all four
cells. These four-replication descriptions are too small to establish replication
or rank methods. Resampling counts still differ descriptively from the paper:
d5 means 1.50/1.25 versus 6.93, and d20 means 11/11 versus 27.61. The paper's realized
observation datasets are unavailable, so these are not paired table comparisons.

## What the saved failures explain mathematically

The weighted fit regresses log backward targets on an intercept, d linear terms
and d diagonal quadratic terms. With design D and weights W, it minimizes

    sum_i w_i [log(b_i) - D_i beta]^2,

whose curvature is 2 D' W D. There are 2d+1 coefficients: 161 at d80. The saved
unweighted designs have full numerical rank 161. Under the target weights,

    effective weight count = (sum_i w_i)^2 / sum_i w_i^2

is only 1.0000008–2.37115. The ten largest weights contain at least 99.989% of the
mass. This count explains concentration; the direct weighted QR check establishes
the numerical rank failure. Across 40 d80 failure records (including repeated
sample/population contexts), 36 have weighted numerical rank 82–160. The other four
retain rank 161 but produce a nonnegative quadratic coefficient. A diagonal
Gaussian requires q_j<0, since variance_j=-scale_j^2/(2q_j).

Weighted design condition numbers range from 1.95e8 to 1.68e11. Failures occur at
times 95–99 during the first backward sweep, iteration 0. The controller does not
evaluate its stopping condition until iteration > 5; changing the SD denominator
cannot repair these recorded failures. Positive floor choices affect backward
targets but did not make any tested d80 fit complete.

These findings identify numerical ill-conditioning and invalid fitted Gaussian
curvature in this weighted regression. They are not a theorem that every
weighted solver, every dataset, or the original iAPF must fail. See the
[saved-cloud diagnostics](runs/diagnostic-failures-v2/results/weighted-failure-diagnostics.csv)
for ranks, effective counts, concentration, source paths and SHA-256 hashes.

## A real implementation defect was repaired

Final review found that a shared check wrongly rejected a weighted-log fit when
the explanatory Equation15 density loss underflowed. It affected peak/N^4,
d20 dataset 2, replica 3. The declared objective was the weighted log regression;
Equation15's absolute loss was explanatory only. The original rejection was
wrong relative to that contract.

Source-v3 retains the underflow flag and applies the rejection only to F1/F2,
which actually optimize that loss. A saved-fixture test confirms that the repaired
function returns the identical already-computed finite positive-variance fit.
The identical full-filter seed then completes: likelihood ratio 1.085109,
N=1000, seven learning likelihood estimates, 11 resampling steps. Only this record
is superseded in final aggregation; its original failure is preserved.
Peak/N^4 consequently passes 4/6 rather than 3/6 calibration cells. Recomputed
selection still admits no candidate, so no downstream validation is omitted.

A separate passive-output repair retains learning-pass floor quantiles on success
and fitting errors. The original source-v1 runs lack these richer failure fields;
they are not retroactively attributed to them. Exact old-core parity confirms
unchanged likelihood histories, particle clouds and prefix values. All eight
focused checks pass in [tests-v3](runs/tests-v3/worker.log).

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this bounded hypothesis series | No combination passes all six calibration cells | d80 fitting fails; three timed workers remain explicitly censored | Few datasets and four repetitions per cell | Design a focused d80 fitting repair against the saved failures | General failure of iAPF |
| Retain weighted-target fit as a limited reference | d5/d20 calibration screens pass | d80 prevents broader qualification | Changed objective and rare likelihood tails | Require a repaired fit before untouched multi-dataset validation | Author implementation or full paper replication |
| Keep filtering-use assessment separate | Original-prefix conditional MSE versus simple filters | Seven observed conditional losses in the full comparison | Small samples, unequal costs and rare weights | Preserve the vetoes and collect adequate paired evidence before promotion | Statistical superiority or inferiority |

| Inference item | Status |
|---|---|
| Hard veto screen | Direct numerical failures are supported; timeout and outer resource caps are separately classified |
| Statistically supported ranking | None established across methods or datasets |
| Descriptive differences | Likelihood summaries, N, resampling, time and observed conditional MSE differences |
| Default readiness | Not evaluated; independent CPU R reference only |
| Next evidence needed | Numerically valid d80 fitting, then untouched multi-dataset validation and sufficient rare-tail replication |

The seven observed heuristic losses involve the loose Equation15 fit at d5 and
QR's ordinary-prefix errors on one d80 dataset. They retain the project's
conservative promotion veto; four-replication comparisons do not establish a
general statistical ranking. Failing a candidate does not reject the research
direction.

The strongest alternative explanation is a particular solver/regression failing
on particular learning clouds, rather than an intrinsic failure of the algorithm.
A stable fit satisfying the same declared objective on these clouds would
overturn that candidate rejection. The weakest evidence for accuracy remains
the tiny calibration sample. No further unchanged QR replication or SD sweep
answers the newly localized fitting problem.

## Execution closeout

Git commit 6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef, branch surrogate-hmc;
source snapshots preserve the relevant dirty state. All source and external-test
hashes in source-v1/v2/v3 verify. Paired observation hashes verify. Runs used
R 4.1.2 with CUDA_VISIBLE_DEVICES=-1 and one BLAS/OpenMP thread per worker,
at most two workers. GPU devices were deliberately hidden.

There were 69 launches and 2188.624 aggregate worker seconds of the 7200-second
ceiling; 5011.376 seconds remain unused in this bounded allocation. Elapsed time
from the first test to the final report was 1646.611 seconds. Cumulative campaign
use is 109578.421 of 172800 worker seconds. The original deadline remains
2026-09-21T20:04:26Z. No worker remains active. Exact commands, seeds, attempt
times, repair supersession and output paths are in the
[manifest](manifest.json) and [attempt ledger](attempts.jsonl).

The campaign ended on its predeclared scientific decision, with its localized
implementation repair completed. Budget, permission and reviewer availability
did not force the stop. A new fitting design needs its own bounded evidence plan;
no paper-scale expansion is implied by the unused allocation.
