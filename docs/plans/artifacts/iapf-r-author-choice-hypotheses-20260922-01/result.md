# Tests of plausible iAPF numerical choices

The bounded author-choice campaign executed. The chosen fit for floor/controller
tests was `wlog1`. Untouched-validation
candidates: **none**. 0 validation records were obtained.
These are independent CPU R reconstructions; the authors' choices are still unknown.

There are 7 observed conditional losses against the constructed heuristic comparators. These block the corresponding promotion under the frozen conservative screen; small samples and rare tails prevent a general inferiority claim.

## Fitting procedures

| Fit | Complete full-filter probes / attempted | Recorded failures |
|---|---:|---|
| f1_strict | 6/14 | 7 Eq15 optimizer did not converge; 1 iteration_cap |
| f1_loose | 6/10 | 2 floating domain boundary; 1 density loss underflow; 1 iteration_cap |
| f2_loose | 0/16 | 13 nonconcave log-quadratic regression; 1 floating domain boundary; 1 non-finite value supplied by optim; 1 density loss underflow |
| f2_strict | 0/16 | 4 Eq15 optimizer did not converge; 12 nonconcave log-quadratic regression |
| wlog1 | 16/16 |  |
| wlog2 | 9/16 | 7 nonconcave weighted log fit |

The fixed-cloud study includes exact-Gaussian controls and independent predictive
and smoothing evaluation draws. F2 cannot test previous-iteration initialization
on a filter that fails before finishing its first backward sweep. Low absolute
Equation15 loss does not establish a useful guide. Weighted log fits explicitly
change the objective. Complete diagnostics are in summary.json and worker CSV/RDS.

## Positive floors and stopping convention

| Fit / floor / SD | Calibration cells passing / six | Validation disposition |
|---|---:|---|
| wlog1 / tail8 / sample | 4/6 | not eligible |
| wlog1 / tail8 / population | 4/6 | not eligible |
| wlog1 / peak8 / sample | 4/6 | not eligible |
| wlog1 / peak4 / sample | 3/6 | not eligible |
| wlog1 / peak2 / sample | 2/6 | not eligible |

All floor variants rerun guide learning and filtering. Floor probabilities refer
to proposals actually used. Sample/population SD comparisons retain six estimates;
history replays explain threshold crossings and actual reruns measure the changed
trajectory. Paired effects and floor-probability summaries are in summary.json.
Every dataset remains fixed across methods and repetitions; source and observation
hashes were checked. No dataset was selected for resemblance to the paper.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve tested fitting hypotheses and failures | Per-cell Kalman likelihood screens retained | Numeric/completion failures remain visible | Local optimizer termination versus useful fit | Inspect scale/shape diagnostics before a new fit hypothesis | Failure of iAPF theory |
| Validation selection follows frozen order | All six calibration cells required | Missing/failed cells cannot pass | Four-replica intervals miss rare tails | Validate only fully eligible combinations | Table matching or author identity |
| General filtering remains separately assessed | Original-prefix conditional MSE | Observed heuristic losses retained | Rare-weight tails and few datasets | Fresh paired evidence needed for any promotion | Terminal success implies accurate prefixes |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Structured nonconvergence, boundary, nonconcavity, nonfinite and incomplete results retained |
| Statistically supported ranking | No overall or multiplicity-corrected ranking established |
| Descriptive differences | Resampling, N, runtime, four-replica calibration differences and controller effects |
| Default readiness | Not evaluated; CPU R reference only |
| Next evidence needed | Untouched multi-dataset replication of a numerically viable method; larger samples for rare tails |

Strongest alternative explanation is small-sample or observation-specific luck.
The weakest statistical evidence is calibration with four repetitions; published
tables are descriptive comparators, never selection losses. A fresh validation
failure would overturn calibration viability. Gaussian-limit checks concern a
zero-floor approximation and cannot prove divergence of the positive-floor filter.

Execution provenance, exact commands, versions, CPU settings, seeds, source hashes,
attempt wall times, failures and remaining budget are in manifest.json,
attempts.jsonl and source-*/sha256.json. The plan is
../../iapf-r-author-choice-hypotheses-2026-09-22.md. Deadline and budget were not renewed.
