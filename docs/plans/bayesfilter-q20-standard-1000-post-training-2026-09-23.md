# Standard 1,000-point NeuTra post-training diagnostic

Owner instruction: make the existing 1,000-point Gaussian score-residual test
standard post-run verification after training. This replaces the abbreviated
32-point report for new serious q20 ordinary and repair training endpoints.

## Contract and skeptical audit before implementation

Question: does every exported completed training checkpoint receive the full
declared diagnostic, with reproducible statistics and honest completion status?
Baseline is the September 22 diagnostic definition and the current shared
post-training call chain. The underlying quantity remains
`g(z)=grad_z[log pi_beta(T(z))+log|det J_T(z)|]+z`, with iid standard-normal
base draws, and `r(z)=log pi_z(z)+sum(z*z)/2`. An exact Gaussian pullback has
g=0 and constant r. No residual threshold is invented to certify training,
global coverage or HMC convergence.

Standard settings: 1,000 points by owner directive, fixed batches of 20 from
the previously successful diagnostic (50 batch-native GPU/XLA calls). Seed
derivation remains separate from training and paired-loss validation. Generate
the small 1,000-by-dimension bank with one CPU TensorFlow RNG call; the q20 bank
is only 32 KB, so a separate multiprocess generator would add overhead without
useful target parallelism. Retain exact points and per-point residual vectors
and r values in checksummed batch artifacts. Numerical kernels have one stable
input signature; Python coordinates batches, never individual target rows.
Tiny smoke tests and disposable sanity pilots may use an explicitly labeled
short bank and cannot satisfy serious admission.

Record minimum, median, mean, vector-norm RMS, p95, p99, maximum, fractions
above 1 and above the corresponding norm(z), plus the full historical quantile
and exceedance table and the range of r. Use the original nearest-lower quantile
definition, floor(p*(n-1)); distinguish vector-norm RMS from per-coordinate RMS.
Aggregate r globally before centering: averaging centered batch RMS values
would hide between-batch shifts and is wrong. Preserve numerical/status
failures and mark incomplete/invalid reports; never resample invalid rows.

Cache by map/target/beta/source/seed/count/batch/policy identity and persist
each completed batch. Check budget at each batch boundary. A resumed diagnostic
uses the saved prefix and cannot count a partial bank as the full test. New
schemas prevent the old 32-point cache/report from satisfying the new policy.
Both actual training producers and serious HMC consumers enforce the contract.
The completion summary displays the requested statistics. Include the added
probe work in training pricing rather than retaining obsolete cost estimates.

Review passes with these corrections: a sparse bank is insufficient for the
requested quantiles; a 1,000-row single graph risks unnecessary memory growth;
per-batch centering is an incorrect substitute for global r variation; finite
residual magnitude is not an HMC veto; checkpoints and old evidence must remain
immutable; timeout and missing pricing cannot silently imply verification.
The primary pass criteria are analytic statistic checks, exact resumability,
real producer/consumer wiring, and a complete 1,000-row CPU/XLA engineering
fixture. The known unrelated binary-tail-ESS integration issue is not relaxed.

## Execution scope

Implement the shared Python procedure, update both training entry points,
pricing and report rendering, and run focused CPU/reference tests with GPU
devices hidden before imports. Ten minutes per test invocation is the existing
engineering timeout. Preserve all unrelated work on main. Artifacts go to
`docs/plans/artifacts/q20-standard-1000-post-training-2026-09-23/`.
No new training, HMC or research parameter search is part of this code change.
Existing 32-point results stay historical diagnostics; installation of the
standard does not retroactively count as testing the saved maps on 1,000 points.
The current research balances remain 144386.38292394514 campaign seconds and
418.80836451620416 diagnostic seconds unless actual new GPU work is charged.

## Implementation and terminal review

Implemented September 23 on the existing main worktree. The shared
`bayesfilter.inference.neutra_post_training.PostTrainingProbe` now defaults to
the full bank. `post_training_settings` supplies the same standard to ordinary
q20 training (`_evaluate_rung`) and repair training (`run_repair_arm`). The
report is stored as `assessment.post_training.geometry` in exported training
records. This is an endpoint diagnostic, not an extra check after every
optimizer update.

The v2 report contains the requested vector-norm statistics and companion
log-density range. `score_residual_rms` remains the explicitly labeled
per-coordinate RMS; `score_residual_norm.rms` is the vector-norm RMS. Quantiles
retain the original lower-index convention, including the lower median for an
even number of points. The geometry statistics describe standard-normal base
draws and are not posterior sample diagnostics.

`FrozenLossCache` preserves all 50 batches, their latent points, residual
vectors, and r values under `validation-cache/*/geometry-*/batch-*.json`.
The summary is emitted only after all declared batches exist. Cache reuse
checks map/target/beta/source/seed/settings identities and ordinary checksums;
a changed checkpoint receives its own diagnostic. A budget pause preserves the
prefix. Ordinary training resume carries the cache forward, and repair workers
report incomplete post-training assessment explicitly when their allocation
expires. Saved 32-point reports do not satisfy the new serious HMC consumer
check; a complete report with finite statistics and matching checkpoint is
required. Large finite residuals remain explanatory rather than a newly
invented admission threshold.

Pricing now measures diagnostic batch setup, steady throughput, and summary
cost separately. These measurements include only two target batches and a
synthetic summary for timing, explicitly not full verification. Reservations
charge the required endpoint banks. Historical pricing without the new fields
is incomplete, and the campaign forecast names the missing training category.
The repair watcher displays point count and all requested summary statistics.

Verification artifacts are in
`docs/plans/artifacts/q20-standard-1000-post-training-2026-09-23/`; the manifest
records the available commands, environment, test times, and source hashes.

| Check | Result | Interpretation |
| --- | --- | --- |
| Initial focused probe/admission checks | 24 passed in 23.39 s | Includes analytic and report-consumer checks |
| Training, repair, pricing, reporting and master regressions | 117 passed, 1 failed, 1 deselected in 219.83 s | Failure was an obsolete expectation that an old plain-map rejection skips fresh plain-map trials |
| Follow-up controller and pricing checks | 22 passed, 1 deselected in 51.21 s | Corrected the stale test and verified the missing diagnostic pricing category |
| Whitespace and syntax checks | Passed | Engineering checks only |

The obsolete test now requires preserved historical failure evidence followed
by fresh plain-map training and tuning. Controller behavior was not changed to
restore the retired skip rule. The deselected test is
`test_master_stops_after_valid_plain_estimate_and_reuses_completed_stages`:
the earlier post-training implementation run already reproduced its unrelated
unavailable tail ESS for the binary `positive_theta_2` functional. Its scientific
checks remain unchanged. These are focused regressions, not a full repository
test-suite claim.

The numerical tests include a complete 1,000-point CPU/XLA evaluation checked
against independent Python standard-library statistics, a nonlinear transform
with its log-Jacobian score, exact Gaussian and badly scaled Gaussian cases,
global centering of r, and exact continuation after a pause at 40 points. They
also check rejection of stale, short, partial, numerically invalid, or
incomplete-statistic reports and that the probe leaves training state unchanged.
GPU devices were deliberately hidden before TensorFlow imports. No new q20
training, HMC, or saved-map 1,000-point evaluation was run.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use the standard procedure at new q20 training endpoints | Producer wiring, exact statistics, cache continuation and consumer regressions pass | Invalid or incomplete reports cannot satisfy serious trial admission | Actual target-specific GPU timing and learned-map geometry remain to be measured | Include the full bank in the next funded post-training run | Existing maps are well trained or HMC has converged |

Terminal self-review found no remaining implementation blocker for this
procedure. The weakest evidence is hardware coverage: the new complete bank
was tested on CPU/XLA analytic fixtures, not the saved q20 GPU target. Even a
small residual on every sampled base point can miss posterior regions the map
does not reach; downstream convergence and reference checks remain necessary.
The research balances above are unchanged; these were engineering checks.

## Subsequent saved-map execution

The owner subsequently requested an actual GPU run. The standard procedure has
now produced complete 1,000-point reports for control, repaired clipping and
depth four; batch 128 paused with 440 saved points at the diagnostic budget
limit. See the [saved-map results](bayesfilter-q20-saved-maps-1000-point-results-2026-09-23.md)
for numerical outcomes, current accounting and the outstanding continuation.
The engineering-run balances above are historical after that GPU execution.
