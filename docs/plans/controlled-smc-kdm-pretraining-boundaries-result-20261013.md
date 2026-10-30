# KDM pretraining boundary result — 13 October 2026

The saved-checkpoint investigation completed at 19:07:01 Hong Kong time
(11:07:01 UTC). All 863 controller checks, 271 numerical/coverage checks and
40 terminal-review checks passed. The author reference was verified against
27 pinned source files. No new translation defect was found in these cases.

## Evidence

Actual author Trainer.train and the TF host controller give identical decisions
when replaying the saved validation histories. The TF decisions also match its
saved event records and best-checkpoint metadata. Epochs are zero-based below.

| Saved history | Completed epochs | Best epoch | LR reduction epochs |
|---|---:|---:|---|
| unit01 author | 132 | 85 | 75,106 |
| unit01 TF | 126 | 77 | 74,100 |
| unit02 author | 164 | 122 | 112,138 |
| unit02 TF | 150 | 103 | 124 |

Thus different observed checkpoint choices are explained by different loss
histories, not a different selection/stopping rule on the same history.

The GPU experiment covers both units, author-best and TF-best measurement
parameters, training batches of64 and8, and validation batches256/256/256/232.
All24 comparisons use identical source clouds and restored parameter states.
TF-best tests restore the actual saved Adam moments, step and rates in both
implementations. Author-best tests use fresh diagnostic Adam state; no claim
is made that the author's best optimizer history was reconstructed.

| Matched-state quantity | Maximum absolute difference |
|---|---:|
| Batch mean NLL | 4.768371582e-7 |
| Normalized particle weights | 1.668930054e-6 |
| Raw / regularized gradients | 1.013278961e-6 |
| Parameters after Adam update | 3.725290298e-9 |
| Adam first moment | 1.043081284e-7 |
| Adam second moment | 1.862645149e-9 |

The four complete validation aggregates differ by at most2.441406250e-7 nat.
Every validation set covers1000 sequences /5000 selected truth points. Validation
leaves parameters and optimizer state unchanged. Instrumented author updates
match uninstrumented updates; TF repeats are exact. All eight exercised graph
signatures trace once. The RTX5080 executes XLA FP32, TF32 off, with verified
memory growth; allocator peak is32,681,216 bytes. The author runs CPU with CUDA
explicitly hidden. No NumPy numerical path was added to the TF runtime.

The independent reference helper was hardened to construct from pinned native
YAML and check equality against saved JSON. Integer dimension keys are checked.
It now supports saved TF Adam state and explicitly labelled author-best probes.
The algorithmic runtime and its numerical defaults were not changed.

## Interpretation and remaining questions

The earlier intervention removed95.26% of the unit01 final MDPF gap by replacing
only its initial measurement weights. These new tests show those measurement
models obey the same loss, gradient, optimizer and selection operations at the
checked states. Their pretraining validation NLLs are also very close on the
same clouds (unit01:4.737298 TF-best versus4.737229 author-best), despite different
later MDPF outcomes. One diagnostic draw is not a quality ranking.

Source cloud generation changes both training and validation losses. Different
random streams and ordinary FP32 trajectory sensitivity are therefore plausible
explanations; this result does not establish them as the sole cause. Existing
sampler checks cover the generic component distribution and replay, but the
composed center-plus-cloud law should be tested through native pretraining's
actual consumer before treating that alternative as sufficiently checked.
A-MDPF and independent downstream variability are not evaluated by this stage.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close saved-pretraining boundary diagnostic | Controller exact;24 same-state cases pass the localization screen | No source, coverage, state, nonfinite or device failure | Whether native composed cloud generation and stochastic trajectory variation explain the downstream saved gap | Check composed native clouds, then consolidate remaining MDPF/A-MDPF endpoint evidence before another training run | Whole KDM acceptance, population noninferiority, paper defect, or improved method |

| Inference status | Finding |
|---|---|
| Hard veto screen | None in the completed diagnostic |
| Statistically supported ranking | None attempted |
| Descriptive-only differences | Tiny paired arithmetic differences and saved-model NLLs |
| Default-readiness | Unchanged; diagnostic evidence does not promote a default |
| Next evidence needed | Native pretraining cloud-law coverage, then a concrete inventory of remaining downstream replication checks |

Post-run red team: shared source clouds deliberately remove RNG differences, so
this pass cannot validate native cloud generation itself. Sparse tested states
cannot prove all gradients correct. A source/TF mismatch under identical native
base draws or an incorrect composed distribution would overturn the current
no-defect finding for that boundary. No full training repetition is justified
solely by a desire for identical independently generated trajectories.

## Execution and accounting

Plan: controlled-smc-kdm-pretraining-boundaries-plan-20261013.md.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B
docs/benchmarks/run_kdm_pretraining_boundaries.py.
Artifacts: artifacts/controlled-smc-kdm-replication-20260928/
pretraining-boundaries-20261013/attempt01/.
The manifest, controller/result.json, diagnostic/result.json, compressed paired
source/TF records and terminal-review.json preserve commands, hashes and results.
Cost:74.797689734 CPU wall seconds /70.825339633 GPU wall seconds; charged once.
Remaining active allocation:205607.526882 CPU /77178.166648 GPU seconds.
No run is active and no conditional additional48-hour allocation was assumed.
