# Frozen T=50 independent validation

Owner authorization: repeat the promising matched comparison on independent
observations and particle designs, with additional Zhao–Cui fits; document,
commit, integrate into main and origin/main, and synchronize sqmc-development.
The completed matched campaign is a43d0e3c3. Its results remain preserved.

## Question and evidence contract

Does the covariance-guided mixture retain likelihood and score accuracy on new
data, especially for SIR, when its calibrated settings are frozen? Test all four
models separately: P44 LGSSM, seven-component KSC, predator–prey and SIR d=18.
Use T=50, N=1008, FP64 GPU/XLA with TF32 disabled and verified memory growth.
Use data seeds 26100831 and 26100832, distinct from the previous test and from
calibration/validation seeds 26100820/21. Use eight new paired designs per data
seed, 261008301–308 and 261008401–408. Initial and process tensors must have
identical hashes across old/new arms. Extra mixture categorical uniforms are
recorded separately. No settings are selected using these data or references.

Freeze the exact four tuning.json files from
docs/plans/artifacts/ledh-matched-comparison-20261008-01/*-tuning/ and all old/new
controls from that run; verify their scope, validity and SHA-256 at launch.
This is a test of transfer to independent observations within the same model,
horizon and data-generating regime, not retuning or default promotion.

Primary measurements: raw log likelihood, every score coordinate, absolute
likelihood error and per-design Euclidean score error, with paired new-minus-old
errors. Report the error of the averaged score separately; cancellation must
not masquerade as individual-run accuracy. Report paired t intervals within
each dataset (df=7), and dataset-level paired summaries (only two independent
datasets, df=1). Numerical reference uncertainty remains separate from those
intervals; they cannot certify a ranking when reference bias is unbounded.

References: exact Kalman and the previously checked refined KSC grid; nonlinear
bootstrap/Fisher N=1008,32768,131072 with four independent replications; and
two independent rank-20 Zhao–Cui fits per nonlinear dataset, fit seeds 41/53,
smoothing seeds 3041/3053, 100000 paths each. Reuse the current-target linear
author-code adapter, including the existing PP tail precision repair. Quadratic
score fits use 128 training and 64 heldout parameter points and two predeclared
radii: PP .000625/.0003125; SIR .000625/.00015625. Report both radii and fit
replications; the smaller radius is the predeclared displayed estimate.
The quadratic design seed is 43017. Save source and observation identities.

The author route is not an exact oracle or a new paper-number replication.
The existing callbacks are an extension for the fixed current models. Source
anchors remain Zhao–Cui JMLR 2024, Eq.26/Algorithm 4 and the pinned author
models/full_sol.m:139–206 (commit 80034dccb99eb1d86284a1839b4a12067d13b9da).
Use logmeanexp of raw importance weights for the likelihood; mean(log weight)
returned by the author's diagnostic is a different quantity. Preserve both.

## Intent, assumptions and skeptical audit

| Item | Role / provenance | Failure and early check |
|---|---|---|
| New mixture versus frozen marginal/ancestor LEDH | Main mechanism and comparator, previous matched run | A favorable single dataset; test both predeclared new datasets |
| Old covariance-only reset, same-N nonlinear bootstrap, exact/refined reference | Constructed simple adversaries: remove high moments, remove flow, solve tractable model | Headline any conditional loss; do not optimize against them |
| Fixed beta and 16 flow / 40 reset steps | Frozen hypotheses, not universal defaults | Check exact tuning scope; report dataset-specific failures without retuning |
| Dual-cap old controls and covariance ridges | Existing numerics retained for comparability | Keep their health/validity diagnostics; a new protection requires a separate evaluation |
| Two datasets / eight designs | Bounded replication choice | Weak across-data inference; no broad ranking/default claim |
| Rank20, 100000 paths, regression radii | Existing reference settings; approximation hypotheses | Two fits, two radii, heldout residuals, path ESS and bootstrap ladder |
| FP64 GPU/XLA | Explicit reference exception to production FP32/TF32 | Trusted GPU preflight, dtype and memory policy in manifests |

Evaluate each model and each dataset separately. The data-generating regimes
remain those of the existing simulations; this does not cover arbitrary tails,
parameter values, horizons or SIR-specific tuning. The heuristic set is a
falsification check, not a tuning objective.

Promotion vetoes: inaccurate candidate relative to the stated references/simple
comparators, invalid values/scores, incomplete trajectories, or unresolved
reference disagreement. These reject promotion, not the research direction.
Continuation vetoes: wrong target/data, changed frozen settings, corrupt or
overwritten evidence, or exhausted budget. A local harness/resource failure is
a repair trigger within budget. High error with otherwise valid computation
is evidence and does not authorize skipping later planned datasets.
ESS, runtime, regression residuals and derivative consistency explain outcomes;
they do not replace likelihood/score agreement.

Pre-mortem: unchanged seeds alone do not prove matched inputs; compare hashes.
Two datasets times eight designs are not sixteen independent datasets. Reference
fits can agree because of shared rank/support bias; bootstrap and radius checks
remain necessary. A finite derivative can belong to a poor finite filter.

Skeptical review passed before implementation: same target, horizon, particles,
hardware and observations across methods; no post-result tuning or selection;
old evidence is not relabeled as independent. Additional reference-fit/radius
uncertainty is explicitly tested. No pfor or alternate numerical implementation
is introduced. The narrow master program calls existing shared authorities.

## Execution and limits

Master: docs/benchmarks/run_ledh_independent_validation.py, commands prepare,
run, status and report. Bounded workers: one GPU queue and at most two CPU
reference queues; CPU reference jobs intentionally hide GPUs. No model agents,
package changes, external messages or default changes are required.
Total budget: 24 aggregate worker-hours, at most 40 attempts, maximum 12 hours
wall time; SIR fits cap at four hours each, PP fits at 30 minutes, quadratic
scores at 90 minutes, GPU workers at 20 minutes. Timeouts/failures are retained;
localized retries consume the same budget. Stop any child at the remaining cap.

Versioned output root:
docs/plans/artifacts/ledh-independent-validation-20261008-01/.
Record git commit, source/tuning hashes, exact argv, interpreter, CPU/GPU status,
memory policy, seeds, observations, wall times, attempts and all result paths.
Result: docs/benchmarks/ledh-independent-validation-results-20261008.md.
Checkpoint: docs/reset-memos/ledh-independent-validation-20261008.md.

After execution, inspect numerical validity first, then raw comparisons,
reference uncertainty and paired errors. Include decision/inference tables and
the strongest alternative explanation. Update the master-program summary and
reset memos. Commit all task changes on sqmc-development, merge into the clean
main worktree, fetch/merge origin/main, resolve conflicts with focused checks,
push main, merge main back and verify all three refs and clean worktrees.

## Pre-launch checks

The independent-data loader, frozen-input checks, shared proposal tests and
nonlinear reference harness passed 36 focused CPU-only tests in 22.82 seconds.
Command, environment and full log are preserved under
docs/plans/artifacts/ledh-independent-validation-checks-20261008-01/.
Paper Eq.26/Algorithm 4 and author full_sol.m:139–206 were inspected directly;
the source distinction between mean log weights and logmeanexp is retained.

The eight datasets were prepared successfully at ae4d41ba2 using CUDA index 1
(RTX 5080, as recorded by TensorFlow). CUDA indices differ from nvidia-smi
indices here. Worker execution selects the idle RTX 4080 SUPER by UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b; every attempt records that selection.
This resource-isolation repair changes neither data nor frozen controls.
