# SIR runtime remeasurement with 1008 particles

23 October 2026. Branch `ledh-refactor-with-policy-fix`, HEAD
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649` plus preserved working changes.

## Invalidation and question

Owner directive: disqualify all previous results in this controlled-SMC/KDM/LEDH
investigation based on 64 particles. They must not establish runtime, scaling,
training efficacy, moment or score improvement, algorithm rejection, or readiness.
Preserve their files for provenance; do not pool or baseline their measurements,
reuse learned checkpoints, or transfer their conclusions. Source code and test
fixtures may be inspected, but any decision requires fresh eligible evidence.
The minimum particle count for this investigation is now 1008. This notice does
not invalidate unrelated investigations already using larger particle counts.

Question: at N=1008, how do the pre-merge, checked ungrouped repair, and current
grouped implementations compare on the actual plain, conditional, and physical
mixture SIR consumer? Measure all three anew; the old timing ratios are withdrawn.
These consumers have guide=None. Conditional means the current-observation
Gaussian proposal, not a trained backward KDM or controlled-SMC guide.

## Evidence contract and research intent

- Mechanism: larger particle batches change GPU occupancy, linear algebra
  batching, and the relative cost of quadratic OT versus per-particle checks.
- Comparators: archived pre-merge source at 4d6915cc; checked repair snapshot
  `ledh-sir-merge-repair-20261022/candidate-02-final-source`; fresh current source
  snapshot. Archived numerical results and weights are not consumed.
- Primary engineering criterion: finite valid execution with stable signatures
  and matched fresh inputs; report per-arm synchronized warm latency and paired
  current/comparator ratios. A speed improvement requires a paired block
  bootstrap 95% interval wholly below one AND numerical/flag parity. Non-harm
  requires its upper limit <=1.05. No pooled overall superiority claim.
- Numerical veto: value, directional analytical score, first/raw second moments
  differ by more than 1e-8 absolute + 1e-8 relative; common validity flags differ;
  or either endpoint rejects. Report failed comparisons without a speed ranking.
  Current/checked-repair must also preserve all common numeric diagnostics except
  the intentionally changed determinant grouping metadata.
- Continuation veto: wrong branch/source/input, invalid current endpoint,
  source drift, GPU/memory-policy failure, unexpected retracing, OOM, or budget
  exhaustion. A comparator parity failure vetoes only that comparison, not the
  remaining valid comparator. Local harness failures may be repaired and retried.
- Explanatory diagnostics: compile/first-call cost, memory peak, per-dataset
  ratios, actual grouping sizes, individual cold/warm latencies, source hashes.
- Scope: single cloud, SIR d18, T5, theta0, N1008, FP64 with TF32 disabled,
  GPU/XLA. FP64 is an explicit diagnostic exception isolating particle count;
  this does not test FP32/TF32 production speed, long horizons, native parameter
  batches, trained-guide efficacy, particle convergence, or score accuracy.
- Heuristic adversaries for this engineering question: plain flow (no physical
  proposal), conditional-only proposal (beta1), physical mixture (beta0.9).
  Evaluate each separately for zero and nonzero score directions and each fresh
  dataset. They are workload controls, not statistical accuracy baselines.

## Defaults, preparation, and skeptical review

N1008 is the owner's minimum, not a convergence finding. Chunk policy must return
K=N=1008. Keep OT epsilon1/Sinkhorn5, 32 terminal balances and the existing
physical controls fixed for this engineering comparison. No claim or tuning
artifact is issued for the changed scope. Existing protection constants are
comparability hypotheses: reject nonfinite/invalid paths and record their flags;
this run does not justify them as scientific defaults.

Generate two fresh SIR datasets (seeds 263103001/263103002), 256 calibration paths
(seed263103010), reset design (263103020), and particle streams
(263103031/263103032). CPU-only generation is a small reference preparation;
GPU workers create their paired particle inputs and verify serialized hashes.
No previous loss, moment, score, timing or trained weight is read.

Use the existing `make_moment_endpoint` and shared analytical LEDH kernel with
stable signatures. Both zero and e0 directions run the score recurrence. Three
resident isolated workers execute serial requests; rotate source and arm order.
Warm each case, then freeze 16 paired blocks. Synchronize every output before
stopping its timer. Record serialization separately. Intervals resample complete
blocks (4096 fixed-seed replicates); individual datasets and pooled-within-arm
summaries remain visible. Intervals describe timing uncertainty on this machine,
not stochastic efficacy or cross-machine generalization.

Skeptical audit PASS: the prior wrong baseline was the small workload, now
removed; current/checked-repair parity is tested separately from known possible
pre-merge drift. No old measurement or trained selection enters. Changes in
normalization use fresh calibration shared by all sources. All observed consumer
checks remain live. Source batching is not mislabeled as native parameter-batch
execution. Fresh compile cost and shared-device occupancy can otherwise mislead;
exclude compilation from warm timing and record device/process state.

Preflight found both GPUs occupied by unrelated user jobs. Do not interrupt them.
The benchmark uses the same RTX4080SUPER throughout with alternating serial
requests and per-block device-utilization records. Any interval is conditional
on scheduling during this run; even a passing comparison does not explain another
agent's different dtype, horizon or parameter-batch benchmark.

The actual GPU launch occurred after preflight: its own device snapshots show
the target GPU idle with 11MiB allocated at both run boundaries. Thus the earlier
busy preflight does not establish concurrent workload during the measurement.
Process ownership was not polled during timing. Report that uncertainty directly;
do not assert actual contention from the earlier preflight. A post-run reporting
repair removes the hardcoded shared-device assertion from the analysis, preserving
the original executed driver, raw result and original analysis. Numerical rows
and timing estimates are unchanged.

## Execution and budget

Artifacts: `docs/plans/artifacts/ledh-sir-n1008-runtime-20261023/`, unique
`attempt-01` (and at most two localized repair attempts). Total cap 45 minutes
elapsed across attempts, with at most three resident GPU workers (135 minutes
summed worker lifetimes); only one worker receives a numerical request at a time.
No learned training, tuning, package change, or algorithm repair is included.
Existing exhausted/remaining budgets from the invalidated comparison are not
used to justify additional work; this is the newly requested bounded measurement.

Commands use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` and
`scripts/benchmark_sir_n1008_runtime.py`; preparation sets CUDA_VISIBLE_DEVICES=-1.
GPU launch requires escalated access, memory growth verified before initialization,
and `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (existing WSL compatibility
exception, identical for all sources). Exact commands, source and executable
hashes, memory policy, device identity, seeds, wall time and outputs go in manifests.
Focused harness tests precede execution. Results will include numerical and
timing verdicts, inference-status/decision tables, uncertainty and next action.

Preparation repair: the first CPU-only import in tf-gpu failed because the local
Sylvester extension was built against the self-built TensorFlow in tftwogpu.
No inputs or measurements were produced. Use tftwogpu (the environment bound
to these source snapshots); do not rebuild packages or change the extension.
Retry in fresh `run-02` under the artifact root. The saved error log preserves
the failed setup attempt. This is a local environment-selection repair.
