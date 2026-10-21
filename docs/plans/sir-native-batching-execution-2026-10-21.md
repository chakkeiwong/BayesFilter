# Native analytical-score batching: implementation and execution

## Checkpoint

2026-10-21, branch `ledh-graceful-failure-execution-20260917`, baseline commit
`f6bb90a00`. Native B/all-P model, proposal, reset and correction algebra,
packed pair contractions, and optional native-batched sharding are implemented.
The old/new full-horizon FP64 scores agree within 4.35e-11 relative L2.
Uniform FP32 remains unpromoted: maximum old/new score difference is 2.274%,
and the worst row differs from the FP64 recurrence by 5.769% (old) and 8.057%
(native), exceeding the unchanged 1% criterion. FP64 is a diagnostic reference
for this finite filter, not an exact model-likelihood oracle or a new default.

GPU B4/8/16/32 complete. B64 completes two evaluations within about 8 GiB
but reaches the repeat-budget cap. B128 exceeds the 12 GiB allocator cap;
logical B128/native S4 completes in 333.77 seconds including compilation,
uses 0.761 GiB and matches repeated B4 values/scores exactly. CPU B4 completes
two calls, B8 one, and B16/B32 none before their 600-second caps. All arms
have ended. The isolated source-tree regression passes 119 tests. No HMC or
NeuTra optimizer update has been run. Training adapter/whole-update parity
remains in the parent integration plan.

This bounded batching execution is complete, with numerical promotion withheld.
The result and next repair are recorded in
[the result note](sir-native-batching-results-2026-10-21.md). The next research
action is target-preserving FP32 score localization within the parent plan,
followed by matched full-horizon remeasurement; it has not been launched here.
Recorded GPU arms use 45.89 minutes; CPU arms and logged tests use 57.83 minutes,
both below the two-hour ceilings. The execution history,
review, failed attempts and numerical evidence are retained under
`artifacts/sir-native-batching-20261021-01/` and in
[sir-native-batching-implementation-review-2026-10-21.md](sir-native-batching-implementation-review-2026-10-21.md).

## Question and evidence contract

Can the existing finite covariance-proposal filter and its total analytical
score evaluate independent parameter rows without duplicating the model body
at trace time or constructing pair-by-state/sensitivity temporaries? Preserve
the mathematical target, fixed random design, and failed-row isolation.
Baseline is the executed source snapshot under
`artifacts/sir-fp32-batch-capacity-20261020-02/gpu-b4/source/`, with matching
data and inputs. Scalar directional and dense FP64 computations are independent
diagnostics, not replacements for the analytical production score.

Promotion requires executable multirow parity, constant graph structure across
batch sizes, bounded compilation, and completed full-horizon measurements under
resource limits. Likelihood difference is scaled by max(1, abs(reference));
score difference uses vector L2 divided by max(1, reference L2), each <=1%.
Primitive FP64 checks use 1e-8 relative/absolute, FP32 1e-4 with documented
scale; exact status/steps, row permutation and isolation must agree. Near-zero
component percentages cannot veto or promote. A precision mismatch above 1%
blocks numerical promotion and triggers localization, not abandonment.

Explanatory diagnostics: compilation/warm latency, allocator/RSS peaks, graph
count, CPU/GPU discrepancy, full per-parameter scores. No posterior correctness,
convergence, trained NeuTra quality, or all-model completion follows from a
capacity smoke. A failed candidate proceeds to its planned localized repair.
Stop only for broken data/target/artifact, missing diagnostics, unsafe resource
use, or the budget below. Preserve every attempt in a new output directory.

## Implementation sequence

1. Native model/proposal/reset algebra across B and all parameter directions;
   compact shared matrices; no Python loop over B or scalar target map. Retain
   the direct contraction while checking this change in isolation. Cover initial
   state, UKF, flow, all covariance partials, density and optional correction.
2. Remove pair-by-state and pair-by-sensitivity temporaries using centered
   matrix products and packed weighted moments. Test each algebraic identity
   against the direct FP64 calculation, including large translations, tangent
   offsets and finite Sinkhorn derivatives. Keep the exact N/K chunk policy.
3. Exercise the actual fixed-design callback, row permutations, different and
   duplicated parameters, invalid-row isolation, and all existing model adapters.
   Measure a small complete batch first; add batched sharding only if measured
   memory requires it. A training adapter/update is not yet implemented in the
   parent integration plan: do not claim training readiness from filter tests.
4. Repeat bounded FP32 CPU/GPU capacity checks from the existing supervisor,
   preserving full outputs, synchronized timings and both clocks. Start B4/8;
   advance only after numerical correctness. Larger B32/64/128 timeouts/OOMs
   are capacity results, not grounds to change the mathematical target.

## Defaults and assumption audit

| Choice | Provenance/status | Failure mode and first check |
|---|---|---|
| FP32, TF32 off, N4032/T50 | Owner workload; baseline | CPU/GPU order sensitivity; matched value/score |
| K2016, exact tiles | Repository policy | Changed finite program; selector/wiring tests |
| flow16, reset5, epsilon1, correction0 | Frozen workload, not fresh tuning admission | Scope transfer; no scientific/default promotion |
| SIR source half-step k4 | Existing source law | Accidental model repair; transition/tangent parity |
| Same fixed uniforms and beta | Prior design | FP32 ancestor boundary drift; reuse prepared indices |
| Compact shared matrices | Structural model property | Invalid sharing for nonlinear H; nonshared tests |
| Matrix distance identities | Candidate, not assumed accurate | Cancellation; centered and translated fixtures |
| Batch >1 / all-P recurrence | Owner training requirement | Hidden row map or pfor; graph and call-chain tests |

Cheap engineering adversaries are the scalar reference, the old batched kernel,
and repeated small native batches. Evaluate identical/distinct rows, one invalid
row, small/large state scale and particle counts separately. These are
implementation comparators, not competing statistical estimators; filter
accuracy against Kalman/oracles is outside this unchanged-target optimization.

## Budget, environment, commands and artifacts

Total ceiling: 2 GPU device-hours and 2 CPU wall-hours, at most three localized
repair attempts per failing phase. No full HMC or NeuTra training is authorized
by this plan. Tests/short diagnostics first; long capacity ladder within the
remaining ceiling, at most 600 seconds per arm. Reuse supervisor host 24GiB
RSS / 6GiB available-RAM guards and GPU logical 12GiB limit exception.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; CPU tests via
`env CUDA_VISIBLE_DEVICES=-1 ... -m pytest <focused tests>`. GPU runs use trusted
permissions, explicit UUID from the baseline, and verified memory policy.
Exact executed commands, commit and dirty-source hashes, seeds, hardware, times
and results are recorded under `artifacts/sir-native-batching-20261021-01/`.
Benchmark invocation uses `scripts/benchmark_sir_fp32_batch_capacity.py run`
with a new output root, selected arms and total seconds explicitly specified.
Do not launch the full ladder until early checks pass.

## Skeptical review before execution

The old CPU score is not an oracle: its 8.274% GPU discrepancy remains an open
precision diagnostic. Graph shrinkage cannot establish memory fit; filter fit
cannot establish a complete training-step fit. All parameter-dependent covariance
and normalization terms must remain in the score. Reducing validity over B would
silently couple independent rows. Scalar mapping, omitted optional corrections,
or a SIR-only numerical filter fork would fail the intended implementation claim.
Keep model laws in adapters and one shared filtering authority. These risks are
addressed by separate parity, graph, precision and capacity phases. Plan passes
for bounded engineering execution; scientific admission and training remain
separate. Record failures and next repairs here instead of enlarging scope.
