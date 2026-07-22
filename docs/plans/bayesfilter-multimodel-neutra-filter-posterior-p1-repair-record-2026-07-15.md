# P1 Repair Record

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

## Attempt 01: Script-Path Import Failure

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-01-20260715T093516Z/`

Classification: `INFRASTRUCTURE_LAUNCH_PATH`.

Observed failure: the script created its fresh output root, imported TensorFlow,
then failed with `ModuleNotFoundError: No module named 'bayesfilter'` before GPU
configuration, target construction, training, HMC, or evidence emission. Python
sets `sys.path[0]` to `docs/benchmarks` when the launcher is invoked by path, so
the repository root was not importable in this environment.

Scientific contract impact: none. No target, data, method, criteria, hardware
class, privacy boundary, or budget changed. No model cell ran and no result was
produced. This consumes one P1 infrastructure attempt and negligible GPU budget.

Repair: derive the repository root from `__file__` and prepend it to `sys.path`
before BayesFilter imports. This is the same self-contained script-path pattern
used by repository benchmark launchers; it does not alter the canary target or
execution graph.

Focused regression:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m py_compile docs/benchmarks/run_multimodel_neutra_p1_canary.py
python docs/benchmarks/run_multimodel_neutra_p1_canary.py --help
```

Retry rule: preserve attempt 01 and launch attempt 02 in a fresh root. If import
still fails, stop and classify the repeated launcher defect rather than changing
the scientific contract.

## Attempt 02: Missing Transformed-Target Status Telemetry

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-02-20260715T093710Z/`

Classification: `HARNESS_INTEGRATION_STATUS_TELEMETRY`.

Observed result: the trusted RTX 4080 SUPER was visible, TensorFlow created the
GPU device, XLA compiled the batch target and training graph, and the bounded
64-step batched training completed and froze a target-bound dense-IAF artifact.
The subsequent transformed-HMC health smoke failed before sampling because
`FixedTransportValueScoreAdapter.target_status_telemetry` correctly required
the base adapter to expose `target_status_telemetry`, while the synthetic
adapter exposed the same status only through its training batch method.

Scientific contract impact: none. This is a missing integration surface, not a
target, training, sampler, GPU, or numerical failure. The attempt-02 training
artifacts remain valid engineering evidence for that attempt but are not reused
for the fresh terminal attempt.

Repair: add a graph-native `target_status_telemetry` method to the synthetic
adapter with the same all-valid status semantics as its batch target. Do not
weaken the fixed-transport wrapper or disable status tracing.

Focused regression: construct a loaded target-bound transport, build the
transformed adapter, call status telemetry on a rank-2 batch, and run the
CPU-hidden campaign integration tests. Then launch attempt 03 from a fresh root
under the unchanged 64-step/16-draw budget.

## Attempt 03 Terminal Audit: Incomplete HMC Execution-Surface Binding

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-03-20260715T094009Z/`

Classification: `IDENTITY_INTEGRATION_TERMINAL_AUDIT`.

Observed result: attempt 03 passed all declared GPU/XLA canary checks, including
64-step training, frozen transport loading, transformed-HMC health, memory
growth, and unchanged blocked model-cell states. During terminal code/result
review, the supervisor found that the typed identity bound the inspected
`neutra_batch_log_prob_and_grad_status` training surface but not the separate
`log_prob_and_grad` surface used by plain and transformed HMC. Replacing the HMC
surface after issuance would therefore not have changed the typed digest.

Verdict: attempt 03 is valid runtime evidence for its then-issued identity but
cannot close P1 because the identity contract was incomplete. No scientific
claim or model cell is affected.

Repair: bind and replay the exact adapter-bound `log_prob_and_grad` callable,
its inspectable source, and repository dependency module/callable closure in the
typed identity. Add a negative post-issuance callable-replacement test. Expand
terminal artifact hashes from top-level JSON files to every recursively emitted
file, including training checkpoints, progress, and frozen transport.

Retry rule: because the repair intentionally changes the target signature and
therefore invalidates transport identity reuse, launch attempt 04 in a fresh
root after the full CPU-hidden suite passes. Do not copy or re-sign attempt 03.

## Attempt 04 Terminal Contract-Coverage Audit

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-04-20260715T094454Z/`

Classification: `TERMINAL_CONTRACT_COVERAGE_AUDIT`.

Observed result: the strengthened typed identity and GPU/XLA canary passed. A
final subplan-coverage audit then found that the campaign training wrapper
should itself require the already-verified memory-growth record and a fresh
output directory, state transitions should optionally persist append-only
events, typed sequential HMC should be integration-tested with disjoint archive
paths, direct production-helper reuse should be rejected in recomposition, and
artifact hashes should cover every recursive file.

Repair: enforce these at the campaign boundary, add focused tests, and retain
the generic trainer/controller interfaces for existing diagnostic lanes. The
target, method, hardware, criteria, and budget remained unchanged.

## Attempt 05 Terminal Pass

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-05-20260715T095202Z/`

Classification: `TERMINAL_PASS`.

Result: all strengthened checks passed, 71 CPU-hidden compatibility tests
passed, all eleven recursively emitted artifacts replayed their SHA-256 hashes,
and bounded Claude terminal review returned `VERDICT: AGREE`. No further P1
repair is indicated.

## Attempt 06: Common Status-Identity Reopen

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-06-20260715T101223Z/`

Classification: `COMMON_STATUS_IDENTITY_REOPEN_PASS`.

P2 target-adapter design showed that transformed HMC executes
`target_status_telemetry` as a hard-veto surface. P1's typed identity bound the
batch training and HMC value/score callables but not that third callable. P1 was
reopened under its common-repair budget, bound the status source/dependency
closure, added a post-issuance status-mutation rejection test, and reran the
bounded canary. Seventy-two CPU-hidden compatibility tests and all attempt-06
recursive hashes passed. The new typed signature is
`7fd7ec3c835da2730ce80704396b32aea60371da511b81266eb19e8b4110ba53`.
