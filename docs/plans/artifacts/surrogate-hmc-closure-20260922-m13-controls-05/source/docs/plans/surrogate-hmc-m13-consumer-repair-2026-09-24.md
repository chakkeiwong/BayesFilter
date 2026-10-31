# M13 repaired-default replay — 2026-09-24

Current interpretation, after the owner's materiality decision: the
saved-run difference is an explanatory warning, not a continuation blocker.
The saved replay passes the mandatory engineering checks under the
[implemented policy](surrogate-hmc-fp32-warning-policy-2026-09-24.md).
This is a decision-policy change over immutable saved numerical evidence;
the original execution report and comparison values below are preserved.
The one-replay allowance does not make optional repeatability a prerequisite.
M13 scoped controls and posterior testing remain pending.

Status: approved and executed 2026-09-24. FP64 checks pass; the FP32
regression criterion still fails. The single replay allowance is used.

## Purpose and boundary

The campaign tests surrogate HMC: whether canonical analytical DCTR forces
can drive valid posterior sampling, with the exact/model and finite-target
comparisons kept separate. This stage answers only the prerequisite
engineering question: does restoring the established five scalar correction
diagnostics recover the saved FP32 finite-program values and total analytical
scores while preserving the full FP64 derivative checks?

This is part of the existing closure master. It changes neither the target,
observations, method, numerical controls, particle count, nor acceptance
thresholds. It is not a new scientific direction or an HMC run. Approval is
requested solely for one additional numerical attempt beyond the original
M13 consumer plan's explicit three-attempt limit. The repaired replay receives a separate reservation within the existing total
campaign ceilings; it cannot reuse the exhausted consumer reservation.

## Evidence contract

Baseline: the N256 FP32/T20 arm saved in
artifacts/surrogate-hmc-closure-20260922-m13-consumer-03/m13-consumer-localize/report.json,
with the same nine parameter points, observations, fixture, warm-start
controls, TensorFlow/GPU/XLA environment and source-bound model law.

The repaired default requests only the previous five scalar cap diagnostics.
Expanded diagnostics remain opt-in for the FP64 inspection arm; they are
not eligible for default promotion on FP64 evidence alone.

Primary engineering pass: every FP32 point is valid, and absolute value and
score differences divided by one plus the corresponding absolute saved
quantity are each at most 3e-5. The existing FP64 N256/T20 full analytical
score versus central-difference criterion at h=1e-5 remains 3e-5. The
h=1e-4 refinement check is preserved as explanatory evidence. All existing
determinism, trace/value/score and diagnostic-presence checks must also pass.

Continuation vetoes for this replay: non-finite or invalid finite programs,
source or fixture mismatch, missing required reports, failed derivative
parity, unavailable trusted GPU/memory-growth policy, exhausted allocation,
or failure of the one-attempt ceiling. The GPU command has no automatic
numerical retry. A test failure blocks the numerical child and remains a
bounded engineering repair.

Likelihood errors relative to the qualified nine-point importance reference,
ESS, covariance/condition diagnostics and runtime are explanatory here.
The simple adversaries are (1) the archived default finite program, which
directly detects regression; (2) independent central differences of the same
finite scalar, which detect omitted score terms; and (3) the qualified
trajectory importance likelihood, which preserves the distinction between
finite-program mechanics and likelihood accuracy. Evaluate all nine saved
noise-scale settings separately; do not hide low-noise failures in averages.
This is not a comparison or ranking of tuned inference methods.

A pass cannot establish M13 likelihood accuracy, scoped control qualification,
a full-support parameter reference, posterior agreement, surrogate-HMC
convergence, efficiency, or all-model completion. No prior or posterior
support is introduced. No tolerance is relaxed.

## Defaults and skeptical audit

| Choice | Provenance and role | Failure mode | Earliest check |
| --- | --- | --- | --- |
| N256, T20, nine saved points, fixed observations | Frozen baseline from consumer-03 | Different scope could hide a regression | Fixture equality and saved point/hash checks |
| Existing warm-start DCTR controls | Untuned hypothesis; not an admitted default | Low-noise weight collapse and large likelihood error | Preserve per-point validity, ESS and reference errors |
| Five scalar diagnostic outputs by default | Established consumer contract before expanded-output attempt | An output-graph change can alter FP32 XLA arithmetic | Saved FP32 value/score replay |
| Expanded FP64 diagnostics | Inspection only | FP64 success could be misused to promote FP32 expansion | Keep expansion opt-in and preserve failed consumer-04 evidence |
| Existing 3e-5 scaled tolerances | Frozen original parity/no-harm criteria | Loose tolerances could conceal target changes | Inspect actual per-point errors and finite differences at both steps |
| GPU/XLA, memory growth, original dtype/TF32 policy | Repository execution policy and archived baseline | CPU/eager or changed precision could give misleading agreement | Run manifest and device/memory policy checks |

Pre-run skeptical audit: the comparator is the saved same-scope finite
program, not the model reference treated as an exact finite-program oracle.
A successful replay answers numerical non-harm only. It cannot repair the
known untuned likelihood approximation. The unchanged nine points cannot
stand in for an untouched tuning holdout or full parameter support. The
entrypoint must retain the original three-attempt limits and allow only one
separately counted repaired-default replay. The command must use the current
master's subprocess budget enforcement and source snapshots.

Pre-mortem: output tensors can change XLA arithmetic even if the Python
recurrence is unchanged. Conversely, a callback or score bug can pass a
pointwise value check. The saved FP32 replay plus full FP64 derivative
checks discriminate these failures. A repeated no-harm failure is retained
as a repair finding, not evidence against surrogate HMC.

## Budget, command and artifacts

One numerical replay, at most 1,200 GPU-wall seconds and 900 worker CPU
seconds. Its required test review is capped at 240 CPU seconds; the parent
controller is capped at 120 CPU seconds. Thus the whole pass reserves at most
1,260 CPU seconds and 1,200 GPU seconds from the remaining global campaign
balance, in a separate m13-repair reservation. No additional total campaign
compute is requested. The earlier CPU-only repair omitted CPU measurement;
its six saved invocation-to-completion intervals are conservatively charged
as 6,340 CPU seconds (198.121 seconds times all 32 logical CPUs, rounded up).
This is a cost bound, not measured CPU usage. The recovered receipt is saved
in m13-consumer-repair-01/accounting.json. It makes the original consumer
reservation unavailable. Before the present command preflight, global charged
CPU was 42,150.338525 seconds, including that bound. The measured command
preflight then charged 5.925223 CPU seconds, recorded in
surrogate-hmc-closure-20260924-command-preflight-01/accounting.json. The
current charged CPU total is therefore 42,156.263748 seconds; GPU remains
33,507.6772886668 seconds; global ceilings remain 54,000 / 226,800 seconds. The master re-reads all receipts before dispatch and fails closed
if the remaining allocation cannot cover a job. Preserve all prior attempts.

Working directory:
/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917

Environment: /home/chakwong/anaconda3/envs/tftwogpu; escalated GPU access;
memory growth verified before device initialization. The master hides GPUs
for its CPU-only pytest subprocess and records that choice.

Exact proposed command:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-consumer-repair --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repair-01
```

The existing allow-list prefix is the absolute Python executable followed by
scripts/run_surrogate_hmc_program.py. Verify both the complete command and
the same command with --dry-run using codex execpolicy check; allow-list
matching does not constitute approval of the fourth numerical attempt.

Expected artifacts in the fresh output directory: top-level and worker run
manifests, source snapshots, pytest log, m13-consumer-repair/report.json,
report.json, accounting.json and result.md. Manifests preserve commit,
commands, environment, seed/data/fixture identities, source hashes, device
policy, wall time and output paths. The worker reuses the same parity
implementation and records this plan as its authority.

## Decision after the replay

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Engineering repair verified | All original parity/no-harm checks pass | None fire | Untuned M13 likelihood remains inaccurate | Continue the existing master's scoped-control/reference preparation | Posterior or surrogate-HMC success |
| Engineering repair unresolved | Any check fails | Report exact failure | Compiler-output interaction or numerical instability | Preserve failure; determine a concrete repair before any new attempt | Rejection of the research direction |

| Inference status | Interpretation |
| --- | --- |
| Hard veto screen | Report finite-state, derivative and no-harm checks first |
| Statistically supported ranking | None; deterministic repair validation |
| Descriptive-only differences | Likelihood-reference residuals, ESS and runtime |
| Default readiness | Only restoration of the existing output contract is tested |
| Next evidence needed | Scoped controls, full-support reference and posterior tests |

Afterward update the active checkpoint and consumer result. The master
remains incomplete regardless of this replay's outcome. Claude review
remains deferred by the owner.

## Execution record

The owner approved the exact command, which completed once. See
[saved result](surrogate-hmc-m13-consumer-repair-result-2026-09-24.md) for the
76-test review, per-point errors, source comparison and remaining budget.
The repair remains unresolved; the unchanged no-harm threshold is not relaxed.
No additional GPU attempt is authorized by this one-replay plan.
