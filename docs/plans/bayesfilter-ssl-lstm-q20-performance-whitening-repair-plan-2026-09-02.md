# SSL-LSTM q=20 performance and transport-score repair plan

Date: 2026-09-02  
Status: `COMPLETED_E2_NO_SAFE_BATCH_REPAIR`  
Governing master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

## Purpose and boundary

The governing master is terminal at the M3 canary resource veto.  This plan is
a new, bounded continuation investigation; it does not silently retry M3, use
partial M3 calls as tuning evidence, open Phase 9B, or make a posterior claim.
It asks two narrower questions:

1. Which execution changes can reduce the serial Phase 9A cost while retaining
   the same target, kernel, candidate pairs, seeds, and target-call accounting?
2. Why are the fresh reverse-KL charts far from the Gaussian pullback score
   identity: a diagnostic/score mismatch, insufficient optimization or
   capacity, or a limitation of the objective/chart?

The plan is diagnostic and repair-oriented.  A passing result permits a future
performance or Phase 9A plan; it does not authorize a retained HMC campaign.

## Research-intent ledger

| Item | Definition |
|---|---|
| Main question | Can a batch-native execution design and a validated score diagnostic separate the current resource failure from the unresolved transport-quality failure? |
| Baseline | Existing `FixedTransportReusableRunnerPool` and existing independent reverse-KL trainer, with the frozen q=20 target signature and C5 K=2 protocol. |
| Candidate mechanism | Group independent candidate evaluations into TensorFlow batches where the HMC kernel permits it; retain one compiled graph per static contract; use an affine oracle and value/score finite-difference checks before interpreting q=20 residuals. |
| Primary performance criterion | A proposed batch execution must preserve candidate semantics and pass numerical-equivalence checks; speed is measured only after equivalence. No minimum speedup is promoted as a scientific result. |
| Primary transport criterion | The affine oracle and q=20 value/score parity must be finite and internally consistent. A fresh trajectory may only classify residual behavior; it cannot promote whitening. |
| Hard vetoes | Wrong target/bridge signature, non-finite value or score, failed memory-growth policy, forbidden pfor/row-mapped target, changed seed/target/kernel identity, artifact collision, or failed analytic equivalence. |
| Continuation vetoes | Required artifacts cannot be made durable; analytic identity fails after focused repair; GPU launch violates the memory policy; or the bounded diagnostic budget is exhausted. |
| Repair triggers | Batch path is slower, trace count grows, finite-difference error is unstable, fresh training is seed-sensitive, or score residuals do not improve. These trigger a narrower repair or a new hypothesis; they do not establish that the research direction is impossible. |
| Explanatory-only quantities | Acceptance, short-run R-hat/ESS, training loss, residual RMS, compiler warnings, wall time, and allocator readings. None is a standalone promotion criterion. |
| Must not conclude | No IID-Gaussian whitening, exhaustive mode discovery, posterior correctness, convergence, sampler superiority, high-dimensional scaling, production readiness, or default readiness. |

## Evidence contract

Every execution phase writes a fresh directory below
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/`
and records the command, Git state, Python/ TensorFlow versions, device,
memory-growth receipt, seeds, target/bridge signatures, timings, source hashes,
and failure classification.  Existing M0--M3 artifacts are read-only inputs;
the new diagnostic never overwrites them.

The target is the frozen q=20 bridge with signature
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, strict
`tensorflow_eigh_strict`, float64, TensorFlow/TFP, XLA enabled, and GPU as the
default execution target.  CPU-hidden execution is permitted only for the
analytic fixture and static checks.  `TF_FORCE_GPU_ALLOW_GROWTH=true` must be
set before TensorFlow import and verified on every visible physical GPU.

The score diagnostic evaluates

\[
 r_s(z) = J_T(z)^\mathsf{T}\nabla_\theta\log \pi_\beta(T(z))
          + \nabla_z\log|\det J_T(z)| + z.
\]

For an exact Gaussianizing chart, `r_s` is zero.  The log-density residual is
also centered before its RMS is reported.  These are finite-bank diagnostics;
they do not prove a global density identity.

## Defaults and assumption audit

| Choice | Provenance | Failure mode | Earliest check | Status |
|---|---|---|---|---|
| One visible GPU, on-demand allocation | Repository owner policy and prior successful q=20 mechanics runs | Contention or allocator growth can hide the execution cost | Pre-import environment and post-init growth receipt | Reviewed execution baseline |
| Batch sizes 8, 16, 32 for target timing | Static-shape contracts already used by the bridge; sizes are diagnostic hypotheses | A larger batch may increase peak memory or trigger a separate compile | Memory telemetry and trace-count receipt | Measured hypothesis, not a default |
| Central-difference steps `1e-3`, `1e-4`, `1e-5` | Numerical localization convention | Cancellation or target noise can make the estimate unstable | Step-size convergence table and finite/status checks | Diagnostic only |
| Fresh chart: `(16,16)` tanh, two stages, eight updates | C5 compact-high family; eight updates keeps this a localization run | Too few updates can mimic an optimization failure | Initial/final held-out residuals and update validity | Fresh diagnostic baseline |
| Candidate batching grouped by fixed leapfrog length | HMC calls with different `L` cannot share one scalar loop budget; grouping preserves kernel semantics | Broadcasting step size or seeds incorrectly can couple chains | Analytic sample/acceptance equivalence and shape checks | Prototype hypothesis |
| No pfor, `tf.map_fn`, or scalar target loop | Repository TensorFlow policy | A convenience vectorization could violate the route contract | Static source scan | Hard requirement |

## Phase protocol

Every phase is `E_k -> R_k`: execute, preserve the attempt, classify the
result, run the smallest regression, and refresh the next phase.  A local
failure is repaired without changing the target or scientific contract when
possible.  A new performance design or budget is required before any future
Phase 9A replay.

### E0/R0: evidence and source audit

Read the terminal M3 result, localized M2 result, C2 score diagnostics, and
current source.  Confirm that the old replay is not being resumed, that the
active route has no explicit CPU worker pool, and that the HMC runner evidence
shows reuse.  Run focused policy/mechanics tests, compilation, and
`git diff --check`.

**Exit:** an immutable audit receipt and a fresh output namespace.  A stale or
corrupt prerequisite is a continuation veto.

### E1/R1: performance and batch-feasibility diagnostic

Use the new diagnostic harness to measure, after warm-up, target value/score
evaluation for equal total rows in serial static batches and in one larger
batch.  Record first-trace and steady-state time separately.  Run the analytic
HMC fixture with independent seeds and compare the existing scalar-candidate
route with a prototype that groups candidates by fixed `L` and broadcasts
step sizes over an independent chain axis.  The prototype is diagnostic until
its samples, accept/reject states, target-call count, and trace schema agree.

**Exit:** a cost decomposition and an equivalence decision.  If equivalence
cannot be established, do not integrate the prototype; retain the timing only
as diagnostic evidence.

### E2/R2: score-authority diagnostic

First run the exact affine Gaussian fixture and require near-roundoff
pullback residuals.  Then evaluate the q=20 bridge score against central
finite differences of the same value program at fixed points and three step
sizes.  Finally build one fresh chart, train eight batch-native updates at
`beta=0.5`, and compare initial/final held-out pullback residuals on a fresh
bank.  This trajectory is not a tuning handoff and is not reused by Phase 9A.

**Exit:** classify the residual as (a) diagnostic/score inconsistency, (b)
optimization/capacity evidence, or (c) unresolved objective/chart behavior.
An unstable finite-difference check is a hard diagnostic veto until localized.

### E3/R3: smallest justified code repair

Only if E1 identifies a safe equivalent path, implement the smallest core
change: preferably a batch-native candidate evaluator grouped by fixed `L`,
with explicit static signatures and no pfor.  Only if profiling shows trainer
construction is material, consider a functional shared training kernel; do not
change optimizer reset semantics or checkpoint identities implicitly.  Add
focused tests for shape, seed independence, target-call accounting, trace
counts, and numerical equivalence.

If E2 points to under-training or capacity, do not silently change the active
default.  Write a separate target-specific training ladder with disjoint
validation data and a declared budget.  If the score authority is wrong, repair
that boundary before any new training.

**Exit:** tests and equivalence receipt pass, or the proposed repair is rejected
as unsafe and the reason is recorded.

### E4/R4: closeout and next-plan refresh

Write a result note with decision and inference-status tables, uncertainty
limits, a post-run red-team paragraph, and exact next action.  Refresh a future
performance/Phase 9A plan only when its changed schedule, budget, and evidence
contract are explicit.  Phase 9B remains closed in all outcomes of this plan.

## Pre-mortem

The run could appear successful while still misleading us if batch timing is
compared before compilation is amortized, if a batched RNG stream changes the
candidate distribution, or if a lower score residual on eight updates is
mistaken for whitening.  The harness therefore separates first-trace and
steady-state time, records every seed and target call, and labels residuals
diagnostic-only.

It could fail for infrastructure rather than science if the custom strict
backend is unavailable, GPU memory growth is applied too late, or the output
directory already exists.  These are launch/artifact failures and must not be
interpreted as transport failures.  It could fail scientifically because the
value program and analytic score disagree or because the chart objective is
mode-seeking; the affine and finite-difference stages distinguish those cases
before any expensive run.

## Skeptical plan audit

Audit performed before execution on 2026-09-02.

| Audit question | Finding and repair |
|---|---|
| Wrong baseline or stale context? | The terminal M3 schedule is explicitly rejected; all new diagnostics use fresh output/seed namespaces and treat old artifacts as read-only context. |
| Proxy promoted as a gate? | Acceptance, short R-hat/ESS, loss, and residual RMS are explicitly explanatory. Equivalence, finite/status, and identity checks are the only hard execution gates here. |
| Missing stop condition? | Per-phase budgets, artifact-collision failure, memory/XLA failure, unstable score parity, and exhausted diagnostic budget are stated continuation vetoes. |
| Hidden batching assumption? | Candidate groups are separated by fixed `L`; seed independence, target-call accounting, shape, and analytic sample/acceptance equivalence are required before integration. |
| XLA/pfor mismatch? | Pure TensorFlow kernels may be JIT compiled; Python orchestration remains host code. The route scan rejects pfor, `map_fn`, and row-mapped scalar targets. |
| Unfair data use? | Fresh diagnostic banks are disjoint from prior tuning and claim banks. No result is eligible for Phase 9A selection. |
| Resource realism? | The old 1,800-second cap is not widened. E1 measures a smaller diagnostic first; any changed schedule or budget requires a new plan. |
| Scientific overclaim? | The plan states that neither a passing affine check nor a lower fresh residual establishes whitening, mode discovery, convergence, or scalability. |

**Audit verdict:** `PASS_WITH_BOUNDED_SCOPE`.  The plan is safe to execute
through E0--E2.  E3 is conditional on an equivalence receipt; no conditional
code repair or Phase 9B launch is pre-authorized by a proxy result.

## Execution commands and budget

The initial bounded budget is 1,200 seconds of GPU wall time and 300 seconds
of CPU-only test time.  The diagnostic launcher is required to create a new
attempt directory and fail closed on collisions.  The intended GPU command is:

```text
BAYESFILTER_PERF_WHITENING_ATTEMPT_ID=attempt-01 \
bash scripts/run_ssl_lstm_q20_performance_whitening_diagnostic_gpu.sh \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-01-gpu
```

The CPU analytic command is:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_performance_whitening_diagnostic_2026_09_02.py \
  --cpu-analytic-only \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-01-cpu
```

The GPU launcher must set memory growth before importing TensorFlow, expose
one selected GPU, and record the trusted managed-session basis.  No package
installation, network access, destructive Git action, or external message is
part of this plan.

## Closeout artifact requirements

The result note must include the command actually run, a run manifest, source
hashes, phase statuses, timing/trace tables, finite-difference error tables,
the affine-oracle result, fresh-chart initial/final residuals, a decision table,
an inference-status table, and the strongest alternative explanation.  A
future plan must state exactly which performance design or budget changed; it
may not simply retry the terminal M3 command.

## Execution ledger and repair/refresh (2026-09-02)

| Phase | Outcome | Repair or refresh action |
|---|---|---|
| E0/R0 | Completed. The terminal M3 result was treated as read-only; the active route scan found no `tf.map_fn`, `tf.vectorized_map`, Jacobian-pfor, or explicit CPU worker pool. Focused policy/mechanics tests passed (`92 passed`, 40.36 s). | A first CPU harness attempt exposed an import-path defect; a second exposed repeated grouped-fixture graph construction. The harness was repaired by adding the repository root to `sys.path` and caching the grouped graph by row count. The repaired CPU analytic run passed and was preserved as `attempt-03-cpu`; the final smoke is `attempt-04-cpu`. |
| E1/R1 | Completed. GPU target batching was finite and each requested static batch size traced once. The grouped HMC prototype was finite, but its random streams and candidate partition were not equivalent to the serial route; `integration_allowed=false`. | The grouped prototype was deliberately not integrated. The measured target batching opportunity is carried forward as a hypothesis, not as a default. |
| E2/R2 | Completed. The affine oracle was at roundoff, q=20 central differences were stable at practical steps, and the fresh chart remained far from the Gaussian pullback identity after eight updates. | The score-authority boundary is not the immediate repair target. A target-specific optimization/capacity ladder is required before any whitening claim. |
| E3/R3 | Not entered: the E1 equivalence receipt did not pass. | No active sampler or optimizer semantics were changed. |
| E4/R4 | Completed in `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-result-2026-09-02.md`. | The refreshed next plan is `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`; Phase 9B and the old M3 replay remain closed. |

The bounded campaign used 131.40 GPU seconds and 4.59 CPU seconds for the
analytic-only receipt, within the stated budget. The GPU run was launched with
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and recorded one
visible GPU. The TensorFlow warning about retracing arose while deliberately
switching among a finite set of static batch-size specializations; it is not a
proof that the entire Python program is XLA compiled, nor evidence of an
unbounded retracing loop. Python orchestration, file I/O, validation, and
candidate loops remain host-side by design.
