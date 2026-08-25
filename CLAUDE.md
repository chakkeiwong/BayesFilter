# BayesFilter Claude Governance

Claude Code reviewers and workers must follow `AGENTS.md`.

## Backend Rule

The repository implementation backend is TensorFlow / TensorFlow Probability.
NumPy may appear only in explicitly diagnostic code: tests, comparison
fixtures, independent reference solutions, closed-form or finite-difference
checks, and post-run diagnostic inspection. Serialization, reporting, training,
data generation, inference, tuning, candidate selection/admission, artifact
construction, and benchmark kernels are not blanket NumPy exceptions.

Do not approve a non-diagnostic NumPy import or a NumPy-backed runtime path.
TensorFlow tensors may be materialized only at host-side assertion, diagnostic,
or artifact boundaries and must not feed NumPy numerical computation. Existing
violations are migration debt, not precedent. Differentiable or gradient-bearing
paths require TensorFlow / TensorFlow Probability unless the owner explicitly
approves another autodiff backend. PyTorch and JAX are non-default and require a
reviewed exception.

## Default Execution Target

The repository default execution target is GPU.  For DPF transport work, the
default production algorithm target is the GPU-oriented LEDH-PFPF-OT TF32 route:
TensorFlow/TFP, `float32` tensors, TensorFlow TF32 execution enabled, and
streaming/chunked transport where applicable.  CPU, FP64, and FP32-no-TF32 arms
remain explicit reference, comparison, smoke, or fallback modes.

Treat this as a human owner directive, not as a scientific proof.  Do not
reopen the default-vs-experimental question without new evidence or human
instruction, and do not turn this policy into unsupported posterior
correctness, HMC readiness, statistical superiority, dense Sinkhorn equivalence,
or broad scientific-validity claims.

## LEDH Per-Scope Tuning Rule

Every claim-bearing LEDH model run requires an offline tuning artifact for the
exact model/target, route/reset family, horizon/prepared-data regime, particle
count, dimensions, dtype/backend, chunk policy, and route-specific control
family used by that run. Any changed bound field is a new tuning scope. A
setting selected for another model, route, or horizon is a warm-start candidate
only and must never be treated as a universal or inherited default.

Streaming OT routes tune their own Sinkhorn/balance controls. Contract E--TP
and other routes tune their applicable feature, basis, lookahead, chart,
ridge/KKT, or other controls; do not mislabel these as Sinkhorn tuning. Require
disjoint tuning and untouched claim partitions and an exact repository-issued
scope match before accepting a claim. A failed claim triggers fresh
scope-specific tuning with new tuning data under the campaign budget; it does
not authorize tuning on the failed claim data, threshold relaxation, or
skipping later model-specific tuning. Runtime adaptation inside HMC is
forbidden.

## DPF Transport Chunk Rule

Active DPF canonical, candidate, benchmark, leaderboard, and production-target
routes must use `dpf_transport_exact_divisor_cap3000_v1` from
`bayesfilter.highdim.transport_chunk_policy`. Row and column chunks are equal.
For `N<=3000`, the only valid chunk extent is `K=N`; for larger `N`, use the
largest divisor of `N` no greater than 3000. Reject a case with no divisor
greater than 1 and reject every caller override that differs from the selector.

Contrary historical settings are wrong and archival only. Do not approve them
as diagnostic, comparison, timing, tuning, or candidate evidence. In
particular, never promote lower-rung `K=16` fixtures into a larger particle run.
Primitive mechanics tests must also use exact `K=N` chunks. Review must verify
the central selector and active-source discovery guard rather than accepting a
matching local constant.

## TensorFlow GPU Memory Rule

TensorFlow GPU processes must not reserve the whole device eagerly by default.
Before any logical-device or GPU runtime initialization, enable and verify
memory growth on every visible physical GPU. Serious GPU runs must fail closed
if this cannot be done and must record the verified policy in their manifest;
silently ignoring `set_memory_growth` failure is not acceptable.

Memory growth is not a hard memory cap and may eventually consume most or all
available memory. If a run must reserve memory for another process, require a
reviewed logical-device `memory_limit` configuration instead. Memory growth and
logical-device limits are mutually exclusive TensorFlow configurations and the
artifact must state which mode was used. Whole-device preallocation is a
non-default reviewed exception.

## NeuTra Batch-Native Training Rule

Claim-bearing NeuTra HMC uses canonical policy
`bayesfilter_neutra_sequential_hmc_v1` and the shared TensorFlow/TFP controller
under `bayesfilter.inference.neutra_hmc`. Retain and archive warm-up while
excluding it from posterior estimates; use recent-window max(rank-normalized
split R-hat, folded rank-normalized split R-hat) readiness; grow retained draws
cumulatively under modern R-hat and declared ESS/downstream gates; and cap
warm-up and retained sampling at 10,000 per chain. Fixed discarded burn-in and
fixed terminal sampling are historical/smoke/reference exceptions only.

The versioned route ledger is mandatory. Its discovery guard must fail on an
unledgered qualifying route, stale or duplicate classification, active route
without the canonical policy binding, or active fixed-budget implementation.

All BayesFilter NeuTra optimizer updates must be batched with batch size greater
than one. The transport, log determinant, target value/score, loss, gradient,
and optimizer computation must preserve the leading batch dimension in
TensorFlow/XLA.

Do not approve a Python sample loop, scalar target replay, or `tf.map_fn`,
`tf.vectorized_map`, or `tf.while_loop` that merely maps a scalar target over
training rows as a batch-native training implementation. An eligible route uses
batch-native tensor/linear-algebra operations, or persistent multicore workers
that each evaluate a batched shard for GPU transport training. Scalar and
row-mapped routes are parity/reference diagnostics only and must not update
NeuTra parameters.

Training evidence must record batch size, batch-native target backend, device,
XLA status, and scalar-fallback/sample-loop status. Batch size one, any scalar
fallback, or any row-mapped scalar target is a hard veto for NeuTra training,
including smoke and CPU-only training. Existing violations are migration debt,
not precedent for approval.

## Configuration-Status-First Reporting Rule

Adopted 2026-08-25 after the Q3 board presented non-production,
untuned score cells beside exact references without a banner, inviting
days of debugging the wrong object.

Any artifact, report, or user-facing summary that shows benchmark or
comparison numbers must state, BEFORE any number is shown or quoted:

1. which program each cell ran — `production` or a named variant, with
   the configuration diff from production;
2. per-scope tuning status — the tuning artifact path, or UNTUNED. An
   UNTUNED or non-production cell carries no per-model claim and must
   not be presented as the production algorithm's performance;
3. what must NOT be debugged or concluded from the table.

Enforcement is artifact-layer, not narrative: leaderboard cells carry
`program` and `tuning` fields, report builders hard-refuse unlabeled
cells (G-5-style quarantine), and the estimand gate
(`test_production_score_lane_value_is_likelihood_estimand`) pins the
production score-lane program to the exact-reference anchor. Chat
summaries of results must lead with the same configuration-status
information — a plan-file caveat does not discharge this duty.

Amendment 2026-08-26 (owner finding #2, same failure class): a REQUIRED
mechanism of the production program (the dual-cap trust region — the
designed covariance-explosion control) was silently omitted from
claim-bearing cells because a code default (`dual_cap_enabled=False`)
quietly defined "production" downward, and the gap register then
misclassified activation as a pending decision. Rule: the production
program is DEFINED IN CODE (`LEDH_PRODUCTION_PROGRAM_V1` in
`ledh_alg1_contract.py`), "production" labels are validated against it
(wiring gate), and omitting a required mechanism must surface as a
labeled deviation in the configuration-status table — never as a
default. Warm-start parameter values inside the registry are declared
as such in `tuning`; the mechanism itself is not optional.
