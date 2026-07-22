# LEDH Generic Transport Optimization Master Plan

Date: 2026-07-19  
Status: `PHASE1_COMPLETE_PHASE2_REJECTED_LATER_PHASES_DEFERRED`

## Research Intent

Determine whether semantics-preserving optimization of the shared LEDH
transport/JVP layer reduces steady-state value-and-score time for arbitrary
parameter dimension without changing the declared finite transport, Contract E
total derivative, numerical gates, or scope-specific tuning policy.

The implementation target is the generic streaming transport primitive under
`experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py` and
its callers. LGSSM is a diagnostic witness only; no LGSSM flow formula, Kalman
oracle, parameter count, or model-specific control may enter the generic
optimization.

## Evidence Contract

Question: does a generic cached-geometry transport/JVP path compute exactly the
same finite value and total JVP as the current streamed path while reducing
steady-state execution time?

Baseline: the current repository streaming transport/JVP implementation with
the same inputs, dtype, TF32/XLA setting, chunk policy, Sinkhorn/balance counts,
and batch/parameter dimensions.

Primary promotion criterion: on exact same-input fixtures, cached and baseline
primal/JVP outputs agree within declared dtype tolerances and all finite,
shape, marginal, and replay checks pass.

Performance criterion: after correctness passes, the cached path has lower
steady-state wall time on at least one trusted GPU witness without a memory-cap,
XLA, or numerical veto. Runtime differences are descriptive unless replicated
on the declared benchmark matrix.

Promotion vetoes: output mismatch beyond tolerance, non-finite values, changed
gradient target, failed marginal/reset gates, graph without XLA `While`, Python
horizon unrolling, chunk-policy violation, OOM/cap breach, or inability to
separate compile from warm execution.

Explanatory diagnostics: HLO operation count, allocator peak, ptxas register
spill messages, per-stage counters, and candidate-level runtime. These explain
results but do not establish mathematical correctness alone.

Nonclaims: no universal Sinkhorn/balance setting, no claim of optimality,
posterior correctness, HMC readiness, statistical superiority, or cross-scope
tuning transfer follows from this campaign.

## Scope And Genericity Contract

The optimization must support arbitrary positive parameter dimension `P`, state
dimension `D`, batch size `B`, particle count `N`, and valid row/column chunks.
No implementation may index a fixed number of parameters or assume LGSSM
coordinates. The cache is allowed only within one fixed OT solve where the
source/query cloud and its tangent are unchanged while potentials iterate. It
must be invalidated when the cloud, tangent, dtype, batch, or chunk scope
changes. No cache may cross a filtering time step or a parameter evaluation.

The finite program and its total derivative remain unchanged. Caching changes
evaluation order and storage only; it must not omit source-moment derivatives,
row-quotient terms, or Contract E dependencies.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Promotion |
| --- | --- | --- | --- | --- | --- |
| Cache same-cloud pairwise cost/tangent within one OT solve | Code inspection: geometry is invariant while potentials iterate | Removes repeated (C(X)) and (D C(X)) work | Extra memory or register pressure | Exact parity, allocator peak, XLA compile, warm timing | Hypothesis until benchmarked |
| Keep streamed reductions and chunk policy | Repository chunk policy and finite transport contract | Preserves scope and numerical target | Cache may not help multi-tile routes | Run K=N and a multi-tile fixture if available | Baseline constraint |
| Use arbitrary final tangent axis `P` | User requirement and generic score contract | Avoids LGSSM-specific assumptions | Shape/transposition bug | P=1, P=3, P=5, P=18 fixtures | Required design |
| Select runtime only after numerical gates | Scientific evidence policy | Prevents speed from masking wrong gradients | Fast but invalid candidate | Parity/marginal gate before timing | Required rule |

## Skeptical Plan Audit

Verdict: `PASS_WITH_BOUNDED_FIRST_OPTIMIZATION`.

- Wrong baseline risk: avoided by same-input, same-control comparisons.
- Proxy promotion risk: runtime is secondary; exact value/JVP parity is primary.
- Hidden assumption: cache is restricted to fixed-cloud iterations and arbitrary
  `P`; no cross-time or cross-parameter reuse is allowed.
- Memory risk: explicit cache can increase peak use; allocator and 8 GiB cap are
  hard vetoes.
- Unfair comparison risk: baseline and candidate use the same dtype, TF32,
  XLA, seeds, chunk policy, and controls.
- Scientific drift risk: no Sinkhorn/balance count or model-specific route is
  changed by this plan.
- Missing stop condition: any parity, memory, XLA, or budget veto stops the
  optimization phase and preserves the baseline.

## Phase Protocol

Every phase must record the command, environment, source hashes, dimensions,
parameter count, controls, wall time, peak memory, graph identity, numerical
parity, decision, and next handoff. A phase close record must refresh the next
subplan before execution continues.

### Phase 0: Baseline And Generic Fixture

Objective: freeze a small TensorFlow/XLA fixture with arbitrary `P` and record
baseline value/JVP, shapes, work, memory, and warm timing.

Entry: current transport tests pass and no active finite-program changes are
assumed.

Required artifacts: baseline fixture test, benchmark result, run manifest,
source hashes, and a phase close record.

Checks: Python compilation, focused transport tests, `git diff --check`, exact
finite/JVP output capture, and trusted GPU/XLA probe for the serious witness.

Handoff: baseline artifact is readable and all dimensions/controls are explicit.

Stop: missing GPU for the GPU witness, invalid fixture, or baseline failure.

### Phase 1: Cached Same-Cloud Geometry/JVP

Objective: add a generic optional cached geometry path for arbitrary `P` while
retaining streamed reductions and the existing default route.

Entry: Phase 0 baseline is complete.

Required artifacts: cache helper, parity tests for `P=1,3,5,18`, dtype tests,
and candidate benchmark artifacts.

Checks: primal/JVP parity, finite/marginal checks, XLA graph check, peak memory,
and warm timing against the baseline on identical inputs.

Handoff: promote only if parity passes and the candidate is not slower after
compile amortization, or retain it as an optional diagnostic optimization with
the measured tradeoff recorded.

Stop: mismatch, OOM, cap breach, non-XLA graph, or no useful result within the
declared campaign budget.

### Phase 2: Generic Tangent-Basis Batching

Objective: remove serial parameter-direction mapping in shared reset/transport
helpers using a final tangent axis, without assuming a parameter count.

Entry: Phase 1 parity and memory evidence pass.

Required artifacts: vectorized helper, old-vs-new parity tests, P ladder,
benchmark comparison, and close record.

Checks: exact same-scalar derivative parity, P=1/3/5/18, XLA, memory, and
regression tests for Contract E moment and row-quotient terms.

Handoff: promote only a measured improvement that preserves the canonical total
gradient; otherwise keep the serial helper as the default.

### Phase 3: Lean Value-and-Score Output

Objective: provide a generic production/HMC-facing callable returning only the
declared value, score, and recursive state while keeping the diagnostic callable
unchanged.

Entry: Phase 1 and any Phase 2 changes are certified or explicitly rejected.

Checks: same-scalar parity, no diagnostic-history materialization in the lean
graph, XLA, replay, and model-adapter integration checks.

Handoff: only after the callable is wired through a generic route factory; no
model may silently switch routes.

### Phase 4: Cross-Model Performance Witnesses

Objective: benchmark the promoted generic optimization on each available LEDH
route with that route's own tuning scope and controls.

Entry: generic implementation and tests are closed; each model has a valid
scope-specific selected tuning artifact.

Checks: paired baseline/candidate value-score parity and steady-state timing for
LGSSM, latent SIR, actual SV, generalized SV, KSC-SV, and predator-prey where
their tuners and claim fixtures exist.

Handoff: prepare a performance summary only for routes with complete artifacts.

Stop: any cross-model route lacks a tuner or exact scope artifact; do not infer
generic benefit from LGSSM alone.

### Phase 5: Documentation And Terminal Audit

Objective: document the measured bottleneck and optimization mathematics in the
LaTeX transport/custom-gradient chapter only if the evidence contract passes.

Required checks: LaTeX build, equation/notation review, source-to-artifact
traceability, and terminal red-team note.

Forbidden claims: no universal speedup, no production/HMC/scientific readiness,
and no claim that caching changes the mathematical transport object.

## Initial Execution Decision

Execute Phase 0 and Phase 1 only in this turn. Phase 2 requires a fresh review
of the Phase 1 parity and memory results because vectorizing the reset has a
larger graph and derivative-risk surface. Phase 3--5 remain conditional.

## Execution Closeout

Phase 0 and Phase 1 passed. The opt-in same-cloud cost/tangent cache achieved
direct generic parity and approximately `10x` descriptive steady-state speedup
on paired `T=50,N=1024` B=1 and B=16 GPU/XLA witnesses. Phase 2 was then
audited, implemented as an exact final-axis Cholesky/reset JVP, verified for
`P=1,3,5,18`, and rejected because the full-filter median warm time increased
from `0.302396 s` to `0.413242 s`. The rejected Phase 2 code was removed.

Phase 3 was not executed: after the cache, the measured warm runtime is already
about `0.30 s` at B=1, and no evidence shows diagnostic-history materialization
is the next dominant bottleneck. Phase 4 remains a future integration task
because the current nonlinear routes do not call the same fused forward-JVP
primitive. Phase 5 documentation was completed for the measured cache result.

Result record:
`docs/plans/bayesfilter-ledh-generic-transport-optimization-phase1-result-2026-07-19.md`.
