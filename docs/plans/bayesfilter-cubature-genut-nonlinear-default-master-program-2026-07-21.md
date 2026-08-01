# Cubature/GenUT Nonlinear Default and Leaderboard Master Program

Date: 2026-07-21

Status: `PHASE5_SCALAR_SCALING_PASSED_COMPARATOR_HIERARCHY_CORRECTED_20260722`

> **Comparator correction, 2026-07-22:** The centered-Gaussian Contract E
> residual route is no longer the universal baseline. Use an exact
> model-specific oracle where available. Where no oracle exists, use the
> target-matched fixed-variant Zhao-Cui route as an independent diagnostic,
> never as an oracle. Centered-Gaussian Contract E is an optional residual
> ablation only. See
> `docs/plans/bayesfilter-genut-comparator-hierarchy-correction-2026-07-22.md`.

## Objective

Determine whether the staged positive-OT -> barycentric -> residual ->
Cholesky-restoration Cubature/GenUT candidate can become a reusable default
route for nonlinear model cells in the high-dimensional leaderboard.

This program is a promotion investigation. It does not change the canonical
Contract E--Chol default while evidence is being collected. LGSSM is a
controlled diagnostic for nonlinear/high-dimensional filtering; it is not an
LGSSM estimation objective and this program is not a NAWM experiment.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can one model-independent positive Cubature/feasible-GenUT route compute a declared finite value and its total recursive score for nonlinear rows with adequate numerical precision, high-dimensional feasibility, and reproducible provenance? |
| Candidate | Non-fused Cubature residual and, only when all weights are positive and representable, GenUT residual in the existing staged Contract E order. |
| Comparator hierarchy | Exact model-specific oracle when available; otherwise target-matched fixed-variant Zhao-Cui diagnostic; centered-Gaussian Contract E only as an optional residual-design ablation. |
| Expected failure modes | Generic adapter mismatch, incomplete total JVP, negative GenUT weights, target-measure mismatch, score variance, XLA compile/memory failure, scope tuning drift, or absent same-target comparator. |
| Promotion criterion | Every promoted model cell has a repository-issued candidate identity, target-matched total value/score pair, model-scope tuning artifact, nonlinear full-horizon evidence, float32/TF32 XLA evidence, replicated uncertainty, and final completeness validation. |
| Promotion veto | Any partial score, caller-stamped identity, negative/unsupported OT mass, target substitution, stale tuning, failed individual FD direction, nonfinite/invalid branch, unsupported full-horizon extrapolation, or incomplete leaderboard dependency. |
| Continuation veto | Missing target law/observations, missing model adapter ownership, invalid finite program, unrepairable mathematical contradiction, unavailable trusted GPU, or campaign budget exhaustion. |
| Repair trigger | Focused test/compile failure, derivative mismatch, conditioning failure, variance above the declared precision budget, or model-specific timeout with a bounded engineering remedy. |
| Explanatory only | Runtime, prefix success, descriptive mean, one-seed score, raw FD, and CI inclusion without a predeclared precision criterion. |
| Nonclaims | No exact nonlinear filtering theorem, unbiasedness, method superiority, HMC readiness, NAWM result, or default promotion before Phase 8. Zhao-Cui diagnostic output is never an oracle. |

## Evidence Contract

The candidate is compared using the hierarchy in the comparator correction.
When an exact model-specific oracle exists, it is the value/score accuracy
reference. When no exact oracle exists, a target-matched fixed-variant Zhao-Cui
route may be used as an independent diagnostic comparator; its output is never
treated as truth. The centered-Gaussian Contract E residual route is optional
ablation evidence only. The admitted score is the total derivative of the
exact finite value scalar emitted by the candidate. Finite differences remain
validation only. Every serious run must preserve a manifest containing commit,
target identity, route identity, tuning scope, seeds, dtype/TF32/XLA, memory
policy, wall time, and artifact hashes.

The current repository policy remains binding:

- The active owner-directed Contract E policy remains in force until a separate
  default-policy decision records an amendment; this plan no longer treats its
  centered-Gaussian residual arm as the universal scientific comparator.
- A new Cubature/GenUT route is candidate/experimental until an explicit policy
  decision follows successful evidence.
- Every claim-bearing scope requires disjoint tuning and a repository-issued
  scope-matched tuning artifact.
- Algorithmic TensorFlow paths default to XLA; no-XLA runs are diagnostic
  exceptions and cannot establish default readiness.

## Pre-Execution Skeptical Audit

The initial proposal was revised before execution for these material risks:

1. **Wrong baseline risk.** The old universal Contract E-Gaussian baseline was
   removed. Exact oracles are used where available; no-oracle models use
   fixed-variant Zhao-Cui only as a diagnostic comparator, with no accuracy
   interpretation.
2. **Proxy promotion risk.** CI inclusion, finite reset residuals, or short
   prefixes cannot admit a full model row. Full-horizon and scope-specific
   evidence are explicit gates.
3. **Identity drift risk.** The benchmark-local runner cannot issue a canonical
   identity. Phase 1 creates a candidate-only repository identity and rejects
   caller-stamped route fields.
4. **GenUT positivity risk.** GenUT is not automatically a positive OT marginal.
   Negative central weights or nonrepresentable masses are hard route vetoes;
   Cubature remains the positive fallback candidate, not a silent alias.
5. **Derivative risk.** A reset covariance identity does not prove a total
   score. Phase 2 requires component-wise tangent checks through OT, moments,
   residual injection, Cholesky, and later recursion.
6. **Variance risk.** The current `T=50` intervals are wide. Phase 4 declares
   a score-precision budget and evaluates same-scalar variance reductions rather
   than ranking methods from descriptive means.
7. **XLA/scaling risk.** The LGSSM comparison used no XLA and hard-coded `d=3`.
   Phase 5 requires loop-native/staged XLA, float32/TF32, dimension/particle/
   horizon ladders, and memory-growth manifests.
8. **Leaderboard contamination risk.** Existing blocked/value-only/sidecar
   artifacts cannot be upgraded. Phase 7 uses fresh versioned artifacts and
   final dependency/completeness validation.

Audit decision: `PASS_WITH_PHASED_SCOPE`. The candidate route may proceed as an
experimental implementation, but no default or leaderboard status is allowed
before Phase 8.

## Phase Program

### Phase 0: Policy and Boundary Freeze

**Question:** Are the target, route role, candidate identity, baseline, and
promotion/nonclaim boundaries explicit?

**Actions:** bind this plan to the active Contract E policy; inventory current
LGSSM candidate code, nonlinear adapters, tuning registry, and leaderboard
dependencies; preserve the current LGSSM result as diagnostic evidence.

**Pass:** a machine-readable inventory and reviewed boundary note exist;
canonical Contract E remains unchanged; no candidate artifact is admitted.

**Stop:** target ambiguity, policy contradiction, or dirty-path conflict that
prevents safe candidate work.

### Phase 1: Generic Candidate API and Identity

**Question:** Can model-specific transition/observation/initial-law adapters
call one generic positive Cubature/GenUT design and candidate route contract?

**Actions:** add a TensorFlow-only candidate module for standardized Cubature,
feasible GenUT weight checks, immutable model/route scope metadata, and a
repository-issued candidate identity digest. Add tests for divisibility,
moments, positivity, representation, mapping-order determinism, and rejection
of caller identity overrides. Do not wire this into canonical production paths.

**Pass:** generic design/identity tests pass CPU-hidden; existing benchmark
tests remain green; no NumPy enters the runtime module.

**Veto:** negative/nonrepresentable GenUT mass, identity forgery acceptance,
or any canonical route change.

### Phase 2: Generic Finite Value and Total-JVP Core

**Question:** Does the candidate compute one finite scalar and its complete
total score for arbitrary model adapters?

**Actions:** define an adapter protocol for initial cloud, transition/noise,
observation log density and tangents, parameter chart, and target metadata;
extract the recursive loop; include all OT, cost-scale/floor, moment, residual,
Cholesky, design, and weight dependencies. Add same-scalar FD audits at
representative points, branch/invalid fixtures, and per-time score increments.

**Pass:** exact value replay, analytic/JVP versus FD parity on every tested
coordinate and fixture, complete tangent dependency ledger, and no runtime FD
or autodiff.

**Veto:** partial derivative, branch-dependent mismatch, or score/value pair
computed by different finite programs.

### Phase 3: Nonlinear Adapter Pilots

**Question:** Does the generic core work on nonlinear models without changing
their target measures?

**Order:** actual non-Gaussian SV, predator-prey, then KSC-SV. Generalized SV
is a separate feature-family experiment after the current negative result.
Austria SIR is blocked until its clipped/atomic probability measure and total
score owners are repaired.

**Pass:** target-bound full value/score at short diagnostic horizons, then
target horizon; support/measure/finite-state gates pass; the appropriate exact
oracle or explicitly diagnostic fixed-variant Zhao-Cui comparator is available
and separately labeled.

**Veto:** target substitution, unsupported density, missing derivative owner,
or prefix evidence presented as target-horizon evidence.

### Phase 4: Score-Variance and Precision Program

**Question:** Can score uncertainty be reduced without changing the declared
finite scalar?

**Actions:** predeclare and test common random numbers, antithetic innovations,
replicated deterministic designs, score control variates/baseline subtraction,
per-time score decomposition, increased `N`, and independent-cloud averaging.
Measure variance, bias, MCSE, memory, and wall time. Separate same-target
repairs from target-changing alternatives.

**Pass:** a declared precision budget is met for each promoted coordinate/model,
or the route is honestly classified as diagnostic-only.

**Veto:** variance reduction changes the target without a new explicit target
  contract, or precision is claimed from one/few seeds.

### Phase 5: GPU/XLA and High-Dimensional Scaling

**Question:** Is the generic route feasible under repository default execution?

**Actions:** loop-native/staged `jit_compile=True` implementation; float32/
TF32 parity against an FP64/reference arm; memory-growth verification; exact
dimension, particle-count, and horizon ladders; record compile and warmed
runtime separately.

**Pass:** no silent CPU fallback, finite outputs, bounded memory, stable score
and value, and documented scaling at the intended diagnostic dimensions.

**Veto:** XLA-only failure, allocator policy failure, graph explosion, or
unsupported high-dimensional memory behavior.

### Phase 6: Per-Model Tuning and Untouched Claims

**Question:** Are controls selected fairly for each route/model scope?

**Actions:** use disjoint calibration/validation/claim partitions; tune
epsilon, balancing steps, ridge, residual design, chart/basis, and model
controls applicable to each route; issue scope-bound tuning artifacts; freeze
before claims.

**Pass:** every claim consumes a repository-issued, exact-scope tuning artifact;
selection does not use claim seeds or target-horizon holdout scores.

**Veto:** cross-model setting transfer promoted as a default, stale scope,
claim-seed leakage, or runtime retuning.

### Phase 7: Same-Target Comparator and Leaderboard Assembly

**Question:** Can admitted Cubature/GenUT cells be compared and assembled
without changing existing row semantics?

**Actions:** run fresh paired candidate claims against the exact oracle where
available. For no-oracle rows, run a target-matched fixed-variant Zhao-Cui
diagnostic with source-anchor classification, and preserve its value/score
outputs in a diagnostic ledger. Keep centered-Gaussian Contract E comparisons
as optional residual ablations. Preserve score/value pairs, five-seed/FD
evidence required by the active leaderboard contract, target/source/config
identities, and sidecar separation. Repair or explicitly leave blocked
Zhao-Cui rows blocked.

**Pass:** all dependencies for each claimed cell validate; no historical or
sidecar artifact is silently upgraded; complete matrix status is truthful.

**Veto:** missing full-horizon row, wrong comparator, incomplete seed set,
failed individual FD, or stale target/source identity.

### Phase 8: Default Decision and Closeout

**Question:** Is there sufficient evidence to amend the default policy?

**Actions:** independent terminal audit; decision table for each row/model;
compare GenUT to the exact oracle where available, and to the fixed-variant
Zhao-Cui diagnostic where no oracle exists. Include Contract E only when the
residual-design ablation answers a stated question. Document strongest
alternative explanations and residual nonclaims. Only then request/record any
owner policy change.

**Pass:** default change is supported by target-specific, replicated,
float32/TF32/XLA, full-horizon, scope-tuned evidence and no promotion veto.

**Failure:** retain candidate/experimental status and write a blocker or
negative-result closeout; do not force a leaderboard default.

## Compute and Artifact Budget

Phase 0/1 are local CPU-hidden engineering checks. Phases 2-6 use bounded
diagnostic ladders before any full claim. Each serious phase gets a fresh dated
artifact root, a run manifest, and a stop condition. A phase failure consumes
its phase budget but does not invalidate unrelated historical evidence.

## Current Execution Slice

Executed in the current turn:

- Phase 0: inventory/boundary artifact and plan consistency checks;
- Phase 1: candidate Cubature/GenUT design and identity module plus focused
  CPU-hidden tests. Result: `PASS_PHASE1_CANDIDATE_DESIGN_IDENTITY_LAYER`.
- Phase 2: generic finite value/total-JVP core and nonlinear toy same-scalar
  gate. Result: `PASS_PHASE2_GENERIC_FINITE_VALUE_TOTAL_JVP_TOY_GATE`.
- Phase 3: exact transformed-SV target-bound pilot. Result:
  historical non-DGP finite-program mechanics only; no SV target evidence.
- Phase 4: bounded antithetic score-variance diagnostic. Result:
  historical non-DGP finite-program variance only; it cannot nominate an SV
  variance policy.
- Phase 5 tiny GPU/XLA smoke: repaired fail-closed CPU/GPU placement and
  verified float32/TF32 XLA execution. Result:
  `PASS_PHASE5_GPU_XLA_TINY_SMOKE_ONLY`.
- Phase 5 scalar scaling: checkpointed `N={12,24,48,96}` and `T={2,10,50}`
  all completed with finite GPU/XLA outputs. Result:
  `PASS_PHASE5_SCALAR_GPU_XLA_SCALING_DIAGNOSTIC`.

The accepted Phase 5 scaling evidence remains scalar (`d=1`) and diagnostic;
its long-horizon compile cost and XLA autotuner warnings require graph-structure
repair before any production-style scaling claim. Phase 4 is complete only as
the stated diagnostic; Phases 3 and 6-8 remain incomplete for promotion
purposes.

## Required Artifacts

- This master program;
- Phase 0 inventory/boundary result;
- Phase 1 candidate module and focused tests;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase0-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase1-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase2-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase3-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase5-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase5-scaling-subplan-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase5-scaling-result-2026-07-21.md`;
- later phase-specific plans/results under fresh dated paths;
- final default decision and leaderboard integrity audit.
