# BayesFilter PP-UKF Ten-Phase Tuning Repair Plan

Date: 2026-07-21  
Status: executed through Phase 7; Phases 8-10 correctly blocked by gates  
Related prior result: `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-repair-and-pp-ukf-rerun-result-2026-07-21.md`

## Objective

Repair the PP-UKF tuning path that was shown to spend most of its runtime in
retained-target health replay and to continue serious warmup after bootstrap
acceptance repair was exhausted. Then run the smallest evidence-bearing
PP-UKF tuning campaign that fits a bounded resource budget. Sequential HMC is
not part of the campaign unless a terminal tuning result is independently
verified and admitted.

This is an engineering and sampler-mechanics campaign. It does not authorize
transport retraining, posterior sampling, publication, or claims of posterior
correctness, convergence, superiority, default readiness, or scientific PP-UKF
validity.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can the repaired public tuner evaluate PP-UKF retained target health through its real batch-native value/score/status route and stop safely when bootstrap has no acceptance-promoted kernel? |
| Candidate/mechanism | Frozen PP-UKF NeuTra transport; fixed-identity HMC tuner; retained-health batch protocol |
| Expected failure mode | Capability metadata is dropped by wrappers, causing one-draw replay; status telemetry invokes the expensive target a second time; `repair_budget_exhausted` falls back to geometry and permits a long run with no promoted bootstrap kernel |
| Primary promotion criterion | A fresh tuning artifact reaches a terminal result while fixed identity remains unchanged, source/transport identities match, required target-health checks pass, and the selected kernel is acceptance-promoted rather than a geometry fallback |
| Promotion veto | Any mass/coordinate mutation, non-finite target or score, invalid status telemetry, target/transport signature mismatch, stale artifact, scalar fallback, missing terminal result, or use of a non-promoted bootstrap kernel |
| Continuation veto | Contract/data/method/hardware/privacy boundary changes; campaign cap is exhausted; required diagnostic is unavailable; or the target is mathematically/numerically invalid |
| Repair trigger | Focused test failure, call-count/parity failure, benchmark instability, resource projection over cap, or localized runner/serialization interruption under the unchanged contract |
| Explanatory diagnostics | Bootstrap acceptance, step size, leapfrog count, target-health rate, status counts, elapsed time, memory, and phase timing; these do not prove convergence or rank candidates |
| Nonclaims | No posterior correctness, convergence, sampler superiority, production readiness, default promotion, transport quality, or scientific claim follows from this tuning campaign |

## Evidence Contract

- Comparator: the same frozen PP-UKF target and transport identity used by the
  prior attempt; no adaptive-mass alternative and no retrained transport.
- Exact target identity: frozen transport SHA-256
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221` and
  target signature
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`, unless
  a new artifact explicitly records and justifies a changed scope.
- Primary pass/fail: the retained-health evaluator uses a declared compatible
  batch contract, evaluates value/score/status at most once per batch when the
  combined protocol is available, and the terminal tuner result is admitted
  only with an acceptance-promoted bootstrap kernel.
- Hard vetoes: fixed-identity mutation, non-finite/invalid target status,
  failed parity, missing source binding, scalar fallback in the batch route,
  stale/missing artifacts, process failure before required artifacts, or
  projected wall time beyond the remaining cap.
- Explanatory-only metrics: acceptance, runtime, throughput, and tail summaries
  without uncertainty analysis. They cannot rank tuning candidates.
- Required artifacts: plan, skeptical audit, focused test log, benchmark JSON,
  preflight JSON, fresh per-attempt manifest/progress/result files, and a final
  result note with decision and inference-status tables.

## Default and Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Promotion status |
| --- | --- | --- | --- | --- | --- |
| TensorFlow/TFP GPU/XLA | Repository default and PP-UKF route | Matches the owner execution policy and actual target implementation | Sandbox/device/compiler failure | Escalated GPU/XLA preflight and manifest | Reviewed default |
| `float64` PP-UKF target | Existing PP-UKF adapter and frozen transport contract | Required by the current corrected-time-order UKF implementation | Precision or memory cost dominates | Batch/scalar parity and target-cost benchmark | Frozen target choice |
| Fixed identity mass | Prior repaired public tuner and owner policy | Keeps the tuning question unchanged and prevents hidden covariance adaptation | Wrong target if any update slips through | Schedule/runtime invariant tests and event-log audit | Reviewed policy |
| Flat retained batching | PP-UKF callable accepts rank-2 `[batch, dim]`; retained samples are `[draw, chain, dim]` | Flattening preserves logical values while using graph-native batch operations | Incorrect reshape/accounting or unsupported target | Shape/parity/call-count tests | Candidate contract until verified |
| Combined value/score/status call | PP-UKF exposes `neutra_batch_log_prob_and_grad_status` | Removes duplicate expensive target evaluation in retained health | Wrapper drops status or changes semantics | One-call parity test and status-shape test | Candidate protocol until verified |
| Bootstrap promotion required | Prior attempt ended `repair_budget_exhausted` with `geometry_preflight_fallback` | A serious run must not treat a non-promoted geometry seed as tuned evidence | Long run spends budget without a viable kernel | Unit test and pre-launch manifest check | Reviewed stop policy |
| Four GPU-hours total | Bounded serious-campaign cap selected for this continuation | Prevents another unbounded 47-minute/100-transition attempt | Too little budget for terminal tuning | Rate projection against cap before Phase 8 | Campaign boundary |
| Fresh versioned roots | Academic reproducibility policy | Preserves interrupted evidence and avoids overwrite | Ambiguous result lineage | Root nonexistence check | Required |

## Skeptical Plan Audit

The plan was audited before execution.

1. The baseline is the unchanged frozen PP-UKF target, not a faster but
   scientifically different target. A benchmark may measure cost only; it may
   not be used to claim a better sampler.
2. Acceptance, throughput, and target-health rate are not promotion metrics.
   They can veto invalid runs or project resource use, but cannot rank kernels
   without a predeclared multi-seed uncertainty analysis, which is outside this
   tuning campaign.
3. The prior 47-minute attempt demonstrates that transition count alone is an
   unsafe budget. Phase 4 therefore requires an actual target-cost measurement
   and Phase 5 requires a prospective wall-time projection before any serious
   continuation.
4. Batch support is not inferred from the existence of a batch-native training
   method. It must be propagated through every runtime wrapper and proven by
   rank-2 parity, logical-draw accounting, and call-count tests.
5. The plan distinguishes candidate rejection from research-direction
   rejection. A resource stop triggers a cheaper validated route or a bounded
   continuation; it does not invalidate PP-UKF mathematics.
6. A `repair_budget_exhausted` bootstrap is not silently promoted. If the
   bootstrap policy repair blocks the serious run, that is a valid terminal
   engineering result for this attempt.
7. No artifact from the prior interrupted attempt may be selected, extended,
   or overwritten. Every launch uses a fresh root.

## Phase Gates and Execution

### Phase 1: Freeze the repaired identity contract

Verify the existing fixed-identity propagation, schedule, runtime, and event
log invariants. Gate: focused tests and compile/diff checks pass; no code
change is needed if already satisfied.

### Phase 2: Advertise PP-UKF retained flat batching

Add `supports_retained_flat_batch = True` to the PP-UKF batch-native adapter
boundary and propagate it through `BatchNativeBoundAdapter`,
`FixedTransportValueScoreAdapter`, `_BootstrapFixedMassLatentValueScoreAdapter`,
and `_AffineWarmupAdapter`. Reject conflicting draw/flat declarations.

Gate: PP-UKF rank-2 value/score/status parity, wrapper propagation, and retained
logical-draw accounting tests pass.

### Phase 3: Add the combined retained value/score/status protocol

Add an optional `log_prob_and_grad_status` protocol to the PP-UKF adapter and
wrappers. Update retained health to use one combined call per batch when
status is requested, while retaining the existing two-call fallback for
adapters that do not implement the protocol. Do not cache TensorFlow tensors or
state inside the HMC/XLA controller.

Gate: combined/fallback parity, status-shape validation, failure localization,
and call-count tests pass.

### Phase 4: Measure target cost on the real route

Create a diagnostic-only benchmark that times scalar, flat-batched, and
combined value/score/status calls on the frozen PP-UKF target using the actual
GPU/XLA route and memory-growth policy. Record batch sizes, warm-up exclusions,
device/compiler settings, finite checks, call counts, and artifact hashes.

Gate: finite parity is exact within the declared tolerance and the benchmark
has stable repeated timing measurements. A failed parity or target status is a
hard stop.

### Phase 5: Add prospective resource and wall-time preflight

Use the measured retained-health rate plus fixed counts for bootstrap, Phase 4
warmup, Phase 5/6 tuning, and serialization to project the complete attempt.
The preflight must include a hard wall-time cap and remaining campaign budget;
it must fail closed if no measured rate exists or the projection exceeds the
cap. Timeout/heartbeat metadata must be enabled or explicitly recorded as
unsupported by the runner.

Gate: the projected attempt fits within the remaining 4 GPU-hours with margin
(target: <=75% of remaining budget). Otherwise proceed only to a bounded canary
or stop with a resource result.

### Phase 6: Repair bootstrap fallback policy

Change serious tuning admission so `repair_budget_exhausted` or any other
non-promoting bootstrap status cannot seed a full warmup from
`geometry_preflight_fallback`. Preserve that fallback for explicit geometry or
timing diagnostics only. Add a small-canary override only when the plan labels
it non-promoting and the artifact records the reason.

Gate: unit tests prove serious tuning fails closed without an
acceptance-promoted bootstrap kernel and that diagnostic-only routes remain
available.

### Phase 7: Run a cheap repaired canary

Run the frozen PP-UKF route with a small predeclared transition/retained-health
count under the same adapter and fixed-identity contracts. This is a mechanics
and resource canary, not a tuning result.

Gate: no mass mutation, no scalar fallback, finite target/status, terminal
canary artifact, and observed rate consistent with Phase 5. Any failure
terminates the campaign and triggers a localized repair.

### Phase 8: Run the full PP-UKF tuning-only campaign

Only if Phases 1-7 pass and the projected cost remains inside the cap, launch a
fresh full tuning-only attempt with the frozen transport. Record exact command,
commit, environment, GPU, TF32/XLA, memory policy, seeds, wall time, and all
artifact paths. No sequential sampling is launched here.

Gate: terminal result, fixed-identity invariants, source/transport identity,
finite target/status, and acceptance-promoted bootstrap. A resource stop is
incomplete infrastructure evidence, not a candidate failure.

### Phase 9: Verify terminal tuning admission

Independently validate the terminal tuning result and manifest: schema,
signatures, policy, bootstrap lineage, target-health summaries, event log, and
freshness. Write a result note with decision and inference-status tables.

Gate: only an admitted terminal tuning artifact can authorize Phase 10. No
terminal artifact means the campaign ends here.

### Phase 10: Run retained sequential HMC only if admitted

If and only if Phase 9 passes, run the shared sequential HMC controller under a
new bounded plan/root with its own warm-up readiness, cumulative retained
draws, R-hat/ESS, finite-state, energy-error, and downstream posterior gates.
This phase is not executed from a tuning-only result by implication.

Gate: any convergence or posterior veto stops sampling claims. A successful
tuning result alone is not posterior evidence.

## Budget and Artifact Roots

- Total campaign cap: 4 GPU-hours wall time, including failed and diagnostic
  attempts; no paid or expanded compute.
- Phase 4 benchmark: at most 20 GPU-minutes.
- Phase 7 canary: at most 30 GPU-minutes.
- Phase 8 full tuning: only the remaining budget after the measured projection,
  with a hard per-process wall-time cap and fresh root.
- Planned roots (all must be fresh):
  - `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/`
  - `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-02/`
  - subsequent attempts increment the suffix and never overwrite prior roots.

## Required Run Manifest Fields

Every serious or benchmark artifact records the git commit, exact command,
conda/environment identity, TensorFlow/TFP versions, GPU/device visibility,
TF32/XLA settings, memory-growth verification, dtype, target and transport
hashes, random seeds, batch sizes, wall time, plan path, result path, and
nonclaims. `N/A` is used only when a field genuinely does not apply.

## Decision and Inference Tables

The terminal result note must include:

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Adapter/runner repair | Parity, call-count, and fixed-identity tests | Any mismatch or scalar fallback | Target cost beyond tested batch sizes | Repair locally or stop | No sampler/scientific claim |
| Tuning admission | Terminal fixed-identity result with promoted bootstrap | Invalid target, mutation, stale/missing artifact, resource cap | Short/one-seed mechanics evidence | Independent verification only | No convergence/posterior claim |
| Sequential HMC | Separate posterior gates | R-hat/ESS/finite/energy/downstream veto | Monte Carlo uncertainty | Stop or continue declared confirmation | No superiority from tuning |

| Evidence class | Required statement |
| --- | --- |
| Hard veto screen | Which engineering/numerical vetoes fired or passed |
| Statistically supported ranking | Normally none for this one-scope tuning run |
| Descriptive-only differences | Runtime, acceptance, and target-health rate only |
| Default readiness | Not established by tuning-only evidence |
| Next evidence needed | Terminal admission, then separate sequential HMC evidence |

## Execution Record

Execution completed through Phase 7. Phases 8-10 were correctly not executed
because the resource and bootstrap-promotion gates failed. See:

- `docs/plans/bayesfilter-pp-ukf-ten-phase-tuning-repair-result-2026-07-21.md`
- `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase4-target-cost.json`
- `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase5-budget-preflight.json`
- `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase7-canary/run_manifest.json`
