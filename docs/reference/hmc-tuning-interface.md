# HMC Tuning Interface

Last checked: 2026-09-15. This reference describes the common candidate-set
procedure. Read it with `HMC_TUNING_INTERFACE_CAPABILITIES` before changing an
HMC consumer. The generated [interface inventory](../generated/hmc_tuning_route_table.md)
classifies public tuners, preparation helpers, chain runners, and historical
readers. A chain runner is not a tuner.

## One procedure, with target-specific preparation

`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel` use
`HMCTuningCandidateSetController` and return `HMCTypedCandidateSetRun`. Its
`result` is an `HMCTuningCandidateSetResult`. Configuration types translate
preparation and initial proposals; they no longer select independent
first-admission or efficiency-winner schedulers.

| Input | Preparation and transition | Result authority |
| --- | --- | --- |
| `HMCKernelTuningConfig` or omitted ordinary config | Operational windowed mass preparation, or explicit fixed identity; exact TF/TFP value and score. | All verified members can seek checked numerical replay. |
| `HMCControllerConfig` plus numerical `candidate_set_adapter` | Caller already obtained repository-issued frozen geometry and starts. | Same controller, evidence, and replay checks. |
| `FixedTransportHMCKernelTuningConfig` plus `frozen_transport_payload` | Reconstruct the frozen diagonal-affine or dense-IAF map; identity mass in latent coordinates. | Same controller and checked numerical replay. |
| `TensorFlowHMCKernelTuningConfig` plus `runner_binding` | Affine preparation and the declared position-only proposal field with exact endpoint potential. | Conditional mechanics only; no exact-score retained-member authority. |

Unsupported custom transports or bare runner callbacks fail before execution.
Supply a supported frozen payload or a repository-issued numerical binding.
Legacy custom verification callbacks and stage retry policies must migrate to
shared evidence/budget settings. Historical helper implementations and result
readers remain for inspecting earlier evidence; they are not public tuning
alternatives. Existing consumers must migrate from a single `final_kernel_payload`
to explicit candidate IDs.

The registry schema is `bayesfilter.hmc_tuning_capability_registry.v2`; the
position-field runner schema is `bayesfilter.hmc_tuning_runner_binding.v2`.
Import their constants rather than copying the strings.

## Search, qualification, and retention

1. Prepare and freeze the target, geometry, coordinate transform, four-chain
   start bank, numerical backend, telemetry policy, and source dependencies.
2. Declare broad L coverage and epsilon hypotheses. The ordinary convenience
   grid is `(3, 5, 9, 13, 18, 25)`, subject to the explicit maximum. A supplied
   `epsilon_by_l` may contain several epsilons at each L. `initial_epsilon` is
   a warm start when an explicit grid is unavailable; each L is still measured
   independently. An optional per-L pilot proposes a frozen pair and never
   qualifies it for replay.
3. Measure every admitted exact pair. Close the current cohort before admitting
   its repair children. A first passing member does not end the cohort.
4. Give each measurement survivor its own fresh fixed-kernel verification.
   Measurement, verification, and evidence extensions use separate recorded
   streams. Neither adaptation nor tuning draws enter posterior estimates.
5. Preserve valid directional decisions. A supported smaller epsilon may repair
   rejection-induced immobility while the immobile parent remains unpromotable.
   Every epsilon repair is an immutable same-L child preserving mass, target,
   starts and coordinates. The child must be measured and independently verified.
6. Extend inconclusive measurement or verification using the predeclared finite
   `evidence_rungs`. These are multipliers of the corresponding base draw count;
   the convenience sequence is `(1, 2, 4)`. Each rung is a fresh run of the same
   candidate. An exhausted rung sequence gives `inconclusive_at_cap`, never a
   verified member. The operational repeated-look screen does not claim nominal
   sequential confidence coverage.
7. Refine around every surviving family when `refinement_rounds` is positive.
   Epsilon factors and additional L values are declared in
   `epsilon_refinement_factors`, `refinement_l_grid`, and `expansion_l_grid`.
   Every exact setting is deduplicated. Opposite directional evidence can propose
   an unvisited geometric interior epsilon; this is a new measured hypothesis,
   not an assumption of monotone acceptance. Trajectory alerts can propose only
   declared additional L values, as new candidates rather than same-L repairs.
8. Retain all verified members. `verified_candidate_ids` is the complete set of
   members eligible for checked replay. `viable_candidate_ids` also includes
   candidates still validating and those terminally inconclusive at the cap;
   inspect `candidate_states` before use. `nominee_id` is always `None`.

The automatic preparation translation enables one pilot per L and one bounded
refinement round. Direct `HMCControllerConfig` defaults to a supplied-pair search
with refinement disabled. Both use the same lifecycle; the optional stages and
limits are serialized. Refinement factors `(0.8, 1.25)` are reciprocal proposal
hypotheses, not target-specific defaults. Multiple epsilons at one L remain
separate candidates. No descriptive acceptance distance, ESS, runtime or R-hat
ranking removes a viable member.

The final-metric epsilon bound from operational preparation is a proposal safety
bound. The binding must preserve the final mass/coordinate identity that produced
it. Direct preparation bindings intersect the requested domain with the bound.
It never transfers acceptance qualification from one L to another. A directional
repair crossing the domain can measure its unvisited boundary before stopping.

## Evidence roles

`HMCAcceptancePolicy` uses four chain means, temporal blocks and a compatibility
interval, with a default minimum of four blocks of sixteen decisions per chain.
It reports mean Metropolis probability separately from realized acceptance.
Finite acceptance alone does not qualify a kernel.

| Evidence | Tuning role |
| --- | --- |
| Valid acceptance evidence with no promotion veto | Qualifies the exact measured kernel; fresh verification remains required. |
| Supported directional acceptance failure | Proposes a smaller or larger same-L child. |
| Repeated states, insufficient movement or recurrence | Vetoes current promotion; may coexist with an eligible directional repair. |
| Candidate-local invalid target, score or transition health | Rejects that candidate and preserves the reason. |
| Corrupt shared execution or accepted-state consistency | Stops the scope and disables all its replayable members. |
| Resource/runtime interruption | Preserves attempted work and pauses for unchanged-scope resume. |
| High, missing, nonfinite or computation-error R-hat | Reporting-only; not a tuning gate, ranking score, repair trigger or delay. |
| ESS and short-chain runtime | Descriptive diagnostics; disabled for ordinary tuning admission. |

The exact numerical binding checks accepted/proposed states and targets, endpoint
score finiteness, momentum, Metropolis state consistency, native divergence when
available, and declared target status. Target-specific intermediate integrator
telemetry and the correctness of the target value/score require consumer evidence.
R-hat/ESS still belong in separate cumulative posterior assessment; tuning does
not establish convergence.

## Public imports and execution

```python
from bayesfilter.inference import (
    HMCControllerConfig, HMCCandidateExecutionConfig, HMCAcceptancePolicy,
    bind_hmc_candidate_set_execution,
    bind_hmc_candidate_set_execution_from_preparation,
    tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
    resume_hmc_candidate_set_tuning, load_numerical_tuning_checkpoint,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    load_hmc_candidate_retained_runner,
)
```

The numerical factories require explicit target/data/prior lineage, source paths,
frozen mass or transport, and four-chain starts. The automatic legacy-config
translation can derive an adapter-signature-only lineage when a consumer omits
one; its metadata explicitly records that incomplete coverage. Supply
`target_lineage` and `source_paths` for real targets. A source hash and start probe
cannot establish that unlisted data or prior dependencies are unchanged.

For already prepared execution, pass the original target and
`binding.initial_active_state` to the public dispatcher, with
`candidate_set_adapter=binding.typed_adapter` and `config=search`.
For automatic ordinary preparation, pass `HMCKernelTuningConfig` plus optional
`search_config` and `execution_config`. Geometry hints are `negative_hessian`,
`initial_covariance`, and `parameter_scales`; keep the estimated center and its
covariance in the same coordinates. The operational factory preserves both
bootstrap and final affine layers and the actual post-warmup bank.

GPU/XLA is the normal execution policy. Memory growth must be enabled before
TensorFlow import and verified before GPU initialization. CPU and non-XLA
settings are explicit small reference/debugging exceptions. The public ordinary
preparation config defaults to XLA. That choice is owner policy, not evidence
that arbitrary targets are XLA qualified. The numerical binding rejects targets
without the required full-chain XLA capability.

## Checkpoints, budgets, and restart

A new run requires a fresh output directory. Complete results are written to
`candidate_set_result.json` and are never replaced. During numerical tuning,
`tuning_checkpoint.json` is atomically updated, alongside `execution_spec.json`,
immutable numerical evidence files, and completed numerical chunks. Failed and
zero-verified searches preserve observations and explanations too.

```python
continued = resume_hmc_candidate_set_tuning(
    "run/tuning_checkpoint.json", adapter=original_target,
)
```

Resume reconstructs the frozen geometry, complete numerical inventory, queue,
attempts and accounting. It rejects changes to scope, source, numerical policy,
versions, target probes, or tensor checksums. `max_work_items` pauses at a work
boundary without changing the search. Position-field mechanics preserve their
preparation and observations separately; use
`resume_position_field_candidate_tuning` with the original target and runner
binding to resume `controller_checkpoint.json`, including immutable completed
chunks within a longer stage. A controller-only artifact
cannot manufacture missing numerical evidence.

`total_budget_units` bounds dispatched attempts. The minimum candidate reserve
covers its mandatory pilot/measurement/verification stages; later rungs and
retries can use free budget without taking another admitted member's reservation.
`max_gradient_work` bounds a conservative transition-times-(L+1) work estimate;
it is not a measured count of every target invocation.
`max_wall_time_seconds` includes recorded preparation time and execution time.
The numerical adapter uses chunks of at most `chunk_max_results` (convenience
default 256), checks time between chunks, and records elapsed time including the
first call's tracing/compilation. Native calls and compilation cannot be
preempted by a Python deadline. Unfinished work reports `partial_budget` or
`paused_infrastructure`, not `complete`. Budget-limited searches may still have
verified members. `shared_invalidity` disables all replay.

## Retained sampling and posterior assessment

Choose a candidate ID explicitly and pass its result and numerical binding to
`build_retained_bound_hmc_archive_runner_from_candidate_set_result`. The two
`build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result` and
`build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result`
variants share verification checks; the claim-eligible variant additionally
requires actual GPU/XLA evidence. Callback adapters cannot grant this authority.

A repaired child needs its own passing verification and valid ancestry; a failed
parent need not pass. Export a member with `runner.export(path)` and reload it
with `load_hmc_candidate_retained_runner(path, adapter=original_target)`.
`runner.run(..., previous_archive=...)` continues from the verified/predecessor
endpoint with fresh seeds and frozen settings. Every block excludes tuning draws.
Current retained-member archives embed their evidence and validate predecessor
history; target-scale memory and restart performance remain unqualified.

Posterior estimation needs a separate declared warmup and cumulative
model-coordinate R-hat/ESS/health procedure. A verified tuning member is an input
to that procedure; `runner.run_sequential` connects the selected member to the
existing posterior controller. Posterior rejection cannot alter tuning membership.
A tuning member is neither posterior-ready nor statistically superior to
other verified members.

## Historical interfaces

Earlier `HMCKernelTuningResult`, fixed-transport selection results, and typed
`TensorFlowHMCKernelTuningResult` archives retain their original identities.
Their readers and low-level diagnostics are preserved for historical inspection.
The single-kernel `build_retained_*_from_tuning_result` and
`build_mechanics_only_frozen_kernel_hmc_adapter_from_tuning_payload` interfaces
are historical compatibility readers. They do not define the new candidate-set
procedure or allow a caller-edited authority flag to grant authority.

`run_full_chain_neural_force_hmc`, `select_fixed_transport_candidate_set`, discovery,
refinement, dual-averaging and stage helpers remain diagnostic helpers. Direct
fixed `M=I`, fixed `L=1` execution is one kernel experiment, not whole tuning.
Use `bind_neural_force_hmc_tuning_runner` for the conditional position-field
mechanics route; never disguise an arbitrary field as a frozen nonlinear map.
