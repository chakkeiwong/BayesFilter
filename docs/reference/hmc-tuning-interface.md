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
   grid is `(3, 5, 9, 13, 18, 25)`. The ordinary legacy config fixes its maximum
   at 25; the position-field config declares its own maximum. Supplied initial,
   refinement and expansion grids must obey the preparation config's maximum. A supplied
   `epsilon_by_l` may contain several epsilons at each L. `initial_epsilon` is
   a warm start when an explicit grid is unavailable; each L is still measured
   independently. An optional per-L pilot proposes a frozen pair and never
   qualifies it for replay.
3. Measure every funded admitted exact pair. Close the current cohort before
   admitting its repair children. A peer whose required work exceeds remaining
   budget is explicitly deferred, allowing affordable mandatory work to finish.
   Its evidence allocation is preserved, and the search remains incomplete.
   A first passing member does not end the cohort.
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
   Every exact setting is deduplicated. Repairs remember the latest valid
   directional evidence at each exact pair and prioritize unvisited geometric
   interiors toward nearby opposing observations. Finite refinement also tests
   unresolved directional intervals when no family survives. This proposes
   measured hypotheses without assuming monotone acceptance. Trajectory alerts can propose only
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
| Available native divergence | Vetoes current promotion; finite valid acceptance evidence can still support a directional child repair. |
| Candidate-local invalid target, score or transition health | Rejects that candidate and preserves the reason. |
| Corrupt shared execution or accepted-state consistency | Stops the scope and disables all its replayable members. |
| Resource/runtime interruption | Preserves attempted work and pauses for unchanged-scope resume. |
| High, missing, nonfinite or computation-error R-hat | Reporting-only; not a tuning gate, ranking score, repair trigger or delay. |
| ESS and short-chain runtime | Descriptive diagnostics; disabled for ordinary tuning admission. |

The exact numerical binding checks accepted/proposed states and targets, endpoint
score finiteness, momentum, Metropolis state consistency, native divergence when
available, and declared target status. Target-specific intermediate integrator
telemetry and the correctness of the target value/score require consumer evidence.
Both execution routes check numerical health over discarded warmup too. The
position-field route checks states, log acceptance and energy errors before
excluding warmup from acceptance statistics.
R-hat/ESS still belong in separate cumulative posterior assessment; tuning does
not establish convergence.

Acceptance compatibility and promotion eligibility are separate typed fields.
An in-band acceptance receipt with a promotion veto is a valid record of a
rejected setting. Writers and readers preserve it; a verified member still
requires its own fresh verification with valid evidence and no veto.

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

Configuration types and conflicting options are checked before preparation.
An issued numerical binding rejects redundant execution, search, lineage,
source-path, or frozen-payload overrides; create a new binding when those inputs
change. Automatic routes accept explicit `search_config` and `execution_config`.
The latter owns candidate evidence counts and acceptance policy while its XLA
and target-status settings must agree with preparation.

On the position-field route, the legacy `step_adaptation_results` count supplies
the fixed-pair pilot allocation when a pilot is enabled. It no longer invokes
the historical adaptive candidate selector. `verification_results` supplies
the measurement and verification allocations. An explicit
`HMCCandidateExecutionConfig` replaces those stage allocations, with optional
`pilot_num_results` (otherwise the measurement count). Pilot evidence never
grants replay authority. The route uses batched chains and rejects threaded
chain execution. Both preparation configs default to XLA; position-field
non-XLA diagnostics must supply `non_xla_reason`.
Historical candidate-policy fields in `TensorFlowHMCKernelTuningConfig.payload()`
are explicitly labeled as metadata for the historical graph helper. The active
result's shared search and execution configs determine candidate stages.

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

Before frozen candidate execution exists, `preparation_progress.json` records
phases, elapsed time and any failure. Ordinary preparation checks deadlines
between its available progress boundaries. Position-field affine preparation
is one compiled graph, checked before and after the call. A failed preparation
requires a fresh output directory for retry; it cannot be resumed as numerical
candidate evidence. Source or search-policy changes also require a fresh run;
historical controller results remain readable but cannot resume under a new
controller policy.

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
`max_gradient_work` bounds a conservative transition-times-(L+1) work estimate.
The scheduler quotes unfinished stage work; each numerical chunk is charged
and checkpointed before its native call. Completed chunks are not charged
again on resume. An interrupted native call remains conservatively charged
even when it returned no usable chunk; invocation budgets charge its retry too.
This is not a measured count of every target invocation. Unfundable work stays
pending while other affordable reserved work may proceed.
`max_wall_time_seconds` includes recorded preparation time and execution time.
The numerical adapter uses chunks of at most `chunk_max_results` (convenience
default 256), checks time between chunks, and records elapsed time including the
first call's tracing/compilation. Native calls and compilation cannot be
preempted by a Python deadline; a hard process limit must be applied externally.
When both preparation and search time limits are supplied, the smaller limit
applies. Unfinished work reports `partial_budget` or
`paused_infrastructure`, not `complete`. Budget-limited searches may still have
verified members. `shared_invalidity` disables all replay.

`complete` means that all work allowed by the declared search policy reached a
terminal disposition. It can coexist with an empty verified set and does not
prove that no viable kernel exists. `HMCTuningScopeCollection` resolves members
across scope/search identities; its completeness requires all included searches
to be complete and all explicitly expected scopes to be present.

Acceptance qualification uses the complete evidence predicate. After health,
chain-conflict, temporal-conflict and trajectory-pathology screens, the
compatibility interval must overlap the practical band, lie wholly in the
repair band, and every chain mean must lie in the repair band. With practical
band [0.65, 0.75] and repair band [0.55, 0.85], a pooled mean 0.76 with interval
[0.73, 0.79] can pass these conditions. Closeness to 0.70 never ranks members.
An interval wholly above/below the practical band supports directional repair;
inconclusive evidence receives its declared additional allocation.

An adapter's `classify_target_exception(error) -> bool` can explicitly classify
a native domain error as candidate-local. It produces a typed failure with no
invented acceptance, preserves attempted seeds, and allows peers to continue.
Resource/unavailable/deadline/aborted errors remain infrastructure failures;
unknown exceptions are not silently declared model-domain failures. Shared
invalidity preserves past verification history but disables every replay.
Budget-deferred work is reconsidered when a member releases reserved work.

Retired fixed-transport selection and R-hat configuration overrides are rejected
at the public boundary. Use `execution_config.acceptance_policy` and the shared
posterior policy for their active replacements. All-chain movement remains
mandatory; setting `require_all_chain_movement=False` is rejected by the config.

## Retained sampling and posterior assessment

Choose a candidate ID explicitly and pass its result and numerical binding to
`build_retained_bound_hmc_archive_runner_from_candidate_set_result`. The two
frozen-kernel builder variants share verification checks; the claim-eligible
variant additionally requires actual GPU/XLA evidence. A repaired child needs
its own passing verification and valid ancestry. A failed parent need not pass.

`runner.export(path)` writes a compact member and a shared, checksummed evidence
bundle beside it. Copy both when relocating the export. Use
`runner.export(path, portable=True)` for a standalone file. Both formats reload
with `load_hmc_candidate_retained_runner(path, adapter=original_target)`.
Immutable evidence is written once; externally changed files are rechecked.
Numerical analyses are cached by their current content hash, while live source,
geometry, target and evidence mutation checks remain active. Predecessor archive
validation uses an iterative walk. History hashing and target-scale memory costs
remain; the implementation does not claim constant-cost restart.

`runner.run(..., previous_archive=...)` writes a fixed mechanics block from the
verified/predecessor endpoint. It excludes tuning draws, checks numerical health
and fresh seeds, and preserves frozen settings. It does not assess posterior
burn-in or precision. Use `run_hmc_posterior(member=runner, config=...)` (or
`runner.run_sequential`) for discarded equilibration and cumulative retained
assessment. It preserves the member's numerical runner, scalar/batched topology
and declared target-status policy. The entire bounded derived seed schedule is
checked against tuning and attempted calls before sampling. R-hat, ESS and MCSE
cannot alter tuning membership.

The `SequentialNeuTraHMCConfig` keyword API is shared with ordinary members;
its historical name does not require a learned transport. `step_size`,
`num_leapfrog_steps` and `jit_compile` must match the member. The inherited
warmup screen starts after 2,000 transitions per chain, uses the latest 1,000,
requires modern R-hat <= 1.05, and caps warmup at 10,000. These are operational
owner-policy defaults, not estimates of a universally sufficient burn-in.
Stan likewise configures a warmup budget and adapts its metric and step size;
its schedule does not prove stationarity or automatically certify sufficient
burn-in for arbitrary targets.

`HMCPosteriorAssessmentPolicy` can add warmup/retained bulk and tail ESS floors,
`warmup_consecutive_checks`, and an `HMCPrecisionPolicy`. Additional requirements
are explicit; default ESS floors are zero and one successful warmup look is
required for compatibility. Overlapping successful windows are not independent
replications. Every look reports modern R-hat, bulk/tail ESS, original-scale mean
ESS and MCSE. These assess the monitored quantities in the explored region;
chains trapped together in a missed mode may still pass. A failed screen
continues within the declared cap. Exhaustion reports inconclusive equilibration
or insufficient retained evidence, never sufficient burn-in by fiat.

Retained draws grow cumulatively, excluding all warmup. Core R-hat and health
checks cannot be replaced by `retained_diagnostic_fn`; callbacks can add
requirements or vetoes. No accuracy target produces `precision_not_requested`,
even if the other checks pass. The public `passed` flag means the *declared*
checks passed, and does not imply requested precision when none was declared.
A precision policy names every estimand and requires an absolute MCSE tolerance,
an MCSE/posterior-SD tolerance, or both. A mean target can monitor an event
indicator; quantiles use their own probability and indicator-ESS/order-statistic
MCSE. Supply `quantities_fn(draws)` returning named `[draw, chain]` tensors and a
stable `quantities_id` for scientific functionals or event probabilities. These
quantities receive the same R-hat and ESS checks as model coordinates.

The mean estimators are `autocorrelation`, `batch_means`, and `lugsail`.
Autocorrelation retains the explicitly identified TFP 0.25 positive-pairs
estimator; it is not Stan's initial-monotone estimator. Batch means use complete
batches within each independent chain. Lugsail combines estimates at batch
sizes b and floor(b/r) as `(LRV_b - c*LRV_small)/(1-c)`. The pooled mean variance
is `sum(chain_LRV)/(chains**2 * draws_per_chain)`. Lugsail r=3, c=0.5 and
square-root batch size are literature baselines; the minimum 20 batches is an
operational floor, not calibrated coverage. All are configurable. Negative,
zero, nonfinite or underbatched estimates cannot grant precision; raw per-chain
LRV and excluded terminal counts are reported. Quantile ties yielding zero
width are unavailable evidence, including unobserved rare events.

These MCSE calculations assume the relevant moments and mixing/CLT conditions.
Lugsail estimates retained mean uncertainty; it does not estimate burn-in.
Repeated MCSE checks provide an operational accuracy screen, not anytime-valid
confidence coverage. Finite-chain calibration does not justify a universal new
stopping default. Declared target-specific posterior checks remain necessary.

`checkpoint_store` accepts `DurableTensorCheckpoint` to preserve numerical
chunks. Restart the call with the same member, policy, seeds, names, coordinate
transform and quantity definition; completed transitions reload and diagnostics
are recomputed from the same cumulative draws. Include consumer callback and
transform identities in the store identity. Changed assessment settings require
a distinct store. The legacy archived and exact-transition wrappers use the
same assessment arithmetic; the archived API retains its stricter `<` R-hat
boundary and declared ESS/coordinate screens for compatibility. Its parameter
names default to `parameter_0`, etc.; supply scientific names for accuracy targets.

The diagnostic identity is `bayesfilter.hmc_diagnostic_math.v2`. The rank formula
now uses `(rank-3/8)/(S+1/4)` and R-hat is the square root of the variance ratio.
Earlier reports using the old arithmetic are historical evidence and must be
recomputed from draws before comparing thresholds. Rank diagnostics run as
bounded-shape TensorFlow reporting graphs without XLA because rank grouping
uses data-dependent segment operations. This exception does not change the
member's HMC XLA execution policy; the optional batch-means kernel defaults to XLA.

The separate legacy Phase 29 warmup screen still uses configured adjacent-epoch
drift thresholds as heuristic rejection criteria. Its standardized differences
omit covariance between epoch means, so they are descriptive statistics, not
calibrated z tests. These extra drift criteria are not part of the common
posterior assessment policy.

See [the posterior example](../examples/hmc_posterior_precision.py) and
[the combined repair and evidence plan](../plans/bayesfilter-hmc-overall-repair-plan-2026-09-15.md).

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
