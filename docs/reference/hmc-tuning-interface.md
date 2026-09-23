# HMC Tuning Interface

Last checked: 2026-09-23. This reference describes the common candidate-set
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

### Supplied whitening and difficult geometry

Positive funnel tuning tests assume a frozen whitening or partial-whitening
map. An exact noncentered chart is a useful analytic control; a supplied
nonlinear map with known residual curvature tests partial whitening. Learning
such a map is an upstream task with its own validation. Ordinary affine mass
adaptation cannot remove a funnel's position-dependent conditional scale.
An empty centered-coordinate search is a useful bounded-failure outcome, not
by itself a defect in epsilon/L tuning or a requirement to extend the search
until something passes.

Use `tune_fixed_transport_hmc_kernel` for a supported frozen map, with its
exact transformed density and score including the Jacobian. Its
`initial_position` is in **latent coordinates**. To compare maps at identical
model starts, apply the inverse coordinate chart and inverse map first, then
check the forward roundtrip. Map changes require new tuning scopes and fresh
verification. An imperfect map may leave no usable pair within the declared
budget; record that residual-geometry outcome. All verified members remain
retained, and convergence and precision in model quantities are assessed
separately.

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

A multiplicative epsilon repair can jump from high acceptance directly to
nonfinite trajectories. The nonfinite result supplies no valid acceptance
direction, so it cannot form the opposing-evidence interval used by refinement.
`HMCControllerConfig(explore_failed_intervals=True, refinement_rounds=2)`
optionally explores unvisited geometric interiors between a valid directional
parent and its numerically rejected same-L child. Two rounds are an explicit
example, not a calibrated default. The nearest rejected child bounds only an
exploration interval; it is not evidence of monotone acceptance or stability.
Each proposal is a new same-family child of the valid parent. Both source
receipts are recorded, the failed endpoint remains rejected, and fresh
measurement and verification are required. Missing telemetry, unknown failures,
invalid retained states and shared corruption cannot supply these endpoints.
Family, candidate, cohort and work budgets still apply. The option defaults to
false; without it, an explicit intermediate grid can investigate that gap under
a fresh scope.
Completion of the declared search does not establish that the unexplored
epsilon interval contains no useful setting.

The final-metric epsilon from operational preparation supplies a starting step
and, by default, the search ceiling. The short preparation probe does not prove
a stability limit or that this ceiling contains a suitable pair for every L.
`HMCKernelTuningConfig(candidate_search_bound_expansion_steps=1)` explicitly
widens the exploration cap by the existing `step_repair_factor`, retaining the
same final geometry. Zero preserves the inherited ceiling; larger counts must
not exceed `max_attempts`. This is an optional search hypothesis, not a newly
calibrated default. Every proposed pair still needs numerical health checks,
measurement and fresh verification.

Direct preparation bindings accept `preparation_bound_expansion_steps` and
intersect the requested domain with the resolved finite cap. They record the
original bound, expansion, final metric/coordinate signatures and new search
identity. Existing scopes and receipts are immutable. A directional repair
crossing the domain can measure its unvisited boundary before stopping.
Failed geometry preparation writes its stage and available veto/repair causes
to `preparation_progress.json` before raising `HMCPreparationFailure`.
Nonfinite diagnostic numbers are explicitly marked in that JSON record so
reporting preserves the original numerical failure instead of masking it with
a serialization error.

Ordinary preparation consumes an optional `negative_hessian`,
`initial_covariance`, or `parameter_scales` in the active coordinates. It does
not estimate the target Hessian when these inputs are absent: it starts from
identity geometry and unit curvature frequencies. A finite density or a
near-zero score at the initial point says little about a usable step size on
a strongly scaled target. Inspect a preparation failure before enlarging the
candidate budget; candidate search cannot repair a bootstrap that never
reaches its handoff. A checked local geometry hint is an available preparation
input, and still requires subsequent adaptation and candidate verification.

Check `operational_metric_update_count` separately from preparation success.
The existing `allow_valid_incumbent` policy permits zero empirical metric
updates; a successful preparation can therefore preserve an unsuitable initial
scale. The M9 regression diagnosis found exactly this behavior without a hint.
Operational preparation runs one chain and constructs the four-chain candidate
bank afterwards. Its temporal covariance assessment receives the original
draw/coordinate array. Explicit multi-chain inputs to `assess_metric_covariance`
retain time within each chain for the temporal information estimate; covariance
uses the pooled N-1 centered second moment with correlation shrinkage. This is
a finite-window geometry proposal, not an unbiased posterior covariance claim.
Split R-hat is reporting-only, including undefined values or computation errors;
it cannot reject or delay metric adaptation.

`preparation_progress.json` records `windowed_mass_metric_schedule` and
`windowed_mass_metric_decision` events for ordinary preparation. The schedule
report says whether any slow window can meet the dense and diagonal state-count
floors. Each completed window preserves the metric decision, individual failed
checks, and any transform, affine-parity or reasonable-epsilon rejection stage.
Metric changes start their fresh epsilon probe from the old coordinate system's
bounded dual-averaging average, clipping before exponentiation. The old bound
limits only that starting hypothesis: the probe may expand in the new
coordinates and must independently qualify its step. The unconstrained average
can grow extremely large while executed steps remain capped; it must not be
used directly as the next search's starting step. Rejected probes preserve the
starting value and available attempts. A nonfinite warmup trace reports the
failed fields, counts and first indices while retaining the numerical veto.
`metric_adaptation_status` distinguishes an applied update from a retained
incumbent. State-count and temporal-information floors, rank, condition and
shrinkage-discrepancy checks remain heuristic preparation requirements; they
are separate from posterior ESS and candidate admission.

`HMCKernelTuningConfig(metric_evidence_policy="finite_window")` optionally
allows a numerically qualified finite-window covariance proposal before the
temporal-information floor is met. The default, `"temporal_information"`,
retains that floor. Both policies record temporal ESS and preserve all count,
within-chain movement, rank, condition, shrinkage and boundary checks. The
option changes the role of ESS, not the covariance formula. It can help escape
poor initial scaling, but may also propose poor geometry; it remains experimental.
Its first matched no-hint regression run applied two updates and later failed
a nonfinite warmup trace. A short reasonable-step probe is not a global
stability guarantee. R-hat is reporting-only under both policies, and the
final frozen candidate still needs independent measurement and verification.
The option requires windowed adaptation and cannot be combined with fixed identity.
For automatic preparation with this option, the slow-window schedule preserves
the total transition budget and both buffers, while allocating windows large
enough for the inherited dense state-count floor when affordable. If only the
diagonal floor is affordable, it uses that floor. An undersized final remainder
is merged into the preceding window. When neither floor fits, the schedule
reports insufficient capacity. This removes avoidable count failures; temporal
dependence, conditioning, covariance discrepancy and target-health checks still
matter. Directly supplied schedules and the default `temporal_information`
schedule retain their declared window lengths.

`HMCKernelTuningConfig(metric_probe_num_results=16)` optionally tests each
metric-boundary epsilon using four independent short chains of sixteen
transitions. The default is one transition per probe. Longer probes visit
evolving positions and check every proposed and retained endpoint, score,
momentum and energy correction before the existing acceptance bracket can
qualify the step. Probe length is recorded with the evidence. Retained-state
inconsistency or unclassified execution failure stops preparation; a failed
proposal rejects that probe. The allocation is experimental and does not
guarantee global stability. Every later warmup window must still pass its
numerical checks, followed by candidate measurement and fresh verification.
This option changes preparation only; it does not add R-hat admission or
automatically retry a failed warmup window.

`HMCKernelTuningConfig(preparation_max_restarts=3)` enables a separate bounded
recovery hypothesis; the default is zero. A rejected nonfinite proposal can
discard its entire preparation attempt only after retained states, targets,
scores, acceptance-state consistency and declared telemetry pass independent
checks. Preparation then restarts its full schedule from the failed window's
validated starting checkpoint, retaining its qualified coordinate transform.
Covariance and dual-averaging statistics reset. Failed and earlier discarded
draws remain archived and never enter posterior estimates or the new statistics.
Fresh streams qualify a step no larger than half the consumed failing step;
that probe cannot expand beyond its contracted ceiling. High acceptance may
nominate this conservative preparation step, while final candidate measurement
and verification retain their unchanged acceptance requirements. Metric changes
still require fresh qualification in the new coordinates. Attempt limits,
cumulative transition/probe accounting and the existing wall budget apply.
Retained/shared corruption, unknown execution errors and invalid telemetry are
fatal. The option is currently limited to ordinary operational preparation.
Three restarts is a development allocation, not a calibrated universal default.

The `standard` preset is a local diagnostic allocation. With its default metric
policy, six dimensions use 150 preparation transitions and slow windows of 30, 60 and 30, all below
the inherited dense minimum of 64. More candidate-search budget does not enlarge
those windows. Use the explicit `serious` preset when its larger preparation
allocation is intended, and inspect its actual window decisions too: a larger
budget alone does not guarantee an update or posterior equilibration.
Candidate-set membership remains based on separate exact-pair evidence;
posterior assessment is still required.

The preparation helper accepts `initialize_bootstrap=True` for an optional
finite startup search before bootstrap. It keeps the supplied affine mass and
target, tests four batched momentum proposals at each epsilon/L pair, and
decreases epsilon when proposals are invalid or acceptance is too low. It
preserves the first invalid proposal and uses a separate seed stream. The
subsequent bootstrap and operational mass checks still run; startup nomination
cannot issue a tuning artifact. The q20 pricing and classical tuning consumers
explicitly enable this repair; unrelated consumers retain their existing policy.

The geometry's full `artifact_hash` retains bootstrap-probe wall times for
audit. New ordinary candidate scopes use its separate `numerical_hash`, which
excludes only those probe clock fields. Clock variation therefore cannot change
the candidate random streams. The full geometry hash remains in the execution
binding, while seeds, numerical probe outcomes, mass, starts and target still
bind the numerical identity. Existing saved checkpoints retain their original
scope and streams.

Bootstrap decisions use the mean Metropolis probability
`mean(exp(min(log_accept_ratio, 0)))` over **every recorded proposal**, including
rejections. Discarded burnin is excluded. The historical `acceptance_rate` field
and the explicit `binary_acceptance_rate` remain binary reporting fields;
`mean_acceptance_probability` controls the screen and directional epsilon
repair. Missing, nonfinite, empty, misaligned or incomplete probability traces
stop the screen without falling back to binary acceptance. For example, a mean
probability of 0.6846376853 inside [0.65, 0.75] passes this acceptance check even
if 13 of 16 proposals were accepted. This short preparation check provides no
posterior or final candidate qualification.

A bootstrap trial that raises a narrowly adapter-declared target-domain
`InvalidArgumentError` can trigger a smaller fresh trial only when the
repository's failure recorder locates it in a proposal target callback after
a finite pre-transition state. An adapter's `classify_target_exception(error)`
must return a Python boolean and recognize its own domain failures specifically.
Initial-state, retained/trace, unattributed, device, programming and classifier
failures still stop preparation. Original exception and first-failure records
remain attached to the failed round. The failed epsilon bounds future trial
proposals without being assigned an acceptance probability. A smaller measured
parent and the failed bound nominate a logarithmic midpoint; otherwise the
existing repair multiplier shrinks epsilon. L is recomputed and its clamp is
recorded. A failed reusable runner is replaced, each retry consumes the existing
repair budget, and only a later completed screen permits a repaired handoff.
This rule does not retry arbitrary TensorFlow errors or remove target assertions.
Repository failure attribution is currently available for `tf_function` without
XLA. Eager or XLA exceptions without that attribution remain terminal; enabling
retry does not relax execution-mode policy or manufacture missing evidence.

The ordinary public config also exposes `bootstrap_initialization_rounds`.
Zero, its default, preserves the existing preparation policy. A positive integer
enables the same probe with that finite round cap and uses the existing
`warmup_startup_only` bootstrap role before ordinary adaptation. For example,
`HMCKernelTuningConfig(bootstrap_initialization_rounds=20)` permits at most twenty
probe rounds; this is a bounded startup hypothesis, not an automatic geometry
estimate. The probe obeys `target_status_trace_policy`: `none` still checks
finite retained/proposed states, values and scores but does not fabricate target
status; `per_chain_step` requires complete valid telemetry. Initial or retained
invalidity stops preparation. A rejected proposal may trigger shrinking and
remains in the record. Startup's lower acceptance floor only nominates work for
adaptation; final candidate acceptance requirements remain unchanged.

q20 preparation additionally uses checkpointed four-transition bootstrap chunks.
Its explicit `warmup_startup_only` role requires the inherited lower repair floor
and finite retained/proposed state, score, target and status throughout startup;
high acceptance does not trigger upward pre-adaptation refinement. This role does
not declare the startup epsilon a stability ceiling: operational warmup performs
its existing reasonable-step search. Final candidate measurement, verification,
metric-update and posterior requirements still apply. The master estimates cost
before each chunk and before complete mass adaptation, preserving an unaffordable
stage as deferred. Checkpoints resume only with matching sources, target, starts
and numerical configuration, into a fresh output directory.

## Evidence roles

Posterior transforms may report a different number of model quantities than
the active HMC dimension. For example, two unconstrained simplex coordinates
can produce three named probabilities. The transform must preserve draw and
chain axes and match `parameter_names`; retained checkpoints and continued HMC
states keep the original active dimension. This changes posterior reporting,
not tuning membership or qualification.

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

Initial geometry is implemented in `hmc_geometry.py`; bootstrap configuration,
screening and bounded epsilon repair are implemented in `hmc_bootstrap.py`.
Windowed preparation, its timeout policy and the frozen-mass/start-bank handoff
are implemented in `hmc_mass_adaptation.py`.
Public imports and historical `hmc_kernel_tuning` aliases resolve to the same
definitions. Automatic preparation calls these implementations directly;
public presets, configuration translation and geometry-scaled budgets live in
`hmc_configuration.py`. The old module retains historical orchestration and
readers, with aliases for the extracted definitions.
These preparation helpers do not issue candidate-set tuning authority.
Numerical bindings hash these implementations and their shared preparation
helpers; seed records keep their existing identifiers while recording the
current physical source locations. Old source-bound results retain their
original provenance.

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
Malformed provider data is recorded as an execution error before it can become
an observation eligible for replay. An unchanged-scope retry retains its prior
cost. Valid shared-failure observations remain in the checkpoint.

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
prove that no viable kernel exists. Inspect the recorded limiting reason:
`max_candidates`, `max_repairs_per_family`, `epsilon_domain`, or exhaustion of
representable unvisited proposals. Refinement-cap counts include only eligible
pairs still unadmitted in that round. `HMCTuningScopeCollection` resolves members
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
The entire live evidence inventory is checked for shared invalidity, including
evidence recorded after the supplied result. An earlier result cannot restore
membership after the same binding records a shared transition failure.

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

A short recent window can contain too little effective information even after
initialization bias has decayed. Increasing `warmup_max_results` alone leaves
the information per readiness check unchanged. Declare the recent-window
length and total allowance together, assess their cost with pilot dependence
estimates, and validate the resulting posterior policy on fresh replications.
A changed policy needs a new checkpoint; it does not change tuning membership.

The shared keyword configuration and `SequentialExactTransitionConfig` retain
the default maximum of 10,000 per chain. A reviewed larger allocation can set
`max_results_per_chain` explicitly, together with a nonempty
`count_budget_reason`, and set the warmup/retained maxima within that bound.
The result records `count_budget_policy="explicit_nondefault_posterior_allocation"`.
This optional finite budget changes neither diagnostic thresholds nor the
canonical default allocation. The reason records the scientific justification;
it does not certify sufficiency. Default configuration payloads are unchanged.
Validation designs pass this same option in `options.posterior_count_budget`;
their posterior and fixed-comparator counts must fit its declared limit.
They can declare the existing mean estimator with `options.posterior_precision_method`;
omitting it preserves lugsail. This option changes no tuning decision.

Bulk/tail ESS uses the Stan/ArviZ initial-positive, initial-monotone recursion,
identified by `bulk_tail_ess_method` in the assessment policy and result. The
preserved-transition reporting API in `hmc_posterior_diagnostics` uses that
recursion for its original-scale and per-chain mean ESS too. The separate
`HMCPrecisionPolicy` autocorrelation estimator retains its named TFP convention
below. These finite-sample estimates can differ; their identifiers distinguish
them. A changed assessment policy requires a new checkpoint identity.

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

Choose tolerances in the units of the requested quantity and check their cost
before interpreting a cap as a mixing failure. For independent normal draws,
the mean MCSE is `SD/sqrt(N)` and the asymptotic median MCSE is
`sqrt(pi/2)*SD/sqrt(N)`. These are planning comparisons, not lower bounds for
correlated HMC. Broad scales and heavy tails can make an absolute target costly
even with favorable R-hat. Retain the unmet target, inspect the quantity-level
report and other predeclared members, and distinguish insufficient sampling
precision from numerical tuning failure. Lugsail does not estimate burn-in bias.

For four independent unit-variance stationary AR(1) chains with correlation
0.995, the long-chain mean variance is approximately `399/(4*n)`. MCSE 0.05
then needs about 39,900 retained draws per chain, beyond a 10,000 cap. This is
a derived planning comparison, not a universal HMC count or sufficient burn-in.
The actual controller must be tested separately from fixed-count intervals.
The [M25 continuation](../plans/bayesfilter-hmc-gap-closure-continuation-2026-09-22.md)
keeps readiness, interval coverage and precision delivery separate.

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
A positive finite estimate and twenty batches do not establish adequate
bandwidth. The M21 exact-Gaussian diagnostic found substantial downward bias
with `sqrt(n)` batches under strong persistence. Inspect dependence and compare
justified batch lengths or the existing autocorrelation estimator before
relying on a demanding precision claim. The [estimator diagnosis](../plans/bayesfilter-hmc-repair-m21-estimator-diagnosis-2026-09-22.md)
preserves the calculation and observed coverage; its alternatives remain
development evidence and do not change the default estimator.
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

Both `hmc_convergence.rank_normalized_hmc_diagnostics` (draw, chain, parameter)
and `hmc_posterior_diagnostics.rank_normalized_bulk_tail_ess` (chain, draw,
parameter) now call the shared Stan/ArviZ initial-positive/monotone ESS
implementation, identified as `bayesfilter.stan_initial_positive_monotone_ess.v1`.
Bulk and tail ESS split chains; odd lengths omit the middle draw from the
split, while tail cutoffs use the full pooled draws. Both tail indicators use
`x <= q`; equal percentile endpoints remain exactly equal. Constant draws
remain nonpromotable. Public schemas and threshold values are unchanged, but
older convergence reports need recomputation from their saved draws. The
explicitly named TFP mean/quantile precision option and the covariance-window
ESS heuristic are separate; this repair does not change either computation.

The separate legacy Phase 29 warmup screen still uses configured adjacent-epoch
drift thresholds as heuristic rejection criteria. Its standardized differences
omit covariance between epoch means, so they are descriptive statistics, not
calibrated z tests. These extra drift criteria are not part of the common
posterior assessment policy.

See [the posterior example](../examples/hmc_posterior_precision.py) and
[the active repair master program](../plans/bayesfilter-hmc-repair-master-program-2026-09-16.md).

## Testing the procedure and its diagnostics

The [inference validation suites](../validation/README.md) exercise the existing
public procedure through separate numerical, invariance, search, SBC, reference
accuracy, stopping and defect-detection experiments. The
[generated coverage table](../generated/inference_validation_coverage.md)
records actual target, route and device scope, including unavailable and
incomplete work. A model listed in the target catalog is not automatically a
tested model.

Inferred suite requirements identify every planned design. A completed step-size
or kernel-power cell cannot cover an unrun peer. Explicit category requirements
can instead request any matching assessed design. Older saved plans lacking
design IDs require inspection of individual rows for whole-suite completion.

Automatic preparation, supplied geometry, and frozen transport have distinct
coverage. Small complete-path cases assess every verified member. Full SBC uses
fresh datasets and independent complete fits, with one declared output per fit;
candidate siblings never inflate replication counts. Failed fits remain in the
denominator. Data-dependent likelihood quantities help detect fitting procedures
that ignore the observations even when parameter ranks appear uniform.

Pipeline reports distinguish `requested_members`, requested
`posterior_unavailable_members`, and `unassessed_by_design_members`.
`all_members_without_posterior_output` includes both unavailable requested
members and deliberately unassessed siblings. Historical reports before this
accounting correction included the siblings in `posterior_unavailable_members`;
inspect their individual records rather than treating that count as failed fits.

For a bounded sibling study, numerical validation designs can declare
`member_rule="shortest_verified_l"`, `posterior_members="selected"` and a
positive `posterior_member_count`. This orders distinct verified L values
increasingly and takes the smallest candidate ID within each of the requested
L values. The list is saved before posterior sampling; a shortage stays
explicit, and changing it on resume fails. Each ordinal member slot has its
own `member_slot_assessments`, with complete fits as the denominator. Siblings
are not pooled into independent replications or selected using posterior
outcomes. Shorter trajectories are a cost-oriented development hypothesis,
not a guarantee of faster mixing. Every verified tuning candidate is retained.

Numerical `search`, `accuracy` and `stopping` validation designs may declare
`options.isolate_fits=true` and `options.fit_process_timeout_seconds` to run
each complete fit in a fresh process. The timeout must fit within the design's
total budget. The coordinator does not initialize TensorFlow; each child checks
device policy, retains the full tuning and posterior evidence, and exits
normally. Exit receipts and resource measurements distinguish a saved result
from successful shutdown. Failed processes and missing outputs remain in the
replication denominator. Resume requires the same source/design, preserves
earlier attempts and consumes the remaining fit budget. A completed assessment
from an abnormal exit is preserved for inspection and is not automatically
rerun or counted as a successful replication. This optional validation setting
does not alter the tuner, member selection, numerical defaults or HMC kernels.

Fixed-kernel invariance tests use the reversible random-position construction
or independent two-sample experiments. Adapting warmup draws cannot replace
those experiments. Identity and recurrent kernels demonstrate why invariance
alone says nothing about useful exploration. A diagnostic passing on stationary
arrays also does not establish its behavior at an adaptive stopping time;
separate experiments compare actual stopped outputs with exact references.
Fixed-length comparison arms reuse the same verified member with independent
streams and predeclared discarded/retained counts. Missing arms remain in the
comparison denominator. Larger replicated experiments can assess a member
selected by tuning identity before sampling, while retaining all unassessed
siblings explicitly.

An invariance design may set `options.kernel_power` to a positive integer s.
Each experimental transition then composes s complete Metropolis transitions
of the same frozen kernel, using separately derived substep seeds. This is the
K^s extension in Gandy–Scott section 2.2, which preserves reversibility under
the null. Power one preserves the original transition and stream. The engine
checks finite states and log ratios across every substep and checks deadlines
between graph calls; an invalid substep is missing numerical evidence, never
statistical defect detection. The option is restricted to frozen invariance
experiments. It provides no independence claim for posterior draws.

An invariance design may separately enable `options.sequential` with
`max_looks` and `sample_multiplier`. This implements Gandy–Scott Algorithm 3:
each look runs a fresh, independently seeded complete experiment; sample count
increases once after the first look. The declared family dimension is fixed and
Bonferroni adjustment is applied once to the raw component p-values. Theorem 3.1
bounds type-I error only when look vectors are independent and each component
p-value is superuniform under the null. The wrapper cannot establish those
assumptions for arbitrary callbacks. Numerical failures are invalid evidence,
never statistical defect detection. This diagnostic option changes neither
tuning membership nor posterior stopping and supplies no repeated-look coverage
guarantee for their separate rules.

The Gaussian mechanics experiment independently reconstructs the Metropolis
log ratio from the analytic density and actual TFP endpoint momenta, and checks
the selected state against the accepted mask. This directly tests the energy
calculation, including the reversed-ratio defect. It does not measure the power
of distributional tests. `options.profile_execution=true` saves an attempt's
host profile, including engine failures. Profile time can include TensorFlow
compilation and execution; it is not a separate GPU kernel timing. An unwritable
profile is reported without hiding the numerical outcome.

Validation of native automatic initialization passes no supplied search
configuration. A fixed epsilon declared before preparation is a different
experiment and can exceed the prepared metric's bound. Even with native
initialization, the bounded search may return no verified candidate; failed
fits must remain visible in calibration rather than being dropped or rerun
until they pass.

The [September 16 validation campaign](../plans/bayesfilter-inference-validation-24h-result-2026-09-16.md)
also found a mixture run whose local posterior checks passed while its retained
draws missed a mode. `POSTERIOR_DECLARED_CHECKS_PASSED` means the declared checks
passed. For multimodal targets, include relevant mode or other global quantities
in posterior assessment and examine sensitivity to starting locations. A small
MCSE within one visited mode does not quantify error relative to the full
posterior.

Validation designs can declare `global_quantities: ["left_mode_probability"]`
for the mixture. The indicator then participates in posterior R-hat, ESS and
mean precision under a stable quantity identity. Constant or unobserved events
cannot grant precision. Stopped-interval reports use the independent mixture
CDF and preserve this quantity's planned denominator when retained draws are
unavailable. `mode_dispersed` is an explicit start regime for mixture
routes preserving a supplied four-chain bank; it does not alter automatic
preparation or guarantee unknown-mode discovery. The diagnostic `acceptance`
engine measures the actual candidate screen with known-mean independent or
persistent acceptance marks, retaining its measurement/verification/rung logic.
Its results describe that screen, not HMC transitions or posterior convergence.

The same `acceptance` engine now has explicitly separate numerical routes.
`frozen` measures stationary TFP-HMC acceptance and requires exact reference
starts; it does not test public qualification. `prepared` calls the public
`tune_hmc_kernel` on one declared identity-metric Gaussian pair, with fixed
tuning starts, fresh measurement/verification streams, declared evidence rungs
and no repair children. Separate iid exact anchors and an independent endpoint
energy calculation supply a stationary acceptance reference. They do not
supply tuning starts or decisions. Whole searches are the independent units;
their qualification rates do not certify sequential coverage or automatic
preparation. Acceptance may be nonmonotonic in epsilon at fixed L.

For frozen Gaussian invariance, `options.gaussian_energy_test=true` adds the
exact chi-square test for the sum of squared standardized independent
endpoints. `options.invariance_quantities` can predeclare a subset of the
available rank/KS quantities. Multiplicity includes every declared test,
including energy. Adjacent chain states do not qualify as iid endpoints.
Stopped/fixed reports give both all-planned and available-interval coverage;
conditional repeated fits of one dataset are distinct from prior-predictive
SBC. A finite MCSE or a posterior runtime-check pass does not certify coverage.

The active repair sequence, budgets and remaining evidence requirements are in
the [master program](../plans/bayesfilter-hmc-repair-master-program-2026-09-16.md).

The validation adapters for noncentered eight-schools and `sblrc-blr` regression
match pinned posteriordb Stan laws, data and coordinate Jacobians. Their campaign
calls the same public ordinary tuner and assesses a predeclared member against
ten-chain Stan references afterward. Combined lugsail uncertainty accounts for
both finite samples. These case-specific comparisons cannot establish nominal
coverage or validate a different consumer; the generic external-reference cells
still require their own matched inputs. See the
[M8 execution note](../plans/bayesfilter-hmc-repair-m8-result-2026-09-18.md).

The [M15--M17 results](../plans/bayesfilter-hmc-repair-master-program-2026-09-16.md)
extend the evidence to 246 complete fresh fits over 82 datasets, 576 numerical
acceptance-boundary searches, 128 sequential defect experiments per device,
21 geometry/route cells and four matched-reference fits. Their limits remain
material: the small-rank SBC design has weak sensitivity to subtle bias;
several sequential null intervals are too wide; centered funnels and mixtures
can fail the bounded procedure; precision caps remain possible even when
reference means agree. Frozen nonlinear transport and conditional position-field
checks exercise replay and verification, not learned-transport training.
The exact original MacroFinance reference is still unavailable. See the
[M17 result](../plans/bayesfilter-hmc-repair-m17-result-2026-09-21.md) for each
model, device, denominator and failure classification.

These are development validation tools. They preserve tuning qualification:
R-hat, ESS and MCSE do not reject, rank, repair or delay tuning candidates.
Missing references, underpowered experiments, non-rejection and descriptive
error comparisons do not grant posterior or default-readiness claims.

## Historical interfaces

Earlier `HMCKernelTuningResult`, fixed-transport selection results, and typed
`TensorFlowHMCKernelTuningResult` archives retain their original identities.
Their readers and low-level diagnostics are preserved for historical inspection.
The private ordinary campaign checkpoint helper also remains for historical
diagnostics. Public tuning rejects its `campaign_checkpoint_dir`,
`campaign_time_budget_s`, and `campaign_interrupted_elapsed_s` options. Use the
shared candidate checkpoint/resume API and shared search budgets described above.
The single-kernel `build_retained_*_from_tuning_result` and
`build_mechanics_only_frozen_kernel_hmc_adapter_from_tuning_payload` interfaces
are historical compatibility readers. They do not define the new candidate-set
procedure or allow a caller-edited authority flag to grant authority.

`run_full_chain_neural_force_hmc`, `select_fixed_transport_candidate_set`, discovery,
refinement, dual-averaging and stage helpers remain diagnostic helpers. Direct
fixed `M=I`, fixed `L=1` execution is one kernel experiment, not whole tuning.
Use `bind_neural_force_hmc_tuning_runner` for the conditional position-field
mechanics route; never disguise an arbitrary field as a frozen nonlinear map.
