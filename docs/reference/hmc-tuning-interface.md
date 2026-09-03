# HMC Tuning Interface

Last checked: 2026-09-03. The prose contract is exercised by
`tests/test_hmc_tuning_documentation_contract.py`; the route table is generated
from the executable capability registry.

Read this before changing an HMC consumer. Exactly two routes are active and
may issue BayesFilter replayable engineering artifacts. Several diagnostic or
historical records also have `interface_kind="public_tuner"`; that field alone
does not confer active status or artifact authority. A chain runner or stage
helper is not a complete tuner. Replayable artifact authority is distinct from
scientific/promotion authority: the ordinary runtime currently carries a
known NumPy-policy blocker, so its public result is explicitly non-admitting
for claims, default promotion, and posterior admission until that debt is
repaired or a reviewed exception is recorded.

At this revision, the only canonical artifact-authority entry points are
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`. Exported discovery,
refinement, campaign, runner, and stage helpers remain diagnostics unless the
capability registry and active route table explicitly say otherwise.

Import and compare the schemas rather than copying their values:

```python
from bayesfilter.inference import (
    HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA,
    HMC_TUNING_ORDINARY_RHAT_THRESHOLD,
    HMC_TUNING_RUNNER_BINDING_SCHEMA,
)
```

At this revision the capability-registry schema is
`bayesfilter.hmc_tuning_capability_registry.v1`, the runner-binding schema is
`bayesfilter.hmc_tuning_runner_binding.v2`, and the ordinary fixed-kernel
handoff threshold is defined by `HMC_TUNING_ORDINARY_RHAT_THRESHOLD`.

## Route Decision

1. Identify the exact log target, matching score, and coordinates.
2. For an ordinary adapter target, use `tune_hmc_kernel` and
   `HMCKernelTuningConfig`.
3. For one genuine frozen nonlinear transport with the Jacobian-corrected
   transformed value and matching score, use
   `tune_fixed_transport_hmc_kernel` and
   `FixedTransportHMCKernelTuningConfig`.
4. For a raw-coordinate frozen position-only proposal field, create a
   repository binding with `bind_neural_force_hmc_tuning_runner` and pass that
   binding to `tune_hmc_kernel`. The binding must supply the exact endpoint
   potential and must label the proposal field honestly; this branch does not
   relabel a non-gradient field as the exact adapter score.
5. Stop when none of those prerequisites holds. Do not relabel a chain runner
   or historical helper as a tuner.

This choice is conditional on what the supplied field actually computes. If it
is the exact score used by ordinary HMC, the default ordinary runner is the
right mechanism. If it is a different deterministic position-only proposal
field, it must be labeled as such and use the typed endpoint-corrected binding.

The registry is queryable without running a chain:

```python
from bayesfilter.inference import (
    HMC_TUNING_INTERFACE_CAPABILITIES,
    hmc_tuning_capability_registry_payload,
    hmc_tuning_interface_capability,
)
```

`hmc_tuning_interface_capability` accepts the short `interface_name`, such as
`"tune_hmc_kernel"`; it does not accept the fully qualified name printed in the
table. Accept a route only when `capability_status="tested_supported"` and
`artifact_authority=True`, and confirm that the corresponding route record is
active.

The generated complete table is
[hmc_tuning_route_table.md](../generated/hmc_tuning_route_table.md). Its source
is `HMC_TUNING_INTERFACE_CAPABILITIES`, and
`scripts/render_hmc_tuning_interface_docs.py --check` rejects drift.

## Downstream Static Audit

Before changing a consumer, run the bounded standard-library audit and inspect
its branch, consumer-role, NumPy, and provenance ledgers:

```bash
python scripts/audit_ordinary_hmc_migration_surface.py \
  --downstream-root /home/ubuntu/python/MacroFinance \
  --downstream-root /home/ubuntu/python/dsge_hmc
```

The report is a source-classification aid, not numerical evidence. A row marked
`unknown_dynamic_import`, `unresolved_dynamic_attribute`, or
`mixed_public_and_lower_level` requires manual role classification before a
claim-adjacent consumer can be admitted. Generated reports belong under the
ignored plan-artifact root; the authored execution note records the command
and the unresolved rows.

## Ordinary Default Policy

For `tune_hmc_kernel` with `HMCKernelTuningConfig` or an omitted config, the
resolved variant is `ordinary_hmc` with algorithm ID
`operational_paired_fixed_trajectory_selection_v3`. The default path performs
windowed mass warm-up, then screens the bounded
`{floor(anchor/2), anchor, 2*anchor}` trajectory candidates using one
shared/frozen epsilon and three replications; it retunes epsilon at the nominated `L` and
then runs fresh verification. This is not joint epsilon/L selection in the
screen. The explicit `joint_l_epsilon_grid_fixed_mass_hmc` identifier reaches
the alternate per-L epsilon grid only through an internal/legacy diagnostic
construction and is rejected by the public artifact-authority facade.

The route payload reports three separate roles: `operational_authority` for a
stage route, `artifact_authority` for a replayable route artifact, and
`scientific_promotion_authority` for a scientific/default claim. The first two
are not evidence of the third. Inspect `result.payload()["resolved_policy"]`
and require `claim_bearing_artifact_authority=True` only after the backend and
target-specific evidence gates have passed; the current ordinary result sets
it to `False` with blocker `ordinary_runtime_numpy_policy_pending`.

The route can be inspected without constructing a chain:

```python
from bayesfilter.inference import HMCKernelTuningConfig
from bayesfilter.hmc_route_contract import (
    HMC_TOP_LEVEL_SELECTION_STAGE,
    resolve_hmc_algorithm_route,
)

config = HMCKernelTuningConfig.standard()
route = resolve_hmc_algorithm_route(
    algorithm_id=config.algorithm_id,
    stage=HMC_TOP_LEVEL_SELECTION_STAGE,
    chain_execution_mode=config.chain_execution_mode,
    use_xla=config.use_xla,
)
print({"config_variant": "ordinary_hmc", "preset": config.preset, **route.payload()})
```

This is a construction-only inspection. It does not tune, initialize an HMC
runner, or establish numerical validity.

## Exact Public Imports

Ordinary target:

```python
from bayesfilter.inference import (
    HMCKernelTuningConfig,
    tune_hmc_kernel,
)
```

Frozen transport:

```python
from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    tune_fixed_transport_hmc_kernel,
)
```

Typed neural-force mechanics inside the ordinary ladder:

```python
from bayesfilter.inference import (
    BoundRetainedHMCArchiveConfig,
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    bind_neural_force_hmc_tuning_runner,
    build_retained_bound_hmc_archive_runner_from_tuning_result,
    tune_hmc_kernel,
)
```

`tune_hmc_kernel(..., runner_binding=binding)` accepts only a repository-issued
`HMCTuningRunnerBinding`; a bare callable is invalid. The binding does not own
tuning authority. It makes the ordinary tuner use the bound runner throughout
mass adaptation, epsilon tuning, leapfrog-count selection, screening, fresh
verification, and repair.

The default ordinary runner requires an exact log target and matching score.
The v2 typed deterministic-field branch instead requires an exact endpoint
potential and a frozen, honestly labeled position-only proposal field. The
field need not equal the potential gradient, and neither the binding nor this
documentation promotes it to an exact score. The binding records identities and
coordinate semantics; it cannot prove that a caller's endpoint potential is the
claimed target. That equality still requires a target-specific check.

## Ordinary Workflow And Geometry

The caller and tuner have different geometry responsibilities. The caller
supplies a center and may supply a local geometry hypothesis. The tuner
validates that input, constructs the affine fixed-mass adapter, performs
windowed mass adaptation by default, screens the bounded operational
`{floor(anchor/2), anchor, 2*anchor}` trajectory candidates with one
shared/frozen epsilon and three replications, retunes epsilon at the nominated
`L`, screens the frozen candidate, runs fresh verification, and applies bounded
repair.

For `mass_policy="windowed_adaptive"`, geometry hints are tried in this order:

1. `negative_hessian`, interpreted as `-d^2 log posterior` in the same
   unconstrained coordinates as `initial_position`;
2. `initial_covariance` at that position;
3. positive diagonal `parameter_scales`; and
4. identity covariance.

When `allow_geometry_fallback=True`, an invalid higher-priority hint is recorded
and the next hint is tried. When it is false, the invalid hint fails closed.
The explicit `mass_policy="fixed_identity"` ignores all supplied hints. Identity
is useful for a mechanics smoke, but it is only a convenience fallback and is
not automatically defensible for an anisotropic posterior.

A covariance-first caller should keep the center and covariance together:

```python
covariance_result = estimate_sequential_map_covariance(...)
if not covariance_result.accepted:
    raise RuntimeError("local covariance was not accepted")

tuning_result = tune_hmc_kernel(
    adapter=adapter,
    initial_position=covariance_result.map_candidate,
    initial_covariance=covariance_result.covariance,
    config=tuning_config,
)
```

Before that handoff, establish that the center and covariance use the adapter's
unconstrained coordinates, the covariance was estimated at that center, exact
target/score replay passes there, the covariance is finite, symmetric, and
positive definite, and all regularization or fallback is recorded. The checked
stubbed binding is in
[hmc_tuning_covariance_first.py](../examples/hmc_tuning_covariance_first.py).

With the default TFP runner, final handoff requires finite health and acceptance
diagnostics, the configured minimum tuning draws, and finite rank-normalized
split and folded split R-hat values at or below
`HMC_TUNING_ORDINARY_RHAT_THRESHOLD` (`1.01` at this revision). Bulk and tail
ESS are disabled for ordinary tuning admission; retained posterior ESS is a
separate check. Neither acceptance nor tuning R-hat proves retained posterior
convergence.

The ordinary ladder currently imports NumPy and uses host numerical and
serialization paths. This is BayesFilter-owned backend migration debt under
`AGENTS.md`; until repaired, the ordinary public result is non-admitting for
claim-bearing use. The TensorFlow-only route exercises the same typed mechanics
and has two evidence roles. `diagnostic_only` can never hand off. `candidate`
may hand the same frozen transition to a retained pilot only when it selected a
predeclared trajectory length, performed a rank-eligible valid metric update,
recorded zero final-verification divergences, and passed the declared four-chain
acceptance screen. Its artifact still serializes `artifact_authority=False`,
`posterior_admission_authority=False`, and `admission_supported=False`: the
handoff is mechanics authority, not posterior or scientific admission.

Fresh retained R-hat and ESS are explanatory posterior diagnostics and do not
enter this tuning handoff. The ordinary config currently defaults to
`use_xla=False`, which is a documented policy mismatch under `AGENTS.md`, not
an implicit qualification. A claim-adjacent consumer must wait for an XLA-on
default or a scope-bound reviewed exception. The route's exposed tuning hyperparameters have no numeric
constructor defaults. The implementation fixes four chains and `float64`. A
`candidate` must supply an explicit initial-position bank with shape `[4,d]`.
The tuner preserves caller row order, uses the equal-row mean as the initial
affine center, and maps each raw row into that chart. A `diagnostic_only` call
may still supply one `[d]` position; BayesFilter replicates it deliberately and
records `initial_position_was_replicated=True`. The remaining fixed policies are
`L=1` in metric windows, a powers-of-two trajectory grid, and a stateless
seed-offset scheme. The route also inherits TensorFlow Probability's internal
dual-averaging defaults except for the explicit adaptation count and target
acceptance. These are interface policies rather than evidence of posterior
convergence or default readiness.

## Fixed Transport

When `engineering_probe_covariance_multiplier` configures the private P4-E
engineering-probe bank, the public tuner creates and owns the required
`G2PreboundarySeedUseRegistry` internally. It passes one instance through the
bootstrap, operational warmup, and P4-E boundary consumers, binds the registry
to the active source-coverage hashes, and keeps the registry and raw seed
values private. Callers continue to use the ordinary `tune_hmc_kernel`
signature; they must not construct, inject, or interpret a P4-E registry.

`tune_fixed_transport_hmc_kernel` constructs and identity-binds the transformed
target before any scoped runner is called. For `theta = T(z)`, it requires
`log pi_z(z) = log pi_theta(T(z)) + log|det J_T(z)|` and the corresponding total
score. It uses fixed identity mass in `z`, tunes epsilon, selects `L`, and runs
fresh candidate or held-out verification. It does not perform ordinary mass
adaptation. An arbitrary force without a frozen transport cannot enter this
route.

Its ESS and verification requirements are selection-policy dependent. The
ordinary ESS-disabled status and `1.01` R-hat threshold do not transfer to this
route. Read the selected fixed-transport configuration and result schema rather
than borrowing ordinary-tuner thresholds.

### Keep the operational layers separate

BayesFilter exposes fixed-transport diagnostic procedures in addition to the
public tuner. They solve different problems and their artifacts are not
interchangeable:

1. `discover_fixed_transport_hmc_candidates` is diagnostic candidate
   nomination. It does not select or confirm a kernel.
2. `refine_fixed_transport_hmc_candidates` is diagnostic, staged comparison of
   nominated candidates. It does not issue an authoritative handoff.
3. `tune_fixed_transport_hmc_kernel` is the active public tuner. Only this
   layer can issue the fixed-transport tuning artifact described by the route
   registry.

Do not assemble the first two helpers into a new de facto tuning route in a
consumer. If their policy is wanted for an authoritative handoff, either feed
the resulting proposal into an active public tuner that independently owns its
required stages or implement and review a separate artifact-authority route.

### Diagnostic candidate discovery

The discovery helper evaluates the fixed primary grid
`L=(3, 5, 9, 13, 18, 25)`. It performs separate dual-averaging adaptation for
each `L`, using four chains, then evaluates the tuned epsilon in two fresh
fixed-kernel replications. The configured adaptation, returned-draw, and screen
budgets are explicit caller inputs; do not infer them from the public tuner's
defaults.

For each `L`, the statistical unit is a replication mean across the four
chains. If the two replication means are `a1` and `a2`, discovery nominates the
candidate only when the clipped interval `[mean(a1,a2)-sd(a1,a2),
mean(a1,a2)+sd(a1,a2)]` intersects `[0.65,0.75]`. This is a deliberately broad
nomination rule, not confirmation. The result records
`selection_performed=False`, `confirmation_performed=False`, and does not
authorize a final kernel or retained sampling.

### Diagnostic candidate refinement

The refinement helper accepts at most one nominated epsilon for each `L` and
uses four chains. Its configured stages are `(500,500)` returned transitions.
Stage 1 first discards 500 burn-in transitions; stage 2, when run, continues
from surviving stage-1 endpoints without another burn-in block. Each attempt
reports rank-normalized split R-hat plus bulk and tail ESS diagnostics and uses
the four chain-level acceptance means for its nomination interval. These are
candidate diagnostics, not retained-posterior convergence evidence.

A stage permits at most one candidate-specific epsilon repair, and that repair
is attempted only if the first attempt leaves no survivors. Low acceptance
multiplies epsilon by `0.80`; high acceptance multiplies it by `1.20`. A hard
veto also follows the lower-epsilon branch. Before describing this as a
two-stage confirmation, inspect the actual stage list: stage 2 is skipped when
stage 1 leaves zero or one survivor. Therefore the implemented policy is not an
unconditional "500+500 confirmation" of every candidate.

### Active public tuner budgets and control flow

For the public `FixedTransportHMCKernelTuningConfig` defaults, a ladder or
fixed-grid screen uses 4 discarded burn-in plus 16 returned transitions per
chain. Fresh candidate verification also uses 4 plus 16 per chain. These are
short tuning screens; they are not posterior convergence runs. The default
`selection_num_results=64` is consumed only by the
`replicated_min_bulk_ess_per_gradient` selection policy. It does not lengthen
the default `acceptance_target_distance` branch.

For the public fixed-grid branch, scales are evaluated in their declared order
and the first healthy in-band screen stops traversal. That candidate then gets
fresh verification. If this fresh verification fails, the public call ends
without trying later declared scales. A plan must not promise continued scale
search after failed verification unless the implementation is first changed
and reviewed.

For the public dual-averaging ladder, an in-band healthy screen stops the
candidate ladder. A low-acceptance screen repairs toward a lower epsilon by
dividing by `step_repair_factor`; a high-acceptance screen repairs toward a
higher epsilon by multiplying by that factor. The default factor is `2`. This
is a bounded multiplicative directional repair, not a continuous repair
proportional to the measured distance from the acceptance band.

The direction is intentional: all else equal, lowering epsilon normally raises
acceptance and raising epsilon normally lowers it. Do not, however, infer an
untested neighboring result from that local rule. At fixed finite `L`, leapfrog
resonance can make acceptance non-monotone in epsilon, so a proposed neighboring
epsilon must be measured with the declared screen and fresh seeds.

### Promotion and failed-attempt discipline

Discovery and refinement artifacts remain diagnostic even when their numerical
checks pass. They cannot be relabeled as canonical tuning handoffs. Promotion
requires an active public tuner or a separately implemented, tested, registered,
and reviewed artifact-authority route. Preserve every failed or superseded
artifact as evidence; do not overwrite its root or retrofit its status. A later
attempt must use a fresh versioned output root with its own target, transport,
configuration, seed, source, and code lineage.

`run_full_chain_neural_force_hmc` owns transition mechanics for a supplied
configuration. It does not tune mass or choose `L`, cannot issue a tuning
artifact, and has a diagnostic direct identity-mass fallback. That fallback is
rejected by the typed public binding.

## DO

- Read `hmc_tuning_interface_capability(name)` at the BayesFilter commit used by
  the consumer.
- Bind target scope, adapter or transport identity, coordinates, backend, dtype,
  runner identity, and source closure.
- Let the selected public tuner own its full stage sequence.
- Record the geometry hint selected, its coordinate system, provenance,
  regularization, and any fallback.
- Treat tuning draws as discarded and run retained posterior assessment
  separately.
- Preserve failed candidates as tuning evidence and follow only declared repair
  triggers and budgets.

## DO NOT

- Do not call a chain runner and describe it as full tuning.
- Do not use a bare runner callback with `tune_hmc_kernel`.
- Do not pass an arbitrary position-only force to the fixed-transport tuner.
- Do not claim ordinary mass adaptation for fixed identity mass in latent `z`.
- Do not reuse a tuning artifact after any target, coordinate, transport,
  dimension, backend, dtype, or bound source identity changes.
- Do not treat acceptance alone as convergence or handoff evidence.

A high acceptance rate after a short run with fixed `M=I`, fixed `L=1`, and
only epsilon adaptation describes that under-tuned configuration. It is not
evidence against HMC, the target, or the neural-force proposal.

## STOP

Stop without issuing or consuming a handoff when:

- no active public tuner matches the target and coordinates;
- the exact value and score do not describe the same probability measure;
- a frozen transport or its Jacobian identity is missing;
- neural force and endpoint target coordinates differ;
- required transition telemetry or source identity is missing; movement has no
  additional candidate threshold unless the selected policy declares one;
- fresh verification fails or exhausts its cap; or
- a candidate supplies only one initial position, rather than an explicit
  four-chain bank; or
- a consumer requires XLA-qualified tuning, because the ordinary default is
  currently non-XLA and no reviewed exception record is present; or
- result, route, or capability schemas are unsupported by the consumer's
  pinned BayesFilter commit.

## Artifact Acceptance

Accept a tuning result only after checking all of the following:

- The capability record is `tested_supported`, has `artifact_authority=True`,
  and its route record is active. `interface_kind="public_tuner"` alone is not
  evidence of authority.
- The result, capability-registry, and, when applicable, runner-binding schemas
  are explicitly supported by the consumer.
- Target scope, coordinate signature, dimension, backend, dtype, XLA mode, and
  chain execution mode match the intended run.
- Adapter, mass or transport, runner-binding, and source-closure identities
  match the actual call.
- Final status is passed and the route-required fresh verifier passed. A failed
  verifier must have no final kernel or public success payload.
- The BayesFilter-owned private handoff exists for replay; redacted public
  status alone is not replayable.
- Tuning draws are excluded from retained posterior inference.

### Durable typed TensorFlow replay

A passing TensorFlow `candidate` is already bound to the endpoint target,
proposal field, affine geometry, selected epsilon and `L`, final chain state,
and source closure. Consume it only through the repository-issued binding:

```python
runner = build_retained_bound_hmc_archive_runner_from_tuning_result(
    tuning_result=tuning_result,
    runner_binding=binding,
)
pilot = runner.run(
    BoundRetainedHMCArchiveConfig(
        num_results=pilot_draws,
        seed=pilot_seed,
        output_dir=pilot_output,
        budget_provenance=pilot_budget_provenance,
    )
)
```

The builder rejects a diagnostic or failed candidate, a changed binding, or a
missing durable tuning manifest. It runs the same frozen bound transition; it
does not switch to ordinary exact-gradient HMC. For an extension, pass the
immediately preceding archive as `continuation_manifest`. BayesFilter verifies
the predecessor and begins from its final active-coordinate state. The caller
must not reconstruct the mass map or restart from the original tuning endpoint.
This is mechanics replay, not posterior admission.

### Durable ordinary replay

The in-memory `HMCKernelTuningResult` contains private geometry and final-mass
arrays that the redacted public JSON does not. Before process exit, convert a
passed result to the durable admitted mechanics form:

```python
from bayesfilter.inference import (
    admitted_kernel_mechanics_payload_from_tuning_result,
    build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
)
```

`admitted_kernel_mechanics_payload_from_tuning_result` reuses the validated
geometry already bound into the result. It preserves and signs both the initial
and adapted mass artifacts, including nonidentity geometry, and requires no
caller reconstruction of hidden hints. Persist the returned JSON-ready mapping.
`build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload` validates its
fingerprint, target, scope, execution settings, adapter signatures, initial
position, and both mass signatures; it invokes neither tuning nor HMC.

The older serialized tuning-payload replay API is different: when reconstructing
geometry from such a payload, the caller must provide the same explicit geometry
inputs used originally. A public status payload without the private mass arrays
is not replayable.

Never infer compatibility from a matching schema string alone. Record the
BayesFilter Git commit in the consumer, compare the current registry payload,
and run that consumer's contract tests. A downstream lock update is a separate
owner-controlled migration.

## Checked Sources

- Normative explanation: [HMC Tuning Interfaces](../chapters/ch21b_hmc_tuning_interfaces.tex)
- Ordinary example: [hmc_tuning_ordinary.py](../examples/hmc_tuning_ordinary.py)
- Covariance-first binding example:
  [hmc_tuning_covariance_first.py](../examples/hmc_tuning_covariance_first.py)
- Neural-force binding example:
  [hmc_tuning_neural_force_binding.py](../examples/hmc_tuning_neural_force_binding.py)
- Fixed-transport example: [hmc_tuning_fixed_transport.py](../examples/hmc_tuning_fixed_transport.py)
- Route-selection example: [hmc_tuning_route_selection.py](../examples/hmc_tuning_route_selection.py)
- Documentation contract: `tests/test_hmc_tuning_documentation_contract.py`
- Bounded downstream audit: `scripts/audit_ordinary_hmc_migration_surface.py`
- Ordinary admission and binding tests:
  `tests/test_hmc_kernel_tuning_public_api.py` and
  `tests/test_hmc_kernel_tuning_outer_loop.py`
- Fixed-transport tests: `tests/test_fixed_transport_hmc_tuning.py` and
  `tests/test_fixed_transport_hmc_binding.py`
- Fixed-transport diagnostic discovery and refinement implementation:
  `bayesfilter/inference/fixed_transport_hmc_candidate_discovery_tf.py`
- Fixed-transport diagnostic discovery and refinement tests:
  `tests/test_fixed_transport_hmc_candidate_discovery.py`
- Neural-force binding tests: `tests/test_neural_force_hmc.py`
- Dispatcher and TensorFlow diagnostic/candidate mechanics tests:
  `tests/test_hmc_tuning_dispatch.py`

These checks establish interface behavior for their fixtures. They do not
establish posterior convergence, target correctness, sampler superiority,
performance, scientific validity, default readiness, or GPU/XLA readiness.
