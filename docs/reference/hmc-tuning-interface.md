# HMC Tuning Interface

Last checked: 2026-08-28. The prose contract is exercised by
`tests/test_hmc_tuning_documentation_contract.py`; the route table is generated
from the executable capability registry.

Read this before changing an HMC consumer. Exactly two routes are active and
may issue BayesFilter tuning artifacts. Several diagnostic or historical
records also have `interface_kind="public_tuner"`; that field alone does not
confer active status or artifact authority. A chain runner or stage helper is
not a complete tuner.

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
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    bind_neural_force_hmc_tuning_runner,
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
windowed mass adaptation by default, tunes epsilon and `L`, screens candidates,
runs fresh verification, and applies bounded repair.

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

The legacy ordinary ladder currently imports NumPy and uses host numerical and
serialization paths. This is BayesFilter-owned backend migration debt under
`AGENTS.md`. A TensorFlow-only diagnostic prototype exercises typed mechanics,
but it serializes `artifact_authority=False` and
`admission_supported=False`. It has neither the ordinary fresh-R-hat handoff
gate nor XLA qualification, so its acceptance and metric screens cannot issue a
retained-kernel handoff or close a downstream TensorFlow/XLA requirement. Its
screen role is named `diagnostic_candidate_screen`. Its exposed tuning
hyperparameters have no numeric constructor defaults, but the implementation
still fixes four chains, `float64`, four identical zero states in the current
affine coordinates, `L=1` in metric windows, a powers-of-two trajectory grid,
and a stateless seed-offset scheme. It also inherits TensorFlow Probability's
internal dual-averaging defaults except for the explicit adaptation count and
target acceptance. These are unqualified diagnostic choices, not admitted-route
defaults.

## Fixed Transport

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
- required telemetry, movement evidence, or source identity is missing;
- fresh verification fails or exhausts its cap; or
- a consumer requires TensorFlow-only or XLA-qualified ordinary tuning; the
  current diagnostic TensorFlow prototype is not an admitted replacement; or
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
- Ordinary admission and binding tests:
  `tests/test_hmc_kernel_tuning_public_api.py` and
  `tests/test_hmc_kernel_tuning_outer_loop.py`
- Fixed-transport tests: `tests/test_fixed_transport_hmc_tuning.py` and
  `tests/test_fixed_transport_hmc_binding.py`
- Neural-force binding tests: `tests/test_neural_force_hmc.py`
- Dispatcher and non-promoting TensorFlow diagnostic tests:
  `tests/test_hmc_tuning_dispatch.py`

These checks establish interface behavior for their fixtures. They do not
establish posterior convergence, target correctness, sampler superiority,
performance, scientific validity, default readiness, or GPU/XLA readiness.
