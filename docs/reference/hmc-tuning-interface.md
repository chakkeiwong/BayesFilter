# HMC Tuning Interface

Read this before changing an HMC consumer. BayesFilter has two public
artifact-authority tuners. Exported helpers and chain runners are not additional
public tuners. A chain runner or stage helper is not a tuner.

Registry schema: `bayesfilter.hmc_tuning_capability_registry.v1`.

## Route Decision

1. Identify the exact log target, matching score, and coordinates.
2. For an ordinary adapter target, use `tune_hmc_kernel` and
   `HMCKernelTuningConfig`.
3. For one genuine frozen nonlinear transport with the Jacobian-corrected
   transformed value and matching score, use
   `tune_fixed_transport_hmc_kernel` and
   `FixedTransportHMCKernelTuningConfig`.
4. For a raw-coordinate frozen position-only neural force, create a repository
   binding with `bind_neural_force_hmc_tuning_runner` and pass that binding to
   `tune_hmc_kernel`.
5. Stop when none of those prerequisites holds. Do not relabel a chain runner
   or historical helper as a tuner.

The registry is queryable without running a chain:

```python
from bayesfilter.inference import (
    HMC_TUNING_INTERFACE_CAPABILITIES,
    hmc_tuning_capability_registry_payload,
    hmc_tuning_interface_capability,
)
```

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

## What Each Tuner Owns

`tune_hmc_kernel` owns geometry, windowed mass adaptation by default (or the
explicit fixed-identity config policy), epsilon tuning, joint epsilon and `L`
selection, screening, fresh verification, and bounded repair. With the default
TFP runner, final handoff requires finite health and acceptance diagnostics,
the configured minimum tuning draws, and finite rank-normalized split and
folded split R-hat values at or below `1.01`. The threshold is inherited from
the existing verifier policy. Bulk and tail ESS are disabled for ordinary
tuning admission; retained posterior ESS is a separate check.

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
- result, route, or capability schemas are unsupported by the consumer's
  pinned BayesFilter commit.

## Artifact Acceptance

Accept a tuning result only after checking all of the following:

- The registry record is `tested_supported`, is a `public_tuner`, and has
  `artifact_authority=True`.
- The result schema and capability-registry schema are explicitly supported by
  the consumer.
- Target scope, coordinate signature, dimension, backend, dtype, XLA mode, and
  chain execution mode match the intended run.
- Adapter, mass or transport, runner-binding, and source-closure identities
  match the actual call.
- Final status is passed and the route-required fresh verifier passed. A failed
  verifier must have no final kernel or public success payload.
- The BayesFilter-owned private handoff exists for replay; redacted public
  status alone is not replayable.
- Tuning draws are excluded from retained posterior inference.

Never infer compatibility from a matching schema string alone. Record the
BayesFilter Git commit in the consumer, compare the current registry payload,
and run that consumer's contract tests. A downstream lock update is a separate
owner-controlled migration.

## Checked Sources

- Normative explanation: [HMC Tuning Interfaces](../chapters/ch21b_hmc_tuning_interfaces.tex)
- Ordinary example: [hmc_tuning_ordinary.py](../examples/hmc_tuning_ordinary.py)
- Fixed-transport example: [hmc_tuning_fixed_transport.py](../examples/hmc_tuning_fixed_transport.py)
- Route-selection example: [hmc_tuning_route_selection.py](../examples/hmc_tuning_route_selection.py)
- Documentation contract: `tests/test_hmc_tuning_documentation_contract.py`
- Ordinary admission and binding tests:
  `tests/test_hmc_kernel_tuning_public_api.py` and
  `tests/test_hmc_kernel_tuning_outer_loop.py`
- Fixed-transport tests: `tests/test_fixed_transport_hmc_tuning.py` and
  `tests/test_fixed_transport_hmc_binding.py`
- Neural-force binding tests: `tests/test_neural_force_hmc.py`

These checks establish interface behavior for their fixtures. They do not
establish posterior convergence, target correctness, sampler superiority,
performance, scientific validity, default readiness, or GPU/XLA readiness.
