# BayesFilter HMC Tuning Interface Downstream Migration Guidance

Date: 2026-08-28

Status: `BAYESFILTER_READY_DOWNSTREAM_MIGRATION_NOT_EXECUTED`

This note records the concrete MacroFinance and dsge_hmc changes required after
the BayesFilter HMC tuning interface work is committed. It does not modify a
downstream repository, update a backend lock, or certify a downstream sampler.

## Version Snapshot

| Repository | Inspected commit | Relevant state |
| --- | --- | --- |
| BayesFilter | implementation began at `1a284ec2d09b7776b7e44fecd211e9f8e7a3ade3`; final implementation commit pending | capability registry schema `bayesfilter.hmc_tuning_capability_registry.v1`; two active artifact-authority tuners |
| MacroFinance | `98cca3bbed5289c770ff67d174808357ac8fd595` | no direct `run_full_chain_neural_force_hmc` consumer found; many ordinary `tune_hmc_kernel` consumers; worktree contains unrelated extensive user work |
| dsge_hmc | `da060c6b4952925b7a1c58bb969300aea56e45c8` | active BGS scripts call the low-level neural-force runner directly; worktree contains an uncommitted integration plan and other user work |

The inspected dsge_hmc lock file names BayesFilter commit
`4c2b2d7856b9f2cb6882e8f0b32592da81720840`. That lock is already an
uncommitted downstream change from `e39913b024485f75ff1706da8a0e89e1c312e171`.
This BayesFilter task did not edit or select either value.

## Shared Compatibility Rule

A downstream consumer must bind all three of these identities:

1. the exact BayesFilter Git commit;
2. capability registry schema
   `bayesfilter.hmc_tuning_capability_registry.v1`; and
3. the selected interface record from
   `hmc_tuning_interface_capability(interface_name)`.

A matching schema string alone is insufficient. The downstream contract test
must also check the route kind, `tested_supported` status, artifact authority,
target/coordinate prerequisite, and relevant result or runner-binding schema.

The stable policy text to add to each downstream `AGENTS.md` is:

```text
Before changing an HMC consumer, inspect the capability registry at the pinned
BayesFilter commit. Use only a tested public artifact-authority tuner matching
the target and coordinates. A chain runner or stage helper is not a tuner.
Fail closed when no supported route matches, and never route an arbitrary force
through the fixed-transport tuner.
```

Detailed stage descriptions should link to BayesFilter
`docs/reference/hmc-tuning-interface.md`; they should not be copied into policy
and allowed to drift.

## MacroFinance Patch

### Current classification

No MacroFinance Python, Markdown, JSON, or AGENTS file in the inspected search
imports or names `run_full_chain_neural_force_hmc`. Its active source contains
many calls to `tune_hmc_kernel`, so the ordinary route choice is generally
correct. The migration risk is stale admission and schema interpretation, not
the neural-force bypass observed in dsge_hmc.

### Reviewable changes

1. Add the shared stable policy paragraph above to MacroFinance `AGENTS.md`,
   with a link to the guide at the declared BayesFilter commit.
2. Add a focused contract test, for example
   `tests/test_bayesfilter_hmc_tuning_interface_contract.py`, which imports the
   pinned checkout and asserts:
   - exactly `tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel` are active
     artifact-authority tuners;
   - the ordinary record owns mass, epsilon, and `L`;
   - fresh verification is required and acceptance alone cannot hand off;
   - ordinary tuning ESS admission is disabled; and
   - the consumer records the BayesFilter commit and capability schema.
3. Audit current `tune_hmc_kernel` consumers for assumptions that a failed
   sequential R-hat verifier can still yield a final handoff. The repaired
   BayesFilter behavior rejects that handoff. Callers must branch on the typed
   final status and presence of the final kernel, not acceptance alone.
4. Keep historical and diagnostic broad-grid results historical. Do not infer
   artifact authority from an exported symbol or a function name containing
   `tuning`.

### MacroFinance gate

Run the new contract test plus the existing focused tests for every active
ordinary-tuner wrapper. No HMC campaign is needed for this compatibility gate.
MacroFinance has no single inspected backend lock analogous to the BGS lock;
each claim-bearing run must still record the exact BayesFilter commit it loads.

## dsge_hmc Patch

### Current classification

The current BGS route in
`scripts/run_bgs_full_estimation_hmc_stage.py` calls
`run_full_chain_neural_force_hmc` directly. Its tuning stage fixes the incoming
mass and `L`, and optionally dual-averages epsilon. That remains a mechanics
diagnostic and cannot issue BayesFilter tuning authority.

The uncommitted downstream plan
`docs/plans/bgs-bayesfilter-adaptive-hmc-interface-integration-plan-2026-08-28.md`
correctly says the direct route is not final, but it was written against the
pre-binding BayesFilter baseline. It says no neural-force binding is needed and
proposes passing the frozen BGS field through ordinary TFP HMC. That is wrong
relative to the plan's own arbitrary position-only-field claim. Ordinary TFP
HMC interprets the adapter score as the Hamiltonian gradient. The
endpoint-corrected arbitrary-field mechanics must instead enter the public
ladder through `bind_neural_force_hmc_tuning_runner`.

### Required call shape

At the compatible BayesFilter commit, dsge_hmc must construct:

```python
from bayesfilter.inference import (
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    HMCKernelTuningConfig,
    bind_neural_force_hmc_tuning_runner,
    tune_hmc_kernel,
)

binding = bind_neural_force_hmc_tuning_runner(
    force=frozen_force,
    target=exact_endpoint_potential,
    target_scope=target_scope,
)
result = tune_hmc_kernel(
    adapter=adapter,
    initial_position=initial_position,
    config=HMCKernelTuningConfig.serious(target_scope=target_scope),
    runner_binding=binding,
)
```

The actual BGS objects must preserve the sign convention: the endpoint object
is the deterministic potential consumed by the neural-force kernel, while the
adapter binds the same underlying target scope. The force and endpoint target
must both declare raw coordinates. The binding rejects mismatched coordinates,
missing telemetry, identity drift, and the direct identity-mass fallback.

### Reviewable changes

1. Revise the downstream integration plan and its Claude review disposition to
   account for `HMCTuningRunnerBinding` and the final BayesFilter commit.
2. Replace claim-bearing direct runner calls in the BGS full-estimation stage
   with the typed binding and public `tune_hmc_kernel` call. Keep any retained
   direct-runner canary explicitly diagnostic and prevent it from emitting a
   canonical tuning or retained-sampling handoff.
3. Update BGS contract tests to reject:
   - a claim-bearing import or call of `run_full_chain_neural_force_hmc`;
   - a bare `runner_binding` callable;
   - missing binding/source-closure identity;
   - `coordinate_route="direct_fixed_transport_z"`;
   - mass or `L` tuning claims issued by the chain runner; and
   - a final kernel when required fresh verification failed.
4. Add positive tests that the same typed binding reaches mass adaptation,
   epsilon tuning, `L` selection, screening, verification, and repair, and that
   its binding hash is serialized in the public tuning result.
5. Keep R-hat and ESS wording route-specific. The default ordinary TFP runner
   gates handoff on its configured R-hat check; the current typed neural-force
   binding exposes fresh health and acceptance verification and leaves ordinary
   tuning ESS disabled. Neither route establishes retained posterior
   convergence.

### Independent blocker: TensorFlow-only BGS policy

The typed binding closes the public ownership and identity gap. It does not
make the full ordinary tuning ladder TensorFlow-only. The inspected
`hmc_kernel_tuning.py` still imports NumPy and contains host materialization in
the existing geometry and serialization path. Claude's downstream review
already classified the missing TensorFlow-only backend as a blocker.

Therefore the current dsge_hmc policy forbids claim-bearing BGS migration even
after adopting the typed binding. Resolve this with a separately reviewed
BayesFilter TensorFlow-only tuning implementation, or an explicit downstream
policy decision that changes the requirement. A DSGE-local TFP runner, copied
tuner, or silent NumPy exception is not an acceptable repair.

### Lock and compatibility gate

Do not update `config/bgs-backend-lock.json` until all of the following hold:

1. this BayesFilter work has a final pushed commit;
2. the dsge_hmc integration plan is revised and independently accepted;
3. the TensorFlow-only blocker is resolved under dsge_hmc policy;
4. the binding and negative contract tests pass against that exact commit; and
5. the user explicitly selects the new lock revision.

No downstream retained HMC run, posterior claim, or scientific claim is
authorized by this migration guidance.

## Status Table

| Consumer | Route decision | Patch readiness | Remaining veto |
| --- | --- | --- | --- |
| MacroFinance ordinary tuners | `tune_hmc_kernel` | guidance ready; downstream edit not made | consumer tests and commit/schema binding not yet run |
| dsge_hmc BGS arbitrary field | typed `bind_neural_force_hmc_tuning_runner` passed to `tune_hmc_kernel` | API guidance ready; downstream plan must be revised | TensorFlow-only tuning path absent; backend lock not selected |
| dsge_hmc frozen nonlinear transport routes | `tune_fixed_transport_hmc_kernel` only with a genuine frozen transport | existing route remains conceptually separate | transport-specific downstream tests and declared pin still required |

The BayesFilter-only verification matrix cannot close any of these downstream
gates. Each repository must test its patch against its own declared pin.
