# Reply to MacroFinance: numerical candidate-set and retained bridge

Date: 2026-09-14  
Implementation commit: `9d06fbefe77d8653e6b3a9c4cd42bc39a8a0aff4`  
BayesFilter branch: `main`

The requested numerical evaluator and retained bridge are implemented and
tested. All three requested builder names are exported from both `bayesfilter`
and `bayesfilter.inference`. The shared candidate procedure now owns numerical
acceptance/health evaluation, frozen geometry, verification traces and retained
continuation. No MacroFinance files were edited and no MacroFinance campaign
was launched.

The bridge request was correct. Three details of the original memo needed
correction: failed ancestors may lead to a successfully repaired child; a
fully checked live binding need not first be written to disk; and the old
ordinary NumPy blocker must be interpreted against the actual source version.
Your pinned `b603363d` predates the ordinary TF/TFP backend repair. This change
preserves `ordinary_tf_tfp_runtime_v1` and does not rehabilitate old blocked
artifacts.

## What MacroFinance must change

`daily_asset_midas_phase14_candidate_set_adapter.py::OperationalCandidateAdapter.observe`
currently uses 32 measured draws, non-XLA execution and caller-side NumPy
inspection, then reports `passed` whenever acceptance is finite and the listed
health checks pass. That is not a valid implementation of the documented
acceptance decision. It neither enforces the acceptance region and uncertainty
rule nor supplies durable repository-owned verification evidence.

Replace that observation method as the tuning authority with the new
BayesFilter execution binding. Keep your original model-specific TF value/score
adapter, data/prior identity and BayesFilter preparation. Do not wrap an already
transformed `OperationalCandidateAdapter` in another mass map. The preparation
factory takes the original model adapter and the actual mapping returned by
`prepare_operational_windowed_mass_handoff`; it reconstructs and verifies both
affine layers and the active four-chain start bank itself.

Use the campaign's declared numerical budgets and acceptance policy. The
repository `HMCAcceptancePolicy` requires at least 64 measured decisions per
chain (four blocks of at least sixteen), so the old 32-draw callback cannot
qualify. Preserve the broad `L=(3,5,9,13,18,25)` search, your approved per-L
epsilon proposals/repair policy, target, prior, mass policy and MIDAS margin.
The new generic example's smaller L grid and wider acceptance band are testing
fixtures and must not be copied into the campaign.

`scripts/daily_asset_midas_phase14_active_support.py::supported_typed_to_retained_bridge`
also needs an integration update: it hardcodes both `supported=False` and
`exact_api_qualified=False` even when the names are present. Update the pinned
snapshot, check the public imports and registry metadata, and run an actual
target-specific binding/retained smoke. Then replace the deliberate M2/M4B
stops. Merely finding three function names is insufficient qualification.

The supplied memo reports M0/M1 completed, M2 blocked and zero new numerical
seconds. It does not describe an already completed candidate tune. This
delivery does not convert callback-only records into numerical verification
evidence; the corrected evaluator must execute the actual tuning work.

## Public API and integration

The preparation and execution types are:

```python
from bayesfilter.inference import (
    HMCAcceptancePolicy,
    HMCCandidateExecutionConfig,
    HMCCandidateExecutionBinding,
    HMCControllerConfig,
    bind_hmc_candidate_set_execution,
    bind_hmc_candidate_set_execution_from_preparation,
    tune_hmc_kernel,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
    load_hmc_candidate_retained_runner,
)
```

For Phase 14, bind the already prepared ordinary target as follows. Variables
named `campaign_*` below are your reviewed campaign settings, not new defaults:

```python
execution_config = HMCCandidateExecutionConfig(
    measurement_num_results=campaign_measurement_draws,
    verification_num_results=campaign_verification_draws,
    num_warmup_steps=campaign_discarded_warmup,
    seed=campaign_tuning_seed,
    acceptance_policy=campaign_acceptance_policy,  # HMCAcceptancePolicy
    target_status_trace_policy="per_chain_step",
    use_xla=True,
    chain_mode=campaign_chain_mode,
)
binding = bind_hmc_candidate_set_execution_from_preparation(
    adapter=original_model_adapter,
    preparation=prepared_windowed_handoff,
    target_lineage=campaign_model_data_prior_identity,
    config=execution_config,
    source_paths=campaign_target_source_dependencies,
    scope_id=campaign_scope_id,
    search_id=campaign_search_id,
    epsilon_domain=campaign_epsilon_domain,
    repair_factor=campaign_repair_factor,
    max_repairs_per_family=campaign_max_repairs,
)
run = tune_hmc_kernel(
    adapter=original_model_adapter,
    initial_position=binding.initial_active_state,
    config=campaign_controller_config,  # HMCControllerConfig
    candidate_set_adapter=binding.typed_adapter,
    output_dir=tuning_output_dir,
)
verified_ids = run.result.verified_candidate_ids
# Choose an explicit member under the campaign plan; no implicit best member.
runner = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
    candidate_set_result=run.result,
    candidate_id=campaign_retained_candidate_id,
    retained_binding=binding,
)
member_path = runner.export(member_output_path)
first = runner.run(
    num_results=campaign_retained_block_size,
    seed=campaign_first_retained_seed,
    output_dir=first_retained_output_dir,
)

# After a process restart: supply the original model, not a reconstructed mass.
restored = load_hmc_candidate_retained_runner(
    member_path, adapter=original_model_adapter,
)
second = restored.run(
    num_results=campaign_retained_block_size,
    seed=campaign_second_retained_seed,
    output_dir=second_retained_output_dir,
    previous_archive=first["archive_path"],
)
```

All three builders have exactly the keyword arguments `candidate_set_result`,
`candidate_id`, and `retained_binding`; they use the same validation and
execution implementation. The result may be a live `HMCTuningCandidateSetResult`,
its checksummed payload, or its durable JSON path. Returned samples have shape
`[draw, chain, parameter]`; `samples` are active coordinates and
`position_samples` are original model coordinates. Retained draws contain no
tuning/warmup draws. A `.run` call performs fixed-kernel sampling, with no
adaptation or epsilon/L override.

The claim-bearing-named builder additionally requires the exact target's
full-chain XLA capability and actual GPU verification evidence. Its
`claim_eligible=True` means it satisfies this numerical/backend policy, not
that it certifies scientific claims. CPU/non-XLA exceptions remain mechanics
only. Set `claim_eligible=True` explicitly on reload if that stricter boundary
is intended. R-hat is reporting-only in tuning and in these retained block
reports; the consumer's sequential retained convergence, ESS and scientific
checks remain necessary before posterior admission.

For frozen transports, the same execution factory accepts a complete supported
affine-diagonal or dense-IAF artifact and explicit latent starts. The fixed
transport dispatcher takes `base_adapter=original_model_adapter`,
`fixed_transport=binding.fixed_transport`, and the same typed adapter/config.
Unsupported transports or arbitrary forces fail closed.

## Exact execution and evidence path

```text
prepare_operational_windowed_mass_handoff
  -> bind_hmc_candidate_set_execution_from_preparation
     -> build_operational_fixed_mass_hmc_adapter (revalidation)
     -> HMCCandidateExecutionBinding.typed_adapter
tune_hmc_kernel(HMCControllerConfig)
  -> run_typed_hmc_candidate_set
     -> HMCTuningCandidateSetController.run
        -> HMCCandidateExecutionBinding.observe(work, candidate)
           -> build_independent_chain_tfp_hmc_runner
              -> ReusableFullChainHMCRunner -> tfp.mcmc.sample_chain
           -> evaluate_hmc_acceptance_evidence + transition-health checks
           -> captured numerical evidence and bound verification receipt
     -> write_candidate_set_result
build_*_from_candidate_set_result
  -> common _validate_member
     -> require_verified_member + source/geometry/state validation
     -> recompute acceptance and health from captured tensors
  -> HMCCandidateRetainedRunner
     -> export / load_hmc_candidate_retained_runner
     -> run -> same independent-chain TFP runner
     -> retained_archive.json with checked predecessor endpoint
```

Every candidate keeps its own epsilon, L, family/parent identity, scope, mass,
source, backend, dtype and XLA identity. Fresh verification has its own stream.
The selected child's incoming repair must be executed and freshly verified;
all older repair links must be valid, but their candidates need not have passed.
A failed A followed by a failed B followed by a verified C is a valid lineage.
Inconclusive, pending, unfunded and failed selected members cannot replay.

Export saves complete numerical geometry, result, traces, start bank and
verification endpoint. Reload checks TF/TFP versions, source files, device and
TF32/XLA policy, target identity/probes, record hashes, traces and endpoint.
Continuation recursively checks predecessor files and starts at the preceding
block's final active state, with a seed not used in tuning or previous blocks.
Keep predecessors available at their recorded paths. Files are not overwritten.

This is ordinary trusted-workspace provenance. The source list and target
signature must actually cover your model, data and prior. Probe agreement at
the start bank does not prove target math or completeness of a caller's lineage.

## Capability discovery and validation

`hmc_tuning_capability_registry_payload()["candidate_set_retained_bridge"]`
now reports:

```text
schema: bayesfilter.hmc_candidate_retained_member.v1
status: supported_with_repository_numerical_binding
binding_factory: bind_hmc_candidate_set_execution
preparation_factory: bind_hmc_candidate_set_execution_from_preparation
builders: the three requested public names
loader: load_hmc_candidate_retained_runner
coordinates: ordinary_affine_mass, frozen_affine_diag, frozen_dense_iaf
rhat_role: reporting_only
callback_observations_can_grant_numerical_authority: false
posterior_convergence_authority: false
```

The final affected tests contain **394 distinct passing cases**. The final
numerical candidate suite passes 37 tests, including a real repaired child,
two-layer nonidentity ordinary preparation, both frozen transport codecs,
wrong-source/target/kernel/tensor/endpoint controls, forged callback authority,
bad-R-hat reporting-only behavior, and continuation in a fresh interpreter.
Other affected suites cover ordinary compatibility, target-status tracing,
verification, public dispatch, artifact authority and documentation. The guide
and reference are updated, generated route documentation is consistent, the
inventory check passes, and the 557-page guidebook builds.

The actual CPU and trusted GPU/XLA examples, commands, environments, seeds,
source hashes, elapsed times and validation XML are committed under
`docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/`. The definitive GPU
run is `gpu-xla-02`: 65.38 seconds, memory growth enabled, actual XLA compilation,
two verified members, exact same-backend transition agreement and deterministic
CPU leapfrog/energy error below 1.8e-15. Its representative member has
`candidate_id="docs-bridge:example-1:candidate:000005"`, epsilon 1.3 and L=3.
The first archive starts at verification endpoint hash
`2a803f95d6feeefc320cc7e750e26f88eef159eb41e4ce37d4a4d154029fc095`.
The second starts at the first archive's final state hash
`9390a1fbbbd8bee4792ac3b17887f8a9d5029264328e7b7782f15b671d9e2998`.
These hashes differ, demonstrating continuation from the retained predecessor.

The first GPU attempt is preserved as a failed cross-mode random-stream
comparator, not passed validation. The corrected test compares draws within
one execution mode and checks deterministic proposals and energy separately
using captured momentum. Neither test ranks samplers or establishes a
MacroFinance posterior. No source in the final GPU execution closure changed
between that successful run and the implementation commit.

The exact commands and audit are in
`bayesfilter-typed-candidate-retained-bridge-execution-audit-2026-09-14.md`;
the executable integration example is
`docs/examples/hmc_candidate_set_retained.py`; API details are in
`docs/reference/hmc-tuning-interface.md` and guidebook Chapter 42.

Main implementation files are `hmc_candidate_set_execution.py`,
`hmc_candidate_set_retained.py`, the shared controller/artifact validator,
`hmc.py` health tracing, the public dispatchers, exports and `tuning_contract.py`.
The preparation math and underlying TFP HMC kernel were reused. The owner can
now proceed with MacroFinance's target-specific integration and checks under
its existing campaign authorization and budget; this memo itself is not a
new campaign launch.
