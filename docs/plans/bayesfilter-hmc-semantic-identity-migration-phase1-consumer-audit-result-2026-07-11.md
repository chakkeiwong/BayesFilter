# Phase 1 Result: Runtime Consumer Audit And Field Classification

Date: 2026-07-11

Status: `PASSED_TO_PHASE2_SCHEMA_IMPLEMENTATION`

## Direct Verdict

The Phase 7 transition is determined by a graph-native LGSSM base target, two
ordered affine mass transforms, a float64 TFP Hamiltonian Monte Carlo kernel,
the exact float64 step size, and the leapfrog count. The prior whole-payload
hash includes many fields that do not affect that transition.

The audit found no unclassified value consumed by the current replay or Phase 7
runtime. It did find one material correction to the initial proposal: both the
Phase 4 geometry mass and the final adapted mass affect the latent target and
must be bound by transition identity. Hashing only the final adapted mass would
be wrong relative to the executed transition.

## Consumer Graph

```text
fixture + source contract + XLA evidence
  -> DeterministicLGSSMPosteriorAdapter (base target)
  -> reconstructed geometry PrecomputedMassArtifact
  -> Phase 4 affine latent adapter
  -> final adapted PrecomputedMassArtifact
  -> final affine latent adapter
  -> FixedSizeHMCChunkConfig + current state + per-chunk seed/count
  -> FixedSizeHMCChunkRunner
  -> TFP HamiltonianMonteCarlo one_step in a JIT-compiled tf.while_loop
```

Source anchors:

- Base target construction and replay call:
  `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:618`.
- Replay geometry and Phase 4 transform reconstruction:
  `bayesfilter/inference/hmc_kernel_tuning.py:4044`.
- Final adapted-mass and final transform reconstruction:
  `bayesfilter/inference/hmc_kernel_tuning.py:4087`.
- Worker initial-state and fixed-size runner construction:
  `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:641`.
- Dynamic transition inputs:
  `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:688`.
- Fixed runner float64 state and target construction:
  `bayesfilter/inference/hmc.py:2916`.
- Runtime seed, step, state, and active-count conversion:
  `bayesfilter/inference/hmc.py:2947`.
- HMC chunk static configuration fields:
  `bayesfilter/inference/hmc.py:637`.

## Field Classification

Primary role means the schema that owns the raw value. Other schemas may bind
the owning schema's hash, but must not duplicate and independently normalize
the raw value.

### Transition Identity

| Field/object | Why transition-bearing | Source consumer |
| --- | --- | --- |
| Transition schema version | Defines canonical semantics and fail-closed compatibility. | New typed builder consumed by worker. |
| Kernel family `tfp.mcmc.HamiltonianMonteCarlo` and leapfrog integrator route | Changes the transition proposal. | `bayesfilter/inference/hmc.py:2896` and runner metadata/tests. |
| Target scope | Binds authority and target route. | `bayesfilter/inference/hmc_kernel_tuning.py:4194`; `bayesfilter/inference/hmc.py:2914`. |
| Base adapter signature, class/module, and runtime backend | Identifies the graph-native target implementation. | `bayesfilter/inference/hmc.py:5174`; driver adapter capability at `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py:187`. |
| Target-bearing fixture hash | Adapter signature currently binds observation shape, not observation values; the fixture changes the target. | Worker reads observations at `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:618`. |
| Target dimension and parameter order | Changes state interpretation. | Driver adapter signature and fixture parameter names; worker dimension at `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:642`. |
| State/target dtype `float64` | Changes numerical transition. | `bayesfilter/inference/hmc.py:2918`; base target casts at driver line 308. |
| Phase 4 transform center, factor, orientation, log-Jacobian convention, and runtime route | First map from Phase 4 latent space to raw target coordinates. | `bayesfilter/inference/hmc_kernel_tuning.py:4071`; transform application at `bayesfilter/inference/hmc_kernel_tuning.py:8411`. |
| Final transform center, factor, orientation, log-Jacobian convention, and runtime route | Second map from final HMC coordinates to Phase 4 coordinates. | `bayesfilter/inference/hmc_kernel_tuning.py:4087`; fixed-mass builder at `bayesfilter/inference/hmc_budget_ladder.py:1479`. |
| Canonical transform-array dtype, shape, byte order, and bytes hash | Prevents list/JSON normalization from hiding array representation changes. | Both runtime transforms consume centers/factors directly. |
| Exact float64 step bits | Runtime converts this scalar to state dtype for every transition. | `bayesfilter/inference/hmc.py:2967`; Phase 7 call at `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:696`. |
| Leapfrog count | Changes integration length and proposal. | Static chunk config at `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:648`. |

The covariance arrays and explanatory mass metadata need not be duplicated in
transition identity when the exact runtime transform center/factor bytes and
validated mass/adapter signatures are bound. Artifact integrity still covers
the full mass payload. Any future runtime that consumes covariance directly
must revise the transition schema.

Existing full transform-adapter signatures are reconstruction-integrity
cross-links, not raw transition-identity owners. Their current construction
includes full mass-artifact signatures, which include provenance and nonclaims.
Placing them directly inside transition identity would recreate provenance-only
hash drift. Transition identity therefore owns adapter runtime route, ordered
base relation, and exact transform mechanics; replay reconstruction separately
requires the historical full signatures to match.

### Execution Contract

| Field/object | Why execution-bearing | Source consumer |
| --- | --- | --- |
| Transition identity hash | Binds the run to one transition. | New execution contract. |
| Initial-state offset formula, range, alternating-sign pattern, and global chain order | Changes starting states and reproducibility. | `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:643`. |
| Root seed and derivation formula, including compile-probe seed | Changes random transition sequence. | `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py:112` and line 215. |
| Worker count, chains per worker, chain order, persistent-process requirement | Changes partition and deterministic orchestration. | Config validation lines 74-100 and worker rounds lines 814-860. |
| CPU visibility and thread environment | Changes execution environment/reproducibility. | Worker environment lines 921-929. |
| XLA/JIT, `tf_function`, sequential worker compilation | Changes execution route. | Config validation lines 80-100 and worker chunk config lines 648-659. |
| `max_results`, active chunk counts, burn-in/retained schedules, check window, and caps | Change compiled shape, transition sequence grouping, and stopping behavior. | `_runtime_counts` lines 750-775 and controller loops lines 231-377. |
| Diagnostic definitions and thresholds | Do not change one transition but change run termination/promotion. | `_aggregate_diagnostics` lines 778-811. |
| Wall-time cap and no-resume policy | Change machine termination and continuation semantics. | `_remaining_wall_time` lines 1078-1087 and master plan. |
| Trace policies and target-status trace policy | Static runner route and diagnostics availability. | Chunk config lines 654-657. |
| TensorFlow/TFP/Python versions | Reproduction environment; not mathematical transition parameters. | Worker metadata lines 733-745. |

### Selection Provenance

| Field/object | Role |
| --- | --- |
| Tuning config, target acceptance, acceptance/repair bands, candidate policy | Explains selection, not retained transition once step/leapfrog/mass are fixed. |
| Bootstrap/window/fixed-step/trajectory stage hashes and candidate indices | Historical lineage. |
| `handoff_screen_policy` and trajectory-window policy | Selection-policy provenance; this caused the legacy hash drift. |
| Screen/tuning/verification seeds and budgets | Historical evidence for selection; Phase 7 uses a new execution seed schedule. |
| Acceptance observations, diagnostics, repair triggers, timings | Explanatory/veto history, not transition mechanics. |
| Reviews, timestamps, nonclaims, artifact paths | Governance and reporting provenance. |

### Artifact Integrity

| Artifact | Integrity requirement |
| --- | --- |
| Private replay | Exact file SHA-256 plus canonical embedded artifact hash and byte count. |
| Public kernel artifact | Canonical embedded hash; public private-replay reference must match file SHA-256 and bytes. |
| Migration certificate | Canonical embedded hash and explicit source-artifact hashes. |
| Phase 7 config and results | Canonical embedded hash; private retained archive separately reopens and verifies. |

### Excluded Or Derived

| Field/object | Classification |
| --- | --- |
| `trajectory_length` | Derived from exact step and leapfrog count; may be asserted, not independently owned. |
| Acceptance and elapsed time | Explanatory only. |
| Mass covariance/eigen summaries | Integrity/provenance unless a runtime starts consuming covariance directly. Transform factor bytes are transition-bearing. |
| Private/public legacy final-kernel hashes | Historical integrity/provenance after an approved migration; currently remain the active blocker. |
| Public redaction booleans and nonclaims | Governance provenance. |

## Canonicalization Requirements

- Arrays must be normalized to a declared C-contiguous canonical byte order,
  with original/semantic dtype, canonical dtype string, shape, byte-order rule,
  and SHA-256 recorded. JSON lists alone are insufficient.
- Step size must use the exact float64 IEEE-754 bits used by the runtime, not a
  rounded decimal string.
- Unknown schema versions and unknown transition fields fail closed.
- Identity builders operate on validated replay objects and runtime config
  objects. Raw private JSON is used for artifact integrity and provenance, not
  as an independent mechanical projection.
- Execution contract binds `transition_identity_hash_v1`; it does not duplicate
  transition arrays or step mechanics.

## Evidence Decision

| Decision | Status |
| --- | --- |
| Primary criterion | Passed: all current replay/runtime consumers are classified and source-anchored. |
| Material finding | Both ordered mass transforms must be bound; final mass alone is insufficient. |
| Unknown execution fields | None in the current route. Future unknown schema fields fail closed. |
| Old/new transition equality | Not checked in Phase 1. |
| Next action | Implement typed identity schemas and canonical hashing from validated runtime objects. |
| Not concluded | No baseline adoption, Phase 7 readiness, convergence, recovery, or scientific claim. |

## Review

Fresh Codex substitute review is required because the managed Claude rejection
applies to this program's workspace review gates. The Phase 2 subplan must pass
that review before source edits.
