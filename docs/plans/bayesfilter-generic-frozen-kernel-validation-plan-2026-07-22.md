# Generic Frozen-Kernel Validation Plan

Date: 2026-07-22

Status: `COMPLETE_FOCUSED_IMPLEMENTATION`

## Objective

Create one reusable BayesFilter Python orchestration function for validating
frozen HMC/NeuTra candidate controls across PP-UKF and future model adapters.
The generic layer must enforce identity, tuning scope, fresh validation data,
control provenance, hard-veto accounting, artifact completeness, and unranked
candidate promotion without assuming PP-UKF-specific names or mathematics.

The model adapter remains responsible for evaluating its target and executing
its fixed kernel. The generic function must not retune, alter the target,
select a winner from descriptive metrics, or claim posterior correctness.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can a common executor validate frozen candidate controls for multiple model adapters without losing scope or evidence provenance? |
| Mechanism under test | A typed candidate/scope/artifact contract plus callback-based frozen-kernel execution |
| Primary criterion | Every candidate result is either a complete viable record or an explicit hard-veto/failure record; viable candidates are the unranked next-round set |
| Promotion veto | Scope mismatch, stale/missing tuning artifact, validation-data overlap, control/provenance mismatch, missing required diagnostics, non-finite adapter status, or explicit adapter hard veto |
| Continuation veto | Contract construction failure, duplicate candidate identity, invalid artifact binding, or invalid validation partition |
| Explanatory diagnostics | Acceptance, ESS, R-hat, runtime, tail summaries, reference gaps, and model-specific metrics unless explicitly declared as hard vetoes |
| Ranking | Forbidden by this generic executor; selection requires a separate reviewed policy |
| Nonclaims | No posterior convergence, sampler superiority, default readiness, model correctness, or scientific validity |

## Evidence Contract

- Baseline: the adapter's fixed-kernel implementation under the exact frozen
  candidate controls; no cross-model baseline is silently invented.
- Candidate identity: model ID, target signature, tuning-scope signature,
  control values, provenance, parent relation, and execution seed.
- Tuning binding: a repository-issued tuning artifact must match model,
  target, tuning scope, and artifact signature exactly.
- Data binding: calibration/tuning and validation data signatures must differ;
  the validation partition identity is recorded and cannot be omitted.
- Required diagnostics: the policy names the required adapter diagnostics;
  missing values are a hard veto, not a pass.
- Artifact: the result contains candidate records, scope, tuning binding,
  validation partition, policy, command/environment fields supplied by the
  caller, and explicit nonclaims.
- Statistical interpretation: continuous metrics are descriptive unless a
  separate reviewed uncertainty policy promotes them. The executor does not
  rank candidates.

## Generic API Shape

Add `bayesfilter/inference/frozen_kernel_validation.py` with immutable records:

- `FrozenValidationCandidate`: generic control mapping, provenance, optional
  parent candidate, and fixed execution seed.
- `FrozenTuningArtifactBinding`: model/target/scope-bound artifact identity.
- `FrozenValidationScope`: model, target, tuning scope, validation partition,
  calibration partition, dtype/backend, and data signatures.
- `FrozenValidationPolicy`: required diagnostics, candidate/attempt bounds,
  and nonclaims.
- `FrozenValidationObservation`: adapter status, hard vetoes, repair triggers,
  diagnostics, and runtime.
- `FrozenKernelValidationResult`: candidate records, viable unranked set,
  hard-veto records, and an artifact payload.

The public executor will be:

```python
run_frozen_kernel_validation(
    *,
    candidates,
    tuning_artifact,
    scope,
    policy,
    runner,
) -> FrozenKernelValidationResult
```

`runner(candidate, scope, seed)` is the only model-specific execution hook.
The executor catches and records candidate-local exceptions so one failed model
candidate does not erase independent candidates. It does not catch invalid
contract construction or scope violations.

## Implementation Phases

1. **Contract implementation**
   - Add immutable records and strict validation.
   - Add deterministic candidate identity and payloads.
   - Enforce exact inherited-control provenance for parent-linked candidates.
2. **Execution orchestration**
   - Validate the tuning artifact and fresh partition binding.
   - Run candidates independently through the adapter callback.
   - Convert missing diagnostics and explicit adapter vetoes into candidate
     hard-veto records.
   - Expose viable candidates without ranking.
3. **PP-UKF integration seam**
   - Add a conversion helper from the existing PP-UKF next-round records into
     generic candidates, preserving primary versus inherited-epsilon metadata.
   - Do not launch the expensive PP-UKF validation campaign in this phase.
4. **Focused tests**
   - Scope/artifact mismatch rejection.
   - Calibration/validation overlap rejection.
   - Duplicate and malformed candidate rejection.
   - Exact inherited-control enforcement.
   - Candidate-local failure isolation.
   - Missing diagnostics and explicit veto classification.
   - Unranked viable union and deterministic payload identity.
5. **Documentation and closeout**
   - Record the skeptical audit and implementation limits.
   - Mark this plan complete only after focused tests pass.
   - Create a separate PP-UKF validation campaign plan before any GPU run.

Implementation result: the generic executor and immutable records are in
`bayesfilter/inference/frozen_kernel_validation.py`. The PP-UKF benchmark
driver now exposes `build_pp_ukf_frozen_validation_candidates`, which preserves
primary/coverage provenance and exact inherited epsilon while converting the
existing next-round records into the generic contract.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Callback adapter boundary | Existing model-specific benchmark drivers | Adapter may silently retune or mutate controls | Runner receives immutable candidate and tests assert exact controls |
| Required diagnostics declared by policy | Model differences make one universal metric set invalid | Missing metric could be mistaken for success | Missing key becomes a hard veto |
| Scope signatures are caller-provided bindings | Existing artifact/target signature patterns | Caller could stamp a false signature | Artifact and candidate signatures must match; mathematical target checks remain adapter responsibility |
| Validation and tuning partition signatures | Existing tuning-scope governance | Reused data creates optimistic evidence | Identical or missing signatures fail closed |
| No generic ranking | Statistical evidence standards prohibit descriptive ranking | More viable candidates remain for later reviewed selection | Result explicitly records `ranking_performed=False` |
| Candidate-local exception capture | Independent candidates should not be erased by one infrastructure failure | A shared failure may be misread as candidate failure | Exception is recorded per candidate; campaign result remains partial/inconclusive |

## Skeptical Plan Audit

- **Wrong baseline:** the executor does not define a universal model baseline;
  each adapter supplies the exact frozen-kernel comparator.
- **Proxy promotion:** acceptance, ESS, R-hat, runtime, and reference metrics
  are not promoted unless explicitly declared required or veto diagnostics by a
  model-specific reviewed policy.
- **Missing stop conditions:** invalid scope, partition overlap, artifact
  mismatch, duplicate identity, and missing required diagnostics fail closed;
  candidate-local failures are preserved without stopping unrelated candidates.
- **Unfair comparisons:** all candidates use the same scope, validation data,
  policy, and provenance checks; model adapters cannot silently change controls.
- **Environment mismatch:** backend, dtype, and device metadata are bound in
  the scope and must be echoed by the adapter record.
- **Artifact insufficiency:** the result preserves candidate identity,
  controls, tuning artifact, partitions, diagnostics, vetoes, nonclaims, and
  deterministic payload identity.
- **Stale context:** PP-UKF's ten-value next-round set is an adapter input,
  not a generic default. Other models must provide their own candidates.

Audit decision: `PASS_FOR_FOCUSED_IMPLEMENTATION`. The remaining GPU campaign
and model-specific scientific validation are explicitly out of scope and need
their own plan, budget, and fresh validation artifacts.

## Focused Execution Result

- Python compilation passed for the new module, lazy export surface, adapter
  seam, and tests.
- Four generic contract tests passed: viable unranked union, scope/partition
  vetoes, missing-diagnostic/control-mutation vetoes, and candidate-local
  failure isolation.
- Existing PP-UKF operational and statistical compatibility tests also passed.
- No GPU or long validation run was launched. The ten-candidate PP-UKF run
  still requires a separate fresh-data campaign plan and compute budget.

## Verification And Stop Conditions

Run only focused CPU-safe tests in this implementation phase. Stop before any
GPU or long validation run. Completion requires the new module to compile, all
focused tests to pass, and no unrelated worktree changes to be staged.

## Planned Nonclaims

This plan creates a reusable validation contract. It does not establish that
any model's target, transport, HMC implementation, convergence, posterior, or
scientific claim is correct. It does not authorize the PP-UKF ten-candidate
GPU validation run.
