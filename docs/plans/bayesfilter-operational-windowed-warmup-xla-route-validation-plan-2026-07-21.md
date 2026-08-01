# Operational Windowed Warmup XLA Route Validation Plan

Date: 2026-07-21
Status: `PLAN_READY_FOR_EXECUTION`

## Scope

Validate the existing `operational_interleaved_windowed_warmup_v2` route under
TensorFlow/XLA so the PP-UKF public tuner can use its declared serious GPU
contract. This plan does not run PP-UKF claim sampling, does not change the
scientific target, and does not authorize a fallback to non-XLA or legacy
routes.

## Research Intent And Evidence Contract

| Field | Frozen decision |
| --- | --- |
| Question | Does the existing operational interleaved warmup execute safely under TensorFlow/XLA while preserving its typed window, target, epsilon, and coordinate-lineage contracts? |
| Candidate | The repository operational warmup implementation, with the smallest explicit XLA compilation control added to its existing `tf.function` runner if required by the implementation audit |
| Comparator | Existing eager and non-XLA `tf_function` operational warmup tests; these are mechanics comparators, not scientific baselines |
| Promotion criterion | Trusted GPU/XLA warmup completes with finite diagnostics, complete window schedule, passed artifact invariants, and explicit XLA metadata in the result |
| Hard vetoes | XLA compile/runtime error, nonfinite state/value/score/acceptance/step, missing target-status telemetry, discontinuous coordinate/metric/epsilon lineage, fixed-identity mutation, invalid artifact, or missing XLA provenance |
| Explanatory diagnostics | Runtime, acceptance, proposal tails, and metric update count; none establish posterior convergence or sampler superiority |
| Nonclaims | No PP-UKF posterior claim, no HMC convergence claim, no route superiority claim, no default-readiness claim, no production claim |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| Existing operational warmup remains the route | Owner route policy and current implementation | A new parallel route silently changes semantics | Route identity and artifact schema assertions | Required |
| XLA is enabled at the runner boundary | Repository XLA policy and public tuner config | XLA requested but only eager execution occurs | `tf.function` concrete function/JIT metadata and trusted GPU log | Required repair |
| Toy Gaussian target for route validation | Existing mechanics fixtures | Toy success is mistaken for PP-UKF validity | Artifact nonclaims and separate PP-UKF retry plan | Diagnostic only |
| Fixed identity mass | PP-UKF route policy | Adaptation mutates mass signature | Existing fixed-identity signature checks | Required |

## Skeptical Audit

- Wrong-baseline risk is controlled by testing the same operational warmup
  implementation in a declared mechanics target; no PP-UKF claim is inferred.
- Proxy promotion is forbidden: finite toy warmup is only route evidence, not
  posterior or sampler evidence.
- The current blocker is explicit route-contract validation, not a numerical
  failure. Removing it without XLA execution evidence would be invalid.
- The test must exercise the actual `tf.function` runner with
  `jit_compile=True`, not merely inspect a flag.
- Existing eager/non-XLA tests remain unchanged as parity authorities.
- Stop if the implementation needs a new algorithm, changes target semantics,
  changes mass policy, or cannot produce a complete XLA artifact.

Audit verdict: `PASS_FOR_IMPLEMENTATION_AND_FOCUSED_EXECUTION`.

## Implementation And Test Work

1. Add an explicit XLA compilation option to the operational warmup runner,
   defaulting to the current non-XLA behavior for existing callers.
2. Propagate the public windowed-stage `use_xla` setting to that runner.
3. Add focused trusted-GPU tests that run the operational warmup with XLA and
   assert finite windows, complete lineage, XLA runner metadata, and no mass
   mutation.
4. Update the route contract only after those tests pass, changing only the
   blocker for the validated operational warmup route.
5. Run the focused route/warmup/windowed-stage suites.

## Execution Boundary

Use trusted GPU permissions for the XLA test. Write only test/validation
artifacts under a fresh versioned root. Do not rerun PP-UKF until this plan
passes and a new PP-UKF tuning-only plan/result note is created.

## Required Artifacts

- This plan with the skeptical audit.
- Focused XLA route-validation test output.
- A fresh JSON validation record containing commit, command, GPU/memory
  policy, XLA setting, route identity, target fixture, result hash, and
  nonclaims.
- A result/reset note stating whether the route blocker was removed or
  preserved and why.
