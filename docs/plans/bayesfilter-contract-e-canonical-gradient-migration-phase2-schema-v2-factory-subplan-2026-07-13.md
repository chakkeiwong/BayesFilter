# Phase 2 Subplan: Schema V2 And Non-Overridable Canonical Factory

Date: 2026-07-13

Status: `REVIEWED_ACTIVE`

Master program:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`

## Phase Objective

Define separate forward and score schema-v2 contracts whose canonical semantic
identity can be issued only by a repository-owned factory from actual callable
objects, actual prepared inputs, and a complete source-dependency closure. Keep
v1 artifacts historical and prove that caller metadata, a raw callable, or an
incomplete identity cannot enter an admission, leaderboard, default, or HMC
consumer.

Phase 2 establishes identity mechanics, not numerical or scientific
correctness. Because the production reset and full filter callables are created
in later phases, the factory must fail closed rather than issue an admitted
canonical identity from placeholders.

## Entry Conditions Inherited From Phase 1

- `contract_e_chol_v1` is the only reset contract eligible to seek canonical
  status.
- The derivative composition is
  `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.
- The row rule is `streaming_positive_transport_row_mass_quotient_v1`.
- The finite program uses likelihood increment before reset, normalized weights,
  population covariances, fixed residual design, and a parameter-independent
  prepared ridge.
- The complete pullback and coordinate conversions are defined in the Phase 1
  normative specification.
- V1 forward and score artifacts remain readable only as
  `historical_raw_barycentric_diagnostic_only`; no silent migration exists.
- Phase 1 numerical adequacy blockers remain binding at their named later
  phases. Schema v2 cannot discharge them.
- The platform-blocked Claude disclosure path is not retried or bypassed; a
  fresh bounded Codex reviewer is the approved substitute.
- Dirty work from other lanes remains untouched.

## Proposed Artifact And Factory Boundary

Add one repository-owned identity module and separate v2 validators/builders;
do not retrofit v1 payloads:

- `bayesfilter/highdim/ledh_contract_e_identity.py`: frozen identifiers,
  immutable factory product, deterministic tensor/prepared-input hashing,
  callable/source binding, and fail-closed implementation registry.
- `bayesfilter/highdim/ledh_forward_contract_v2.py`: v2 forward artifact
  validator/builder.
- `bayesfilter/highdim/ledh_score_contract_v2.py`: v2 score artifact
  validator/builder tied to the exact v2 forward identity digest.
- `tests/highdim/test_ledh_contract_e_schema_v2_factory.py`: positive identity
  mechanics on a private test fixture and negative forgery/reachability tests.

If inspection shows that a smaller module split is clearer, the result must
record the deviation before review. Existing v1 constants, validators, and
builders remain behaviorally unchanged except for a narrow version-dispatch
helper if a consumer requires one.

The factory product must bind at least:

- route factory ID;
- reset contract ID;
- derivative composition ID;
- row-normalization policy ID;
- residual-design ID plus hash of the realized tensor and its dtype/shape;
- ridge-policy ID plus hash of the realized prepared ridge and its dtype/shape;
- a repository-owned registered route-specification ID and extractor which
  enumerates every mandatory prepared-input field, its semantic role, accepted
  dtype/rank/shape relation, and whether any extra field is permitted;
- hash of every prepared-input value required by that registered specification,
  using sorted field names and an unambiguous dtype/shape/byte encoding; missing
  and extra fields are rejected rather than delegated to caller declaration;
- target scalar, output field, parameter coordinate system, parameter names and
  ordering, issued by the same registered route specification rather than copied
  from an artifact payload;
- exact module, qualified name, source file, and source-file hash for reset,
  value, and gradient callables;
- deterministic BayesFilter-owned dependency-manifest entries and closure hash,
  plus repository-owned external-primitive provenance entries; and
- one identity digest over the canonical serialization of all preceding fields.

Callers supply actual callables and a mapping of actual prepared values, never an
identity string, semantic field, required-field list, source/tensor hash, or
dependency entry. The registered route specification owns target and parameter
semantics and extracts/validates the complete prepared-input mapping. The factory
computes every identity field. Unknown dynamic BayesFilter dependencies,
unserializable prepared values, missing source, source/code mismatch, or an
unregistered exact symbol must block issuance.

The dependency boundary is explicit. For BayesFilter-owned callables, the
factory conservatively walks the registered code-object dependency graph,
records module/qualified-name/source-file/source-file-hash plus a code-object
digest, and rejects unresolved repository-owned calls. It verifies that the
loaded code object's source segment and bytecode-derived digest correspond to
the inspected module/source state, so editing a file after import cannot be
silently represented by the new on-disk hash. TensorFlow, TensorFlow Probability,
Python standard-library, and other reviewed external primitives are not treated
as BayesFilter source files: a repository-owned allowlist binds their module
roots, distribution/version provenance, and allowed role. An unallowlisted
external primitive or missing/version-ambiguous provenance blocks issuance.
Callers cannot extend either closure or allowlist.

The production registry is initially empty or names only exact future symbols
that do not yet resolve. A private test-only registry may be dependency-injected
only into a private factory constructor in the test module; it must be
unreachable from public artifact builders. The public factory must reject raw,
benchmark, experiment, wrapper, lambda, partial, monkeypatched, and caller-
annotated callables. Phase 3 and Phase 5 will activate the exact reset and
value/gradient symbols after their implementations and tests exist.

## Schema V2 Rules

The proposed versions are:

- `bayesfilter.highdim.ledh_forward_scalar_artifact.v2`;
- `bayesfilter.highdim.ledh_score_artifact.v2`;
- `bayesfilter.highdim.contract_e_route_identity.v2`.

An artifact builder accepts a factory product object, not an arbitrary identity
mapping. A serialized artifact contains the full identity record and digest for
audit, but a validator may declare it canonical only when an independently
recomputed expected factory product is supplied and every byte-relevant field
matches. Self-asserted serialized metadata is provenance only.

Forward and score artifacts must share exactly the same route digest, prepared-
input digest, callable dependency closure, row ID, target scalar, parameter
coordinate system, and parameter ordering. Those target/parameter fields are
issued by the registered route specification and independently compared during
validation, never jointly trusted because two payloads agree. The score schema
additionally binds the same-scalar value callable and the total-gradient
callable. Missing evidence gates produce an explicit candidate/blocked state,
never admission by omission.

No v2 artifact may claim admitted status in Phase 2 because the production
callables and later numerical gates do not yet exist. The only permitted Phase 2
state is mechanically valid identity candidate or explicitly blocked; neither
is leaderboard/default/HMC eligible.

## Skeptical Plan Audit

Decision: `PASS_FOR_SCHEMA_ONLY_WITH_INERT_PUBLIC_FACTORY`.

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | V1 is historical input for negative compatibility tests, not a template to upgrade. Phase 0 and Phase 1 identities are the authority. |
| Proxy promotion | Matching hashes proves identity, not reset math, same-scalar gradient correctness, Kalman agreement, memory feasibility, or HMC readiness. V2 admission remains impossible in this phase. |
| Caller-forged identity | Validators never trust a payload's identity alone. They compare against an independently factory-issued object whose fields were computed from actual objects. |
| Placeholder callables | The public implementation registry stays unresolved/inert until exact production symbols exist. A benchmark helper cannot stand in for production. |
| Incomplete dependency closure | The factory blocks unknown dynamic calls and records every source file used to compute the closure. A caller cannot submit a closure list. |
| External dependency ambiguity | A repository-owned allowlist binds external module roots and installed distribution/version provenance; unallowlisted or ambiguous primitives block issuance without pretending TensorFlow source is part of the BayesFilter closure. |
| Stale loaded code | Both source-segment/file state and code-object digests are checked; a callable whose loaded code cannot be tied to inspected source blocks issuance. |
| Hash ambiguity | Canonical serialization includes type tags, sorted names, dtype, rank, shape, byte length, and bytes before hashing. Plain string concatenation is forbidden. |
| Stale prepared input | The realized residual design, ridge, and all declared prepared tensors are hashed, not merely their seed/config labels. |
| Caller-omitted semantics | A repository-owned route specification owns required prepared fields and target/parameter semantics; missing/extra values and payload self-labels fail closed. |
| Separate value/score semantics | Score validation requires exact forward identity and same route/prepared/source digests. |
| Environment mismatch | Phase 2 is pure schema/CPU work. No GPU evidence or XLA claim is needed or allowed. |
| Dirty worktree collision | Inspect status and hashes immediately before each edit; add new modules/tests where possible and do not touch dirty model harnesses. |
| Commands not answering the question | Focused tests exercise construction, serialization, mutation, raw-callable rejection, v1 behavior, and real consumer rejection. No benchmark is run. |

The audit found no reason to run a numerical experiment. It did find that a
factory which accepts caller-supplied hashes or issues an identity for a
placeholder would not answer the phase question; both actions are forbidden.

## Required Artifacts

- The identity/factory and v2 contract modules above.
- Focused v2 schema/factory tests.
- A machine-readable schema/factory manifest containing exact public symbols,
  schema fields, consumer reachability, and source hashes.
- Phase result:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase2-schema-v2-factory-result-2026-07-13.md`.
- Focused check log under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase2/`.
- Phase 3 cloud-level implementation subplan.
- Updated execution ledger and stop handoff.

## Required Checks, Tests, And Reviews

1. Inventory v1 validators/builders and every adapter, aggregator, leaderboard,
   default, and HMC-facing consumer from the Phase 0 manifest; refresh stale line
   anchors.
2. Unit-test deterministic serialization across mapping order and equivalent
   immutable tensor values, and inequality after any dtype, shape, byte, field,
   callable, or dependency change.
3. Prove direct construction or mutation of the immutable identity product is
   rejected or cannot affect its digest.
4. Prove the public factory rejects raw/benchmark/experimental/wrapper/lambda/
   partial/unregistered callables, caller-provided identity fields, missing
   source, stale loaded code, unresolved BayesFilter dependency closure,
   unallowlisted external primitives, and ambiguous external version provenance.
5. Prove the registered route specification, not the caller, owns the exhaustive
   prepared-input names/types and target/parameter semantics; missing, extra,
   wrong-dtype, wrong-rank, wrong-shape-relation, and self-labeled fields fail.
6. Prove a forged v1 or v2 payload cannot become canonical even if every visible
   ID string is copied from the freeze manifest.
7. Prove forward/score identity mismatch, prepared-input mismatch, source closure
   mismatch, row/parameter mismatch, and missing gate fields fail closed.
8. Prove current adapters, inclusive aggregator, leaderboard consumer, and
   default/HMC entry paths reject an unadmitted v2 candidate.
9. Re-run the Phase 0 revocation suite and clean central v1 contract suites to
   prove no compatibility regression.
10. Run Python compilation, manifest JSON validation, source/path/hash checks,
   and `git diff --check`.
11. Obtain a bounded fresh read-only review of identity completeness,
    non-overridability, serialization ambiguity, consumer reachability, and
    Phase 3 handoff. Repair and repeat for material findings, up to five rounds.

All TensorFlow-importing tests must set `CUDA_VISIBLE_DEVICES=-1`. No GPU,
benchmark, HMC, or scientific run is authorized in Phase 2.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can schema v2 mechanically distinguish a factory-bound Contract E candidate from raw or caller-labeled artifacts without claiming implementation correctness? |
| Exact comparator | Frozen Phase 0 identity fields, Phase 1 semantics, current v1 validators/builders, raw callables, and every inventoried consumer. |
| Primary criterion | A repository-owned route specification owns exhaustive inputs and scientific semantics; only the factory computes IDs/hashes from actual objects and checked dependency provenance; serialized artifacts require independent factory recomputation; all forgery and reachability tests fail closed. |
| Promotion vetoes | Caller-supplied semantic hash; raw/unknown callable accepted; incomplete dependency closure; mutable identity; forward/score digest mismatch; v1 silent upgrade; or any Phase 2 artifact marked admitted. |
| Repair triggers | Serialization collision, missing consumer, public test bypass, stale source anchor, v1 regression, or review-identified ambiguity. |
| Continuation vetoes | A reproducible actual-object identity cannot be serialized; the required dependency closure cannot fail closed; unrelated concurrent edits overlap an in-scope file; five material review rounds do not converge; or campaign budget exhaustion. |
| Explanatory only | Digest values, payload size, validation runtime, and number of dependencies. |
| Not concluded | Contract E math correctness, production callable existence, numerical adequacy, same-scalar FD, Kalman agreement, streaming feasibility, HMC readiness, leaderboard completeness, or release readiness. |

## Forbidden Claims And Actions

- Do not relabel or auto-migrate v1 artifacts.
- Do not accept identity strings, scientific/parameter semantics, required-input
  lists, source hashes, tensor hashes, dependency lists, or external allowlist
  entries as caller authority.
- Do not use the dense benchmark helper as the registered production reset.
- Do not claim a Python object type or underscore name alone prevents forgery;
  validation must recompute and compare the actual binding.
- Do not issue an admitted canonical identity before exact production callables
  and later evidence gates exist.
- Do not change a public default, numerical threshold, model file, or leaderboard
  result.
- Do not implement the production reset, streaming composition, or full filter in
  this phase.
- Do not edit dirty model-specific harnesses from other lanes.
- Do not launch GPU, HMC, nonlinear, leaderboard, detached, or long commands.

## Exact Next-Phase Handoff Conditions

Phase 3 may begin only if:

- schema-v2 fields and canonical serialization are complete and unambiguous;
- the public factory is non-overridable and inert for unresolved production
  symbols;
- the registered route specification rejects omitted/extra prepared fields and
  independently issues target/parameter semantics;
- deterministic actual-object hashing, loaded-source/code correspondence,
  BayesFilter dependency closure, and external provenance checks pass;
- raw, forged, mutated, mismatched, v1-upgraded, and unregistered identities
  fail closed;
- adapters, aggregators, leaderboard, default, and HMC-facing reachability tests
  reject unadmitted candidates;
- v1 historical readability/revocation behavior is unchanged;
- no Phase 2 artifact claims admission or numerical/scientific correctness;
- the schema/factory manifest, result, logs, and run manifest are complete;
- a bounded review converges; and
- the Phase 3 subplan names the exact reset symbols that will activate only the
  reset portion of the registry, carries every Phase 1 numerical blocker, and is
  reviewed for consistency and feasibility.

## Stop Conditions

Stop and write a blocker result if actual-object identity cannot be made
reproducible without trusting caller omissions, the dependency boundary cannot
be made complete and feasible, a required repair would cross the phase or owner
authority boundary, an in-scope concurrent edit appears, five material repair
rounds fail to converge, or the campaign budget cannot support a valid close
record. A raw digest collision, accepting consumer, v1 regression, or other
ordinary focused-test failure is first a repair trigger and promotion veto; stop
only if it remains after the allowed repair loop or reveals one of the preceding
continuation vetoes.

## Phase-End Protocol

1. Run all focused CPU-hidden checks and source/manifest audits.
2. Write the Phase 2 result or blocker result with decision and inference-status
   tables plus a run manifest.
3. Draft or refresh the Phase 3 cloud-level implementation subplan.
4. Review the result and next subplan for identity coverage, boundary safety,
   correctness, feasibility, and preservation of Phase 1 blockers.
5. Patch visible material findings and rerun focused checks, up to five rounds.
6. Update the ledger and stop handoff.
7. Advance only when every exact handoff condition passes.
