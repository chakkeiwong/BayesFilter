# HMC Semantic Identity Migration Phase 2 Implementation Review, Iteration 1

Date: 2026-07-11

Review type: findings-first Codex implementation audit before repair.

Claude status: unavailable for this program after the governed one-path review
was rejected by the managed external-disclosure gate. The rejection is not
being retried or bypassed.

## Findings

1. `CanonicalArrayIdentityV1` records `dtype.str`, so equal native- and
   big-endian arrays can have different semantic dtype labels even though their
   canonical bytes are identical. Semantic dtype must be endian-independent.
2. `FrozenHMCTransitionIdentityV1` binds the whole fixture hash and the weak
   base-adapter signature instead of the actual observation values, parameter
   order, and mathematical source-contract fields consumed by the LGSSM
   target. The fixture also contains simulation and reporting provenance that
   is not part of one transition.
3. `FrozenHMCExecutionContractV1.execution_payload` accepts any non-empty
   mapping. It neither proves coverage of the Phase 7 controller nor rejects
   omitted, misspelled, or extra execution/stopping fields.
4. `SelectionProvenanceIdentityV1.provenance_payload` has the same open mapping
   defect. It should preserve a declared source schema, canonical full-payload
   hash, and typed named lineage references without retaining mutable arbitrary
   payloads.
5. `FrozenHMCTransitionIdentityV1.from_replay` permits callers to replace the
   kernel family and integrator route. Those are runner facts and must be
   hard-coded for schema V1.
6. String and digest validation is not fail-closed: `None` becomes the string
   `"None"`, prefixed hashes are checked only for their prefix, booleans can be
   accepted as integers, and nested mappings remain mutable after construction.
7. Tests do not cover endianness-independent semantic dtype, actual target
   mutations, complete SHA-256 syntax, `None`/whitespace rejection, immutable
   nested state, strict Phase 7 policy fields, or caller route-override denial.

These are engineering schema defects. They do not imply an HMC, target,
convergence, or scientific failure. They are fixable within Phase 2 without
changing serializer/validator behavior or adopting a baseline.

VERDICT: REVISE
