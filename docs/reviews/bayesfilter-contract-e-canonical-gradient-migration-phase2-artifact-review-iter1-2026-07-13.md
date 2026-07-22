# Contract E Phase 2 Artifact Review, Iteration 1

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute after Claude repository disclosure was
platform-blocked. Scope was the forward and score v2 contract modules.

## Material Finding

The public evidence-gate and score-correctness dictionaries were mutable. A
caller could mutate the module objects, after which builders emitted and
validators accepted the same promoted state.

## Verdict

`VERDICT: REVISE`

## Repair

The canonical specifications are now private immutable tuples. Public helpers
return fresh serialization dictionaries, builders use fresh copies, validators
reconstruct the canonical copies, and a regression test mutates returned copies
without changing later results. A focused iteration-2 review is required.

## Iteration 2 Follow-Up

The first repair protected module-level canonical specifications, but the
iteration-2 reviewer found that validators still returned shallow outer copies
whose nested identity, gate, correctness, and nonclaim objects were caller-
owned. The validators now rebuild every normalized field from checked scalar
values, the independently issued identity, and fresh canonical helper outputs.
A regression test mutates inputs and normalized outputs in both directions and
checks that neither aliases the other. An iteration-3 review is required.
