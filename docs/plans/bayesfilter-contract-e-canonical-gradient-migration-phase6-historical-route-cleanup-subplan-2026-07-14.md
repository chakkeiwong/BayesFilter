# Phase 6 Subplan: Mechanical Historical-Route Cleanup

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `DRAFT_PENDING_HANDOFF_REVIEW`

## Phase Objective

Make every executable raw-barycentric LEDH score entry point explicitly
historical and fail closed. Raw compact-sensitivity and full-history/manual
reverse routes may remain runnable only after an explicit diagnostic opt-in and
must emit or normalize to
`historical_raw_barycentric_diagnostic_only`. They must never emit a full,
tiny-admitted, default, leaderboard, canonical, or HMC-facing status, and a
Contract E error must never fall back to them.

This phase changes route classification and reachability only. It does not
rewrite historical derivatives, register a canonical factory route, implement
Contract E for nonlinear models, regenerate results, or use historical output
as scientific evidence.

## Entry Conditions Inherited From Phase 5

- Contract E--Chol remains the only reset eligible for canonical use.
- Phase 5 passed the checked tiny one-graph derivative gate at `0 ULP` but all
  numerical/scientific promotion blockers remain open.
- The production v2 factory is empty and all v2 artifacts remain unadmitted.
- Central v1 forward/score validators already normalize raw routes to
  historical status and reject them for admission.
- Historical benchmark entry points and clean tests still contain obsolete
  local `FULL`, `TINY`, default, or compact-admitted expectations.
- No historical route may serve as a canonical implementation fallback.
- The current continuation clock remains
  `2026-07-14T01:32:19+08:00` through approximately
  `2026-07-14T09:32:19+08:00`.

## Frozen Scope And Artifacts

Before code edits, create a machine-readable inventory of exact path/symbol
roots. It must cover public imports/APIs, CLIs and defaults, dispatchers,
artifact constructors, canonical factories, aggregators/leaderboard consumers,
and exception/fallback handlers. Preserve every discovery query and classify
every hit as `edit`, `test`, `central_guard`, `historical_data`, or a justified
exclusion. The initial named roots include the LGSSM score mode in
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`, the five
historical nonlinear score harnesses, their route-contract tests, the
cross-model wiring test, central v1 validators/emitters, v2 factory registries,
and inclusive leaderboard consumers. Phase 6 cannot close with an unclassified
discovery hit.

The implementation should use the smallest mechanical pattern that applies to
all reachable historical routes:

1. an explicit CLI/API diagnostic opt-in is required before raw score work;
2. local artifact construction emits the central historical status constant;
3. obsolete compact/full admission branches are removed rather than retained
   behind a condition;
4. historical route identity and wrong-target nonclaim are explicit; and
5. tests bind every executable inventory root to the applicable no-opt-in,
   exact-historical-status, aggregate-rejection, and no-fallback obligations.

If a named file has changed unexpectedly after its Phase 6 hash inventory, stop
that edit and request direction rather than overwrite concurrent work.

## Research Intent Ledger

| Field | Binding Phase 6 intent |
| --- | --- |
| Main question | Can every remaining raw route be made explicitly diagnostic-only at its executable boundary, in addition to central validation? |
| Candidate/mechanism | Mechanical status replacement plus explicit diagnostic opt-in and negative reachability tests |
| Expected failure mode | A local harness still emits `FULL`/compact-admitted, a default silently selects raw, a forged field passes, or Contract E failure falls back to raw |
| Primary criterion | All classified executable route roots fail closed without opt-in and emit the exact historical status with opt-in; discovery has zero unclassified hits |
| Promotion veto | Any raw default/admission result, fallback, forged canonical identity, or changed historical mathematics |
| Continuation veto | Required cleanup cannot be separated from new scientific implementation, concurrent in-scope edit, or campaign budget exhaustion |
| Repair trigger | Focused test exposes a remaining reachable obsolete branch or consumer |
| Explanatory diagnostic | Counts and locations of historical symbols; no scientific promotion role |
| Must not be concluded | Canonical implementation correctness, Kalman agreement, nonlinear readiness, HMC, leaderboard, or release readiness |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Exact historical status string | Repository Contract E policy and central score contract | One fail-closed classification avoids consumer ambiguity | Local aliases may drift | Import central constant and assert exact output | Required |
| Explicit diagnostic opt-in | Master Phase 6 contract | Prevents raw route from remaining an implicit default | Tests or internal helpers bypass CLI | Exercise CLI/parser and direct artifact boundary | Required |
| No historical-math rewrite | Phase objective | Scientific target is cleanup, not repair of a wrong target | Scope drifts into nonlinear implementation | Diff/source audit | Required |
| No canonical fallback | Owner directive | Raw scalar is wrong relative to canonical target | Exception handler silently selects raw | Negative exception/reachability tests | Required |
| Update historical tests visibly | Current tests encode obsolete claims | Tests must describe current policy | Deleting coverage hides reachability | Preserve fixtures and invert admission assertions | Required |
| Freeze numerical kernel bodies | Reviewer finding | Whole-file hashes cannot distinguish classification edits from math edits | A route label patch silently alters raw arithmetic | AST/source-slice hashes plus representative pre/post raw outputs and branch traces | Required |

## Skeptical Plan Audit

Decision: `PASS_FOR_MECHANICAL_IMPLEMENTATION_AFTER_HANDOFF_REVIEW`.

- Baseline is the current central fail-closed policy, not the historical
  benchmark's local status strings.
- The primary criterion is reachability/classification, not FD agreement,
  runtime, memory, or a score magnitude.
- No arbitrary numerical threshold is introduced.
- The plan does not treat Phase 5's tiny certificate as permission to admit a
  route.
- Historical tests will be updated to assert rejection rather than deleted.
- Exact numerical-kernel function bodies will be hashed before edits and
  rechecked after edits. Representative raw value/score/branch outputs will be
  frozen before edits and required bitwise identical after opt-in execution.
- Phase 9 nonlinear implementation remains separate.
- A local artifact that merely contains a raw historical score cannot become a
  leaderboard input even if all its old numerical checks pass.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Are raw-barycentric routes explicitly diagnostic-only at all inventoried executable and artifact boundaries? |
| Comparator | Central Phase 0 fail-closed validators and the owner-selected Contract E route policy |
| Primary pass criterion | Without opt-in, raw execution fails before numerical work; with opt-in, emitted/normalized status is exactly historical; discovery has zero unclassified hits; all forgery, aggregate, no-fallback, and numerical-kernel preservation tests pass |
| Hard vetoes | Local full/tiny/compact admission, implicit raw default, Contract E-to-raw fallback, canonical identity forgery, or historical-math rewrite |
| Explanatory only | Source inventory counts and old numerical diagnostic values |
| Artifact | Phase 6 result, focused check log, source hashes, and updated Phase 7 subplan |
| Not concluded | Any numerical or scientific correctness beyond fail-closed route classification |

## Required Checks, Tests, And Review

1. Emit the machine-readable discovery inventory, exact queries, hit
   classifications, exclusions, whole-file hashes, and exact numerical-kernel
   function-body hashes before edits; require zero unclassified hits.
2. Inspect every public import/API, CLI default, dispatcher, local admission
   decision, artifact constructor, factory, aggregator/leaderboard consumer,
   and exception/fallback boundary.
3. Add one shared explicit historical-diagnostic opt-in convention without
   adding a caller-controlled canonical identity.
4. Replace local raw full/tiny/compact-admitted outcomes with the exact central
   historical status.
5. Bind each executable root to applicable tests for no opt-in before numerical
   work, opt-in historical emission, forged canonical metadata, inclusive
   aggregation, and injected Contract E failure that cannot call a raw sentinel.
6. Run central Phase 0/2 factory-forgery and fail-closed tests plus every
   modified historical route-contract test.
7. Recompute exact numerical-kernel body hashes and representative diagnostic
   outputs/branch traces; require bitwise identity. Run Python compilation,
   source discovery again, JSON parsing, scoped `git diff --check`, verify zero
   unclassified hits, and verify the canonical factory remains empty.
8. Write the Phase 6 result and Phase 7 documentation subplan.
9. Use the already planned one terminal independent Codex review for the
   material Phase 5 close/Phase 6 handoff; obtain another review only if Phase
   6 implementation introduces a genuinely material scientific or engineering
   ambiguity.

## Required Artifacts

- pre-edit machine-readable Phase 6 discovery, classification, whole-file hash,
  numerical-kernel body hash, and representative-output record;
- focused code/test changes;
- Phase 6 focused-check log and source-search result;
- Phase 6 result with decision and inference-status tables;
- Phase 7 LaTeX reconciliation subplan; and
- updated master, ledger, and stop handoff.

## Forbidden Claims And Actions

- Do not register or admit the Phase 5 callable.
- Do not implement a Contract E fallback by calling raw code.
- Do not preserve an obsolete admission outcome under a renamed alias.
- Do not accept caller-stamped canonical route identity.
- Do not alter raw numerical mathematics, thresholds, prepared randomness, or
  model definitions.
- Do not start nonlinear Contract E implementation or leaderboard regeneration.
- Do not claim that mechanical cleanup establishes Kalman, HMC, scientific, or
  release correctness.

## Exact Next-Phase Handoff Conditions

Phase 7 may begin only when:

- all inventoried raw CLI/API routes require explicit diagnostic opt-in;
- every raw artifact emits or normalizes to
  `historical_raw_barycentric_diagnostic_only`;
- every discovery hit is classified and every exclusion is justified;
- no local full/tiny/compact-admitted outcome or implicit default remains;
- factory-forgery, inclusive-consumer, and no-fallback tests pass;
- exact numerical-kernel body hashes and representative raw outputs/branch
  traces match their pre-edit values;
- the production canonical factory remains empty;
- historical mathematics is unchanged; and
- the result and source search preserve every unresolved Phase 5 scientific
  blocker.

The handoff carries documentation-reconciliation authority only. It carries no
Phase 8 run, nonlinear, admission, HMC, leaderboard, or release authority.

## Stop Conditions

Stop and write a blocker if a raw route cannot be separated from a canonical
consumer without choosing new scientific behavior, an unexpected concurrent
edit appears in an in-scope file, cleanup would require a public API/default
decision beyond the owner directive, five materially distinct repairs fail for
the same reachability defect, or the campaign budget expires. A focused test
failure is a repair trigger first.

## Phase-End Protocol

1. Run the full fail-closed and modified-route focused suite.
2. Preserve source hashes and source-search output.
3. Write the Phase 6 result/manifest.
4. Draft the exact Phase 7 LaTeX reconciliation subplan.
5. Review only material unresolved ambiguity under current proportional
   governance.
6. Update the master, ledger, and stop handoff.
