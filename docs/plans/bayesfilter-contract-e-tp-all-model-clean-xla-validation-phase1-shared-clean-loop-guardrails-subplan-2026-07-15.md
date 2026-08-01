# Phase 1: Shared Clean-Loop Primitives And Static Guardrails Subplan

Date: 2026-07-15

Status: `DRAFT_READY_FOR_REVIEW`

Program: `contract-e-tp-all-model-clean-xla-validation`

## Objective And Boundary

Create reusable reporting/test guardrails that inspect exact compiled-route
roots for forbidden Python time, lookahead, RK4, or solver recursion and inspect
concrete TensorFlow graphs for functional loop bodies. Preserve the completed
LGSSM finite scalar, total score, fail-closed behavior, and graph gates.

Phase 0 does not justify a shared cross-model algorithmic loop driver: LGSSM,
scalar SV, predator--prey, and structural support carry different state shapes,
features, time ordering, and solver needs. Phase 1 therefore shares audit
primitives only. It does not refactor nonlinear model mathematics.

## Inherited Entry Conditions

- Phase 0 is `PASS_CLOSED_HANDOFF_READY` with registry SHA-256
  `704522e3909f446eb1b3d32584bb37cff90159a69b6c1af2f13a8d15fc1aacee`.
- The inherited close record is
  `docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase0-registry-topology-default-audit-result-2026-07-15.md`;
  the registry is
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-00/registry/attempt-01-20260715/phase0_registry.json`.
- LGSSM is the completed clean-XLA reference.
- Actual SV and KSC-SV remain eligible for Phase 3; predator--prey remains
  eligible for Phase 5.
- Generalized SV is excluded to Phase 10; SIR/DSGE remain Phase 7 target
  blockers; the structural fixture is reserved for Phase 2.
- Remaining budget is 95.99 CPU core-hours, 32 GPU-hours, and three full-horizon
  attempts per eligible model.

## Required Implementation And Artifacts

Add one reporting-only module under `bayesfilter/testing/` or
`docs/benchmarks/` that:

1. resolves repository-owned route roots to exact source symbols;
2. classifies reachable Python loops by role rather than substring alone;
3. rejects Python loops declared as dynamic time/window/RK4/solver work;
4. counts top-level and function-library `While`/`StatelessWhile`/`Scan`
   operations, nodes, functions, and GraphDef bytes from a concrete function;
5. reports rather than promotes fixed small parameter/dimension loops;
6. checks audited gradient-runtime modules for NumPy/SciPy and host callbacks;
7. supports per-route allow/deny specifications without embedding LGSSM
   lookahead, dimensions, charts, dtype, or tolerances as global defaults.

Write a Phase 1 JSON result and quiet log under
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-01-20260715/`.

The exact structured artifact is `guardrail_audit.json`; the exact log is
`phase1_checks.log`. The exact close record is
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase1-shared-clean-loop-guardrails-result-2026-07-15.md`,
and the execution ledger is the existing program ledger. JSON, log, close
record, and ledger readiness table are all required; none substitutes for the
others.

The JSON must exercise:

- the LGSSM clean root as a passing control with at least two functional loops;
- the historical LGSSM unrolled root as a rejecting control;
- current scalar-SV and predator--prey roots as expected-rejecting inventories,
  not failures of Phase 1;
- a synthetic fixed-small-loop fixture that is reported but not confused with
  horizon recursion;
- a synthetic hidden-helper route proving reachability cannot be evaded by a
  clean wrapper token.
- a synthetic forbidden recursion with neutral function/variable names, an
  aliased `range`, and a bound derived through `len(sequence)`, proving the
  classifier uses reachable AST structure plus an explicit dynamic-bound role
  specification rather than tokens such as `time`, `window`, `RK4`, or `solver`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | can one reusable audit distinguish clean functional routes from hidden or direct Python dynamic recursion without changing model programs? |
| Baseline | exact LGSSM loop/unrolled roots and Phase 0 source inventory |
| Primary criterion | all positive and negative source/graph fixtures classify correctly; LGSSM regressions remain unchanged |
| Vetoes | substring-only acceptance, wrapper-token acceptance, missed reachable helper, changed LGSSM scalar/score, lost fail-closed behavior, or runtime NumPy/SciPy |
| Explanatory | raw loop counts, GraphDef nodes/bytes, trace time |
| Not concluded | nonlinear core repair, nonlinear XLA, scientific accuracy, canonical/default/HMC/leaderboard readiness |
| Artifact | Phase 1 JSON/log/result with source and artifact hashes |

## Defaults And Assumptions

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| source roots are repository-owned callables | Phase 0 requirement | caller hides helper or stamps identity | hidden-helper fixture |
| AST plus explicit route specifications | engineering hypothesis | dynamic aliases evade static resolution | conservative reject on unresolved calls |
| GraphDef inventory is separate from source audit | reviewed master | wrapper has loop token but hidden body unrolls | positive/negative graph fixtures |
| LGSSM graph thresholds `1.10/1.25` | reviewed reference topology gates | become scientific tolerances | label topology-only in schema/tests |
| no shared model loop driver yet | Phase 0 evidence | premature abstraction changes scalar | Phase 1 touches reporting/tests only |

## Checks

1. Unit tests for direct loops, hidden helpers, fixed-small loops, unresolved
   helper fail-closed behavior, functional loop graph counting, and policy roles.
2. Existing LGSSM recursive/progressive suites.
3. Existing compiled invalid-chart artifact hash replay; no GPU rerun is needed
   because Phase 1 does not change LGSSM algorithm code.
4. CPU-hidden concrete-graph inventory at `T=10,50`, verifying the controlling
   `4014/3712/4` topology and master ratios against fresh source.
5. `compileall`, JSON parse/hash checks, and focused `git diff --check`.
6. Write Phase 1 result, draft Phase 2 exact structural subplan, review it, and
   continue automatically if `NEXT_PHASE_READINESS` passes.

## Exact CPU-Hidden Commands

Every command sets `CUDA_VISIBLE_DEVICES=-1` before Python starts, hence before
any TensorFlow import:

```bash
mkdir -p docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-01-20260715
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/build_contract_e_tp_clean_xla_phase1_guardrails.py --phase0-registry docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-00/registry/attempt-01-20260715/phase0_registry.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-01-20260715/guardrail_audit.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-01-20260715/phase1_checks.log 2>&1
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/highdim/test_contract_e_tp_clean_xla_guardrails.py tests/highdim/test_ledh_contract_e_tp_lgssm_recursive.py tests/highdim/test_ledh_contract_e_tp_lgssm_progressive.py >> docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-01-20260715/phase1_checks.log 2>&1
CUDA_VISIBLE_DEVICES=-1 python -m compileall -q bayesfilter/testing/contract_e_tp_clean_xla_guardrails.py docs/benchmarks/build_contract_e_tp_clean_xla_phase1_guardrails.py tests/highdim/test_contract_e_tp_clean_xla_guardrails.py
git diff --check -- bayesfilter/testing/contract_e_tp_clean_xla_guardrails.py docs/benchmarks/build_contract_e_tp_clean_xla_phase1_guardrails.py tests/highdim/test_contract_e_tp_clean_xla_guardrails.py
```

The builder fails if `CUDA_VISIBLE_DEVICES` is not exactly `-1`; this phase
cannot silently initialize a GPU. Full output is preserved in the declared log.

## Forbidden Actions And Claims

- Do not modify scalar-SV, predator--prey, structural, or LGSSM algorithm code.
- Do not define a generic shared loop body before model-specific state contracts
  prove compatibility.
- Do not accept a route merely because its wrapper contains `tf.while_loop`.
- Do not reject fixed small parameter loops without showing dynamic scaling;
  report their role.
- Do not call source or graph audit proof of numerical/scientific correctness.
- Do not run a new GPU campaign in Phase 1.

## Budget And Readiness Calculation

Phase cap: 6 CPU core-hours, zero GPU-hours, zero full-horizon attempts.
`minimum_entry_budget`: 2 CPU core-hours. `repair_reserve`: 2 CPU core-hours.
Available 95.99 CPU core-hours is at least 4; GPU and full-horizon inequalities
are zero. Entry budget gate: `PASS`.

## Repair And Stop Conditions

Repair localized AST resolution, graph counting, schema, or test defects within
the phase and rerun focused checks. Stop only if a reusable audit cannot
distinguish the known LGSSM clean/unrolled routes without model-specific false
claims, LGSSM controlling topology no longer reproduces and the cause cannot be
localized, evidence is corrupted, or the phase CPU cap is exhausted.

Expected rejection of scalar-SV or predator--prey current roots confirms the
inventory and is not a program stop.

## Exact Phase 2 Handoff

Before Phase 1 closes, create
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase2-structural-support-regression-subplan-2026-07-15.md`.
It must inherit the exact structural fixture, declare its smallest trusted
compiled configuration, support/completion/tangent/off-support/fail-closed
checks, fresh artifact paths, budget, and a continuation veto that prevents new
nonlinear GPU campaigns if shared singular-support semantics fail.

The Phase 1 result and ledger must contain the runbook's evidence-linked
`NEXT_PHASE_READINESS` table. On `READY`, enter Phase 2 `PRECHECK` immediately.

## Skeptical Pre-Execution Audit

Status: `PASS_DRAFT_FOR_REVIEW`.

- The baseline is exact clean/unrolled LGSSM routes, not a wrapper token.
- Raw loop count is explanatory; reachable role classification is primary.
- Shared audit code cannot silently become shared model mathematics.
- Expected nonlinear-root rejection drives later repair and is not candidate
  rejection.
- Phase 1 uses CPU-hidden traces only and cannot support a GPU claim.
- The output directly answers whether guardrails can detect hidden unrolling.
