# Phase 0: Registry, Topology Inventory, And Default Audit Result

Date: 2026-07-15

Status: `PASS_CLOSED_HANDOFF_READY`

Program: `contract-e-tp-all-model-clean-xla-validation`

## Verdict

Phase 0 passed. The clean-XLA program now has a dedicated eight-item registry,
not a relabeled copy of the earlier six-row gradient-comparison registry. Seven
entries are model/client rows and the structural deterministic fixture is a
separate `shared_regression_item` that cannot be counted as a completed model.

The inventory supports clean-loop engineering for actual SV, KSC-SV, and
predator--prey; preserves LGSSM as the completed reference; carries generalized
SV directly to terminal synthesis as a negative result for the tested feature
family; and keeps Austria SIR and DSGE/NAWM target-blocked. No new GPU campaign
was run in this phase.

## Artifacts And Checks

Controlling registry:

- `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-00/registry/attempt-01-20260715/phase0_registry.json`;
- SHA-256 `704522e3909f446eb1b3d32584bb37cff90159a69b6c1af2f13a8d15fc1aacee`.

Quiet log:

- `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-00/registry/attempt-01-20260715/phase0_checks.log`;
- SHA-256 `ee36c181a7fc3dfbad4adb16c95223145b9ee3b20c09680a4484478096af38b2`.

Checks actually run:

| Check | Result |
| --- | --- |
| CPU-hidden builder | pass; emitted eight unique items |
| focused Phase 0 suite | `5 passed in 0.83s` |
| independent JSON parse/classification print | pass |
| controlling LGSSM and target-registry hashes | pass |
| runtime source NumPy/SciPy import scan | no findings in the audited modules |
| compileall for builder/test | pass |
| focused `git diff --check` | pass |

The AST's raw loop counts are explanatory only. The repair-driving inventory
identifies three exact reachable groups: scalar-SV filter/backward-continuation
loops, predator--prey filter/backward-continuation loops, and predator--prey
fixed-substep RK4 loops when reached by the compiled route.

## Row Decisions

| Item | Classification | Binding next action |
| --- | --- | --- |
| LGSSM `d_x=3,p=5,T=50` | `reference_pass` | Phase 1 regression/guardrail oracle |
| actual SV `d_x=1,p=2,T=1000` | `eligible_inventory_required` | Phase 3 loop-native core after shared guards and structural regression |
| KSC-SV `d_x=1,p=2,T=1000` | `eligible_inventory_required` | Phase 3 independently gated from actual SV |
| generalized SV `d_x=1,p=3,T=1008` | `negative_result` | exclude from implementation/GPU phases; retain for Phase 10 |
| predator--prey `d_x=2,p=6,T=20` | `eligible_inventory_required` | Phase 5 loop-native filter, lookahead, and reachable RK4 work |
| Austria SIR `d_x=18,p=3,T=20` | `target_blocked` | Phase 7 target/derivative re-audit only; GPU forbidden |
| structural deterministic fixture | `required_shared_regression` | early Phase 2 support/tangent/negative controls |
| DSGE/NAWM | `target_blocked` | Phase 7 client re-audit only; no proxy client |

## Defaults And Scientific Boundaries

LGSSM lookahead 8, order 5, charts, and float64 results remain reference-only.
Actual SV, KSC-SV, and predator--prey retain unset target-specific lookahead,
quadrature, chart, parameter-region, and dtype choices until their own phases.
`0.05*sqrt(p)` remains an individual-direction same-scalar FD screen only.

The target is the finite Contract E--TP scalar actually executed. Loop counts,
prefix comparisons, graph topology, and GPU execution cannot establish
scientific filtering accuracy or cross-method equivalence. Contract E--TP is
experimental; Contract E--Chol remains canonical.

## Repairs And Budget

Before execution, bounded review led to explicit item kinds and exact artifact/
handoff paths. Source precheck then found a master handoff typo: structural/high-
dimensional Phase 8 admission must come from the SIR/DSGE Phase 7 re-audit, not
the predator--prey Phase 6 result. The master was corrected before any related
experiment.

Phase 0 used less than 0.01 CPU core-hours of the 4-hour phase cap, zero GPU
hours, and zero full-horizon attempts. Conservatively recording 0.01 CPU
core-hours leaves 95.99 CPU core-hours, 32 trusted GPU-hours, and all three
full-horizon attempts per eligible model.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| close Phase 0 | eight items have source-backed identity/topology/default classification | no registry or artifact veto | exact reusable guard surface | implement Phase 1 static/graph guards | no nonlinear clean-XLA claim |
| continue eligible rows | actual/KSC/predator have executable finite prefix cores | full-horizon factories still absent | model-specific loop state and features | later target-specific core phases | no full-horizon readiness |
| preserve generalized negative | tested progressive feature family failed | candidate veto, not shared-core veto | materially new features untested | Phase 10 synthesis unless separately replanned | not rejection of TP overall |
| preserve SIR/NAWM blockers | target/client inputs incomplete | row-local continuation veto | future owner target/client decision | Phase 7 re-audit | no proxy substitution |

## Post-Run Red Team

The strongest alternative explanation is that source names, not actual reachability,
drive the loop inventory. The registry avoids that overclaim: raw AST counts are
explanatory and the repair groups name exact current core symbols; Phase 1 adds
reusable reachability and GraphDef guards before model code changes.

The weakest part is that no nonlinear graph was built in Phase 0. That is
intentional: current Python-unrolled full horizons would answer neither the
bounded-topology nor scientific question. Prefix parity and loop-native graph
evidence belong to later phases.

## NEXT_PHASE_READINESS

| Ready-gate clause | Status | Evidence |
| --- | --- | --- |
| current result/check consistency | `PASS` | registry and log hashes above |
| legal inherited classifications | `PASS` | Row Decisions table and registry `rows` |
| explicit identities | `PASS` | registry target/factory/derivative/reference/device fields |
| audited assumptions | `PASS` | Defaults section and registry `default_audit` |
| declared criteria/vetoes/nonclaims/fresh paths | `PASS` | Phase 1 subplan evidence contract |
| executable commands answer Phase 1 | `PASS` | Phase 1 focused AST/GraphDef/LGSSM tests |
| no unresolved material finding/human boundary | `PASS` | Phase 0 review `AGREE`; no external boundary |
| componentwise budget | `PASS` | available `95.99 CPU,32 GPU,3 attempts`; Phase 1 minimum+reserve `4 CPU,0 GPU,0 attempts` |

Overall: `READY`.

Handoff:
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase1-shared-clean-loop-guardrails-subplan-2026-07-15.md`.

