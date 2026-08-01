# Phase 3: Scalar-SV Loop-Native Core Subplan

Date: 2026-07-15

Status: `REVIEWED_ACTIVE_EXECUTION`

Review record: bounded review returned `REVISE` on inherited-default provenance,
topology-threshold interpretation, and exact CPU-hidden commands/artifacts. The
same plan was patched with source anchors, falsification diagnostics, explicit
engineering-heuristic labels, and commands. Focused re-review returned
`VERDICT: AGREE`.

Execution amendment after skeptical close audit: the master program also
requires a `T=100` rung and a parameter-region decision before `T=1000`.
Phase 3 therefore adds fresh center-scoped `T=100` certification. It does not
invent a parameter box: the frozen registry says no reviewed nonlinear region
exists. Actual SV uses the previously reviewed order-25/lookahead-16 repair
hypothesis; KSC-SV uses its order-41/lookahead-8 refinement hypothesis. Center
and same-scalar FD-endpoint validity may advance to a center-only Phase 4;
full-box and HMC readiness remain explicitly deferred.

## Objective And Separate Targets

Replace Python-unrolled filter-time and backward-continuation recursions in the
actual non-Gaussian SV and KSC-SV finite Contract E--TP programs with functional
TensorFlow loops. Preserve each row's distinct observation target, transform,
time order, fixed charts, value, and total score.

Actual SV uses the exact transformed log-square/log-chi-square observation law,
`transition_before_first_observation=false`, and parameter order
`[gamma_unconstrained,log_beta]`.

KSC-SV uses the offset log-square KSC Gaussian-mixture observation law with
offset `1e-8` and the same time convention/coordinates. KSC is a BayesFilter
mixture target, not the actual-SV law and not an exact Zhao--Cui source model.

Generalized SV is excluded: its tested feature family is a terminal negative
result and cannot enter this repair.

## Entry Conditions

- Phase 2 structural result is `PASS_CLOSED_HANDOFF_READY`.
- Phase 1 final guardrail SHA-256 is
  `344d8480621affbb89c048bdaf61d4e9660ebc2a4ac002c82e41594b38cd370a`.
- Actual/KSC short-prefix finite programs pass same-scalar FD and fixed-chart
  checks; `T=10` comparisons are descriptive only.
- Current scalar source has Python outer-time and backward-continuation loops
  and no XLA factory.
- Remaining budget is 95.87 CPU core-hours and 31.99 GPU-hours; Phase 3 is CPU
  only and consumes no full-horizon attempts.

## Frozen Prefix Baselines And Default Audit

| Choice | Actual SV | KSC-SV | Status/failure diagnostic |
| --- | --- | --- | --- |
| `T=1` | order 25 primitive | order 25 primitive | inherited Phase 5 primitive artifacts; edge/no-reset parity |
| `T=2,3,10` | order 41, continuation order 129, radius 10 | same numeric baseline | Phase 7 prefix plan and Phase 8 refinement ledger; convenience/refinement baseline, not default |
| `T=100` | order 25, continuation order 129, radius 10 | order 41, continuation order 129, radius 10 | Phase 5 target-specific refinement results; actual order-25/lookahead-16 repaired score drift, KSC order-41/lookahead-8 is adjacent refinement; hypotheses only |
| lookahead | 1 at `T=2,3`; 8 at `T=10`; 16 at `T=100` | 1 at `T=2,3`; 8 at `T=10,100` | Phase 5/7 viable-prefix artifacts; target-specific hypotheses only; falsified by loop/unrolled mismatch or graph growth |
| feature family | mass/state/square/target continuation | same form, distinct target | Phase 5 scalar core; row-owned target evaluation |
| chart | fixed prepared square chart | independently prepared fixed chart | preparation artifact/source hash; falsified by nonpositive/rank-invalid chart or parity mismatch; no runtime selection |
| dtype | float64 | float64 | repository numerical reference and existing artifacts; no TF32 claim |
| FD | central step `1e-5`; `0.05*sqrt(2)` individual directions | same | inherited scalar-prefix float64 diagnostic; same-scalar only; step sensitivity is explanatory and a disagreement triggers repair rather than threshold relaxation |

The provenance anchors are immutable evidence, not promotions: Phase 5 scalar
preparation result
`docs/plans/bayesfilter-contract-e-tp-phase5-scalar-nonlinear-preparation-result-2026-07-15.md`,
Phase 7 comparison plan
`docs/plans/bayesfilter-contract-e-tp-phase7-same-target-all-model-comparison-plan-2026-07-15.md`,
and Phase 8 refinement result
`docs/plans/bayesfilter-contract-e-tp-phase8-one-factor-refinement-result-2026-07-15.md`.
Each row must record its exact preparation JSON hash before execution. A
provenance choice is a baseline/hypothesis, never a cross-model default.

Shared numeric choices are tested in both rows but are not scientifically
interchangeable. One row's failure does not fail the other unless a shared-core
defect is proved.

## Required Design

1. Implement a batched backward-continuation `tf.while_loop` over fixed maximum
   lookahead with a terminal mask. Preserve target time indices, grid weights,
   and log-sum-exp order. Compare values/Jacobians for both specs at lookaheads
   `1,2,8`.
2. Static edge dispatch: `T=1` initial-law terminal only; `T=2` initial
   projection plus terminal; `T>=3` adds one fixed-shape intermediate
   `tf.while_loop`.
3. Carry four parents/weights, objective, validity, and fixed-shape histories.
   Use static broadcast/reshape instead of reverse-unsafe repeat/tile.
4. Try reverse autodiff of the scalar through functional loops. If diagnostic
   history derivatives are compiler-incompatible, keep scalar reverse autodiff
   and use explicit diagnostics; never stop gradients or change the scalar.
5. Add XLA-default factories for actual and KSC. Phase 3 builds CPU-hidden
   concrete graphs but does not execute serious GPU/full-horizon ladders.
6. Invalid charts/support return false and poison scalar, score, particles, and
   weights through the exact factory/configuration being classified.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | do loop-native actual/KSC prefix programs equal their own unrolled finite scalars and total scores with bounded graph bodies? |
| Baseline | row-specific unrolled finite program; dense/SGQF remains a separate scientific diagnostic |
| Primary criterion | per-row continuation value/Jacobian and full value/increment/score/state/validity parity at `T=1,2,3,10,100`; source/graph/fail-closed pass |
| Vetoes | target transform/time-order conflation, partial derivative, Python dynamic loop, chart drift, finite invalid state, or wrong-row preparation |
| Explanatory | dense-reference gaps, trace time, graph nodes/bytes, chart conditioning |
| Not concluded | `T=1000`, nonzero-radius parameter-region validity, GPU execution, equivalence, accuracy, canonical/default/HMC/leaderboard readiness |
| Artifacts | separate actual/KSC preparations/results plus shared graph/source audit, logs, result, hashes |

The topology ratios are engineering heuristics inherited from the reviewed
LGSSM repair: they detect accidental horizon-proportional graph duplication and
are not mathematical bounds, numerical tolerances, or scalar-correctness
evidence. A ratio failure triggers graph inspection/repair; it cannot be relaxed
after seeing results.

## Required Artifacts And Checks

Fresh root:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/`.

- copy or regenerate exact `T=1,2,3,10,100` preparations in fresh attempts without
  changing selection rules;
- continuation value/Jacobian tests for actual and KSC;
- loop/unrolled full-program parity and same-scalar FD at each rung;
- Phase 1 reachability guard on exact loop factories and expected rejection of
  historical roots;
- graph inventory at `T=3,100`: both exercise the same intermediate filter-loop
  body and contain functional loops; top/function node ratio `<=1.10`,
  GraphDef-byte ratio `<=1.25`. `T=2` remains an edge-semantics test and is not
  the scaling denominator;
- local compiled same-factory invalid controls;
- existing scalar-SV/adapters and shared LGSSM/structural regressions;
- compileall, JSON/hash parse, and `git diff --check`.

All CPU commands set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. No GPU
command belongs to Phase 3.

## Exact CPU-Hidden Commands

```bash
mkdir -p docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-01-cpu-20260715
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 1 --teacher-order 25 --continuation-order 129 --continuation-radius 10 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-01-cpu-20260715/actual_t1_preparation.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 2 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-01-cpu-20260715/actual_t2_preparation.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 3 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-01-cpu-20260715/actual_t3_preparation.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 10 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 8 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-01-cpu-20260715/actual_t10_preparation.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 100 --teacher-order 25 --continuation-order 129 --continuation-radius 10 --lookahead-steps 16 --chart-mode fixed_square --output <fresh-attempt>/actual_t100_preparation.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 --time-steps 100 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 8 --chart-mode fixed_square --output <fresh-attempt>/ksc_t100_preparation.json
```

Repeat the four commands with row id
`zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` and KSC's exact target
transform. Then run the dedicated Phase 3 harness (to be added at
`docs/benchmarks/run_contract_e_tp_clean_xla_phase3_scalar_sv.py`) with
`--device cpu --row-id ... --time-steps 1,2,3,10`, redirecting output to
`phase3_checks.log`. The harness must refuse existing outputs, record the exact
command/environment/seed/preparation hashes, and emit per-row JSON plus a close
record. The commands above are reference syntax and execution is allowed only
after the harness and row-specific preparation adapters pass their focused unit
tests.

## Forbidden Actions And Claims

- Do not import generalized SV into a shared factory branch.
- Do not conflate target/flow observations or actual/KSC likelihoods.
- Do not transfer an actual chart to KSC or select charts at runtime.
- Do not use NumPy/SciPy in the gradient-bearing runtime.
- Do not relax graph gates after seeing results.
- Do not call prefix parity full-horizon, GPU, scientific, canonical, HMC, or
  leaderboard evidence.
- Do not infer a nonzero-radius parameter region from the center, FD endpoints,
  or one deliberately invalid off-center point.

## Budget And Repair Loop

Phase cap: 16 CPU core-hours, zero GPU, zero full-horizon attempts.
`minimum_entry_budget`: 4 CPU core-hours; `repair_reserve`: 4 CPU core-hours.
Available 95.87 CPU exceeds 8; entry budget gate `PASS`.

Repair localized shapes, Cartesian products, histories, autodiff, XLA CPU,
preparation, or harness defects within the fixed target and budget. Preserve
failed attempts. Stop a row on unreconciled loop/unrolled mismatch, wrong target
identity, invalid chart without a predeclared repair, graph growth, or phase cap
exhaustion. Continue the other row unless a shared-core defect is proved.

## Exact Phase 4 Handoff

Before close, create
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase4-scalar-sv-gpu-xla-subplan-2026-07-15.md`
using only Phase 3-passing rows. Freeze row-specific full-horizon preparations,
factories, short/full graph pairs, derivative and same-factory fail-closed
controls, trusted GPU commands, attempt/budget limits, and scientific nonclaims.

Phase 3 closes only with `NEXT_PHASE_READINESS` and bounded Phase 4 review. On
`READY`, continue automatically.

## Skeptical Pre-Execution Audit

Status: `PASS_DRAFT_FOR_REVIEW`.

- Same finite row scalar, not dense/SGQF agreement, is the topology baseline.
- Actual/KSC laws remain separate despite common state dimension.
- Lookahead/order/chart choices remain prefix hypotheses, not defaults.
- `T=1,2,3,10,100` cover edge, prefix, and intermediate-loop cases; `T=3,100`
  form the amended graph-scaling pair.
- Source and GraphDef evidence, not a wrapper token, determine status.
- Prefix success cannot be promoted to `T=1000` or GPU readiness.
