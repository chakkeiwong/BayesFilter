# Contract E--TP Phase 7 Same-Target All-Model Comparison Plan

metadata_date: 2026-07-15
status: READY_AFTER_SKEPTICAL_AUDIT
program_id: contract-e-tp-all-model-gradient-comparison
phase: 7
execution_target: explicit CPU-hidden TensorFlow float64 diagnostics plus artifact-only aggregation
plan_owner: Codex

## Phase Objective

Build one comparison ledger whose populated cells are mathematically comparable
because they use the same model row, observation prefix, parameter vector and
coordinate order, initial/time-order convention, and observed-data scalar.
Report values, total scores, same-scalar derivative evidence, reference gaps,
and per-time increments where available.  Missing or ineligible methods remain
visible cells rather than being filled by an alias.

This phase answers which currently implemented finite programs are
derivative-correct and how far their center estimates are from a declared
same-target reference.  It does not establish stochastic equivalence, ranking,
canonical/default status, HMC readiness, or full-horizon readiness.

## Entry Conditions

1. Contract E--TP dense/streaming primitives and recursive LGSSM tests pass.
2. LGSSM Contract E--TP has valid center artifacts at `T=2,10,50` against the
   differentiated Kalman oracle.
3. Actual SV, KSC-SV, generalized SV, and predator--prey have row-specific
   Contract E--TP prefix artifacts with own-scalar FD evidence.
4. Phase 6 has classified the scalar adjacent-state squared-TT route as
   `extension_or_invention`, subtype
   `fixed_parameter_adjacent_state_squared_tt_extension`.
5. Zhao--Cui source-route parameter-learning comparators are unavailable for
   all rows.  Contract E--Chol has no admissible all-row artifact in this
   campaign.  Austria SIR observed-data comparison is blocked by the clipped
   simulator versus unclipped Gaussian transition-density measure mismatch.

## Research Intent Ledger

| Field | Phase 7 contract |
| --- | --- |
| Main question | What value and total score does each independently valid same-target finite program return at the frozen row center and prefix? |
| Candidate | Experimental Contract E--TP with each row's prepared feature/chart policy. |
| Baseline | Exact Kalman for LGSSM; refined dense TensorFlow quadrature for scalar SV; semi-analytic or corrected-time-order SGQF reference for predator--prey. |
| Additional diagnostic | Fixed-parameter adjacent-state squared-TT extension for scalar SV only. |
| Expected failure mode | Target transform/time-order mismatch; own-scalar derivative failure; ineligible route relabelled as Zhao--Cui; feature insufficiency; finite-resolution fit error. |
| Promotion criterion | None in Phase 7 because no justified cross-method equivalence margin and no replicated preparation ensemble exist. |
| Promotion veto | Any target-identity, coordinate, scalar, or derivative mismatch; it prevents a cell from being compared. |
| Continuation veto | Corrupt/unreadable controlling evidence; shared-core own-scalar FD failure; nonfinite output; or evidence that the declared reference evaluates a different target. |
| Repair trigger | A valid TP/reference pair with a material descriptive gap, unstable adjacent refinement, or localized increment drift proceeds to Phase 8 one-factor diagnosis. |
| Explanatory diagnostics | Componentwise differences/relative errors, sign changes, increment gaps, chart condition/margin, TT fit residual, runtime. |
| Forbidden conclusion | No equivalence, superiority, Zhao--Cui parity, leaderboard completion, default readiness, HMC readiness, or nonlinear exactness. |

## Evidence Contract

The comparison builder must validate, before emitting a populated cell:

1. exact row id and horizon;
2. exact parameter vector and parameter-name order;
3. target observation policy and transition-before-first-observation flag when
   the schema exposes them;
4. expected algorithm/route identity and allowed classification;
5. finite value and finite score of the expected dimension;
6. passing own-scalar FD status for Contract E--TP and extension cells;
7. passing engineering/chart status for Contract E--TP;
8. reference refinement metadata and its honest classification;
9. SHA-256 of every source artifact.

Cross-method differences are descriptive.  `0.05*sqrt(p)` is admissible only
inside each method's own-scalar FD check and is forbidden as a TP/reference or
TP/extension agreement rule.  The ledger must set
`equivalence_classification=descriptive_only_margin_unavailable`.

The result artifact is:

`docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_same_target_comparison_20260715/comparison_ledger_v2.json`.

The unsuffixed ledger was generated before the mathematical execution review
found that the generalized-SV extension used the wrong first-step time order.
It is superseded, noncontrolling evidence and must not be used for comparison.

The phase result is:

`docs/plans/bayesfilter-contract-e-tp-phase7-same-target-all-model-comparison-result-2026-07-15.md`.

## Frozen Row Matrix

| Row and horizons | Reference | Contract E--TP | Adjacent-state extension | Contract E--Chol | Zhao--Cui source route |
| --- | --- | --- | --- | --- | --- |
| LGSSM `T=2,10,50` | differentiated Kalman | required | unavailable | unavailable in campaign | unavailable |
| actual SV `T=1,2,10` | refined dense quadrature | required | required | unavailable in campaign | unavailable |
| KSC-SV `T=1,2,10` | refined dense quadrature | required | required | unavailable in campaign | unavailable |
| generalized SV `T=1,2,10` | refined dense quadrature | required; `T=10` remains negative feature-family evidence | extension diagnostic required | unavailable in campaign | unavailable |
| predator--prey `T=2,5` | semi-analytic `T=2`; approximate corrected-time-order SGQF `T=5` | required | unavailable | unavailable in campaign | unavailable |
| Austria SIR observed-data | target measure unresolved | blocked | unavailable | unavailable in campaign | unavailable |

Actual, KSC, and generalized-SV Contract E--TP `T=2` artifacts are the only
missing mandatory numeric cells.  Prepare them with the already tested
row-specific policy: teacher order 41, continuation order 129, radius 10 for
actual/KSC SV, radius 12 for generalized SV, and the only possible
one-observation continuation at `T=2`. These are prefix diagnostics, not
cross-model defaults. Every other Phase 7 cell is built from immutable
controlling artifacts.

Execution-review amendment: historical actual/KSC `T=1` preparations bind an
older generated-data prefix and are not controlling Phase 7 evidence. Refresh
both `T=1` cells from the current frozen generator. The KSC extension must use
the same target-transform owner as Contract E--TP so that its executed
observation tensor, not merely its formula label, matches byte-for-byte.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Scalar teacher order 41 | adjacent refinement already run at `T=10`; diagnostic rung | favorable active chart only at one order | preserve order-25/41 reference history and send sensitivity to Phase 8 |
| Continuation order 129/radius 10 | scalar Phase 5 preparation protocol | truncated or under-resolved continuation | reference order 129/257 equality and Phase 8 radius/order refinement |
| Center-only charts | current evidence scope | agreement fails off-center | explicit nonclaim; parameter-region work deferred |
| Adjacent TT degree 8/order 17/rank 2 | Phase 6 warm start, not default | fit residual, rank/degree error | expose maximum fit residual and refine one factor in Phase 8 |
| Descriptive comparison | no justified margin/replicates | qualitative gap mistaken for equivalence | hard-coded no-equivalence classification |
| Artifact reuse | prevents needless reruns | stale or target-mismatched evidence | schema/identity/hash validation before aggregation |

## Exact Commands

All numeric Phase 7 diagnostics are deliberate CPU reference runs and hide GPU
before TensorFlow import:

```bash
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 2 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_actual_sv_t2_order41_lookahead1_preparation_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_contract_e_tp_scalar_sv_prefix.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_actual_sv_t2_order41_lookahead1_preparation_20260715.json --reference-orders 129,257 --reference-radius 10 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_actual_sv_t2_order41_lookahead1_result_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 --time-steps 2 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_ksc_sv_t2_order41_lookahead1_preparation_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_contract_e_tp_scalar_sv_prefix.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_ksc_sv_t2_order41_lookahead1_preparation_20260715.json --reference-orders 129,257 --reference-radius 10 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_ksc_sv_t2_order41_lookahead1_result_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_generalized_sv_synthetic_from_estimated_values --time-steps 2 --teacher-order 41 --continuation-order 129 --continuation-radius 12 --lookahead-steps 1 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_generalized_sv_t2_order41_lookahead1_preparation_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_contract_e_tp_scalar_sv_prefix.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_generalized_sv_t2_order41_lookahead1_preparation_20260715.json --reference-orders 129,257 --reference-radius 12 --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_generalized_sv_t2_order41_lookahead1_result_20260715.json
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/build_contract_e_tp_phase7_comparison.py --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_same_target_comparison_20260715/comparison_ledger_v2.json
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/highdim/test_contract_e_tp_phase7_comparison.py tests/highdim/test_ledh_contract_e_tp_scalar_sv.py tests/highdim/test_ledh_contract_e_tp_lgssm_recursive.py tests/highdim/test_ledh_contract_e_tp_predator_prey.py tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py
```

## Required Checks And Reviews

1. `python -m py_compile` for the new builder/test and touched runners.
2. Focused pytest command above.
3. `git diff --check` on Phase 7-owned files.
4. Mathematical execution review checking same scalar, time order, observation
   transform, parameter coordinates, derivative ownership, reference status,
   and route classification.
5. Result decision and inference-status tables plus post-run red team.

## Forbidden Claims And Actions

- Do not call the adjacent-state extension Zhao--Cui, source-faithful, or a
  fixed HMC adaptation.
- Do not use Kalman twice under different method labels.
- Do not use generic retained-grid routes for LGSSM or predator--prey.
- Do not fill Contract E--Chol cells with v1, raw-barycentric, TP, or caller-
  stamped artifacts.
- Do not compare an SIR component score with an observed-data score.
- Do not convert a descriptive gap into a pass/fail threshold.
- Do not select a favorable resolution after seeing Phase 7 gaps.
- Do not proceed to a long GPU run from a failed same-scalar or target-identity
  gate.

## Skeptical Plan Audit

Status: `PASS_AFTER_BASELINE_AND_INFERENCE_REPAIR`.

The pre-audit Phase 7 master text had two material flaws.  First, it assumed a
certified Zhao--Cui comparator even though Phase 6 proved that the repaired
route removed Zhao--Cui's parameter TT coordinate.  Second, it requested one of
four equivalence classifications despite having neither a justified margin nor
replicated preparation uncertainty.  The amended plan exposes unavailable
cells, labels the scalar route as an extension, and permits descriptive-only
comparison.

The audit also checked:

- **wrong baseline:** Kalman is oracle only, not a Zhao--Cui cell;
- **proxy promotion:** SGQF `T=5` and fit residuals are explicitly approximate
  or explanatory;
- **stop conditions:** target/coordinate/FD/nonfinite defects stop aggregation;
- **fairness:** target transform and time ordering are validated per cell;
- **hidden assumptions:** center-only charts and warm-start TT settings are
  explicit;
- **environment:** all new numerics are deliberate CPU float64 diagnostics;
- **answerability:** the ledger records source hashes, values, scores,
  derivative status, gaps, and unavailable cells needed to answer the phase
  question.

## Handoff And Stop Conditions

Phase 8 begins when the ledger and Phase 7 result exist, focused checks pass,
and every row is either populated, negative, unavailable, or blocked with a
specific reason.  A valid candidate gap is a repair trigger, not a continuation
veto.

Stop Phase 7 without aggregation if a controlling artifact is corrupt, target
identity cannot be reconciled, own-scalar FD fails, or the reference is shown
to compute another scalar.  Stop the campaign only if the defect invalidates
the shared core/reference; a row-specific feature or comparator failure moves
to Phase 8 or remains a row-specific negative result.
