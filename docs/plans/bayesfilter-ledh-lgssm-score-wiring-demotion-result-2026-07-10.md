# LGSSM LEDH Score Wiring Demotion Result

Date: 2026-07-10

## Question

Was the N=10000 LGSSM score failure a wiring problem, and should the wrong
path be demoted?

## Finding

Yes.  The failure was a wiring/category problem, not a new mathematical
derivation failure.

The public `compact-sensitivity` score path in
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` was emitting
compact/memory-efficient route metadata while dispatching through the
full-history reverse score implementation.  That implementation stores
time-indexed filtering auxiliaries and is exactly the route that can consume
large GPU memory at N=10000.

## Repair

- `compact-sensitivity` now dispatches to
  `_compact_value_and_score_from_components`.
- `manual-reverse` now dispatches to `_manual_value_and_score_from_components`
  only as a historical/full-history diagnostic.
- The LGSSM route metadata now emits:
  - `score_route = compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
  - `score_execution_style = compact_forward_sensitivity_no_time_history`;
  - `uses_full_history_reverse_route = false` for compact runs.
- The shared score contract now treats compact no-tape provenance as the
  admissible family and memory-style/manual-total-VJP provenance as historical
  diagnostic for full admission.
- The fixed-SIR memory-result normalizer was aligned with the same contract:
  it emits compact provenance and records the old memory-style/manual route as
  historical metadata.

## Evidence

Static anchors:

- LGSSM route constants:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py:96`.
- LGSSM score dispatcher:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py:2002`.
- LGSSM admission decision:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py:2246`.
- Shared score contract:
  `bayesfilter/highdim/ledh_score_contract.py:70` and
  `bayesfilter/highdim/ledh_score_contract.py:260`.
- Fixed-SIR provenance alignment:
  `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py:1117`.

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py bayesfilter/highdim/ledh_score_contract.py`
  passed.
- `pytest -q tests/test_ledh_lgssm_manual_score_phase4.py tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_score_artifact_emitter_phase1.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  passed: 73 passed, 2 warnings.
- `pytest -q tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/highdim/test_ledh_predator_prey_score_phase4_contract.py tests/highdim/test_ledh_actual_sv_score_phase5_contract.py tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`
  passed: 55 passed, 2 warnings.
- Tiny CPU route smoke passed with `dtype=float32`, TF32 enabled, and emitted
  compact score metadata.  This was route evidence only, not GPU performance or
  N=10000 admission evidence.

Read-only review:

- Claude bounded read-only review returned `VERDICT: AGREE`.
- Review note: `RAW_MEMORY_STYLE_ADMITTED_STATUS` remains a legacy raw status
  name in the LGSSM runner, but Claude found it to be naming residue rather
  than an admission loophole because normalized artifacts emit compact
  provenance and pass through the tightened shared contract.

## Nonclaims

- This does not prove N=10000,T=50 GPU score admission.
- This does not prove exact Kalman score correctness.
- This does not prove HMC/NUTS readiness or posterior correctness.
- The CPU smoke is not GPU memory evidence.
