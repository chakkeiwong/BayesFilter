# LEDH Score Tangent Materialization Phase 3 Codex Review

Date: 2026-07-09 22:44:42 HKT

Claude status: skipped for this program because the approval reviewer rejected
the bounded Claude artifact disclosure path. Review was performed by a fresh
Codex read-only subagent.

## Review Scope

- `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3-sinkhorn-reverse-lifetime-subplan-2026-07-09.md`
- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
- `tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py`

## Verdict

`VERDICT: AGREE`

## Findings

No blocking findings.

The reviewer agreed that `_filterflow_streaming_same_points_softmin_vjp`
preserves the generic same-points softmin VJP semantics while avoiding the old
separate query/key full-cotangent output stacks on equal-chunk paths. The
review checked the query/key signs, value cotangent, epsilon cotangent,
unequal-chunk fallback, total Sinkhorn call sites, source sentinel coverage,
and boundary rules.

Boundary checks passed: no exact Kalman substitution, no production
`GradientTape` or `ForwardAccumulator` in the total reverse helper, no score
admission claim, and no `N=10000` rung before the required `N=1000,T=10,
Sinkhorn=10` gate.

## Local Checks Before Review

```bash
python -m py_compile \
  experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_same_points_softmin_vjp_matches_generic_route \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result: `68 passed, 2 warnings in 29.68s`.
