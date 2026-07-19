# Focused Native Review: Repair-02 Exact Continuation-01

Date: 2026-07-14 (Asia/Shanghai)

Review type: `FOCUSED_NATIVE_READ_ONLY_REVIEW`

Status: `AGREE`

## Scope

Reviewed:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-repair-02-continuation-01-plan-2026-07-14.md`;
- `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py`;
- `tests/test_ssl_lstm_a4_hmc_repair_02_continuation_01.py`;
- repair-02 adaptation and segment-0 public/private receipts and hashes; and
- the reviewed A4 acquisition runtime and admission diagnostics.

## Verification

| Check | Result |
| --- | --- |
| Continuation focused tests | `6 passed` in `5.75s` |
| Compile | Runner and tests compile with GPU hidden |
| Handoff replay | Exact old shard `[250,4,4]`, final state `[4,4]`, finite values, and prior budget `2040.799946242012s` verified |
| Kernel identity | Step `0.37613058552609946`; `L=4`; trajectory `1.5045223421043978` |
| State identity | New call starts from repair-02 segment-0 exact final state; no adaptation or burn-in is repeated |
| Hash identity | Adaptation, segment receipt, private manifest, retained shard, and final-state hashes are frozen and checked |
| Budget | Conservative `1800s` projection fits within `26759.200053757988s` remaining |
| Namespace | Fresh `repair-02-continuation-01`; no overlap with repair-02 |

## Interpretation

The continuation directly tests the unresolved explanation from repair-02:
insufficient retained length versus a deeper geometry problem. It appends 250
draws from the exact final state, preserves all earlier draws, and recomputes
the unchanged A4 gate over 500 draws per chain. It cannot silently convert a
promotion veto into a hard veto, and it cannot use trend metrics as a substitute
for R-hat, ESS, or MCSE admission.

The command must run exactly once. Whether cumulative diagnostics pass or fail,
the continuation stops afterward. A pass would admit the cumulative archive as
an A4 calibration input; it would not establish posterior correctness or
convergence proof. A fail would justify a mass-geometry plan rather than more
unplanned extensions.

## Verdict

`VERDICT: AGREE`

One trusted GPU/XLA continuation block may run. Forecast calibration remains
conditional on cumulative admission and a refreshed calibration execution plan.
