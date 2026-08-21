# Reset Memo: Squared-TT Program — Session Close (2026-08-19)

Status: `CHECKPOINT` (supersedes as entry point:
`bayesfilter-squared-tt-program-reset-memo-2026-08-17.md`; that memo's
Section 6 queue is fully executed).

## What is now true (all verified this session, main @ dae37183, uncommitted)

- P2 CLOSED: scaled-solve repair verified (I-P2-4 adjoint vs forward-JVP
  1e-12 at n in {1,2}; two test-side defects fixed: Cholesky-JVP
  transpose, qo=8-vs-deg-10 fixture). T=120 resource gate failed at
  19.5 GiB then repaired (trace drops design matrices, reverse sweep
  rebuilds bit-identically) -> 1.69 GiB, value+grad bit-identical.
  Execution-log Addenda 5-7; master plan at REVISION 4 (v0.2/v0.3
  semantics, rank-conditioning + quadrature-resolution tuning gate).
- P1B: r*(2) = 6 (attempt03, Sobol 32768 rows, all seeds; +8% margin
  seed flagged scope-marginal). attempt01/02 classified invalid as rank
  evidence (row-sampling bias; memo's shift-jump explanation REFUTED).
  `EngineConfig.row_design="sobol"` added (both engines, V5).
- P3.1-P3.3 CLOSED: XLA value engine
  `bayesfilter/highdim/squared_tt_engine_xla_tf.py`, parity 2.8e-14
  (gate 1e-12), regression test tests/highdim/test_p3_xla_value_parity.py
  GREEN. Backend: CholeskyQR2 + eigvalsh condition (estimator ceiling
  ~1e8, non-finite backstop). Three measured XLA-CPU defects fixed —
  tall-SVD OOM, jitted-QR wide-fit INACCURACY (probe under the claim
  path's compilation mode, not eager!), 6.6s/update SVD. Speedups:
  n=4 cell 522 s warm vs 1386 s eager.
- n=4 mechanism arms (XLA, single seed): rank8 FLAT (+12%), deg16 FLAT
  (-12%), hw4.0 WORSE (2.1x) with better in-sample rms. Declared rule
  fired: STOP n=4 compute; structural-suspects review next (branch/
  boundary-rank growth; 9-axis assembly conditioning; normalizer
  error-mass — build the holdout-mass instrument first).

## Environment notes

- Bash permission classifier outage recurred throughout; project
  allowlist now covers tf-gpu pytest + docs/benchmarks (owner-approved,
  .claude/settings.json), but background/pipe launches still classify —
  prefer foreground allowlisted commands writing their own logs.
- LLVM "Resource tracker defunct" crash after ~3 large XLA compiles in
  one process: run big compile batteries in fresh processes.

## Continuation point (in order)

1. Structural-suspects review for the n=4 plateau (analysis-first;
   holdout-mass instrument; per-step telemetry read). Note Section 5
   has the suspects list.
2. P3.4: adjoint/score path under XLA (same probe-then-port method;
   scoping note has the inventory).
3. Owner-approved checkpoint commit(s) of: engine XLA module + tests,
   adjoint trace repair, row_design option, plan revision 4, all result
   notes. RECOMMENDED before further work.
4. Then per plan: P4 adapters, P5 tuning, P2S (UB-3 focused review
   boundary), P6.

Key artifacts index: notes 2026-08-18/19 under docs/plans (ladder
attempt02/03 results, n4 row-design note w/ Section 5, P3 scoping note),
benchmarks run_p31/p32/p33_*, run_p1b_n4_* + artifacts dirs.
