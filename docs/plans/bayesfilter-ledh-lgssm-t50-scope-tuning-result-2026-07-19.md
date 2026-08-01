# LEDH LGSSM T=50 Scope Tuning Result

Date: 2026-07-19  
Status: `CLOSED_SCOPE_CLAIM_PASS`

## Question And Verdict

Question: for the exact LGSSM M3 scope `T=50,N=K=1024,float32/TF32,GPU/XLA`,
can an offline cheaper-first search select fixed streaming-OT controls that pass
the declared marginal and finite-program gates on calibration, validation, and
then untouched claim seeds?

Verdict: `PASS` for this exact scope. The selected controls are
`sinkhorn_steps=20,balance_steps=8`, bound to scope SHA-256
`451d361ac856bfeea9df93a84f07999967db8199dbb9ce948fa214c074ea5f2d`.
They are not settings for another horizon, data regime, model, or LEDH route.

## Evidence

Calibration used seeds `81800..81807`, validation used `81808..81815`, and the
untouched claim used `81820..81835`. Historical failed-baseline seeds
`81720..81735` were not reused.

| Candidate | Calibration | Validation | Worst validation `TV_col` | Worst validation `E_row` | Decision |
| --- | --- | --- | ---: | ---: | --- |
| `(20,3)` | PASS | FAIL | `5.03e-5` | `0.04571` | Reject; continue cheaper balance ladder |
| `(20,5)` | PASS | FAIL | `2.54e-5` | `0.02193` | Reject; continue cheaper balance ladder |
| `(20,8)` | PASS | PASS | `8.59e-6` | `0.007126` | Freeze for untouched claim |

The untouched claim passed with worst `TV_col=3.44e-6` and
`E_row=0.003273`. Values and scores were finite; chart/reset checks, exact work
counts, and bitwise replay passed. The graph contains `StatelessWhile`, reports
no Python horizon unrolling, uses TF32/XLA on GPU, and stayed below the 8192 MiB
logical limit with about 0.95 GB peak allocator use. Claim node wall time was
65.42 seconds; total attempt wall time was 193.95 seconds.

Artifacts are under
`docs/benchmarks/artifacts/ledh_per_scope_tuning_20260719/lgssm_t50_attempt02/`.
The principal result SHA-256 is
`2f973e3cdaab51996d9a28ad856879340dc4a25def47f397025da8d03fd35b94`.

## Attempt And Traceability Record

Attempt 01 failed before candidate execution: the tuning-scope import
initialized TensorFlow before the logical GPU limit could be installed. The
repair delayed that import until after GPU configuration. Attempt 02 was a
localized harness retry in a fresh directory with unchanged scientific scope,
partitions, search, gates, and budget.

The immutable attempt-02 manifest and nested runner payload name the superseded
2026-07-18 plan because the harness then hard-coded that path. This is a
traceability defect, not evidence that cross-horizon tuning was used: the
artifact's repository-issued scope identity, T=50 calibration/validation
partitions, selected-control record, and untouched T=50 claim are explicit and
internally consistent. The harness default was subsequently corrected to name
the active per-model-scope program and to propagate caller-supplied campaign and
plan identities. Existing evidence was not rewritten.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit `(20,8)` only for this T=50 scope | PASS on disjoint tuning and untouched claim | No marginal, finite, chart/reset, replay, work, XLA, device, memory, or time veto | Only 16 claim seeds; no population failure-rate estimate | Close LGSSM T=50 tuning and implement the latent-SIR scope tuner | No universal control setting, Kalman score agreement, HMC readiness, posterior correctness, nonlinear validity, or leaderboard completeness |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Passed for this exact finite T=50 execution scope |
| Statistically supported ranking | None; candidates were screened against hard gates, not ranked scientifically |
| Descriptive-only differences | Candidate residual sizes and wall times |
| Default readiness | `(20,8)` is scope-bound, not a repository or cross-model default |
| Next evidence needed | A separate implemented tuner, tuning partitions, and untouched claim for every nonlinear model/scope |

## Post-Run Red Team

The strongest alternative explanation is finite-seed luck in the tail of the
row-marginal error distribution. More untouched replications could overturn the
scope admission. The result would also be invalidated by a source change that
alters a scope-bound finite program without producing a new scope/artifact.
The weakest evidence is the 16-seed claim size and the stale manifest plan path.
Neither supports transferring `(20,8)` or claiming score correctness against the
Kalman oracle.
