# LGSSM N=2000/N=5000 Kalman Bias Ladder Result

Date: 2026-07-20
Status: `N5000_QSCALE_BIAS_SMALLER_DESCRIPTIVELY_SCREEN_FAIL`
Plan:
`docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-plan-2026-07-20.md`
Aggregate:
`docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/aggregate_final.json`

## Verdict

Increasing to `N=5000` made the observed `q_scale` score bias materially
smaller, but it did not certify the LGSSM route against Kalman.

- At `N=1024`, mean relative `q_scale` error was `-31.65%`.
- At `N=2000`, it worsened to `-45.75%`.
- At `N=5000`, it narrowed to `-9.91%`, a `68.7%` reduction in absolute mean
  bias from `N=1024`. Relative across-seed SD also fell from `15.73%` to
  `10.28%`.

This is descriptive support for a finite-particle contribution, not a
monotone particle-convergence result or a statistically supported ranking.
`N=2000` moved in the wrong direction, the three rungs used independent seeds,
and no paired uncertainty analysis for the cross-`N` difference was declared.

The frozen `N=5000` screen is `screen_fail`:

- the value-bias interval `[0.1116%,0.1848%]` is wholly above the allowed
  `[-0.1%,+0.1%]` region;
- the `q_scale` interval `[-17.72%,-2.11%]` overlaps but is not contained in
  `[-5%,+5%]`, so that coordinate is inconclusive relative to the tolerance;
- `phi3` is also inconclusive; and
- the remaining score-coordinate intervals are contained.

Therefore `T=10,N=5000` was not launched, nonlinear transfer remains blocked,
and no HMC-facing/default-readiness status is established.

## Claimed And Computed Quantities

| Item | Exact classification |
| --- | --- |
| Claimed target | Exact differentiated Kalman log likelihood and HMC-coordinate score on the float32-rounded production observation prefix at `T=50`. |
| Candidate quantity | Canonical Contract E--Chol finite-particle log likelihood and its total derivative `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`, with independently tuned fixed OT controls for each exact particle/chunk scope. |
| Equality verdict | Different at both `N=2000` and `N=5000`; the frozen simultaneous screen fails. |
| Supporting artifact | `aggregate_final.json`, SHA-256 `fab768b961214fb5d962fd05a7d868802a65ae55a7cd0d73451de7f662dc495e`. |
| Not proved | A convergence rate, asymptotic correctness, nonlinear validity, HMC/posterior readiness, or superiority over another method. |

## Particle Ladder

| N | Independently selected controls | Chunk grid | Engineering claim | Kalman screen | Mean value error | Mean `q_scale` error | Simultaneous `q_scale` interval | Relative seed SD |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| 1024 | `(20,8)` | `1 x 1`, `K=1024` | PASS | `screen_fail` | `0.08472%` | `-31.65%` | `[-43.59%,-19.71%]` | `15.73%` |
| 2000 | `(20,5)` | `1 x 1`, `K=2000` | PASS | `screen_fail` | `0.10027%` | `-45.75%` | `[-58.79%,-32.70%]` | `17.19%` |
| 5000 | `(20,5)` | `2 x 2`, `K=2500` | PASS | `screen_fail` | `0.14819%` | `-9.91%` | `[-17.72%,-2.11%]` | `10.28%` |

Raw final-scope values and HMC scores were:

| N | LEDH value | Kalman value | LEDH mean HMC score | Kalman HMC score |
| ---: | ---: | ---: | --- | --- |
| 2000 | `-135.9395266` | `-136.0759746` | `[2.67102,-2.69709,0.28254,-0.97797,1.97595]` | `[2.72366,-2.67495,0.26532,-0.67101,1.95942]` |
| 5000 | `-135.8743277` | `-136.0759746` | `[2.71803,-2.68320,0.23621,-0.73752,1.92387]` | `[2.72366,-2.67495,0.26532,-0.67101,1.95942]` |

The `N=5000` absolute mean `q_scale` error was `-0.06651`, versus
`-0.21239` at `N=1024` and `-0.30696` at `N=2000`.

## Tuning And Claim Evidence

`N=2000` used calibration `81900..81907`, validation `81908..81915`,
and untouched claim `81920..81935`. Blind tuning rejected `(20,3)` because
calibration `E_row=0.01499`, then selected `(20,5)`. The claim passed with
`TV_col=3.34e-6`, `E_row=0.003350`, finite value/score, exact replay, and exact
scope/work identity.

The first `N=5000` phase used tuning `82000..82015`, selected `(20,3)`, and
failed its untouched claim `82020..82035`: seeds `82024`, `82027`, and `82030`
exceeded the row gate, with worst `E_row=0.020104`. That claim was preserved
and excluded from repair selection and the final bias screen.

The authorized repair used fresh calibration `82200..82207`, validation
`82208..82215`, and untouched claim `82220..82235`. It selected `(20,5)` and
passed with `TV_col=2.35e-6`, `E_row=0.008557`, finite value/total score,
bitwise replay, correct `K=2500,2 x 2` identity, `StatelessWhile`, no Python
horizon unroll, and exact work accounting.

## Multi-Block Harness Repair

The initial `N=5000` resource probe exposed a real implementation gap: the
optimized shared final transport/marginal state still required `K=N`, which is
wrong for the active `N=5000 -> K=2500` policy scope. The repair did not alter
the Sinkhorn or terminal-balance scalar. It:

- reused the existing blockwise value/JVP transport application;
- exposed its explicit row-mass value and total tangent, avoiding a TF32
  all-ones payload GEMM;
- ran one marginal-only blockwise pass from the same frozen potentials and row
  masses, without reconstructing Sinkhorn/balance state; and
- records `marginal_tile_sweeps=1` per active time step for multi-block scopes.

The exact `N=5000,K=2500` resource retry passed. The final claim recorded 800
Sinkhorn states, 800 terminal-balance states, 800 transport sweeps, 800
marginal-only sweeps, and zero diagnostic solver reconstructions across the 16
size-one microbatches.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain `(20,5)` for exact `T=50,N=2000` engineering scope only | Direct claim PASS | Kalman `q_scale` screen fail | Independent finite-seed rung | Preserve as negative particle-ladder evidence | No cross-scope/default control |
| Retain `(20,5)` for exact `T=50,N=5000` engineering scope only | Direct fresh claim PASS | Value interval wholly outside; score screen not closed | Cancellation-sensitive long-horizon score and finite-particle recursion | Run the predeclared time-local active/no-reset/Kalman score decomposition | No Kalman/HMC correctness |
| Classify larger-N hypothesis | `N=5000 q_scale` descriptively closer; `N=2000` worse | Non-monotone sequence and no supported ranking | Whether value and residual score bias are finite-N or structural | Decompose same-stream time-local score components before another N/seed ladder | No `1/N` rate |
| Nonlinear continuation | Prerequisite not met | Continuation veto remains | Nonlinear behavior not tested | Do not launch `T=10,N=5000` or nonlinear scopes | No nonlinear failure claim |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto evidence | `N=2000 q_scale` interval wholly outside; `N=5000` value interval wholly outside the frozen region. |
| Viable candidates | Both final finite programs are engineering-valid for their exact scope; neither is Kalman-certified. |
| Statistically supported ranking | None. Cross-`N` differences are descriptive because seeds are independent and no ranking analysis was predeclared. |
| Descriptive-only differences | `N=5000` smaller mean/SD `q_scale` error; runtimes, memory, OPG diagnostics, and non-monotone `N=2000` behavior. |
| Default readiness | No new default and no HMC-facing status. |
| Next evidence needed | Same-observation/same-stream time-local decomposition of stationary, proposal, likelihood-normalization, carried-weight, and Contract E reset score contributions, with same-partial-scalar derivative checks. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b` plus preserved shared uncommitted work and this campaign's source changes |
| Environment | conda `tf-gpu`; TensorFlow `2.19.1`; Python recorded in per-scope manifests |
| Hardware | NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; verified 8192 MiB logical-device limit |
| GPU policy | `bayesfilter.tensorflow.gpu_memory_policy.v1`, mode `fixed_logical_device_limit`; memory growth correctly not claimed simultaneously |
| Production settings | float32, TF32 enabled, GPU/XLA, `StatelessWhile`, no Python horizon unroll |
| N=2000 microbatch / peak / campaign wall | `4`; `865,357,568` bytes; `679.89 s` |
| N=5000 historical microbatch / peak / final campaign wall | `1` (conservative campaign choice, not active guidance); `388,324,608` bytes; `2742.96 s` |
| All recorded probes/smoke/scope attempts | `6320.82 s` total wall (`1.756 h`), within the 12 GPU-hour campaign budget |
| N=2000 result SHA-256 | `dfa230b243577d929304bc14c776c3841e1e6bfbed6493332adf651d49d1edc4` |
| Failed N=5000 claim SHA-256 | `649017cc3ce117f1abbd9bb0d21c04ee5e09ecb767367f91bfb1ca2134f753d4` |
| Final N=5000 result SHA-256 | `2a940a3c3a911e4331aebe0d5bcab7849bc2b265d2591620f47965ca3c3109dc` |
| Aggregate SHA-256 | `fab768b961214fb5d962fd05a7d868802a65ae55a7cd0d73451de7f662dc495e` |
| Aggregate execution | Explicit CPU-only diagnostic postprocessing with `CUDA_VISIBLE_DEVICES=-1`; no algorithm rerun |

## Checks

- focused campaign/binding/aggregate checks: `47 passed`;
- earlier scope/transport/canonical focused checks: `43 passed`, `5 passed`,
  and `10 passed` subsets;
- Python compilation: pass;
- diff whitespace checks: pass;
- trusted `nvidia-smi` and TensorFlow GPU/fixed-memory-policy probe: pass;
- exact `N=2000,K=2000` and repaired `N=5000,K=2500` resource probes: pass;
- tiny end-to-end microbatch tuning/selection/claim smoke: pass.

## Post-Run Red Team

The strongest alternative explanation for the improved `N=5000 q_scale` mean
is ordinary between-seed variation rather than particle-count convergence. The
sequence is non-monotone, uses independent seed blocks, and has only 16 claim
seeds per rung. A paired common-random-number design or reviewed hierarchical
uncertainty model would be needed to rank particle counts.

The strongest evidence against declaring success is the frozen value interval,
which is wholly outside at `N=5000`, plus the `q_scale` and `phi3` intervals that
remain uncontained. The result would be overturned by a source-matched
time-local decomposition exposing a target/harness error, or by a predeclared
fresh-scope study whose simultaneous intervals all lie inside the frozen
regions. Neither evidence exists.

The weakest implementation evidence is the new multi-block final marginal
composition: its constituent blockwise value/JVP and marginal primitives have
focused checks and the exact production-scale GPU/XLA route passes replay and
marginals, but no small alternative chunk policy is eligible for a synthetic
`K<N` route test. This limitation does not justify treating the scientific
screen as passed.
