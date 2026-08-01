# LGSSM N=10000 Tuned Kalman Certification Plan

Date: 2026-07-20
Status: `COMPLETE_ENGINEERING_PASS_KALMAN_SCREEN_FAIL`
Parent diagnostic:
`docs/plans/bayesfilter-lgssm-n10000-single-seed-kalman-diagnostic-result-2026-07-20.md`

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | After independent tuning for the exact `T=50,N=10000,K=2500,4 x 4` scope, does the canonical Contract E--Chol value and total score agree with the exact differentiated Kalman likelihood under the same frozen screen used for earlier particle counts? |
| Candidate/mechanism | Increasing particle count may reduce finite-particle error, while scope-specific Sinkhorn/balance tuning repairs numerical marginal error without using Kalman information. |
| Exact baseline | The independently tuned final `N=5000` claim and its frozen Kalman screen; the `N=10000` single-seed `(20,5)` run is a resource/warm-start diagnostic only. |
| Promotion criterion | The untouched 16-seed claim passes every hard engineering gate; its simultaneous 95% value relative-bias interval lies inside `[-0.001,+0.001]`; and all five score relative-bias intervals lie inside `[-0.05,+0.05]`. |
| Promotion veto | Any interval wholly outside its frozen region, or any hard route, scope, chunk, replay, finite, chart, reset, marginal, GPU/TF32/XLA, graph, work-accounting, target-identity, or artifact failure. |
| Continuation veto | Invalid target/artifact, inability to run policy chunks with singleton execution under the 8192 MiB limit, exhausted eight-GPU-hour budget, no direct-gate-valid controls within the declared grid, or failure of exact scope/tuning identity. |
| Repair trigger | Direct marginal failure advances the declared control grid. A localized harness/serialization failure permits one fresh versioned retry without changing target, seeds, method, precision, hardware class, or total budget. |
| Explanatory diagnostics | Per-seed value/score errors, SD/SE, simultaneous intervals, marginal residuals, runtime, allocator peak, and descriptive comparison with `N=5000`. |
| Must not be concluded | No monotonic `1/N` rate, statistical superiority, nonlinear validity, HMC/posterior readiness, universal controls, or new default follows from this campaign. |

## Evidence Contract

Tuning is Kalman-blind. Calibration and validation may select controls only by
the direct canonical gates: `TV_col <= 1e-4`, `E_row <= 1e-2`, finite value and
total score, valid chart/reset, exact work accounting, `StatelessWhile`, no
Python horizon unroll, and exact GPU/TF32/XLA/chunk/scope identity.

The selected controls are frozen before the untouched claim. The claim uses the
exact Kalman value and five-coordinate physical score and two-sided
Bonferroni-Student simultaneous intervals over six outputs, 16 seeds, critical
value `3.036283222821165`, value margin `0.001`, and score margin `0.05`.

Artifacts will be written without overwrite under:

`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/`

## Frozen Scope, Seeds, And Grid

| Field | Value |
| --- | --- |
| Model/target | Canonical LGSSM dataset seed `81100`, theta `(0.72,0.55,0.35,0.35,0.45)` |
| Scope | `T=50,N=10000,K=2500`, `4 x 4`, float32, TF32, GPU/XLA |
| Calibration | `82400..82407` |
| Validation | `82408..82415` |
| Untouched claim | `82420..82435` |
| Seed microbatch | `1`; correctness fallback after demonstrated TF32 batch-shape drift |
| First candidate | `(sinkhorn_steps=20,balance_steps=5)`, cross-scope warm start only |
| Balance ladder | `5,8,12,16,25,32` |
| Sinkhorn ladder | `20,25,30,40`, exhausting the balance ladder at each rung |
| Selection rule | First calibration-and-validation direct-gate pass in declared order |

The seed partitions are fresh and disjoint from every prior `N=1024`,
`N=2000`, `N=5000`, and one-seed `N=10000` artifact. Failed tuning candidates
remain tuning evidence; claim seeds are never recycled into tuning.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Contract E--Chol total derivative | Owner-mandated canonical route | Only eligible value/score reset identity | Route drift tests another scalar | Route/source/preparation identities and graph/work gates |
| `(20,5)` first candidate | `N=5000` selected control; `N=10000` one-seed warm start | Passed one exact-scope resource/hard-gate diagnostic | One seed can hide marginal failures; cross-scope transfer can mis-tune | Fresh 8+8 calibration/validation direct gates |
| Singleton microbatch | Correctness fallback after measured TF32 batch-shape dependence | Preserves the finite numerical program used by prior singleton evidence | Slow execution | One-seed `N=10000` runtime and allocator witness |
| 16 tuning plus 16 claim seeds | Prior certification protocol | Separates tuning and claim data and supports simultaneous intervals | Still limited precision for small effects | Report SD/SE and avoid unsupported rankings |
| Float32/TF32/GPU/XLA | Repository production target | Holds execution target fixed across the particle ladder | TF32 recursive sensitivity | Fixed singleton geometry, replay, and explicit nonclaims |
| 8192 MiB logical limit | Existing resource contract | Bounds process allocation; one-seed peak was about 382 MiB | External contention or unexpected candidate memory growth | Allocator peak and structured resource failure |

## Skeptical Plan Audit

Verdict: `PASS`.

- The baseline and criterion are unchanged from the earlier ladder; the
  one-seed `N=10000` result is not promoted into a tuning or claim baseline.
- Selection is blind to Kalman and uses disjoint calibration/validation data,
  so proxy marginal diagnostics cannot silently select for score agreement.
- The exact `N=10000` scope receives its own tuning; `(20,5)` is explicitly a
  first candidate, not a transferred default.
- Singleton execution prevents the known TF32 batch-shape semantic mismatch.
- Claim evidence is multi-seed with simultaneous uncertainty; a hard-gate pass
  alone is not evidence of Kalman agreement or superiority.
- The result can fail for tuning/numerical reasons rather than invalidate the
  particle hypothesis; failure classification must remain explicit.
- Existing external GPU load may affect timing but does not change the fixed
  TensorFlow allocator cap or numerical evidence. No clean performance ranking
  will be made.

## Budget, Pre-Mortem, And Stops

- At most eight GPU-hours total and one localized infrastructure retry.
- At most two GPU-hours for one tuning or claim node.
- Preserve every completed or failed versioned artifact; never overwrite.
- Stop after the first direct-gate-valid pair and its untouched claim.
- Stop if no pair passes the declared grid, the budget is exhausted, or a true
  continuation veto fires.

Pre-mortem:

- The run could pass marginals while retaining structural score bias; the
  untouched Kalman intervals distinguish that case.
- The run could fail because `(20,5)` is not tuned for `4 x 4`; the declared
  balance/Sinkhorn ladder distinguishes a repairable control failure.
- A one-seed apparent improvement could disappear across seeds; the 16-seed
  claim and simultaneous intervals address that risk.
- TF32 could make results batch-shape-dependent; singleton execution fixes the
  numerical batch geometry rather than assuming microbatch equivalence.

## Execution Log

Attempt `n10000-tuned-kalman-attempt01` launched on 2026-07-20 under:

`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/`

The command uses the frozen `T=50,N=10000`, singleton-microbatch, seed, grid,
GPU/TF32/XLA, two-hour node-cap, and eight-hour campaign-cap settings declared
above. The skeptical audit passed before launch because tuning remains
Kalman-blind, claim seeds remain untouched, and the artifacts answer the stated
engineering and simultaneous-bias questions.

The first candidate `(20,5)` completed in `2232.58 s`. Calibration passed, but
validation failed only the direct row-error gate: `E_row=0.0112653 > 0.01`;
`TV_col=1.1852e-6` passed. This is a declared tuning repair trigger, not a
continuation veto. The driver advanced to `(20,8)` without using claim seeds.

Candidate `(20,8)` completed in `2081.99 s` and was frozen as the first blind
direct-gate pass. Calibration reached `E_row=5.3406e-5` and
`TV_col=3.0496e-7`; validation reached `E_row=0.00277203` and
`TV_col=3.6288e-7`. Both partitions were finite with valid charts and resets.
The peak TensorFlow allocator value was `396,638,976` bytes. The driver then
started the untouched claim on seeds `82420..82435`; no Kalman quantity was
used to select `(20,8)`.

The untouched claim completed with `status=SCOPE_CLAIM_PASS` at
`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/`.
It passed the direct engineering gates with `TV_col=3.0400e-7`,
`E_row=2.6166e-5`, bitwise replay, and exact work/scope identity. The claim
wall time was `4096.77 s`, the campaign wall time was `8412.73 s`, and the
peak TensorFlow allocator was `400,609,536` bytes.

CPU-only postprocessing produced
`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/aggregate.json`.
The frozen Kalman screen is `screen_fail`: the value interval is
`[+0.1502%,+0.1968%]` against `[-0.1%,+0.1%]`, and the `q_scale` interval is
`[-22.0036%,-9.7941%]` against `[-5%,+5%]`. The mean relative errors are
`+0.1735%` for value and `-15.8989%` for `q_scale`, both descriptively worse
than the independently tuned `N=5000` means (`+0.1482%` and `-9.9115%`).
This is a valid negative result for the larger-N repair hypothesis, not a
harness invalidation or a claim of a monotone particle-count law.
