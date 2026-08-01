# Phase 9 Gate C Result: Generalized-SV Prefix Ladder

Date: 2026-07-11

Status: `GATE_C_FAILED_FD_BLOCKED_BEFORE_T50_AND_GATE_D`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop generalized-SV after the first `T=4,N=10000`, seed-`81120` rung; do not run `T=50,252,1008` or Gate D. | Trusted GPU/XLA score execution, finite output, terminal artifact, source-route target identity, and the `14000 MiB` prefix memory screen passed. The same-scalar FD criterion failed. | Row-local FD veto fired: `max_abs=0.0151546374 > 0.005` and `max_rel=0.442753971 > 0.005`, driven by `log_tau`. No shared harness continuation veto fired. | Current evidence does not distinguish compact-score error from float32 central-FD resolution error or their interaction. | Preserve the row as rejected by the current frozen ladder. Continue only KSC-SV in nonlinear Gate C. Any alternative step, tolerance, precision, or score repair requires a revised reviewed plan and cannot retroactively pass this run. | No full-time generalized-SV score or memory result, five-seed admission, HMC readiness, posterior correctness, runtime or memory superiority, statistical ranking, or rejection of the compact-score research direction. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the source-route prior-mean generalized-SV compact score execute at `T=4,N=10000` under trusted GPU/XLA/TF32, remain within the prefix score-memory budget, and pass the frozen same-scalar FD rule before the next rung? |
| Exact comparator | Float32 central finite differences of the row-matched value-only scalar at the same seed, prepared inputs, source-route prior-mean target, transport settings, and active transformed coordinates. |
| Primary criterion | Failed. Score execution and prefix memory passed, but FD failed both branches of the frozen OR rule. |
| Promotion veto | Same-scalar FD failure at the first `N=10000` prefix. |
| Continuation veto | Row-local continuation to `T=50` and Gate D is vetoed. The terminal trusted artifacts show no corruption, device, XLA, provenance, finite-output, or memory failure that would invalidate KSC-SV. |
| Explanatory only | Compile time, elapsed time, the `35.231 MiB` prefix peak, objective, score magnitude, and coordinate-level error pattern. |
| Artifact | Two Gate C JSON shards, two Markdown summaries, two logs, and this result. |

## Prefix Result

Both shards use seed `81120`, `T=4`, `N=10000`, `float32`, TF32 enabled,
`jit_compile=True`, logical `/GPU:0`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

| T | Score status | Reset score peak MiB | FD max abs / atol | FD max rel / rtol | Frozen rule | Row decision |
| ---: | --- | ---: | --- | --- | --- | --- |
| 4 | Terminal, finite, GPU | `35.23095703125` | `0.0151546374 / 0.005` | `0.442753971 / 0.005` | Fail both branches | Stop before `T=50` and Gate D |

The score peak passes the `N=10000` prefix memory screen but is not full-row
`T=1008` memory evidence. No threshold or FD step changed after observing the
rung.

## FD Detail

The claimed target is the score of the raw source-route prior-mean
generalized-SV scalar in `gamma_unconstrained`, `log_tau`, and `mu`
coordinates. The score shard computed the compact forward sensitivity of that
scalar. The FD shard computed central differences of the row-matched value-only
scalar with step `1e-4`.

| Parameter | Compact score | Finite difference | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| `gamma_unconstrained` | `-0.0152896475` | `-0.0190734863` | `0.0037838388` | `0.1983821243` |
| `log_tau` | `-0.0342281237` | `-0.0190734863` | `0.0151546374` | `0.4427539706` |
| `mu` | `-0.0293109789` | `-0.0286102295` | `0.0007007495` | `0.0239074063` |

`gamma_unconstrained` and `mu` pass the coordinate-level absolute threshold,
while `log_tau` fails both coordinate-level thresholds. Because the declared
gate uses all-coordinate maxima, the candidate fails. The quantities differ
beyond the frozen criterion, so the current generalized-SV score candidate is
wrong relative to this admission claim at `T=4`. This result does not by itself
identify whether the compact recurrence, the float32 finite-difference
comparator, or their interaction causes the discrepancy.

## Engineering Correctness Ledger

- Score and FD processes emitted terminal structured artifacts.
- Score and value outputs were finite and placed on `/GPU:0` under XLA JIT.
- Production `float32`, TF32 enabled, singleton seed `81120`, the expected
  generalized-SV row id and source-route prior-mean target, and the
  managed-session trust basis are recorded.
- The FD shard names the exact score JSON, binds score SHA-256
  `3fb140284b74a02efb8fe57562f0f33ee75a1012bd1dd6cdc554be71c59e71d6`,
  and matches prepared-input fingerprint
  `d8e322dbaac08835fc0499c5161d77ba02777deeac2d4f26022e77ca3202c643`.
- The prefix score peak is `35.23095703125 MiB`, below the frozen
  `14000 MiB` budget. It is not full-time memory evidence.
- The FD failure is numerical evidence against this candidate under the frozen
  screen, not an execution or artifact failure.

## Numerical Validity Ledger

- The first `N=10000` prefix fails the predeclared absolute-or-relative
  same-scalar FD rule.
- The row is blocked before longer prefixes and Gate D even though the score is
  finite and below budget.
- No alternative FD arm was run after observing the failure.

## Scientific Interpretation Ledger

- This is engineering and same-scalar numerical-screen evidence only.
- Candidate rejection is not research-direction rejection. KSC-SV remains a
  separately identified target and may continue under its own Gate C ladder.
- No stochastic ranking is supported by a single seed, and none is attempted.
- No posterior, filtering-accuracy, or HMC claim follows from either the prefix
  memory pass or the FD failure.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| Score JSON | `3fb140284b74a02efb8fe57562f0f33ee75a1012bd1dd6cdc554be71c59e71d6` |
| FD JSON | `edc896ef4a41772e29487257b3c6e01c8543b780aacf930fa607dd43479b8b08` |

The shards also bind these unchanged decision inputs:

- Gate B result SHA-256:
  `cbfe9ab65929745345d32a765c7067a1c9875dc95d18095abfc84f205010c605`.
- Final Gate B review SHA-256:
  `ac1c25f3d24cf329abc0cfabf9cc928f0367d95064e778a749b39f0f3fd70312`.
- Runner SHA-256:
  `fa1e9602023e96e3a1b68e1a6547397eef7cf1f1d4cd95b2c62b9a61fa13e8fe`.
- Exact-command manifest SHA-256:
  `ffa2d232c0582d28d13d57a5ca188ad8d9f30e279555093532561aa264559e3d`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the dirty worktree recorded inside both shards |
| Commands | The two literal generalized-SV `T=4,N=10000` Gate C `shell_command` entries in `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json`, score before FD |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; Python `3.11.14`; TensorFlow `2.19.1` |
| CPU/GPU status | Trusted GPU 0; NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; TF32 enabled; XLA JIT; output on `/GPU:0` |
| Data version | `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`, SHA-256 `5afb71144576bdb0070080f684b5d5b41f33de77889105b10bcd78e36b77dd77` |
| Random seeds | Singleton seed `81120` |
| Wall time | Score elapsed `75.9381s`; FD elapsed `46.0792s` |
| Output artifacts | `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/generalized-sv-t4-n10000-seed81120-{score,fd}.{json,md}` and matching logs under `docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/` |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Same-scalar FD fails at `T=4`; longer prefixes and Gate D are blocked. Prefix memory, finite-output, device, XLA, trust, and artifact screens pass. |
| Statistically supported ranking | None. One fixed seed cannot support a ranking. |
| Descriptive-only differences | Runtime, prefix peak, objective, score, and coordinate error magnitudes beyond their hard pass/fail use are descriptive only. |
| Default-readiness | No new default-readiness conclusion. The owner-directed algorithm default is unchanged, while this generalized-SV score candidate is not admitted. |
| Next evidence needed | A revised and reviewed diagnostic plan is required before investigating the generalized-SV discrepancy. KSC-SV may continue its reviewed Gate C ladder. |

## Post-Run Red Team

- Strongest alternative explanation: float32 cancellation at FD step `1e-4`
  may account for the mismatch, especially the identical quantized FD values
  for `gamma_unconstrained` and `log_tau`, rather than the compact recurrence
  being wrong. That explanation is not established and does not erase the
  failed criterion.
- Result that would overturn the candidate decision: a predeclared, reviewed,
  production-relevant derivative check of the unchanged source-route scalar
  that passes without post-hoc threshold selection. It would create new
  evidence, not retroactively change these shards.
- Weakest evidence: one seed, a short prefix, and one frozen FD step cannot
  isolate the source of disagreement.

## Gate Boundary

Generalized-SV `T=50,252,1008`, Gate D score/FD shards, and aggregation are
blocked. KSC-SV remains eligible to continue its independently reviewed Gate C
ladder. LGSSM remains blocked by the existing Phase 9 gate ordering.
