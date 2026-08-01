# Phase 9 Gate C Result: KSC-SV Prefix Ladder

Date: 2026-07-11

Status: `GATE_C_FAILED_FD_BLOCKED_BEFORE_T50_AND_GATE_D`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop KSC-SV after the first `T=4,N=10000`, seed-`81120` rung; do not run `T=50,250,1000` or Gate D. | Trusted GPU/XLA score execution, finite output, terminal artifact, KSC mixture-target identity, and the `14000 MiB` prefix memory screen passed. The same-scalar FD criterion failed. | Row-local FD veto fired: `max_abs=0.0102410018 > 0.005` and `max_rel=0.0369351506 > 0.005`. No shared harness continuation veto fired. | Current evidence does not distinguish compact-score error from float32 central-FD resolution error or their interaction. | Preserve the row as rejected by the current frozen ladder. Do not run nonlinear Gate D. Any alternative step, tolerance, precision, or score repair requires a revised reviewed plan and cannot retroactively pass this run. | No full-time KSC-SV score or memory result, five-seed admission, native actual-SV likelihood evidence, HMC readiness, posterior correctness, runtime or memory superiority, statistical ranking, or rejection of the compact-score research direction. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the KSC log-chi-square Gaussian-mixture compact score execute at `T=4,N=10000` under trusted GPU/XLA/TF32, remain within the prefix score-memory budget, and pass the frozen same-scalar FD rule before the next rung? |
| Exact comparator | Float32 central finite differences of the row-matched KSC value-only scalar at the same seed, prepared inputs, mixture-surrogate target, transport settings, and synthetic unconstrained coordinates. |
| Primary criterion | Failed. Score execution and prefix memory passed, but FD failed both branches of the frozen OR rule. |
| Promotion veto | Same-scalar FD failure at the first `N=10000` prefix. |
| Continuation veto | Row-local continuation to `T=50` and Gate D is vetoed. The terminal trusted artifacts show no corruption, device, XLA, provenance, finite-output, or memory failure. |
| Explanatory only | Compile time, elapsed time, the `35.226 MiB` prefix peak, objective, score magnitude, and coordinate-level error pattern. |
| Artifact | Two Gate C JSON shards, two Markdown summaries, two logs, and this result. |

## Prefix Result

Both shards use seed `81120`, `T=4`, `N=10000`, `float32`, TF32 enabled,
`jit_compile=True`, logical `/GPU:0`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

| T | Score status | Reset score peak MiB | FD max abs / atol | FD max rel / rtol | Frozen rule | Row decision |
| ---: | --- | ---: | --- | --- | --- | --- |
| 4 | Terminal, finite, GPU | `35.22607421875` | `0.0102410018 / 0.005` | `0.0369351506 / 0.005` | Fail both branches | Stop before `T=50` and Gate D |

The score peak passes the `N=10000` prefix memory screen but is not full-row
`T=1000` memory evidence. No threshold or FD step changed after observing the
rung.

## FD Detail

The claimed target is the score of the KSC log-chi-square Gaussian-mixture
surrogate scalar in `gamma_unconstrained` and `log_beta` coordinates. The score
shard computed the compact forward sensitivity of that scalar. The FD shard
computed central differences of the row-matched value-only scalar with step
`1e-4`.

| Parameter | Compact score | Finite difference | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| `gamma_unconstrained` | `-0.2772698104` | `-0.2670288086` | `0.0102410018` | `0.0369351506` |
| `log_beta` | `0.1562928855` | `0.1621246338` | `0.0058317482` | `0.0359707735` |

Both coordinates fail both coordinate-level thresholds. The all-coordinate
maxima therefore fail the declared gate. The quantities differ beyond the
frozen criterion, so the current KSC-SV score candidate is wrong relative to
this admission claim at `T=4`. This result does not by itself identify whether
the compact recurrence, the float32 finite-difference comparator, or their
interaction causes the discrepancy.

The target is explicitly
`ksc_log_chi_square_gaussian_mixture_surrogate`; it is not a native actual-SV
likelihood.

## Engineering Correctness Ledger

- Score and FD processes emitted terminal structured artifacts.
- Score and value outputs were finite and placed on `/GPU:0` under XLA JIT.
- Production `float32`, TF32 enabled, singleton seed `81120`, the expected
  KSC-SV row id and mixture-surrogate target, and the managed-session trust
  basis are recorded.
- The FD shard names the exact score JSON, binds score SHA-256
  `232c28ae76c945efc843f296e412f58ef30d3db38e28e958d47be633c9311dae`,
  and matches prepared-input fingerprint
  `d33a1590422ac12605c2bf651c275f3c63c8ac88b19c0febf3c8111502bbc285`.
- The prefix score peak is `35.22607421875 MiB`, below the frozen
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
- Candidate rejection is not research-direction rejection.
- No stochastic ranking is supported by a single seed, and none is attempted.
- No native actual-SV, posterior, filtering-accuracy, or HMC claim follows from
  either the prefix memory pass or the FD failure.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| Score JSON | `232c28ae76c945efc843f296e412f58ef30d3db38e28e958d47be633c9311dae` |
| FD JSON | `288f997acb7dcc0440a5bcd653e34ca626884fc1421e35e2c3fd25048bde366d` |

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
| Commands | The two literal KSC-SV `T=4,N=10000` Gate C `shell_command` entries in `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json`, score before FD |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; Python `3.11.14`; TensorFlow `2.19.1` |
| CPU/GPU status | Trusted GPU 0; NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; TF32 enabled; XLA JIT; output on `/GPU:0` |
| Data version | `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`, SHA-256 `9883721faf8af9fbe96ef75c209f86eda5732aec6ca5e602980d4cf27338b3b6` |
| Random seeds | Singleton seed `81120` |
| Wall time | Score elapsed `76.4188s`; FD elapsed `41.4762s` |
| Output artifacts | `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/ksc-sv-t4-n10000-seed81120-{score,fd}.{json,md}` and matching logs under `docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/` |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Same-scalar FD fails at `T=4`; longer prefixes and Gate D are blocked. Prefix memory, finite-output, device, XLA, trust, and artifact screens pass. |
| Statistically supported ranking | None. One fixed seed cannot support a ranking. |
| Descriptive-only differences | Runtime, prefix peak, objective, score, and coordinate error magnitudes beyond their hard pass/fail use are descriptive only. |
| Default-readiness | No new default-readiness conclusion. The owner-directed algorithm default is unchanged, while this KSC-SV score candidate is not admitted. |
| Next evidence needed | A revised and reviewed diagnostic plan is required before investigating the KSC-SV discrepancy. No nonlinear row is eligible for Gate D. |

## Post-Run Red Team

- Strongest alternative explanation: float32 cancellation at FD step `1e-4`
  may account for the mismatch rather than the compact recurrence being wrong.
  That explanation is not established and does not erase the failed criterion.
- Result that would overturn the candidate decision: a predeclared, reviewed,
  production-relevant derivative check of the unchanged KSC surrogate scalar
  that passes without post-hoc threshold selection. It would create new
  evidence, not retroactively change these shards.
- Weakest evidence: one seed, a short prefix, and one frozen FD step cannot
  isolate the source of disagreement.

## Gate Boundary

KSC-SV `T=50,250,1000`, Gate D score/FD shards, and aggregation are blocked.
All nonlinear rows now have terminal Gate B or Gate C decisions, and no
nonlinear row is eligible for Gate D. LGSSM remains blocked by the existing
Phase 9 gate ordering.
