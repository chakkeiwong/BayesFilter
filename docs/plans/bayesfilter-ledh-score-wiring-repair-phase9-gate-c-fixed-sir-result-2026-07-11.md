# Phase 9 Gate C Result: Fixed-SIR Prefix Ladder

Date: 2026-07-11

Status: `GATE_C_FAILED_FD_BLOCKED_BEFORE_GATE_D`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop fixed-SIR after the full-time `T=20` seed-`81120` rung and do not run Gate D. | Trusted GPU/XLA score execution, finite output, terminal artifacts, and the `14000 MiB` score-memory screen passed at `T=1,5,20`; FD passed at `T=1,5` but failed at `T=20`. | Row-local same-scalar FD veto fired at `T=20`: `max_abs=7.853515625 > 0.01` and `max_rel=0.0566700101 > 0.05`. No shared harness continuation veto fired. | Current evidence does not distinguish compact-score error from float32 central-FD resolution/accumulation error or their interaction. | Preserve the row as rejected by the current frozen ladder. Continue only unrelated reviewed Gate C rows. Any alternative step, tolerance, precision, or score repair requires a revised reviewed plan and cannot retroactively pass this run. | No five-seed admission, HMC readiness, posterior correctness, exact nonlinear-likelihood correctness, runtime or memory superiority, statistical ranking, or rejection of the compact-score research direction. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the fixed-SIR compact same-scalar score execute at `N=10000` under trusted GPU/XLA/TF32 through full `T=20`, remain within the per-seed score-memory budget, and pass the frozen same-scalar FD rule at every prefix? |
| Exact comparator | Float32 central finite differences of the row-matched value-only scalar at the same seed, prepared inputs, target, transport settings, and log-scale coordinates. |
| Primary criterion | Failed. Score execution and memory passed all rungs, but the full-time FD shard failed both branches of the frozen OR rule. |
| Promotion veto | Same-scalar FD failure at `T=20`. |
| Continuation veto | Row-local continuation to Gate D is vetoed. The terminal trusted artifacts show no corruption, device, XLA, provenance, finite-output, or memory failure that would invalidate other rows. |
| Explanatory only | Compile time, elapsed time, sub-budget peak differences, objectives, score magnitudes, and which coordinate dominates each error measure. |
| Artifact | Six Gate C JSON shards, six Markdown summaries, six logs, and this result. |

## Prefix Results

All shards use seed `81120`, `N=10000`, `float32`, TF32 enabled,
`jit_compile=True`, logical `/GPU:0`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

| T | Score status | Reset score peak MiB | FD max abs / atol | FD max rel / rtol | Frozen rule | Row decision |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | Terminal, finite, GPU | `185.3544921875` | `0.0791292191 / 0.01` | `0.0171573609 / 0.05` | Pass by relative tolerance | Continue to `T=5` |
| 5 | Terminal, finite, GPU | `348.32421875` | `2.568359375 / 0.01` | `0.0020681783 / 0.05` | Pass by relative tolerance | Continue to `T=20` |
| 20 | Terminal, finite, GPU | `414.44677734375` | `7.853515625 / 0.01` | `0.0566700101 / 0.05` | Fail both branches | Stop before Gate D |

Passing by one branch of the declared OR rule at `T=1` and `T=5` is a pass;
it is not a claim that the absolute errors were favorable. No threshold or FD
step changed after observing a rung.

## Full-Time FD Detail

The claimed target is the score of the realized finite-`N` fixed-SIR LEDH
scalar in `log_kappa_scale`, `log_nu_scale`, and `log_obs_noise_scale`
coordinates. The score shard computed the compact forward sensitivity of that
scalar. The FD shard computed central differences of the row-matched value-only
scalar with step `1e-3`.

| Parameter | Compact score | Finite difference | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| `log_kappa_scale` | `-2990.702392578125` | `-2982.848876953125` | `7.853515625` | `0.0026259769` |
| `log_nu_scale` | `27.304161071777344` | `25.756834030151367` | `1.5473270416` | `0.0566700101` |
| `log_obs_noise_scale` | `422.55389404296875` | `416.9921569824219` | `5.5617370605` | `0.0131621957` |

The quantities differ beyond the frozen all-coordinate criterion. Therefore the
current fixed-SIR score candidate is wrong relative to this admission claim at
`T=20`. This result does not by itself identify whether the compact recurrence,
the float32 finite-difference comparator, or their interaction causes the
discrepancy.

## Engineering Correctness Ledger

- Every score and FD process emitted a terminal structured artifact.
- Every score and value output was finite and placed on `/GPU:0` under XLA JIT.
- Production `float32`, TF32 enabled, singleton seed `81120`, the expected
  fixed-SIR row id, and the managed-session trust basis are recorded.
- Every FD shard names the corresponding score JSON and binds its exact SHA-256.
- The full-time score peak is `414.44677734375 MiB`, below the frozen
  `14000 MiB` budget. This is a hard memory-screen pass, not a comparative
  memory claim.
- The full-time FD failure is numerical evidence against this candidate under
  the frozen screen, not an execution or artifact failure.

## Numerical Validity Ledger

- `T=1` and `T=5` pass the predeclared absolute-or-relative same-scalar FD rule.
- `T=20` fails both branches of that rule.
- The row is blocked from Gate D even though its full-time score is finite and
  below budget.
- No alternative FD arm was run after observing the failure.

## Scientific Interpretation Ledger

- This is engineering and same-scalar numerical-screen evidence only.
- Candidate rejection is not research-direction rejection. The compact-score
  route remains separately testable for other row targets.
- No stochastic ranking is supported by a single seed, and none is attempted.
- No posterior, filtering-accuracy, or HMC claim follows from the memory pass or
  the FD failure.

## Artifact Hashes

| Rung | Score JSON SHA-256 | FD JSON SHA-256 |
| --- | --- | --- |
| `T=1` | `ca991fa7c4dad820d715fcecedc3f591c5d52a144308cc68bdf2d5287240e794` | `606c194d7dc2d7264bd41f1f7a0b7ebcf08219fb74ff429ca97492280fbc1c4a` |
| `T=5` | `2a19ef266749e876f741a0955556d92d56cf15178c041585ac9671fa3aa878f1` | `18cc1c4fc18bb1800109cf54440fa33cadab3f0367f34349e4d50afb4b09b299` |
| `T=20` | `7acf4612b4082533cfa076635f1788015ffae43da94f15eb4e818e57c2036773` | `00944bcb7f756f914b56f920b62709e9c4d9a950b5dffcf8589ac83fd68f0036` |

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
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the dirty worktree recorded inside every shard |
| Commands | The six literal fixed-SIR Gate C `shell_command` entries in `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json`, executed in ascending `T` with score before FD |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; Python `3.11.14`; TensorFlow `2.19.1` |
| CPU/GPU status | Trusted GPU 0; NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; TF32 enabled; XLA JIT; output on `/GPU:0` |
| Data version | `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json`, SHA-256 bound in each shard |
| Random seeds | Singleton seed `81120` |
| Wall time | Score elapsed `17.3579/72.7900/532.9583s`; FD elapsed `10.9973/21.4934/49.5846s` for `T=1/5/20` |
| Output artifacts | `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/fixed-sir-*.json`, matching Markdown files, and matching logs under `docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/` |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Same-scalar FD fails at full `T=20`; Gate D is blocked. Memory, finite-output, device, XLA, trust, and artifact screens pass. |
| Statistically supported ranking | None. One fixed seed cannot support a ranking. |
| Descriptive-only differences | Rung-to-rung peaks, runtimes, objectives, scores, and error magnitudes beyond their hard pass/fail use are descriptive only. |
| Default-readiness | No new default-readiness conclusion. The owner-directed algorithm default is unchanged, while this fixed-SIR score candidate is not admitted. |
| Next evidence needed | A revised and reviewed diagnostic plan is required before investigating the fixed-SIR discrepancy. Unrelated reviewed Gate C rows may continue. |

## Post-Run Red Team

- Strongest alternative explanation: accumulated float32 central-difference
  error at full `T=20` may make the frozen FD comparator insufficiently
  resolved, rather than the compact recurrence being wrong. That explanation
  is not established and does not erase the failed criterion.
- Result that would overturn the candidate decision: a predeclared, reviewed,
  production-relevant derivative check of the unchanged scalar that passes
  without post-hoc threshold selection. It would create new evidence, not
  retroactively change these shards.
- Weakest evidence: one seed and one frozen FD step cannot isolate the source of
  disagreement.

## Gate Boundary

Fixed-SIR Gate D score, FD, and aggregation commands are blocked. Generalized-SV
and KSC-SV remain eligible to continue their independently reviewed Gate C
ladders. LGSSM remains blocked by the existing Phase 9 gate ordering.
