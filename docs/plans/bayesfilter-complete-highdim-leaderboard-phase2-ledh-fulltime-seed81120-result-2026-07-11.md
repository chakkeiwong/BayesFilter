# Complete High-Dimensional Leaderboard Phase 2 Result

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `STATUS_COMPLETE_WITH_BLOCKERS_NO_PHASE3_ELIGIBLE_ROWS`

## Outcome

Phase 2 is closed with a truthful terminal classification for every LEDH row.
No row produced a passing exact full-time seed-`81120` score/FD pair, so no row
is eligible for Phase 3, no Phase 3 authority may be created, and no
leaderboard cell is admitted.

The shared harness repair passed its checks and the trusted GPU/XLA/TF32
preflight passed. The subsequent evidence separates three numerical FD vetoes
from three row-specific 900-second score timeouts. These are candidate/runtime
failures, not evidence that the shared target, harness, GPU, data, or math is
invalid. Prefix successes remain immutable diagnostic checkpoints only.

## Claimed Target And Computed Quantity

- Claimed Phase 2 target: for each of six current-source LEDH routes, execute
  the ordered seed-`81120` ladder through exact full time and require a finite,
  trusted GPU/XLA/TF32 score plus separately executed same-scalar FD evidence
  satisfying every individual direction.
- Quantity actually computed: complete ordered ladders until the first
  row-specific FD or timeout veto, under the frozen target, source,
  configuration, route, seed, endpoint, command, device, XLA, TF32, and dtype
  identities.
- Relationship: the artifacts are `correct` for the executed prefix/full-time
  engineering screens. They do not supply a passing full-time pair for any row
  and therefore do not establish LEDH cell admission, a complete leaderboard,
  HMC readiness, posterior correctness, ranking, confidence coverage, default
  readiness, or scientific validity.
- Zhao-Cui classification: all six adapters remain owner-approved
  `extension_or_invention`; no source-faithful or author-reproduction claim is
  made.

## Row Decisions

| Row | Last valid paired checkpoint | Terminal executed evidence | Row classification | Phase 3 eligible |
| --- | --- | --- | --- | --- |
| Fixed-SIR | `T=5` FD pass | Exact full `T=20` score completed; FD failed | `row_candidate_blocked` | No |
| Predator-prey | `T=1` FD pass | `T=5` score completed; FD failed | `row_candidate_blocked` | No |
| LGSSM | `T=10` FD pass | Exact full `T=50` score completed; FD failed | `row_candidate_blocked` | No |
| Actual-SV | `T=4` FD pass | `T=50` score hit exact 900s timeout; artifact remains nonterminal | `row_candidate_blocked` | No |
| Generalized-SV | `T=4` FD pass | `T=50` score hit exact 900s timeout; artifact remains nonterminal | `row_candidate_blocked` | No |
| KSC-SV | `T=4` FD pass | `T=50` score hit exact 900s timeout; artifact remains nonterminal | `row_candidate_blocked` | No |

All remaining unrun Phase 2 manifest rungs are `deadline_incomplete` at phase
close. This execution status does not replace the row terminal classifications
above: each row already fired a row-specific hard veto. No unrelated-row
continuation veto or shared invalidity fired.

## Numerical Veto Evidence

The FD-only policy remained frozen throughout:

`max_coordinate_relative_error <= 0.05 * sqrt(number_of_parameters)`.

It is an individual-direction validation threshold, not a general score
tolerance and not a computed confidence interval.

| Row/rung | Maximum relative error | Threshold | Failing coordinate | Score | FD | FD artifact SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Fixed-SIR full `T=20` | `0.1366906979` | `0.0866025404` | `log_nu_scale` | `27.2798843` | `31.599201` | `a3620206a37dc9da6ae83efe2ed3521d8fedd2e3b55740975c6985baeb6dbbb6` |
| Predator-prey `T=5` | `0.2405297074` | `0.1224744871` | `a` | `-0.2287312` | `-0.301172` | `dbcd1eee4ea7932534e308eb302fdddb6b3603462d721c8362eeb437055708f2` |
| LGSSM full `T=50` | `0.5088113106` | `0.1118033989` | `q_scale` | `11.2701368` | `5.535764` | `8c58faa76ee07c627cbb30b719f282d66837db52ea6934925838c3cc4aeff844` |

The repaired predator-prey evidence contains finite, distinct endpoints and
nonzero FDs. The historical zero-FD anomaly is absent.

## Timeout Evidence

The three commands below exited through the exact external supervisor timeout
with status `124`. Because the process received external termination while the
compiled call was active, each structured JSON remains
`artifact_status=initialized` and `terminal_artifact=false`; none is a valid
score shard and no paired FD was run.

| Row/rung | JSON SHA-256 | Log SHA-256 | Classification |
| --- | --- | --- | --- |
| Actual-SV `T=50` score | `47069e05255f2b6db2e0fdf9cc7c835c348fb69e33897c50c5af8a232447d065` | `cafc83962c534436a70b6074827496eb498cd88ad183ecbde283ae754ddb049c` | Exact 900s timeout; nonterminal |
| Generalized-SV `T=50` score | `19ea617815a962b0e460d0d4359f7f487ea122effeb8c2f676167e88c137a27f` | `61c3dfc609b90bb40960c34c36183968643703f4216924d2fb6082e5800ec725` | Exact 900s timeout; nonterminal |
| KSC-SV `T=50` score | `f0ff5c06d4a1ee1757d2cfb6628b6e0503a631bbbbf32ba5d1b86d6ddb917c8c` | `b743c546a5658ae033268c2476c5cf0efcd5f013d5c8c73cbd194267edd80d97` | Exact 900s timeout; nonterminal |

## Preserved Prefix Evidence

| Row/rung | Value | Score | Maximum FD error | Threshold | Score JSON SHA-256 | FD JSON SHA-256 |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Generalized-SV `T=4` | `-5.4150171` | `[-0.0152617, -0.0342291, -0.0293059]` | `0.00487299` | `0.08660254` | `a0db1dea78bd9203fb0f364ddb3ec31614df7ef29131834f8445808c8c8842ca` | `5ac6e4605f74f5366b65f7815f4ca71db828aebd6ea8f247c04ef033dca835d1` |
| KSC-SV `T=4` | `-7.5785789` | `[-0.2772164, 0.1562585]` | `0.000271163` | `0.07071068` | `b652df16ea2618f5b547ae9609c9cc89b35d6042264cb9068e23dba3e41add65` | `cfd7d466948ada41f5b2f68c2352af341ba68e18bbe3b22ed8ac3062770cd07c` |

Both pairs passed the current raw score and raw FD validators. They remain
prefix diagnostics and cannot substitute for exact full-time evidence.

## Shared Harness Repair

The first repaired-run FD attempt exposed a shared validator defect:
`_load_score_reference` checked a score shard using the FD command's timeout.
The repair validates the score shard using its own recorded argv, then compares
only genuinely shared score/FD fields. Old evidence was preserved and all
repaired outputs use new `repair1` paths.

| Artifact | SHA-256 |
| --- | --- |
| Repaired harness | `2849903065533976efb7701b24fe8d56720978087b78e53751cdd9a42b5e0cd9` |
| Repair1 exact-command manifest | `bc8a8a9aa67b64b72ff5e9431bf8ea993bc8a97acbb62e52af0cef421bf4229f` |
| Repair result | `c5c96c0fbcf8eb7c4b43003f7eb289ffbd086da8f3b55ec22b5b1f19da6dac5e` |
| Repair review | `9b2f91e6221693a7fca92de2a4d8f3284cbeaf1fd28aabc2aebfc28f4d59507e` |
| Repair1 Phase 2 authority | `e6d8915c9afb499fc661f1301c32efde71a4eaf14b72e02114bfdd9673cd6665` |

Post-repair CPU-hidden checks passed: focused validators `61 passed`, dedicated
harness `132 passed`, six row/cross-model suite `146 passed`, deterministic
manifest `--check`, Python compilation, and scoped diff hygiene. The trusted
GPU preflight passed with artifact SHA-256
`8daebd9efde58a807c699daab45c0cdd1a0c2beffa86549cc2e06f23f3dc17e2`.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Close Phase 2 with blockers and zero Phase 3-eligible rows |
| Primary criterion | Failed for every row because no exact full-time seed-`81120` score/FD pair passed |
| Hard veto status | Fixed-SIR, predator-prey, and LGSSM: FD veto; Actual-SV, generalized-SV, and KSC-SV: exact 900s score-timeout veto |
| Shared continuation veto | None; repaired harness, target identities, command authority, preflight, and raw validators remained valid |
| Main uncertainty | Whether bounded score/gradient correctness or full-time compilation/runtime repairs can recover the six candidates without changing the frozen target |
| Next justified action | Diagnose and repair Phase 2 candidates under a new reviewed repair plan; then rerun the smallest discriminating affected rungs before any Phase 3 execution |
| What is not concluded | No cell admission, complete leaderboard, ranking, superiority, HMC/posterior correctness, confidence coverage, source-faithfulness, default readiness, release, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Three numerical FD vetoes and three exact timeout vetoes are supported |
| Viable candidates | None is eligible for five-seed admission; prefix checkpoints remain repair candidates only |
| Statistically supported ranking | None; one seed and no uncertainty analysis cannot rank candidates or methods |
| Descriptive-only differences | Values, scores, runtimes, memory, and relative-error magnitudes outside their hard-screen role are descriptive only |
| Default-readiness | Not established |
| Next evidence needed | Reviewed root-cause repairs followed by current-source trusted full-time paired score/FD evidence, then exactly five paired seeds and aggregate validation |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command | Exact per-rung `shell_command` entries in the repair1 manifest; each score used a 900s supervisor and each prefix FD used a 1200s supervisor |
| Environment | `tf-gpu`; Python `3.11.14`; TensorFlow `2.19.1`; TFP `0.25.0` |
| CPU/GPU status | Trusted NVIDIA GeForce RTX 4080 SUPER GPU; `/GPU:0`; XLA JIT; float32; TF32 enabled. Validators/checks deliberately hid GPU with `CUDA_VISIBLE_DEVICES=-1` |
| Data version | Canonical target SHA-256 `1cc83076b491b7c059fadbef85cacbb138c974a39502f5418d9018c17ef8fec8` |
| Random seeds | Singleton `81120` for every executed Phase 2 rung |
| Wall time | Per-command elapsed times are stored in terminal JSON; Actual-SV, generalized-SV, and KSC-SV `T=50` score commands each exhausted exactly 900s |
| Output artifacts | `docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh-repair1/` and matching logs under `docs/plans/logs/complete-highdim-leaderboard/phase2-ledh-repair1/` |
| Plan | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-subplan-2026-07-11.md` |
| Repair result | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-score-reference-timeout-repair-result-2026-07-12.md` |
| Result | This file |
| Review | `docs/reviews/bayesfilter-complete-highdim-leaderboard-phase2-result-phase3-subplan-local-review-2026-07-12.md` |

## Post-Run Red Team

- Strongest alternative explanation: full-time FD failures may reflect a
  bounded score implementation defect, numerical accumulation problem, or
  frozen FD comparator sensitivity rather than evidence against the underlying
  filtering idea. The timeout rows may be compile-time/runtime engineering
  failures rather than numerical failures.
- Result that would overturn this close classification: a reviewed repair that
  preserves the exact target and produces current-source terminal full-time
  score/FD shards passing every identity, memory, and individual-direction FD
  gate.
- Weakest evidence: only one seed was attempted and the three timeout artifacts
  contain no terminal numeric result. No stochastic ranking is supported.

## Handoff

The dedicated Phase 3 subplan is drafted but blocked with an empty eligible-row
set. Do not create Phase 3 execution authority and do not run any Phase 3 seed
or aggregate command. Resume through a new Phase 2 repair subplan that binds
this result, preserves every failed artifact, states the exact root-cause
hypothesis and repair discriminator, and regenerates/reviews identities if any
computation-relevant source changes.
