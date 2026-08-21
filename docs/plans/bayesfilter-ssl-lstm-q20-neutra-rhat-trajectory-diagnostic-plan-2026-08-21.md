# SSL-LSTM q=20 NeuTra R-hat trajectory diagnostic plan (2026-08-21)

Status: `TERMINAL_DIAGNOSTIC_SCREEN_FAILED_AT_4000`

This plan governs one diagnostic-only continuation authorized by the user's
2026-08-21 instruction to test whether the failed SSL-LSTM q=20 seed-2,
`L=5` fixed-HMC verification was primarily short of samples and whether its
R-hat is declining. It does not reopen or overwrite the terminal `r2`
campaign. The first launcher invocation is preserved as a terminal harness
failure under:

`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory`

The only GPU-initializing diagnostic attempt, after the bounded launcher
repair, is written under the fresh root:

`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | When the exact failed seed-2, `L=5` fixed kernel is run for 4,000 draws per chain, how do cumulative and recent-window R-hat evolve, and does every chain actually cross the sign boundary of `observation_weight.0.0`? |
| Mechanism under test | The simple longer-run explanation: the prior 2,000-draw verification may have stopped while a genuinely mixing common kernel was still approaching stationarity. |
| Candidate under test | Frozen transport seed 2 and fixed identity-`z` HMC with `L=5`, step size `0.2460072308515237`, four prior initialized chains, 64 burn-in steps, and stateless seed `(20260820, 52000)`. |
| Expected alternative failure | Cumulative R-hat may decline while recent-window R-hat remains high or one or more chains remain sign-locked. That would make extra draws look promising without demonstrating global mixing. |
| Primary diagnostic endpoint | At 4,000 draws, report the all-parameter physical R-hat screen, observation-weight R-hat, sign-indicator R-hat, and direct per-chain sign occupancy/transitions. |
| Diagnostic pass | Physical and sign-indicator rank-normalized split/folded R-hat are all `<=1.01`, and every chain visits both signs and transitions at least once. This means only that doubling to 4,000 draws was sufficient for this diagnostic screen. |
| Diagnostic fail | The 4,000-draw diagnostic pass is not met. This means only that doubling the verification length was insufficient at this horizon. |
| Run-validity veto | Immutable identity drift; nonfinite state, target, score, or log acceptance; invalid target status; sample/trace shape drift; GPU/XLA/memory-growth failure; corrupt or incomplete raw archive; or artifact collision. |
| Continuation veto | The measured forecast cannot fit with closeout, the external timeout fires, or the one GPU-initializing attempt otherwise cannot complete. The preserved pre-TensorFlow launcher failure permits only the audited import-path repair and fresh-root retry; no further retry or second candidate is authorized. |
| Explanatory diagnostics | Acceptance, runtime, cumulative checkpoint series, recent-window series, per-chain sign counts/transitions, adjacent checkpoint changes, and the 2,000-to-4,000 net change. |
| Must not be concluded | Posterior admission, canonical warm-up readiness, ESS adequacy, posterior correctness, mode weights, predictive validity, transport/kernel superiority, universal sample sufficiency, model adequacy, production readiness, or default readiness. |

A failed endpoint rejects neither the frozen transport nor the NeuTra research
direction. A passing endpoint does not reinstate the `r2` candidate or bypass
canonical tuning and sequential HMC admission.

## Exact baseline and immutable bindings

The runner must validate the full `r1` training hash graph, full `r2` HMC
hash graph, target/adapter signatures, frozen transport identity, and these
top-level artifacts before target construction:

| Artifact | SHA-256 |
|---|---|
| `r2/hmc/result.json` | `9092b82d25f8e63d1708c63c7d48284ef3c55a5edc3e225833ba505f0b65e706` |
| `r2/hmc/manifest.json` | `c662e56370fe6dd111916232dfede31ac169ce214d03a95431583fe9ca6a92d6` |
| `r2/hmc/artifact-hashes.json` | `d874cf937fe6b7cca69be6c3aa0274ad3ca3ba80a0d409a5a943370425f5a14e` |
| Seed-2 `L=5` tuning result | `b9006654df5fe52a44c1057dc52646ab403de427afdc8b0bfe56e780575987fe` |
| `r2` continuation runner | `eeef1880cb26a7649ccf76230b909518fa1ca4a3e94e3bbc35e38de654d57723` |

The baseline terminal verification used four chains and 2,000 draws per chain.
Its observation-weight rank and folded R-hat were
`1.0786399742045318` and `1.0875996310350042`; the latter was the maximum
physical R-hat and failed the `1.01` screen. Raw chains were not preserved.

The new 4,000-draw call deliberately reuses the same stateless seed. Its
2,000-draw prefix is compared with the prior reported rank/folded R-hat and
acceptance. Exact equality of every saved summary float is required to call the
new prefix a deterministic replay of the prior result; maximum absolute
residuals are also reported descriptively. Failing this tie-out does not
invalidate the new within-run trajectory, but forbids calling it an exact
extension or treating cross-run differences as a time series.

## Evidence contract

### Engineering evidence

The runner must use the same batch-native TensorFlow target, frozen transport,
exact total transformed score, shared TensorFlow/TFP fixed-HMC mechanics,
float64, XLA JIT, TF32 disabled, and one trusted GPU. It must configure and
verify TensorFlow memory growth before logical-device or target initialization.
The active q=20 canonical runner remains unchanged; this new route is classified
as `smoke_mechanics_or_reference` because it is diagnostic-only and cannot
admit a posterior.

The runner archives, with SHA-256 receipts:

- latent and physical samples in `[draw, chain, parameter]` layout;
- sign labels in `[draw, chain]` layout;
- acceptance, log-acceptance, target-value, proposed-target-value, score, and
  available target-status trace tensors;
- checkpoint diagnostics, launch receipt, result, manifest, and complete
  artifact inventory.

### Diagnostic evidence

Cumulative checkpoints are frozen at
`{500,1000,1500,2000,2500,3000,3500,4000}` draws per chain. At every
checkpoint, compute physical rank-normalized split/folded R-hat, the same
diagnostic for the binary observation-weight sign indicator, and direct
per-chain sign counts and transition counts. At checkpoints from 1,000 onward,
repeat those diagnostics on the trailing 1,000 draws. The 1,000-draw recent
window is inherited from `bayesfilter_neutra_sequential_hmc_v1`; it is an
explanatory localization diagnostic here, not canonical warm-up evidence.

The result reports the checkpoint values without fitting an extrapolation.
`R-hat dropping` is represented by the signed 2,000-to-4,000 change and the
number of adjacent checkpoint decreases, not by a claim that the series is
monotone or will cross `1.01`. Cumulative decline alone is insufficient:
recent-window behavior and direct sign transitions must be shown beside it.

### Claim boundary

This call retains only 64 burn-in steps and is not
`bayesfilter_neutra_sequential_hmc_v1`. Even if its 4,000-draw endpoint passes,
the samples remain diagnostic and cannot be used for posterior summaries,
predictive work, leaderboards, or default selection. A future claim-bearing run
would still need a reviewed candidate protocol and the complete canonical
warm-up, retained R-hat/ESS, status, movement, energy, and cross-sign gates.

## Diagnostic role table

| Diagnostic | Role |
|---|---|
| Immutable hashes, target/transport identity, route policy, GPU memory policy | Run-validity/continuation veto |
| Finite samples, values, scores, log acceptance, and target status | Run-validity veto |
| Raw archive hashes and shapes | Run-validity/artifact veto |
| 4,000-draw physical plus sign R-hat and direct sign transitions | Primary diagnostic endpoint, not promotion |
| Cumulative and recent-window checkpoint R-hat | Explanatory trajectory |
| 2,000-draw prefix tie-out | Replay-identity evidence; mismatch downgrades cross-run interpretation only |
| Acceptance and `max_abs_log_accept_ratio` proxy | Explanatory only |
| Native divergence | Hard veto only if exposed and positive; unavailable is not zero |
| Runtime forecast and timeout | Resource/continuation veto only |

## Budget and attempt contract

The user's original incremental grant was `64,800 s`. The completed `r2`
HMC process used `23,280.603976539 s`. The failed launcher then used
`0.050816870003473014 s` before any repository TensorFlow import or GPU
initialization, leaving `41,519.345206590995 s` of aggregate campaign process
time. The fresh retry has an external bound of `36,000 s` and an internal work
bound of `35,820 s`, retaining `180 s` for closeout. At the maximum, aggregate
campaign process wall would be `59,280.654793409005 s`; actual aggregate GPU
wall would be at most `59,280.603976539 s`. Both remain below the prior
`61,200 s` HMC envelope and the `64,800 s` total grant. The unused predictive
reserve remains unused.

The measured `r2` `L=5` call took `13,188.602818158004 s` for 2,064
post-start HMC states per chain. Linear scaling to 4,064 gives
`25,968.256711721962 s`. The inherited `1.25` allowance gives
`32,460.320889652452 s`; adding the 180-second closeout remains below the
new internal cap. This is a feasibility forecast, not a performance law.

At most two launcher invocations and exactly one GPU-initializing attempt are
authorized. Launcher invocation 1 is already terminal and did not reach
TensorFlow; invocation 2 is the fresh-root retry. There is no retuning,
retraining, changed step size, second seed, second leapfrog length, further
retry, sequential HMC, or predictive phase.

## Default and assumption audit

| Choice | Provenance/status | Justification | Misleading failure mode | Early diagnostic |
|---|---|---|---|---|
| Seed 2, `L=5` | Exact failed `r2` pair; scoped baseline | Directly answers the user's question and had the lowest descriptive terminal maximum among completed q=20 verifications | Mistaken for a candidate ranking | Explicit no-ranking/non-promotion language |
| 4,000 draws | Convenience hypothesis: exactly twice the failed count | Smallest material extension that can reveal whether 2,000 was simply early while fitting the remaining grant | Still too short, or a passing cumulative statistic hides recent locking | Recent 1,000-window diagnostics and direct sign transitions |
| Checkpoints every 500 | Inherited canonical chunk extent; reviewed diagnostic default | Gives eight observations without changing the chain | Multiple looks mistaken for repeated experiments | Label all points descriptive and make no uncertainty/ranking claim |
| Recent window 1,000 | Canonical NeuTra warm-up window; transferred explanatory diagnostic | Distinguishes old-draw dilution from current behavior | Transfer mistaken for canonical readiness | Explicitly forbid readiness/admission interpretation |
| Same seed and 64 burn-in | Exact baseline replay hypothesis | Enables a potential 2,000-prefix tie-out | Larger static shape changes the random stream or compiler behavior | Prefix summary and acceptance tie-out; downgrade on mismatch |
| Frozen step `0.2460072308515237` and `L=5` | Measured `r2` tuning output; failed-pair identity | Changing them would answer a different question | Failure attributed to length when kernel is itself poor | Preserve identity and classify only horizon sufficiency |
| R-hat threshold `1.01` | Existing q=20 verification and canonical retained screen | Keeps the diagnostic endpoint comparable | Passing screen mistaken for convergence proof | Require explicit nonclaim and no posterior archive admission |
| Sign boundary at zero | Existing material region definition | This coordinate generated the known sign-separated modes | Other undiscovered modes ignored | State two-region scope and no exhaustive mode claim |
| Float64/XLA/TF32-off/GPU 1 | Exact prior execution mode | Minimizes environment drift | Device or source drift breaks tie-out | Prelaunch source hashes and trusted device receipt |
| Exact summary tie-out | Derived from the exact-extension claim | Without a prior raw archive, exact equality of every saved summary float is the strongest available replay check | Same summary could theoretically arise from different raw draws | State that raw-prefix identity remains unproved and report numeric residuals |

No choice is promoted as a general default. This plan tests one scoped
explanation for one failed pair.

## Pre-mortem

| How the run could mislead or fail | Smallest discriminating check | Disposition |
|---|---|---|
| Cumulative R-hat falls because old draws dominate while chains are currently locked | Trailing-1,000 R-hat and sign transitions at every eligible checkpoint | Report both; cumulative trend alone cannot support sufficiency |
| Pooled chains contain both signs but individual chains never cross | Per-chain sign counts and adjacent transitions | Diagnostic fail |
| A 4,000-draw invocation is not a deterministic extension of the old 2,000-draw invocation | Prefix R-hat/acceptance tie-out | Keep within-run trajectory, forbid exact-extension claim |
| Same random stream is called independent replication | Seed provenance in manifest and result | Explicitly classify as replay/extension diagnostic |
| R-hat passes while ESS remains poor | No admission or posterior use regardless of result | Future canonical run must evaluate ESS |
| Nonfinite or invalid target rows are averaged away | Full trace finite/status checks | Run-invalid terminal |
| Native divergence is unavailable and reported as zero | Preserve availability status | State unavailable is not zero |
| Timeout loses all evidence | Launch receipt before call and raw/terminal writes immediately after return | Preserve launch as resource evidence; no scientific conclusion |
| Result artifacts overwrite prior evidence | Fresh root reservation | Fail closed before TensorFlow import |

## Execution phases

### Phase 1: implementation and CPU-hidden checks

Create
`docs/benchmarks/run_ssl_lstm_q20_neutra_rhat_trajectory_diagnostic_2026_08_21.py`,
add the non-promotional route classification, and add focused tests for frozen
arguments, checkpoint scheduling, trajectory/sign diagnostics, budget
arithmetic, identity validation, raw archive receipts, and terminal closeout.
Use `CUDA_VISIBLE_DEVICES=-1` for tests.

### Phase 2: prelaunch audit

Before GPU execution, inspect the implemented runner against every skeptical
audit row, compile it, run focused tests, verify the complete `r2` inventory,
record plan/runner/route hashes, confirm the output root is absent, and inspect
trusted GPU availability. Any material mismatch keeps the GPU gate closed.

### Phase 3: single bounded diagnostic

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
timeout 36000s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_rhat_trajectory_diagnostic_2026_08_21.py \
  --device 1 \
  --time-cap-seconds 35820 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01
```

### Phase 4: terminal interpretation

Write a terminal result and reset memo. Include the exact command, environment,
GPU/memory policy, source/data identities, seed, wall time, archive hashes,
checkpoint table, decision table, inference-status table, engineering/sampler/
scientific ledgers, failure classification, candidate-versus-direction
distinction, uncertainty/nonclaims, and post-run red team.

## Skeptical audit checklist

| Required audit | Pass condition |
|---|---|
| Wrong baseline | Exact failed seed-2 `L=5` pair and complete immutable `r2` graph are bound |
| Proxy promotion | Neither checkpoint nor endpoint can admit HMC, posterior, or predictive work |
| Missing stop conditions | One GPU-initializing attempt, fresh retry root, forecast, timeout, closeout, identity, finite/status, and artifact stops are executable |
| Unfair comparison | No seed/kernel ranking; only one pair and one horizon question |
| Hidden assumptions | Draw count, windows, thresholds, seed, tolerance, budget, dtype, and hardware are audited above |
| Stale context | Current terminal `r2` result and post-run plan identity are distinguished |
| Environment mismatch | Trusted GPU, memory growth, float64, XLA, and TF32-off fail closed |
| Artifact cannot answer | Raw chains plus cumulative/recent R-hat and per-chain sign transitions are preserved |
| Pass while misleading | Recent-window and direct transitions accompany cumulative diagnostics; ESS/posterior nonclaim remains |
| Fail for wrong reason | Resource, harness, identity, artifact, numerical, candidate-horizon, and research-direction classes remain separate |

The skeptical audit and prelaunch audit records will be appended before their
respective gates open.

## Skeptical plan audit record

Codex audited the draft against the immutable `r2` result and tuning receipt,
the shared fixed-HMC mechanics, the rank-normalized R-hat implementation, the
route-policy ledger, and the repository research-governance requirements.

The first draft had two material interpretation defects:

1. It did not define what would count as evidence that more samples helped.
   The revised contract makes a complete 4,000-draw physical/sign R-hat plus
   direct-transition screen the only affirmative sufficiency result. A
   downward but still failing R-hat is reported only as descriptive movement
   and cannot support extrapolation.
2. It transferred the `1e-10` frozen transport-parity tolerance to stochastic
   replay identity without relevant provenance. The revised tie-out requires
   exact equality of all saved 2,000-draw summary floats and explicitly states
   that raw-prefix identity remains unproved because `r2` did not archive its
   samples.

The audit also checked the following:

| Risk | Audit disposition |
|---|---|
| Wrong baseline | Exact seed-2 `L=5` transport, step, initial bank, burn-in, seed, and failed `r2` summary are bound. |
| Proxy promoted to conclusion | Checkpoints and the endpoint remain diagnostic-only; no HMC, posterior, predictive, or default admission is possible. |
| Missing stop condition | One GPU-initializing attempt, hard process/internal caps, closeout, fresh root, input identity, finite/status, and artifact gates are explicit. |
| Unfair comparison or ranking | Only one pair is run; no seed, kernel, or method ranking is permitted. |
| Hidden defaults | Every material number and transfer is listed with provenance, failure mode, and early diagnostic. |
| Stale context | The terminal `r2` graph is the baseline; the current working plan is separately hash-bound at launch. |
| Environment mismatch | The exact prior numerical mode and trusted memory-growth GPU policy are required. |
| Artifact cannot answer the question | Raw chains, full traces, cumulative/recent R-hat, and per-chain sign transitions are required. |
| Pass while misleading | Recent-window behavior and direct transitions must accompany cumulative R-hat; ESS and posterior claims remain forbidden. |
| Fail for the wrong reason | Run validity, resource, horizon sufficiency, pair rejection, and research-direction conclusions remain separate. |

Plan-audit verdict:
`PASS_FOR_IMPLEMENTATION_PENDING_ACTUAL_RUNNER_AND_PRELAUNCH_AUDIT`.

## Harness prelaunch audit

Codex inspected the implemented runner and focused tests against every plan
gate. No scientific or default claim is opened by this audit.

| Check | Evidence and disposition |
|---|---|
| Immutable baseline | The runner validates the complete 30-entry `r1` training graph, complete 15-entry `r2` HMC graph, five top-level `r2`/runner hashes, target and adapter signatures, frozen seed-2 transport state, initial bank, exact transformed-gradient identity, and fixed identity-`z` mass matrices. |
| Same failed pair | Step `0.2460072308515237`, `L=5`, burn-in 64, seed `(20260820,52000)`, target scope, chain bank, float64, XLA, and status tracing are fail-closed. |
| Source closure | Shared fixed-HMC mechanics SHA-256 is `a04aaea2824bb972881ce9af023d89ba473a6343cf1d437bca2812a90f564616`; R-hat source SHA-256 is `b7544346f4beb63946c7482a8a9a2341f4d5cabe72a2b29fab7c4f4ab8408dd7`. |
| Single attempt/non-promotion | AST/source audit finds exactly one shared fixed-HMC call and no local TFP sampler, canonical sequential call, HMC admission token, or predictive path. Result schemas hard-code no posterior, predictive, or candidate reinstatement. |
| Checkpoint evidence | Tests exercise cumulative and recent-window R-hat, direct per-chain sign locking/crossing, trajectory deltas, exact summary replay tie-out, and raw TensorFlow archive roundtrip. |
| Artifact integrity | Fresh root is absent. Writes are atomic and refuse reuse. Raw sample/trace tensors are immediately parsed back and checked for SHA-256, byte count, dtype, and shape before terminal interpretation. |
| Route policy | Repository audit passes with 42 discovered and 42 classified routes. This runner is `smoke_mechanics_or_reference`; the active claim-bearing q=20 route is unchanged. Route-ledger SHA-256 is `f1d4c9268774ed302e386ba5ce260a60daaeb17112d04c5f12363a129a14f0f5`. |
| Budget | The allowance-scaled forecast plus closeout is below `35,820 s`; a `36,000 s` external cap leaves aggregate wall below the prior HMC envelope and 18-hour grant. The monolithic XLA call cannot be interrupted gracefully by the internal timer, so the internal value is an admission check; the shell timeout is the actual hard stop. |
| Environment | GPU environment and memory-growth variables are set before repository/TensorFlow imports. Trusted inspection found two RTX 4080 SUPER GPUs. GPU 1 had 598 MiB resident and no TensorFlow/CUDA compute process; only the existing desktop/remote-session services were listed. |
| Validation | Compilation and `git diff --check` pass. CPU-hidden focused suite: `57 passed`; 188 TensorFlow Probability/Gast deprecation warnings are non-failures. Full `r2` inventory rehash: 15 entries, zero mismatches. |

Prelaunch identities before this status amendment: runner SHA-256
`1caa14095d13dd592ef1bf184b2237a71e8fe2145ed79081473c8267de121292`,
route-ledger SHA-256
`f1d4c9268774ed302e386ba5ce260a60daaeb17112d04c5f12363a129a14f0f5`,
and focused-test SHA-256
`acc41cbcc3cc327cdaa0fa81e9a7eb18bdf4a3c4193960163032b20c2cc501ab`.
The runner records the final post-amendment plan hash at launch.

Historical prelaunch verdict:
`PASS_FOR_ONE_BOUNDED_DIAGNOSTIC_PROCESS`. That launch was consumed by the
terminal pre-TensorFlow failure below. Its old no-retry sentence is superseded
only by the proportionate bounded-repair amendment; the scientific contract,
GPU-attempt limit, and claim boundary are unchanged.

## Preserved launcher failure and bounded repair amendment

The exact first Phase 3 invocation reserved its fresh root and failed after
`0.050816870003473014 s` with `ModuleNotFoundError: No module named
'bayesfilter'`. The traceback ends in `_route_policy_audit()` before the
runner's TensorFlow import, GPU memory-policy helper, target construction,
or HMC call. Its partial-artifact map is empty. This is an implementation
harness failure and supplies no sample-length or sampler evidence.

| Preserved artifact | SHA-256 |
|---|---|
| Failed result | `f3a0112caf08738c8fcc83af0b28b4ac41504295edb076ace3018cf89e7befa8` |
| Failed manifest | `198339c610841f43341e31be5085fd4dd52ed8e03648adc446ab2b063294019c` |
| Failed two-entry inventory | `e607f09c9d9f61b4b0b236cd049ca6b98220feef2403e154b5afcd15b5b98f0e` |
| Launch-time plan | `28dc43a858e0d00dc8af9b890c98339d70e34092249dcd72b2db576bd019ef8f` |
| Failed runner | `1caa14095d13dd592ef1bf184b2237a71e8fe2145ed79081473c8267de121292` |

The failed abort artifact added its process elapsed time to a field named
`aggregate_gpu_wall_seconds`. That label is wrong relative to actual GPU-use
accounting because the runner did not reach TensorFlow or GPU initialization.
The repair records the `0.050816870003473014 s` only as campaign process wall;
it does not count it as GPU wall.

The defect was stale import ordering: repository root was inserted into
`sys.path` only after `_route_policy_audit()` attempted the first
`bayesfilter` import. The bounded repair moves that insertion immediately
after `ROOT` is derived, before any repository import. It does not change the
target, data, transport, initial states, HMC kernel, seed, draw count,
diagnostics, hardware class, criteria, vetoes, or total campaign budget. The
failed root is immutable; the retry is forced to the absent `retry-01` root.

Repair-audit verdict:
`PASS_FOR_REVALIDATION_BEFORE_ONE_FRESH_ROOT_GPU_ATTEMPT`. The GPU gate remains
closed until compilation, focused tests, route audit, source/hash audit,
failed-artifact rehash, retry-root absence, and trusted GPU inspection pass.

## Repair revalidation record

All repair gates passed after the amendment:

| Gate | Evidence and disposition |
|---|---|
| Focused validation | CPU-hidden compilation passed; the trajectory, global-mixing, and shared fixed-transport tuning suites report `57 passed`. The 188 TensorFlow Probability/Gast deprecation warnings are non-failures. |
| Route enforcement | Repository route audit again reports 42 discovered and 42 classified routes with no errors. The diagnostic remains `smoke_mechanics_or_reference`; the active q=20 claim-bearing route remains unchanged. |
| Input integrity | `_validated_inputs()` revalidated the complete immutable `r1` and `r2` graphs and the three preserved launcher-failure artifacts. The failed two-entry inventory has zero mismatches. |
| Attempt/budget enforcement | Executable budget reports two launcher invocations, one GPU-initializing attempt, retry index 1, and all three resource-fit booleans true. |
| Output isolation | The failed root remains present and unchanged; `r3-rhat-trajectory-retry-01` is absent. The runner rejects every other output path. |
| Source hygiene | Compilation and `git diff --check` pass. Direct untracked-file whitespace checks emit no findings. |
| Trusted GPU state | Both RTX 4080 SUPER devices were visible. GPU 1 reported 581 MiB used and 0% utilization; the only listed process was `gnome-remote-desktop-daemon` at 251 MiB. No TensorFlow/CUDA compute owner was present. |

Post-repair identities are runner SHA-256
`b4d83c0c49a215e4d031c1cd60016ca363b20a69a13dec20bddbb6b26c746c7a`,
focused-test SHA-256
`0a217b8a3af961d3ba98472715727ca41f9e44922567938318eee75edcc8c770`,
and route-ledger SHA-256
`f1d4c9268774ed302e386ba5ce260a60daaeb17112d04c5f12363a129a14f0f5`.
The launch receipt binds the final plan SHA-256 after this status amendment.

Final retry verdict: `PASS_FOR_ONE_FRESH_ROOT_GPU_DIAGNOSTIC_ATTEMPT`. This
opens only the exact amended Phase 3 command. A timeout, numerical failure,
harness failure, or completed result is terminal for this diagnostic; no
further retry, alternate kernel, sequential HMC, posterior use, or predictive
work is authorized.

## Terminal execution record

The fresh-root retry completed normally on 2026-08-21. It returned
`DIAGNOSTIC_SCREEN_FAILED_AT_4000` and
`DOUBLING_TO_4000_INSUFFICIENT_FOR_DECLARED_SCREEN` with no run-validity hard
vetoes.

The 2,000-draw prefix exactly reproduced every saved prior R-hat and acceptance
summary float. Cumulative observation-weight R-hat then fell from
`1.0875996310350042` at 2,000 to `1.0489500982500948` at 4,000, while the
endpoint sign-indicator R-hat was `1.0761213959625822`. Both failed `<=1.01`.
The cumulative trajectory had five decreases and two increases, and the final
trailing-1,000 observation-weight R-hat was `1.0595875339465644`; therefore the
result supports descriptive improvement but not monotone convergence or a
pure sample-shortage explanation.

Every chain visited both signs and transitioned. Cumulative transition counts
were `[33,49,29,46]`, and final trailing-window counts were `[13,19,11,16]`.
This rejects literal sign locking in the observed chain but does not repair the
failed R-hat screen.

The HMC call used `25797.081125429017 s`; retry process wall was
`25813.99267909798 s`; aggregate campaign process wall was
`49094.647472506986 s`. The 23-entry terminal inventory has zero rehash
mismatches, and all 16 raw TensorFlow receipts independently passed hash,
byte-count, dtype, and shape verification.

Post-run closeout repeated compilation and the CPU-hidden focused suite:
`57 passed`. Exact structured terminal assertions, `git diff --check`, and
ASCII/whitespace scans also passed.

Terminal records:

- `docs/plans/bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-result-2026-08-21.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-reset-memo-2026-08-21.md`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/checkpoint-diagnostics.json`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/raw-archive.json`

No candidate, posterior, predictive, scientific, or default admission follows.
The next justified direction is a new target-specific geometry/kernel repair
plan, not another unreviewed extension of this fixed chain.
