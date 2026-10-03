# Phase 9A Full-Replay R0 Audit Result

Date: 2026-09-02  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `PASS_R0_REFRESHED_R1_PROFILE_REPAIR_COMPLETE`

## Scope

R0 audited the immutable chart-1/beta-0 attempts before a new GPU launch. It
checked target and backend identities, embedded manifest hashes, seed
namespaces, per-call records, reusable-runner evidence, and the cost model.
No historical file was rewritten or used as a warm start.

## Audited evidence

| Attempt | Profile | Status | Calls | Per-call sum (s) | Wall time (s) | Embedded manifest hash | Interpretation |
|---|---|---|---:|---:|---:|---|---|
| `attempt-03` | v3 minimal | pass partial | 9 | 199.862577 | 344.628621 | `761ed1c6371e8fad9d41843b5a0d0a9bc28681f8b2081b04dc5d858be04458ab` | metadata-defect mechanics probe; not fresh replay evidence |
| `attempt-04` | v3 minimal | pass partial | 9 | 216.346595 | 370.885823 | `114635c897e794fdb581214492fdcea1894ce3202fb6da9a57156234e835978d` | deterministic provenance replay; reused v3 seeds |
| `attempt-05` | v4 fresh | pass partial | 13 | 336.644644 | 494.568909 | `88bab1482475d59d1c610f1dda4391d8b1424f15b69536895995950e327dfea7` | fresh localized mechanics evidence only |

The attempt-05 embedded hash was recomputed from the manifest with its
`manifest_hash` field removed and matched exactly. The current file checksum
is `6ec0d0b155bc039bb4c79170c9f938e0dbff60fe17b7f5e72f8ba8b1896ea8ee`; this
ordinary file checksum is recorded separately from the manifest's canonical
content hash.

The historical manifests predate the new timing fields, so they cannot be
retroactively interpreted as compile-versus-steady telemetry. Their call
durations remain valid descriptive inputs. The repaired runner now records
`runner_created_on_call`, `timing_role`, static configuration, runner-pool
trace counts, and the underlying `sample_chain_call_s` for every new call.

## Seed and output audit

The historical namespaces are `20260901/730xx--775xx` (attempts 03--04) and
`20260901/770xx--775xx` (attempt 05). The new canary profile owns
`20260902/780xx--785xx`; the full profile owns the disjoint
`20260902/790xx--795xx` namespace. The new output root is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/phase9a-full-replay/`.
No path in that root existed when R0 closed, and the launcher rejects a
collision before TensorFlow import.

## Cost model

Attempt-05's 13-call sum is `336.644644` seconds, or `25.895742` seconds per
call. A six-scope replay with the same measured-grid schedules is expected to
have more calls plus two chart builds, reliability, and the mechanics
transition. The profile-bound canary cap is `1800` seconds and the profile-
bound full cap is `7800` seconds. These are hard ceilings, not values a caller
can override. The canary is the cost discriminator; its selection and held-out
draws are not reused by the full replay.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close R0 | Identity/hash/seed audit and executable profile contract | No R0 veto | Historical records lack compile/steady split | Run focused R1 checks and the fresh canary | No performance guarantee |
| Accept instrumentation repair | Existing numerical route unchanged; new telemetry is additive | Python/shell/tests pass | Runtime behavior on GPU can still differ by graph cache | R2 canary under 1800 s | No numerical equivalence beyond existing mechanics tests |
| Authorize R2 | User explicitly requested continued execution; target, data, hardware, profiles, and caps unchanged | Platform GPU trust still required at launch | Canary may expose a second resource failure | Launch canary, then mandatory closeout | No Phase 9B or posterior claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the R0 audit and focused checks |
| Statistically supported ranking | None; no new stochastic comparison was made |
| Descriptive-only differences | Historical wall/call times and seed sensitivity |
| Default-readiness | Not assessed and not promoted |
| Next evidence needed | Fresh canary telemetry, then a canary closeout and refreshed full-replay ledger |

## Post-run red team

The main alternative explanation is that the new timing fields still understate
asynchronous device work or that separate chart construction causes additional
compilation not visible in a single call. The canary's first-run/steady-run
records and GPU allocator telemetry are the direct discriminator. A timeout,
nonfinite artifact, or missing call record will be classified as a resource or
implementation failure and will trigger the declared closeout; it will not be
silently treated as a tuning result.

## Closeout receipt

- R0 command: read-only manifest/hash/seed audit plus focused profile checks.
- Focused regression: `12 passed` in `tests/test_ssl_lstm_q20_phase9a_repair_runner.py`.
- Python compilation and shell syntax: passed.
- Runner source hash after repair: `ceb162f4af18f30823eb5d00c2616f3cd871117a632b4db72e631638b657fe1e`.
- Launcher source hash after repair: `7a2fe7404d39affccd8207d09e820657170ea433cee919faede5d497238db70e`.
- Refreshed next command: `BAYESFILTER_PHASE9A_ATTEMPT_ID=attempt-01 bash scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh --profile phase9a_full_replay_canary_v1 --scope-start 3 --scope-limit 1`.
- Real blocker: none. R2 is permitted under the unchanged M3 contract.
