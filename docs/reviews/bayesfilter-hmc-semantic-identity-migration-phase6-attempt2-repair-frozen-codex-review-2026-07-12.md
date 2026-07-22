# Phase 6 Attempt-2 Repair Frozen Codex Review

Date: 2026-07-12

Review type: fresh bounded independent Codex read-only implementation review.

Claude was not retried because the managed external-disclosure rejection
remained binding. Codex remained supervisor and executor; the reviewer did not
edit files, create authority, launch workers, or run HMC.

## Scope

- `bayesfilter/inference/hmc_smoke_authority.py`
- `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py`
- `scripts/build_hmc_phase6_smoke_authority_proposal.py`
- `scripts/build_hmc_phase6_smoke_authority.py`
- `scripts/run_hmc_phase6_typed_identity_smoke.py`
- `tests/test_hmc_smoke_authority.py`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md`

## Review Loop

The first review returned `VERDICT: REVISE` because attempt-1 integrity was
procedural rather than mechanically enforced. The repair added an exact
13-file immutable ledger and a retained evidence session.

The second review returned `VERDICT: REVISE` because same-size overwrite,
concurrent mode change, hard-link, claim-boundary, semantic, and complete-ledger
gaps remained. The repair added no-follow descriptor pinning, shared locks,
owner/group and one-link checks, capture-time complete stat signatures, double
reads, semantic validation, preclaim/postclaim checks, and adversarial tests.

The third review found one remaining stage-boundary overclaim: drift detected
after the initial progress write can preserve those already-durable bytes. The
subplan was corrected to distinguish reservation-time drift from later drift,
and a regression now proves that post-progress detection preserves only the
prior progress record, creates no worker, performs no later write, and does not
seal an infrastructure terminal.

## Frozen Snapshot

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/inference/hmc_smoke_authority.py` | `71b0f01270ddf6e743cdd4040f815845a31a5abe6af1a767ada5fda11dabb670` |
| `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` | `109e67f12476093b276a6954252f8c124edb7fe743d9e56036b3ed046790fbb3` |
| `scripts/build_hmc_phase6_smoke_authority_proposal.py` | `0004d73093d581bed38877e44598c2264a6144481696af41daf335f0ec15397f` |
| `scripts/build_hmc_phase6_smoke_authority.py` | `5ebfde054ef2c0a6ffe2a034403b10c58c63341f1f2580c45328d79102c113d2` |
| `scripts/run_hmc_phase6_typed_identity_smoke.py` | `6730d7ec33e59e2ddba19f51bb2e335cee417bf63716dbdff5cc51974b0356bb` |
| `tests/test_hmc_smoke_authority.py` | `0b94071c9aa7ae1c4608d08978a0c3e3b8227adc96db80b4761b8f96cc543a9b` |
| `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md` | `0fdb4378c5cd50b3c790609f49b7a5334bf1fbec83900b487ec7d8cc295856b3` |

## Verification Evidence

| Check | Result |
| --- | --- |
| Final stage-specific drift matrix | `3 passed, 2 warnings` |
| Complete authority module | `106 passed, 2 warnings in 312.64s` |
| Combined eight-module migration gate | `251 passed, 2 warnings in 340.78s` |
| Python compilation | Passed |
| Scoped whitespace check | Passed |
| Immutable attempt-1 evidence | `INTEGRITY_OK 13` |
| V3/attempt-2 paths before materialization | All absent |

The warnings were the existing TensorFlow Probability `distutils.version`
deprecations. All checks were deliberately CPU-hidden; no worker, HMC
transition, or XLA compile transition ran.

## Findings

No blocking finding remained on the final frozen snapshot.

- Capture-time stat signatures are retained rather than silently recaptured.
- Final reservation verification returns the partial session for deterministic
  descriptor cleanup.
- Typed attempt-1 drift bypasses controller failure writing and launcher
  infrastructure sealing while retaining bounded worker teardown.
- Reservation-time drift can leave only the permanent claim and empty
  reservations; later detection preserves only bytes completed before detection
  and permits no further write.
- Attempt-1 artifacts remain immutable and V2 approval cannot authorize attempt
  2.

## Verdict

`VERDICT: AGREE`

This verdict authorized only V3 proposal materialization and review. It did not
authorize attempt-2 authority creation, a worker, an HMC transition, serious
Phase 7, Phase 8, or NeuTra.
