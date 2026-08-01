# Complete High-Dimensional Leaderboard Phase 0 Authority Repair Result

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `PASS_PHASE0_OWNER_AUTHORITY_REPAIR`

## Outcome

The owner-authorized SIR observation identity and six exact-row extension
classifications are now bound by an additive authority overlay and a regenerated,
independently audited Phase 0 machine freeze. The old Phase 0 freeze remains a
historical reviewed record; its hash is preserved in both the overlay and the
new machine artifact.

The repaired fixed-SIR target identity is:

`fixed_bayesfilter_sir_observations_from_dataset_seed_81103_not_author_matlab_rng1_reproduction`

This states directly that the authoritative main-row observations are the
current deterministic BayesFilter `_sir_dataset(81103)` bytes and are not an
exact author MATLAB `rng(1)` reproduction.

## Supersession Ledger

| Item | SHA-256 / status |
| --- | --- |
| Original Phase 0 freeze | `4115ef55114ffd73255363f0c62c4a19dd85d7ca3241d002c48409cb9004f878` (`historical_superseded_authority_identity`) |
| Owner-authority amendment | `171b8d42ec5f31869181003636a223b4e3063e38c03a66e1df7f77782d81ce90` |
| Repaired Phase 0 freeze | `4c7d602e16f4856457873347d254b2d48b78ecd3e034f30992f3b45aa72d356b` |
| Failed preapproval P1-A receipt | `3299d3b797aa41b028fe77ec4d3aabb639fa176da73767fe3b4905a2d614ff67` (`preserved_failure_not_reinterpreted`) |
| Frozen pre-edit harness | `2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58` (`unchanged`) |

## Changes

- Phase 0 schema advanced from `v1` to `v2`.
- The machine artifact now binds the owner-authority amendment path/hash and
  original Phase 0 freeze hash.
- The SIR target-generation token now declares seed `81103` and explicitly
  denies author-MATLAB reproduction.
- The independent auditor encodes the same authority as its own constant and
  rejects the old token or a missing supersession ledger.
- Focused tests cover both rejection cases.

No row, parameter coordinate, time horizon, LEDH execution seed, algorithm id,
sidecar boundary, target scalar, FD policy, or numerical cell status changed.

## Checks

| Check | Result |
| --- | --- |
| Phase 0 generator write | `PHASE0_FREEZE_WRITTEN` |
| Independent Phase 0 audit | `PHASE0_INDEPENDENT_AUDIT_PASS` |
| Combined Phase 0/P1-A focused tests | `19 passed` |
| Python compilation | Pass |
| Scoped diff check | Pass |
| Harness SHA-256 | Unchanged at `2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58` |

The TensorFlow tests were deliberate CPU-hidden engineering checks with
`CUDA_VISIBLE_DEVICES=-1`. TensorFlow emitted CUDA plugin-registration noise,
but GPU devices were intentionally hidden and no GPU evidence is claimed.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Pass the Phase 0 authority repair and allow P1-A/P1-B to be reissued |
| Primary criterion | Owner decision is hash-bound and independently enforced in the regenerated freeze |
| Veto status | No target contradiction remains; original/historical identities remain visible |
| Main uncertainty | Numerical evaluator and later Zhao-Cui route correctness remain untested |
| Next justified action | Refresh Phase 1 entry bindings, rerun P1-A, close P1-B, then enter P1-C only if both receipts pass |
| Not concluded | No cell admission, source faithfulness, GPU readiness, ranking, HMC/posterior correctness, or scientific validity |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command | `python scripts/build_complete_highdim_leaderboard_phase0_freeze.py --output docs/plans/artifacts/complete-highdim-leaderboard/phase0-boundary-freeze-2026-07-11.json` |
| Independent command | `python scripts/audit_complete_highdim_leaderboard_phase0_freeze.py --input docs/plans/artifacts/complete-highdim-leaderboard/phase0-boundary-freeze-2026-07-11.json` |
| Environment | Current BayesFilter shell; metadata generator/auditor do not import TensorFlow |
| CPU/GPU | Metadata checks only; focused framework tests used `CUDA_VISIBLE_DEVICES=-1` |
| Data version | Repaired Phase 0 JSON and its SHA-bound inputs |
| Random seeds | No randomness executed; target/execution seed metadata only |
| Plan authority | Owner-authority amendment plus original Phase 0 subplan |
| Result | This file |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for authority metadata only |
| Statistically supported ranking | None |
| Descriptive-only differences | None; no stochastic run |
| Default-readiness | Not evaluated |
| Next evidence needed | Superseding P1-A canonical-target receipt and P1-B source-availability receipt |

## Post-Run Red Team

- Strongest alternative explanation: the authority records could be consistent
  while the canonical bytes differ from `_sir_dataset(81103)`. Control: P1-A
  independently reconstructs and byte-checks those observations.
- Result that overturns this pass: a stale amendment hash, old SIR identity,
  missing original-freeze hash, or failed independent reconstruction.
- Weakest evidence: owner approval permits the extension classifications but
  is not evidence that any extension is correct.

