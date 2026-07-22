# Phase 2 Score-Reference Timeout Repair Result

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `PASS_SHARED_HARNESS_REPAIR_RETRY_REQUIRES_REBOUND_AUTHORITY`

## Failure Classification

The first exact fixed-SIR `T=1`, seed-`81120` score command completed under
trusted GPU/XLA/TF32 and passed the raw score validator for the pre-repair
source. Its paired FD command failed before numerical evaluation with:

`shard run manifest command_timeout_seconds mismatch`

The claimed target was a score shard validated under its own frozen score
command. The quantity actually checked was the score shard against the FD
command's stage-specific timeout. Those are different. The validation was
wrong relative to the intended score-reference contract.

This is a shared harness implementation-validation failure. It is not an FD
numerical failure, target/data/math invalidity, memory failure, GPU failure, or
evidence against fixed-SIR or the compact-score direction.

## Preserved Evidence

| Artifact | SHA-256 | Classification |
| --- | --- | --- |
| Pre-repair exact-command manifest | `fa77f32fbf50333c0ae5e0e1a0c26e9772f9b568d9e0a017790e3d86c3d27433` | Superseded command/source identity |
| Pre-repair fixed-SIR `T=1` score | `bd6475fe2b457557b175fb03f23dab8ee940787c703e0cdff793e5c9d1951771` | Valid trusted score only for pre-repair source; not reusable as current evidence |
| Pre-repair fixed-SIR `T=1` FD failure | `240b086fad509ce6a7e024cbca936eaa7675b60cbce5dc12ee208dca5c39f0ab` | Preserved terminal implementation failure |
| Trusted GPU preflight | `8daebd9efde58a807c699daab45c0cdd1a0c2beffa86549cc2e06f23f3dc17e2` | Passed device/XLA/TF32 continuation gate |

No artifact was overwritten.

## Root Cause And Fix

`_load_score_reference` called `_validate_raw_score_shard` with the FD
invocation's arguments. Score and FD commands intentionally have different
stage, output, reference, and timeout fields. The validator therefore rejected
the valid score timeout `900` because the FD timeout was `1200`.

The repair parses and validates the score shard using its own recorded argv,
then separately compares the true shared score/FD fields: row, seed, `T`, `N`,
source artifact, memory budget, device policy, dtype/TF32, configuration, and
route. Stage-specific timeout/output/reference fields are no longer treated as
shared computation identity.

## Repair Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Repaired harness | `2849903065533976efb7701b24fe8d56720978087b78e53751cdd9a42b5e0cd9` |
| Repaired command builder | `79c37f3b862e1adb32e968a26359f88b267d470c5ea176d20518d5f35d57bd87` |
| Repaired harness tests | `b38b1bbca29b673a246549a8e651a5d28606cb3c195a1e281c3a1f226e6c9ad8` |
| Repair1 exact-command manifest | `bc8a8a9aa67b64b72ff5e9431bf8ea993bc8a97acbb62e52af0cef421bf4229f` |
| Repair1 command-set SHA-256 | `8bf4f4fd4f0890b04a0199254bc188d21af1ade15f276d1d22e770ad7c888597` |

The `repair1` manifest uses entirely new `phase2-ledh-repair1` and
`phase3-ledh-repair1` JSON/Markdown/log paths and explicitly binds the old
manifest as superseded.

## Checks

| Check | Result |
| --- | --- |
| Focused score-reference/shard validators | `61 passed` |
| Full dedicated harness | `132 passed` |
| Six row contracts plus cross-model provenance | `146 passed` |
| Repair1 command-builder `--check` | Pass |
| Python compilation | Pass |
| `git diff --check` | Pass |

All checks were deliberate CPU-hidden engineering checks and are not GPU
numeric evidence.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Accept the bounded shared-harness repair and permit one new-path retry after review and authority rebinding |
| Primary criterion | Stage-specific timeout regression passes and all broader contract suites remain clean |
| Veto diagnostic status | No target, data, math, GPU, memory, or FD numerical veto fired; old computation identity is superseded |
| Main uncertainty | Repaired FD execution has not yet run on GPU |
| Next justified action | Review this repair, bind a repair1 Phase 2 authority receipt, and restart at the smallest fixed-SIR `T=1` score/FD pair using new paths |
| What is not concluded | No row eligibility, cell admission, ranking, HMC/posterior correctness, source-faithfulness, or release claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Shared harness defect repaired locally; numerical screen not yet rerun |
| Statistically supported ranking | None |
| Descriptive-only differences | Pre-repair score value/runtime/memory are historical diagnostics only |
| Default-readiness | Not evaluated |
| Next evidence needed | New-path repaired score/FD GPU artifacts |

## Post-Run Red Team

- Strongest alternative explanation: parsing the score's own argv could hide a
  real score/FD mismatch. Control: the repaired code explicitly compares every
  true shared computation/device/configuration field and has a negative
  shared-field mismatch test.
- Result that overturns the repair: a forged score reference with a changed
  shared field passes, or the repaired GPU FD still fails before numerical
  evaluation for another shared validator defect.
- Weakest evidence: the fix is verified locally but not yet by a repaired GPU
  retry.
