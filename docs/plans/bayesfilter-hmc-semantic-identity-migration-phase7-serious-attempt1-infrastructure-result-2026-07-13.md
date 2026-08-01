# Phase 7 Serious Attempt-1 Infrastructure Result

Date: 2026-07-13

Status: `TERMINAL_INFRASTRUCTURE_FAILURE_AUTHORITY_CONSUMED`

Decision: `BLOCK_PHASE7_SERIOUS_LAUNCHER_INFRASTRUCTURE`.

## Direct Verdict

The exact approved Phase 7 serious attempt consumed its one-use authority and
permanent claim, then failed during secure output reservation before worker
creation or any HMC transition. This is an implementation failure in the
historical-result retirement verifier. It is not a convergence failure, target
failure, XLA failure, HMC failure, or evidence against the scientific
direction.

The approval and claim cannot be reused. No retry is authorized.

## Exact Authority

The owner supplied:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS bound to Phase 7 authority proposal manifest sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330.`

The resulting authority is
`sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d`.
The permanent launch claim is
`sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c`.

## Terminal Evidence

| Artifact | Embedded hash | File SHA-256 | Bytes | Mode |
| --- | --- | --- | ---: | --- |
| Authority | `sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d` | `ee58976627310dd8eb3c491f7b810e1240bccb1a6620bc0d5316c6e6d1949b52` | 1638 | `0600` |
| Permanent claim | `sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c` | `9dd3a18b045efd863e0fc5c2a60ffca7b1daa104b540b2a947d320b2312b423e` | 2053 | `0400` |
| Infrastructure failure | `sha256:f571a196112af7b0fa04b63bda24fe30e4ece7cf8a2fe11dc63429bce7baf173` | `06eb4261f452754f658f1acf07c0d47190875286985a92393173122678dfa4f6` | 1227 | `0400` |
| Infrastructure manifest | `sha256:7c8a1fc2a886ca1ae9930e1099c3e0f36830875fac47451183b57ab04f7e440e` | `2ce6fce8d09d4dc6723988f47a8e0ba17e555646065530abdfc57657ee2213d8` | 3500 | `0400` |
| Reserved ordinary output manifest | N/A, empty reservation | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Reserved public result | N/A, empty replacement | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Reserved log | N/A, empty reservation | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |

The terminal failure stage is
`secure_output_reservation:historical_result_replacement` with reason
`infrastructure_error:RuntimeError`. The infrastructure manifest backward-binds
the exact archive manifest, authority, claim, infrastructure failure, empty
public result, and empty log. No progress or private-sample reference exists.

The immutable historical archive remains mode `0400`, 2378 bytes, file SHA-256
`3b34cf56062950a9ba835f6b4839421510a8921545a0edd36203f39eac4ec0d6`.
The configured live result pathname now intentionally names the empty reserved
attempt-1 result inode bound by the terminal infrastructure manifest. It must
not be restored, deleted, rewritten, or reused.

## Root Cause

Production opens the historical live result twice before replacement:

1. `SeriousInheritedEvidenceSession` pins one read-only descriptor; and
2. `PinnedSmokeOutputDirectories` opens a second read-only descriptor to the
   same inode for the output capability.

After atomic replacement,
`SeriousInheritedEvidenceSession.retire_replaced_path()` incorrectly requires
the two numeric file-descriptor values to be equal:

```python
if held_fd != original_fd or not self._retired_signature_matches(...):
    raise RuntimeError(...)
```

Different descriptors to the same inode are expected and safe. The verifier
should compare their device/inode identity, constrained retired metadata, and
bytes, not their process-local integer values.

The descriptor-level unit fixture passed the same descriptor as both roles, so
it failed to represent production and masked the defect. The repair must add a
two-distinct-descriptor regression.

## Research Question Guardian

The attempt did not reach the fixed transition or sampler. Therefore:

- the harness/output reservation failed;
- the target, data, transition math, HMC implementation, and XLA path were not
  evaluated;
- the current candidate was not rejected;
- no diagnostic cap, convergence threshold, or numerical veto fired; and
- the planned repair addresses exactly the observed implementation defect.

The justified continuation is a separately reviewed attempt-2 mechanical
repair and new proposal. It is not an authorized runtime retry.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Block attempt 1 and preserve all evidence | Worker/HMC entry was not reached | Post-claim output-reservation continuation veto fired | Whether the fixed serious sampler passes remains entirely unknown | Versioned attempt-2 no-runtime repair, tests, proposal, review, then new human approval | No convergence, recovery, ranking, production, GPU, Phase 8, NeuTra, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Infrastructure veto only; sampler/numerical screens not run |
| Statistically supported ranking | None |
| Descriptive-only differences | None; no samples or diagnostics exist |
| Default readiness | Not evaluated |
| Next evidence needed | Correct two-descriptor/output-version repair followed by a separately approved unchanged serious run |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command | Exact proposal-bound serious launcher command |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11`; deliberate `CUDA_VISIBLE_DEVICES=-1`; fixed thread environment |
| GPU status | GPU deliberately hidden; no GPU evidence |
| Worker/HMC status | No worker PID, XLA compile transition, HMC transition, burn-in, retained draw, or diagnostic |
| Seeds | Fixed by proposal but unused |
| Wall time | Approximately 9 seconds through authority/claim/reservation failure |
| Output paths | Exact attempt-1 paths listed above |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md` |
| Result | This file |

## Post-Run Red Team

Strongest alternative explanation: another invariant inside the same
retirement method could also fail after the descriptor-number defect is fixed.
The attempt-2 regression must exercise two independently opened descriptors,
the full archive/parent/replacement checks, and a versioned result path without
using runtime.

What would overturn this classification: evidence of a worker process, HMC
transition, corrupted archive, or invalid terminal manifest. Current process,
artifact, and archive checks show none.

Weakest evidence: the terminal failure records only `RuntimeError`, not the
message. The exact executed code path and the guaranteed distinct production
descriptors identify the first failing condition; a focused regression must
reproduce it before repair.

## Stop

Do not rerun attempt 1, delete its claim, restore its live result pathname,
overwrite any reservation, or reuse its approval. Phase 8 and NeuTra remain
unauthorized.
