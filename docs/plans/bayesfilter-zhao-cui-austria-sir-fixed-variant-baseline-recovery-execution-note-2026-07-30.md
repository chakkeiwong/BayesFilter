# Zhao-Cui Austria SIR Fixed-Variant Baseline Recovery Execution Note

Date: 2026-07-30

Status: `LANE_A_COMPLETE_BLOCK_EXACT_P88_RECOVERY_EXHAUSTED`

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-plan-2026-07-30.md`.

## Research Intent

Determine whether the exact missing historical P88 fixed-TTSIRT identity is
present in local Git objects, snapshots, worktrees, ignored artifacts, logs, or
checkpoints. This is an identity-recovery audit, not an attempt to regenerate a
plausible transport.

## Evidence Contract

| Field | Contract |
|---|---|
| Exact baseline | P88 artifact SHA-256 `ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e` and density branch `265f9a06877e9babbba22dde187487fde4b50d08d8ecb98cd26b16467b6c1f10`. |
| Primary criterion | A structured immutable payload or chain binds the full frame, CDF configuration, frozen references, retained identity, input identity, and source closure to exact P88. |
| Hard veto | Any missing or guessed group; regenerated fields without an independent expected hash; scalar-only agreement. |
| Explanatory only | Filename, prose, target-id, normalizer, frame-log-determinant, and source-code matches. |
| Artifact | `docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/recovery-attempt-01/inventory.json`. |
| Nonclaims | No complete filter, value, score, T2/T20, GPU, HMC, correctness, or production claim follows from inventory matches. |

## Skeptical Execution Audit

| Risk | Disposition |
|---|---|
| Wrong baseline | Exact P88 SHA and branch are the only historical baseline. |
| Proxy promoted | Text and scalar matches are leads only; all identity groups are required. |
| Stale code | Current code cannot issue historical identity and is not used to regenerate fields. |
| Incomplete search | Scan workspace files plus every reachable and unreachable Git blob, including registered worktrees and the preserved source snapshot through the workspace scan. |
| Unbounded scan | Skip individual files/blobs over 64 MiB; record every skip. P88 manifests and expected compact identity artifacts are far below this cap. |
| Environment mismatch | Inventory uses Python standard library and Git only. TensorFlow, CUDA, GPU, and XLA are not initialized. |
| Misleading success | A syntactic complete lead pauses for manual admission; the runner cannot itself declare recovered P88. |

Audit verdict: `PASS_LANE_A_INVENTORY_EXECUTION`.

## Command

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m scripts.run_zhao_cui_austria_sir_fixed_variant_recovery_inventory --repository-root . --output docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/recovery-attempt-01/inventory.json
```

The environment variable records deliberate GPU hiding even though the runner
does not import TensorFlow. No replay runs unless a candidate passes manual
admission against every identity field.

## Budget

- one inventory attempt, at most 30 minutes;
- one repaired inventory only for a harness/schema failure;
- no numerical replay without an admitted candidate;
- no training, package mutation, GPU execution, parameter work, or HMC.

## Attempt Ledger

| Attempt | Classification | Outcome |
|---|---|---|
| 01 | Harness failure | Git batch reader used the subprocess input pipe for response reads; no inventory JSON was written and no scientific candidate was evaluated. |
| 02 | Repaired terminal inventory | Git/workspace inventory completed; two syntactic Phase-0-result leads were manually rejected because they declare the fields absent rather than serialize their values. |

Terminal result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-result-2026-07-30.md`.

Repaired command:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m scripts.run_zhao_cui_austria_sir_fixed_variant_recovery_inventory --repository-root . --output docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/recovery-attempt-02/inventory.json
```
