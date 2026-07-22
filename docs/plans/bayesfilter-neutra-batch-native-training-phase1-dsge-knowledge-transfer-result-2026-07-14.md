# Phase 1 Result: DSGE Batch-Native Knowledge Transfer

Date: 2026-07-14

## Outcome

**PASS_PHASE1_AND_CONTINUE.** The inspected DSGE work has been transferred as a
bounded mechanism ledger rather than copied as a monolithic recipe. The
selected local route is one GPU/XLA target call over a leading parameter batch
with stateless in-graph reverse-KL noise and custom-gradient score injection.
Persistent CPU worker sharding is retained as an alternative Phase 6 topology,
not described as offline training-data generation.

Specification:
`docs/plans/bayesfilter-neutra-batch-native-training-phase1-dsge-knowledge-transfer-spec-2026-07-14.md`

Source ledger:
`docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase1/source_transfer_ledger.json`

## Material Findings

- The DSGE adapter's direct batch path is reusable, but its `tf.map_fn` prior
  fallback is policy-incompatible and must not transfer.
- The NK canary's compiled target boundary and score injection are reusable;
  its Python optimizer loop and NumPy training orchestration are not.
- The SGU/Rotemberg 96-worker route evaluated each current optimizer batch on
  CPU workers. It did not pre-generate a fixed NeuTra training dataset.
- Historical launch summaries establish configured topology only; they do not
  establish measured speed, completion, parity, or LGSSM suitability.
- The local Cholesky batch Kalman implementation supplies batch shapes and
  control-flow design, while scalar SVD/eigh graph-status code remains the
  mathematical authority.

## Checks

| Check | Result |
| --- | --- |
| Required source hashes and exact line anchors | pass; recorded in ledger |
| JSON validation | pass |
| Unsupported speed/default claim audit | pass; all such material is explanatory or rejected |
| Batch/sample-generation terminology audit | pass; base noise, target batches, offline data, and HMC samples separated |
| Master Phase 2-7 dependency coverage | pass; selected, alternative, target-local, or rejected classification recorded |
| Phase 2 skeptical suitability audit | pass |
| `git diff --check` | pass |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Transfer GPU/XLA batch tensor mechanism | direct leading-batch graph design is source anchored | no target substitution | batch SVD FP64 performance unknown | implement target-local materialization | no speed claim |
| Keep CPU shards as alternative | historical route is precisely identified | host/NumPy bridge prevents selected-route admission | may outperform GPU FP64 target | compare only after correctness, if repair triggered | no default change |
| Reject DSGE mapping fallbacks | `tf.map_fn` and scalar row loop violate active policy | hard policy veto | none | fail closed if batch helper absent | no claim that all DSGE code is deficient |

## Handoff

Phase 2 entry conditions are met. The reviewed next subplan is
`docs/plans/bayesfilter-neutra-batch-native-training-phase2-lgssm-materialization-subplan-2026-07-14.md`.
It fixes exact tensor shapes and treats row parity plus primal/differentiated
Lyapunov residuals as hard gates for the shared multi-right-hand-side solve.

