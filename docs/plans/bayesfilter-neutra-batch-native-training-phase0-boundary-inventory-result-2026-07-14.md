# Phase 0 Result: NeuTra Batch-Native Boundary And Inventory

Date: 2026-07-14

Master program:
`docs/plans/bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-2026-07-14.md`

Subplan:
`docs/plans/bayesfilter-neutra-batch-native-training-phase0-boundary-inventory-subplan-2026-07-14.md`

## Outcome

**PASS_PHASE0_AND_CONTINUE.** NeuTra optimizer updates now fail closed unless
the adapter exposes a repository-inspected, owner-bound method named
`neutra_batch_log_prob_and_grad_status`. The current exact `T=120` LGSSM
adapter is correctly rejected before output-directory creation because it still
uses scalar row mapping and has no eligible batch method.

The retired affine, bounded-GPU, and topology benchmark optimizer routes reject
through both their public entry points and preserved private historical bodies.
Their old source remains readable as migration evidence but is non-executable.

## Implemented Artifacts

- `bayesfilter/inference/neutra_batching.py`: repository-issued bound-method
  contract, source audit, integrity checks, status contract, and custom-gradient
  score injection.
- `bayesfilter/inference/neutra_training.py`: binding is required before runtime
  validation, output creation, flow construction, or optimizer execution.
- `tests/test_neutra_batching.py`: valid binding and negative forgery, mapping,
  delegation, XLA, singleton, and detached-callable tests.
- `tests/test_neutra_training.py`: generic training migration plus real exact
  LGSSM rejection before side effects.
- Retired entry-point tests in `tests/test_lgssm_neutra_training_tf.py` and
  `tests/test_neutra_gpu_bounded_training_tf.py`.
- Inventory:
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase0/neutra_optimizer_entrypoint_inventory.json`.

## Attempts And Repairs

| Attempt | Finding | Classification | Repair | Focused evidence |
| --- | --- | --- | --- | --- |
| Initial retirement suite | Retirement test omitted `LGSSMNeuTraTrainingError` import | mechanical test defect | Imported the exception | Suite moved from `1 failed, 16 passed` to pass |
| Source/inventory audit | Inventory statuses did not say public routes now fail closed | artifact/documentation mismatch | Refreshed current-status fields | JSON read and diff check pass |
| Skeptical boundary audit | Renamed historical functions remained directly callable | policy bypass | Added unconditional no-side-effect errors to private historical functions | New direct-call tests pass |
| Claude advisory review | Health and one-path read probes passed; substantive one-path reviews returned no output | reviewer prompt-surface limitation | Narrowed prompt per probe ladder, then recorded limitation | `CLAUDE_PROBE_OK`, `MASTER_PROGRAM_READ_OK`; no substantive verdict |

## Checks Actually Run

| Check | Result |
| --- | --- |
| Python compilation of changed modules, benchmark, and tests | pass |
| Boundary, real-adapter rejection, and retired-route suite | `22 passed` |
| Generic trainer group 1 | `5 passed` |
| Generic trainer group 2 | `5 passed` |
| Generic trainer group 3 | `5 passed` |
| Direct exact LGSSM row-mapped rejection | pass; no output directory created |
| Source scan for mapping, callbacks, NumPy, and Python loops | active generic training uses no sample-axis map/loop or host callback; reporting materialization remains; retired historical source is unreachable |
| `git diff --check` on Phase 0 paths | pass |

The 15 generic trainer tests were deliberately split across three CPU-hidden
processes because a prior all-in-one run accumulated XLA compilation memory.
Splitting changes test process topology, not tested semantics.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 0 | Ineligible adapters cannot reach an optimizer update or artifact side effect | no enforcement bypass observed | Direct-source AST audit is intentionally conservative and is not whole-call-graph proof | Freeze exact knowledge-transfer anchors in Phase 1 | LGSSM is not yet batch-native or fast |
| Keep current LGSSM serious training blocked | Exact adapter lacks eligible batch method | row-mapped scalar target remains a hard training veto | none for the current implementation state | Implement batched materialization and SVD/status kernel in Phases 2-4 | no claim against the scientific NeuTra direction |
| Continue despite missing Claude verdict | Local skeptical audit and focused tests answer Phase 0 risk | reviewer unavailability is not a scientific veto | Claude substantive prompt surface remains unreliable | Use local phase-transition reviews; retry bounded Claude only when material | no claim of Claude agreement |

## Ledger Status

| Ledger | Status |
| --- | --- |
| Engineering enforcement | pass for discovered NeuTra entry points |
| Numerical parity | not applicable in Phase 0 |
| Performance | not measured in Phase 0 |
| Scientific interpretation | no NeuTra quality or HMC claim |

## Handoff

Phase 1 entry conditions are met. The next subplan is
`docs/plans/bayesfilter-neutra-batch-native-training-phase1-dsge-knowledge-transfer-subplan-2026-07-14.md`.
Its local suitability audit explicitly rejects `tf.map_fn` DSGE fallbacks and
distinguishes in-graph reverse-KL noise from CPU worker target evaluation.

