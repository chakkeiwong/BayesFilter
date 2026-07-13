# Kalman QR Batched XLA Lean Repair Result

Date: 2026-07-13

Status: `CLOSED_NO_REPAIR_CANDIDATE_PROMOTED`

Plan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-plan-2026-07-13.md`.

## Result

The three planned local counterfactuals are complete. All preserved the tested
Kalman QR value/score semantics, but none established a memory repair:

| Candidate | Correctness | Structural result | CPU/XLA memory evidence | Decision |
| --- | --- | --- | --- | --- |
| Differentiate `tf.reduce_sum(value)` | Passed all four cells and row checks; output digests identical to explicit seed | Graph bytes increased about 0.21%; nodes increased about 0.43% | Not run because structural nomination failed | Not promoted; structural nomination failed |
| Build only value tensors with explicit batch-shaped bases | Passed all four cells, analytical parity, and row checks | Bytes reduced 5.14%-5.71%; nodes reduced 7.19%-8.23% | At `D=10,T=8,P=150,B=4`, both arms compiled and matched, but candidate peak RSS was not lower: 632,452 KiB versus 627,720 KiB. The 0.75% magnitude is descriptive only. | Not promoted; prospective first-pair RSS trigger failed |
| `tf.map_fn` singleton-row VJPs | Passed all four cells, analytical parity, and row checks | Bytes increased 25.76%-26.75%; nodes increased 2.77%-4.10% | Not run because no-regression veto failed | Not promoted; structural no-regression veto failed |

The production benchmark dispatch was not changed. The mapped candidate remains
diagnostic-only because the existing benchmark contract forbids batch mapping.
The output-seed and model-construction builders also remain unselected
counterfactual helpers; no public API or BayesFilter algorithm default changed.

## Evidence Contract Assessment

| Field | Result |
| --- | --- |
| Question | Answered for the three planned local constructions: none demonstrated a true-batched autodiff compile-memory repair. |
| Baseline | Current full-helper true-batched explicit-ones VJP on identical deterministic fixtures. |
| Correctness veto | Passed for every executed candidate/cell; consolidated focused regression passed. |
| Structural nomination | Passed only for explicit value-only construction. |
| Small-cell RSS promotion trigger | Failed prospectively: `candidate_peak_rss < baseline_peak_rss` was false in the first valid pair, so the plan did not authorize replication or promotion. This is a promotion decision, not evidence that the mechanism cannot help at another scale. |
| Explanatory only | Trace/first/warm call times and the exact 0.75% paired peak-RSS magnitude. GraphDef size served only as the predeclared nomination screen. |
| Not concluded | No historical `T=120,B=16` repair, universal memory/performance result, GPU viability, production/default readiness, HMC/posterior correctness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the current true-batched comparator; do not promote any tested candidate | Output-seed and mapped-row arms failed structural screens; value-only passed structural nomination but failed the prospective first-pair RSS promotion trigger | No numerical veto fired. Structural screens reject the output-seed and mapped-row constructions for this route; the value-only mechanism remains unpromoted rather than scientifically rejected | Whether fixed compiler costs or noise masked a value-only benefit, and whether the reverse functional `while` over the batched time scan dominates memory at realistic `T/B` | Design one new Tier 2 discriminator comparing current reverse-mode time scan with a mathematically equivalent checkpointed/custom-gradient or forward-sensitivity route; establish derivation and correctness before runtime | No repair completion, claim that value-only cannot help at another scale, GPU readiness, speed ranking, or historical large-cell viability |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Passed: candidate isolation, fail-closed compile harness, benchmark no-map contract, syntax/JSON/whitespace checks, and 165-test focused regression. |
| Numerical validity | Passed only for tested deterministic cells/dtypes and analytical comparisons. This is not broad posterior or model validation. |
| Memory/performance | No improvement established. One small CPU/XLA pair completed; candidate peak RSS was descriptively higher. No timing ranking is supported. |
| Scientific interpretation | Output-seed replacement and mapped-row VJP are structurally unfavorable on this ladder. Value-only construction is structurally favorable but remains unpromoted because its first RSS trigger failed. Neither that mechanism nor the broader memory-repair direction is scientifically rejected. |

## Artifacts

| Artifact | SHA-256 | Role |
| --- | --- | --- |
| `docs/benchmarks/kalman_qr_output_seed_counterfactual_2026-07-13.json` | `eb9d7f593a5fe80bdd56238ab9fb6082fb27a751767242db751e97fff17f797c` | Output-seed trace/correctness result |
| `docs/benchmarks/kalman_qr_model_construction_counterfactual_2026-07-13.json` | `f95316e2eeea0b13a1b062b5b3509164c703c2448a3bc724ba54cffbbfcb0f5a` | Three-arm model-construction trace/correctness result |
| `docs/benchmarks/kalman_qr_model_construction_cpu_xla_2026-07-13.json` | `fe39945313abbf57b20349995f04df0041841ca655e806ca33b39f5e1b673297` | Isolated small-cell CPU/XLA and peak-RSS result |
| `docs/benchmarks/kalman_qr_mapped_row_vjp_counterfactual_2026-07-13.json` | `10cdc2c17266a634da3ec0c150ad45fa7b7f631778e97fcfb91d25face7265ba` | Mapped-row trace/correctness result |

Run manifest common fields: Git commit
`a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`; Python 3.13.13; TensorFlow
2.20.0; deterministic synthetic nested fixture; `float32` diagnostic ladder;
GPU deliberately hidden with `CUDA_VISIBLE_DEVICES=-1`; one TensorFlow thread;
XLA disabled for trace diagnostics and enabled for the isolated compile pair.
Exact commands, environment, per-child wall times, source hashes, and output
paths are embedded in each JSON artifact.

## Checks

Focused consolidated command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_batched_fixture.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py \
  tests/test_kalman_qr_mapped_row_vjp_diagnostic.py
```

Result: `165 passed` in 67.04 seconds. TensorFlow AutoGraph emitted existing
Python 3.13 `gast` deprecation warnings; no test failed.

Focused Claude Opus read-only result review used only this result path. Round 1
returned `REVISE` because the binary RSS promotion trigger and the descriptive
0.75% magnitude were conflated. The note was repaired as above. Round 2 found
no material issue and returned `VERDICT: AGREE`. Claude did not inspect the
cited JSON artifacts; local validation and hashes carry that evidence burden.

## Negative-Result Classification

- Implementation failure: no candidate numerical implementation failed.
- Tuning failure: not applicable; no tuning comparison was performed.
- Diagnostic failure: the first CPU/XLA launcher attempt lacked its `/tmp`
  output directory and exited before TensorFlow import. The harness was fixed,
  checked fail-closed, and rerun without reusing target evidence.
- Evidence against the ideas: explicit seed replacement and mapped row VJP are
  structurally unfavorable on this ladder. Removing unused derivative
  construction is structurally favorable but was not promoted because the
  predeclared first-pair RSS trigger failed; the exact RSS difference is
  descriptive and does not establish absence of benefit at another scale.
- What remains viable: a change to the time-scan differentiation mechanism or
  analytical/forward-sensitivity comparator, subject to mathematical parity and
  bounded compiler-memory evidence.

## Post-Run Red Team

The strongest alternative explanation is that process-level peak RSS at this
small cell is too noisy or dominated by TensorFlow/compiler fixed costs to
detect a real large-cell advantage from the smaller graph. That possibility
does not rescue the candidate under the predeclared contract: the first pair
did not nominate replication, and the historical large cell was not authorized.

Evidence that would overturn the close decision is replicated paired peak RSS
showing a stable reduction, or a larger bounded cell where baseline and
candidate differ in compile completion or memory without changing semantics.
Such a run requires a new prospective plan amendment because the current
candidate failed its small-cell replication trigger.

The weakest evidence is the single-pair memory comparison. The strongest is
the deterministic four-cell correctness/structure evidence and the focused
regression suite. Therefore this note rejects the two structurally unfavorable
constructions and declines promotion of the value-only construction; it does
not reject the value-only mechanism at all scales or the broader research
direction.
