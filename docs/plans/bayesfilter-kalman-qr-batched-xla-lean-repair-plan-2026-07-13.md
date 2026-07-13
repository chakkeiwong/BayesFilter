# Kalman QR Batched XLA Lean Repair Plan

Date: 2026-07-13

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `COMPLETED_NO_REPAIR_CANDIDATE_PROMOTED`

## Question

Which smallest local construction change reduces the true-batched autodiff
GraphDef/XLA compile-memory burden while preserving the current value and score
semantics, and does that repair allow the previously blocked bounded CPU/GPU
XLA cases to compile and run?

The purpose is to fix and measure the memory/performance problem, not to prove a
unique historical TensorFlow construction origin.

## Current Baseline

- Comparator: true-batched analytical QR score versus true-batched reverse-mode
  autodiff on identical fixtures, dtype, batch, parameter count, and device/JIT
  settings.
- Historical failures and original grid:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-reset-memo-2026-07-10.md`.
- Current structural evidence: the reviewed discriminator result cited by the
  governance reset. It is explanatory only and found no unique repair point.
- The original memory/performance problem remains open. Passing unit tests or a
  smaller GraphDef alone will not close it.

## Hypotheses And Smallest Discriminators

1. `output_seed`: explicit `tf.ones_like(value)` VJP seeding contributes
   avoidable shape/constant construction. Compare it with differentiating
   `tf.reduce_sum(value)` without an explicit output gradient.
2. `broadcast_gradients`: broadcasted base-plus-`einsum` model-tensor
   construction creates dynamic broadcast-reduction gradients. Test one
   algebraically equivalent, shape-explicit construction in isolation.
3. `single_vjp_envelope`: one batched VJP causes a large reverse functional
   while. Compare a graph-native mapped row VJP (`tf.map_fn` or another reviewed
   TensorFlow-native mapping) against the current VJP; do not use a Python row
   loop as the promotion comparator.

Run these in order. Stop exploring later hypotheses when an earlier
counterfactual produces a material structural reduction and passes correctness;
move that candidate to bounded compile measurement.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact question above. |
| Baseline | Current true-batched autodiff callable and the matching analytical callable on the same frozen fixture. |
| Diagnostic criterion | At the smallest representative cells, the candidate preserves finite value/score shape and parity while reducing the structure implicated in compile pressure. GraphDef nodes/bytes are nomination evidence, not proof of a memory fix. |
| Repair pass criterion | The candidate passes focused correctness/parity tests and a bounded method-isolated XLA compile that previously failed or shows a reproducible reduction in measured peak compile memory against the exact baseline. |
| Vetoes | Value/score mismatch, non-finite output, row-dependence violation, unfair comparator, stale artifact reuse, compile crash, missing memory provenance, or a candidate whose apparent reduction comes from changing the mathematical target. |
| Explanatory only | Source-region counts, node names, GraphDef/HLO size, trace time, and one-off memory/timing observations. |
| Nonclaims | No universal speed ranking, production/default readiness, HMC/posterior correctness, framework defect, or scientific validity from this repair alone. |
| Result artifact | Update this plan during diagnostics, then write one concise result/reset note with exact commands and artifact paths after a material run. |

## Execution Sequence

1. Add focused tests or a diagnostic harness for one counterfactual only.
2. Run CPU-hidden non-JIT or trace-only correctness/shape checks as debugging
   evidence; label them non-default.
3. Compare GraphDef structure at a small fixed ladder, initially dimension 10,
   `P in {50, 150}`, and `B in {1, 4}`. Add `B=16` only for a nominated
   candidate.
   Nominate a candidate only if total serialized GraphDef bytes or total node
   count falls by at least 5% at both `P=150` cells, no tested cell regresses by
   more than 1% on either measure, and all correctness checks pass. This is a
   prospective diagnostic threshold, not evidence that compile memory is fixed.
4. For a correctness-passing candidate, run one method-isolated CPU XLA compile
   with a five-minute initial limit and record peak RSS, compile stage, GraphDef
   bytes/nodes, and output parity. A timeout is a result, not a crash diagnosis.
5. If the CPU gate is valid, run the smallest trusted managed-session GPU/XLA
   case and record device, TF32, JIT, dtype, peak device/host memory where
   available, and failure stage.
6. Patch the production-target implementation only after the counterfactual
   identifies a local change and focused correctness passes. Rerun the relevant
   linear Kalman/QR suite.
7. Run a bounded comparison ladder only after both analytical and autodiff arms
   compile on that lane with equivalent semantics. Use uncertainty-bearing
   repetitions before any timing ranking.

## Resource And Stop Rules

- Start with the smallest diagnostic; do not launch the historical full grid.
- Initial compile attempts are limited to five minutes per method/cell and one
  cell at a time. Increase a budget only after inspecting a valid stage artifact.
- Stop a candidate for incorrect math, corrupted/missing evidence, uncontrolled
  memory growth, or an invalid comparator. Do not stop the whole repair merely
  because one candidate or backend fails.
- Destructive operations, dependency changes, external services, public/default
  changes, and overnight or broad-sweep execution require separate Tier 3
  approval/planning.

## Skeptical Audit

Status: `PASSED_FOR_SMALL_COUNTERFACTUAL_DIAGNOSTICS_ONLY`.

- Wrong baseline is controlled by using the current true-batched pair, not the
  historical Python row-loop comparator.
- Graph size is explicitly a nomination proxy; compile completion and measured
  peak memory carry the repair claim.
- CPU and GPU failures are lane-local and will record exact failure stages.
- The plan preserves parity and row-independence so a cheaper but different
  computation cannot be promoted.
- No command for an overnight grid or scientific/default claim is authorized by
  this plan.

## Next Action

Implement and test only the `output_seed` counterfactual first. If it does not
materially change the graph, record that negative result briefly and proceed to
`broadcast_gradients` without creating another phase plan.

The first diagnostic writes
`docs/benchmarks/kalman_qr_output_seed_counterfactual_2026-07-13.json`; a
material decision is recorded in
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-result-2026-07-13.md`.

## Live Progress

### Output Seed Counterfactual

Status: `REJECTED_AS_REPAIR_CANDIDATE`.

The candidate passed all value/score, analytical-parity, finiteness, shape, and
row-independence checks. It produced identical output digests to the explicit
seed baseline, but increased serialized GraphDef bytes by about 0.21% and total
nodes by about 0.43% in every cell. No XLA compile was justified.

### Model Construction Counterfactual

Status: `NOT_PROMOTED_AFTER_SMALL_CELL_RSS_TRIGGER`.

The fresh baseline graphs contain zero `BroadcastGradientArgs`, so merely
adding `tf.broadcast_to` would not test the stated dynamic-gradient mechanism.
Code tracing instead found that the shared `_batched_model_tensors` helper
constructs eight analytical-derivative outputs even though the autodiff
likelihood consumes only its first eight value tensors. The next diagnostic
therefore compares, on the same four cells:

1. the unchanged full-helper explicit-seed baseline;
2. a value-only helper with the same implicit base broadcasting; and
3. a value-only helper with explicit static batch-shaped bases.

Both candidate arms retain the explicit all-ones VJP seed, isolating model
construction from the rejected output-seed change. Apply the same correctness
veto and 5%/1% nomination thresholds above. Prefer the simpler implicit arm
unless the explicit arm adds at least a further 1% reduction in bytes or total
nodes at both `P=150` cells without regressing the other metric. The artifact is
`docs/benchmarks/kalman_qr_model_construction_counterfactual_2026-07-13.json`.

The diagnostic nominated `value_only_explicit`: all correctness checks passed,
and the four cells reduced serialized GraphDef bytes by 5.14%-5.71% and total
nodes by 7.19%-8.23%. The next CPU/XLA gate compares the unchanged full-helper
baseline with that candidate at `D=10`, `T=8`, `P=150`, `B=4`, `float32`, one
TensorFlow thread, in separate fresh processes. Each method has a 300-second
TERM limit plus a 10-second KILL grace. Record first-call compile/execution
completion, warm-call completion, output parity, and GNU-time peak RSS in
`docs/benchmarks/kalman_qr_model_construction_cpu_xla_2026-07-13.json`.

This small-cell compile is a continuation gate only. It cannot close or repair
the historical `T=120`, `B=16` failure. If both arms complete and candidate
peak RSS is lower, repeat the isolated pair before making a memory-reduction
claim. If the reduction replicates, patch the benchmark autodiff builder and
run focused tests before attempting the larger bounded cell.

The first launcher attempt exited before TensorFlow import because its GNU-time
output directory was absent. It produced no target evidence. The harness now
creates that directory before launch and its aggregate checks fail closed when
no child passes; rerunning repetition 1 does not reuse target work.

The repaired CPU/XLA pair completed for both methods with finite, shape-correct,
single-trace outputs and baseline/candidate parity. The candidate retained its
smaller GraphDef but used 632,452 KiB peak RSS versus 627,720 KiB for the
baseline, a descriptive 0.75% increase. This fails the prospective trigger for
replication, production patching, or a measured memory-reduction claim. The
binary trigger outcome governs promotion; the 0.75% magnitude is descriptive
only and does not establish that the mechanism cannot help at another scale.

### Mapped Row VJP Counterfactual

Status: `REJECTED_AS_REPAIR_CANDIDATE`.

Compare the unchanged true-batched full-helper VJP against one TensorFlow
`tf.map_fn` over rows. Each mapped body uses a singleton-batch view of the row,
the same `_batched_model_tensors` helper, the same batched-static likelihood,
and one row-local reverse-mode gradient. Use `parallel_iterations=1`; do not use
a Python row loop or scalar Kalman implementation. Apply the same four-cell
correctness, row-independence, and 5%/1% GraphDef nomination contract. The
artifact is
`docs/benchmarks/kalman_qr_mapped_row_vjp_counterfactual_2026-07-13.json`.

The mapped arm passed every correctness and row-independence check but increased
serialized GraphDef bytes by 25.76%-26.75% and total nodes by 2.77%-4.10%.
It failed the no-regression veto and was moved out of the benchmark source into
its diagnostic-only harness to preserve the existing no-map benchmark contract.

## Close Decision

All three planned counterfactuals are complete. None supports a production
patch, replicated memory-reduction claim, GPU launch, or historical-repair
claim. The unchanged full-helper true-batched autodiff route remains the
benchmark comparator. Exact results and the next justified hypothesis are in
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-result-2026-07-13.md`.
