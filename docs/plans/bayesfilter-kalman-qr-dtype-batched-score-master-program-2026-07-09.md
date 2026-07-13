# BayesFilter Kalman QR Dtype And Batched Analytical Score Master Program

Date: 2026-07-09

## Status

`PHASE_7B_BLOCKED_GPU_TENSORFLOW_VISIBILITY`

## Objective

Remove hard-coded floating dtype assumptions from the TensorFlow QR Kalman
value and analytical-score lanes, then build a true batch-native analytical QR
score path for simultaneous evaluation across independent parameter proposals.

This program is engineering infrastructure for the existing Kalman QR lane.  It
does not claim HMC readiness, posterior correctness, method superiority,
default-readiness, or Zhao-Cui source faithfulness.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the QR Kalman value, autodiff-score, analytical-score, benchmark, and planned batched analytical-score paths honor requested `float32`/`float64` dtype while preserving correctness and XLA compatibility? |
| Mechanism under test | Dtype-polymorphic TensorFlow helper infrastructure followed by QR value cleanup, analytical score cleanup, benchmark dtype controls, and batch-native analytical score implementation. |
| Expected failure mode | Hidden `tf.float64` casts remain in helpers, FP32 benchmark silently runs FP64 kernels, FP32 tolerance is too strict or too loose, XLA compilation fails for one dtype/device, or a vectorized scalar wrapper is mistaken for a batch-native score kernel. |
| Promotion criterion | Phase-specific only: dtype contract tests must prove requested dtype reaches outputs and relevant graph calls; score/value parity must hold within dtype-appropriate tolerance; batch-native score must match scalar analytical rows and small autodiff references. |
| Promotion veto | Any core QR kernel coercing floating tensors to `tf.float64` after the dtype phase, missing observed dtype in benchmark artifacts, failed parity, nonfinite output, unsupported speed claim, or treating `tf.vectorized_map(tf_qr_sqrt_kalman_score)` as the final batch-native implementation. |
| Continuation veto | Dtype contract cannot be stated clearly, local checks fail without a reviewed repair, Claude/Codex review does not converge after five rounds for the same material blocker, or a phase would require an unapproved GPU/runtime/default-policy boundary. |
| Repair trigger | Hidden dtype cast found, output dtype mismatch, FP32/FP64 parity drift beyond declared tolerance, compile failure, artifact missing requested/observed dtype, or benchmark/device provenance incomplete. |
| Explanatory diagnostics | Compile+first-call time, warm-call time, TF32 flag, device placement, XLA logs, dtype inventory counts, tolerance deltas, and scalar-vs-batched timing. |
| What must not be concluded | Runtime superiority, statistical ranking, posterior correctness, HMC readiness, default-policy readiness, broad GPU production readiness, or scientific validity. |

## Phase Index

| Phase | Name | Objective | Subplan | Required result |
| --- | --- | --- | --- | --- |
| 0 | Contract and dtype inventory | Lock governance, review plan boundaries, and inventory hard-coded dtype sites before source edits. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-result-2026-07-09.md` |
| 1 | Dtype infrastructure | Add or localize dtype helpers and tests that preserve input dtype and reject mixed floating dtypes where required. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-result-2026-07-09.md` |
| 2 | QR value dtype cleanup | Make scalar, while-loop, batched-static, and masked QR value paths dtype-polymorphic. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-result-2026-07-09.md` |
| 3 | Analytical score dtype cleanup | Make public QR analytical score and needed derivative helpers dtype-polymorphic. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-result-2026-07-09.md` |
| 4 | Benchmark dtype controls | Add benchmark `--dtype` controls, requested/observed dtype recording, and fail-closed dtype checks. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-result-2026-07-09.md` |
| 5 | Batched analytical score contract | Specify `[B, ...]` model tensors, `[B, P, ...]` derivative tensors, output shapes, limitations, and reference baselines. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-result-2026-07-09.md` |
| 6 | Batch-native analytical score implementation | Implement a true batch-native analytical QR score kernel without using a vectorized scalar wrapper as the final path. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-result-2026-07-09.md` |
| 7 | Correctness and benchmark ladder | Run CPU/GPU XLA correctness and descriptive timing across dtype, batch size, parameter count, and dimensions. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-result-2026-07-09.md` |
| 8 | Closeout | Record decision tables, inference-status table, manifest, remaining risks, and nonclaims. | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase8-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase8-closeout-result-2026-07-09.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the QR Kalman lane become dtype-polymorphic and batch-native for analytical score without silently changing precision or overstating runtime evidence? |
| Baseline/comparator | Current hard-coded FP64 QR value/score paths, existing scalar analytical score, existing batched-static QR value path, and the 2026-07-09 FP64 CPU/GPU XLA benchmark artifacts. |
| Primary pass criterion | All reached phases pass local checks, review gates, dtype-output assertions, parity checks, and artifact requirements. |
| Veto diagnostics | Hidden FP64 coercion after the relevant cleanup phase, requested/observed dtype mismatch, failed scalar-vs-batched parity, failed CPU/XLA smoke, unapproved GPU use, missing result artifact, or unsupported claim. |
| Explanatory diagnostics | Timing tables, compile+first-call deltas, warm-call medians, TF32 global flag, device placement, XLA compile logs, and dtype inventory counts. |
| Numeric provenance | Tolerances and benchmark sizes are phase hypotheses until stated in the phase subplan. Existing dimensions/parameter counts are inherited from the 2026-07-09 benchmark request. |
| Not concluded | Runtime superiority, statistical ranking, HMC readiness, posterior correctness, broad production readiness, default-policy change, or scientific validity. |
| Artifacts | Master program, runbook, ledger, stop handoff, phase subplans/results, review bundles, JSON/Markdown benchmark outputs, and logs. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Preserve FP64 as allowed dtype | Historical implementation and existing tests | Avoids breaking current numerical reference behavior | FP64 treated as only supported dtype | Phase 1/2 dtype tests include FP32 and FP64 | hypothesis |
| Add FP32 support before batched score | User directive and current hard-coding discovery | Avoids building new kernels on top of a flawed precision contract | Batched score repeats FP64-only design | Phase 0 inventory and Phase 1 helpers | reviewed target |
| Use CPU-hidden checks first | Repo policy allows CPU debug/reference checks | Keeps dtype repairs cheap and isolates GPU boundary | CPU evidence mistaken for GPU/default readiness | Artifacts label CPU-hidden and nonclaims | hypothesis |
| GPU/XLA checks later | BayesFilter default execution target is GPU | Needed for final benchmark/device evidence | Untrusted or unrecorded GPU run interpreted as evidence | Phase 7 trusted GPU provenance gate | future approval |
| `tf.vectorized_map` only as reference/comparator | Current scalar score is not batch-native | Prevents false completion of batch-native objective | Wrapper mislabeled optimized kernel | Phase 5/6 tests inspect source/contract | veto |

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is current FP64-only behavior and existing scalar/batched value tests, not an FP32 implementation. |
| Proxy metrics promoted | Fixture dtype is not sufficient; observed kernel output dtype and benchmark artifact dtype are required. |
| Missing stop conditions | Each subplan includes review, local-check, dtype, GPU, and unsupported-claim stop conditions. |
| Unfair comparison | Timing is descriptive until replicated; dtype and batch axes are separated in later benchmarks. |
| Hidden assumptions | Numeric tolerances, batch sizes, and benchmark grids must be restated in each executable subplan. |
| Stale context | Phase 0 inventories current source before edits and records the actual hard-coded sites. |
| Environment mismatch | CPU-hidden checks cannot support GPU/default-readiness claims; GPU work is a later trusted gate. |
| Artifact mismatch | Each phase must write a result and refresh/review the next subplan before advancing. |

Audit status: `PASSED_FOR_PHASE_0_DOCUMENT_REVIEW_ONLY`.

## Review Protocol

Codex is supervisor and executor.  Claude is read-only reviewer only.

Use `/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh` for
material plan, subplan, result, and boundary reviews only when that external
disclosure boundary is separately approved.  The first Phase 0 attempt was
rejected by the approval reviewer as external disclosure risk, so Claude review
is unavailable for this run unless the user explicitly re-approves that risk.

When Claude is unavailable because of approval-policy rejection, transport
failure, or probe failure, replace the material review with a fresh bounded
Codex substitute review and record the weaker review status.  If Claude is
approved and the probe succeeds but the material review times out or returns no
verdict, treat Claude as alive and redesign the review bundle or prompt to a
smaller exact path before retrying.  Stop after five review rounds for the same
material blocker.

Claude cannot authorize human, runtime, model-file, funding, product,
default-policy, scientific-claim, or GPU trust boundaries.
