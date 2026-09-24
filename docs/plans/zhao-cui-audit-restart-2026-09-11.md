# Zhao-Cui Audit Restart Checkpoint

Updated: 2026-09-11, Hong Kong time. This is the entry point for a NEW audit
conversation. Read this file first, then only the source/results needed for the
next stage. The saved conversations are incident evidence, not required input.

For execution, start from the shorter
[next-stage brief](zhao-cui-next-analysis-stage-2026-09-11.md). It isolates the
aggregate-validity repair first; rank activation follows. This checkpoint is
the broader evidence index and need not be loaded into every new conversation.

## Task And Current State

The bounded audit of the active Algorithm 3 derivation and implementation has
returned REVISE. The frozen-score and conditional-law fixtures pass, but two
runtime defects remain, the manuscript misstates the finite-grid probability
law, and there is no non-test Algorithm 3 consumer. No overall scientific
approval has been established. Runtime source has not been changed by this
continuation; repair and end-to-end validation remain next.

Research checkout: `/home/chakwong/BayesFilterZhaoCui`.
Recorded base: `47176bce7cdaa91ddd4466c39f90a11ad005c800`, with uncommitted
implementation work. Preserve unrelated changes in both checkouts.
Audit documents and diagnostic output live in `/home/chakwong/BayesFilter`.
Recheck the relevant source hashes before treating an old diagnostic as current.
All six hashes matched at this checkpoint's creation.

## Preserved Evidence

Read the result JSON, not the old conversation:

`/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-audit-20260911-01/diagnostic-result-01.json`

The companion `diagnostic_audit.py` is an independent CPU/reference diagnostic.
It recorded TensorFlow 2.19.1, GPU intentionally hidden, JIT disabled, and
2.22448 seconds inside the diagnostic (excluding framework startup).

| Check | Saved result | Meaning |
| --- | --- | --- |
| Rank activation | Requested rank two stays rank one; RMS 0.234375. Active rank-two warm start gives RMS 1.03e-10. | Reproduced initializer/ALS defect; not evidence against TT representational capacity. |
| Finite-output guard | `valid=true`, log likelihood `-inf`, particle weight sums 2.0. | Invalid result accepted; repair required before affected use. |
| Preparation/score wiring | Identity ancestry; path-sum value error 0; score finite-difference error 1.85e-12. | Pass for the tested frozen finite scalar only. |
| Gram and bounded initial density | Quadrature error 2.67e-15; density/Jacobian error 1.64e-10. | Tested mechanics pass. |
| Sigma-point guide | Moment and invalid-covariance fixtures pass. | This positive cubature projection is not an exact Gaussian posterior update. |

The first results above have a saved rerun in attempt 02. Findings are recorded
in the audit note:
`/home/chakwong/BayesFilter/docs/plans/zhao-cui-algorithm-audit-2026-09-11.md`.

## Completed Continuation

Read [the stage result](artifacts/zhao-cui-audit-20260911-03/stage-result.md)
for the new multidimensional conditional-law, numerical-CDF, real-C2 score,
consumer-inventory, and Lean checks. The six runtime hashes in the new
diagnostic still matched the research checkout at closeout. Lean passed with
the installed toolchain in 30.59 seconds; it checks finite algebra only.

The incident investigation also confirms excessive tool output: 657,799 saved
output characters across 28 results, seven truncated. Recovery had already
consumed 148,657 context tokens before the audit; the later counter reached
249,888 against a 244,800 compaction trigger. All calls have results. The
gateway failed both ordinary responses and compaction; its internal cause is
unconfirmed. See the structured session-context-summary.json beside the stage
result. Character counts are not token counts.

## Next Bounded Stage

1. Repair rank activation in the shared initializer with its consumers accounted
   for, or introduce an explicit initializer choice in the shared fitting API.
   Do not create an Algorithm 3-only solver fork. The positive rank-two target
   must fit, and healthy fixtures must retain their stated correctness checks.
   The exploratory constant perturbations are not an approved initializer.
2. Add fail-closed checks for per-step and aggregate values, scores, weight
   normalization, means, centered scores, and ESS in Algorithm 3. Preserve the
   invalid-input counterexample and run a healthy-score no-fire regression.
3. Reconcile master section 5 with the actual frozen coefficients/states and
   common-parameter target; reconcile its ESS roles. Correct manuscript lines
   2788-2790 to distinguish the interpolated CDF's cell-slope density from the
   smooth TT density and finite inversion error.
4. Wire one generic non-test consumer, preserving the guide's extension status.
   GPU/XLA and C2 claim-bearing runs wait for their recorded prerequisites;
   bounded repair diagnostics may continue. No C2/GPU launch budget was spent.

The governing audit contract is in the existing audit note. The research
checkout's repair plan is
`docs/plans/bayesfilter-zhao-cui-algorithm3-audit-repair-2026-09-10.md`.
Its master is
`docs/plans/bayesfilter-zhao-cui-algorithm3-ukf-guided-tt-master-program-2026-09-09.md`.
The analytical target freezes coefficients, charts, states, uniforms, and
proposal densities. It is not the adaptive-TT derivative or exact-model score.

## Session Discipline

Use a fresh conversation for each bounded stage while remote compaction is
unreliable. Do not resume or fork failed thread
`01a08be9-85ba-7f21-8e53-a2dbf566ab7d` to obtain a clean context.
Limit individual reads to relevant ranges and about 2,000 output tokens, and
combined tool output to about 6,000 tokens per batch. Per-command limits add
up in a batch; set an outer exec output limit too. Return selected fields from
results instead of emitting every complete tool-result object. Never search the entire
Codex session directory and print matching JSONL lines. Parse only the exact
session if incident evidence is actually needed.

Write findings immediately after each substantive check. Keep long command
logs in versioned files and return concise results. Start the next fresh stage
from the updated checkpoint. The launcher's 120,000-token compaction threshold
is a conservative trial setting, not a measured gateway limit or a guarantee.

Incident diagnosis and prevention details:
`/home/chakwong/BayesFilter/docs/plans/codex-zhao-cui-compaction-recovery-2026-09-11.md`.
