# Saved SIR precision-trace inspection

Question: at what observation does the already-reported FP32/FP64 score
discrepancy become material, and which saved diagnostics accompany it?
Compare the two existing final-precision-01 SIR results at identical controls,
data, particle design and parameter positions. This is host-side analysis of
saved evidence, with no new filter evaluation or GPU allocation.

The one-percent scaled final-score criterion remains unchanged. Incremental
and cumulative discrepancies locate the failure; ESS, reset errors, and final
particle-tangent magnitudes are explanatory only. No local threshold becomes
an additional promotion or continuation veto. Reconstruct cumulative scores
from the stored score increments using ordinary diagnostic host arithmetic,
and report the reconstruction gap to the saved final scores separately.

Do not infer the faulty primitive or a causal explanation from the first
visible discrepancy. FP64 is the same finite filter, not an independent model
oracle. The two saved rows do not describe an entire posterior region.
Preserve inputs and emit sir-fp32-trace-diagnostic-01.json plus a short result
note. The existing mixed-precision tuning continues independently.

Skeptical audit: score_trace stores increments, not cumulative scores; summing
these explicitly avoids mislabeling the comparison. Any material input/scope
mismatch invalidates this analysis. Deterministic arithmetic comparisons do not
support statistical accuracy or speed rankings.
