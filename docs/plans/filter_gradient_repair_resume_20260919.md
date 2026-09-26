# Filter and gradient repair resume checkpoint

Checkpoint through 04261, September 27. Repair branch HEAD `9f0a5508f` plus
this checkpoint contains the GenUT diagnostic follow-up and corrected cost
cohort. The loop-bound runtime repair remains installed; precision alternatives
remain uninstalled. F01--F20 and main promotion remain open.

The eight CPU diagnostics 04208--04215 reject both tested weight-gradient
regroupings. The source-feature cut changes the forward program and cannot
explain the original failure. A whole-program forward diagnostic retains the
graph values but still fails coordinate 18; XLA higher-order conversion fails.
No ill-conditioning conclusion or comparison-bound change follows. See
[precision results](filter_gradient_genut_weight_precision_result_20260926.md).

The corrected cost cohort 04245--04260 freezes the same FP32 objective
coefficients for every arm and its FP64 reference, saves every output and
gradient, and checks array hashes after execution. FP64 references 04246/04247
pass replay and finite differences at two steps. Twelve CPU/GPU graph/XLA
cost arms pass finite/replay/trace/owner-collection checks at N=1,000,d=3 and
N=10,000,d=18. Before/after graph records are bitwise identical. Every FP32
weight-gradient arm fails the independent bound; the cap report remains
mode-dependent. Timing and memory observations are descriptive, with no
promotion. The earlier 04217--04242 pilot is superseded because it generated
the wrong objective coefficients. See the [cost result](filter_gradient_genut_bounded_gradient_cost_result_20260926.md).

Policy 04261 passes all 129 checks after the final source additions: 270 guarded
sources and 1,424 exact allowances. This is a reviewed subset, not a whole-repo
compliance claim. The two new evidence receipts archive every raw file through
04261, including failed diagnostics and superseded pilots, with SHA-256 checks.
The corrected receipt is `genut-bounded-gradient-cost-verification-04261.json`;
the earlier unit is `genut-weight-and-cost-pilot-verification-04244.json`.

Charges through 04261: 86888.150888 CPU / 78707.470026 GPU seconds, leaving
31.864403 CPU / 30.136814 GPU hours under the 56/52-hour caps. The added 24 CPU
hours are already included. No numerical worker is active.

Next commit this checkpoint and integrate fetched remote main `52888d771`
(FAB-only changes) into the repair branch, then run focused FAB, GenUT and
policy checks before pushing. Continue public staged-locator/initializer and
consumer call-chain coverage, reporting/numerical failures, DZ5 integration,
native memory/capacity attribution and terminal source-frozen F01--F20 review.
Canonical NeuTra remains `bayesfilter_neutra_iaf_author_v1`; the user excluded
the canonical LEDH rebuild from this execution-policy campaign.
