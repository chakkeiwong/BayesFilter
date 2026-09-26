# Filter and gradient repair resume checkpoint

Current checkpoint through 04164, September 26. Prior pushed commit:
`a6609df85`. No numerical worker is active. The SIR and public TTSIRT log
execution repairs are committed; see their September 26 result notes.
GenUT CPU/GPU FP64/FP32, reset/callback and native-cost refresh 04141--04152
passes. LEDH safety/stage/consumer refresh 04153--04160 passes 23 checks per
backend without canonical admission. Policy 04164 passes 129 checks over
270 guarded sources / 1,424 exact allowances. All these checks are bounded
endpoint evidence, not whole-repository closure.

GenUT cap-report root cause is resolved: XLA's division/power rewrite changes
FP32 rounding across the unchanged 1e-7 reporting predicate. Strict cross-mode
report equivalence remains open. Reduced-primal capacity passes independent
moments and exact replay through N=10,000,d=18; GPU XLA peaks at 6 MiB versus
47.05 MiB for native graph. This does not establish full reset/score capacity.
Original comparator failures are localized against independent FP64 records.

CPU d=18 native XLA remains slower than the original. Last-axis layout trial
04161 passes numerics but is slower and is rejected. Highest-precision XLA dot
trial 04162/04163 passes CPU/GPU same-mode comparisons, reduces CPU d=18 median
from 80.7 to 34.8 ms, but slows GPU d=3 (1.38 to 1.78 ms) and has no TensorFlow
registered gradient. Neither trial changes runtime. See
`filter_gradient_genut_reduction_layout_result_20260926.md` and its unit for
next qualification: native pullback, graph compatibility, ownership, isolated
memory and evidence for any static size dispatch. No unreviewed score or
backend change is allowed. Reporting semantics need a validated uncertainty
contract; no tolerance waiver or mode-specific normalization is installed.

Evidence archive `genut-capacity-evidence-04140-r2.tar.gz` repairs r1's omitted
per-run capacity JSON/NPZ files and contains full raw data/analyzers. r1 remains
preserved. Refresh/trial evidence is in `genut-refresh-layout-evidence-04164.tar.gz`.
Git write/NVIDIA discovery failures were sandbox restrictions; trusted retries
passed. No evidence supports a disk or GPU hardware failure.

Charges through 04164: 85892.592118 CPU / 77830.415020 GPU seconds, leaving
32.140947 CPU / 30.380440 GPU process-hours under unchanged 56/52-hour caps.
The user's added 24 CPU hours are already included. Remote main
`5e16df06f586c16bc58fb76bc62d4f6451e7690d` is contained in the branch. Main is
not merged; F01--F20 terminal dispositions and current-source terminal coverage
remain open. Canonical LEDH rebuilding is excluded; NeuTra remains
`bayesfilter_neutra_iaf_author_v1`.

Next continue bounded highest-dot qualification and unresolved public-consumer
repairs, especially staged locator error/ownership integration, full reset,
initializer/supervisor, DZ5 and repeated-constructor retention. Pending
reporting/isotropic proposals are not approved by elapsed time. The merge gate
must not turn green merely because registered diagnostics passed.
