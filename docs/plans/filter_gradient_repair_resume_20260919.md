# Filter and gradient repair resume checkpoint

Current checkpoint through 04140, September 26. Prior pushed commit:
`be3a9a662`. SIR terminal follow-up 04061/04062, refreshed costs 04063--04068,
and mixed KR/TTSIRT shared-owner tests 04092/04093 are qualified for their
bounded execution scopes. Consumers 04106/04107 pass 12 checks per backend.
Policy 04108, 04138 and 04140 pass 129 checks; the current source guard covers
270 sources with 1,424 exact allowances. See the SIR, mixed-KR and GenUT result
notes for exact evidence.

GenUT cap localization 04109--04113 explains the one-of-216 FP32 report
mismatch: XLA rewrites power/division into multiply by a negative power, and
FP32 rounding crosses the existing 1e-7 reporting predicate. Identical operand
replays, 80-digit Decimal, exact instrumentation and a diagnostic optimization
barrier reproduce the mechanism on CPU/GPU. The strict report comparison stays
open; no barrier, threshold, tolerance or runtime precision change was made.

Reduced-primal capacity 04114--04134 passes native finite/value/replay,
radial-cap and independent-moment checks through N=10,000,d=18 on CPU/GPU.
GPU XLA allocator peak is 6.3 MiB versus 49.3 MiB for native graph at that
fixture. A matched CPU d=18 probe 04139 finds native XLA median warm time 134.5
ms versus 42.0 ms for frozen original XLA (3.21x descriptive ratio), despite
smaller HLO; this remains an open performance limitation. FP64 references
04135--04137 show the non-report full-record failures in several original arms,
while native non-report fields pass against FP64. No comparator tolerance was
weakened. Capacity evidence is bounded to the reduced primal and does not
qualify full LEDH/reset consumers.

Charges through 04140: 85657.405133 CPU / 77576.305836 GPU seconds, leaving
32.206276 CPU / 30.451026 GPU process-hours under unchanged 56/52-hour caps.
The extra 24 CPU hours are already included. No campaign worker is active.
Remote main `5e16df06f586c16bc58fb76bc62d4f6451e7690d` remains contained in this
branch; main is not merged. Canonical LEDH rebuilding remains excluded; current
NeuTra architecture is `bayesfilter_neutra_iaf_author_v1`.

Next: decide and qualify an uncertainty-aware cap-report contract without
silently selecting graph or XLA rounding; continue registered public-consumer,
LEDH/reset, initializer/supervisor, DZ5, repeated-constructor and F01--F20
terminal repairs. Target/full-consumer capacity remains open. Main merge stays
blocked until the master gate passes. Pending reporting proposals are not
approved by elapsed time.
