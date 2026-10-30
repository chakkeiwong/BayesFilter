# Dense initializer cloud execution unit

E5's next dependency encloses the actual dense initializer's ordered partition
evaluation, finite/validity veto and strict incumbent selection. The clean local
clone is `/tmp/macrofinance-dz5-filter-xla-20260923`, branch
`repair/filter-xla-initializer-20260923`, at external commit2f386f75. Its
`bayesfilter_estimation_initialization.py` exactly matches the live initializer;
SHA2567c4d5598959fcd9605bed01d16d416f2002fd442301e82264ff396b1aa050f75.
The live dirty worktree and completed CDF/frozen campaigns remain untouched.
No current CDF/commodity target or sampler run is part of this unit.

Question: can the existing training/selection/audit partition loop become one
reusable operand-bound XLA program while preserving every target call and
selection? This is accepted TF/TFP execution repair under E5, initially an
internal dependency. Baseline is the exact external loop from that pinned
function. Diagnostic AST extraction may expose its locals for comparison, but
must retain the original numerical statements and order. It must not replace
the original with a hand-written supposedly equivalent loop.

Prepared offsets are explicit operands, with fixed active row counts for the
three roles. Use the original eager TF stateless draws to freeze identical
diagnostic offsets in both arms. This does not establish a new RNG default or
claim seeded equality between eager and XLA generators. Native random preparation
and full attempt/fitting integration are later E5 steps. Do not pad target calls:
each callback receives exactly its role's original row extent. Zero-padding is
only for bounded history storage. Each active callback returns value, score and
validity atomically. No scalar-row fallback or duplicate telemetry call.

The native loop must preserve all partition target values/scores/validity,
separately rounded `center + offsets * scale`, scaled scores, earliest argmax
and strict greater-than promotion, short-circuit at the first invalid cloud,
row/callback counts, candidate center/value and complete completed history.
No later partition may execute after invalidity. Compare against the exact
original loop for healthy, improving, tied, invalid and nonfinite cases with
unequal training/selection/audit extents, changed operands and return-to-original.
Cover one-dimensional and non-diagonal three-dimensional targets. Require one
trace, unchanged HLO across changed runtime operands and callback/graph owner
collection. NumPy may occur only inside this independent external reference.

Reserve at most12 workers/2400 charged seconds,120s per ordinary worker and300s
for a justified combined case, under the unchanged32CPU/52GPU campaign caps.
Use registered runner groups and fresh numbered output directories. Freeze
runtime/scripts/tests during each worker/matrix. CPU is an explicit reference;
GPU uses the unchanged selector, memory growth and provenance. Any numerical,
decision, callback order/count or lifetime mismatch stops promotion and triggers
localized repair at unchanged comparisons. Compilation failure propagates with
no eager retry. This unit cannot close full initializer, RNG, curvature input
validation, public/cost, actual target/transition or external deployment gates.

Skeptical review: treating zero-padded rows as real target inputs would change
both exact-row accounting and target behavior; use static-shape branches around
actual calls. Fusing the original two affine operations may change an incumbent,
so retain their already qualified rounding boundary. Passing a scalar mapper
as a batch callback would not qualify the actual NeuTra consumer. Reusing the
existing numeric fitter later must also carry its disjoint-partition, selection
and error checks; calling the internal fitter alone cannot bypass them. The
isolation clone is source preparation, not a renewed CDF campaign or permission
to replace its frozen numerical bytes.

First worker03350 fails all seven D1 cases during tracing: a combined dynamic
index/static slice loses the target's known row extent. No native call executes.
Use gather for the partition and retain the static slice with an explicit shape;
this preserves values and gives the batch-native callback its required extent.
Preserve03350 and retry within the12-worker reservation at unchanged comparisons.

03351 passes every D1 original numerical/decision/callback comparison, but six
cases fail unchanged-HLO checks; the constant case passes. Preserve this
candidate failure and save the HLO diff in a one-case localization before any
runtime repair. No HLO gate relaxation. The exact external source is now also
retained as a byte-identical diagnostic text fixture in the BayesFilter checkout
so these tests do not depend on the isolation clone surviving.

03352's raw HLO diff proves real specialization: changed center, scale and clouds
appear as changed constants, rather than runtime operands. The value-dependent
`points[best]`/`values[best]` StridedSlice path is the same specialization hazard
already repaired elsewhere in this campaign. Replace it with tf.gather, which
keeps the selected row an operand and preserves its value. Re-run the unchanged
full-record/HLO checks before qualification. Store the unchanged external source
as a diagnostic `.py` reference fixture so the runner hashes its bytes too.

03353/03354 pass all14 complete CPU cases after that indexing repair. Every
case compares three invocations (changed center/scale/clouds, then return),
all arrays and archive fields, exact target row/call order, earliest ties,
invalid-first/second and nonfinite-value/score exits. Each has one trace,
unchanged HLO and collected program/graph/callback. The frozen diagnostic offsets
are TF stateless-normal draws; they are supplied identically to the extracted
original loop and candidate, not claimed as the external ball-RNG stream.
Actual seeded cloud generation remains a separate obligation. GPU renewal and
an enclosing consumer still remain before integration.

GPU03359/03360 now pass the same14 complete cases, including changed operands,
unchanged HLO, exact callback order/counts and owner/graph/callback collection.
The unit uses7/12workers and76.916920/2400seconds, including the three preserved
failed workers. The complete native module is guarded with no new exception:
policy03371 passes129checks and direct enforcement reports230sources/1333exact
exceptions. A compiled enclosing consumer, actual seeded cloud generation,
curvature input validation and full initializer integration remain required.
