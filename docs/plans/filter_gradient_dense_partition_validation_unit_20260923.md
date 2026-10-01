# Dense initializer partition validation dependency

Completed through03407:03401 CPU and03402 GPU each pass all68 validation
scenarios in two dimensions. Complete original errors/order, signed-zero and
subnormal identity, enclosing compilation, changed-input HLO and collection pass.
Affected fixed-geometry suites03403 CPU/03404 GPU each pass121 checks. Policy03406
passes129 checks; the guard covers231 sources/1333 exact exceptions, with no new
waiver. Eight workers are charged to this12-worker unit including the two
preserved failures, final policy renewal and explicit-CPU inventory03407.
The checker and public subnormal repair qualify their tested scope. Its result
still must gate the actual fitter inside the E5 enclosing initializer.

E5 needs the fixed-center fitter's input validation inside its enclosing
initializer. Calling the existing internal fit program alone would bypass finite
input and copied-row checks. This unit implements only that prerequisite; full
locator/cloud/fitting/attempt composition and public wiring remain separate.

Question: can finite-input checks and exact copied-row detection preserve the
original public fitter's validation order in a stable XLA program? The authority
is original3582b4ac `fixed_center_curvature.py`, with current public checks also
compared. Use its actual `_vector`, `_cloud_partitions`, `_cloud_pair` and
`_require_independent_partitions` functions in an independent diagnostic loader,
not a rewritten reference. Shape/count/configuration checks are fixed construction
metadata. Prepared padded partitions occupy disjoint logical ranges; caller
arrays still require the public shared-buffer check at preparation. This program
cannot authorize arbitrary overlapping caller buffers or replace that host check.

Preserve first error order: center, center score, training offsets/scores,
selection offsets/scores, audit offsets/scores, then copied rows. Compare only
active rows; padding has no numerical meaning. Match cross-partition exact
float64 row identity, normalizing signed zero exactly as the original. Repeated
rows within a partition are allowed. Return the first overlapping partition pair
in original traversal order. Use TensorFlow loops and bitwise float64 identity
to avoid GPU subnormal comparison changes and a full all-partitions product.
No tolerance, threshold, solver, RNG or numerical-result change is introduced.

Tests cover unequal row extents, dimensions1/3, healthy clouds, duplicates in
every role, within-partition repeats, signed zeros, distinct subnormals, nonfinite
input precedence and nonfinite padding. Check complete original exception type/
message, changing same-shape clouds without trace/HLO change, enclosing XLA use
and callback/program collection. No fitting or target calls are expected in this
pure checker. Its result must later gate actual fitting with original error
ordering; a passing checker does not establish the enclosing call chain.

Reserve at most12workers/1800charged seconds,120s ordinary/300s localized worker,
under unchanged32CPU/52GPU-hour caps. Use the stable registered campaign runner,
fresh numbered directories, explicit CPU reference followed by GPU qualification,
verified growth and the existing GPU selector. Runtime/scripts/tests stay frozen
during workers and matrices. Stop promotion on any numerical/error/HLO mismatch;
localize and repair without weaker comparisons. Stop execution on invalid
evidence, changed scientific contract or exhausted budget.

First worker03399 fails in diagnostic loading, before numerical execution: the
generic frozen loader follows unrelated HMC imports and treats the runtime package
as a missing runtime.py module. Extract the four unchanged original validation
function ASTs with their only numerical dependency, NumPy, instead. Record full
source and excerpt hashes and preserve every statement; no runtime, numerical
authority or comparison changes. Retain03399 and retry within the reservation.

Worker03400 exposes a real existing validation defect after the loader repair:
current `_require_independent_partitions` uses floating equality to normalize
zero, which collapses distinct subnormal values and falsely reports copied rows.
The new native checker matches original3582b4ac on those cases. Share its bitwise
signed-zero normalization with the current host validator through the existing
stable XLA kernel boundary. This restores original exact identity, changes no
overlap criterion or accepted numerical result, and adds no exception. Preserve
03400 and renew full checker and affected fitter input/consumer checks.

Skeptical review: row equality is a numerical decision and cannot remain Python
byte-set selection inside the enclosing controller. Full tensor-product row
comparisons can waste memory; an ordered pair loop bounds temporary size to two
partitions. Padding must not create false duplicates or hide nonfinite active
inputs. Signed zero and subnormal cases discriminate exact identity from an
accidental approximate/flush-to-zero comparison. Separate preparation checks
remain mandatory because a tensor checker cannot reconstruct a caller's original
buffer aliasing. Execute only after the frozen GPU cost cohort finishes.
