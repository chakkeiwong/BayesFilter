# Bounded GenUT gradient memory and execution costs

The two GenUT recurrences now declare their existing configured limits for XLA
reverse storage, with a minimum one-slot allocation for the zero-trip compiler
case. Their bodies, conditions and numerical controls are unchanged. The
original XLA differentiated program failed compilation in 04171 and cannot be
used in speed ratios. Compare the working original graph at `28cbdb536`, the
repaired graph, and the repaired XLA program in fresh processes. The uninstalled
highest-dot candidate is outside this cost cohort; its earlier value-only
timings do not establish the cost of its repaired pullback.

Use exactly the saved rounded FP32 operands from 04114 (N=1,000,d=3) and 04122
(N=10,000,d=18), four diagonal and four pairwise iterations, and the complete
reduced-primal value plus the same smooth scalar functional and derivatives
used in the bounded-derivative tests. There is no canonical LEDH score, reset,
particle-filter capacity or HMC claim. Saved inputs are diagnostic fixtures;
the executable kernel uses only TensorFlow. Record input/source hashes and all
outputs. The function takes stable operands and must trace once, produce
finite derivatives and valid particles, and replay exactly.

For each arm/extent/backend, record process RSS/PSS and high-water memory before
building, after the first synchronized call, after ten synchronized reuses and
after Python owner collection; record TF GPU allocator current/peak bytes and
sampled GPU sharing separately. Timings include transfer/materialization of
the complete result in every arm, with array equality checked outside timing.
Export the XLA HLO size/hash after timing. Do not interpret retained native
allocations as a Python-reference leak or NVIDIA reservation as live tensors.
No cost assertion hides raw results: this is an explanatory cost cohort, and
numerical qualification/interpretation happens before any speed comparison.

Compare complete original/repaired graph records at unchanged FP32 2e-5 bounds.
Compare repaired XLA records and derivatives to that graph with the known
thresholded report disagreement explicitly recorded. If any smooth numerical
field disagrees, use independent FP64 graph values/derivatives on the identical
rounded inputs to identify the erroneous comparator; do not relax tolerances
or rank a rejected arm by speed. Reserve two such CPU references up front.
Original-XLA failure is recorded as unavailable, not a zero-time baseline.

Run `genut_transitive_gradient_cost_<arm>_<count>_<dimension>_<device>` via
`/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test`
with explicit CPU/GPU and a 300-second timeout. Arms are `before_graph`,
`after_graph`, `after_xla`; reference groups are
`genut_transitive_gradient_cost_fp64_<count>_<dimension>_cpu`.
Reserve eight CPU and six GPU workers / 2,400 CPU and 1,800 GPU seconds within
the unchanged 56/52-hour campaign caps. This allocation is separate from the
pullback correctness unit. All outputs use fresh numbered campaign directories;
trusted GPU runs use the existing device selection, growth and provenance
mechanisms. No source mutation while a worker runs.

Skeptical review: one fresh process per arm/extent is a descriptive pilot, not
an uncertainty-supported performance ranking. Large compiler RSS is not live
tensor storage, and GPU process sharing invalidates timing comparisons. A
>15% warm-time or >25% allocator/RSS increase nominates attribution; it cannot
be waived by successful values. Unbounded original XLA derivatives cannot be
compared by inventing a runtime or assigning zero cost. Adding a loop bound
could still alter compiler lowering; same-mode complete values, exact replay,
independent FP64 checks and saved HLO make that observable. Stop on source/input
drift, invalid reference, or exhausted allocation; preserve ordinary candidate
failures for repair rather than promoting them.

Renewal after the first cost cohort found that its JSON summary retained finite
and replay checks but omitted the materialized arrays and a direct comparison to
the newly generated FP64 references. That is an artifact completeness gap, not
evidence of a numerical failure. The renewal adds compressed output arrays,
SHA-256 hashes, per-field maxima and unchanged `2e-5` comparisons against the
FP64 graph references on the same saved FP32 operands. It repeats the six
before/after graph/XLA arms at both extents on CPU and GPU, using fresh process
directories and the same memory/timing contract. The comparison is descriptive:
any failed field remains a veto on ranking that arm and does not alter the
runtime or tolerance policy. The bounded extension uses at most six CPU and six
GPU workers, each under 300 seconds, within the unchanged global caps.

Review after 04244 rejects 04217--04242 as the planned numerical/cost cohort.
The pilot harness used `cos(index)`, not the declared frozen FP32
`cos(index * .11)`, and recomputed those coefficients in FP64 for the reference.
It therefore compared derivatives of different finite objectives. Its recorded
FP64 failures cannot establish the size of the runtime error. The earliest
pilots also omitted full arrays, and HLO inspection preceded collection without
a separate memory checkpoint. Preserve all pilots, but do not promote their
speed ratios, post-collection memory interpretation or numerical comparisons.
04216 was a collection-selector failure. 04243 found missing default GPU group
registration; 04244 passes all 129 checks after registration. Explicit GPU flags
in the pilots and their device records did select actual GPUs.

The corrected cohort uses one shared differentiated function with an explicit
coefficient operand, generated on CPU in FP32 exactly as in the precision
diagnostic and cast without recomputation for FP64. Save its bytes/hash and
the same rounded input hashes in every arm. The independent FP64 graph
reference must pass centered finite differences at two step sizes, separately
for source, weights and reset. Preserve complete arrays, loss, gradients, exact
replay and all report fields. The post-run analyzer verifies artifact identities
and recomputes every comparison from arrays, keeping the cap fraction separate
from smooth values/gradients without waiving its unresolved gate.

Record memory immediately before and after HLO inspection, release its string
and all returned arrays before owner collection, and record Python owner
collection separately from native memory. Save the HLO file itself. The
uninspected after-replay checkpoint is the primary steady-memory observation;
post-HLO/collection values cannot establish compiler eviction. No timing ratio
is interpreted for an arm that fails independent smooth values or gradients.

Reserve ten CPU and six GPU workers, each at most 300 seconds, under the
unchanged total caps: two FP64 references, twelve corrected cost arms, two
analysis/policy workers. Actual registered names are
`genut_bounded_gradient_fp64_<N>_<d>_cpu` and
`genut_bounded_gradient_cost_<arm>_<N>_<d>_<cpu|gpu>`; explicit devices remain
required. Freeze sources for this entire corrected cohort. A short harness
repair can be retried under the unused total reservation with its failure
recorded. This changes neither numerical controls nor scientific acceptance.

Skeptical review: finite/replay tests alone cannot certify cost comparability.
An identical coefficient hash, original source, fixed controls, independent
derivative check and array-level analysis are prerequisites. HLO extraction
can allocate additional memory and must not be mistaken for a collection leak.
The corrected design addresses those specific defects; the preexisting GenUT
precision and report gates remain open even if its cost worker completes.
