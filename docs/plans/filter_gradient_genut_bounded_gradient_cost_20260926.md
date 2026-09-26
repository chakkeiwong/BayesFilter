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
