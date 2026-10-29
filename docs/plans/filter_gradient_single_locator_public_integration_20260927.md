# Public single locator execution repair

After staged-locator qualification, wire `locate_joint_center` to the existing
`make_joint_center_program` numerical authority. The public API currently
constructs new optimizer closures and performs initial replay, endpoint replay,
incumbent selection and score summaries outside the optimizer's graph. The
native program already has original-record, accounting and rounding tests.
This unit changes execution and ownership only, preserving all settings,
callback order, caps, target values, derivative boundaries and returned fields.

Use `3582b4ac` with its complete frozen numerical imports as the reference.
The existing GPU reference helper records its four int64 accounting
substitutions; it must not be described as the untouched GPU original.
The public result formatter consumes only completed native tensors. Use one
retained program for a callback/configuration/dimension/device signature,
including receiver/function identity for Python bound methods and independent
dependency ownership. Starts and scales remain operands. Execute and format
under the program's invocation lock.

The explicit non-JIT host-clock diagnostic remains available under a named
private diagnostic, never as an execution fallback. Construction errors already
represented by the native program retain those result records; true compiler
or device errors propagate. Validate the public rank/finite/positive input
boundary before target use. No tolerance, optimizer, numerical model, NumPy
exception, pfor engine or scientific acceptance criterion changes.

The known internal callers are
`posterior_local_initializer.initialize_posterior_local_location_scale` and
`quadratic_map_covariance._run_locator`. Test their actual call chains as well
as the exported API. The posterior initializer still has an outer host
controller; repairing its locator alone cannot qualify that whole initializer.
The quadratic consumer already catches locator errors and records a fallback;
preserve and inspect that boundary instead of claiming all consumer exceptions
propagate. Unsupported target callbacks must not be silently called eagerly.

Reserve 20 CPU and 16 GPU workers, 7,200 combined charged seconds, inside the
existing global 56/52-hour caps. Use the same registered runner prefix and
fresh numbered output directories, normally 300 seconds per focused worker.
A documented compilation-capacity issue may use a 900-second retry within the
unchanged unit reservation. Run CPU correctness before GPU checks; observe the
existing selector and growth policy. Keep sources frozen during each matrix.

Acceptance requires public complete-original records and exact callback order
at D1/D3 on healthy, nonlinear, invalid, constant, cap and iteration-limit
cases; existing exported API and two consumer checks; unchanged frozen
derivatives; changed-input trace/HLO stability; callback/method/configuration
isolation and Python collection; compiler-failure/no-fallback evidence; policy
checks; and fresh original/graph/XLA public cost observations at two extents.
Costs include construction, completed records and host boundaries. Retain
graph/XLA rounding differences independently; compare each mode with its
original for same-mode numerical qualification. Do not rank failed arms.

Skeptical review: moving initial and final evaluations into XLA can change
finite arithmetic or exception timing. Full records, exact order and actual
consumer tests must expose that rather than comparing optimizer positions only.
Reusing a callback graph can capture mutable Python configuration; only tensor
operands/resource state may change between reuse calls. Changing static
configuration requires a different owner. A single retained Python program
cannot prove native executable eviction. Passing this unit cannot close the
outer initializer, actual DZ5 target, GenUT precision/reporting or terminal
F01--F20 gates. Preserve failures and localize the first mismatch before adoption.

Pre-execution review, September 27: staged qualification is now complete through
04292 and archived. The single-locator caller audit confirms the two internal
consumers above. The original record comparator, not endpoint distance alone,
is the engineering acceptance criterion. Existing numerical settings are frozen
for execution parity, not calibrated or promoted for any new target. D1/D3 are
bounded mechanics fixtures; actual DZ5 qualification stays separate. Cost
triggers inherit the staged unit: cold above 2x, warm above 1.2x, host RSS above
256 MiB extra or 2x, or GPU allocator peak above 2x requires attribution before
cost acceptance. A failed numerical arm cannot enter performance ranking.
Source/environment/input/reference drift invalidates a cohort and triggers a
fresh bounded retry; budget exhaustion stops new workers. Compiler failure is
tested with a real unsupported XLA operation and must not produce an eager
retry. These checks resolve the wrong-baseline, hidden-fallback and unfair-cost
risks sufficiently to execute the bounded unit. Results and raw manifests go to
fresh `run-NNNNN` directories under the existing campaign artifact root. No
independent reviewer has been used for this localized wiring change.

CPU correctness 04293--04301 passes 32 distinct cases, including both actual
initializer call chains and a real XLA compilation error. 04302 adds two
adjacent staged-versus-single assertions and renews the seven public API cases;
all nine pass, for 34 distinct CPU correctness cases. No runtime correction was
needed. The only intervening harness changes add those assertions and fix cost
import formatting; predecessor snapshots are preserved. Original replay tests
include original, changed and original operands and exact target ordering.
The AST audit against e56307029 confirms unchanged numerical factory, affine
rounding, formatter, input validator, configuration and result schemas.

Proceed to six CPU costs, policy, then the same GPU correctness and six cost
arms using the selector. The existing 20/16-worker and 7,200-second reservation
covers this sequence (17 CPU and 15 GPU planned, including the API renewal).

Recovery correction through 04324: standalone 04302 and policy 04309 were
launched without `--device CPU`, so the CLI's documented default selected GPU.
Their manifests and charges are correct; earlier CPU labels and hand-summed
CPU charges were wrong. Run the adjacent API assertions explicitly on CPU and
renew policy after the launch-guard fix. CPU correctness before that retry is
32 distinct cases; GPU correctness 04310--04318 passes all 34.

All six GPU costs preserve same-mode original records, but a foreign compute
process is present in every preflight and in-run monitor. The independent
analyzer correctly rejects timing eligibility. The new public staged/single
cost groups were omitted from `require_unshared_cost_preflight`; add both
registered batches and test every new arm. This preserves existing admission
criteria and prevents wasting workers on known shared-device costs. Keep all
failed timing evidence. A later uncontended six-arm cohort is required before
GPU cost acceptance; correctness evidence remains usable.

Allow up to 24 GPU workers for this unit, including the two incorrectly labeled
standalone GPU workers and one full six-arm cost renewal. The 20 CPU-worker and
7,200-second combined reservation and global 56/52-hour caps are unchanged.
This local harness repair/retry changes no target, data, method, criterion,
hardware class or campaign compute budget. If compute GPUs remain occupied,
continue the independently authorized CPU enclosure unit and retain this cost
gate; do not use a desktop GPU or weaken uncontended timing requirements.
