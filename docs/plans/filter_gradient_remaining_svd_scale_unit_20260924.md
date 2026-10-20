# Remaining inference SVD consumer scale diagnostic

The 28-call audit leaves dense-score singular values, COD condition reporting,
sequential score rank, block score ranks and quadratic score fitting to check.
The previous rectangular SVD repair does not qualify these consumers. This unit
asks whether their full numerical outputs preserve a known, well-conditioned
quadratic when the offset design is rescaled.

Use deterministic non-coordinate orthogonal frames, separated singular values
(3,1.5,0.7) and near-tied values (1+1e-8,1,1-1e-8), with magnitudes1,1e-4,1e-10,
1e4. The exact precision is diag(1.3,1.7,0.8). Exercise the existing native
dense, block, sequential and quadratic fit routines and the COD condition
helper, in explicit graph reference and enclosing XLA modes. CPU is an explicit
reference; GPU remains the default qualification target. These are fixed
diagnostic matrices, not stochastic performance samples.

Require relative/absolute1e-10 precision/condition agreement with the exact
quadratic and independent NumPy SVD, exact expected design ranks and accepted
statuses. Record every raw result and failure before asserting. Numerical
failures reject the tested implementation and trigger source attribution;
they do not justify a tolerance change. No ill-conditioned matrix is used.
The unchanged zero-ridge option isolates the homogeneous exact-quadratic
problem; it is not a new default or general regularization recommendation.

First run one CPU and one GPU worker under the stable campaign runner, with
registered remaining_svd_scale_cpu/gpu groups and 300-second bounds. Reserve
at most8 workers /1800 charged seconds under unchanged32CPU/52GPU caps for
the diagnostic, a localized harness repair and any subsequently specified
numerical repair checks. Save source/command/environment/device/growth and
elapsed time in the numbered master artifacts. One numerical worker at a time;
freeze source during each run. Preserve unexpected failures and stop unchanged
retries. Stop for invalid evidence, scope change or exhausted budget.

Do not modify numerical routines based on the search alone. A reproduced
consumer discrepancy must identify its SVD operation and preserve existing rank
cutoffs, ridge and derivative semantics in any proposed repair. Primitive scale
passes cannot replace complete original-record and public consumer renewal.
No Zhao-Cui behavior, target, sampler or scientific admission changes here.

Review: observing condition changes only in a helper would miss rank/precision
decisions downstream. Calling actual fitting routines tests those effects;
independent exact precision avoids treating graph output as automatically
correct. Near-tied singular values probe convergence while non-coordinate
frames prevent a diagonal fixture from passing without Jacobi work. Large and
small magnitudes test homogeneity without approaching binary64 underflow.
No inherited scientific threshold is reinterpreted. This is a primary-agent
review; broader call-chain and scale coverage remain open.

CPU03676 and GPU03677 reproduce the same failures at magnitude1e-10; every
graph arm passes. Block symmetric-design rank is5 rather than6 and rejects the
exact SPD quadratic. Quadratic precision errors reach2.04e-7. Near-tied COD and
dense-score conditions are1.0000000177313435 rather than1.0000000200000000.
These are well-conditioned matrices; the ill-conditioned reporting proposals
do not apply. Sequential rank/precision passes the tested spectra, which is
limited evidence and does not close its near-rank-cutoff behavior.

Source attribution identifies the unnormalized SVD in `_block_fit`,
`_quadratic_fit_kernel`, `_singular_values_xla` and COD `condition_number`.
Use the already-qualified ops/accurate_svd_tf.py operation at these calls.
It factors A/max(abs(A)) and rescales singular values, preserving the exact
SVD target, rank thresholds, ridge, statuses, clipping and derivative conventions.
The dense values-only pullback remains U diag(ds) V'. Keep explicit graph arms
on the original TensorFlow SVD. Do not alter the sequential routine without a
reproduced downstream failure. Thin output shapes must remain valid for tall,
square and wide reduced designs.

After the localized repair, repeat the unchanged16-case scale test on each
backend and run focused affected rank, precision, pullback/frozen-preparation
and endpoint checks. The two failed launches consumed27.586887seconds of the
8-worker/1800-second unit; six workers remain. Any broader full-record renewal
and matched costs must be separately scheduled inside the existing global caps.
Review: magnitude normalization repairs an algebraically homogeneous operation;
it does not authorize changing selection or treating a wrong original XLA
answer as a valid performance baseline. Preserve both failed reports and every
subsequent discrepancy.

Continuation after03682: scale qualification03678/03679 passes unchanged on
CPU/GPU, and46 derivative/COD cases pass per backend03680/03681. Endpoint
worker03682 aborts after47 test progress dots during
`test_center_refinement_accepts_nearby_mode`; pytest capture hides the native
fatal message. No completed endpoint or ownership qualification can be claimed.
Preserve this failure. First run only that test with capture disabled using
`remaining_svd_endpoint_crash_cpu`, then attribute the actual native error before
repair or GPU repetition. This is an infrastructure/compiler diagnostic, not
authority to weaken a numerical or collection check.

Revise this local allocation from8 workers/1800 seconds to16 workers/3600
seconds, including the seven completed workers03676--03682, within unchanged
global32CPU/52GPU-hour caps. The additional reserve covers native-error
localization, an evidence-based repair and focused endpoint/ownership renewal.
Review: splitting a captured crash is necessary because an unchanged large-suite
retry cannot reveal its cause; this keeps the same fixture, acceptance criteria,
runtime and resource class. Stop unchanged retries after a reproduced cause.

Isolated test03683 passes in30.881 charged seconds. This rules out a deterministic
single-fixture abort but not an order/resource failure. Next run the same complete
endpoint sequence with uncaptured verbose test output and the existing parent
process-memory observer (`remaining_svd_endpoint_sequence_cpu`,900-second bound).
It records mapping count, RSS, thread count and memory limits even if the native
child aborts. This changed observation is needed to distinguish retained native
executables from a new callback-ownership regression; neither cause is assumed.

03684 reproduces the abort at the same test. Uncaptured output identifies
`LLVM ERROR: Unable to allocate section memory!` from
`contiguous_section_memory_manager`. The last nonempty parent samples reach
60986 mappings against vm.max_map_count65530, about6.64GiB peak RSS,564 constant
threads and more than175GiB system memory available. This supports the known
native executable-mapping pressure, not a failed numerical assertion or exhausted
physical RAM. The1-second samples do not observe the exact failing allocation;
do not assert that a kernel mapping-limit failure was directly measured.

Reuse the existing small geometry/block test groups in separate fresh processes
to finish endpoint qualification. This is containment of the test process,
not evidence of native executable eviction. First run the existing
`geometry_fit_lifetime_cpu` case alone to check the new custom-gradient closure
without native mapping pressure. Preserve the combined-suite abort as a capacity
limitation, and keep the broader process-lifetime/capacity work open.

03685 confirms a separate ownership regression: both complete numerical records
pass, but the fit FuncGraph stays alive after its public function is deleted.
The callback and captured resource are collected. The newly inlined
`_xla_thin_svd` custom-gradient closure retains tensors from that FuncGraph in
TensorFlow's gradient registry. Isolate this resource-free primitive in a
shape-cached, explicitly signed XLA function traced under the existing
`independent_trace_scope`. Use the same scope for the dense values-only wrapper,
whose prior tf.init_scope does not detach graph ancestry. Preserve the entire
SVD primal and pullback; no gradient removal or numerical-policy change is
authorized. Existing owner-collection tests must pass without registry deletion
or global cache clearing. Renew scale, derivative and SRUKF endpoint checks
because the shared primitive boundary changes.

The observed native-memory abort requires split endpoint workers. Set the
revised local cap to32 workers/6000 charged seconds, including03676--03685,
inside the unchanged global caps. Allow two bounded numerical-regression
workers (CPU/GPU), separate ownership workers and block/public-geometry shards,
plus localized repair attempts. Each worker retains unchanged test selection
and assertions; the complete selection must cover the original failed group.
Review: the independent primitive owns no target/resource and reuses a static
shape signature. It preserves active derivatives while preventing each caller
from becoming permanent gradient-registry state. Source-derived ownership is
not enough: require executable weak-reference checks and gradient renewal before
cost measurements. These checks do not establish native executable eviction.

Post-repair CPU03687--03692 passes147 cases across all six shards. GPU
03693--03697 passes145 cases: four ownership,86 numerical/derivative,41 block
and14 public geometry checks. The final two-case GPU capacity shard03698 hits
the300-second supervisor deadline after one progress dot. It is an incomplete
run, not a numerical failure or qualification. The prior public GPU shard took
299.116 seconds, so this launch allowance was too short for compilation at these
capacities. Retry exactly the same capacity tests and source set once with the
existing900-second ceiling; reserve that time within the32-worker/6000-second
local allowance. Keep03698 and count its300.520 seconds. No further unchanged
retry if900 seconds is insufficient; inspect the blocked stage instead.

The unchanged retry03699 passes both capacity cases in492.685 seconds. The
completed qualification is147 pytest cases on CPU03687--03692 and147 on
GPU03693--03697/03699, including ownership, changed-input/derivative/SRUKF,
block and public geometry checks. Both backends bind the identical runtime and
harness closure; GPU2 uses verified growth. The local unit03676--03699 has
24 workers and2976.621635 charged seconds, within32 workers/6000 seconds.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain the four-consumer normalization repair | Independent scale/rank/precision checks and affected records pass | Preserved original small-scale errors repaired without changing tolerances | Additional shapes and near-cutoff sequential cases | Complete call-chain audit and matched costs | No full-repository or actual-DZ5 qualification |
| Retain the independent SVD tracing boundary | Four ownership cases pass on both devices and active pullbacks pass |03685 graph-retention regression is repaired | Shape churn/native executable retention | Capacity and process-lifetime follow-up | Python collection does not prove native eviction |
| Use fresh-process endpoint shards | Every original selected case passes on CPU/GPU | Combined LLVM allocation failures and300-second timeout remain preserved | Exact failing native allocation was not sampled | Keep compiler mapping pressure in capacity disposition | No unrestricted long-lived-process readiness |

Post-run review: a passing scale fixture could miss a derivative or ownership
change. The full pullback/SRUKF and weak-reference checks address those risks at
their tested shapes. The strongest remaining alternative is a failure near a
rank cutoff or under much larger shape churn; those observations would reopen
the affected consumer. A passing split suite cannot erase the combined-suite
failure, and this unit does not qualify the drafted cost harness or stale DZ5
snapshot. No numerical-loop/NumPy allowlist entry was added.
