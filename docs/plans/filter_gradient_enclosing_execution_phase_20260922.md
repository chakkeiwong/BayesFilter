# Additional repair phase: complete execution boundaries and ownership

The owner requests execution of this phase after the diagnosis checkpoint
`ca920bac5`. It extends the existing master campaign; cumulative caps remain
32 CPU / 52 GPU process-hours. At run02636, 17.42 CPU / 38.21 GPU hours remain.
Use the same approved campaign runner and numbered artifact root, one numerical
worker, source freeze during workers and paired measurements, GPU UUID selection
with desktop protection, and verified memory growth. No new approval language,
package change, paid compute, score substitution or numerical tolerance change
is introduced. The separate canonical LEDH rebuild remains excluded.

The engineering question is whether the full numerical endpoint can execute
with TensorFlow/XLA, preserving the original algorithm, accepted outputs,
failure decisions, target call order and evaluation accounting, with understood
compile/memory costs. Original3582b4ac is the numerical authority on frozen
clouds. Current ca920bac5 is the mechanism/cost baseline. The approved versioned
TensorFlow stream is the RNG authority; historical PCG64 draws are not the
required seeded-output target. The rejected posterior precision rule continues
to require rejection, error reporting and no downstream use, without imposing
equality on a discarded matrix.

| Step | Change | Required evidence before closing the step |
| --- | --- | --- |
| E1. Active-row geometry and RNG | Carry training/holdout counts through fitting, reductions and rank diagnostics. Add a fixed-capacity permutation with runtime count and unchanged Philox words. | Complete original fit records on CPU/GPU at D1/D3/D5, changed active counts/inputs, rank and holdout boundaries, poisoned inactive rows, rejected invalid counts with no target calls, exact permutation parity, one enclosing XLA call with stable HLO/runtime count operands. |
| E2. Pilot and full geometry | Preserve exact active callback batch extents and scalar/batched ordering; connect direction/design/partition/fit/replay and iterative recentering. | Full public original payloads, seeded draw/call order, failures/early exits, target counts, compiled outer invocation, coupled changing-input tests. No padded target calls or hidden scalar fallback. |
| E3. Public controller integration | Wire native posterior and sequential controllers; compute report metrics/mass preparation in tensors; add ordered block recurrence. | Public endpoint tests, accepted/rejected records, progress buffering, budgets, asymmetric terminal behavior, coupled block replay/rollback and every supported configuration. |
| E4. Coordinated ownership | Bound callback-dependent program caches as one lifetime, preserving live compiled handles and useful reuse. | Distinct target identities, weak-reference collection after eviction, nested dependency release, continued execution of retained handles, same-signature reuse versus signature churn, native-residency attribution in fresh CPU/GPU processes. |
| E5. Consumer telemetry and supervision | Migrate actual DZ5 callbacks to tensor telemetry and independent parent deadlines before consuming buffered progress. | Real target/transition wiring, fixed18-batch behavior, invalid-row counts, healthy quiet and blocked workers, deadline termination/cleanup, exact source/pin provenance. Prepare external edits in an isolated checkout and obey its local policy and platform write permissions. |
| E6. Costs, audit, integration | Version GPU cost analyzers around physical UUID/provenance; renew signed-word direction costs and all affected public costs; finish F01–F20 endpoint dispositions, remote integration and terminal review. | Mixed-UUID/shared-device timing rejection tests, identical inputs/full outputs, original/current graph/XLA arms, two extents and three fresh-process repeats, full source-frozen tests, current remote integration/retest. Main merge only after every terminal gate passes. |

Dependencies: E1 precedes E2; sequential public integration precedes ordered
blocks; E4 may be implemented between completed E1/E2/E3 units. E5 must precede
promotion of progress-buffering consumers. E6 cannot close from internal
component parity or source counts alone. The master ledger remains open until
endpoint evidence is complete.

E1 implementation review: dispatch compact QR over the active sample count,
padding its Q factor only as an intermediate storage representation. Use one
shared small SVD outside row-count branches, then dispatch compact projections
and reductions so padding never becomes an observation. Preserve the original
`eps * max(active_design.shape)` rank threshold and do not substitute COD or
normal equations. Dispatch holdout reductions separately, avoiding a quadratic
cross-product of training and holdout counts. Existing configuration-only
`_active_row_operation` builds the finite branch table before numerical
execution; its exact static-topology exemption remains the only relevant Python
iteration exception. Numerical branches contain no Python iteration/NumPy.
Counts outside the declared domain return invalid status and cannot trigger
target evaluations; callers must not silently clamp malformed counts into valid
data. Retain fixed-extent behavior for its qualified consumers while adding the
count-bearing signature in the same fitter implementation.

Start E1 with 120/300-second CPU/GPU groups, no more than 30 qualification workers
including at most three localized retries per unchanged fixture. Reserve at most
12 additional 300-second fresh cost workers, for 42 workers total and a shared
7,200 charged-second ceiling in this first tranche. Compile/memory profiling
uses D3 capacities16/24 with identical active12 training/2 holdout rows, one cold
and20 synchronized warm calls, then changed13 training/2 holdout rows and20
alternating calls. Compare pinned ca920bac5 compact XLA against current active
graph and active XLA, CPU/GPU. Record the compact program's required second
shape compilation and the active program's reused HLO separately. Every arm
must match the original3582b4ac complete healthy records at unchanged tolerances.
These twelve single-process observations diagnose branch costs; terminal timing
ranking still requires E6 repeats. The broader phase uses only remaining campaign
budget; no attempt is allowed to reserve past a cumulative cap. Preserve failed
attempts. Diagnostic graph mode is explicitly non-default; XLA stays the
candidate default and GPU the intended target.

Keep existing investigation triggers: cold time above2x, repeat-median warm
regression above20%, device peak above2x, extra host RSS above256MiB or2x, and
continuing warm growth. These trigger diagnosis and repair, not relaxed
equivalence or automatic abandonment. No performance ranking from a single
process. Record trace count, graph size, compilation, RSS/PSS, GPU allocator
current/peak and device reservation separately. Collect HLO after timed/memory
snapshots. Do not infer native executable eviction from Python collection.

Stop the affected worker/arm on budget exhaustion, source mutation, unsupported
callback contract, unexpected numerical mismatch, missing provenance or corrupt
artifacts; repair the cause before resuming that arm. A failed candidate is not
evidence against the scientific direction. Acceptance requires original full
records at unchanged per-field tolerances, exact discrete decisions, no host
numerical feedback, and the declared public/cost gates. Diagnostic prototypes
cannot establish whole-repository compliance, HMC readiness or merge readiness.

Thorough pre-execution review: the earlier proposal of masked padding alone
would change the intercept, variance and rank cutoff; exact compact operations
are now explicit. Dynamic slicing can specialize HLO despite one trace, so
runtime operands and unchanged HLO are mandatory. Duplicating the entire SVD
and proposal controller per count would inflate compilation; share those pieces
and measure the remaining branch growth. Inactive NaNs must be excluded before
arithmetic, not merely multiplied by zero. Zero/invalid counts must not execute
a usable fit. Callback caches and native executable caches need separate tests.
Progress events cannot serve as a hard deadline. CPU results, one-extent costs,
target-only DZ5 qualification and a partial policy guard each leave their wider
gates open. No unexamined solver/threshold/RNG default change is needed. Proceed
with E1; reassess any concrete scientific/API change before adopting it.

Commands use:

```text
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py test --group <registered-group> --device <CPU-or-GPU> --test-timeout-seconds 300
```

Exact invocations, source hashes, seeds, environment, hardware and results are
recorded in the numbered run manifests under the existing shared campaign root.

Execution finding and reviewed E1 amendment:02637 passes four complete D3
original-record cases across changing counts;02638 passes seven edge/enclosure
checks but fails a newly introduced near-rank-threshold precision comparison.
The attribution in02639 shows the active and pinned ca920bac5 compact XLA
kernels produce identical complete raw records on all four fixtures. At two
samples and a2e-15 weak direction, both retain rank2 with condition about1e15
and return precision1.9165377 where the original NumPy answer is2.0. At16
samples the original dimension-aware rank cutoff drops that direction. This
is an inherited accepted-but-unreliable solve, not an active-row regression.

Under the owner's instruction to report ill-conditioning and the repository's
Class B fail-closed guard policy, add `fit_design_ill_conditioned` rejection
before covariance construction, refinement or replay in both fixed and active
fit routes. Use a scale/dimension-aware roundoff indicator
`eps * max(active_design.shape) * largest_singular / smallest_retained_singular`;
flag it when greater than`sqrt(eps)`. The threshold declares that the retained
solve may lose more than half the available significand digits under the
dimension-scaled roundoff model. It is an engineering guard, not a certified
forward-error bound. Singular directions already removed by the unchanged rank
rule are excluded; empty retained subspaces follow the original projected-fit
semantics. This does not certify identifiability of a rank-deficient fit.

This is a new rejection criterion, explicitly replacing the plan's unchanged
failure-decision requirement for this unreliable retained solve only. It changes
no accepted numerical result, rank cutoff, regularizer or comparison tolerance.
The error report must contain numerical rank, retained condition, indicator and
limit, while exposing no usable geometry and making no post-fit target calls.
Healthy original suites provide the required no-fire regression. Test just
above/below the declared relative guard margin, original/current error records,
active-count-dependent rank behavior, and CPU/GPU outcomes. Preserve the two
failed attempts. This repair requires no renewed campaign approval: the owner
already requested ill-conditioned errors and the local Class B policy directs
adoption after a no-fire regression. No broader condition waiver is authorized.

Recovery review:02640 passes all nine edge checks with the guard.02641's four
D1 cases match complete original records and counts, then fail an overly strict
test operand count: XLA legitimately removes the empty rank-zero basis. Count
nonempty operands plus the callback counter; retain the unchanged runtime-count
and stable-HLO requirements. This is a test repair only. The next frozen
qualification includes independent closed-form guard margins at1% above/below
the crossing, fixed/public healthy regressions and CPU/GPU runs.
02642--02644 pass all12 active-fit cases, including changing counts and complete
original records.02645 passes eight edge cases; the guard test then fails on
Python identity comparison with a NumPy boolean, before testing the new margins.
Convert the independently computed threshold to a Python float and retry only
the affected group. The runtime remains unchanged by either test correction.

E2 implementation review for the next dependency: add active-count behavior to
the existing cloud and pilot factories, keeping their fixed-shape API unchanged.
The scalar route must loop only over actual rows in original interleaved
plus/minus order. The batched route must dispatch a single call with exactly
twice the retained direction count, including the original zero-row call when
rank is nonzero. It must never use padded calls or scalar fallback. Preserve
the original all-plus-then-all-minus order for batched targets. Use one shared
curvature sketch/eigensystem/QR after the callback dispatch, with inactive rows
excluded and invalid counts prevented from calling the target. Host reporting
may trim completed records; it cannot feed a numerical decision back into the
program. Test counts0/1/intermediate/capacity, corrupted inactive rows, changing
centers/scales, invalid outputs and shape-sensitive callbacks against the full
original pilot records. Record graph growth before enclosing the complete
geometry controller. Shape-dispatch tests at small capacities do not establish
feasibility at production sample/direction capacities; that remains an E2/E6
requirement. Graph-reference and XLA arms retain their already-declared distinct
SVD/eigensolver implementations and must not be described as identical-graph
compiler ablations.

E1 result: see [numerical qualification and cost review](filter_gradient_active_rows_result_20260922.md).
The166 CPU/GPU checks establish the E1 dependency; shared-device timing leaves
GPU costs open under E6. E2's first pilot tranche allows at most20 numerical
workers (including at most three localized retries per fixture), 300 seconds
each and6000 charged seconds total, inside the unchanged cumulative caps. Run
`geometry_active_pilot_cpu`/`geometry_active_pilot_gpu` test matrices, stopping
the affected matrix on the first failure. The new tests include coupled batch
targets, so a silently padded callback cannot satisfy the comparison. This
tranche qualifies pilot execution only; full geometry and iterative public
controllers retain their independent E2 gates.

E2 attribution02675 resolves02674's failures. At one retained direction, the
original D3 sketch has eigenvalues about3.6323,4.8e-16,-4.8e-16. Its requested
second basis vector lies in an unresolved null eigenspace. Active and pinned
40f169fcc compact XLA bases are identical, and both differ from the original
basis by0.3193. Counts0/4/9 agree at ordinary floating-point error. This is an
inherited basis-identification failure, not padded callback evaluation.
Following the owner's ill-conditioning instruction and the Class B rule, add a
`pilot_eigenbasis_ill_conditioned` status before this basis can feed a fit.
For a nonzero positive sketch, use the smallest adjacent eigenvalue gap needed
to identify each retained eigenvector, including the retained/discarded boundary.
The engineering indicator is `eps * max(active_direction_count, dimension) *
max(abs(eigenvalues)) / minimum_required_gap`, limited by `sqrt(eps)`.
In a2x2 symmetric perturbation, `tan(2*rotation)=2*off_diagonal_error/gap`;
this motivates the relative gap check but does not certify an error bound.
The original zero-positive-curvature identity fallback and rank-zero pilot are
deterministic and retain their semantics. No eigenvalue, eigenvector or accepted
fit is regularized or changed. Healthy original tests and1% guard-margin cases
must pass; rejected pilots expose no usable basis in their report, and the full
controller must reject before design/fitting. This exception changes only the
unreliable-basis rejection criterion. Public wiring remains an E2 obligation.

The HLO difference is solely Grappler's nested
`StatefulPartitionedCall/zeros*/_N` dummy-source metadata suffix. Extend only
that diagnostic normalization; all instructions, constants, operand identities,
shapes and other metadata still require exact equality. Preserve the raw HLO
exports and02674 failure. No executable-reuse or numerical tolerance is relaxed.

E2 CPU qualification02676--02683 passes78 checks. GPU02684 stops at the first
group because the test fixture uses int32 resources, which TensorFlow places on
CPU even inside its GPU device scope. Use int64 recording counters, as existing
GPU fixtures do; no runtime numerical change is required. Preserve the failed
worker and rerun the affected GPU matrix under the same300-second deadlines.
The original20-worker tranche covers this retry and policy checks; any additional
full CPU renewal after the fixture-only correction belongs to the next bounded
integration tranche, not an unrecorded expansion.

E2 enclosing-controller review before implementation: compose the center target,
direction preparation, active pilot, design evaluation, seeded partition, fit,
incumbent selection and replay inside one stable-signature program. Pre-generate
the versioned CPU/XLA raw directions and ball offsets and prepare the permutation
seed in the original normal/ball/permutation stream order; these are runtime
operands, not captured numerical constants. Permutation consumes the finite
design count inside the enclosing program. Rank-zero preparation does not draw
unused directions. Preserve the original center-failure public accounting of
zero recorded evaluations (one attempted callback); record both counts explicitly
in native results so this legacy reporting behavior cannot hide actual work.

Preserve earliest strict-finite incumbent ties and logical indices: design
indices start after twice the retained direction count, regardless of pilot
storage capacity. Rejected pilots stop before any design, fit, refinement or
replay. Completed host records may format statuses and arrays but cannot feed
numerical decisions back to the program. Preserve legacy callback construction
failures through the already-qualified adapter; unsupported runtime callbacks
still fail closed without an eager retry.

Bound the fit branch table using the existing required-finite rule. For finite
count N, required count R and holdout fraction f, H(N)=min(floor(f*N),N-R).
For R<=N<=S, H and N-H are nondecreasing, so training counts lie in
[R,S-H(S)] and holdout counts in[0,H(S)]. This only removes unreachable branch
shapes; use the same compact QR, projections, rank threshold and reductions.
Keep general fitter defaults unchanged and reject invalid counts outside any
declared restricted domain. If S<R, compile the original insufficient-samples
exit without an unreachable fitter. Test all legal counts at small capacity
and public/default capacities separately; small fixtures cannot close compile
feasibility. Exact batch dispatch still scales with the pilot capacity and is
an explicit compile-memory risk, not a justified assumption of low overhead.

The next integration tranche allows30 workers,300-second focused jobs and at
most900 seconds for a predeclared public-capacity or consumer job, with a shared
7200 charged-second ceiling inside the cumulative caps. At most three localized
retries per fixture. Use original3582b4ac complete payloads on identical frozen
raw directions, clouds and permutation draws;40f169fcc is the current mechanism
baseline. Include changed centers/scales/counts, scalar and coupled batch
callbacks, center/pilot/design/fit failures, no-holdout and exact-tie cases,
runtime-operand/HLO checks, actual call logs and public consumers. Renew CPU
checks affected by the pilot fixture's counter-only change in this tranche.
Record timing/memory descriptively here; E6 retains the fresh-process repeated
cost gates. Integration is accepted only after these gates pass, not merely
after creating an internal factory.

Renewed skeptical review: the original source remains numerical authority,
while current-checkpoint attribution cannot certify correctness. Versioned RNG
changes are approved, but frozen-input comparisons remain mandatory. Pilot and
fit rejection exceptions are limited to the documented ill-conditioning guards.
No accepted tolerance, numerical regularizer or target changes. GPU correctness
may run on a shared eligible device; GPU cost comparison may not. Public
endpoint, native-memory ownership, external consumer and terminal gates remain
open. The bounded next implementation answers the execution-boundary question;
it cannot establish posterior validity, canonical LEDH status or merge readiness.

E2 pilot result:02685--02692 pass78 GPU checks after the fixture repair;
02693 passes102 policy checks. Together with02676--02683, the pilot has78
CPU and78 GPU passes. Twenty workers used in the pilot tranche. Static guard
215 sources /1306 existing exceptions; no numerical exemption added.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Qualify the active pilot dependency | Complete original records, exact calls/extents and HLO checks pass CPU/GPU | Ill-conditioned bases reject; invalid counts do not call target | Full/public composition and large-capacity costs untested | Enclose geometry and qualify its actual outputs/calls | No whole-endpoint, performance, HMC or merge claim |

Post-run review: component tests can pass while public code still uses the old
controller. That is the weakest part of current execution evidence and remains
an explicit E2 gate. Source guard counts also do not prove endpoint closure.
