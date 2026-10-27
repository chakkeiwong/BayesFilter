# Design for a complete Stan Euclidean NUTS port

Status: source-grounded design answering the owner's question, October 27,
2026. No new transition kernel or benchmark campaign was executed for this
design. The existing standalone scope and production defaults are unchanged.

Subsequent execution is recorded in the
[reviewed execution plan](bayesfilter-full-stan-nuts-port-execution-2026-10-27.md)
and [result](bayesfilter-full-stan-nuts-port-result-2026-10-27.md). The status
above describes the original design stage, not the later implementation.

## Meaning of a complete transfer

The proposed deliverable is the Stan 2.37.0 Euclidean multinomial NUTS
transition, all three constant metrics, initialization on unconstrained
coordinates, step-size/metric adaptation, freezing and sampler diagnostics,
implemented in Python with TensorFlow/XLA numerical kernels for generic
TensorFlow targets. Completion must be judged against a source coverage matrix
and native behavior tests, not a label such as “Stan-style.”

Three different compatibility questions must remain separate:

| Compatibility | Proposed requirement |
| --- | --- |
| Algorithm and options | Same transition, stopping, selection, adaptation and supported sampler options as the pinned Euclidean Stan implementation |
| Randomized behavior | Same choices under supplied common random variates; native runtime may use TF stateless RNG with recorded per-chain state |
| Identical Stan seed streams | Separate compatibility work: Stan 2.37 uses Boost MIXMAX plus Boost distribution transforms and conditional RNG consumption; matching an integer seed alone does not establish this |

Bitwise CPU/GPU/native trajectories cannot be promised even with matching
random streams: reduction, transcendental and fused arithmetic can change a
decision near its boundary. The default fidelity claim should be algorithmic
equivalence plus deterministic common-variate tests and statistical validation.
If exact Stan seed compatibility is included in “whole,” the RNG/distribution
translation and corresponding tests are required before claiming completion.

Stan's language compiler, generated quantities, arbitrary model constraint
transformations, Stan Math distribution library, variational inference and
other samplers are separate systems. A generic target API owns its density,
Jacobian and parameter decoding. It can implement Stan initialization in the
unconstrained space without implementing Stan's model language. CmdStan remains
the way to execute the installed native Stan model implementation; invoking it
from Python does not make the sampler an XLA GPU kernel.

## Inspected source and missing work

All source paths below are relative to
.localresources/cmdstan/cmdstan-2.37.0/stan/src/stan.
The existing adaptation implementation and audit are retained:
[audit](bayesfilter-stan-adaptation-code-audit-result-2026-10-27.md).

| Layer | Source authority | Local work |
| --- | --- | --- |
| Transition and recursive tree | mcmc/hmc/nuts/base_nuts.hpp | Replace the TFP transition with a direct translation; preserve endpoints, proposal state, integrated momentum, weights and early returns |
| Unit/diagonal/dense metrics | mcmc/hmc/hamiltonians/{unit,diag,dense}_e_metric.hpp | Direct physical-coordinate kinetic energy, velocity and momentum draws; genuinely vector-only diagonal path |
| Leapfrog and target cache | mcmc/hmc/integrators/{base,expl}_leapfrog.hpp; hamiltonians/base_hamiltonian.hpp | Explicit value/score cache and correct retained proposal selection; domain-error status represented in tensors |
| Step search and jitter | mcmc/hmc/base_hmc.hpp | Reuse audited search mechanics, complete runtime jitter and option behavior |
| Metric/step adaptation | mcmc/{windowed,stepsize,var,covar}_adaptation.hpp; nuts/adapt_{unit,diag,dense}_e_nuts.hpp | Reuse audited arithmetic; add unit metric, supplied initial metric, disabled adaptation and complete schedule edge cases |
| Initialization and metric validation | services/util/initialize.hpp; validate_{diag,dense}_inv_metric.hpp | Supplied/zero/random unconstrained initialization, declared-coordinate partial initialization, retries and initial metric validation |
| Services and telemetry | services/sample/hmc_nuts_*; util/{run_adaptive_sampler,generate_transitions}.hpp | Match initialization/adaptation order, thinning semantics, nominal versus realized epsilon, tree depth, leapfrog count, acceptance, divergence and selected-state energy |
| Randomness | services/util/create_rng.hpp and Boost distribution code | Explicit per-chain state, source-order consumption in common-variate tests; separate native-seed compatibility option |

Some layer dependencies still need deeper inspection before implementation:
generate_transitions, unconstrained random initialization details, all CLI
option boundaries, Boost distribution algorithms and remaining Stan tests.
This design does not mislabel an uninspected dependency as audited.
Preserve Stan's BSD copyright/license notice for translated source, and audit
dependency licenses before copying any Boost implementation.

## The important transition details

The source uses two different proposal updates. Let W be an existing
trajectory weight and W_new the next complete subtree weight. At the outer
doubling level, replacing the current candidate uses

    min(1, W_new / W).

Inside a recursive merge with child weights W_left and W_right, the right
child candidate is selected with

    W_right / (W_left + W_right).

These are read directly from transition() and build_tree() in base_nuts.hpp.
Both weights are accumulated in log space. Replacing both rules by the same
generic multinomial merge changes the implemented transition. Their
conditional uniform draws also have different consumption patterns.

The stopping rule tests integrated momentum rho against endpoint velocities
p_sharp = C p. It performs three checks at merges: across the complete merged
segment and across both adjoining subsegment combinations. The same structure
appears inside subtrees and at the outer doubling level. A single endpoint
displacement check is not an equivalent transcription.

Invalid-subtree early returns, strict comparison signs, which proposal survives,
depth increments and accounting for work done in rejected subtrees all need
native tests. Acceptance is averaged over evaluated leapfrogs, including
rejected subtrees. Preserve the divergence criterion inside tree construction;
the owner's rule that divergence telemetry alone cannot reject a tuning
candidate does not remove an algorithmic tree stop.

## TensorFlow/XLA architecture

Use one implementation of metric operations, one transition implementation
and the shared adaptation arithmetic. Keep the target callable separate from
the sampler. A useful contract is an unconstrained FP64 batch of positions
returning row-independent values/scores and explicit numerical status.
An autodiff adapter can compute scores inside the compiled kernel; tested
analytic scores can use the same interface. No Python target callbacks,
NumPy runtime or pfor are required.

Translate recursive build_tree into an explicit depth-first stack inside
tf.while_loop. Each frame holds its phase, partial child result, proposal,
endpoints, rho and log weight. A fixed maximum depth bounds stack allocation,
while the loop exits dynamically on the same conditions as Stan. The design
target is O(d K) trajectory workspace for dimension d and depth K, excluding
the metric and saved chain outputs. A transition can still require up to
2^K-1 leapfrogs. Avoid Python-unrolling every potential tree node or allocating
the entire exponential tree. Confirm these complexity claims from the actual
traced graph and peak memory before accepting the implementation.

For inverse metric C, work directly in physical q coordinates:
velocity C p, kinetic energy pᵀ C p/2, momentum covariance C^-1. Unit and
diagonal metrics should need O(d) operations/storage. Dense metrics need
O(d²) matrix-vector work; cache their factorization while the metric is fixed.
This removes the current generic affine wrapper and its unnecessary dense
operations in diagonal mode. Retained position, value and score caches remain
in one coordinate system.

Numerical initialization, momentum generation, tree traversal, gradients,
warm-up and retained transitions must be inside stable-signature
tf.function(jit_compile=True) kernels. Host configuration, status checking,
serialization and reports stay outside. Validate compiled outputs explicitly;
TF assertions alone are insufficient under XLA.

First establish one-chain FP64 source fidelity. Then support batched chains
with per-chain RNG/state and masks that cannot alter another chain's kernel.
A separate chain consuming more random numbers must not advance its neighbors'
streams. Masked GPU work is a performance concern to measure. Single-chain
CPU, parallel CPU chains and batched GPU are execution layouts of the same
transition, not independently maintained samplers.

## Ordered implementation and verification

1. **Freeze the fidelity specification and oracle.** Pin source hashes and
   record license notices. Inventory options and expected edge cases.
   Extend the native C++ oracle to expose complete transitions, subtree
   validity, proposals, diagnostics and random consumption. Use Stan's own
   unit tests for rho aggregation, endpoint selection and divergence as
   additional cases; they are not a substitute for cross-implementation tests.
2. **Translate the transition without adaptation.** Implement metric operations,
   leapfrog and an iterative tree. Verify individual leaves, merges, backward
   expansion, exact stop boundaries, invalid leaves/subtrees and complete
   transitions. Start with small explicitly supplied momenta and uniforms.
3. **Complete adaptation and lifecycle options.** Attach the existing audited
   arithmetic, add missing unit/initial-metric/no-adaptation modes, jitter,
   initialization, zero/short-window cases, output thinning and exact diagnostic
   meanings. Include Stan's no-parameter service behavior in the compatibility
   inventory, even if the generic sampler reports it as a degenerate case.
   Verify each option against the native service, including failures.
4. **Validate enclosing CPU/GPU XLA.** Check input signatures, no callbacks or
   pfor, full-kernel HLO, device placement, memory growth, graph growth,
   allocator peaks and compilation costs. Only then optimize batching, memory
   reuse and RNG overhead, with semantic tests run after each change.
5. **Run a tuning-inclusive comparison.** Compare native Stan, the current
   audited Stan-adaptation/TFP route, the direct port and properly tuned fixed-L
   HMC using the repository's HMC tuning protocol. Charge each arm's full
   initialization/tuning/search, verification and retained sampling costs.

A common-random-input oracle must retain actual Stan control logic. Prefer
templated momentum/RNG hooks and inspection of overridable criteria.
If internal trace hooks require a diagnostic instrumented source copy, record
the exact instrumentation diff and first show that it reproduces the
unmodified native sampler. An independently rewritten Python “reference”
would not establish Stan fidelity. Random tapes are test inputs only; runtime
random generation must also compile within XLA.

Require exact discrete agreement on designed fixtures with arithmetic values
well separated from branch boundaries. Test ties and strict inequalities with
exactly representable scripted states. Record near-boundary FP differences;
do not conceal an unexplained branch mismatch by widening an epsilon tolerance.
The existing 2e-12 FP64 arithmetic allowance is an inherited test hypothesis,
to be checked for each new operation rather than treated as a universal error
bound. Preserve value/gradient caches and RNG state through chunk/resume tests.

## Research question and evidence contract for the later comparison

Question: does a direct source-faithful Stan transition reduce the measured
overhead of the current TFP-based implementation while preserving numerical
and posterior validity? The mechanism is direct metric handling plus a compact
compiled traversal, not a changed stopping rule or weaker adaptation.

Engineering promotion requires the complete declared feature/source matrix
and deterministic oracle/XLA checks. Scientific comparison requires fresh
target-specific posterior/reference diagnostics. Efficiency promotion requires
uncertainty evidence on elapsed time to a predeclared precision objective,
including tuning. A fixed-budget ESS/time study can be reported separately
with its narrower interpretation.

Initial targets should reuse the six existing models: isotropic Gaussian,
rotated anisotropic Gaussian, banana, centered funnel, noncentered control and
regression. Add at least one costlier existing XLA target only after checking
native Stan/TF density and gradient parity and securing an independent
reference or suitable posterior validation. Include both the previous
favorable starts and a separately declared overdispersed initialization study.
Settings adequate on the Gaussian cannot be silently transferred as tuned
choices for the funnel.

| Diagnostic | Role |
| --- | --- |
| Native oracle mismatch, invalid retained state/cache, wrong target or missing XLA proof | Engineering veto; repair trigger before posterior comparison |
| Required R-hat, ESS/MCSE or reference agreement fails | Posterior/efficiency promotion veto for that target; target-specific repair trigger |
| Time to declared precision, with tuning charged and replicated uncertainty | Primary efficiency promotion criterion after validity |
| Leapfrogs, gradient calls, per-step time, tree depth, divergence flags, masking and compilation time | Explanatory; divergence telemetry alone does not direct epsilon tuning |
| Invalid oracle/data, corrupted artifacts, resource violation or exhausted budget | Continuation veto for the affected execution |

No speed claim follows from transplanting the algorithm alone. Stan's C++
control flow and target-gradient implementation remain different from XLA.
Cheap low-dimensional targets may favor CPU despite GPU compilation; larger
batched gradient-heavy targets may favor GPU. Select a layout with calibration
and evaluate it on fresh runs. Report cold compilation and native build costs
separately as well as total workflow costs; do not count only leapfrogs.

Before any execution campaign, fill a bounded run manifest under a unique
versioned output root with exact executable commands, chosen model sizes,
seeds/replication count, diagnostic thresholds, total worker/attempt budget
and stopping rules. This question has not allocated a new compute campaign;
the previous audit budget is closed and must not be silently reused.
A failed centered funnel weakens the configured sampler/parameterization
claim, not the mathematical correctness of a port that passes its oracle.

## Skeptical design review

The main correctness risk is translating textbook NUTS rather than the pinned
implementation. The distinct selection rules, cross-subtree criteria, early
returns and random consumption above are explicit because each can change
the transition. A full transition oracle is therefore the first deliverable.

The main performance risk is preserving semantics but creating excessive
XLA state copying or masked work. O(d K) workspace is an architecture target,
not measured performance; graph/memory measurements must precede a speed
claim. Removing dense diagonal operations is a justified optimization target,
but its contribution to the prior timing gap has not been measured.

The main completeness risk is implementing missing features one at a time
while continuing to call the whole port “complete.” Completion requires a
closed source/feature matrix, including initialization, optional modes and
diagnostics. The known 2.37 short-budget fallback returns without restarting
its window counters: reproduce observed behavior in a compatibility fixture
or explicitly record a corrected deviation; never silently replace it with
an intended schedule while claiming exact release behavior.

The main inference risk is ranking a faster but poorly exploring sampler.
Validation precedes efficiency ranking, tuned fixed-HMC costs remain included,
and small-seed results remain descriptive without adequate uncertainty.
Framework equality is not assumed to imply target equality.

The design passes this source-level self-review as a roadmap. It is not an
implementation result, an independent review, an execution-ready benchmark
manifest or authority to alter production defaults.
