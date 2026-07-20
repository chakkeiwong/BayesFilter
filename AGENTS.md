# BayesFilter Agent Governance

## Academic Research Governance Profile

Owner directive, 2026-07-13: BayesFilter is a trusted local academic research
repository. Governance must prioritize scientific validity, reproducibility,
bounded compute, and research progress. It must not import production-service
security ceremony without a concrete threat that applies to the work.

This section supersedes older BayesFilter plans, runbooks, and implementation
notes where they require procedural controls that are stricter than this
profile. Preserve historical artifacts, but do not treat their old launch gates
as active authority requirements for new work.

### Proportional Risk Model

The default threat model is accidental error in a trusted local workspace, not
an adversarial multi-tenant service. The controls that remain strict are:

- scientific question, baseline, success criteria, vetoes, and nonclaims;
- mathematical, numerical, and statistical validity;
- exact commands, environment, seeds, hardware, wall time, and artifact paths
  for serious runs;
- unique versioned output directories that do not overwrite prior evidence;
- bounded compute and attempt budgets;
- secrets, private data, external publication, package/environment mutation,
  destructive operations, and other irreversible or externally visible acts;
  and
- platform or sandbox permissions that the agent cannot waive.

Absent a documented concrete adversarial risk or an explicit user request, do
not create or require:

- hash-bound natural-language approval statements;
- one-use authority, launch-claim, or approval-token files;
- inode, descriptor, hard-link, immutable-empty-file, or crash-durable output
  reservation protocols;
- custom cryptographic schemas when Git provenance, ordinary SHA-256 checksums,
  and versioned artifacts answer the research-integrity question;
- separate approval for each local retry when the scientific contract and
  campaign budget are unchanged; or
- mandatory review chains for each proposal, manifest, subplan, result, and
  handoff.

These mechanisms may be used only when the plan identifies the applicable
threat, explains why ordinary academic reproducibility controls are
insufficient, and shows that the added ceremony is proportionate.

### Execution Tiers

1. Routine local work: implementation, unit tests, focused diagnostics, import
   checks, and short smokes need no special governance approval beyond the
   user's task and normal tool permissions.
2. Serious local research campaigns: use one concise experiment plan with an
   evidence contract, compute/attempt budget, versioned output root, and stop
   conditions. A plain-language user request to execute or resume that plan is
   sufficient campaign authorization. No magic wording or manifest hash is
   required.
3. External or irreversible work: require explicit human approval for public
   release, external messages or publication, credentials/secrets, paid or
   materially expanded compute, package/environment changes with broad impact,
   destructive operations, or a material change in scientific/product
   direction.

Within an authorized serious campaign, a localized infrastructure or harness
failure may be repaired and retried automatically when all of the following
remain unchanged: scientific target, data, method, promotion criteria, vetoes,
hardware class, privacy boundary, and total campaign budget. Record the failure,
repair, and retry in the campaign result. Reapproval is required only when one
of those boundaries changes.

### Review Proportionality

Review is advisory by default. Use it where independent scrutiny materially
reduces scientific or engineering risk, especially for source-faithfulness,
publication-grade claims, major API/default changes, expensive campaigns, or
user-requested review. One material plan review and one terminal result review
are normally enough. Reviewer unavailability or a procedural documentation
disagreement must not block trusted local research when the scientific plan,
focused checks, budget, and artifacts are adequate; record the limitation and
continue.

This profile does not weaken any scientific evidence standard below. It
separates scientific rigor from launch-security ceremony.

## Contract E Canonical LEDH Reset Policy

Owner directive, 2026-07-13: Contract E--Chol is the only LEDH reset semantics
eligible for canonical value, total-gradient, score-admission, leaderboard,
default-readiness, or HMC-facing status. The frozen reset identifier is
`contract_e_chol_v1`. The frozen total-derivative composition identifier is
`contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.

Raw-barycentric reset routes are historical diagnostics and are wrong relative
to this canonical target. This includes compact forward-sensitivity routes and
full-history/manual reverse routes, even when their derivative is algebraically
correct for the raw-barycentric scalar. They must emit or normalize to
`historical_raw_barycentric_diagnostic_only` and must never be used as a fallback
when Contract E is unavailable or fails a gate.

All v1 LEDH forward and score artifact schemas predate reset identity and cannot
prove which reset callable executed. They are therefore ineligible for canonical
admission, default selection, leaderboard contribution, and HMC-facing use.
Existing v1 artifacts remain readable as historical evidence but must not be
silently upgraded. Adding caller-supplied fields such as `reset_contract_id` to a
v1 payload does not make it admissible.

Canonical identity must be issued by a non-overridable repository-owned route
factory from the actual callable and settings. Callers may not stamp, override,
or self-attest canonical route identity. The identity must bind at least reset
semantics, row-mass normalization, fixed residual design, ridge policy and
realized input, derivative composition, prepared-input identity, and source
dependency closure.

The claimed canonical gradient is the total derivative of the same finite
value program. Contract E depends directly on source-cloud moments and weights
as well as on the transported cloud, so its pullback must include
`G_X = G_X^moments + G_X^transport` and
`G_w = G_w^moments + G_w^transport`, plus any declared residual-design or ridge
dependence. A transported-cloud-only adjoint is a partial derivative and is
wrong relative to the canonical total-gradient claim.

This policy selects the only route eligible to seek admission. It does not by
itself establish implementation correctness, covariance restoration, Kalman
agreement, nonlinear validity, production feasibility, HMC readiness,
leaderboard completeness, or scientific validity; those remain evidence gates.

## LEDH Per-Scope Tuning Policy

Owner directive, 2026-07-19: every claim-bearing LEDH run must have an offline
tuning step for its own execution scope before its final claim or leaderboard
run. There is no universally applicable LEDH numerical or feature setting.

A tuning scope binds at least the model/target identity, LEDH route and reset
family, observation horizon and prepared-data regime, particle count, state and
parameter dimensions, dtype/TF32/backend, chunk policy, and the complete family
of tunable controls used by that route. A change to any bound field creates a
new tuning scope. In particular, a setting selected at one horizon, such as
`T=10`, is not tuned for another horizon, such as `T=50`, even for the same
model.

Prior settings from another model or scope may seed the first candidate only.
They are warm starts, not defaults or promotion evidence. Each route must tune
its own controls: annealed-Sinkhorn/terminal-balance counts for the streaming OT
route; feature, basis, lookahead, chart, ridge/KKT, or other applicable controls
for Contract E--TP and other LEDH routes. Do not force unrelated routes into a
shared parameter vocabulary.

The tuning procedure must use disjoint calibration/validation data and freeze
the selected controls before an untouched claim run. The claim run must consume
a repository-issued tuning artifact whose scope identity exactly matches the
claim scope; fail closed on a missing, stale, mismatched, caller-stamped, or
cross-model artifact. Runtime or parameter-dependent retuning inside HMC remains
forbidden.

A failed tuned candidate or untouched claim is a repair trigger for fresh
scope-specific tuning under the remaining campaign budget. It is not a reason
to transfer a setting, relax a gate, tune on the failed claim data, skip tuning
for later models, or reject the LEDH research direction. Preserve the failed
claim as holdout evidence and use fresh tuning partitions.

## DPF Transport Chunk Policy

Owner directive, 2026-07-18, supported by the June 24 GPU/XLA exact-tiling
artifacts: active DPF transport canonical, candidate, benchmark, leaderboard,
and production-target routes must use policy
`dpf_transport_exact_divisor_cap3000_v1` from
`bayesfilter.highdim.transport_chunk_policy`.

For particle count `N` and row/column chunk extent `K`:

- row and column chunks must be equal;
- `K` must divide `N`, giving an exact `(N/K) x (N/K)` block grid;
- for `N <= 3000`, `K=N`, so the grid is `1 x 1`;
- for `N > 3000`, `K` is the largest divisor of `N` not exceeding 3000; and
- if no divisor greater than 1 exists under that cap, fail closed. Never fall
  back to a tiny fixture chunk or an independently chosen local default.

Required examples are `N=1000 -> K=1000`, `N=1024 -> K=1024`,
`N=10000 -> K=2500`, and `N=10240 -> K=2560`. Selection and validation happen
at configuration time before TensorFlow tracing; do not implement the selector
inside an XLA graph.

All contrary chunk policies, constants, CLI defaults, plans, and results are
historical and wrong relative to this policy. Historical files remain preserved
only as archival provenance. They are not eligible even as diagnostic,
comparison, tuning, timing, admission, or scientific evidence for a new run.
Primitive unit tests may use small arrays only when they still obey `K=N`; that
tests mechanics and does not define an alternative chunk policy.

The repository-owned selector is non-overridable for active canonical and
candidate routes. Caller-supplied row/column values must be validated against
it and rejected on mismatch. A new chunk rule requires a new reviewed policy
identifier and evidence; editing a benchmark constant or copying a historical
fixture cannot change this policy.

## Default Implementation Backend

BayesFilter algorithmic implementation defaults to TensorFlow and TensorFlow
Probability.

## Default Execution Target

The BayesFilter repository default execution target is GPU.  CPU-only execution
is allowed for explicit reference checks, small smoke tests, debugging, and
sandbox-safe diagnostics, but it must not be described as the default
production target unless a reviewed plan explicitly changes this policy.

### TensorFlow GPU Memory Allocation Policy

Owner directive, 2026-07-14: BayesFilter TensorFlow GPU processes must disable
eager whole-device memory reservation. The default configuration is memory
growth on every visible physical GPU, applied and verified before TensorFlow
creates a logical GPU or initializes any GPU tensor, operation, or compiler
context.

- Prefer setting `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and
  also call the repository memory-policy helper to set and verify growth on
  every visible physical GPU.
- Serious, candidate, benchmark, training, HMC, and production-target GPU runs
  must fail closed if memory growth cannot be configured or verified before
  device initialization. Do not silently catch and ignore the failure.
- Serious run manifests must record the memory-policy schema/mode, every
  physical device and its verified growth value, and whether a logical-device
  memory limit was used.
- Memory growth prevents TensorFlow from reserving almost all device memory at
  startup. It is not a hard cap: TensorFlow may grow to use most or all
  available memory and normally retains allocations for process-local reuse.
- When a run must guarantee memory remains available to another process, use a
  reviewed logical-device `memory_limit` configured before initialization.
  TensorFlow does not allow memory growth and virtual-device memory limits on
  the same physical device, so the manifest must identify this explicit
  `logical_device_memory_limit` exception instead of claiming growth.
- Exclusive whole-device preallocation is non-default and requires a reviewed
  exception stating why growth or a logical-device limit is unsuitable. A
  performance preference alone is not sufficient to make silent whole-device
  reservation the repository default.

Historical GPU artifacts remain valid under the policy active when they were
created, but they do not prove the new memory-policy field. New serious GPU
artifacts must record it.

## NeuTra Execution Target Policy

BayesFilter NeuTra training is a GPU workload by owner directive.  Any
BayesFilter-owned learned NeuTra transport training, including affine,
dense-IAF, normalizing-flow, or future learned transport families, must plan for
GPU execution by default and must run with trusted/escalated GPU access under
the local GPU/CUDA sandbox policy.

CPU-only NeuTra training is allowed only as an explicitly labeled tiny smoke,
reference, or sandbox-diagnostic exception under a reviewed plan.  Such an
exception must not be described as the default, serious, production, or
preferred NeuTra training route, and it must not support claims about learned
transport quality, HMC readiness, posterior correctness, production readiness,
or scientific validity.

### NeuTra HMC Sequential Sampling Default

Owner directive, 2026-07-15: the canonical claim-bearing NeuTra HMC policy is
`bayesfilter_neutra_sequential_hmc_v1`. New serious, confirmatory, posterior,
robustness, or default-readiness NeuTra HMC routes must use the shared
TensorFlow/TFP sequential controller under `bayesfilter.inference.neutra_hmc`.

- Retain and archive every warm-up chunk, but exclude all warm-up draws from
  posterior estimates.
- Check warm-up readiness on a predeclared recent window using the maximum of
  rank-normalized split and folded rank-normalized split R-hat. The default is
  at least 2,000 warm-up transitions per chain, latest 1,000-transition window,
  threshold `<=1.05`, and maximum 10,000 per chain.
- Grow retained sampling cumulatively. Tuning admission uses modern R-hat
  `<=1.01`; confirmation additionally uses declared bulk/tail ESS and downstream
  posterior gates. The retained maximum is 10,000 per chain.
- Apply finite state/target/log-acceptance, target-status, all-chain movement,
  and declared energy-error vetoes to every chunk. Acceptance is nomination or
  explanation only, never convergence evidence.
- Fixed discarded burn-in and fixed terminal sample counts are allowed only in
  explicitly classified historical, mechanics, smoke, reference, or debugging
  routes whose nonclaims forbid convergence and posterior promotion.

The committed NeuTra-HMC route ledger and its discovery/enforcement test are a
persistent default guard. Every qualifying repository-owned Python route must
be classified exactly once. Unledgered routes, stale or duplicate ledger paths,
and active routes that bypass the shared controller are test failures.

### NeuTra Batch-Native Training Requirement

Owner directive, 2026-07-14: all BayesFilter NeuTra training must be batched.
Every optimization update must consume a batch with more than one sample, and
the transport forward/log-determinant computation, target value/score
evaluation, loss reduction, gradient calculation, and optimizer update must
preserve that batch dimension in TensorFlow/XLA.

A Python loop over samples, repeated scalar target calls, or a `tf.map_fn`,
`tf.vectorized_map`, or `tf.while_loop` that merely applies a scalar target
implementation independently to each training row does not by itself satisfy
this requirement. An eligible target route must use batch-native tensor and
linear-algebra operations across the leading sample dimension, or explicitly
shard the batch across persistent multicore workers where each worker evaluates
a batched target shard. The latter is a CPU value/score generation lane feeding
GPU transport training, not permission for scalar workers.

Scalar and row-mapped target routes may remain as independent parity authorities,
tiny reference diagnostics, or debugging aids, but they must not update NeuTra
parameters and must not be described as training routes. A smoke or CPU-only
exception remains subject to the batching requirement if it performs an
optimizer update.

Serious NeuTra plans and result artifacts must record the training batch size,
batch-native target backend, device placement, XLA status, and whether any
sample-wise loop or scalar fallback was used. A scalar fallback, batch size of
one, or row-mapped scalar target is a hard training veto. Existing NeuTra paths
that violate this rule are migration debt and are ineligible for new serious
training until repaired; historical artifacts remain historical evidence.

NeuTra sample generation is a separate execution lane.  Pre-generating replay
samples, target/evaluation samples, proposal clouds, or training datasets should
use multicore CPU parallelism by default, with worker count, seeds, target
signature, and output artifact hashes recorded.  Independent CPU sample
generation must not be conflated with GPU NeuTra training.  In-graph random
noise needed inside a GPU training step may remain part of the GPU training
graph, but external sample/dataset generation should be planned as multicore
CPU work unless a reviewed plan justifies otherwise.

For DPF transport work, the default production algorithm target is the
GPU-oriented LEDH-PFPF-OT TF32 route: TensorFlow/TFP implementation, `float32`
tensors, TensorFlow TF32 execution enabled, streaming/chunked transport where
applicable, and explicit FP64 or FP32-no-TF32 only for reference/comparison
arms.  The historical module path under `experiments/dpf_implementation` is not
a reason to demote this route; future agents should treat the current owner
directive as the default production direction and should avoid reopening the
default-vs-experimental question without new evidence or human instruction.

This default-policy promotion is a product and engineering direction.  It does
not by itself certify posterior correctness, HMC readiness, statistical
superiority, dense Sinkhorn equivalence, or broad scientific validity.  Those
claims still require their stated evidence gates and artifacts.

## NumPy Diagnostic-Only Policy

Owner directive, 2026-07-14: BayesFilter code must not import or use NumPy
outside an explicitly diagnostic or independent-reference role. TensorFlow and
TensorFlow Probability are the array, numerical, statistical, training,
inference, and algorithmic backends for this repository.

NumPy is permitted only in code whose diagnostic status is explicit in its
path, name, or module documentation, including:

- tests and comparison fixtures;
- independent reference solutions and closed-form or finite-difference checks;
- post-run inspection, reporting, and diagnostic benchmark analysis; and
- historical diagnostic readers that cannot affect runtime decisions.

NumPy is forbidden in production or candidate runtime paths, training and data
pipelines, inference and tuning implementations, selection or admission logic,
artifact/manifest construction, and executable benchmark kernels. Serialization
and ordinary reporting are not blanket exceptions: use TensorFlow operations and
Python standard-library types instead. A TensorFlow tensor may be materialized
at a host-side assertion, diagnostic, or artifact boundary, but the materialized
value must not become a NumPy numerical-computation path.

Diagnostic/reference NumPy code must not be imported by an admitted runtime
path, and its outputs cannot establish production, default-readiness, HMC, or
scientific status without an eligible TensorFlow/TFP computation. A narrowly
reviewed exception requires explicit owner approval; recording an exception in
a plan alone is insufficient.

Existing non-diagnostic NumPy imports are migration debt, not precedent. Do not
add new violations. When modifying a legacy non-diagnostic module, remove its
NumPy dependency from the touched execution path or record and execute a bounded
migration before promoting that path.

Differentiable or gradient-bearing implementation paths must use TensorFlow /
TensorFlow Probability unless a reviewed plan explicitly authorizes another
autodiff backend.

PyTorch and JAX are non-default backends for this repository.  They require a
reviewed exception before use in BayesFilter-owned algorithmic implementation
paths.

## Evidence Discipline

Reference, comparison, smoke, and diagnostic-reporting code must not be
represented as the BayesFilter default implementation. If an explicitly
diagnostic lane uses NumPy for a reference or comparator, the artifact must say
so and preserve the gap to a TensorFlow / TensorFlow Probability implementation.
NumPy prototypes are not eligible runtime candidates.

GPU-oriented LEDH-PFPF-OT TF32 is now the default production target by owner
directive.  Evidence artifacts may still record unresolved scientific or HMC
gates, but they should not downgrade the default target back to "no production
default" merely because those separate gates remain open.

## XLA JIT Default Policy

Owner directive, 2026-06-26: BayesFilter-owned TensorFlow/TensorFlow
Probability algorithmic, differentiable, gradient-bearing, benchmark, and
production-target execution paths must default to XLA JIT compilation
(`tf.function(..., jit_compile=True)` or an equivalent project API option that
defaults to true).

`--no-jit-compile`, `jit_compile=False`, eager-only execution, or graph mode
without XLA is allowed only as an explicit non-default exception for reference
checks, small smoke tests, debugging/localization, sandbox-safe diagnostics, or
a reviewed artifact that records why XLA is not being used.  Such runs must not
be described as the BayesFilter default execution path, default-readiness
evidence, production-target evidence, or a replacement for the GPU/XLA route.

New CLI harnesses that expose a JIT switch must default to JIT on.  If a
non-JIT escape hatch is kept, artifacts must record `jit_compile`, and result
notes must label non-JIT runs as debug/reference exceptions.

## Managed-Session GPU Trust

Owner directive, 2026-06-25: visible non-elevated GPU runs in the managed
BayesFilter Codex session are trusted BayesFilter GPU evidence when all of the
following hold:

- the run uses the repository TensorFlow/TFP GPU/XLA path;
- GPU visibility/provenance is recorded in the artifact;
- TF32/XLA/device settings are recorded in the artifact;
- the command writes structured JSON/Markdown/log artifacts under the reviewed
  plan;
- the artifact states the trust basis as
  `owner_designated_managed_session_visible_gpu_trusted`;
- no package install, network fetch, destructive git operation, model-file
  edit, public API/default-policy change, HMC runtime, or scientific/default
  promotion claim is smuggled into the run.

This directive resolves the local execution-boundary question for BayesFilter
GPU benchmark artifacts.  It does not lower the scientific evidence bar:
posterior correctness, HMC readiness, statistical superiority, threshold
calibration, public API readiness, package readiness, and broad scientific
validity still require their stated gates and artifacts.

## GPU Memory Growth Policy

Owner directive, 2026-07-14: every BayesFilter-owned TensorFlow GPU process
must enable GPU memory growth before TensorFlow initializes a logical GPU.
Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and, where the
process controls TensorFlow initialization, call
`tf.config.experimental.set_memory_growth(gpu, True)` for every visible
physical GPU before `tf.config.list_logical_devices("GPU")` or any tensor
operation initializes the device.

Full-device preallocation is forbidden. A GPU path must fail closed when it
cannot establish memory growth before initialization; it must not continue by
reserving almost all visible VRAM. Serious GPU benchmarks and capacity
diagnostics should record TensorFlow allocator current/peak bytes and must not
interpret `nvidia-smi` process reservation or TensorFlow's device memory limit
as live tensor memory.

This is a resource-sharing and allocator policy, not a claim that every
workload fits concurrently. GPU runs must still respect task-specific
utilization, ownership, cleanup, and capacity gates.

## Zhao-Cui Lane Source-Anchor Gate

For all Zhao-Cui high-dimensional filtering work, "faithful" has a binding
meaning: the agent must inspect and cite both the Zhao-Cui paper/math claim and
the local author source code before implementing, reviewing, or approving any
new source-route behavior.

## Zhao-Cui Production Route Boundary

For Zhao-Cui leaderboard and production work, the generic all-axes
multistate retained-grid route is diagnostic/historical only.  This includes
`bayesfilter/highdim/filtering.py::multistate_nonlinear_fixed_design_tt_value_path`
and `bayesfilter/highdim/filtering.py::multistate_nonlinear_fixed_design_tt_score_path`.
These routes may remain useful for tiny fixture diagnostics, lower-rung
tie-outs, and historical blocker preservation, but they must not be selected
as the production Zhao-Cui leaderboard evaluator.

The production-admissible Zhao-Cui direction is the fixed-variant source-route
path that avoids the generic full tensor-product retained-grid transition
route.  Do not revive the retained-grid path as a production candidate merely
because it emits a finite lower-rung value or score; use the fixed-variant
route for production planning and leaderboard wiring.

Every proposed Zhao-Cui implementation choice must be classified before code is
written:

- `source_faithful`: matches a cited author paper/source operation, with
  source file and line anchors.
- `fixed_hmc_adaptation`: preserves the author's algorithmic route but freezes
  randomness, ranks, bases, schedules, or samples for differentiability/HMC.
  The frozen operation must still cite the author source route it adapts.
- `extension_or_invention`: not present in the author paper/source. It may be
  useful, but it must not close a Zhao-Cui source-faithfulness gap unless the
  user explicitly approves that extension as the target.

Veto rule: if a plan, implementation, result, or Claude review uses
"faithful", "source-faithful", "paper-scale Zhao-Cui", or equivalent language
without paper anchors and author source file/line anchors, block with
`BLOCK_SOURCE_UNGROUNDED`.  If a fixed-gradient need changes the route rather
than merely freezing the author's route, classify it as `extension_or_invention`
unless explicitly approved otherwise.

Claude/Codex review loops for this lane must verify anchors, not merely
internal consistency.  A review that does not inspect the cited paper/source
anchors cannot emit a valid `VERDICT: AGREE` for source-faithfulness.

## Zhao-Cui Training Regularization Default

For Zhao-Cui training-base runs, L1 regularization with explicit L1 weight
tuning is the default procedure going forward.  This is a lane policy, not a
global `P75TrainableTTConfig` scalar default: `l1_weight=0.0` may remain an
allowed comparator arm inside a reviewed tuning grid, but fixed unreviewed
`l1_weight=0.0` training must not be treated as the default Zhao-Cui decision
path.

L1 weight tuning must preserve validation/audit separation.  Validation or
holdout data may nominate, select, or veto candidates only under a reviewed
plan; audit data remains reserved for final-only checks and must not be used
for tuning.  A selected L1 value requires a reviewed tuning/selection ledger
before it can support rank-convergence, HMC, production, or scientific claims.

The reviewed Phase 6T diagnostic showed lower LR plus L1 repaired a rank-5
training pathology, but that diagnostic does not by itself prove production
readiness, posterior correctness, HMC readiness, source-faithful TT-cross
training, or final rank convergence.

## Claude Review Prompt Shape

Claude review must start with the smallest exact path that can answer the gate.
The default prompt shape is:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line: <one path>. Do not
edit, run commands, launch agents, or review the whole repo. Question: <one
question>. End with VERDICT: AGREE or VERDICT: REVISE.
```

Do not send artifact packets, broad path lists, pasted code chunks, whole-file
bundles, or repo-wide instructions as the first review attempt.  If Claude
needs more context, let Claude request the next exact path or line range, then
send only that bounded target.

This rule is deliberately operational.  It avoids repeated review stalls,
approval blocks, and over-broad external disclosure.  A review that can be
answered from a single result/subplan path should be asked as a one-path review.
