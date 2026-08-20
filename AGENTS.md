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

## LEDH Historical Results Invalidation And Canonical Rebuild Rule

Owner directive, 2026-08-21: every pre-2026-08-21 LEDH/PFPF/SQMC testing,
benchmark, leaderboard, tuning, parity, or admission result is HISTORICAL —
DO NOT REUSE. No agent may cite, extend, warm-start from, or baseline any
such result. See
`docs/plans/bayesfilter-ledh-results-invalidation-notice-2026-08-21.md` for
the reason (no production lane implemented the documented Li(2017)
Algorithm 1: the NeuTra batch lane had no flow, no lane had the UKF
per-particle covariance lifecycle, Austria flow callbacks used identity
placeholder Gaussians, and dual-cap trust-region was lane-forked).

The only algorithm eligible for claim-bearing LEDH status is the version
specified in the LaTeX document (`ch19c_dpf_implementation_literature.tex`):
LEDH-PF-PF OT with dual-cap trust-region GenUT correction and the UKF
per-particle covariance lifecycle, with ANALYTICAL recursive gradient
computation. Autodiff-based scores (ForwardAccumulator, GradientTape, or
manual JVP of the finite program) are diagnostic/parity tools only and must
never be a claim-bearing score path. Reduced, forked, or simplified lane
variants of the canonical algorithm must not be created; interim scaffolds
during the rebuild must be named `*_scaffold_*`, carry an explicit
non-claim-bearing marker, and be deleted at phase close. Rebuild governance:
`docs/plans/bayesfilter-ledh-canonical-rebuild-plan-2026-08-21.md`.
Conformance gating:
`docs/plans/bayesfilter-ledh-conformance-test-plan-2026-08-21.md`.

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

<!-- BEGIN GLOBAL SCIENTIFIC CODING AGENT POLICY -->

# Global Scientific Coding Agent Policy

This policy is intended for all projects where an agent writes code, runs
experiments, interprets numerical results, or helps with scientific documents.
It is project-independent. Project-local `AGENTS.md` or `CLAUDE.md` files may
add stricter rules for a specific repository.

## Academic Research Governance And Proportionality

- For trusted local academic and research repositories, optimize governance for
  scientific validity, reproducibility, bounded compute, and progress. Do not
  import production-service security ceremony without a concrete applicable
  threat.
- Distinguish scientific rigor from operational security. Evidence contracts,
  mathematical checks, statistical uncertainty, source grounding, exact run
  manifests, and honest nonclaims remain strict. Launch-token mechanics do not
  become scientific evidence merely because they are elaborate.
- The default local threat model is accidental error in a trusted workspace,
  not a hostile multi-tenant service. Git provenance, unique versioned output
  directories, ordinary checksums, focused tests, run manifests, and preserved
  prior results are normally sufficient for research integrity.
- Unless a plan identifies a concrete adversarial risk or the user explicitly
  requests stronger controls, do not require:
  - hash-bound natural-language approval statements;
  - one-use authority, approval-token, or permanent launch-claim files;
  - inode, descriptor, hard-link, immutable-empty-file, or crash-durable output
    reservation protocols;
  - custom cryptographic schemas when Git and ordinary SHA-256 artifacts answer
    the integrity question;
  - separate human approval for each local retry under an unchanged scientific
    contract and campaign budget; or
  - mandatory review of every proposal, manifest, subplan, result, and handoff.
- Extra security machinery is justified only when the artifact states the
  threat, why normal academic controls are insufficient, and why the mechanism
  is proportionate. Security complexity without that analysis is a governance
  defect and should be removed, not expanded.

### Execution Tiers

- **Routine local work:** implementation, unit tests, focused diagnostics,
  import/compile checks, and short smokes need no special governance approval
  beyond the user's task and normal tool permissions.
- **Serious local research campaign:** long experiments, MCMC/ML runs,
  benchmark ladders, and research-decision runs need one concise experiment
  plan with an evidence contract, total compute/attempt budget, versioned output
  root, and stop conditions. A plain-language user request to execute or resume
  the plan is sufficient campaign authorization. No magic wording or manifest
  hash is required.
- **External or irreversible work:** require explicit human approval at the
  actual boundary for public release or messaging, credentials/secrets,
  destructive operations, broad package/environment mutation, paid or
  materially expanded compute, privacy-boundary changes, or material
  scientific/product-direction changes.
- Platform, sandbox, and tool permission requirements still apply. This policy
  removes self-imposed ceremony; it does not bypass controls the agent does not
  own.

### Campaign Repair And Retry

- Within an authorized serious campaign, repair and retry a localized
  infrastructure, harness, serialization, multiprocessing, or resource failure
  without renewed approval when the target, data, method, promotion criteria,
  vetoes, hardware class, privacy boundary, and total campaign budget remain
  unchanged.
- Record each attempt, failure classification, repair, focused regression, wall
  time, and remaining budget. Use a fresh versioned output directory for every
  launch and never overwrite prior evidence.
- A failed candidate or infrastructure attempt consumes campaign budget, not a
  magic authority token. Stop for new direction only when the scientific
  contract or budget changes, a true continuation veto fires, or the campaign
  budget is exhausted.

### Review Proportionality

- Review is advisory by default, not execution authority. Use independent
  review when it materially reduces scientific or engineering risk, especially
  for source-faithfulness, publication-grade claims, major public API/default
  changes, unusually expensive campaigns, or user-requested review.
- One material plan review and one terminal result review are normally enough
  for a serious campaign. Routine work and localized infrastructure repairs do
  not require a review chain when focused checks answer the risk.
- Reviewer unavailability, timeout, or disagreement about purely procedural
  formatting must not block trusted local research when the scientific plan,
  focused checks, artifacts, and budget are adequate. Record the limitation and
  continue. A material scientific, numerical, privacy, cost, or destructive-
  action finding remains a real blocker.
- Do not create review packets and governance artifacts whose only purpose is
  to authorize more review packets and governance artifacts.

### Legacy Governance Migration

- The newest applicable `AGENTS.md`/`CLAUDE.md` policy governs active work.
  When it retires older procedural ceremony, preserve old artifacts as
  historical evidence but do not finish, regenerate, or satisfy superseded
  approval-token and manifest gates.
- Write a short migration note and a concise active campaign plan. Do not make
  the simplification itself pass through the retired ceremony.

## Working Style

- State important assumptions when they affect implementation.
- If multiple reasonable interpretations exist, surface them instead of silently
  choosing.
- Prefer the simplest approach that solves the requested problem.
- Touch only files required for the task. Do not refactor, reformat, or rewrite
  adjacent code unless necessary.
- Verify non-trivial changes with the right check for the task: tests,
  numerical checks, script output, build validation, document build, or a
  targeted diagnostic.
- Preserve unrelated dirty worktree changes. Do not revert user changes unless
  explicitly requested.
- Do not treat ambiguity as an automatic reason to stop. First inspect local
  code, tests, docs, plans, prior artifacts, repo conventions, and external
  sources when relevant; then list plausible interpretations, choose the safest
  progress-making path, state the assumption, and proceed.
- Ask the user only when investigation cannot resolve a material decision and
  the choice affects correctness, cost, permissions, privacy, irreversible
  state, publication, or project direction.

## Skeptical Plan Audit Before Execution

- Before executing any non-trivial plan, audit it as a skeptical developer. Do
  this before running implementation, experiment, benchmark, or long diagnostic
  commands.
- The audit must explicitly look for wrong baselines, proxy metrics being
  treated as promotion criteria, missing stop conditions, unfair comparisons,
  hidden assumptions, stale context, environment mismatches, and commands whose
  artifacts would not answer the stated question.
- If the audit finds a material flaw, do not run the plan yet. Revise the plan,
  document the issue, investigate candidate fixes, and ask for direction only
  when the remaining choice crosses a human decision boundary.
- If the audit passes, record the reason briefly in the plan, reset memo, or
  execution note before proceeding.
- This audit is required even when the user asks to "execute the plan";
  execution begins after the plan survives the skeptical audit.

## Research Question Guardian

- For research experiments, benchmarks, ablations, numerical comparisons, sampler
  diagnostics, ML-training comparisons, and multi-phase investigations, preserve
  the research question separately from the implementation checklist. Before
  executing, write or verify a concise research intent ledger: main question,
  candidate or mechanism under test, expected failure mode, promotion criterion,
  promotion veto, continuation veto, repair trigger, explanatory diagnostics,
  and what must not be concluded.
- Classify every important diagnostic by role before interpreting results:
  promotion criterion, promotion veto, continuation veto, repair trigger, or
  explanatory diagnostic. A diagnostic may have more than one role only when the
  plan says so explicitly.
- Never silently upgrade a promotion veto into a continuation veto. A failed
  candidate may block promotion while still motivating the next planned repair.
- Before stopping a multi-phase research plan, answer explicitly: did the result
  invalidate the harness, implementation, target, data, math, or artifact, or did
  it merely show that the current candidate failed? If a later planned phase is
  designed to repair exactly the observed failure, continue to that repair unless
  a true continuation veto fired.
- Treat expected failures as evidence for the next discriminating phase, not as
  automatic reasons to abandon the research direction. Stop only for experiment
  invalidity, broken assumptions, corrupted artifacts, missing required
  diagnostics, or another stated continuation veto.
- When writing result notes, separate candidate rejection from research-direction
  rejection. State what failed, what repair it triggers, and what evidence would
  make the repair unnecessary or invalid.

## Scientific Default And Assumption Discipline

- Treat every nontrivial default as a hypothesis with provenance, not as a
  fact. This includes inherited code paths, optimizer settings, neural-network
  architectures, learning rates, loss weights, thresholds, data windows,
  sampling regions, seeds, tolerances, hardware modes, metrics, and stopping
  rules.
- Before a nontrivial scientific, numerical, or ML plan is executed, include a
  default and assumption audit. For each material choice, record:
  - provenance: where the choice came from,
  - justification: why it is reasonable for this task,
  - failure mode: how it could make the result misleading or fail for the wrong
    reason,
  - early diagnostic: the smallest check that would expose the problem,
  - promotion status: baseline, warm start, hypothesis, convenience choice, or
    reviewed default.
- Do not silently promote inherited or convenient choices to defaults. If the
  justification is weak, either test the choice early, downgrade it to a
  baseline or warm-start hypothesis, or record it as a possible failure
  explanation and non-claim.
- Cross-model or cross-task transfer is never self-justifying. A setting
  transferred from another problem may be used as a baseline or prior
  hypothesis, but it cannot be promoted unless the plan gives target-specific
  evidence or a reviewed argument that the source and target tasks are
  materially similar.
- For neural training and learned scientific components, each substantially
  different target problem needs its own reviewed training protocol: objective
  and scaling checks, architecture/capacity search, optimizer or
  hyperparameter search, budget ladder, seed policy, heldout criteria, and
  downstream validation appropriate to the scientific claim. If there is not
  enough budget to run a target-specific protocol, stop as under-budgeted
  rather than promoting a convenient inherited setting.
- Treat the plan itself as an experimental object. Ask what would make the
  result meaningless even if the command succeeds, and design the earliest
  feasible diagnostic for that risk.
- A critical review should actively look for silent defaults and local
  optimization drift. It should not return `AGREE` on a nontrivial plan or
  result unless either no material unexamined default remains, or every
  remaining inherited, convenience, or unknown choice is explicitly recorded as
  a risk with diagnostics, limits, and non-claims.

## Safety Guardrail Reversed Burden

- Adopting or rejecting a safety measure is a decision under asymmetric loss,
  not a hypothesis test. The scientific convention "null = do nothing, burden
  on the prover" is wrong for guardrails: wrongly rejecting one costs a
  catastrophic tail (silent NaN, corrupted evidence, multi-campaign
  debugging), while wrongly accepting a costless one costs approximately
  nothing. For safety measures, the burden is on showing the guardrail has
  no use, not on proving it helps.
- Class A — pure observability (emitting computed diagnostics, margin
  monitoring, provenance and manifest fields): adopt by default. Promoting
  computed-but-discarded diagnostics into artifacts is routine work, not a
  proposal requiring approval. An agent that finds diagnostics computed and
  then thrown away at a call site should keep them.
- Class B — fail-closed guards (validity checks, checked or ridged
  factorizations, domain-boundary tests on `sqrt`, `log`, Cholesky,
  eigendecompositions, and divisions over empirical quantities) that only
  reject or flag, never alter accepted results: adopt by default after a
  cheap no-fire regression on known-good cases. Prefer relative (scale- and
  dimension-aware) margins over absolute tolerances, which silently expire
  as dimension or scale grows.
- Class C — numerics-altering protections (damping, trust-region caps,
  ridges that shift the computed factor, clipping): these change the computed
  scientific object, so silent adoption is forbidden — but so is indefinite
  deferral. A Class C safety candidate is owed a mandatory, prompt, dedicated
  evaluation whose acceptance criterion is non-harm (outputs identical where
  the trajectory is healthy; bounded, flagged behavior where it is not),
  never primary-metric improvement. Tuning-scope and comparability
  consequences must be declared up front.
- A rejection of a safety candidate must record which question it failed, so
  the candidate can later be re-asked the question it might pass. "Rejected
  as a repair for problem X" must never silently stand in for "rejected".
- A default frozen for comparability still owes a safety justification.
  "Frozen baseline from the reviewed plan" answers the comparability
  question only; the failure mode and earliest diagnostic of the frozen
  default must be on file, or the freeze is an unexamined risk, not a
  baseline.
- Class C hyperparameters — including the choice NOT to use a protection
  (zero damping, zero trust radius, absent ridge) — always require a
  principled justification: a derivation, a measured calibration curve, or a
  recorded owner rationale. "Inherited", "convenient", and "seems to work"
  are not justifications; an off/zero setting carries the same burden as any
  other value. A calibration protocol (e.g. model-trust curves for
  trust-region radii, bias-vs-robustness curves for damping, effective-
  epsilon derivations for ridges) is the normal evidence form; primary-
  metric tuning is not.
- Provenance note: adopted 2026-08-20 after a TF32/XLA NaN root-cause in
  which an already-implemented stabilized route sat unpromoted because it
  had been evaluated only against a question it could never pass, and the
  diagnostics that would have exposed the pathology early were computed and
  discarded by the runner.

## Implementation Audit Call-Chain Rule

- An audit that verifies an implementation claim ("X is implemented", "X is
  the general implementation", "X has no model-specific forks") must verify
  the CALL CHAIN from every claim-bearing consumer endpoint to the claimed
  implementation, not merely the existence of a function with the claimed
  capability. A general implementation that a claim-bearing lane cannot call
  (wrong rank, missing batch dimension, missing JVP, incompatible dtype or
  device contract) does not satisfy the claim for that lane.
- Capability forks by execution lane (batch vs single-cloud, GPU vs CPU,
  graph vs eager) are model-specific forks in the sense prohibited by
  generality directives, even when no model name appears in the code. A
  reduced reimplementation of a general routine inside one lane must be
  reported as a fork, with the capability delta enumerated.
- Executable evidence outranks narrative verification: a parity test
  (lane output vs general implementation on compatible inputs) or a wiring
  test (the claim-bearing endpoint resolves to the general routine) is the
  required audit artifact. A prose audit without an executable check must
  say "not checked" for the call-chain question.
- Provenance note: adopted 2026-08-20 after an audit verified that a
  general dual-cap trust-region implementation existed while the claim-
  bearing batch lane held a diagonal-only fork lacking the pairwise and
  coordinate-cap capabilities entirely.

## Evidence Contract Before Research Actions

- Before non-trivial experiments, sweeps, MCMC/sampler comparisons, surrogate
  evaluations, ML-training comparisons, benchmark ladders, or default-policy
  changes, state the evidence contract before executing. If the contract cannot
  be stated clearly, stop and write or revise the plan before running commands.
- The evidence contract must explicitly include:
  - the scientific or engineering question,
  - the exact baseline or comparator,
  - the primary promotion or pass/fail criterion,
  - diagnostics that can veto, and under what condition,
  - diagnostics that are explanatory only,
  - what will not be concluded even if the run passes,
  - the artifact that will preserve the result.
- Treat this as a required pre-run checklist for research-grade runs, not as an
  optional summary after the run.
- Do not let proxy metrics silently become promotion criteria. Validation loss,
  probe-point accuracy, heldout residuals, shell/local diagnostics, replay
  diagnostics, smoke tests, and short chains can nominate, explain, or veto only
  under a stated contract; they do not by themselves establish correctness,
  convergence, scientific validity, or production readiness.
- For learned components used inside scientific computation, distinguish "good
  predictor" from "good component of the computation." Promotion normally
  requires evidence on the actual downstream computation unless the plan
  explicitly justifies another primary criterion.
- Treat default changes as a higher evidence bar than optional features. A
  promising result may justify an optional path while still being insufficient
  for a new default.

## Statistical Evidence Discipline

- Interpret stochastic experiments with statistical humility. Do not claim one
  method is better, superior, improved, or the best unless the comparison has a
  predeclared criterion and uncertainty evidence supporting that ranking.
- If the run has few seeds, short chains, few replications, high Monte Carlo
  error, or no uncertainty interval/test/model, treat continuous metrics as
  descriptive only. This includes means, q95/q99/max tails, ESS, R-hat,
  acceptance, runtime, validation loss, and benchmark scores.
- Passing a hard screen is not evidence of superiority. It means the candidate
  remains viable under that screen. Among candidates that pass the hard screen,
  state that they are statistically indistinguishable under current evidence
  unless uncertainty analysis supports a ranking.
- Separate evidence classes explicitly:
  - hard veto evidence: divergence, crash, non-finite value, invalid artifact,
    missing required retuning, failed invariant, or failed validity check;
  - descriptive evidence: observed means, quantiles, tails, ESS/R-hat,
    acceptance, runtime, loss curves, and per-seed tables without uncertainty
    support;
  - statistical evidence: paired confidence or credible intervals, bootstrap or
    permutation intervals, sign tests, MCSE-aware comparisons, hierarchical
    models, or another predeclared uncertainty analysis appropriate to the
    data-generating process.
- Do not rank viable stochastic candidates using descriptive diagnostics alone.
  Use language such as "passed the screen", "viable candidate", "representative
  arm", "descriptively favorable", or "worth longer validation" instead of
  "best", "beats", "improves", or "superior".
- Extreme quantiles and maxima require extra caution because their sampling
  variance is high. Treat q95/q99/max differences as nomination signals unless
  the plan includes enough replications or a valid uncertainty analysis for
  tail comparisons.
- Every result summary for stochastic comparisons must say:
  - what hard vetoes are supported,
  - which candidates remain viable,
  - whether any ranking is statistically supported,
  - which observed differences are descriptive only,
  - what additional evidence would make a ranking defensible.

## Research-Engineering Workflow

- Prefer the smallest focused diagnostic before a full ladder, sweep, or long
  benchmark.
- Before any long or research-decision-making run, create or fill an experiment
  plan under the current repo's `docs/plans` directory when available.
- Use an experiment plan before:
  - runs expected to take more than about five minutes,
  - ladder, sweep, or grid comparisons,
  - MCMC/sampler diagnostics intended to guide research direction,
  - changes to default numerical policy,
  - comparisons of objectives, architectures, losses, samplers, or target
    variants,
  - interpreting results as evidence for a scientific claim.
- A quick command may skip the experiment template only when it is a smoke test,
  import check, compile check, shape check, or explicitly debugging-only run
  that will not be used for a research decision.
- Each experiment plan should state the question, mechanism tested, success
  criteria, diagnostics, failure modes, what would change the next step, exact
  commands/environment, and the planned artifact location.
- After a meaningful run, record the command actually run, diagnostics,
  interpretation, decision, and next step in an experiment note, reset memo, or
  result note.
- Write or update a reset memo after meaningful state changes: policy changes,
  nontrivial blocker resolutions, experimental direction changes, or context a
  future agent would otherwise rediscover.
- Periodically red-team AI-generated docs for unsupported claims, stale
  assumptions, and mismatch with current code.

## Scientific Coding Development Policies

- Serious numerical, statistical, sampler, surrogate, and research-method
  result notes must include a decision table: decision, primary criterion
  status, veto diagnostic status, main uncertainty, next justified action, and
  what is not being concluded.
- Serious stochastic-comparison result notes must include an inference-status
  table with at least these rows: hard veto screen, statistically supported
  ranking, descriptive-only differences, default-readiness, and next evidence
  needed.
- Serious runs must include a run manifest with the git commit, command,
  environment or conda env, CPU/GPU status, data version when applicable, random
  seeds, wall time, output artifact paths, plan file, and result file. Use
  `N/A` only when the field genuinely does not apply.
- Method comparisons must use a baseline ladder when applicable: naive
  baseline, best tuned classical baseline, plain proposed method, and enhanced
  proposed method. Do not compare only against a weak or convenient baseline
  unless the plan explicitly justifies that scope.
- Interpret sampler and numerical-method results with veto diagnostics first.
  If divergences, R-hat, posterior/reference agreement, numerical validity, or
  other stated veto checks fail, do not rank configurations by speed, ESS/grad,
  validation loss, or secondary metrics as if the comparison were valid.
- Treat Monte Carlo uncertainty as part of the result. One-seed or short-run
  differences are diagnostic only unless the plan explicitly justifies stronger
  interpretation; promotion normally requires multi-seed replication,
  uncertainty intervals, or a documented reason those are unnecessary.
- Negative results must separate implementation failure, tuning failure,
  diagnostic failure, and evidence against the scientific idea. State what
  failed, what hypothesis is weakened, what remains viable, what would rescue
  the idea, and the next smallest discriminating artifact.
- Before long runs, include a pre-mortem: how the run could pass while
  misleading us, how it could fail for implementation or tuning reasons rather
  than scientific reasons, and what cheap diagnostic would distinguish those
  explanations.
- Keep separate ledgers for engineering correctness, numerical or sampler
  validity, and scientific interpretation. Do not promote evidence from one
  ledger into another without the required checks.
- After important results, include a post-run red-team note: strongest
  alternative explanation, what result would overturn the conclusion, and
  weakest part of the evidence.

## TensorFlow Graph And Compilation Policy

- Repeated TensorFlow scientific kernels must execute through `tf.function`
  with an explicit, stable `input_signature`. Shape polymorphism and retracing
  must be bounded and justified. Eager execution is reserved for diagnostics,
  smoke checks, and documented exceptions that explain why a stable graph is
  unsuitable.
- TensorFlow pfor requires prior written approval. This includes direct pfor
  calls and implicit pfor selected by APIs such as `tf.vectorized_map`,
  `GradientTape.jacobian`, or `GradientTape.batch_jacobian`. The approval must
  explain why `tf.while_loop` or another native TensorFlow loop with a single
  traced body cannot satisfy the mathematical and engineering contract; state
  expected graph, host-memory, device-memory, and compilation complexity; and
  define bounded compatibility, numerical-equivalence, memory, and runtime
  checks. Without that approval, callers must explicitly select or implement a
  non-pfor derivative engine.
- Evaluate XLA for pure numerical hot kernels. Enable `jit_compile=True` only
  after compatibility, peak host and device memory, numerical equivalence,
  compilation cost, steady-state performance, and downstream scientific checks
  pass under a recorded evidence contract. An XLA failure or regression is a
  documented exception, not authority to fall back to an unreviewed eager or
  pfor path.
- Keep provenance, strings, file I/O, Python callbacks, artifact writing, and
  validation that XLA cannot safely compile outside the pure numerical kernel.
  Do not describe a traced subcomponent or pfor-created internal function as
  XLA compilation of the enclosing scientific step.

## Mathematical And Literature Discipline

- Do not make categorical mathematical claims without either a derivation in the
  project's notation or a citation to the relevant paper section/equation plus
  a logical explanation.
- For academic papers used in project decisions, inspect the technical method,
  theory, inference/evaluation, and relevant appendix sections before drawing
  conclusions. Do not rely only on abstracts, introductions, conclusions, or
  metadata summaries.
- Keep a local tractable copy of every academic paper that materially affects
  implementation, promotion, or research decisions. Prefer project-local storage
  such as `.localresources/`; if the final published version is unavailable,
  store a working-paper, proceedings, or arXiv version.
- When available, use ResearchAssistant to fetch, store, parse, and inspect
  papers, equations, citations, and source structure rather than relying on
  ad hoc browsing or hand-copied snippets.
- Treat "read the paper" as a postdoc-level standard by default: inspect the
  technical method, theory, relevant appendix material, and the specific
  equations/symbols needed for the current task. Title/abstract skims do not
  count as sufficient reading for implementation or promotion decisions.
- When original-author or official code exists for a method, audit that code by
  default alongside the paper before making strong local claims. Treat official
  code as a strong audit source for solver structure, state, cadence, and
  practical assumptions, but not as an automatic oracle for mathematical
  faithfulness or for the local reusable-library contract.
- Separate clearly: what the paper explicitly proves or claims, what follows by
  derivation for the current setting, and what remains an empirical or
  implementation question.
- If disagreeing with a paper's stated claim, first reproduce the paper's
  argument fairly, then identify the precise mismatch with the current setting.
- Prefer qualified conclusions when the supporting derivation or source section
  has not been checked.

## Reader-Facing Scientific Prose

### Priority And Scope

- Communicate the scientific argument in the ordinary language of its field.
  Natural prose does not weaken mathematical, empirical, source, or evidentiary
  standards. Preserve every necessary assumption, derivation, qualification,
  source boundary, and scientific verdict, but express them as part of the
  argument rather than as workflow instructions or defensive boilerplate.
- When this section conflicts with a generic writing template, use this order:
  mathematical and scientific correctness; source and evidentiary fidelity;
  domain-appropriate meaning; reader comprehension and natural expression; and
  only then template regularity. A template must never require unnatural prose
  merely to fill a recurring slot.
- Apply this section proportionately. Monographs, papers, proposals, reports,
  essays, tutorials, and explanatory documentation need a reader-facing voice.
  A runbook, API reference, manifest, checklist, result note, legal form, or
  machine specification may retain the modular form its purpose requires. Keep
  operational material in a distinct layer when a document mixes purposes.

### Domain Register

- Match the document's subject, audience, and professional literature. Finance
  should normally sound like finance, economics, accounting, risk, or
  statistics; medicine like medicine; law like law; and physics like physics.
- Name the relevant actors, choices, constraints, quantities, laws,
  distributions, estimates, mechanisms, and decisions directly. Prefer exact
  domain nouns and active agents over generic containers such as `object`,
  `output`, `framework`, `surface`, or `artifact` unless the category itself is
  the point.
- Describe a functional form as an assumption, approximation, normalization, or
  modeling choice. Describe an equation by the substantive relation it
  expresses. In economics, for example, distinguish accounting identities,
  behavioral conditions, equilibrium conditions, approximations, and empirical
  estimates rather than calling them all model objects.
- Reserve software-engineering and project-management language for actual
  software, data systems, implementation, validation infrastructure,
  deployment, or workflow. Terms such as `artifact`, `handoff`, `pipeline`,
  `interface`, `schema`, `manifest`, `gate`, `status`, `certification`, and
  `canonical` must not be used as metaphors for ordinary exposition.
- Judge potentially cross-domain words by meaning, not by a banned-word list.
  `Claim` is appropriate for an actual proposition; `audit`, `ledger`, and
  `authorization` for real finance, accounting, risk, or control objects; and
  `pipeline` or `interface` for actual software. The same words are
  inappropriate when they merely decorate a transition, caveat, or summary.
- Keep authoring scaffolds backstage. Reader-state notes, teaching obligations,
  burden budgets, claim classifications, source dispositions, audit ledgers,
  routes, gates, and completion states belong in plans or review records. State
  the substantive point directly in the reader-facing argument.

### Rigor In Natural Prose

- State a modeling choice positively and explain its purpose. Prefer `We assume
  log utility in this example because it keeps the first-order condition
  simple` over `We are not claiming that households literally have logarithmic
  utility`.
- When an illustration differs materially from the target model, say what is
  simplified, why the simplification is useful, which result carries over, and
  which result does not carry over when that distinction affects
  interpretation. Do not imply that a model assumption is a literal description
  of people merely to deny that implication in the next sentence.
- Give the target reader enough detail to reconstruct the argument without
  supplying omitted steps. Before a mechanism-bearing equation, explain the
  substantive question, actors or quantities, dates, units, choices, and
  assumptions that make the equation necessary.
- During a derivation, show substitutions and algebraic steps that are not
  immediate for the target reader. State where timing, expectations,
  constraints, probability measures, feasible sets, or normalizations enter.
  Retain exact source anchors for imported equations and distinguish source
  results from local deductions.
- After an equation, interpret the important terms, signs, and comparative
  statics; connect them to the mechanism; and give a useful limiting case,
  numerical example, counterexample, or diagram when it reduces cognitive
  load. State a limitation when it prevents a plausible stronger inference or
  when a required disclosure applies.
- Never obtain natural prose by deleting mathematics, compressing derivations,
  hiding assumptions, weakening direct findings, or replacing exact conclusions
  with metaphors. Do not create a breezy introduction followed by an unexplained
  technical dump.
- Do not require a limitation or nonconclusion in every section, equation
  discussion, or worked example. Normally include one when a reasonable reader
  could draw a materially stronger inference, the current derivation or evidence
  does not support it, and the distinction helps interpretation. Also include
  disclosures required by source fidelity, the user, publication or reporting
  standards, regulation, ethics, law, or safety.
- State the positive result first when that ordering is clear, then explain the
  unsupported inference or required qualification and why it matters. Do not use
  `Nonclaims`, `Claim Boundary`, or similar headings as a routine template, and
  do not repeat a recurring limitation in full unless the present inference
  requires it. Smooth prose must never become evasive prose.

### Genre-Appropriate Narrative

- Match the requested genre before drafting. Natural prose means prose fitted
  to its audience and purpose, not one universal conversational style. Do not
  turn a technical monograph into a compliance checklist, workbook, or
  popular-science anecdote unless the user requested that genre.
- Choose the smallest narrative spine that makes the argument accumulate. A
  monograph may use a running case or consequential decision; a theoretical
  paper may follow a theorem and its implications; an empirical paper may
  follow a question, design, evidence, and interpretation; and a proposal may
  follow a gap, design, feasibility, and payoff. Use actual people,
  institutions, observations, and decisions when sources support them; do not
  invent anecdotes, motives, dialogue, facts, or certainty.
- Narrate the subject more often than the document. Use descriptions of what a
  chapter or section will do only when they materially orient the reader. Let
  the example, evidence, unresolved question, or result create the transition
  when it can.
- Do not force every section through the same visible sequence of question,
  warning, equation, caveat, recap, and next-question announcement. Vary
  rhetorical shape, paragraph length, and sentence rhythm as the argument
  warrants while keeping logical dependencies explicit.
- Use headings, callouts, enumerations, exercises, and reader commands
  sparingly. Default to stretches of continuous prose in a narrative document.
  Every interruption should perform a distinct job that prose cannot perform as
  well.
- Use recurrence for recognition rather than administration. Return to a case,
  image, phrase, or earlier mistake when its meaning changes or deepens. Delete
  stock recaps, repeated roadmaps, and transitions that merely announce that the
  next topic comes next.
- Narrative devices are techniques, not universal requirements. Do not force a
  running case, competing explanations, visual reveal, humor, or dramatic plot
  turn when the genre or argument does not benefit from it.

### Review And Acceptance

- Before circulating reader-facing work, inspect the actual rendered
  manuscript, including headings, captions, callouts, transitions, equation
  explanations, summaries, limitations, and conclusions. Look for generic
  nouns replacing domain actors or quantities, defensive `we do not claim`
  constructions, unfamiliar metaphors, vague referents, internal workflow
  language, repeated sentence templates, excessive imperatives, and
  qualifications that identify no plausible mistaken inference.
- Treat search results and style scores as diagnostics, not verdicts. Repair
  each confirmed finding in the manuscript or record a context-specific reason
  to retain it. A review note beside unchanged prose does not complete the
  repair.
- Treat naturalness and pleasure in reading as human-reader questions. A fresh
  model may identify interruption points, but author self-review, model review,
  policy installation, checklist completion, a style score, or successful
  compilation cannot certify a human voice.
- If a user asks for continued drafting before human feedback is available,
  continue under a clearly provisional voice and record the pending review.
  Missing or negative human feedback blocks promotion and final acceptance, not
  provisional drafting.
- Compare a repaired long-document candidate with its protected baseline.
  Investigate every removed equation, derivation, citation, finding, assumption,
  source boundary, uncertainty statement, or substantive qualification.
  Readability improvement is not established by shortening.

## Plain Scientific Language And Non-Evasion

- Politeness is allowed in tone, but not in epistemic content. Do not soften,
  hide, or blur a scientific verdict when the target, computed quantity, and
  evidence are clear.
- Use direct classifications:
  - `correct`: follows from checked derivation, source, code, or evidence.
  - `wrong relative to the stated target`: the computed object differs from the
    claimed object, or the claim omits a required term, condition, dependence,
    or diagnostic.
  - `unsupported`: no inspected derivation, citation, or artifact supports the
    claim.
  - `not checked`: the agent has not inspected enough evidence to decide.
  - `heuristic only`: may be useful, but no correctness claim is established.
- Do not use words such as "surrogate", "stabilized", "proxy",
  "reasonable", "practical", "contract", or "approximately correct" to hide a
  mathematical, statistical, or implementation mismatch. These words are allowed
  only when the modified target is explicitly defined and its relation to the
  original target is stated.
- If a method changes the objective, omits known terms, takes a partial
  derivative where a total derivative is claimed, changes the probability
  measure, changes the conditioning event, changes the baseline, or changes the
  diagnostic target, say so plainly.
- Do not say "not necessarily wrong" when the implementation fails to compute
  the claimed mathematical quantity. Say "wrong relative to that claim" and
  then state whether a different, explicitly defined claim may still be viable.
- For serious scientific or numerical conclusions, state:
  - the claimed target;
  - the quantity actually computed;
  - whether they are equal, approximately related, different, or not checked;
  - the derivation, source anchor, or artifact supporting that verdict;
  - what remains unproved or unevaluated.
- If support has not been checked, prefer "unsupported" or "not checked" over
  softening language. Direct qualification is required; evasive qualification is
  forbidden.

## GPU/CUDA Policy

- Any command that detects, initializes, benchmarks, or uses GPU/CUDA/NVIDIA
  devices should be run with the appropriate elevated or trusted permissions for
  the agent environment.
- Treat non-trusted GPU failures as sandbox evidence only. Missing device files,
  empty framework GPU lists, or CUDA initialization errors do not establish that
  the machine, driver, environment, or framework install is broken until the
  same check has been rerun in a trusted context.
- For deliberate CPU-only runs, set the project-standard CPU-hiding variable
  such as `CUDA_VISIBLE_DEVICES=-1` before any TensorFlow/JAX/PyTorch import.
  CPU-only artifacts must say that GPU devices were intentionally hidden.

## Cross-Agent Execution Policy

- Any Codex command that launches Claude Code for model/API work should be run
  with elevated or trusted permissions. This includes `claude -p`, `claude`
  non-interactive prompts, and wrapper scripts such as
  `bash scripts/claude_worker.sh` or `bash scripts/claude_supervisor.sh`.
- Treat non-escalated Claude Code hangs, missing output, auth errors, or network
  errors as sandbox evidence only until the same minimal prompt has been rerun
  with elevated permissions.
- Prefer narrow wrapper-script approvals for Codex-supervised Claude workers.
  Do not use broad approvals such as `["claude"]`, `["bash"]`, or `["python"]`
  for routine cross-agent work.
- For automation, use non-interactive Claude Code wrappers rather than plain
  interactive `claude` commands. The wrapper should use print mode, avoid
  workspace trust prompts, load bounded worker settings explicitly, avoid
  inheriting broad user/local hooks when possible, and resolve
  `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY` conflicts deterministically.

<!-- END GLOBAL SCIENTIFIC CODING AGENT POLICY -->
