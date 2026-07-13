# BayesFilter Agent Governance

## Academic Risk-Tier Workflow

Owner directive, 2026-07-13: BayesFilter is an academic research repository.
Use governance proportional to the actual scientific, computational, and
external-impact risk. Progress on engineering and scientific questions should
not be blocked by security ceremony that does not mitigate a credible risk for
the task.

This section governs planning, artifact, and review depth unless a stricter
domain-specific section below applies. It supersedes older BayesFilter plans and
runbooks that require a dedicated subplan, frozen hash, authorized snapshot,
finite write-set manifest, custom guard, or repeated Claude/Codex review for
every diagnostic or phase. Those older documents remain historical evidence;
they are not active execution policy unless the user explicitly reactivates
them after this directive.

### Tier 1: Routine Academic Engineering

Use Tier 1 for read-only investigation, focused debugging, reversible local
source/test edits, import/shape/compile smokes, and targeted checks expected to
finish in about five minutes and not used for a scientific or default-policy
claim.

- A short working plan in the conversation is sufficient.
- Run the smallest relevant check and report its result.
- No plan file, phase subplan, external review, hash ledger, authorized
  snapshot, run manifest, custom filesystem/import guard, or result table is
  required unless it directly answers the engineering question.
- Ordinary source inspection, including installed dependency source, does not
  require a separate source-proof phase.

### Tier 2: Material Research Engineering

Use Tier 2 for numerical algorithm changes, differentiable-path changes,
benchmarks or diagnostics expected to exceed about five minutes, GPU/XLA runs,
meaningful method comparisons, or experiments that guide research direction.

- Maintain one concise live plan under `docs/plans`. It should state the
  question, exact baseline, candidate mechanism, pass/promotion criterion,
  vetoes, explanatory diagnostics, commands/environment, resource stop, result
  artifact, and nonclaims.
- Perform the skeptical pre-execution audit and evidence contract required by
  the global policy. Record them in that live plan; do not create a separate
  governance phase.
- Use one result or reset note after meaningful evidence. Preserve exact
  commands, environment/device/JIT settings, seeds when applicable, wall time,
  and output paths.
- Review is required only when the work changes mathematical/numerical logic,
  a public API or default, or supports a material scientific conclusion. Use
  one focused review followed by a repair/recheck when needed. Repeated
  convergence rounds are not the default.
- A failed candidate is a repair signal, not an automatic stop for the research
  direction. Continue to the next smallest discriminating repair unless an
  evidence-validity, resource, safety, or authority veto fires.

### Tier 3: High-Impact Or Irreversible Work

Use Tier 3 for destructive operations, dependency or model-file changes,
network/external disclosure, releases, public/default-policy promotion,
expensive overnight or large-sweep execution, consequential scientific claims,
or actions that affect systems or people outside the repository.

- Use an explicit reviewed plan and obtain human/tool approval where required.
- State budgets, cancellation and cleanup behavior, external effects, and
  promotion evidence prospectively.
- Review and artifacts should be detailed enough for the concrete risk, but a
  Tier 3 program still does not require a new subplan for every internal step.

### Proportional Artifact And Review Rules

- Hash immutable inputs or outputs only when identity matters to reproducibility
  or comparison integrity. Do not hash every intermediate document or log.
- Use dirty-worktree snapshots, finite write sets, or custom guards only when a
  credible concurrent-write, provenance, or boundary risk justifies them.
  Normal Git status inspection and preservation of unrelated changes are
  sufficient otherwise.
- Claude is optional, read-only, and advisory. Use it only when explicitly
  requested or when a material independent review is worth the disclosure and
  latency. Native focused review is sufficient for ordinary academic work.
- Do not let review mechanics replace decisive experiments. Prefer a bounded
  counterfactual that can falsify a hypothesis over increasingly elaborate
  offline provenance arguments.
- All tiers retain the global scientific evidence rules, statistical humility,
  mathematical sourcing, GPU trust requirements, unrelated-worktree
  protection, and prohibitions on unsupported correctness, performance,
  readiness, or scientific claims.

## Default Implementation Backend

BayesFilter algorithmic implementation defaults to TensorFlow and TensorFlow
Probability.

## Default Execution Target

The BayesFilter repository default execution target is GPU.  CPU-only execution
is allowed for explicit reference checks, small smoke tests, debugging, and
sandbox-safe diagnostics, but it must not be described as the default
production target unless a reviewed plan explicitly changes this policy.

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

New BayesFilter-owned algorithmic code must not use NumPy as its implementation
backend.  NumPy is allowed only for:

- independent reference solutions;
- comparison fixtures;
- closed-form sanity checks;
- serialization, reporting, and lightweight data inspection;
- narrowly reviewed exceptions recorded in a plan or result artifact.

Differentiable or gradient-bearing implementation paths must use TensorFlow /
TensorFlow Probability unless a reviewed plan explicitly authorizes another
autodiff backend.

PyTorch and JAX are non-default backends for this repository.  They require a
reviewed exception before use in BayesFilter-owned algorithmic implementation
paths.

## Evidence Discipline

Reference, comparison, prototype, smoke, and reporting code must not be
represented as the BayesFilter default implementation.  If an experimental lane
uses NumPy for a prototype or comparator, the artifact must say so explicitly
and must preserve the gap to a TensorFlow / TensorFlow Probability
implementation.

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
