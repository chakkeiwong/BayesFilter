# BayesFilter Claude Governance

Claude Code reviewers and workers must follow `AGENTS.md`.

## Backend Rule

The repository implementation backend is TensorFlow / TensorFlow Probability.
NumPy may appear only in explicitly diagnostic code: tests, comparison
fixtures, independent reference solutions, closed-form or finite-difference
checks, and post-run diagnostic inspection. Serialization, reporting, training,
data generation, inference, tuning, candidate selection/admission, artifact
construction, and benchmark kernels are not blanket NumPy exceptions.

Do not approve a non-diagnostic NumPy import or a NumPy-backed runtime path.
TensorFlow tensors may be materialized only at host-side assertion, diagnostic,
or artifact boundaries and must not feed NumPy numerical computation. Existing
violations are migration debt, not precedent. Differentiable or gradient-bearing
paths require TensorFlow / TensorFlow Probability unless the owner explicitly
approves another autodiff backend. PyTorch and JAX are non-default and require a
reviewed exception.

## Default Execution Target

The repository default execution target is GPU.  For DPF transport work, the
default production algorithm target is the GPU-oriented LEDH-PFPF-OT TF32 route:
TensorFlow/TFP, `float32` tensors, TensorFlow TF32 execution enabled, and
streaming/chunked transport where applicable.  CPU, FP64, and FP32-no-TF32 arms
remain explicit reference, comparison, smoke, or fallback modes.

Treat this as a human owner directive, not as a scientific proof.  Do not
reopen the default-vs-experimental question without new evidence or human
instruction, and do not turn this policy into unsupported posterior
correctness, HMC readiness, statistical superiority, dense Sinkhorn equivalence,
or broad scientific-validity claims.

## LEDH Per-Scope Tuning Rule

Every claim-bearing LEDH model run requires an offline tuning artifact for the
exact model/target, route/reset family, horizon/prepared-data regime, particle
count, dimensions, dtype/backend, chunk policy, and route-specific control
family used by that run. Any changed bound field is a new tuning scope. A
setting selected for another model, route, or horizon is a warm-start candidate
only and must never be treated as a universal or inherited default.

Streaming OT routes tune their own Sinkhorn/balance controls. Contract E--TP
and other routes tune their applicable feature, basis, lookahead, chart,
ridge/KKT, or other controls; do not mislabel these as Sinkhorn tuning. Require
disjoint tuning and untouched claim partitions and an exact repository-issued
scope match before accepting a claim. A failed claim triggers fresh
scope-specific tuning with new tuning data under the campaign budget; it does
not authorize tuning on the failed claim data, threshold relaxation, or
skipping later model-specific tuning. Runtime adaptation inside HMC is
forbidden.

## DPF Transport Chunk Rule

Active DPF canonical, candidate, benchmark, leaderboard, and production-target
routes must use `dpf_transport_exact_divisor_cap3000_v1` from
`bayesfilter.highdim.transport_chunk_policy`. Row and column chunks are equal.
For `N<=3000`, the only valid chunk extent is `K=N`; for larger `N`, use the
largest divisor of `N` no greater than 3000. Reject a case with no divisor
greater than 1 and reject every caller override that differs from the selector.

Contrary historical settings are wrong and archival only. Do not approve them
as diagnostic, comparison, timing, tuning, or candidate evidence. In
particular, never promote lower-rung `K=16` fixtures into a larger particle run.
Primitive mechanics tests must also use exact `K=N` chunks. Review must verify
the central selector and active-source discovery guard rather than accepting a
matching local constant.

## TensorFlow GPU Memory Rule

TensorFlow GPU processes must not reserve the whole device eagerly by default.
Before any logical-device or GPU runtime initialization, enable and verify
memory growth on every visible physical GPU. Serious GPU runs must fail closed
if this cannot be done and must record the verified policy in their manifest;
silently ignoring `set_memory_growth` failure is not acceptable.

Memory growth is not a hard memory cap and may eventually consume most or all
available memory. If a run must reserve memory for another process, require a
reviewed logical-device `memory_limit` configuration instead. Memory growth and
logical-device limits are mutually exclusive TensorFlow configurations and the
artifact must state which mode was used. Whole-device preallocation is a
non-default reviewed exception.

## NeuTra Batch-Native Training Rule

Claim-bearing NeuTra HMC uses canonical policy
`bayesfilter_neutra_sequential_hmc_v1` and the shared TensorFlow/TFP controller
under `bayesfilter.inference.neutra_hmc`. Retain and archive warm-up while
excluding it from posterior estimates; use recent-window max(rank-normalized
split R-hat, folded rank-normalized split R-hat) readiness; grow retained draws
cumulatively under modern R-hat and declared ESS/downstream gates; and cap
warm-up and retained sampling at 10,000 per chain. Fixed discarded burn-in and
fixed terminal sampling are historical/smoke/reference exceptions only.

The versioned route ledger is mandatory. Its discovery guard must fail on an
unledgered qualifying route, stale or duplicate classification, active route
without the canonical policy binding, or active fixed-budget implementation.

All BayesFilter NeuTra optimizer updates must be batched with batch size greater
than one. The transport, log determinant, target value/score, loss, gradient,
and optimizer computation must preserve the leading batch dimension in
TensorFlow/XLA.

Do not approve a Python sample loop, scalar target replay, or `tf.map_fn`,
`tf.vectorized_map`, or `tf.while_loop` that merely maps a scalar target over
training rows as a batch-native training implementation. An eligible route uses
batch-native tensor/linear-algebra operations, or persistent multicore workers
that each evaluate a batched shard for GPU transport training. Scalar and
row-mapped routes are parity/reference diagnostics only and must not update
NeuTra parameters.

Training evidence must record batch size, batch-native target backend, device,
XLA status, and scalar-fallback/sample-loop status. Batch size one, any scalar
fallback, or any row-mapped scalar target is a hard veto for NeuTra training,
including smoke and CPU-only training. Existing violations are migration debt,
not precedent for approval.

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
