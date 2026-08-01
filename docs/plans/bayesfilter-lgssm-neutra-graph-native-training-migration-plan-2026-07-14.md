# LGSSM NeuTra Graph-Native Training Migration Plan

Date: 2026-07-14

## Objective

Make the active target-specific LGSSM NeuTra training process satisfy two hard
runtime invariants:

1. all optimization steps execute in one TensorFlow/XLA program with no
   per-step Python loop, host callback, or per-step tensor materialization; and
2. no repository-owned module in the training command's import and execution
   closure imports or calls NumPy.

Static Python used once to construct a fixed neural-network graph, validate
configuration, or serialize the terminal result is not an optimization loop.
Python iteration over fixed layer/variable lists during tracing is unrolled into
the graph and is allowed. Python or NumPy computation over training steps,
training batches, samples, target evaluations, gradients, or optimizer updates
is forbidden.

## Scope

In scope:

- `bayesfilter/inference/neutra_training.py`;
- package initialization reached by importing the training engine;
- the exact LGSSM target and target-specific training harness reached by the
  `train` CLI stage;
- training-state, heartbeat-summary, frozen-transport, parity, held-out screen,
  and GPU/XLA evidence emitted directly by a training job;
- focused tests and a fresh bounded GPU/XLA smoke.

Out of scope:

- Phase 5 tuning and Phase 6 HMC, which are already blocked on their separate
  NumPy migration;
- unrelated legacy BayesFilter modules not imported by the training command;
- eliminating Python from CLI parsing, one-time configuration, terminal
  artifact writes, or static graph construction.

## Research Intent And Evidence Contract

| Item | Contract |
| --- | --- |
| Engineering question | Can the unchanged reverse-KL dense-IAF training computation run as one graph-native XLA program without importing NumPy? |
| Scientific target | The same fixture-bound exact 18D LGSSM posterior, target signature `f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30`. |
| Baseline | Current stateless per-step seeds, dense-IAF/affine composition, reverse-KL objective, manual Adam equations, clipping, recipes, and float64 GPU/XLA target evaluation. |
| Primary engineering criterion | A multi-step call contains graph control flow, completes all requested steps, and returns final state plus cadence-selected tensor diagnostics in one host invocation. |
| NumPy veto | An imported repository-owned module has a direct NumPy import or call; or source/runtime inspection finds `tf.numpy_function`, `tf.py_function`, or a repository NumPy bridge. TensorFlow's own third-party NumPy dependency is outside this source-policy claim. |
| Loop veto | A Python loop invokes the compiled optimization step, any per-step `.numpy()`/host synchronization remains, or the graph lacks a `While`/`StatelessWhile` operation for a multi-step run. |
| Numerical veto | Seed identity changes, resumed and uninterrupted final states differ, nonfinite loss/gradient/state occurs, target status fails, frozen reload/score parity fails, or GPU/XLA placement fails. |
| Explanatory diagnostics | Compile-plus-run wall time, warm execution time, cadence losses, gradient norms, clipping flags, log determinant, target conditioning, and graph operation inventory. |
| Artifact | This plan, focused test output, fresh smoke output under a new versioned root, and a result note under `docs/plans`. |
| Nonclaims | This migration does not establish transport quality, posterior correctness, HMC convergence, speedup, recipe ranking, production readiness, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| `tf.while_loop` inside `tf.function(jit_compile=True)` | Repository XLA-default policy | Expresses all steps as one compiled program | Unsupported target op or resource-variable update under XLA | Tiny two-step CPU-XLA unit test, then trusted GPU smoke | reviewed implementation choice |
| TensorArray cadence collection | Existing heartbeat cadence | Preserves bounded diagnostics without host calls | Dynamic TensorArray/XLA incompatibility or wrong row count | Exact expected step-index test and graph smoke | implementation hypothesis |
| Terminal-only checkpoint | Constraint imposed by no host loop/callback | JSON file I/O cannot execute inside an XLA cluster | Loss of mid-run crash recovery | Short screen before long runs; terminal artifact checks | explicit tradeoff |
| Stateless seed `(root0, root1 + step)` | Existing training implementation | Preserves exact Monte Carlo stream across graph and resume | Off-by-one drift | uninterrupted-versus-resumed exact-state test | frozen baseline |
| Manual Adam equations unchanged | Existing reviewed implementation | Isolates execution topology from optimizer semantics | update-order or bias-correction drift | one-step and resumed parity tests | frozen baseline |
| Terminal tensor materialization | Required artifact boundary | Host serialization occurs only after optimization completes | hidden per-step synchronization | source AST/static checks and call-count instrumentation | reviewed boundary |
| Lazy package exports | Existing top-level `bayesfilter` pattern | Prevents unrelated NumPy-backed modules from loading | public import regression | common inference/testing import compatibility tests | implementation choice |

## Checkpoint And Failure-Semantics Amendment

The old `checkpoint_every=50` contract is incompatible with an uninterrupted
single XLA call because Python JSON writes cannot occur inside XLA. Host
callbacks (`tf.py_function`, `tf.numpy_function`) are forbidden and would also
break device/XLA evidence. Therefore:

- the strict route emits exactly one immutable terminal checkpoint per call;
- `stop_after_steps` remains a deliberate bounded call boundary and can create
  a resumable terminal state for a planned partial run;
- infrastructure interruption inside a call has no recoverable mid-call state;
- the target-specific campaign disables automatic mid-job infrastructure
  resume and records `checkpoint_policy=terminal_only_graph_native_v1`;
- prior smoke artifacts remain historical and are never overwritten;
- fresh post-migration smokes use a new versioned artifact root.

This is an engineering resilience tradeoff, not a scientific change. Restoring
periodic crash recovery would require relaxing the one-call invariant or using
a separately reviewed device-native checkpoint mechanism.

## Skeptical Plan Audit

| Risk | Audit verdict and mitigation |
| --- | --- |
| Wrong baseline | Guarded by exact target/adapter signatures and unchanged recipe/seed/objective/optimizer fields. |
| Proxy promoted | Training and smoke diagnostics remain engineering evidence only; HMC remains the promotion computation. |
| Missing stop condition | Stop on any NumPy/loop/XLA/device/signature/state/parity veto or if the bounded smoke exceeds 15 minutes. |
| Hidden NumPy through package initialization | A subprocess records imported `bayesfilter.*` modules and statically rejects NumPy imports/calls in their source; package initializers become lazy where needed. TensorFlow itself may import NumPy internally. |
| Hidden host loop disguised as chunks | Exactly one compiled training-program invocation is permitted per call, not a Python loop over chunks. |
| Misleading speed claim | Timing is descriptive unless compared in a separately predeclared benchmark with warm-call separation. |
| Artifact no longer supports recovery | Explicitly accepted as terminal-only; contract and tests are updated rather than pretending cadence files still exist. |
| Stale artifacts | New smoke root and terminal filenames; old attempts remain immutable. |
| Environment mismatch | CPU-hidden focused tests and trusted GPU/XLA smoke are both required. |
| Plan could pass while graph silently unrolls | Multi-step concrete graph must contain functional control flow and have one host invocation; operation inventory is recorded. |

Audit conclusion before execution: **PASS AFTER REVISION**. The original idea
was incomplete because it ignored package-import NumPy and the incompatibility
between periodic JSON checkpoints and uninterrupted XLA. This plan now exposes
both boundaries and has artifacts that answer the stated engineering question.

Claude review note: a trusted fixed-token health probe returned
`CLAUDE_PROBE_OK`, but two bounded single-path review prompts returned no
substantive output. This is reviewer/prompt-path unavailability, not an
`AGREE` verdict. No Claude claim is used to authorize execution. The local
skeptical audit above is the operative review, consistent with the repository's
review-proportionality rule.

## Implementation Phases

### Phase 1: Import-Closure Isolation

- Make inference/testing package exports lazy without changing documented
  public names.
- Replace broad package imports in the exact target with narrow module imports.
- Remove the training engine's dependency on NumPy-backed runtime helpers by
  using local standard-library JSON/hash utilities.
- Add a subprocess test that imports the training engine and strict training
  harness, inventories imported `bayesfilter.*` modules, and rejects direct
  NumPy imports or calls in every repository-owned source file in that closure.

### Phase 2: Single-Program Training Engine

- Refactor the optimizer step into a graph body.
- Execute `[completed_steps, terminal_step)` with `tf.while_loop` inside one
  `tf.function(jit_compile=True)` invocation.
- Collect first step, heartbeat cadence, and terminal diagnostics with fixed
  tensor arrays.
- Aggregate hard validity flags inside the graph and fail closed after the one
  terminal synchronization.
- Write one terminal checkpoint, latest record, progress rows, and optional
  frozen transport after the graph returns.

### Phase 3: Strict Target-Specific Training Harness

- Remove the NumPy-heavy parent module from all `train`-stage imports and calls.
- Load and validate target, mass geometry, target signatures, artifact loader,
  probes, parity, and held-out batches through TensorFlow/standard-library
  helpers only.
- Batch the eight held-out seed evaluations in graph control flow or a single
  tensorized call; do not use a Python loop over held-out batches.
- Amend the campaign contract and CLI metadata to the graph-native terminal-only
  checkpoint policy while preserving scientific fields.
- Route Phase 5/6 lazily to their legacy module so training never imports it.

### Phase 4: Verification And Fresh Smoke

Required checks:

1. compile and diff checks;
2. focused CPU-hidden unit tests;
3. source/AST checks for no per-step Python loop, `tf.py_function`,
   `tf.numpy_function`, or direct NumPy import in the closure;
4. subprocess import and completed-training checks showing that every imported
   repository-owned module is free of NumPy imports/calls;
5. exact uninterrupted-versus-resumed final state parity;
6. graph operation inventory proving functional while-loop control flow;
7. trusted GPU/XLA two-step then five-step smoke in fresh versioned directories;
8. terminal frozen reload, explicit-score parity, target-status, and device
   checks.

## Stop Conditions

Stop without launching a screen or serious training job if any required check
fails, if the strict closure imports NumPy, if XLA needs a host callback, if
seed/state parity changes, if target signatures drift, or if GPU outputs leave
the selected GPU. A failed current implementation is an engineering repair
trigger, not evidence against NeuTra.

## Execution Authorization And Budget

The user requested plan execution. Authorized live compute in this migration is
limited to focused CPU-hidden tests and at most two fresh trusted GPU/XLA smokes
of at most five training steps each, with a combined 15-minute wall limit. The
four 500-step screen arms and two 5,000-step final runs are not launched by this
migration plan.
