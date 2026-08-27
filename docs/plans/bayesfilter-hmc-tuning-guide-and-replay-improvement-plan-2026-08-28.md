# BayesFilter HMC Tuning Guide And Replay Improvement Plan

Date: 2026-08-28

Status: `EXECUTED_COMMIT_PENDING`

Baseline: BayesFilter `1ef88766e` (`origin/main` after a clean fast-forward on
2026-08-28).

## Purpose

Repair the agent-facing HMC tuning documentation and the non-identity geometry
replay defect identified by two independent reviews:

- the pasted Claude audit of the interface reference and downstream migration
  guidance; and
- MacroFinance
  `docs/reviews/bayesfilter_hmc_tuning_interface_guide_feedback_2026_08_28.md`.

This is an interface, deterministic replay, documentation, and test task. It is
not an HMC experiment. Verification may run a one-transition, four-row
deterministic TensorFlow mechanics fixture, but no research or claim-bearing
chain is authorized.

## Adjudicated Findings

The following findings are accepted and in scope:

1. The downstream migration snapshot is stale and must identify a concrete
   compatible BayesFilter implementation commit.
2. The guide needs a worked neural-force binding example and explicit equations
   for the potential, proposal force, kick sign, and exact Metropolis correction.
3. The guide must distinguish the two active artifact-authority tuners from the
   eight diagnostic or historical records whose registry kind is also
   `public_tuner`.
4. The current ordinary tuner is not TensorFlow-only. This is BayesFilter-owned
   migration debt under `AGENTS.md`, not authority supplied by a Claude verdict.
5. The guide must name the runner-binding schema, capability-registry schema,
   lookup-key convention, fixed-transport ESS policy, and the implementation
   source of the ordinary `1.01` R-hat handoff threshold.
6. The ordinary workflow must explain caller-supplied geometry hypotheses,
   their precedence, coordinate obligations, and the identity fallback.
7. `admitted_kernel_mechanics_payload_from_tuning_result` currently drops the
   original geometry inputs before replay. A passed result initialized from a
   non-identity covariance can therefore be reconstructed as identity and fail
   its geometry hash check.
8. The downstream negative-search claims need exact commands and path lists.

The following review statements are rejected or narrowed:

- A wrong proposal-force sign does not by itself change the invariant target in
  this kernel. The symmetric, volume-preserving proposal is corrected with the
  declared endpoint potential. A wrong endpoint potential changes the target;
  a poor or sign-reversed proposal force can instead damage acceptance and
  mixing.
- The registry contains ten `public_tuner` records, not nine: two active
  artifact-authority routes and eight diagnostic or historical routes.
- The guide already has focused literal contract coverage. The improvement is
  to bind more of that coverage to registry values and exported constants.
- MacroFinance Phase 14 failed before `tune_hmc_kernel`; this task must not claim
  to repair its incumbent-eligibility wrapper.
- A living reference guide should not hardcode its own Git commit. The dated
  downstream migration snapshot must record the compatible implementation
  commit instead.

## Engineering Questions

1. Can an in-memory passed `HMCKernelTuningResult` initialized with a
   non-identity covariance issue durable admitted mechanics without the caller
   reproducing hidden geometry hints?
2. Can the persisted mechanics payload be JSON-round-tripped and replayed while
   preserving initial and adapted mass signatures and invoking no tuning,
   target evaluation, transition, or HMC?
3. Can an agent determine the correct tuning route, construct a covariance-first
   ordinary call or neural-force binding, and reject unsupported artifacts using
   only checked public documentation and imports?

## Evidence Contract

The exact baseline is BayesFilter `1ef88766e` plus the implementation changes
made under this plan. The primary engineering pass criteria are:

- a deterministic regression first reproduces the non-identity extraction
  failure;
- result-based replay uses the geometry artifact already bound into the passed
  result and verifies its hash, mass signature, adapter identity, dimension,
  target scope, and supplied initial position;
- durable mechanics extraction and JSON-round-trip replay succeed for a
  non-identity covariance;
- initial and adapted mass signatures are unchanged; and
- spies or stubs establish that no tuner, target evaluation, transition, or HMC
  is called during extraction or replay.

Hard vetoes are a geometry, adapter, scope, schema, or mechanics hash mismatch;
loss of existing negative replay checks; a generated-document drift failure;
an example import or construction failure; or a focused test regression.

Documentation rendering, import resolution, literal term checks, and CPU fixture
tests are engineering evidence only. They do not establish target correctness,
HMC convergence, sampler quality, retained-run readiness, GPU/XLA readiness,
TensorFlow-only compliance, or scientific validity.

The preserved result will be the implementation diff, focused test output, and
a dated result note under `docs/plans`.

## Implementation

1. Add a failing non-identity result-replay regression using deterministic
   fixtures and no HMC.
2. Refactor result-based retained replay so it consumes the validated geometry
   already carried by `HMCKernelTuningResult`. Preserve the serialized-payload
   API for callers that intentionally reconstruct from explicit geometry inputs.
3. Add an isolated durable mechanics JSON round-trip regression and preserve
   every existing mismatch rejection.
4. Export a named ordinary-tuning R-hat threshold constant and replace duplicate
   ordinary handoff literals where this can be done without changing behavior.
5. Add exact, importable examples for covariance-first ordinary tuning and
   neural-force runner binding. Stub the tuner in the covariance binding test;
   examples must not launch HMC as part of the verification suite.
6. Revise the Markdown guide and LaTeX chapter to cover authority terminology,
   geometry precedence, identity fallback, neural-force equations, fixed-
   transport diagnostic policy, replay boundaries, schemas, and the current
   TensorFlow-only blocker.
7. Extend documentation tests so examples parse and public imports resolve,
   registry-derived policies agree with prose, and the runner-binding schema and
   lookup convention remain explicit.
8. Update the downstream migration guidance with reproducible downstream search
   commands, policy-and-code authority for the NumPy blocker, conditional BGS
   force classification, and a concrete implementation commit after the first
   commit is created.
9. Write the result note, inspect the staged diff, create a closeout commit that
   records the implementation commit, and push both commits to `main`.

## Verification Commands

All Python checks are deliberate CPU-only checks with
`CUDA_VISIBLE_DEVICES=-1`; no GPU or research HMC process is authorized by this
plan. The dispatcher test contains one bounded transition fixture solely to
exercise the typed graph boundary and prove that handoff remains impossible.

```bash
python scripts/render_hmc_tuning_interface_docs.py --check
python -m pytest -q \
  tests/test_hmc_tuning_documentation_contract.py \
  tests/test_hmc_tuning_contract.py \
  tests/test_hmc_kernel_tuning_geometry.py \
  tests/test_hmc_kernel_tuning_public_api.py \
  tests/test_hmc_kernel_tuning_outer_loop.py \
  tests/test_hmc_tuning_dispatch.py \
  tests/test_neural_force_hmc.py \
  -k 'not test_full_chain_runner_exposes_hmc_diagnostics'
python docs/examples/hmc_tuning_route_selection.py
python docs/examples/hmc_tuning_covariance_first.py
python docs/examples/hmc_tuning_neural_force_binding.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The two construction examples may expose a callable `main` or construction
helper so tests can stub the tuner. Direct execution must remain a tiny,
explicitly non-claim-bearing documentation fixture.

## Compute And Stop Conditions

Budget: one deterministic reproduction, one focused repair cycle, one tiny
typed-graph mechanics fixture, and at most two localized repair reruns. Expected
wall time is below five minutes excluding the document build. Stop and revise
the plan if replay requires a schema-breaking public API change, private
geometry arrays are not present in the in-memory result, a test would need a
research chain, or the TensorFlow-only migration becomes necessary to prove
this replay repair.

## Default And Assumption Audit

No admitted or artifact-authority HMC numerical default is being selected or
changed. The `1.01` value is the existing ordinary fixed-kernel handoff
threshold; this task gives it a named implementation anchor and continues to
label it as tuning admission rather than posterior proof. Identity geometry
remains a convenience fallback, not a scientifically justified default for an
anisotropic target. Covariance-first initialization remains a caller-supplied
hypothesis whose provenance, coordinates, regularization, and acceptance must
be recorded. The new TensorFlow mechanics path is diagnostic-only; its fixed
numeric policies are unqualified hypotheses recorded in its config payload,
not defaults for an admitted tuner.

The replay repair assumes the passed in-memory result retains its private
`HMCGeometryInitializationResult` and array-bearing final mass payload. Source
inspection confirms both at this baseline. The regression will fail closed if
that assumption stops holding.

## Skeptical Pre-Execution Review

Verdict: `PASS_AFTER_SCOPE_REVISION`.

- Wrong baseline: resolved by fast-forwarding a clean BayesFilter worktree to
  `1ef88766e` before planning. MacroFinance and dsge_hmc snapshots are read-only
  observations at their stated commits and dirty worktrees.
- Proxy promotion: prevented. Passing fixture and documentation tests proves
  only deterministic interface behavior; it cannot promote a sampler or target.
- Missing stop conditions: repaired above with schema, private-payload, real-HMC,
  and focused-regression stops.
- Hidden assumption: result-bound private geometry was verified in
  `HMCKernelTuningResult.geometry` and `geometry.payload(include_mass_arrays=True)`.
- Stale context: the replay functions and tests were reread after synchronizing
  `origin/main`, which included later HMC changes.
- Environment mismatch: all framework checks are CPU-only with GPU hidden. The
  task makes no GPU claim.
- Commands answering the wrong question: no stochastic run is included. The
  primary regression directly exercises the non-identity extraction and durable
  replay path identified by the review.
- Disproportionate scope: a full NumPy-to-TensorFlow rewrite of the ordinary
  tuner is excluded. The guide and migration note will fail closed for consumers
  requiring TensorFlow-only tuning, and a separate bounded migration remains
  necessary.

The plan is ready to execute without further review because its remaining risk
is localized deterministic replay and documentation behavior covered by focused
tests.

## Mid-Execution Integration Audit

The first contract run found an incomplete in-worktree runner-binding v2 change:
the capability registry named a public dispatcher that had not yet been added,
while the legacy implementation path remained directly discoverable. This was
not a defect in the synchronized `origin/main` baseline. It was unfinished work
in the implementation diff and therefore could not be documented or committed
as-is.

Disposition: preserve the v2 identity fields and deterministic-field semantics,
add the missing public dispatcher, retain the old module path as an explicit
compatibility delegate, and make route inventory classify only the dispatcher as
the artifact-authority entry point. The tensor-kernel factory is a typed binding
capability, not evidence that the full ordinary tuning ladder is TensorFlow-only.
That backend migration remains excluded. Focused registry, import, identity, and
kernel-construction tests must pass before documentation is rendered.

A concurrent in-worktree TensorFlow-only prototype was then discovered at the
dispatcher boundary. Skeptical review found that it used four-chain mean
acceptance, finite health, and metric rank as its `passed` condition, but did
not implement the ordinary route's R-hat handoff gate and explicitly rejected
XLA. It therefore cannot close the TensorFlow-only blocker or issue an admitted
kernel under the current registry contract. The prototype is preserved as
non-promoting diagnostic mechanics, with `admission_supported=False` serialized
and retained-run construction rejected. This plan does not add the missing
R-hat/XLA qualification campaign.

During closeout, concurrent downstream work temporarily changed that prototype
to set `passed` from the acceptance-and-metric heuristic and enabled retained
archive construction. That was a new tuning-authority policy, not a repair
inside this reviewed plan: it omitted the ordinary fresh-R-hat handoff gate,
retained the unqualified non-XLA path, and contradicted the guide's rule that
acceptance alone cannot authorize handoff. The authority change was rejected.
The useful TensorFlow mechanics, artifact reload, and inclusive-boundary
roundoff regression were preserved; handoff and `passed` remain forced false.
The default audit also removed the prototype's exposed numerical constructor
defaults and renamed its bare `candidate` role to
`diagnostic_candidate_screen`. The remaining fixed choices (four chains,
`float64`, identical zero starts in affine coordinates, `L=1` metric windows,
the trajectory grid, seed offsets, and inherited TFP dual-averaging internals)
are now serialized and documented as unqualified diagnostic policy. This
prevents a non-XLA, non-admitting screen from looking like a canonical candidate
or silently presenting those choices as admitted defaults.

## Execution Closeout

The reviewed scope has been executed. The non-identity durable replay
regression passes through a JSON round trip without invoking target evaluation,
tuning, a transition, or HMC. The guide, LaTeX chapter, capability registry,
typed runner binding, executable examples, and downstream migration guidance
now agree on route authority, geometry ownership, force semantics, and replay
limits.

The complete scoped test command reports `242 passed, 6 failed`. The same six
operational P4-E probe-bank fixture tests fail with the same classifications on
the untouched `1ef88766e` baseline, so they are recorded as pre-existing debt
rather than regressions. Focused replay, dispatcher, binding, documentation,
inventory, example, compilation, and generated-document checks pass. The full
manual builds, the new chapter has no overfull boxes, and representative pages
were inspected from the rendered PDF.

Detailed evidence and remaining nonclaims are recorded in
`docs/plans/bayesfilter-hmc-tuning-guide-and-replay-improvement-result-2026-08-28.md`.
Git commit and remote synchronization remain before this plan is complete.
