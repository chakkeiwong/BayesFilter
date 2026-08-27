# BayesFilter HMC Tuning Interface Documentation And Verification Plan

Date: 2026-08-27

Status: `REVIEWED_READY_FOR_IMPLEMENTATION`

Baseline commit: `553208502e2e43e6883ad9467381eb5c3e82867a`

Companion review handoff:
`docs/plans/bayesfilter-hmc-tuning-interface-documentation-claude-audit-handoff-2026-08-27.md`

Review result:
`docs/plans/bayesfilter-hmc-tuning-interface-documentation-claude-audit-result-2026-08-27.md`

Review disposition: Claude (Fable 5) returned `VERDICT: AGREE` on 2026-08-27
with no blocker or major finding. The two minor observations confirm that the
capability registry remains an architecture hypothesis until Phase 2 passes and
that Phase 4 must characterize ESS without inventing a threshold. Codex accepts
both observations as already enforced by this plan; no substantive plan change
was required.

Scope of agreement: the review establishes that this plan is sufficient to
test and document the interface. It does not establish that the current tuner
implements the intended R-hat or ESS admission behavior, that the proposed
neural-force binding is feasible, or that any sampler is scientifically valid.
Those remain explicit Phase 1--4 evidence gates. The baseline commit above is
the inspected source baseline; the implementation-start commit must still be
recorded and re-audited as required by Phase 0.

## 1. Purpose

BayesFilter has substantial HMC tuning machinery, but its monograph does not
identify the public tuning interfaces, their prerequisites, the stages each
interface owns, or the limits of low-level chain runners. This omission has
already caused a downstream dsge_hmc agent to call
`run_full_chain_neural_force_hmc` directly and describe four transitions of
fixed-mass, fixed-`L` step-size adaptation as a serious BayesFilter tuning rung.
That classification was wrong.

This plan makes the tuning procedure understandable to people and agents, makes
the route guidance derive from a machine-readable repository contract, and
tests the documented claims against executable behavior. It also closes or
clearly records the current public-interface gap for an arbitrary frozen
position-only force.

This is an engineering and documentation plan. It does not authorize a sampler
campaign, a default-policy change, a downstream lock update, or an HMC
scientific claim.

## 2. Direct Classification Of The Current State

Claimed target: a user or agent should be able to select the right BayesFilter
HMC tuning entry point and understand exactly which mass, step-size,
trajectory-length, screening, verification, and repair stages it executes.

Quantity currently provided: a high-level fixed-trajectory tuning discussion in
the monograph, a route-role registry in Python, public API docstrings, internal
stage helpers, and historical audit notes. These pieces are not joined into a
single tested interface guide.

Verdict: the documentation is **incomplete** relative to the claimed target.
The existence of the registry is correct by source inspection, but registry
membership alone does not prove the behavioral claims of either active route.
The correctness of final R-hat and ESS admission is **not checked on the
baseline by this plan artifact**; an existing audit reports material defects
that must be reproduced with focused tests before new prose makes an admission
claim.

## 3. Measured Baseline

The following facts were inspected at the baseline commit. Line numbers are
baseline anchors and may move during implementation.

| Fact | Evidence anchor | Classification |
| --- | --- | --- |
| The route registry has exactly two active artifact-authority interfaces: `tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`. | `bayesfilter/inference/tuning_contract.py:143-155`; `tests/test_hmc_tuning_contract.py:22-30` | `correct by source and focused test contract` |
| The monograph describes a promoted fixed-trajectory algorithm and public/private handoffs, but does not name either public API or its configuration type. | `docs/chapters/ch21_hmc_for_state_space.tex:55-134`; absence search over files included by `docs/main.tex` | `documentation deficiency` |
| `tune_hmc_kernel` owns the ordinary public call but exposes no `run_full_chain` parameter. | `bayesfilter/inference/hmc_kernel_tuning.py:14188-14213` | `correct by signature inspection` |
| The internal tune/verify/repair helper accepts an injected runner but explicitly says it is not the final one-call API. | `bayesfilter/inference/hmc_kernel_tuning.py:11670-11717` | `correct by signature and docstring inspection` |
| The canonical ordinary implementation says BayesFilter owns mass, step size, leapfrog count, screening, verification, and repair. | `bayesfilter/inference/hmc_kernel_tuning.py:13031-13053` | `intended behavior; must be backed by behavioral tests before documentation treats every stage as established` |
| The fixed-transport public tuner accepts a runner injection only after building and identity-binding a genuine frozen transformed target. | `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:694-789` | `correct by source inspection; not a generic arbitrary-force escape hatch` |
| The low-level neural-force runner dual-averages step size only when requested; it receives a fixed `num_leapfrog_steps` and uses an identity inverse mass in its supplied mass coordinates. | `bayesfilter/inference/neural_force_hmc.py:727-816` | `correct by source inspection` |
| If its adapter lacks the native affine map, the neural-force runner falls back to `direct_fixed_transport_z` with identity factor and zero center. | `bayesfilter/inference/neural_force_hmc.py:750-765` | `correct by source inspection; direct use does not establish mass tuning` |
| The neural-force runner currently rejects target-status tracing other than `none`. | `bayesfilter/inference/neural_force_hmc.py:746-750` | `interface limitation` |
| The August 22 audit reports that final ordinary admission does not consume the verifier R-hat pass and that no ESS criterion gates canonical admission. | `docs/audits/bayesfilter-hmc-tuning-full-audit-2026-08-22.md:16-48` | `reported defect; reproduce on the baseline before accepting, repairing, or documenting it as current behavior` |
| MacroFinance has many direct ordinary-tuner consumers, while dsge_hmc contains context-dependent contracts around direct neural-force runner use. | sibling-repository tests found by an exact-symbol search on 2026-08-27 | `migration surface; sibling state must be re-read at execution time` |
| dsge_hmc pins a BayesFilter commit and treats a lock change as an owner-selected compatibility event. | `/home/ubuntu/python/dsge_hmc/config/bgs-backend-lock.json` and its repository instructions | `external migration boundary` |

## 4. Source-Of-Truth Order

When sources disagree, implementation and review use this order:

1. Checked mathematical target and explicit project policy.
2. Executable public signature, fail-closed validators, and behavioral tests at
   the current commit.
3. Machine-readable route and capability registry generated from those checks.
4. Reference guide and LaTeX prose generated from or linked to that registry.
5. Historical plans, audits, result notes, and downstream prose.

The registry is authoritative for route role and artifact authority only. It is
not, by itself, evidence that mass adaptation, trajectory selection, or fresh
verification executed correctly. A docstring describes intent; a behavioral
test establishes the checked behavior for the tested case.

## 5. Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can BayesFilter provide one precise, agent-usable HMC tuning guide whose route selection and stage claims remain synchronized with executable public behavior? |
| Exact baseline | Commit `553208502e2e43e6883ad9467381eb5c3e82867a`, followed by an explicit rebase-of-evidence step if `main` moves before implementation. |
| Primary pass criterion | Every normative route and stage claim is either generated from a validated capability record or tied to a focused behavioral test; the monograph builds; executable examples pass; downstream route guards reject the observed misuse. |
| Hard vetoes | A documented public signature does not exist; a claimed stage is bypassed in a behavioral test; a failed verifier can still issue a final handoff; a low-level runner can issue canonical tuning authority; fixed-transport tuning is presented as generic force injection; generated docs drift from the registry; examples do not execute; or the LaTeX document does not build. |
| Repair triggers | Ambiguous route semantics, a stale source anchor, a red behavioral characterization, unsupported numeric provenance, downstream lock mismatch, or a neural runner that cannot satisfy the canonical stage and identity contracts. |
| Explanatory diagnostics | Route counts, test coverage, generated diff, build log, example runtime, and downstream symbol inventory. They do not prove sampler quality. |
| What will not be concluded | No posterior convergence, target correctness, sampler superiority, performance, GPU readiness, production readiness, or scientific validity claim follows from documentation tests. |
| Preserved artifact | This plan, the Claude audit result, capability data and generated fragments, focused test output, LaTeX build result, downstream migration notes, and a terminal implementation result note. |

## 6. Research Intent Ledger

Although no research run is authorized, the interface work can still create
scientific misinterpretation. This ledger guards that boundary.

| Item | Ledger |
| --- | --- |
| Main question | Does the guide tell an agent which HMC tuner is valid and what evidence that tuner actually establishes? |
| Mechanism under test | Machine-readable capabilities, generated route tables, executable examples, behavioral admission tests, and downstream static guards. |
| Expected failure mode | Documentation repeats intended behavior that executable admission does not enforce, or treats a low-level runner as a full tuner. |
| Promotion criterion | Documentation correctness gates in Section 15 all pass. |
| Promotion veto | Any contradiction between prose, registry, signature, artifact schema, or tested behavior. |
| Continuation veto | The mathematical target is unclear; runner injection changes the target or coordinate measure; required identity cannot be bound; or a downstream pin change needs an owner decision that has not been made. |
| Repair trigger | A red contract test, missing capability field, stale generated fragment, or Claude `REVISE` finding. |
| Explanatory diagnostics | Build logs, route inventories, symbol searches, and example output. |
| Must not conclude | A correctly documented tuner is not thereby a correct target, a converged sampler, or a good scientific default. |

## 7. Skeptical Plan Audit Before Execution

### 7.1 Wrong baseline

Risk: documenting the pre-merge commit, the dsge_hmc pinned commit, or a
historical audit as though it were current `main`.

Control: record `git rev-parse HEAD`, verify a clean worktree, rerun route and
signature inventories, and create a contradiction ledger before edits. If
`main` moved, update the baseline table rather than carrying line numbers
forward silently.

### 7.2 Proxy metrics promoted into proof

Risk: a registry entry, import test, prose build, acceptance rate, or short
chain is treated as proof that the complete tuning/admission procedure is
correct.

Control: route identity, stage behavior, artifact authority, and final
admission each receive separate tests. No stochastic tuning result is a pass
criterion for this documentation project.

### 7.3 Missing stop conditions

Risk: the implementation keeps writing prose after discovering that the public
API cannot support the claimed neural-force workflow or that final verification
does not gate handoff.

Control: stop normative documentation of the affected claim when a behavioral
test is red. Mark it `known defect` or `unsupported` until code repair is
reviewed and green. Do not convert a documentation task into an unreviewed
sampler-policy change.

### 7.4 Unfair comparison

Risk: ordinary HMC, fixed nonlinear transport HMC, and arbitrary neural-force
mechanics are compared as interchangeable tuning procedures.

Control: describe prerequisites and coordinates before stages. The guide uses
a decision tree, not a performance ranking. NUTS and historical grid tools are
out of scope except for route-role warnings.

### 7.5 Hidden assumptions and silent defaults

Risk: the plan invents a target acceptance, R-hat threshold, ESS threshold,
chain count, draw count, or runner contract and then presents it as BayesFilter
policy.

Control: numeric policy is read from current configuration and traced to code,
policy, derivation, or literature. A value without provenance is labeled an
inherited or convenience hypothesis. Tests inject pass/fail sentinels where a
specific scientific threshold is unnecessary.

### 7.6 Stale downstream context

Risk: BayesFilter guidance is correct at its commit but agents use a different
pinned version or copied API.

Control: generated docs include schema/version identity; downstream guides
name their BayesFilter commit; each downstream repository owns a contract test
against that pin. A lock update is a separate explicit migration action.

### 7.7 Environment mismatch

Risk: a documentation test initializes GPU state or a CPU fixture is reported
as GPU/XLA evidence.

Control: all tests in this plan are deliberate CPU-only checks with
`CUDA_VISIBLE_DEVICES=-1` set before TensorFlow import. They make no GPU claim.
No package installation or environment mutation is authorized.

### 7.8 Commands whose artifacts do not answer the question

Risk: a successful LaTeX build is used as the only documentation test, or a
large sampler run obscures a contract failure.

Control: use the smallest tests in Section 14. The generated-doc check answers
drift, behavioral tests answer semantics, examples answer usability, and
`latexmk` answers document integrity. Each result is recorded separately.

Audit result: `CONDITIONAL_PASS_FOR_REVIEW`. The plan has a correct baseline,
explicit nonclaims, falsifiable gates, and stop conditions. Implementation must
not begin until the Claude audit is adjudicated and status is changed to
`REVIEWED_READY_FOR_IMPLEMENTATION`.

## 8. Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Two active public tuners | Measured registry and existing test at baseline | Current repository authority | Registry can describe intended identity while behavior drifts | Registry plus signature and behavioral tests | `measured baseline` |
| New chapter immediately after Chapter 21 | Current `docs/main.tex` organization | Keeps interface detail next to HMC mechanics | Split narrative or duplicate definitions | Build and cross-reference test | `convenience choice` |
| A concise Markdown agent guide in addition to LaTeX | Observed agent use and lack of an operational route guide | Agents need a searchable exact-symbol reference | Two hand-maintained sources drift | Generate shared tables; test cross-links and required clauses | `task-specific requirement` |
| Capability registry as shared table source | Existing route registry | Removes duplicated route classification | Registry overclaims untested behavior | Capability records require evidence anchors and tests | `architecture hypothesis` |
| Keep exactly two active artifact-authority routes if feasible | Existing reviewed registry | Minimizes public surface and preserves route distinction | Ordinary tuner becomes a bag of unsafe raw callbacks | Typed binding and full-stage conformance tests | `preferred compatibility hypothesis` |
| Typed runner binding rather than a bare callable | Current identity gap and internal raw hook | Can bind target, force, coordinates, telemetry, and source closure | Over-designed binding still bypasses a stage | Fake-runner call ledger through every stage | `architecture hypothesis` |
| Direct neural-force runner remains mechanics-only | Its current fixed-mass/fixed-`L` signature and identity fallback | Prevents recurrence of observed misuse | Downstream calls it and labels output canonical | Artifact-authority rejection and downstream static test | `measured limitation, proposed guard` |
| Existing numeric thresholds | Current config/policy only | Documentation should report implementation, not invent policy | Unsupported values harden into defaults | Provenance table and config-reflection test | `must be re-audited` |
| CPU-hidden focused tests | Repository GPU policy and task type | These are interface tests, not performance runs | Accidental GPU initialization or false GPU claim | Environment assertion before TensorFlow import | `reviewed task default` |
| Tracked generated route fragments | User direction: claim-supporting generated files are tracked | They are evidence for normative interface claims | Stale generated text | Renderer `--check` and clean-diff test | `task requirement` |
| Ignored LaTeX and test build products | Existing `.gitignore` policy | Build products do not support promotion claims | Untracked workspace debris | `git ls-files --others --exclude-standard` is empty | `existing repository policy` |

No new scientific threshold is approved by this table.

## 9. Normative Documentation Contract

The finished documentation must define these terms before using them:

- **Tuner**: a public repository-owned orchestration route that may issue a
  canonical tuning artifact only after all required stages and fresh
  verification pass.
- **Chain runner**: mechanics for executing a fixed configuration. It is not a
  tuner merely because it can adapt one parameter.
- **Stage helper**: an internal or diagnostic part of the ladder. Direct use
  does not inherit public artifact authority.
- **Tuning artifact**: scope-bound evidence of the selected fixed kernel and
  the gates that ran. It is not retained posterior evidence.
- **Final handoff**: a replayable private kernel handoff plus a redacted public
  status record, emitted only under the tested admission rule.
- **Ordinary coordinates**: the adapter coordinates in which the ordinary
  tuner owns geometry and mass adaptation.
- **Fixed-transport coordinates**: latent `z` coordinates of one frozen,
  identity-bound nonlinear transport; changing the transport invalidates the
  tuning scope.
- **Neural-force mechanics**: a fixed force and exact endpoint potential used
  by a chain runner. This does not imply a fixed nonlinear transport and does
  not independently tune mass or `L`.

The guide must answer, in this order:

1. What mathematical target and coordinate system are being sampled?
2. Is there a genuine frozen nonlinear transport with an exact transformed
   value/score contract?
3. Which public tuner is eligible for this case?
4. Which parameters and stages does that tuner own?
5. What fresh evidence gates the final handoff?
6. What artifact may the downstream caller consume?
7. Which claims remain forbidden after a pass?

## 10. Route Decision Table To Be Generated

The route registry will be extended, or paired with an immutable capability
registry, so generated documentation can answer more than active/historical
role. Each public or discoverable route record must include:

- qualified interface name and schema version;
- role and canonical artifact authority;
- algorithm family;
- required target/value/score authority;
- coordinate-system prerequisite;
- mass policy (`adaptive`, `fixed`, or `not owned`);
- step-size policy;
- trajectory-length or leapfrog-count policy;
- fresh verification policy;
- target-status telemetry requirement;
- runner-injection policy;
- identity and source-closure fields bound into artifacts;
- replacement route for non-active entries;
- explicit forbidden uses and nonclaims; and
- evidence anchors naming the tests that support each behavioral capability.

Unknown capability is a valid fail-closed value. A renderer must refuse to
print `supported` for a capability without an evidence anchor.

The human decision tree is:

1. For ordinary exact-value/exact-score HMC where BayesFilter owns geometry,
   use `tune_hmc_kernel`.
2. For HMC behind a genuine frozen nonlinear transport satisfying the fixed
   transport protocol, use `tune_fixed_transport_hmc_kernel`.
3. For an arbitrary position-only or learned force without that transport
   contract, no canonical public tuning route exists at the baseline. Do not
   call `run_full_chain_neural_force_hmc` and label it full tuning.
4. Diagnostic and historical routes may debug mechanics only and cannot issue
   canonical artifacts.

Step 3 remains the truthful guide until Phase 3 passes.

## 11. Neural-Force Interface Design Gate

### 11.1 Preferred compatibility design

The first design to test is a typed, repository-owned runner binding accepted
by `tune_hmc_kernel`, while retaining exactly two active artifact-authority
routes. A bare `Callable` is insufficient. The binding must carry and validate:

- stable runner and algorithm identity;
- force identity and exact endpoint-target identity;
- target and coordinate scope;
- compatibility with the affine mass wrapper created by the ordinary ladder;
- support for arbitrary fixed `L` and `epsilon` chosen by the ladder;
- finite, divergence, energy-error, movement, and target-status telemetry;
- deterministic seed handling and chain execution mode;
- XLA/dtype/backend capabilities;
- source dependency closure and artifact serialization; and
- a statement that the runner cannot issue tuning authority on its own.

The binding must be threaded through bootstrap, windowed mass adaptation,
fixed-mass step tuning, `L` selection, candidate screening, fresh verification,
and repair. A fake runner call ledger must show which stages invoked it and
which mass/`L`/`epsilon` values it received.

For a claim-bearing neural-force use, lack of the native affine adapter route
must fail closed at the public binding. The current
`direct_fixed_transport_z` identity fallback may remain a mechanics diagnostic,
but it must not be eligible for canonical tuning artifacts.

### 11.2 Rejection condition

Reject the preferred design if any of the following is true:

- the runner cannot expose target-status or equivalent fail-closed evidence
  required by the canonical ladder;
- the force and exact endpoint target cannot be bound to the same coordinates;
- mass adaptation or `L` selection silently executes with the default TFP
  runner instead of the requested algorithm;
- runner identity cannot be preserved through artifacts and replay;
- the ordinary tuner requires algorithm-specific forks that make its public
  contract misleading; or
- fixed-transport semantics would be claimed without an actual frozen
  transport.

If rejected, stop. Write a reviewed API decision record for a separate
`tune_neural_force_hmc_kernel` route and update the registry only after its own
full-ladder and admission tests pass. Do not route arbitrary force mechanics
through `tune_fixed_transport_hmc_kernel` merely because that function exposes
`run_full_chain`.

## 12. Documentation Deliverables

### 12.1 Normative monograph chapter

Add `docs/chapters/ch21b_hmc_tuning_interfaces.tex` and include it immediately
after `ch21_hmc_for_state_space` in `docs/main.tex`. It will contain:

- terms and source-of-truth rules from Section 9;
- the generated route decision table;
- ordinary tuner inputs, outputs, stages, and scope identity;
- fixed-transport prerequisites and transformed-target equations;
- the neural-force distinction and current/implemented status;
- screening, fresh verification, repair, and handoff semantics;
- diagnostic and historical route restrictions;
- a failure-interpretation table separating implementation failure, tuning
  candidate failure, target invalidity, and scientific nonclaims;
- numeric-policy provenance references; and
- executable example listings rather than copied code snippets.

### 12.2 Agent-facing guide

Add `docs/reference/hmc-tuning-interface.md`. It will be short enough to read
before modifying a consumer and will include:

- a first-page route decision tree;
- exact imports and config types;
- `DO`, `DO NOT`, and `STOP` conditions;
- an artifact acceptance checklist;
- a warning that a high acceptance rate from fixed `M=I`, fixed `L=1`, and a
  short epsilon adaptation is not evidence against HMC or the target;
- current schema/version compatibility instructions; and
- links to the monograph, generated capability table, and tests.

### 12.3 Executable examples

Add focused examples under `docs/examples/`:

- `hmc_tuning_ordinary.py`: smallest valid ordinary public-tuner call using a
  deterministic analytic target fixture;
- `hmc_tuning_fixed_transport.py`: smallest valid frozen-transport call;
- `hmc_tuning_route_selection.py`: pure route/capability selection with one
  expected rejection for a low-level neural-force runner.

The LaTeX chapter must include these files with `\lstinputlisting` or an
equivalent source inclusion so displayed code and executed code are identical.
Examples use tiny CPU-hidden fixtures and make no sampler-quality claim.

### 12.4 Agent policy

Add a concise HMC tuning section to BayesFilter `AGENTS.md` that links to the
agent guide and states only stable rules: consult the capability registry,
choose a public artifact-authority route, never equate a chain runner with a
tuner, and fail closed on an unsupported algorithm/coordinate combination.
Detailed stage facts stay in the generated guide rather than being duplicated
in policy prose.

### 12.5 Generated claim-supporting files

Add `scripts/render_hmc_tuning_interface_docs.py` with `--check`, and track:

- `docs/generated/hmc_tuning_route_table.md`; and
- `docs/generated/hmc_tuning_route_table.tex`.

These generated fragments are tracked because they directly support normative
interface claims. LaTeX auxiliaries, logs, caches, coverage files, and temporary
example outputs remain ignored. At each phase close,
`git ls-files --others --exclude-standard` must print nothing.

## 13. Behavioral Correctness Tests

Add `tests/test_hmc_tuning_documentation_contract.py` and narrowly extend
existing tuner tests. The suite must cover:

### 13.1 Registry and generation

- exactly one record per discoverable tuning route;
- exactly the reviewed active artifact-authority routes;
- capability enums and evidence anchors validate;
- non-active routes name an active replacement and cannot issue authority;
- generated Markdown and TeX are byte-for-byte current;
- every interface named by the guide resolves to its documented signature;
- every example import resolves; and
- changing a capability in a temporary registry fixture makes `--check` fail.

### 13.2 Ordinary tuner

- the public route owns or deliberately fixes mass according to config;
- step-size and `L` stages use the same selected runner and coordinate binding;
- fresh verification uses a seed/draw segment not reused for candidate
  selection;
- repair cannot issue a handoff without rerunning the required screens; and
- a forced verifier result with `passed=False`, including an R-hat-cap failure,
  cannot produce a final kernel or public success artifact.

### 13.3 ESS and admission truthfulness

- characterize whether bulk/tail ESS are disabled, explanatory, or required in
  the current config;
- when an ESS gate is configured as required, force it to fail and assert
  non-admission;
- do not introduce a new default ESS number in this documentation project;
- if a new default threshold is desired, stop and create a separate numerical
  provenance and evidence plan; and
- the prose must render the tested current status, including `known defect` if
  a red admission test is not repaired.

### 13.4 Fixed transport

- changing any frozen transport identity invalidates scope reuse;
- the route uses the transformed value including the declared Jacobian term and
  the matching score contract;
- the identity-`z` mass policy is not described as ordinary mass adaptation;
- injected runner use remains bound to the frozen-transport adapter; and
- a raw arbitrary-force object cannot satisfy the transport prerequisite.

### 13.5 Neural force

- direct low-level execution is classified as mechanics-only and cannot issue a
  canonical tuning artifact;
- the public binding rejects `direct_fixed_transport_z` for claim-bearing use;
- the fake runner receives every mass, `L`, and `epsilon` selected by all
  relevant ladder stages;
- missing telemetry or identity fails closed;
- force and exact endpoint target remain coordinate-consistent; and
- a fixed-transport neural-force example remains valid only when a genuine
  frozen transport exists.

### 13.6 Negative documentation mutations

Tests must temporarily mutate or synthesize claims and prove rejection for at
least these false statements:

- "all exported tuning functions are public tuners";
- "the neural-force runner tunes the mass matrix";
- "dual averaging chooses `L`";
- "fixed-transport tuning accepts an arbitrary force without a transport";
- "acceptance alone verifies convergence"; and
- "a failed fresh verifier may still emit a final handoff."

These mutation cases are documentation tests, not scientific experiments.

## 14. Phased Work Program

### Phase 0: Review and baseline lock

1. Send only this plan path through the companion Claude handoff.
2. Adjudicate every finding in a tracked review result.
3. Change status only after all blockers are resolved.
4. At implementation start, record commit, worktree state, Python/TensorFlow/
   TFP versions, and deliberate CPU-only environment.

Gate: Claude returns `VERDICT: AGREE`, or every `REVISE` item is resolved and a
bounded rereview agrees.

### Phase 1: Contradiction and consumer inventory

1. Inventory exported/discoverable tuning symbols with
   `scripts/inventory_hmc_tuning_routes.py --check`.
2. Map every normative sentence in Chapter 21 and both audit memos to source or
   tests.
3. Inventory MacroFinance and dsge_hmc consumers at their then-current commits.
4. Write a contradiction ledger classifying each item `correct`, `wrong
   relative to implementation`, `unsupported`, `not checked`, or `historical`.

Gate: no unresolved contradiction is silently carried into generated docs.

### Phase 2: Capability registry and renderer

1. Define strict capability types and validation.
2. Bind evidence anchors to behavioral tests.
3. Extend discovery so new unclassified tuning-like routes fail tests.
4. Implement Markdown/TeX rendering and `--check`.
5. Generate and track both route tables.

Gate: registry, discovery, generated output, and negative drift tests pass.

### Phase 3: Neural-force public-interface decision

1. Write characterization tests for the baseline low-level runner.
2. Prototype the typed ordinary-tuner binding without changing route authority.
3. Trace the fake runner through every canonical stage.
4. Verify coordinate, target, mass, telemetry, and artifact identity.
5. Accept the preferred design or stop with the separate-route decision record
   required by Section 11.2.

Gate: no public guide claims neural-force full tuning until an end-to-end
contract test proves the corresponding route.

### Phase 4: Admission characterization and bounded repair

1. Reproduce the August 22 R-hat finding with a forced verifier failure.
2. Characterize ESS role with configuration-driven tests.
3. Repair final admission only if the change restores already-declared
   verifier semantics without selecting a new scientific threshold.
4. If a new ESS default or verification budget is needed, stop and split that
   default-policy decision into its own reviewed evidence plan.

Gate: prose labels the exact green behavior. Known-red behavior is documented
as a defect, never as an implemented guarantee.

### Phase 5: Write and generate documentation

1. Add the normative chapter, agent guide, examples, generated fragments, and
   minimal AGENTS rule.
2. Cite symbols, schemas, equations, behavioral tests, and numeric provenance.
3. Remove or correct contradictory prose in Chapter 21 rather than leaving two
   normative versions.
4. Label intended, implemented, tested, historical, and unsupported statements
   distinctly.

Gate: documentation contract and negative mutation tests pass.

### Phase 6: Build and usability verification

1. Execute examples as tests in the deliberate CPU-only environment.
2. Build the monograph with halt-on-error.
3. Reject undefined references, unresolved citations, duplicate labels, or
   missing included files.
4. Have a fresh agent answer three route-selection fixtures using only the
   agent guide; compare answers against the registry without using sampler
   output as evidence.

Gate: all build, example, and route-selection checks pass.

### Phase 7: Downstream migration

1. Prepare separate, reviewable patches for MacroFinance and dsge_hmc AGENTS
   guidance and contract tests.
2. Replace direct low-level neural-force tuning claims with the supported public
   binding, or mark the route diagnostic until support exists.
3. Record each consumer's BayesFilter schema and commit.
4. Do not change the dsge_hmc backend lock without the explicit owner selection
   and compatibility validation required by that repository.

Gate: each downstream repository passes its own contract tests against its
declared BayesFilter pin. A BayesFilter-only pass cannot certify downstream
migration.

### Phase 8: Terminal audit and closeout

1. Run the complete focused matrix in Section 15.
2. Run the Claude terminal audit on the result note using the same one-path
   protocol.
3. Record a decision table, inference-status table, changed files, commands,
   environment, limitations, and remaining nonclaims.
4. Commit only source, tests, normative docs, claim-supporting generated
   fragments, and review/result evidence. Verify no untracked visible files.

Gate: all acceptance criteria pass and terminal review findings are adjudicated.

## 15. Exact Verification Matrix

Run from `/home/ubuntu/python/BayesFilter`. The commands are deliberate CPU-only
interface checks; they must be prefixed before any TensorFlow import.

```bash
git rev-parse HEAD
git status --short --branch
python scripts/inventory_hmc_tuning_routes.py --check
python scripts/render_hmc_tuning_interface_docs.py --check
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_hmc_kernel_tuning_windowed_mass.py -k 'verification or handoff or runner'
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_fixed_transport_hmc_tuning.py tests/test_fixed_transport_hmc_binding.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_neural_force_hmc.py -k 'tuning or runner or coordinate or artifact'
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_route_selection.py
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_ordinary.py
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_fixed_transport.py
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
test -f main.log
! rg -n 'Undefined references|Citation .* undefined|multiply defined' main.log
cd ..
git diff --check
git ls-files --others --exclude-standard
```

The new documentation contract and example files must be created under the
exact names above. Any later rename requires updating this plan before the
matrix runs. Do not create a broad full-suite or GPU gate merely to compensate
for missing focused tests.

Additional build-log checks must fail on `Undefined references`, unresolved
citations, multiply defined labels, or a nonzero `latexmk` result. A clean
second `latexmk` pass may be used to resolve normal references; the two-pass
allowance is an ordinary TeX build requirement, not scientific evidence.

## 16. Acceptance Criteria

The work is complete only when all of the following are true:

- The monograph names every active public HMC tuner and its exact prerequisite.
- The agent guide yields the same route decision as the validated registry.
- A low-level chain runner is never described as a complete tuner.
- Every mass, epsilon, `L`, verification, repair, and handoff statement has a
  behavioral evidence anchor or an explicit unsupported label.
- Fixed-transport and arbitrary-force semantics are not conflated.
- R-hat and ESS are described according to tested admission behavior.
- False-claim mutation tests fail as intended.
- Generated route tables are current and tracked.
- Executable examples and the LaTeX build pass.
- Downstream migration status and version pins are explicit.
- The result note preserves hard vetoes, uncertainty, nonclaims, and next work.
- `git ls-files --others --exclude-standard` is empty; generated build debris is
  ignored and claim-supporting generated fragments are tracked.

## 17. Stop And Repair Rules

- Stop on a contradiction between mathematical target and implementation.
- Stop on a public runner binding that changes coordinates or probability
  measure without an exact derivation and identity update.
- Stop on a failed fresh-verification admission test; do not document the
  intended behavior as implemented.
- Stop before choosing a new R-hat, ESS, acceptance, budget, or retry threshold
  without numeric provenance and the required evidence plan.
- Stop before changing a downstream lock or public/default API without its
  repository's required human decision.
- Repair a stale generated table by changing the source contract or renderer,
  never by hand-editing the generated fragment.
- Treat an example failure as an interface/usability defect, not as evidence
  against HMC.
- Treat a candidate tuning failure separately from target invalidity,
  implementation failure, and research-direction rejection.

## 18. Pre-Mortem

| How the work could pass but mislead | Cheap discriminating check |
| --- | --- |
| The PDF builds while route claims are false. | Behavioral claim-to-test matrix and negative mutations. |
| The registry says a route is active while it bypasses verification. | Forced failed-verifier test asserts no handoff. |
| A typed neural runner is accepted but only some stages use it. | Per-stage fake-runner call ledger with mass/`L`/`epsilon` assertions. |
| Fixed transport is used as a label for an arbitrary force. | Require transport protocol, manifest identity, and transformed-target test. |
| Numeric thresholds look authoritative but have no provenance. | Numeric-policy ledger rejects unclassified constants. |
| Guide examples import but cannot execute. | Run the exact listed files in CI. |
| Downstream agents read docs for a different commit. | Consumer pin/schema contract test. |
| Generated files conceal hand edits or workspace debris. | Renderer `--check`, `git diff --check`, and zero visible untracked files. |

## 19. Compute, Attempt, And Artifact Budget

- No GPU run, sampler benchmark, posterior chain, sweep, or long experiment is
  authorized.
- All TensorFlow checks are deliberate CPU-only tests with GPUs hidden before
  import.
- Each behavioral question starts with one deterministic fixture and expands
  only when that fixture cannot discriminate the contract.
- A command expected to exceed roughly five minutes crosses the repository's
  serious-run planning threshold and must stop for a narrower test or a new
  experiment plan.
- Package/environment mutation is out of scope.
- Implementation artifacts are written under versioned paths and never
  overwrite historical evidence.

Planned tracked closeout artifacts:

- `docs/plans/bayesfilter-hmc-tuning-interface-documentation-claude-audit-result-2026-08-27.md`;
- `docs/plans/bayesfilter-hmc-tuning-interface-documentation-and-verification-result-2026-08-27.md`;
- the capability registry, focused tests, source examples, normative docs, and
  generated claim-supporting route tables named above.

Ordinary LaTeX output, caches, logs, temporary JSON, and example scratch output
remain ignored.

## 20. Required Closeout Tables

The terminal result note must include this decision table:

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Publish or retain the guide | Pass/fail by Section 16 | List each hard veto | State remaining untested behavior | Exact smallest follow-up | Convergence, superiority, readiness, target validity |

It must also include this inference-status table:

| Evidence class | Required statement |
| --- | --- |
| Hard veto screen | Which interface/document contradictions remain, if any |
| Statistically supported ranking | `N/A`; this plan performs no stochastic ranking |
| Descriptive-only differences | Build times, test times, and consumer counts only |
| Default readiness | Not established by documentation correctness |
| Next evidence needed | Exact test, design decision, or downstream migration still required |

The post-run red-team note must state the strongest alternative explanation for
every material conclusion, what evidence would overturn it, and the weakest
part of the documentation evidence.

## 21. Review Boundary

Claude reviews this plan before implementation using the companion hands-off
memo. The first review is one-path and read-only. `VERDICT: AGREE` is valid only
if the plan can falsify its own route and admission claims, contains no silent
numeric defaults, keeps fixed transport distinct from arbitrary neural force,
and names all material downstream and version boundaries.

Review agreement authorizes preparation for implementation only after Codex
records adjudication and changes this status. It does not authorize a serious
HMC run, new default threshold, downstream lock change, or scientific claim.
