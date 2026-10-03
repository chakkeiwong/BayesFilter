# Claude handoff: thorough audit of the HMC candidate-set unification

Date: 2026-09-12  
Audit type: read-only, source-grounded technical, statistical, and guidebook
audit  
Repository: `/home/ubuntu/python/BayesFilter`  
Plan under review: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`  
Inspected planning baseline: commit `9be4b8fe7bad711deea61e915c6f95bc0d37649f`

This memo is a fresh handoff. It does not replace or modify the initial Claude
audit at
`docs/plans/bayesfilter-hmc-candidate-set-unification-claude-audit-2026-09-12.md`,
and it does not replace the plan. The first audit returned `VERDICT: REVISE`.
The plan was then edited as revision R1. The purpose of this review is to test
R1 independently and more deeply, including whether the proposed unification is
the right abstraction at all.

## Initial worker prompt

Use this as the first prompt. It deliberately names one exact path so the
review starts with the context memo rather than an uncontrolled repository-wide
search.

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited path or line:
/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-candidate-set-unification-claude-thorough-audit-handoff-2026-09-12.md.
Do not edit files, run commands, launch agents, initialize TensorFlow/GPU,
build the book, or review the whole repository. Read the memo, then inspect the
named plan sections and exact source paths needed to answer its questions.
Audit the R1 proposal independently: decide whether retaining every viable
(epsilon,L) candidate through bounded refinement and fresh verification can be
implemented as one honest lifecycle while preserving ordinary, frozen
transport, and typed mechanics-only target contracts. Check scientific/statistical
logic, state-machine completeness, replay and authority boundaries, backend
sequencing, migration impact, and guidebook consistency. Challenge the premise
where necessary; do not rubber-stamp the prior audit or R1. End with exactly one
terminal line: VERDICT: AGREE or VERDICT: REVISE.
```

If a cited source is insufficient, request the smallest exact next path and line
range. Do not infer source behavior from the plan alone. Do not run experiments
or tests during this planning audit; implementation and terminal qualification
are later phases.

## Why this review is needed

The owner observed a confusing HMC tuning result in a downstream M4 run. The
reported table showed only a final handoff pair per attempt:

| Attempt | L | epsilon | verification acceptance |
| ---: | ---: | ---: | ---: |
| 0 | 3 | 0.9342759554564003 | 0.7629769641 |
| 1 | 5 | 0.7826240785901211 | 0.7699239275 |

Both reported acceptance values exceeded the configured upper bound of `0.75`,
so neither final pair passed that particular screen. The natural question was:
if tuning starts from a grid of `L`, why does an attempt appear to contain only
one `L`?

The source explanation was that the grid was in fact measured or proposed, but
the ordinary procedure selected a single handoff candidate after ordering and
limited fresh verification. The full internal candidate records were not the
same thing as a set of verified handoff candidates. In the motivating M4
artifact, the relevant queue recorded three eligible candidates, only two
verification starts, and one candidate left unstarted because of
`start_quota_exhausted`. The top-level result was `budget_exhausted` with no
top-level hard veto. This establishes incomplete candidate evaluation and an
unclear user-facing contract. It does not establish that every unstarted
candidate would pass, that every candidate would fail, or that the reported
acceptance values rank kernels.

The owner remembers a different BayesFilter convention: start with a broad
candidate family, retain every candidate that remains valid, continue tuning or
verifying the survivors, and nominate a representative only as a separate
descriptive operation. The proposed plan attempts to make that convention the
common lifecycle for ordinary coordinates and frozen nonlinear transports,
while keeping their target preparation and geometry rules distinct.

The key question for this audit is therefore not merely whether a list contains
all records. It is whether the proposed procedure can honestly claim all of the
following at once:

1. a broad, finite, explicitly scoped search over `L` and independently
   qualified epsilon values;
2. retention of all non-vetoed candidates, including multiple epsilons at one
   `L`;
3. fair further work for every retained candidate under bounded compute;
4. fresh, candidate-local final verification with no inherited evidence;
5. replayable set-valued results rather than an implicit single winner; and
6. distinct authority boundaries for target correctness, mechanics, artifact
   replay, numerical validity, and scientific promotion.

## Current procedures that must be distinguished

Inspect these sources as needed. The descriptions below are context, not
substitutes for reading the implementation.

### Ordinary coordinate tuning

The ordinary public path is `tune_hmc_kernel`, with implementation and policy
material in:

- `bayesfilter/inference/hmc_kernel_tuning.py`
- `bayesfilter/inference/hmc_tuning_dispatch.py`
- `bayesfilter/inference/tuning_contract.py`
- `bayesfilter/hmc_ordinary_selection_policy.py`
- `docs/reference/hmc-tuning-interface.md`

The current path has a primary `L` grid, a refinement idea, acceptance-related
ordering, a fresh verification queue, a per-attempt start limit, and an early
`first_admission` stop. It can therefore preserve diagnostics while still
returning only one selected candidate and leaving other eligible candidates
unverified. The relevant symbols include
`_select_joint_l_epsilon_candidate` and
`_run_phase7_direct_candidate_queue`.

### Frozen nonlinear transport tuning

The frozen-transport public path is `tune_fixed_transport_hmc_kernel`, with
implementation in:

- `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py`
- `bayesfilter/inference/fixed_transport_hmc_tuning.py`

It prepares a transformed target with the required Jacobian correction and uses
identity mass in the declared latent coordinates. Its measured joint-grid and
fresh-validation rules are not automatically interchangeable with ordinary
mass adaptation or ordinary coordinates.

### Existing all-survivor helper

`bayesfilter/inference/fixed_transport_candidate_selection.py` defines
`select_fixed_transport_candidate_set`. It accepts already prepared candidates,
executes caller-supplied validation callbacks, retains rows that pass its local
screens, and can return a descriptive nominee. It does not itself tune every
`(epsilon,L)` pair, construct the transformed target, own chain state, or issue
the complete artifact-authority handoff. The current chapter 21b workflow first
tunes frozen transports and then invokes this helper.

The audit must decide whether R1's proposed treatment of this helper is
technically and socially honest: a deprecated diagnostic compatibility export,
a read-only result aggregator, or some other explicit disposition. It must not
remain an undocumented second lifecycle merely because its name sounds like the
desired protocol.

### Typed proposal-field branch

The registry also contains a typed deterministic position-field/mechanics
branch. It is not automatically an exact score. Its endpoint potential, binding,
coordinate, and authority requirements must remain explicit. A common lifecycle
may schedule it only where its capability contract is actually satisfied; a
shared facade must not upgrade it by relabeling.

## Proposed R1 design

R1 proposes one common controller after typed target preparation:

- each controller invocation owns exactly one immutable frozen `scope_id`;
- one result, `HMCTuningCandidateSetResult`, belongs to that scope;
- a separate collection can report several scope results but cannot schedule
  work, validate candidates, or issue replay authority;
- ordinary and transformed adapters preserve their distinct target, score,
  coordinate, and mass contracts;
- every primary `L` independently qualifies epsilon, and every resulting pair
  is measured before entering the survivor set;
- multiple viable epsilon values at one `L` remain distinct candidates;
- a stage barrier or equivalent persisted fair queue gives every current
  survivor its declared allocation before any survivor advances alone;
- repair and refinement children are separate records with parent lineage,
  fresh reservations, and fresh verification streams;
- candidate evidence state is separate from campaign completion state;
- a verified parent does not grant evidence to a child, and a failed holdout is
  never silently reused;
- replay requires an explicit verified member ID bound to its scope and source
  identity; and
- nomination is optional, descriptive unless separately supported by
  uncertainty analysis, and never removes retained verified members.

R1 also says that `select_fixed_transport_candidate_set` is diagnostic-only,
that P2 creates stable TensorFlow signatures and a non-default XLA
qualification mode, and that P5 may activate XLA defaults only after per-adapter
compatibility, numerical equivalence, memory, compile-cost, steady-state, and
dependency-closure checks.

The plan sections that contain the normative proposal are:

- section 1, intended outcome, scope, helper disposition, and one-scope result;
- section 3, research intent and evidence contract;
- sections 4.1-4.5, scope, coverage, diagnostic roles, scheduler, result, and
  replay semantics;
- section 5, default and numerical-assumption audit;
- section 6, implementation phases and acceptance evidence;
- section 7, guidebook rewrite specification;
- section 8, tests, commands, budget and backend checks;
- section 9, skeptical audit and pre-mortem; and
- section 11, R1 responses to the first audit findings.

Read the plan at:
`docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`.

## What the first audit found

The initial Claude audit inspected the plan, handoff, and bounded source
excerpts and returned `REVISE`. Its five material findings were:

1. the public survivor helper had no explicit migration disposition;
2. result scope cardinality was ambiguous;
3. fairness was stated without executable rules for queue mutation and resume;
4. P2 wording could make XLA default before P5 qualification; and
5. tests lacked strong cross-entry and holdout-lineage oracles.

R1 claims to address these by classifying the helper as diagnostic-only,
choosing one scope per result, specifying closed cohorts and deferred child
insertion, separating XLA qualification from activation, and requiring shared
trace plus fresh-holdout tests.

Do not assume that accepting the first audit's findings means R1 is correct.
Check whether the revisions actually close the underlying problems and whether
they introduce new inconsistencies. Also test whether the proposed common
controller is over-unified: a common lifecycle may be valid while target
preparation, mass policy, candidate evidence thresholds, and authority remain
typed; the plan must say exactly where that boundary lies.

## Questions the second audit must answer

### 1. Is the unification premise sound?

Determine whether the owner can have one sensible procedure without pretending
that ordinary and transformed HMC are the same algorithm. Check:

- whether the common object is correctly a lifecycle/controller rather than a
  single numerical tuner or a universal threshold policy;
- whether all route-specific preparation and mass rules are frozen before
  candidate comparison;
- whether the fixed-transport measured joint-grid policy and ordinary per-`L`
  epsilon qualification can share candidate states without silently changing
  either target;
- whether the typed proposal-field branch can use the lifecycle while retaining
  mechanics-only authority; and
- whether a separate per-transport result collection is sufficient for
  cross-transport reporting without pooling incompatible latent states.

If the premise needs narrowing, state the smallest corrected formulation. Do not
reject unification merely because adapters differ; reject it if the shared
semantics alter a target, diagnostic, or authority contract.

### 2. Does “retain every viable candidate” have a precise meaning?

Check the definitions of `screened`, `validating`, `verified`, `promotion_failed`,
`pending`, `incomplete`, and campaign completion. In particular:

- Is a screened candidate retained even when final verification is pending?
- Can a candidate be both verified and part of an incomplete campaign?
- Are finite but out-of-band acceptance values correctly treated as proposal or
  efficiency information rather than validity failures?
- Are multiple epsilons at a common `L` and midpoint/refinement children
  separate IDs with separate evidence?
- Does the result distinguish no viable candidate, no verified candidate,
  budget-incomplete coverage, and shared scope invalidity?
- Is a nominee always optional and non-authoritative?

Flag any overloaded field or status whose meaning changes between adapters.

### 3. Can the scheduler be implemented and resumed deterministically?

Red-team the R1 closed-cohort semantics with at least this conceptual trace:

1. candidates A and B enter a cohort;
2. A passes an early rung while B is still pending;
3. A requests repair child A1 and refinement child A2;
4. B still has reserved work;
5. execution is interrupted before the cohort closes; and
6. resume occurs with enough budget for some but not all children.

Check cohort membership, child insertion boundary, reserve ownership, tie
breaking, work-item IDs, draw ranges, parent/child terminal status, and whether
resume can reconstruct a different order. Also check a failed candidate and an
infrastructure retry. A prose promise of fairness is insufficient if two
reasonable implementations yield different traces.

The audit should identify whether the plan needs a small formal transition table
or invariants such as:

- no child starts before its admission boundary;
- no child consumes another candidate's reserved work;
- no completed work item runs twice after resume;
- every evidence record has one scope, candidate, stream, and draw range; and
- campaign completion cannot imply all candidates were verified unless the
  declared cohort and budget say so.

### 4. Are the statistical roles honest?

Check that acceptance, movement, divergences, target status, R-hat, ESS, MCSE,
observables, runtime, and gradient counts are assigned explicit roles. Verify
that the plan does not quietly use acceptance distance as a ranking, an upper
acceptance bound as a correctness veto, or a short-chain screen as posterior
convergence evidence.

Check the following subtle points:

- independent epsilon proposals are hypotheses until measured;
- acceptance does not transfer between `L` values or geometries;
- extension chunks are not counted as independent replications;
- fresh final verification is disjoint from selection data;
- optional stopping and repeated confidence checks are addressed; and
- multiplicity and uncertainty are required before calling a nominee superior
  or best.

If the plan's candidate screens are only mechanics or promotion screens, say so
plainly. If a threshold is inherited rather than target-calibrated, classify it
as a hypothesis or compatibility value.

### 5. Are scope, geometry, and replay identities sufficient?

Inspect whether `scope_id`, `search_id`, candidate ID, transport hash, target
identity, mass signature, start-bank design, backend, dtype, source closure,
RNG lineage, and draw ranges bind every claim-bearing handoff. Check:

- a changed geometry creates a new scope;
- a changed transport cannot reuse latent states or epsilon evidence;
- ordinary adaptation cannot enter identity-mass NeuTra silently;
- a verified member can be replayed without reconstructing hidden state;
- list position cannot select a member; and
- old payloads remain readable without being upgraded to new authority.

Identify any missing binding that would let a caller combine evidence from
incompatible scopes while preserving the same numeric `(epsilon,L)`.

### 6. Is the helper and migration story complete?

Trace package exports, registry records, examples, benchmark callers, and
repository-owned consumers for:

- `select_fixed_transport_candidate_set`;
- `tune_hmc_kernel`;
- `tune_fixed_transport_hmc_kernel`;
- typed force/runner bindings;
- lower-level stage helpers; and
- old selected-candidate result schemas.

Decide whether the R1 compatibility wrapper and diagnostic helper can still
silently own validation or nomination authority. Check whether the guidebook's
cross-transport example accidentally recreates the old lifecycle under a new
name. Downstream MacroFinance and dsge_hmc consumers are assigned to a bounded
P0 inventory, not an automatic cross-repository edit; assess whether that is an
honest boundary.

### 7. Does backend sequencing satisfy repository policy?

Check the complete active dependency closure, not just the top-level controller:
state, selection, artifact construction, replay, numerical adapters, and
runner. Confirm that:

- NumPy is not newly retained in a production/candidate runtime path;
- TensorFlow/TFP remains the algorithmic backend;
- stable signatures are specified before repeated execution;
- XLA qualification precedes default activation for each adapter class;
- memory-growth settings precede TensorFlow initialization on GPU; and
- CPU-hidden tests are not misrepresented as GPU evidence.

Separate a current migration blocker from a future qualification check. Do not
call the plan wrong merely because P5 evidence does not yet exist; call it
wrong if the plan permits activation without that evidence.

### 8. Is the guidebook rewrite scientifically faithful and usable?

Inspect the listed chapter surfaces and the existing conflicting narratives:

- `docs/chapters/ch21_hmc_for_state_space.tex`;
- `docs/chapters/ch21b_hmc_tuning_interfaces.tex`;
- `docs/chapters/ch22_mass_matrices.tex`;
- `docs/chapters/ch25_diagnostics.tex`;
- `docs/chapters/ch26b_neutra_transport_hmc.tex`;
- `docs/chapters/ch26c_hnn_surrogate_hmc.tex`; and
- `docs/reference/hmc-tuning-interface.md` plus generated route tables.

Check whether a reader can distinguish:

- candidate measurement from final verification;
- all retained viable members from an optional nominee;
- transport restarts from multiple HMC pairs within one transport scope;
- acceptance from validity and convergence; and
- a tuning artifact from posterior or scientific authority.

Check that the rewrite preserves equations, citations, assumptions, and source
boundaries, and that the plan requires rendered-book inspection rather than
equating successful compilation with reader-facing correctness. The audit need
not build the book, but it should flag missing chapter surfaces or a narrative
that remains contradictory by inspection.

### 9. Would the proposed tests fail the important regressions?

Assess whether the test obligations would fail each of these deliberately bad
implementations:

1. a renamed first-admission selector under the new facade;
2. a queue that records unserved survivors but never gives them work;
3. a selector that collapses two viable epsilons at one `L`;
4. a wrapper that returns the same schema but uses a different controller;
5. replay that uses list position or a caller-stamped scope;
6. a repair child inheriting its parent's verified status;
7. a repair child reusing failed holdout draws;
8. resume that duplicates a draw range or changes work order;
9. a diagnostic helper that still issues active handoffs; and
10. P2 exposing an XLA default before adapter qualification.

Require injected deterministic outcomes and exact work-item traces where
appropriate. Distinguish a scientifically necessary oracle from an
implementation-specific assertion such as Python object identity.

### 10. Are defaults, budgets, and governance proportionate?

Audit every material number and rule in sections 4-8, including the inherited
`L=(3,5,9,13,18,25)` grid, acceptance target/band, chain bank, R-hat/ESS/MCSE
screens, refinement barrier, transition allocations, CPU/GPU ceilings, retry
limits, and launch count. For each, classify provenance as owner policy,
existing compatibility behavior, measured/derived, convenience choice, or
unproven hypothesis. Check that no number becomes a universal scientific
default merely because it appears in the plan.

Also check proportionality under the repository's trusted academic-research
policy. Ordinary Git provenance, versioned output roots, checksums, bounded
budgets, and manifests are appropriate. Flag any unnecessary approval-token,
one-use authorization, mandatory review-chain, or other production-service
ceremony if the plan introduces it.

## Required classification of findings

For every finding, use one of these classifications:

- `confirmed defect`: follows from inspected code, plan text, or a checked
  contract;
- `unsupported assumption`: the plan relies on a premise that has not been
  source-grounded or operationally defined;
- `design choice`: a defensible choice that should be stated, but is not itself
  a correctness failure;
- `missing evidence`: a later implementation, numerical, GPU, statistical, or
  rendered-document check that cannot be passed by planning prose; or
- `no finding`: the proposed semantics are sufficient on the inspected point.

Do not label every future implementation check a defect. Conversely, do not
call an active contradiction a mere future check. For each material issue give:
the plan section, exact source path/symbol/line where relevant, failure mode or
counterexample, severity, and smallest revision.

## Required deliverable

Return a complete audit response with:

1. an executive verdict on whether R1 is ready for implementation;
2. material findings first, ordered by severity;
3. a separate disposition for each of the five original findings;
4. an assessment of the unification premise itself;
5. an evidence-coverage table listing inspected, unavailable, and not-checked
   sources;
6. a table of remaining defaults and their provenance/limits;
7. a test-oracle assessment against the ten bad implementations above;
8. the next smallest justified action and any true blockers; and
9. exactly one final line: `VERDICT: AGREE` or `VERDICT: REVISE`.

`AGREE` means only that the plan is coherent enough to implement under its
stated boundaries. It does not mean the code exists, tests pass, XLA is
qualified, a GPU run is valid, the guidebook has been rendered, a nominee is
statistically superior, or posterior inference is ready. Use `REVISE` when a
material lifecycle, target, authority, migration, statistical, or guidebook
problem remains unresolved. State when a limitation is simply evidence still
pending for a later phase.

The audit is advisory. Do not edit source or documents, run a broad repository
review, launch another agent, request hash-bound approval language, or require
per-retry permission. Preserve the initial audit and this memo as separate
provenance. A later implementation audit will be responsible for executable
tests, numerical parity, backend qualification, downstream compatibility, and
rendered-book inspection.
