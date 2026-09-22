# Claude handoff: audit the HMC candidate-set unification plan

Date: 2026-09-12.
Status: ready to hand to Claude; audit not yet performed.
Task type: read-only technical and scientific plan audit, not implementation.
Repository: `/home/ubuntu/python/BayesFilter`.
Inspected commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`.

## Start here

The owner requests a single sensible HMC tuning procedure: begin with a broad L
grid, independently tune epsilon, retain every viable pair, and continue tuning
and verification for the full set. The owner also requests a guidebook rewrite
and a thorough Claude audit of the implementation plan.

The plan to audit is
[HMC candidate-set unification and guidebook rewrite](bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md).
Read that one file next, in full. Its sections 2 and 6–8 explicitly identify the
source excerpts that this audit needs. Then inspect those bounded sources one
at a time as necessary; do not substitute a repository-wide review for checking
the actual selectors, verifier, registry, replay boundaries, and manuscript.

Use this as the initial worker prompt, with exactly one path:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-candidate-set-unification-claude-audit-handoff-2026-09-12.md.
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does the referenced plan define one implementable, scientifically defensible
candidate-set HMC tuning procedure and a complete, faithful guidebook migration,
or what must change before implementation? Follow the memo's targeted source
inspection instructions. End with VERDICT: AGREE or VERDICT: REVISE.
```

Reading cited sources with file-reading tools is part of this audit. Do not run
HMC, tests, GPU initialization, package installation, builds, or shell commands.
If a needed source or symbol is outside the cited excerpt, request that exact
next path/range rather than making an unsupported inference. A supervising
agent can supply the requested excerpt. Do not turn missing optional context
into approval-token or mandatory review-chain machinery.

## What has and has not been done

Codex inspected the current interface guide, capability registry, ordinary
selector and verification queue, measured fixed-transport policy, candidate-set
helper, relevant tests, earlier repair plans, and selected guidebook chapters.
The planning edit creates only the plan and this memo. No algorithm, API,
guidebook chapter, consumer, or scientific default has been changed. No new
experiment was run, and no claimed improvement or posterior conclusion exists.

The source-level problem is established:

- The ordinary procedure measures a grid but can stop after the first verified
  pair and starts at most two candidates per attempt.
- Its selection order prioritizes distance from an acceptance target.
- The current all-survivor helper validates already tuned frozen transports;
  it is not the desired per-(epsilon,L) tuner.
- Ordinary, fixed-transport, and typed-force paths have different control flow.
- Chapter 21 still narrates selecting a “best pair” and refining locally, while
  chapter 21b describes a different broad-grid procedure.

The motivating M4 artifact records another BayesFilter commit,
`d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`. Its first attempt had three eligible
candidates, two verifier starts, and one `start_quota_exhausted` record. The
whole result was `budget_exhausted`, with no top-level hard veto. This supports
the incompleteness diagnosis only. Do not infer the other candidates would have
passed, treat all attempts as one unchanged mass scope, or rank samplers from
the acceptance values. The exact JSON path is in plan section 2; inspect it only
if the control-flow example itself is disputed.

## Required audit questions

1. **One executable procedure.** Does the proposed common entry/controller
   actually remove divergent active selection policies? Trace E2–E7. Identify
   any adapter, compatibility wrapper, typed-force branch, or candidate-set
   helper that could still independently choose a winner, stop early, or grant
   authority. A single function name or schema is insufficient.

2. **Mathematical scope.** Are the target measure, exact score, affine mass
   coordinates, nonlinear Jacobian, proposal-field exception, and frozen kernel
   all preserved? Is separate geometry preparation justified without silently
   adding ordinary mass adaptation to NeuTra or promoting an arbitrary force?
   Identify anything still unproved rather than approving it by analogy.

3. **Survivor meaning and fairness.** Does “retain every viable candidate” cover
   multiple epsilon values per L, every frozen transport/geometry scope, and
   candidates pending further evidence? Can early order, batching, refinement,
   or budget reservation starve a later survivor? Does a failed repaired child
   leave unrelated candidates and prior evidence intact? Must any state or
   transition be specified more precisely?

4. **Acceptance and statistical reasoning.** Does the plan really remove
   acceptance-distance elimination and upper-band validity claims? Does it
   measure every epsilon proposal without assuming fixed-L monotonicity? Are
   movement, divergences, R-hat, ESS, MCSE, mode/observable diagnostics, and
   uncertainty assigned appropriate and distinct roles? Check the distinction
   between a short early screen, fresh final verification, descriptive
   nomination, and statistically supported ranking, including optional stopping
   and multiplicity.

5. **Repair, resume, and budget.** Can repairs and longer validation continue
   without restarting healthy geometry or losing the queue? Are continuation
   chunks distinguished from independent replications? Does resume bind the
   RNG, state, settings, draw ranges, source, and remaining work? Are the
   proposed cost ceilings honest engineering choices, with insufficient budget
   reported explicitly rather than used to relax scientific criteria?

6. **Replay and migration.** Can every verified member be replayed without a
   caller reconstructing hidden geometry? Is an explicit member ID required?
   Are partial campaign completion and member verification independent? Are old
   artifacts readable without being silently relabeled, and are in-repo and
   downstream consumers dealt with concretely? Flag any dependence on the old
   single-selected-result shape that P3 overlooks.

7. **Backend feasibility.** Does the bounded ordinary NumPy migration cover the
   actual active dependency path, including state/selection/artifact code?
   Could the shared controller merely hide a noncompliant numerical backend?
   Are stable TensorFlow signatures, XLA, trusted GPU execution, on-demand
   memory allocation, and explicit CPU-reference roles addressed? Separate
   known debt from new defects and from checks not yet performed.

8. **Guidebook fidelity.** Inspect E8–E9 and the chapter 21b material cited in
   E5. Does section 7 replace the conflicting scientific story across all
   active chapters and examples, preserve substantive mathematics/citations,
   and explain the common procedure naturally? Could a reader still confuse
   transport training restarts with L/epsilon candidates, accepted proposals
   with valid sampling, or a tuning result with posterior convergence? The
   rendered-book review is future work, not already passed.

9. **Tests that discriminate.** Inspect E10 and the relevant test contracts.
   Would P5 fail a renamed first-pass implementation, a capped unserved queue,
   two viable epsilons collapsed into one, cross-wired replay, or a repair that
   consumes holdout evidence? Identify missing adversarial cases. Do not ask
   for low-value tests that merely repeat constant declarations.

10. **Hidden assumptions and proportionality.** Examine every material number
    and proposal in plan sections 4–5 and 8. Are provenance, failure mode, early
    diagnostic, and limits explicit? Distinguish decisions settled by the owner
    from implementation choices and future target-specific experimental inputs.
    Flag unsupported defaults and unnecessary bureaucracy with equal care.

## Source-reading sequence

The plan's evidence table is the primary map. After reading the plan:

1. Read E1–E3 to check the reported ordinary behavior and what must be removed.
2. Read E4–E7 to test whether the shared-controller/adapters design is feasible
   and whether the proposed semantics correctly distinguish existing APIs.
3. Read E10 and the public replay contracts in
   `docs/reference/hmc-tuning-interface.md` under “Artifact Acceptance,”
   “Durable ordinary replay,” and “Replay roles and authority.” For a disputed
   implementation detail, request the exact referenced builder's source next.
4. Read E8–E9 and E5's chapter excerpt to audit the guidebook rewrite scope.
5. Read additional exact implementation dependencies only when required by a
   concrete finding. For the known NumPy dependency, begin with
   `bayesfilter/inference/hmc_tuning_state.py:1–25`; request its called paths
   and those of the ordinary executor if assessing migration completeness.

The prior September 5 plan deliberately kept fixed-transport policy separate
and fixed the ordinary grid. It is historical design context, not a veto on
the owner's new unification request. Conversely, this planning request does
not authorize hiding target/coordinate differences or weakening scientific
checks. Do not approve a cosmetic documentation-only patch as unification.

## Deliverable

Return an audit in your response. The supervising agent should preserve it at
`docs/plans/bayesfilter-hmc-candidate-set-unification-claude-audit-2026-09-12.md`
with the actual inspected commit and model identity, if available. Do not
create a placeholder verdict in advance and do not edit files during the audit.

Lead with material findings ordered by severity. For each finding give the
plan section, inspected source path/symbol/line, the precise counterexample or
failure mode, and the smallest required revision. Separate confirmed defects,
unsupported assumptions, and missing evidence. Then include:

- an evidence-coverage table marking required source checks as inspected,
  unavailable, or not checked;
- a concise disposition table for procedure, mathematics, statistics,
  candidate lifecycle, budgets, backend, migration/replay, and guidebook;
- remaining numerical defaults and whether their provenance/limits suffice;
- the next smallest justified action and any material execution blockers; and
- exactly one terminal line: `VERDICT: AGREE` or `VERDICT: REVISE`.

`AGREE` means the plan is sound enough to implement under its stated boundaries;
it does not mean the implementation exists, smokes pass, a GPU campaign is
authorized by your verdict, or posterior inference is valid. Do not emit
`AGREE` if a material unexamined default is silently treated as settled, a
required source check is missing, or a scientific/migration flaw remains.
Use `REVISE` to identify the concrete remaining work, including explicitly
unavailable evidence when it prevents an assessment.

Review is advisory under the current academic-research policy. One material
plan audit and one terminal implementation audit are sufficient unless new
material evidence requires more. Do not request hash-bound approval language,
per-retry permission, launch tokens, a broad repository review, or a new review
chain. If a worker is later launched from Codex, the supervisor must read the
installed `claude-code-workers` skill and use its trusted noninteractive wrapper;
this memo itself launches or sends nothing.
