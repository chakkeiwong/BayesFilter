# Handoff to Claude: review the inference validation infrastructure plan

Date: 2026-09-15. Requested by the owner. This is a review brief, not a request
to implement the infrastructure or launch experiments. The proposed
infrastructure has not been implemented. Earlier HMC repairs and unrelated
work are present in a dirty worktree; preserve them.

## Starting prompt

Use the following single-path prompt to begin the review. This memo explicitly
authorizes the bounded plan read specified below; other source/code reads
should be requested only when needed to resolve a particular finding.

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-inference-validation-infrastructure-claude-handoff-2026-09-15.md
Do not edit, run commands, launch agents, or review the whole repo.
Question: Does the referenced plan provide a scientifically sound, implementable
infrastructure that fully synthesizes the surveyed validation methods for the
BayesFilter HMC pipeline, with appropriate scope and completion criteria?
Use this memo's instructions to inspect the plan and return an independent
review with concrete findings and repairs. End with VERDICT: AGREE or
VERDICT: REVISE.
```

## Exact review target and boundary

Read the complete [infrastructure plan](bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md),
lines 1–539 in the version present when this memo was written. That plan is the
subject of the verdict. Review its scientific design, architecture, coverage,
implementation ordering, and completion criteria. Do not mistake reviewing this
memo for reviewing the plan.

The initial review consists of this context and that one plan. The plan links
the prior literature survey and the interface reference; do not recursively
read every link or scan the repository. If a finding needs evidence beyond the
plan, request the next exact source path and section/line range, explaining the
question it would resolve. Source or implementation parity that has not been
inspected must be described as not checked. Do not make a positive parity claim
from the author's account of a previous inspection alone.

The initial bounded scope follows `AGENTS.md`, “Claude Review Prompt Shape.”
It limits discovery and tool use, not the rigor of the scientific review. Return
the review as Markdown text. The caller can save it; this read-only request does
not authorize edits, test execution, downloads, package installation, or another
agent launch.

## Background: why the owner requested infrastructure

The owner originally challenged an HMC tuning workflow that searched multiple
leapfrog counts but handed off a single selected pair. Ordinary and
fixed-transport tuning had developed inconsistent procedures and documentation.
The subsequent repair direction was a shared candidate-set procedure: broad L
coverage, candidate-specific epsilon measurement and repair, independent fresh
verification, and retention of all verified members. A passing candidate must
not end a funded cohort prematurely.

Further reviews identified R-hat being used as a tuning requirement, fragile
candidate-to-retained bridges, budget/resume corner cases, and incomplete
warmup/precision reporting. The owner explicitly required R-hat to be removed
from tuning qualification. Later work separated posterior assessment from
tuning, added precision reporting including optional lugsail batch means, and
added regression tests. Those changes do not establish broad sampler validity.

When told about the test coverage, the owner challenged whether it tested
mechanisms and corner cases adequately and asked about established practices in
the literature, Stan, PyMC, R, and Dynare. A source survey identified numerical
mechanics tests, Gandy–Scott MCMC tests, SBC with data-dependent quantities,
posterior reference benchmarks, and diagnostic/run-length assessment.

The owner then rejected a recommendation limited to a smaller list of test
additions. The latest request was to synthesize the full survey into proper
infrastructure. The plan under review is the response: common target/reference
definitions, distinct experimental designs, narrow public-procedure adapters,
bounded execution, independent assessment, and coverage/power reporting.

Please assess whether it actually achieves that synthesis. Merely recommending
more Gaussian tests, listing difficult models, or increasing a pytest count
would not answer the owner's concern. Conversely, a generic framework that
delays useful scientific tests indefinitely would also fail the request.

## Current evidence and implementation status

The following is background from the prior work, not a request to rerun it:

- The latest repair result records 277 passing tests followed by 114 overlapping
  tests. These mix unit, contract, mocked-control-flow, and numerical integration
  checks. The counts are not additive and do not count validated model families.
- Earlier automatic preparation/search tests cover isotropic and anisotropic
  Gaussian targets. They establish search completion and candidate records;
  they do not test retained posterior accuracy and precede the latest diagnostic
  changes.
- The latest actual CPU/GPU posterior example uses a two-dimensional Gaussian,
  precomputed geometry and one supplied initial pair. It exercises tuning through
  posterior assessment, but not the whole automatic-preparation/broad-search path.
- Some route-specific tests already examine involution, volume, incorrect energy,
  replay, seeds, budgets, and missed-mode diagnostic failures. The gap is systematic
  coverage and independent statistical validation, not a complete absence of tests.
- Stationary AR(1) precision calibration tests fixed-size MCSE behavior. It does
  not calibrate the actual sequential stopping procedure.
- The inspected evidence does not establish current full-pipeline validation on
  the proposed nonlinear, heavy-tailed, multimodal, hierarchical and MacroFinance
  target collection.

The plan proposes the infrastructure and its delivery milestones. It has not
implemented its proposed package or command interface. The author's skeptical
self-review is included in the plan and should itself be questioned.

## Requirements to preserve

1. Keep the current public tuner/candidate-set procedure as the subject under
   test. The validation executor must not become another tuner, sampler, or
   admission authority.
2. Preserve all verified candidates and distinguish tuning eligibility from
   posterior evidence. R-hat, ESS and MCSE must not qualify, rank, remove, repair
   or delay tuning candidates.
3. Include the full surveyed method families in the final architecture. An early
   complete example is an implementation milestone, not completion of the project.
4. Use independent references and retain failed, incomplete, unassessed and stale
   evidence explicitly. A diagnostic pass does not establish posterior correctness.
5. Keep the work proportionate to a trusted academic repository: ordinary
   provenance, bounded compute, source grounding and reproducible results. Do
   not add approval tokens, review chains, or speculative security machinery.
6. Preserve TF/TFP execution, GPU/XLA policy and memory growth. Independent
   reference/diagnostic tools remain separate from runtime inference decisions.
   Existing CPU/non-XLA checks need honest scope labels.

These requirements come from the owner's direction and current project policy.
Challenge the proposed implementation choices freely within them.

## Review priorities

| Priority | Plan section | Questions requiring a concrete judgment |
| --- | --- | --- |
| Experimental architecture | Architecture; synthesis of methods | Are shared definitions sufficient for the genuinely different experiments? Can an inappropriate ordinary chain or controller double accidentally satisfy a stronger test? Are the interfaces specific enough to implement without redesigning them in each campaign? |
| References and model laws | Shared definitions; reference accuracy; catalog | Are target identity, coordinate transformations, simulator independence, reference uncertainty and moment requirements represented adequately? Can exact likelihood, generating truth and posterior reference be confused? |
| Frozen-kernel tests | Invariance and SBC designs | Are fixed-kernel assumptions, conditional independence of preparation, random-position ranks, ties, and sequential testing correctly scoped? Does the design preserve the distinction between invariance and mixing? |
| Full-procedure SBC | Invariance and SBC designs | Is the actual inference procedure tested, with truth excluded from initialization? Are data-dependent quantities, simulator errors, MCMC dependence, failed fits and selection effects handled? Is the proposed treatment of candidate families statistically defensible and implementable? |
| Candidate-set behavior | Procedure adapters; numerical/search mechanisms | Can the suite independently establish full retention, per-L epsilon repair, fresh verification, cohort completion, budgeting, interruption and replay? Are real numerical tests required where mocks cannot answer the question? |
| Precision and warmup | Reference accuracy and stopping decisions | Does the plan distinguish diagnostic arithmetic from operational readiness and error at actual stopping? Are false readiness, interval coverage, cap failures and missed modes defined precisely enough for later experiments? |
| Test quality | Controlled defects; statistical decisions | Can mutations establish meaningful power without altering the reference or merely causing an unrelated crash? Are null rejection, defect size, uncertainty, multiplicity, and confirmation separation adequate? |
| Coverage and scope | Catalog; reports; delivery plan | Does the mechanism/route/scenario matrix prevent token model coverage? Are required intersections and terminal completion sufficiently concrete? Could the project declare success after only its first example? |
| Engineering feasibility | Repository integration; bounded execution | Are reuse boundaries sound, the records minimal, and execution/resume responsibilities clear? Does the plan avoid historical campaign semantics, duplicate persistence/scheduling, and runtime imports of diagnostic references? |
| Guide and interpretation | Reports and operational profiles | Will the guide report actual, current evidence and its limitations? Are engineering success, sampler validity, statistical accuracy and expected test failures kept distinct? |

Pay particular attention to the following possible weaknesses. These are
questions for review, not instructions to agree with the plan:

- The architecture deliberately postpones experiment-specific numerical
  allocations. Determine which omissions are appropriate before pilots and
  which leave the implementation or scientific test undefined. A usable design
  may still need a more explicit first resolved suite and coverage contract.
- Candidate-set SBC is more complicated than single-output SBC. Varying numbers
  of surviving siblings, absent L families, data-dependent tuning, and correlated
  member results may require a more precise statistical design than the current
  prose provides. Request a concrete repair if the proposed grouping rule is
  insufficient. It must not reintroduce a production nominee.
- The plan lists a broad set of engines and model families. Check whether its
  completion criteria distinguish implementing an engine, calibrating its tests,
  and completing required numerical coverage. “Supported by the schema” is not
  execution evidence.
- Reusing an existing statistic or target factory may preserve an unsuitable
  assumption or a shared bug. Check whether independence and compatibility are
  assessed at an adequate granularity rather than asserted at package level.
- Assess whether the independent assessment layer can detect a falsely favorable
  runtime report, including a changed or incorrect target. Merely recomputing the
  same runtime diagnostic is insufficient.

## Scientific distinctions that matter to the verdict

An identity kernel preserves every distribution but never explores. A Gaussian
resonance case can have perfect endpoint acceptance while cycling. Such controls
should pass the relevant invariance identity and expose an exploration failure;
they must not be treated as evidence that the invariance test is malfunctioning.

SBC parameter ranks can be uniform when an implementation ignores the data and
returns prior draws. Likelihood-based quantities and independently checked model
simulation are therefore material parts of the proposal, not optional plotting
extras. Ordinary SBC with correlated MCMC draws cannot inherit an iid rank null
without justification. Gandy–Scott's random-position construction addresses a
different experiment and does not solve arbitrary full-pipeline dependence.

A wrong proposal force need not break posterior invariance if the proposal is
reversible and volume preserving and the full MH correction is correct. It
should fail an exact-score oracle when that is the claimed target. Review
mutation expectations against the property actually tested.

Small MCSE does not establish removal of burn-in bias, and rank/R-hat diagnostics
can miss unvisited modes. Estimator accuracy at fixed length and at a random
stopping time are separate questions. Cauchy tests cannot claim finite-mean
precision. Posterior references have their own uncertainty; agreement with an
external library requires matching the density, priors, data and Jacobians.

The plan must allow these distinct outcomes without collapsing them into one
green “validated” status. Also check that identifying an expected failure is a
successful test response, not a passing result for the defective sampler.

## Requested review output

Lead with the main judgment and the most consequential findings. Then provide:

1. **Findings in severity order.** For each, cite the plan section/line, explain
   the concrete scientific or engineering failure it could permit, and specify
   a repair. Distinguish design blockers, material improvements, and optional
   refinements.
2. **Synthesis assessment.** State which surveyed methods are actually integrated,
   which are only named, and whether dependencies/completion criteria preserve
   the full intended scope.
3. **Statistical design assessment.** Address frozen-kernel assumptions, SBC
   dependence/candidate selection, reference validity, stopping decisions, and
   false-rejection/power calibration. Identify anything not checked against
   original sources.
4. **Implementation assessment.** State whether the first deliverable and the
   subsequent milestones are concrete enough to execute. Propose the smallest
   structural changes needed without replacing the plan with a generic framework.
5. **Default/assumption audit.** Identify unsupported numerical choices or hidden
   assumptions. Separate architectural gaps from parameters that appropriately
   require a later bounded pilot and experiment design.
6. **Verdict scope.** Say whether the architecture is ready to implement and what
   must be resolved before statistical campaigns. Do not imply that this review
   validates existing HMC results or authorizes a new numerical default.

Use `VERDICT: REVISE` if a material scientific or implementation-design gap
requires changing the plan. Use `VERDICT: AGREE` only if no material unexamined
assumption remains at this architectural stage and the later required evidence
is adequately specified. Findings that require an additional exact source read
should remain explicit rather than being replaced by unsupported confidence.

End with the applicable verdict on its own line. This is an independent review;
the author's self-audit and the existence of citations are not reasons to agree.
