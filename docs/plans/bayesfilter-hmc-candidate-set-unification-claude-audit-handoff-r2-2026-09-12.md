# Claude handoff: R2 audit of unified HMC candidate tuning and epsilon repair

Date: 2026-09-12  
Audit type: read-only, source-grounded plan audit  
Repository: `/home/ubuntu/python/BayesFilter`  
Plan under review: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`  
Plan revision: R2  
Prior thorough audit: `docs/plans/bayesfilter-hmc-candidate-set-unification-claude-thorough-audit-2026-09-12.md`  
New external issue: `/home/ubuntu/python/MacroFinance/docs/plans/bayesfilter_candidate_specific_epsilon_repair_handoff_memo_2026_09_12.md`

This memo requests one final planning audit before implementation. It does not
authorize code changes, experiments, GPU work, downstream edits, or guidebook
rewriting. Preserve the prior audits and the R2 plan as separate provenance.

## Initial worker prompt

Start with this exact bounded prompt and one path:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited path or line:
/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-candidate-set-unification-claude-audit-handoff-r2-2026-09-12.md.
Do not edit files, run commands, launch agents, initialize TensorFlow/GPU,
build the book, or review the whole repository. Read this memo, then inspect
the R2 plan and only the exact BayesFilter or MacroFinance sources it cites.
Audit whether R2 is a coherent repair of the unified candidate-set procedure,
with special attention to candidate-specific same-L epsilon repair, queue
precedence, immutable evidence lineage, typed budget outcomes, and consistency
of the guidebook and tests. Independently challenge the prior AGREE verdict.
End with exactly one terminal line: VERDICT: AGREE or VERDICT: REVISE.
```

If a cited excerpt is insufficient, request the smallest exact next path and line
range. Do not infer implementation behavior from the plan alone. Do not launch
tests or experiments during this planning audit.

## Background: the original unification question

The owner observed a confusing downstream M4 tuning result. The user-facing
table showed one final `(L, epsilon)` pair per attempt even though the procedure
started from a broad `L` grid. The current ordinary route can preserve internal
candidate records while ordering candidates, starting only a limited number of
fresh verifications, and stopping on the first admission. Thus “the grid was
tested” and “only one pair was handed off” can both be true, but the interface
does not make the distinction clear and can leave eligible candidates unverified.

The desired BayesFilter procedure is:

1. freeze the target, coordinates, geometry or transport, mass policy, start
   bank, and execution scope;
2. explore a declared broad family of `L` values;
3. qualify epsilon independently for each `L` and measure every proposed pair;
4. retain every non-vetoed candidate, including multiple epsilons at one `L`;
5. allocate further stages fairly to all retained candidates;
6. repair a candidate locally when evidence supports a repair, preserving its
   identity and evidence lineage;
7. perform fresh final verification for each candidate that reaches it; and
8. return a set of verified members, with an optional descriptive nominee that
   does not erase or outrank the set without uncertainty evidence.

Ordinary coordinates and frozen nonlinear transports do not have identical
target preparation or mass rules. The proposed unification is a shared
lifecycle/controller after typed preparation, not a claim that all routes use
the same target, coordinates, mass adaptation, or authority. Typed proposal
fields whose deterministic force is not the exact target score remain
mechanics-only.

## The newly discovered concrete failure

The MacroFinance handoff reports a current Phase 14 sequence in which:

1. a candidate at `L=3` receives an inconclusive verification;
2. another candidate at `L=5` is selected on a later attempt;
3. a directional higher-epsilon repair is computed for that selected candidate;
4. the repaired `(L=5, epsilon)` pair is not actually verified; and
5. the campaign terminates under generic budget exhaustion.

The external handoff requests a stronger protocol:

- directional evidence for a fixed `L` must repair epsilon at that same `L` and
  mass before another `L` is considered;
- `inconclusive_evidence` and `inconclusive_conflict` must not be converted into
  a directional repair or silently replaced by another `L`;
- a computed repair must be visibly distinguished from an executed and freshly
  verified repair;
- budget exhaustion of a required repair must name the candidate and repair and
  use a typed `repair_budget_exhausted` status; and
- old evidence must remain immutable and replayable without double credit.

The R2 plan incorporates this issue. It interprets “same candidate” carefully:
an epsilon change cannot retain the same immutable candidate record or hash. It
creates a new child record with a new candidate ID/hash, while retaining the
same `candidate_family_id`, parent lineage, exact `L`, scope, target, mass,
coordinates, start-bank design, and warmup protocol. An unchanged retry retains
the candidate record and changes only its `verification_attempt_id`.

This distinction is central to the audit. A family means a fixed-geometry,
fixed-`L` repair lineage; it does not permit evidence from one epsilon to be
treated as evidence for another epsilon.

## What R2 changed

Read the actual plan, especially sections 1, 2 (E11), 3, 4.1-4.5, 5, 6, 7, 8,
9, 11, and 12. The material R2 changes are:

### Candidate and evidence identity

Section 4.1 now requires immutable candidate records containing exact `L`,
epsilon, scope/search, target/adapter, mass, warmup protocol, and a
`candidate_record_hash`. Verification receipts carry candidate hash, exact
`L`/epsilon/mass, stream and draw range, source hash, and a separate
`verification_attempt_id`.

Directional epsilon repair creates a child record. The child must preserve the
same exact `L`, scope, target, mass, coordinates, start-bank design, and warmup
protocol. A geometry repair creates a new scope. An explicit trajectory veto may
create a new `L` family, but it cannot inherit epsilon or verification evidence.

### Directional repair semantics

Section 4.3 defines a typed mapping:

- `repair_step_higher` with one-sided valid support creates a higher-epsilon
  child in the same fixed-`L` family;
- `repair_step_lower` analogously creates a lower-epsilon child;
- `inconclusive_evidence` permits pending or a predeclared evidence extension,
  but no epsilon change or `L` switch;
- `inconclusive_conflict` remains unresolved or follows an explicit conflict
  policy, but cannot become a directional repair;
- trajectory/resonance/stall decisions may change `L` only if the scope policy
  explicitly permits that trajectory repair; and
- finite acceptance outside a band is not itself a validity veto or repair.

The plan requires valid one-sided directional support under a declared
aggregation policy. It does not permit a noisy boundary crossing to silently
change epsilon. Section 4.2 also requires a positive finite epsilon domain, a
bounded repair-factor/proposal rule, and a finite repair limit per candidate
family; exceeding that limit must be a typed terminal outcome.

### Queue precedence and budget outcomes

Section 4.4 keeps the active cohort closed and does not preempt already reserved
work. After that cohort closes, a directional epsilon child receives priority
over not-yet-admitted work for another `L` and over advancement of its parent
beyond the repair barrier. Existing reservations of other candidates remain
untouched.

Repair records have execution, verification, and qualified-handoff statuses. Only
`executed_and_verified` produces a qualified repair handoff. An unexecuted,
executed-but-failed, or budget-pending child has
`not_executed_with_reason` as its qualified status and cannot enter
`verified_candidates` or replay.

If required repair work cannot be funded, the result records
`final_status="repair_budget_exhausted"`, `completion_status="partial_budget"`,
the parent/child IDs, old/new epsilon, exact `L`, mass signature, and remaining
budget. This is not collapsed into generic candidate exhaustion.

### Tests and guidebook

R2 adds test obligations 16-22 for direction mapping, immutable identity,
inconclusive evidence, repair priority, no-handoff statuses, typed budget and
repair-limit exhaustion, and a deterministic reproduction of the MacroFinance
sequence. The guidebook requirements now explicitly explain same-`L` repair,
repair lineage, inconclusive evidence, repair priority, and
computed-versus-verified status.

## Questions for the audit

### 1. Is the repair identity model correct?

Check whether the plan's distinction between immutable candidate record,
candidate family, parent/child lineage, and verification attempt is coherent.
In particular:

- Does changing epsilon necessarily require a new candidate ID/hash?
- Is `candidate_family_id` sufficiently narrow to prevent unrelated candidates
  at the same `L` from sharing evidence?
- Can a retry of the unchanged candidate safely retain its record hash while
  using a new verification attempt ID and fresh or resumed evidence as declared?
- Are exact `L`, mass signature, target/coordinate identity, start bank, and
  warmup protocol enough to establish “same candidate family” for this repair?
- Can any result or replay path accidentally use the parent record hash with the
  child epsilon?

Classify the family interpretation as correct, unsupported, or requiring a
specific additional binding. Do not demand that an epsilon-changing repair
retain an immutable hash merely because the external memo used “same identity.”

### 2. Does the queue guarantee the requested ordering?

Red-team the exact boundary semantics:

1. active cohort contains candidates A and B;
2. A returns `repair_step_higher` or `repair_step_lower`;
3. A's child and an unrelated later-`L` candidate are pending;
4. B still has reserved work;
5. execution is interrupted and resumed; and
6. budget is sufficient for B and only one of the pending children.

Determine whether the plan correctly runs already-reserved B work first, then
gives the same-`L` repair priority over not-yet-admitted different-`L` work,
without stealing B's reserve or preempting a running task. Check what happens
when multiple repairs exist, when the repair is not fundable, and when the
repair child itself fails verification. Identify any remaining ambiguity in
“repair priority interval,” cohort closure, child admission, or parent advance.

### 3. Are directional and inconclusive decisions separated correctly?

Inspect the actual BayesFilter decision/repair symbols named by the plan, such as
`repair_step_lower`, `repair_step_higher`, `inconclusive_evidence`, and
`inconclusive_conflict`. Verify that R2 does not accidentally make every
out-of-band acceptance a repair, while still honoring valid directional evidence.

Check whether the plan needs to bind the aggregation rule, confidence/uncertainty
rule, opposing-direction rule, repair factor/bounds, or maximum same-family
repair count more explicitly. A “one-sided valid support” phrase is insufficient
if two reasonable implementations would make different repair decisions.

### 4. Are terminal statuses and artifacts honest?

Check the state combinations and invariants:

- computed epsilon but no dispatched work;
- dispatched work interrupted;
- child executed but final verification failed;
- child verified;
- child budget-pending;
- parent verified while child exists; and
- scope invalidated after an earlier parent verification.

Confirm that only `executed_and_verified` creates an admissible repair handoff,
that every other case is represented without overwriting the parent, and that
`repair_budget_exhausted` is not misleadingly used when the failure is an
infrastructure retry or scientific veto. Check whether status names are
consistent between candidate evidence state, work-item state, campaign
completion, and public `final_status`.

### 5. Does R2 still preserve the all-survivor objective?

Check that repair priority fixes candidate-specific epsilon behavior without
reintroducing first-admission selection or starving unrelated viable candidates.
The desired rule is bounded priority after already reserved work, not unlimited
repair retries and not a hidden single winner. Verify that multiple epsilon
values at one `L` remain separate, a repaired child does not erase its parent,
and later candidates remain pending/viable or incomplete with truthful status.

### 6. Is the plan faithful to the actual R1 file?

The preserved thorough audit returned `AGREE`, but it referred to plan sections
and formulas absent from the file, including sections 3.1-3.4, 4.6, 6.4,
8.1-8.5 and geometry-scaled budget formulas. Re-read the actual R2 file rather
than relying on that audit's summary. Identify any remaining copied or
unsupported claim. The R2 plan explicitly records this source-fidelity issue in
section 12; determine whether that correction is adequate.

### 7. Are tests sufficient and discriminating?

Check whether tests 16-22 would fail each bad implementation:

1. repair child changes `L` as well as epsilon;
2. two initial epsilons at one `L` collapse into one family or one record;
3. inconclusive evidence triggers a directional repair;
4. a later `L` is dequeued before a required same-`L` repair;
5. repair uses another candidate's mass or start bank;
6. child inherits parent verified status;
7. computed epsilon is emitted as a handoff without execution;
8. executed child fails verification but is marked qualified;
9. repair budget exhaustion is reported only as generic candidate exhaustion;
10. resume duplicates a receipt or changes deterministic child order; and
11. the MacroFinance reproduction is impossible to distinguish from a passing
    candidate because the trace lacks old/new epsilon, hashes, or final status.
12. a repair factor leaves the declared epsilon domain or continues beyond the
    family repair limit without a typed terminal status.

Separate scientifically required assertions from implementation-specific ones.
Exact traces, IDs, hashes, stream domains, and statuses are required where they
answer lineage and scheduling questions; Python object identity is not a
scientific requirement by itself.

### 8. Are guidebook, API, and migration requirements aligned?

Check that sections 7 and 8 describe the R2 repair with the same vocabulary as
sections 4.1-4.5. A reader must not confuse:

- a candidate family with an exact candidate record;
- a computed epsilon proposal with a verified child;
- inconclusive evidence with a directional repair;
- a later `L` candidate with a same-`L` repair;
- a diagnostic helper with the active controller; or
- a partial campaign with a completed tuning handoff.

Check that P0-P5 and the generated registry/replay contracts require the new
fields and statuses. Downstream MacroFinance adoption remains an owner-managed
consumer migration; the BayesFilter plan should provide a concrete compatibility
finding without silently editing that repository.

### 9. Are numerical defaults and governance proportionate?

Audit the actual numbers in the plan: inherited `L` grid, acceptance band,
chain count, R-hat/ESS/MCSE screens, repair limits/factors if specified, CPU/GPU
ceilings, and launch counts. Classify provenance and identify any number that
silently became a universal scientific default. Flag missing bounds for epsilon
repair or family depth if they affect correctness or cost.

The repository is trusted academic research. Versioned outputs, ordinary
checksums, manifests, bounded budgets, and atomic checkpoints are sufficient.
Do not ask for launch tokens, hash-bound approval language, per-retry approval,
or a review chain.

## Required response

Return an audit with:

1. executive verdict on whether R2 is ready for implementation;
2. material findings first, ordered by severity;
3. explicit disposition of the five initial findings and the MacroFinance issue;
4. identity and queue transition assessment;
5. status/artifact and test-oracle assessment;
6. evidence coverage, including any unavailable downstream artifacts;
7. remaining defaults, provenance, and limits;
8. the smallest justified next action and true blockers; and
9. exactly one final line: `VERDICT: AGREE` or `VERDICT: REVISE`.

Use these finding classes: `confirmed defect`, `unsupported assumption`,
`design choice`, `missing evidence`, or `no finding`. For every material issue,
give the plan section, exact source path/symbol/line when relevant, failure mode,
severity, and smallest revision. Do not call implementation evidence “missing”
when it is intentionally assigned to a later phase; do call a contradiction in
the plan a defect.

`AGREE` means only that R2 is coherent enough to implement under its stated
boundaries. It does not certify code, tests, numerical parity, GPU/XLA
qualification, downstream adoption, guidebook rendering, posterior validity,
or statistical superiority.
