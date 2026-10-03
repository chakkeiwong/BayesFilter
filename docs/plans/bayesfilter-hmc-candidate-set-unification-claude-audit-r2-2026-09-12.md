# Claude audit: HMC candidate-set unification R2 (epsilon repair)

Date: 2026-09-12  
Audit type: read-only, source-grounded plan audit  
Plan audited: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md` revision R2  
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`  
Model identity: `claude-opus-5[1m]`  
Prior audits: initial audit (verdict: `REVISE`), thorough audit (verdict: `AGREE`)  
External issue: `MacroFinance/docs/plans/bayesfilter_candidate_specific_epsilon_repair_handoff_memo_2026_09_12.md`

This is the final planning audit before implementation. It focuses on candidate-specific epsilon repair, queue precedence, immutable evidence lineage, and the MacroFinance Phase 14 issue. No code, tests, experiments, GPU work, downstream edits, or guidebook rewriting was performed.

## Executive summary

**Verdict: AGREE**

R2 resolves the MacroFinance epsilon-repair issue with immutable candidate identity, repair-family tracking, deterministic queue precedence for same-`L` directional repairs, and explicit executed-versus-computed repair status. The seven requested protocol changes are specified completely: candidate records are immutable and hashed; directional repairs preserve exact `L` and mass; repaired children have priority over different-`L` work; inconclusive evidence does not silently become a different-`L` repair; repairs are marked `executed_and_verified` or `not_executed_with_reason`; budget exhaustion produces typed `repair_budget_exhausted`; and parent evidence is never overwritten.

The plan is implementable as specified. Remaining qualifications from the thorough audit (start-bank recipe, coefficient provenance, cross-scope ranking uncertainty) still apply but do not block P0–P5. One focused implementation audit after executable artifacts exist is the next review.

## 1. MacroFinance issue resolution

**All seven requested protocol changes are specified; the reproduction failure mode cannot recur under R2 semantics.**

### Issue summary (from handoff memo)

MacroFinance Phase 14 M4 attempt 1 selected `L=5` after `L=3` returned `inconclusive_evidence`, then computed a repair epsilon for `L=5` but never verified it. The result claimed a repair handoff without execution. The intended protocol: same-`L` directional repair with fresh verification before considering different `L`.

### R2 resolution (sections 4.1, 4.3, 4.4)

**1. Immutable candidate identity including `L`, mass, epsilon (section 4.1):**
- Each candidate record is immutable and repository-hashed: `candidate_record_hash`
- Identity includes: `scope_id`, `search_id`, creation ordinal, exact `L`, epsilon, mass signature, target signature, warmup protocol, start-bank design
- Equal numeric `(ε, L)` in different scopes or searches = distinct records
- An epsilon change creates a new child candidate ID and hash

**2. Same-`L` repair preserves exact `L` and mass (section 4.3.2):**
- Directional repair (`repair_step_lower` / `repair_step_higher`) creates child with:
  - Same `candidate_family_id` (fixed-geometry, fixed-`L` lineage)
  - `parent_candidate_id` = source verification's candidate ID
  - Same scope, target, exact `L`, mass signature, coordinate system, start-bank design, warmup protocol
  - New epsilon (computed from parent acceptance and repair factor)
  - New `candidate_record_hash`
- Two independently proposed epsilons at the same `L` = separate families

**3. Repaired same-`L` candidate receives priority and fresh verification (section 4.4, rule 4):**
```
After active cohort closes, directional epsilon repair children have priority 
over:
- not-yet-admitted work for a different L
- advancement of the triggering family's parent to a longer stage

Order repairs by parent work-item order, then proposal order.
Priority interval ends when child receives fresh verification or becomes 
budget/infrastructure pending.
```

**4. Inconclusive evidence does not silently promote different `L` (section 4.3.2):**
- `inconclusive_acceptance` (acceptance interval overlaps target, wide or uncertain) = neutral signal, no directional repair
- `inconclusive_conflict` (multiple candidates with conflicting directions) = retain all, no automatic `L` switch
- Neither becomes a different-`L` repair; both may nominate additional exploration under the declared refinement/exploration policy

**5. Repair status is explicit: `executed_and_verified` or `not_executed_with_reason` (section 4.5):**
- Repair record fields:
  - `parent_candidate_id`, `child_candidate_id`, `candidate_family_id`
  - `old_epsilon`, `new_epsilon`, exact `L`, mass signature
  - `source_verification_hash`
  - `execution_status`: `executed` / `not_executed`
  - `verification_status`: `verified` / `failed` / `pending` / `skipped`
  - `qualified_repair_status`: `executed_and_verified` / `not_executed_with_reason`
  - `not_executed_reason`: `repair_budget_exhausted` / `attempt_limit_reached` / `wall_time_exceeded` / `shared_invalidity`
- A computed epsilon alone = proposal, never a completed handoff
- Unexecuted repairs cannot appear in `verified_candidates`

**6. Budget exhaustion produces typed `repair_budget_exhausted` (section 4.4, rule 5):**
```
If required repair cannot be executed within declared total, attempt, stage, 
or wall-time budget:
- preserve as `proposal_budget_pending`
- set `qualified_repair_status` to `not_executed_with_reason`
- record `repair_budget_exhausted` when budget is the cause
- continue already reserved work
```

Result's `final_status` may be `repair_budget_exhausted`; `completion_status` = `partial_budget`; repair payload names the exact blocked candidate.

**7. Parent evidence preserved, never overwritten (sections 4.1, 4.4):**
- Candidate records are immutable
- Verification receipts have distinct `verification_attempt_id`, source candidate hash, stream ID, draw range, seed lineage
- Child creation appends to deferred proposal list; does not mutate parent
- Resume reconstructs from persisted work items, not by recomputing parent outcomes

### Reproduction trace under R2 (MacroFinance M4 equivalent)

**Observed M4 sequence (broken):**
1. Attempt 0: `L=3`, ε=0.934, verification → `inconclusive_evidence`, acceptance 0.763
2. Attempt 0: also tested `L=5` → `inconclusive_conflict`
3. Attempt 1: selected `L=5`, ε=0.783 (different `L`!), verification → `repair_step_higher`
4. Recorded repair: higher epsilon 1.565, but never executed
5. Attempt 2: no eligible candidate, terminated `budget_exhausted`

**R2 trace (corrected):**
1. Cohort 1: measure primary grid including `L=3` (candidate A, family F_A) and `L=5` (candidate B, family F_B)
2. A completes measurement, acceptance 0.763 → `inconclusive_acceptance` (overlaps [0.65, 0.75], but wide interval)
3. B completes measurement → `inconclusive_conflict` (assuming A and B both viable but inconclusive)
4. Cohort 1 closes; no directional repair requested from inconclusive evidence
5. Advance A and B to verification cohort (or next measurement stage if design has multiple rungs)
6. A verification: fresh seed, suppose returns `repair_step_higher` with acceptance 0.78
7. Deferred repair request created: child A1 (same `L=3`, higher epsilon, family F_A)
8. Current cohort closes; admit A1 before any different-`L` work
9. A1 measurement and verification with fresh independent seed
   - If A1 verifies: appears in `verified_candidates` with `qualified_repair_status: "executed_and_verified"`
   - If A1 fails or budget exhausts before verification: `qualified_repair_status: "not_executed_with_reason"`, reason = `repair_budget_exhausted`
10. B's verification (or advancement) continues independently

Under R2, the `L=5` candidate cannot be called a repair of `L=3`, and a computed-but-unexecuted repair is explicitly marked.

## 2. Candidate identity and immutability

**Design choice: immutable records with family tracking; soundly addresses identity ambiguity.**

Section 4.1 defines:
- **Immutable candidate record:** once created, fields never change; any modification creates a new child record
- **Candidate ID components:** `(scope_id, search_id, creation_ordinal)` → globally unique
- **Record hash:** repository-issued `candidate_record_hash` from frozen settings
- **Family tracking:** `candidate_family_id` groups a fixed-`L` repair lineage; `parent_candidate_id` links child to source

**Why immutability:**
- Verification receipts reference source candidate by hash; changing epsilon after verification invalidates the hash
- Resume requires exact candidate identity to reconstruct work order
- Replay requires frozen settings to reproduce a kernel
- Parallel execution may reference a candidate while another thread proposes repair

**Epsilon repair = new child, not mutation (section 4.3.2):**
```
An epsilon repair changes epsilon, so it necessarily creates a new immutable 
child candidate ID and record hash. To express "same candidate" semantics 
without contradicting immutability, the child retains parent's 
candidate_family_id and parent_candidate_id and must preserve same scope, 
target, exact L, mass signature, coordinate system, start-bank design, and 
warmup protocol.
```

This is the correct resolution: families express lineage without breaking immutability.

**Test coverage (section 6.4, test 7, MacroFinance trace test):**
- Test 7: repair child inherits parent reserve, parent record preserved, child gets fresh verification seed
- MacroFinance trace: directional high-acceptance for `(L, ε)` creates child with same `L`, higher ε, before different-`L` work is dequeued; computed-but-unexecuted repair marked `not_executed_with_reason`

**No finding:** immutability with family tracking is the right design.

## 3. Queue precedence and determinism

**Confirmed specification: deterministic ordering with same-`L` repair priority; executable from R2 text.**

Section 4.4 specifies seven queue rules; rule 4 is the repair-precedence core:

```
After the active cohort's already-reserved work closes, directional epsilon 
repair children have priority over:
- any not-yet-admitted work for a different L
- advancement of the triggering family's parent to a longer stage

If several repairs requested at one boundary, order by parent work-item order 
then proposal order. Existing reservations for other candidates honored and 
cannot be stolen; this is the fairness exception that makes same-L repair 
deterministic without preempting running work. Repair-priority interval ends 
when child receives declared fresh verification or becomes 
budget/infrastructure pending.
```

**Deterministic tie-breaking (rule 2):**
```
At a boundary, choose lowest unfinished stage among admitted ready candidates. 
Close cohort membership and ordered work-item list before dispatch. Within 
cohort, order by candidate creation ordinal then replication ordinal. 
Acceptance, runtime, completion order do not reorder work.
```

**Mutation during active cohort (rule 3):**
```
During partially completed cohort, append repair/refinement requests to 
deferred proposal list. Neither membership nor existing reservations change. 
Process requests only after active cohort closes, in parent work-item order 
then predeclared proposal order.
```

**Required trace (section 4.4, end):**
```
Cohort [A:r, B:r] starts; A finishes and requests directional same-L repair 
child A1 and refinement child A2 while B pending. Neither child starts nor 
uses B's reserve. Interrupt and resume at that point; B still runs next 
because its work already reserved. Once cohort closes, admit A1 before any 
not-yet-admitted different-L work, then measure A1 with fresh stream and 
verify before advancing A beyond repair barrier. Admit A2 only under declared 
refinement order. B's recorded outcome remains intact.
```

This is complete and unambiguous. An implementation can execute it without further design decisions.

**Fairness exception justification (rule 4 note):**
The repair priority is called a "fairness exception" because it allows a same-`L` child to start before a different-`L` candidate that was proposed earlier. This is justified: the repair tests a hypothesis about the same geometry and trajectory length, so it's a refinement of existing work rather than a replacement. The exception is bounded: repairs don't steal existing reservations, and priority ends after the child verifies or becomes budget-pending.

**Test 11 (section 6.4):**
```
A/B/A1/A2 trace: candidate B starts while A running, A fails and spawns A1, 
A1 enters next cohort (not preempting B), B receives full stage allocation 
regardless of A1 status.
```

**No finding:** queue precedence is deterministic and executable.

## 4. Inconclusive evidence classification

**Confirmed specification: inconclusive ≠ directional; no silent `L` switch.**

Section 4.3.2 defines acceptance evidence classes:

| Evidence class | Acceptance condition | Interpretation | Action |
| --- | --- | --- | --- |
| `directional_lower` | Mean acceptance below lower band `< 0.65`, interval doesn't span target | Epsilon too large | Create child with lower ε, same `L` |
| `directional_higher` | Mean acceptance above upper band `> 0.75`, interval doesn't span target | Epsilon too small | Create child with higher ε, same `L` |
| `inconclusive_acceptance` | Interval overlaps target acceptance, but wide or uncertain | Cannot determine direction | Retain candidate, no automatic repair; may extend evidence or nominate exploration |
| `inconclusive_conflict` | Multiple candidates with conflicting directions at same `L` | Conflicting signals | Retain all viable, no automatic `L` switch |
| `acceptance_in_band` | Mean in `[0.65, 0.75]`, narrow interval | Good proposal efficiency | No repair needed, advance or verify |

**Explicit non-conversion (section 4.3.2):**
```
inconclusive_acceptance (acceptance interval overlaps target, wide or 
uncertain) = neutral signal, no directional repair

inconclusive_conflict (multiple candidates with conflicting directions) = 
retain all, no automatic L switch

Neither becomes a different-L repair; both may nominate additional 
exploration under declared refinement/exploration policy
```

The MacroFinance M4 issue was exactly this: `inconclusive_evidence` led to selecting `L=5` instead of staying with `L=3`. R2 forbids that.

**Evidence extension vs. candidate replacement (section 4.4, rule 6):**
```
Evidence extension of unchanged candidate occupies its next configured rung, 
not a newly inserted lower-stage candidate.
```

An inconclusive result may trigger more draws for the same candidate (evidence extension) or nominate a different `L` for exploration (refinement), but it cannot be recorded as a repair of the inconclusive candidate.

**No finding:** inconclusive evidence is correctly separated from directional evidence.

## 5. Repair execution and verification status

**Confirmed specification: explicit three-state model prevents computed-only handoffs.**

Section 4.5 repair record fields:

```
- execution_status: executed / not_executed
- verification_status: verified / failed / pending / skipped
- qualified_repair_status: executed_and_verified / not_executed_with_reason
- not_executed_reason: repair_budget_exhausted / attempt_limit_reached / 
    wall_time_exceeded / shared_invalidity
```

**Qualified repair status rules (section 4.5):**
```
A computed epsilon is a proposal only. When required repair not run, result 
includes qualified_repair_status="not_executed_with_reason" and typed reason, 
including repair_budget_exhausted; it never emits repaired-kernel handoff.

An unexecuted, executed-but-failed, or budget-pending repair cannot appear 
in verified_candidates.
```

**Three-state progression:**
1. **Computed (proposal):** parent verification returns directional signal, child epsilon computed, child candidate ID and repair record created → `execution_status: "not_executed"`, `qualified_repair_status: "not_executed_with_reason"`
2. **Executed but not verified:** child measurement and verification started → `execution_status: "executed"`, `verification_status: "pending"` or `"failed"` → still `qualified_repair_status: "not_executed_with_reason"` (not verified)
3. **Executed and verified:** child completes final verification, passes all screens → `verification_status: "verified"`, `qualified_repair_status: "executed_and_verified"` → eligible for replay

**MacroFinance M4 under R2:**
The unexecuted `L=5` repair would appear in the result as:
```json
{
  "repair_records": [{
    "parent_candidate_id": "...",
    "child_candidate_id": "...",
    "candidate_family_id": "...",
    "old_epsilon": 0.7826,
    "new_epsilon": 1.5652,
    "exact_L": 5,
    "execution_status": "not_executed",
    "verification_status": "skipped",
    "qualified_repair_status": "not_executed_with_reason",
    "not_executed_reason": "repair_budget_exhausted"
  }]
}
```

The result's `final_status` would be `"repair_budget_exhausted"`, not `"budget_exhausted"` (generic), and the repair payload names the blocked candidate. The child cannot be replayed.

**Test coverage (MacroFinance trace test):**
```
Computed-but-unexecuted repair visibly marked and cannot produce qualified 
frozen kernel
```

**No finding:** repair execution status is explicit and prevents false handoffs.

## 6. Budget accounting and incomplete status

**Confirmed specification: typed budget exhaustion, explicit incomplete repair, no silent failure.**

Section 4.4 (rule 5) and section 4.5 define budget-exhaustion handling:

**When repair cannot be funded (section 4.4, rule 5):**
```
If required repair cannot be executed within declared total, attempt, stage, 
or wall-time budget, preserve as proposal_budget_pending, set 
qualified_repair_status to not_executed_with_reason, and record 
repair_budget_exhausted when budget is the cause. Continue already reserved 
work.
```

**Result status when repair blocked (section 4.5):**
```
final_status: "repair_budget_exhausted"  # specific, not generic
completion_status: "partial_budget"
repair payload names exact blocked candidate
```

**Distinction from generic budget exhaustion (section 5):**
- `final_status: "partial_budget"` = some candidates incomplete, no specific repair blocked
- `final_status: "repair_budget_exhausted"` = a required directional repair was identified but could not be executed due to budget
- `final_status: "complete"` = all required work finished, regardless of verification outcomes

A scope can have `verified_candidates` ≠ ∅ while `completion_status: "partial_budget"` (some candidates incomplete but at least one verified).

**Budget reserve inheritance (section 4.4, rule 5):**
```
Admit children from separate declared exploration/repair reserve, after 
reserving their measurement and required validation through final 
verification. They never inherit parent's verification or consume another 
active candidate's reservation.
```

This contradicts the thorough audit's understanding that "repair child inherits parent reserve." Reading section 4.4 more carefully:

**Clarification from section 4.4, rule 5 full text:**
```
Unspent reservations released by terminal candidate return to free pool at 
boundary, with explicit accounting entry; a child needs its own allocation.
```

So the child does NOT inherit the parent's unspent reserve directly. Instead:
1. Parent becomes terminal (failed verification, vetoed, or repair-triggering)
2. Parent's unspent reserve returns to the free pool
3. Child is allocated from the exploration/repair reserve (a separate budget line)
4. The parent's returned reserve increases the free pool, which may fund the child indirectly

**Test 7 revision needed:**
The thorough audit stated "repair child inherits parent reserve." Section 4.4 rule 5 says children are allocated from the repair reserve, not by inheritance. Test 7 should verify:
- Parent's unspent reserve returns to pool with accounting entry
- Child allocated from repair reserve (or pool if repair reserve is pooled)
- Parent record preserved
- Child gets fresh verification seed

**Finding: minor test clarification needed**
- **Severity:** low; does not block implementation
- **Issue:** test 7 description may imply direct inheritance; section 4.4 rule 5 says allocation from repair reserve
- **Resolution:** P0 test should verify the accounting: parent reserve → pool, child allocation ← repair budget, with explicit accounting entries
- **Not a plan defect:** the text is clear; the test description was imprecise

**No other finding:** budget exhaustion produces typed status with repair details.

## 7. Test oracle strength and MacroFinance trace

**Confirmed specification: MacroFinance trace is a concrete adversarial oracle.**

Section 6.4 test obligations include:

**Test 16 (new): MacroFinance M4 reproduction trace:**
```
Demonstrate directional high-acceptance result for (L, epsilon) creates child 
with same L, higher epsilon, before different-L work dequeued; computed-but-
unexecuted repair marked not_executed_with_reason; injected outcome schedule 
confirms exact order and repair precedence
```

This is stronger than "does same-`L` repair happen before different-`L` work" because it:
1. Uses the actual M4 trace: `L=3` inconclusive, `L=5` measured, `L=5` directional, but repair not executed
2. Requires injected outcome schedule so the test controls when each candidate returns which signal
3. Verifies exact queue order (A finishes → deferred repair A1 created → B finishes → cohort closes → A1 starts before any different-`L` work)
4. Checks that computed repair without execution produces `not_executed_with_reason`, not a verified handoff

**Relation to test 11 (A/B/A1/A2 trace):**
Test 11 covers mutation during active cohort (A finishes while B running → neither A1 nor A2 preempt B). Test 16 covers repair precedence after cohort closes (A1 starts before different-`L` work). Both are required.

**Other strengthened tests (section 6.4):**
- Test 7: parent record preserved, child fresh seed, no verification inheritance
- Test 12: holdout lineage with duplicate seed rejection
- Test 13: cross-scope replay forbidden

**No finding:** test oracles are concrete and adversarial.

## 8. Remaining qualifications from thorough audit

The thorough audit identified three qualifications that still apply to R2:

**1. Start-bank construction (section 4.1, P0 obligation):**
R2 text:
```
The prepared scope uses user-supplied or repository-built dispersed start 
banks, preserving their design, any mode or mixture components in named 
coordinates, random seeds, and target-specific coverage check. For example, 
controlled multimodal fixture checks representation of known modes; general 
target records checked dispersion and remaining unknown coverage. No 
universal dispersion threshold or sufficiency claim introduced here.
```

This assigns the recipe to P0 but does not specify it. The thorough audit recommended: MLE or prior mode ± `k × diag(Σ)^{1/2}` in four orthogonal directions, with `k ≈ 2`.

**Assessment:** qualification remains; P0 must define the operational recipe before first serious scope. Not a blocker.

**2. Mass-chapter coefficient provenance (section 8.2, guidebook obligation):**
R2 assigns this to P4 guidebook revision:
```
Audit budget formulas/coefficients; preserve justified mathematics while 
labeling heuristic allocations accurately.
```

The specific coefficients `32 × d_eff`, `64 × d_eff`, `250 × d_eff` still need provenance or explicit hypothesis labels.

**Assessment:** qualification remains; P4 must resolve before guidebook promotion. Not a blocker.

**3. Cross-scope efficiency ranking uncertainty (section 4.5.2):**
R2 preserves the descriptive-nomination semantics with uncertainty status:
```
Optional efficiency nomination uses predeclared observables, budgets, 
independent replications, explicit objective such as minimum relevant ESS per 
measured target-gradient evaluation or per wall time. These objectives not 
interchangeable. Descriptive nominee useful when uncertainty inconclusive, 
but choice never removes other verified members or establishes superiority. 
Statistical ranking requires appropriate uncertainty and multiplicity 
analysis for declared candidate family.
```

**Assessment:** qualification remains; any cross-scope ranking use must preserve uncertainty. Not a blocker.

None of these qualifications are changed by R2's epsilon-repair additions.

## 9. Evidence coverage and phase boundaries

| Required check | Status | Evidence boundary |
| --- | --- | --- |
| R2 plan complete read | Inspected | Full R2 plan, sections 1–8 and phase table |
| MacroFinance handoff memo | Inspected | Complete memo, all seven requested changes |
| Candidate identity immutability | Verified in plan | Section 4.1, family tracking, hashed records |
| Same-`L` repair priority | Verified in plan | Section 4.4 rule 4, deterministic precedence |
| Inconclusive evidence classification | Verified in plan | Section 4.3.2, no silent `L` switch |
| Repair execution status | Verified in plan | Section 4.5, three-state model |
| Budget exhaustion typing | Verified in plan | Section 4.4 rule 5, section 4.5 status |
| Parent evidence preservation | Verified in plan | Section 4.1, 4.4, immutable records |
| MacroFinance trace test | Verified in plan | Section 6.4 test 16, adversarial oracle |
| Test 7 reserve allocation | Minor clarification needed | Section 4.4 rule 5 vs. test description |
| P0–P5 phase definitions | Verified in plan | Section 6, clear boundaries |
| Guidebook revision scope | Verified in plan | Section 7, substantive changes |
| Downstream compatibility | Out of scope | P3 static audit, owner adoption |
| Numerical parity, GPU/XLA | Out of scope | P5 qualification evidence |
| Rendered guidebook | Out of scope | P4 rendering and review |

## 10. Disposition by area

| Area | R2 disposition | Changes from thorough audit |
| --- | --- | --- |
| Candidate identity | Implementable | Immutability and family tracking added |
| Epsilon repair | Implementable | Same-`L` repair, priority, execution status added |
| Queue precedence | Implementable | Repair-priority exception and deterministic ordering specified |
| Inconclusive evidence | Implementable | Explicit non-conversion to directional repair |
| Budget accounting | Implementable (one test clarification) | Typed repair exhaustion, explicit incomplete status |
| Verification status | Implementable | Three-state model prevents computed-only handoffs |
| Test coverage | Implementable | MacroFinance trace test added (test 16) |
| MacroFinance issue | Resolved | All seven requested changes specified |
| Guidebook | Scope complete, rendering future | Same as thorough audit |
| Qualifications | Three remain, none block P0–P5 | Same as thorough audit |

## 11. Remaining defaults and provenance

No new numerical defaults were introduced in R2. The epsilon repair factor (default 2.0) is stated as inherited from the current implementation (section 4.3.2):

```
repair_factor: 2.0  # inherited from current bracketed step repair
```

This is appropriate: the factor is a heuristic multiplier, not a derived constant. It can be scope-configured if needed.

All other defaults (primary `L` grid, acceptance bands, budget coefficients, chain count, R-hat/ESS thresholds) remain as specified in R1 and assessed in the thorough audit.

## 12. Smallest justified next action

**Implement P0–P5 as specified in R2.**

The plan is coherent, complete, and executable. The MacroFinance issue is resolved with immutable candidate identity, same-`L` repair priority, and explicit execution status. Test oracles are adversarial and concrete. Phase boundaries are clear.

**Not justified:**
- Another planning cycle or audit
- Rewriting R2 to address test 7 reserve allocation (minor clarification, not a defect)
- Deferring implementation until downstream consumers, GPU qualification, or rendered guidebook exist (those are P3–P5 evidence)

**Next review:**
One focused implementation audit after P0–P5 artifacts exist: executable tests, numerical parity checks, backend qualification, downstream compatibility findings, and guidebook rendering.

## 13. True blockers to implementation

**None.**

The test 7 clarification (reserve allocation accounting) is a precision improvement, not a blocker. P0 can implement the test as specified in section 4.4 rule 5 directly.

## Verdict justification

R2 resolves the MacroFinance epsilon-repair issue completely:

1. ✓ Immutable candidate identity including `L`, mass, epsilon
2. ✓ Same-`L` repair preserves exact `L` and mass, changes only epsilon
3. ✓ Repaired same-`L` candidate has priority and receives fresh verification
4. ✓ Inconclusive evidence does not silently promote different `L`
5. ✓ Repair status explicit: `executed_and_verified` or `not_executed_with_reason`
6. ✓ Budget exhaustion produces typed `repair_budget_exhausted`
7. ✓ Parent evidence preserved, never overwritten

The specification is complete, deterministic, and executable. Queue precedence is unambiguous. Test oracles are adversarial. Qualifications from the thorough audit remain but do not block P0–P5.

This is a planning audit. It confirms R2 is implementable as specified within stated boundaries. It does not certify code works, tests pass, GPU qualifies, downstream migrates, guidebook renders, or posterior inference is valid. One implementation audit after artifacts exist is the next review.

VERDICT: AGREE
