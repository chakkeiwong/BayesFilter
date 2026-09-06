# LEDH Surrogate-Force HMC — Uninterrupted Execution Contract

**Date:** 2026-09-07
**Binds:** All phases of `ledh-surrogate-hmc-executable-master-program-2026-09-07.md`
**Authority:** CLAUDE.md § Campaign Repair And Retry; owner directive 2026-09-07

---

## The rule

Execution runs from Phase 0 to program completion **without stopping for
approval**. Failures are repaired and retried. Only the five conditions in
§ "Genuine stops" below halt execution.

This is not a relaxation of rigor. Every promotion criterion, veto, and
non-claim in the master program stands unchanged. What is retired is the
reflex of stopping to ask about problems that are mine to fix.

### Why this contract exists

The owner runs ~19 agents across multiple projects. A mid-execution question
costs hours of context reconstruction, and a mathematical question cannot be
answered responsibly without investigation the owner has no budget for while
managing other work. Start-stop cycles are therefore a primary source of
drift, not a safety mechanism. A program that took days to specify should
run to completion on that specification.

CLAUDE.md § Campaign Repair And Retry already grants this:

> Repair and retry a localized infrastructure, harness, serialization,
> multiprocessing, or resource failure **without renewed approval** when the
> target, data, method, promotion criteria, vetoes, hardware class, privacy
> boundary, and total campaign budget remain unchanged. … Stop for new
> direction **only** when the scientific contract or budget changes, a true
> continuation veto fires, or the campaign budget is exhausted.

The handoff notes written 2026-09-07 contained 14 stop-and-report
instructions. On audit, approximately none met that bar; they were
implementation bugs in code not yet written. Those instructions are
superseded by this contract.

---

## Genuine stops (exhaustive)

Execution halts **only** on these. The list is closed: if a situation is not
on it, repair and continue.

### S1. Resource unavailable after escalated retry
GPU absent, driver failure, or environment broken, and an escalated retry
reproduces it. I cannot repair hardware or drivers.

### S2. Campaign budget exhausted
44 GPU-hours or 20 repair attempts consumed. Budget is a hard boundary, not
a guideline.

### S3. The scientific contract would have to change to proceed
The estimand, primary criterion, comparator, veto, or model set would need
altering. Changing what the program certifies is the owner's decision, never
mine — this is the exact failure that produced the true-θ coverage defect.

### S4. A Corollary 5.2 premise is unsatisfiable in principle
Not "my adapter has a bug" but "no implementation can satisfy this premise."
Requires exhausting the repair budget for that premise first, and a written
argument for why the premise cannot hold. Evidence, not suspicion.

### S5. Irreversible or outward-facing action
Publication, external transmission, credential use, destructive operation
outside `results/` and the program's declared artifact paths.

**Nothing else stops execution.** Not a failing test, not a veto firing, not
a negative scientific result.

---

## Repair-and-continue protocol

On any failure not in S1–S5:

1. Classify: implementation bug / harness bug / fixture bug / tolerance
   mis-set / genuine numerical failure.
2. Repair under the unchanged contract.
3. Re-run the failing check.
4. Log to `results/repair-log.jsonl` — one line per attempt:
   `{phase, failure_class, diagnosis, repair, outcome, attempt_n, wall_s, gpu_h_used}`
5. Continue.

**Repair budget:** 3 attempts per distinct failure, 20 total for the
campaign. A failure surviving 3 targeted repairs is reclassified as a
finding, recorded, and execution continues to the next phase that does not
depend on it. If every remaining phase depends on it, that is S4 — and the
S4 write-up must show the three attempts.

Failed attempts consume budget. They do not require permission.

---

## Vetoes are results, not interruptions

The master program's vetoes govern **promotion**, not execution. This
distinction is what the audits kept collapsing.

| Event | Old (defective) behaviour | Contract behaviour |
|---|---|---|
| Phase 2 W₂ > tolerance | stop, ask user | debug wiring, repair, re-run; if it survives repair, record "mechanism fails on analytical gradient" and continue to Phase 3 to localize |
| Phase 3 V1/V2/V3 fails | stop, ask user | repair adapter, re-run; if it survives, record premise-violation finding and continue |
| Phase 4a W₂ > threshold | stop, ask user | this is the diagnostic gate working: skip 4b (saves 6–9 GPU-h), run Phase 3 localization, report |
| Phase 4b W₂ > threshold | stop, ask user | **negative result = program complete.** Corollary 5.2 does not hold for this construction. Write it up. Do not run Phase 5. |
| Phase 5, 1 of 3 models fails | stop, ask user | record per-model, continue remaining models, report the split |

A negative result is a **completed program**, not a blocker. "Surrogate-force
HMC does not sample π_N^ω correctly" answers the research question. It gets
written up and delivered, not escalated mid-flight.

---

## Decision gates: automatic

Every gate in the master program evaluates from measured artifacts, with no
judgment call:

- **Phase 4a → 4b:** W₂ < threshold → run 4b. Else skip 4b, localize, report.
- **Phase 4b → 5:** primary criterion passes and no veto fired → run Phase 5.
  Else report and end.
- **Within Phase 5:** each model independent; one failure does not stop the
  others.

Pre-approved: the full 44 GPU-hour envelope, all phase transitions, and all
repair activity inside it.

---

## Reporting: end of program, or genuine stop

**During execution:** nothing is sent. Progress is observable in files —
`results/phase*-summary.json`, `results/repair-log.jsonl`,
`docs/plans/phase*-complete.md`. Inspect whenever convenient; no
notification will arrive demanding attention.

**At completion, one report:** what was certified, what failed, every repair
made, budget consumed, and the scientific verdict with its non-claims.

**At a genuine stop (S1–S5):** which condition fired, the evidence, what was
completed, and the minimum decision needed to resume.

---

## What is out of scope for repair

Repairs may fix implementation, harness, fixtures, tolerances that were
mis-derived, wiring, and infrastructure.

Repairs may **not** touch: the estimand, the primary criterion, any veto
threshold, the comparator, the model set, the budget, or any non-claim.
Those are contract terms. Needing to change one is S3.

---

## Provenance

Adopted 2026-09-07 after the handoff notes were found to contain 14
stop-and-report instructions, of which an audit found approximately zero met
the CLAUDE.md bar for halting a campaign. Root cause: a stop-on-any-failure
reflex applied to failures that are the ordinary content of execution. The
underlying pattern — an agent optimizing locally, asking the owner to absorb
context the agent should carry — is the same pattern that produced three
months of drift on this program.
