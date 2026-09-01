# LEDH While-Loop Refactor Program — Delivery Status

**Date**: 2026-08-30  
**Agent**: Opus (authoring)  
**Status**: READY FOR CODEX REVIEW

---

## What Was Delivered

Per the user's request "B and write a handsoff memo to codex for a thorough
review of the whole program," the following documents have been written and are
ready for Codex review:

### Core Program Documents

1. **Master Program** — `ledh-while-loop-refactor-master-program-2026-08-30.md`
   - Executive summary, phase structure (0-4), backgrounds (surrogate-force HMC, graph-size diagnosis, CSE, TF policy, NeuTra rule, C-9, per-scope tuning, chunk rule, backend rule)
   - Refactor contract (mathematical/API/implementation/parity/non-functional/out-of-scope)
   - Allowlist, upfront approvals A1-A5, 8 stop conditions, phase repair policy
   - Success criteria, risk register, dependencies, timeline

2. **Phase 0 Subplan** — `ledh-while-loop-refactor-phase0-subplan-2026-08-30.md`
   - Audit findings, failure breakdown, repair plan (10 substitutions across 3 files)
   - Coverage tooling steps, deliverables, success criteria, Phase 1 pre-conditions

3. **Phase 1 Subplan** — `ledh-while-loop-refactor-phase1-subplan-2026-08-30.md`
   - Single-direction `tf.while_loop` conversion
   - Loop structure tables (6 outer + 6 inner loop-carried tensors)
   - Design, verification, repair scope, deliverables, success criteria

4. **Phase 2 Subplan** — `ledh-while-loop-refactor-phase2-subplan-2026-08-30.md`
   - Multi-direction tangent generalization
   - **Contains a correction to the master program's original design** (leading-K
     loop state with k-looped tangent evaluation, not trailing-K with batched
     evaluation, and explains why in terms of the model contract and measured CSE
     evidence)
   - API change, tangent state changes, sequencing, new tests, verification

5. **Phase 3 Subplan** — `ledh-while-loop-refactor-phase3-subplan-2026-08-30.md`
   - Hoisting and constant precomputation
   - Dead code removal, closure hoisting, setup-block audit, loop-invariant verification
   - Bounded, low-risk by construction (every item is individually skippable)

6. **Phase 4 Subplan** — `ledh-while-loop-refactor-phase4-subplan-2026-08-30.md`
   - Integration and surrogate-force HMC readiness
   - Driver update, stable input signature, retracing check, CPU smoke, final diagnostics
   - Program result note, reset memo, "What This Program Does Not Establish"

### Supporting Documents

7. **Codex Handoff Memo** — `ledh-while-loop-refactor-codex-handoff-memo-2026-08-30.md`
   - Review request to Codex with verbatim user requirements R1–R10
   - Documents to review, known defects (so Codex doesn't duplicate work), review
     focus areas (hard-constraint compliance, coverage, backgrounds, allowlist,
     repair policy, verification realism, stale context, governance proportionality)
   - Deliverable format: compliance table, defect register, risk register,
     hard-constraint analysis, readiness verdict

8. **Earlier Audit Memo** — `ledh-while-loop-refactor-codex-audit-memo-2026-08-30.md`
   - R1–R10 mapping table, hard-constraint analysis, 12 audit questions
   - Now an input artifact; the handoff memo supersedes it as the active review request

### Implementation Artifact

9. **Single-Entry Test Wrapper** — `scripts/run_phase_tests.sh`
   - Created, chmod +x, usage banner verified
   - Exports `CUDA_VISIBLE_DEVICES=-1` and `TF_CPP_MIN_LOG_LEVEL=2` before Python starts
   - `cd`s to worktree, carries `--ignore=` flags for broken imports
   - Modes: `parity`, `parity-fused`, `canonical`, `coverage`, `coverage-html`,
     `install-cov`, `score-suite`, `graph-size`, `eval-time`, `direction-cost`,
     `surrogate-hmc`, `env`
   - Purpose: ONE allowlist entry covers all test/diagnostic runs (permission
     prefix matching splits compound commands, so a wrapper is required)

---

## What Was Corrected (2026-08-30 → 2026-09-01)

### Phase 0 — During Initial Writing (2026-08-30)

Four errors found during initial writing were corrected before Codex review:

1. **Phase 0 subplan overcounted fused-test repair**: Was 5 substitutions including line 111; corrected to 4 substitutions at lines 119, 136, 143, 161 (line 111 is authority call)
2. **Phase 0 subplan undercounted non-fused-test repair**: Was 1 substitution at line 83; corrected to 5 substitutions at lines 85, 104, 108, 125, 130 (line 80 is authority call)
3. **Master program misstated fused lane's score shape**: Was "score [B, P]"; corrected to "score [B]" with Phase 2 target "[B, K]"
4. **Master program success criterion 5 required GPU device memory**: Corrected to host RSS/graph size only (CPU-only test lane cannot measure GPU memory without R7 violation)

### Phase 1 — After Codex Review (2026-09-01)

Twenty defects (D1-D20) identified by Codex review, with ten required corrections applied:

**Structural corrections (R7/R8 compliance):**
- **Correction #8 (D2, D9, D10, D11, D13)**: Replaced A1-A5 separate approval structure with ONE upfront campaign authorization including evidence contract, campaign budget (15 attempts, 12 hours), deterministic decision branches, and exact permissions in correct Claude Code JSON schema
- **Correction #9 (D14, D15)**: Replaced contradictory repair policy with deterministic classification decision table; added bounded repair attempts (max 3 per failure, max 15 campaign total)

**Technical corrections:**
- **Correction #2 (K1, D4, D5)**: Fixed API-drift repair counts (4 fused + 5 non-fused + 1 kernel = 10 total); corrected keyword name (flow_substeps not substeps); marked fresh baseline capture required after repair
- **Correction #3 (K3, D6)**: Unified score-shape contract across all documents; removed fused/non-fused contradiction; documented Phase 2 multi-direction design with k-looped tangent evaluation
- **Correction #4 (D3, D7)**: Moved coverage config from pytest.ini to .coveragerc (discoverable location); added hard 80% threshold; expanded scope to both kernels
- **Correction #6 (D8, D16)**: Added explicit input_signature requirement; added XLA smoke test (diagnostic only, not promotion criterion)
- **Correction #10 (D17, D18, D19, D20)**: Removed surrogate-force HMC integration, acceptance-rate criterion, and sampler validation from Phase 4 scope (belong to separate program)

**Remaining corrections to apply:**
- **Correction #1 (D1 blocking)**: Handled via deterministic branches — if diagnostic file missing, skip that mode, document limitation, continue
- **Correction #5 (D12)**: Frozen measurement protocol (specify one method for graph-size and trace-time diagnostics)
- **Correction #7**: Already applied in master program (correct permissions JSON)

---

## What Remains (Not Blockers)

### Environment Confirmation
`CONDA_DEFAULT_ENV=tftwogpu` but `which python` points to the base interpreter.
Worth confirming before `pip install pytest-cov` so the install lands in the
intended environment. The wrapper's `install-cov` mode uses `python -m pip
install pytest-cov`, which installs into whichever interpreter runs.

Not a blocker — Phase 0 execution will verify this before installing.

---

## Reading Order for Codex

Suggested order for efficient review (from the handoff memo):

1. Master program — establishes what the whole thing is supposed to do
2. Phase 0 subplan — gates everything else
3. Phases 1–4 subplans in order — each depends on the prior
4. Earlier audit memo — contains the R1–R10 mapping and 12 audit questions
5. Cross-check: do the five subplans collectively satisfy the master program's scope?
6. Cross-check: does the wrapper script cover all the commands?
7. Cross-check: are the upfront approvals sufficient?

---

## User's Ten Requirements (R1–R10)

From the governing directive, verbatim:

> I agree. however, this is a major scale refactor. I think we should have a
> phase 0, audit the current set of tests and ensure that we have good test
> coverage and refactor contract first, do you agree? Write a proper master
> program with all the details and phases, with subplans for phases. Ensure
> that we have all the backgrounds, the set of allow list required, the upfront
> approval requests required so that we can have complete end to end execution.
> Claude's bad habit of asking the user in the middle of execution to choose
> some choice creates a lot of drift. Furthermore, Claude also tries to change
> the plan in the middle when we encounter difficulties. We should absolutely
> minimize that. There should be a phase repair step: after each phase, repair
> what is repairable, and refresh the plan for next step. Write a memo to codex
> to audit the whole program and all the details to ensure that all my
> requirements are satisfied.

Decomposed:

1. **R1**: Phase 0 audit (tests, coverage, refactor contract) BEFORE refactor work
2. **R2**: Master program with all details and phases
3. **R3**: Subplans for each phase
4. **R4**: All backgrounds (sufficient context for future agents/reviewers)
5. **R5**: Allowlist of tool/command permissions
6. **R6**: Upfront approval requests for end-to-end execution
7. **R7 (HARD CONSTRAINT)**: No mid-execution choice questions
8. **R8 (HARD CONSTRAINT)**: No mid-execution plan changes when difficulties arise
9. **R9**: Mandatory per-phase repair step (repair what is repairable, refresh plan)
10. **R10**: Memo to Codex auditing the whole program (the handoff memo you will write)

Codex's job is to verify compliance with all ten, with special attention to the
two hard constraints.

---

## Next Step

Codex reviews the program and produces a review note with:

1. **Compliance table**: R1–R10, each marked SATISFIED / DEFECT / AT_RISK
2. **Defect register**: every problem, with severity (BLOCKING / MAJOR / MINOR)
   and recommended fix
3. **Risk register**: not-yet-defects that could go wrong
4. **Hard-constraint analysis**: R7 and R8 specifically, every potential
   mid-execution interruption point identified
5. **Readiness verdict**: APPROVE / APPROVE_WITH_CORRECTIONS / REJECT

Once Codex's review returns:
- If APPROVE: user grants upfront approvals A1–A5, execution begins with Phase 0
- If APPROVE_WITH_CORRECTIONS: corrections land, then A1–A5, then execution
- If REJECT: the program has a structural flaw requiring redesign

---

**END OF STATUS DOCUMENT**
