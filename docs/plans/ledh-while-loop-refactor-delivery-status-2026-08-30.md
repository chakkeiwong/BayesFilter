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

## What Was Corrected

Four errors found during writing have been corrected in the documents Codex will
review:

### Error 1: Phase 0 subplan overcounted fused-test repair
- **Was**: 5 substitutions in `test_ledh_canonical_batch_fused.py` including line 111
- **Now**: 4 substitutions at lines 119, 136, 143, 161; line 111 is an authority
  call that takes `flow_substeps=` correctly and must not change
- **Location**: Phase 0 subplan, "Files to Repair" section

### Error 2: Phase 0 subplan undercounted non-fused-test repair
- **Was**: 1 substitution at line 83 in `test_ledh_canonical_batch.py`
- **Now**: 5 substitutions at lines 85, 104, 108, 125, 130; line 80 (not 83) is
  an authority call that must stay `flow_substeps=`
- **Corrected total**: 10 substitutions across 3 files (4 fused + 5 non-fused + 1 kernel)
- **Location**: Phase 0 subplan, "Files to Repair" section

### Error 3: Master program misstated the fused lane's score shape
- **Was**: "theta [B, P] → value [B], score [B, P]"
- **Now**: "theta [B, P] with theta_directions [B, P] → value [B], score [B]"
  plus an explanation that score shape is `[B]` (one directional derivative per
  row), not `[B, P]`, and that Phase 2 extends to `[B, K]`
- **Why it matters**: Phase 2's purpose is turning the `[B]` single-direction
  return into `[B, K]`; the original text made it sound like `[B, P]` already existed
- **Location**: Master program, "NeuTra Batch-Native Training Rule" section

### Error 4: Master program success criterion 5 required GPU device memory
- **Was**: "device memory usage at (B=6, horizon=50, N=252) ≤ 2× current"
- **Now**: "host RSS, graph node count, and GraphDef bytes at (B=6, horizon=50,
  N=252) ≤ 2× current. GPU device-memory validation is deferred as a separate
  owner-scheduled escalated step."
- **Why it matters**: the test lane is CPU-only by design
  (`CUDA_VISIBLE_DEVICES=-1`), so measuring GPU device memory would require an
  escalated mid-execution prompt, violating the zero-interruption requirement (R7)
- **Location**: Master program, success criterion 5 and stop condition 6; also
  added "What This Program Does Not Establish" section recording GPU validation
  as an open item

---

## What Remains (Not Blockers)

### Allowlist JSON Schema
The master program's allowlist section presents JSON in an invented schema. The
real Claude Code settings format is `{"permissions": {"allow": ["Bash(...)"]}}`.
This needs rewriting before the user pastes it into settings, but it is not a
blocker for Codex review — Codex is reviewing the program logic, not the JSON syntax.

The wrapper-script approach is sound: one script, one allowlist entry, covers
all runs.

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
