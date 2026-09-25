# Codex Audit Memo: LEDH tf.while_loop Refactor Program

**To**: Codex (Independent Reviewer)  
**From**: Claude (Program Author)  
**Date**: 2026-08-30  
**Subject**: Audit Request for LEDH Canonical Batch Fused tf.while_loop Refactor Master Program  
**Program ID**: `ledh-while-loop-refactor-2026-08-30`

---

## Audit Request

I request an independent audit of the master program and Phase 0 subplan for the LEDH canonical batch-fused kernel refactor. The user's governing directive requires that "all my requirements are satisfied" before execution begins.

**Audit Scope**:
1. User requirements completeness (all items from governing directive addressed)
2. Hard constraint compliance (no mid-execution questions, no mid-execution plan changes, mandatory repair-and-refresh)
3. Allowlist sufficiency (can the agent execute autonomously within declared permissions?)
4. Stop conditions clarity (when MUST the agent stop vs when CAN it repair autonomously?)
5. Technical soundness (is the tf.while_loop approach viable? Are risks identified?)
6. Phase structure coherence (does Phase 0 prepare Phase 1? Does each phase build on the previous?)

---

## User Requirements (Governing Directive)

**Original Request** (from prior session, preserved in summary):

> "I agree. however, this is a major scale refactor. I think we should have a phase 0, audit the current set of tests and ensure that we have good test coverage and refactor contract first, do you agree? Write a proper master program with all the details and phases, with subplans for phases. Ensure that we have all the backgrounds, the set of allow list required, the upfront approval requests required so that we can have complete end to end execution. Claude's bad habit of asking the user in the middle of execution to choose some choice creates a lot of drift. Furthermore, Claude also tries to change the plan in the middle when we encounter difficulties. We should absolutely minimize that. There should be a phase repair step: after each phase, repair what is repairable, and refresh the plan for next step. Write a memo to codex to audit the whole program and all the details to ensure that all my requirements are satisfied."

**Decomposed Requirements**:

| Requirement | Document Section | Status |
|-------------|-----------------|--------|
| R1: ADD Phase 0 (test audit, coverage, refactor contract) | Master Program, Phase 0 Subplan | ✓ Addressed |
| R2: Write master program with all details and phases | Master Program document | ✓ Addressed |
| R3: Write subplans for phases | Phase 0 subplan written; Phase 1-4 subplans to be written | ⚠ Partial (Phase 0 only so far) |
| R4: Include all backgrounds (sufficient context for future agents/reviewers) | Master Program: Background section (surrogate-force HMC, graph-size diagnosis, CSE behavior, UKF lifecycle, tangent recursion, policies) | ✓ Addressed |
| R5: Include allowlist of tool/command permissions | Master Program: Allowlist section | ✓ Addressed |
| R6: Include upfront approval requests for end-to-end execution | Master Program: Upfront Approval Requests (A1-A5) | ✓ Addressed |
| R7: **HARD CONSTRAINT**: NO mid-execution choice questions to user | Master Program: Allowlist (pre-approved), Stop Conditions (explicit gates), Repair Policy (autonomous bounds) | ✓ Addressed |
| R8: **HARD CONSTRAINT**: NO mid-execution plan changes when difficulties arise | Master Program: Phase Repair Policy (repair within declared bounds, stop for fundamental problems), Stop Conditions | ✓ Addressed |
| R9: Mandatory phase repair-and-refresh step after each phase | Master Program: Phase Repair Policy (binding), Phase 0 Subplan: Deliverables include result doc + Phase 1 pre-condition verification | ✓ Addressed |
| R10: Write memo to Codex to audit the program | This document | ✓ Addressed |

**Audit Question 1**: Are all 10 user requirements addressed in the program documents? If any are missing or incomplete, specify which.

---

## Hard Constraint Compliance

### R7: No Mid-Execution Choice Questions

**User's Concern**: "Claude's bad habit of asking the user in the middle of execution to choose some choice creates a lot of drift."

**Program Mechanisms**:
1. **Pre-Approved Allowlist** (Master Program, Allowlist section):
   - File operations: READ any file in worktree; WRITE/EDIT 7 specific files plus new test/plan files
   - Bash commands: all pytest runs, coverage measurement, git status/diff/add/commit, TF version check, diagnostic scripts
   - Git operations: stage/commit (semantic messages), log, show, cat-file
   - Python packages: `pip install pytest-cov` (Phase 0 only)
   - Explicitly NOT approved: push, merge, rebase, force operations, other package installs

2. **Upfront Approval Requests** (A1-A5):
   - A1: Execute Phase 0 repair?
   - A2: Accept refactor contract?
   - A3: Allowlist sufficient?
   - A4: Repair policy acceptable?
   - A5: Autonomous phase-to-phase execution acceptable?
   - All 5 require YES/NO/REVISE before Phase 1 begins

3. **Phase 0 Subplan Repair Decisions**:
   - API drift repair strategy: Option A selected (preserve `substeps=`, fix tests and forwarding) with explicit rationale
   - Out-of-scope dispositions: missing fixtures, leaderboard artifacts, collection errors explicitly recorded
   - No open "should I do X or Y?" questions remain

**Audit Question 2**: Does the program eliminate mid-execution choice questions? Are there any remaining decision points where the agent would need to ask the user to choose between options?

### R8: No Mid-Execution Plan Changes

**User's Concern**: "Claude also tries to change the plan in the middle when we encounter difficulties. We should absolutely minimize that."

**Program Mechanisms**:
1. **Stop Conditions** (Master Program, 8 explicit conditions):
   - Parity gate failure beyond tolerance (rtol 5e-4)
   - API break (existing callers fail)
   - Silent behavior change (no identified root cause)
   - tf.while_loop compatibility blocker
   - XLA incompatibility introduced
   - Memory regression > 2×
   - Performance regression > 2×
   - Fundamental architectural problem
   - **When a stop condition fires**: STOP, document, preserve artifacts, request user direction (do NOT speculate, do NOT change plan)

2. **Phase Repair Policy** (Master Program, binding):
   - Autonomous repair ALLOWED for: trivial bugs, numerical discrepancies within tolerance, minor API mismatches, documentation drift, dead code, lint/format
   - Autonomous repair FORBIDDEN for: anything triggering a stop condition
   - After repair: refresh plan for next phase IF repair revealed new info, ELSE proceed

3. **Refactor Contract** (Master Program, binding):
   - Mathematical, API, implementation, parity-gate, non-functional constraints are NON-NEGOTIABLE
   - Out-of-scope items EXPLICITLY EXCLUDED (no scope creep)

**Audit Question 3**: Does the program prevent mid-execution plan changes? Are stop conditions clear enough that an agent would know when to stop vs when to repair autonomously?

### R9: Mandatory Phase Repair-and-Refresh

**Program Mechanisms**:
1. **Phase 0 Subplan** includes:
   - Repair Plan (3 repairs with exact changes)
   - Deliverables (repaired files, coverage infra, Phase 0 result doc, git commit)
   - Success Criteria (7 checkboxes)
   - Phase 1 Pre-Conditions (6 items, verified after Phase 0 repair)

2. **Master Program, Phase Repair Policy**:
   - Mandatory Repair Actions (6 categories, NO USER PERMISSION REQUIRED)
   - Refresh Plan for Next Phase (4 steps)
   - Example Repair Scenario (Phase 1 parity test failure within tolerance)

**Audit Question 4**: Is the repair-and-refresh step mandatory and well-defined for each phase?

---

## Allowlist Sufficiency

**Pre-Approved Actions**:
- Read any file in worktree
- Write/Edit 7 specific files + new test/plan files
- Run pytest (CPU-only, no escalation needed)
- Run pytest with GPU (escalation required, user will grant via tool permissions)
- Measure coverage
- Git status/diff/add/commit (semantic messages)
- Install pytest-cov (Phase 0 only)

**NOT Pre-Approved** (require explicit user direction):
- Push to remote
- Merge worktree → main
- Rebase, force operations
- Install any package other than pytest-cov

**Audit Question 5**: Is the allowlist sufficient for autonomous Phase 0-4 execution? Are there any likely scenarios where the agent would need a permission not on the list?

---

## Stop Conditions Clarity

**8 Stop Conditions** (Master Program):
1. Parity gate failure beyond tolerance
2. API break
3. Silent behavior change
4. tf.while_loop compatibility blocker
5. XLA incompatibility introduced
6. Memory regression > 2×
7. Performance regression > 2×
8. Fundamental architectural problem

**For each stop condition**, the program specifies:
- **DO**: Document failure mode, record phase, preserve artifacts
- **DO NOT**: Attempt speculative fixes, change plan, downgrade requirements
- **REPORT**: Failure mode, evidence, phase state, request user direction

**Contrast with Autonomous Repair** (Phase Repair Policy):
- Trivial bugs: fix
- Numerical discrepancies within tolerance: investigate and fix
- Minor API mismatches (no caller breaks): fix
- Documentation drift: fix
- Dead code: remove
- Lint/format: fix

**Audit Question 6**: Are stop conditions distinct from autonomous-repair conditions? Would an agent be able to tell the difference?

---

## Technical Soundness

### Performance Root Cause

**Measured** (docs/benchmarks/diagnose_*.py, 2026-08-30):
- Python unrolling: ~2,200 nodes/timestep × 50 timesteps = 110,628 nodes per LEDH call
- Surrogate-force HMC: 1 value + 5 swept directions = 6 calls = 663,766 nodes
- Swept form: 6.00× graph, 1.23× arithmetic (CSE working)
- Naive batched: 2.00× graph, 3.51× arithmetic (CSE defeated)

**Interpretation**: CSE works for swept calls (deduplicates primal) but fails for batched (tiling breaks structural identity)

**Audit Question 7**: Is the performance diagnosis sound? Is the 6× graph-size problem real?

### Proposed Solution

**tf.while_loop with multi-direction tangent**:
- One loop body (~2,200 nodes), maximum_iterations = horizon × substeps
- Tangent state [m, K, dim], explicit forward-mode JVP
- Full gradient in one loop body → work-sharing structural
- Target: O(10³) nodes vs O(10⁶)

**Technical Risks** (Master Program, Risk Register):
- Loop body size limit (Low likelihood, mitigation: TF supports large bodies)
- Multi-direction tangent defeats CSE differently (Medium, mitigation: early Phase 2 diagnostic)
- Parity fails due to op-order (Medium, mitigation: rtol 5e-4 tolerance)
- Trace time still high (Low, mitigation: tf.while_loop is well-optimized)
- Warm eval slower (Low, mitigation: loop traced once)

**Audit Question 8**: Is the tf.while_loop approach technically sound? Are there any obvious blockers not in the risk register?

### XLA Compatibility

**Current kernel**: all ops are XLA-compatible (det/inverse via QR because MatrixDeterminant/MatrixInverse lack tf2xla kernels, recorded in code)

**Refactor commitment**: preserve XLA compatibility (Stop Condition 5 fires if an op without tf2xla kernel is introduced)

**Audit Question 9**: Is XLA compatibility adequately protected?

---

## Phase Structure Coherence

### Phase 0 → Phase 1 Dependency

**Phase 0 Deliverables**:
- Parity tests pass (3/3 fused + 3/3 non-fused)
- Coverage baseline measured
- Refactor contract written and accepted

**Phase 1 Pre-Conditions** (Phase 0 Subplan):
1. Parity gates functional
2. Coverage baseline established
3. Refactor contract accepted (A2)
4. Allowlist accepted (A3)
5. Repair policy accepted (A4)
6. Execution model accepted (A5)

**Phase 1 Objective** (Master Program):
- Convert unrolled loops to tf.while_loop with one-direction tangent
- Preserve all intermediate state
- Parity gates must still pass

**Audit Question 10**: Does Phase 0 adequately prepare Phase 1? Are the pre-conditions sufficient?

### Phase 1 → Phase 2 → Phase 3 → Phase 4 Flow

**Phase 1**: Single-direction tangent (establish tf.while_loop structure)  
**Phase 2**: Multi-direction tangent (full gradient in one body)  
**Phase 3**: Closure hoisting, constant precomputation (minimize loop state)  
**Phase 4**: Integration, leaderboard validation, surrogate-force HMC smoke test

**Audit Question 11**: Is the phase progression logical? Does each phase build on the previous in a minimal-increment way?

---

## Missing Elements

**Phase 1-4 Subplans**: Not yet written (only Phase 0 subplan exists)

**Rationale**: User's requirement R3 "write subplans for phases" is partially complete. Phase 1-4 subplans should be written before user approval (A1-A5) so the user can review the complete program.

**Audit Question 12**: Should Phase 1-4 subplans be written now (before user approval), or is it acceptable to write them after Phase 0 repair completes and Phase 1 begins?

---

## Specific Audit Questions for Codex

1. Are all 10 user requirements (R1-R10) addressed? If any are missing/incomplete, specify.
2. Does the program eliminate mid-execution choice questions?
3. Does the program prevent mid-execution plan changes?
4. Is the repair-and-refresh step mandatory and well-defined?
5. Is the allowlist sufficient for autonomous Phase 0-4 execution?
6. Are stop conditions distinct from autonomous-repair conditions?
7. Is the performance diagnosis (6× graph size, CSE behavior) sound?
8. Is the tf.while_loop approach technically sound?
9. Is XLA compatibility adequately protected?
10. Does Phase 0 adequately prepare Phase 1?
11. Is the phase progression (Phase 1 → Phase 2 → Phase 3 → Phase 4) logical?
12. Should Phase 1-4 subplans be written before user approval or after Phase 0 repair?

---

## Audit Deliverable Request

Please provide:
1. **Compliance Summary**: For each user requirement R1-R10, state COMPLIANT / NON-COMPLIANT / PARTIAL with brief justification
2. **Hard Constraint Assessment**: For R7, R8, R9, state SATISFIED / NOT SATISFIED with evidence
3. **Technical Soundness**: For the tf.while_loop approach, state VIABLE / QUESTIONABLE / BLOCKED with reasoning
4. **Gap Analysis**: List any missing elements, ambiguities, or insufficiencies in the program documents
5. **Recommendation**: APPROVE / REVISE / REJECT with specific required changes if REVISE

**Audit Standard**: The user's directive is "ensure that all my requirements are satisfied." Apply that standard strictly.

---

## Document Inventory

**For Audit**:
1. [Master Program](ledh-while-loop-refactor-master-program-2026-08-30.md) — 23 sections, ~18 KB
2. [Phase 0 Subplan](ledh-while-loop-refactor-phase0-subplan-2026-08-30.md) — 13 sections, ~13 KB
3. This memo — 6 sections, ~6 KB

**Not Yet Written**:
4. Phase 1 Subplan (single-direction tf.while_loop conversion)
5. Phase 2 Subplan (multi-direction tangent generalization)
6. Phase 3 Subplan (closure hoisting, constant precomputation)
7. Phase 4 Subplan (integration, leaderboard validation)

**Supporting Evidence** (existing):
- `docs/benchmarks/diagnose_graph_size_20260830.py` (Case A/B/C measurements)
- `docs/benchmarks/diagnose_eval_time_20260830.py` (trace + warm eval times)
- `docs/benchmarks/diagnose_direction_cost_scaling_20260830.py` (CSE effectiveness)
- `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md` (retraction + next action)
- `docs/benchmarks/step1_true_surrogate_force.py` (surrogate-force HMC driver, 6-call pattern)

---

**Audit Request Submitted**: 2026-08-30  
**Author**: Claude (Opus 5)  
**Reviewer**: Codex (Independent)  

---

**END OF AUDIT MEMO**
