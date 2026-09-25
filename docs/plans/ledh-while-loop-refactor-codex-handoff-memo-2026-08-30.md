# Handoff to Codex: LEDH While-Loop Refactor Program Review

**To**: Codex (review agent)  
**From**: Opus (authoring agent)  
**Date**: 2026-08-30  
**Subject**: Thorough review of `ledh-while-loop-refactor-2026-08-30` program before execution

---

## Request

Conduct a thorough review of the attached LEDH while-loop refactor program
against the user's ten explicit requirements (R1–R10, verbatim below) and the
two hard constraints. Your review should be adversarial in the sense defined by
the global CLAUDE.md skeptical-audit policy: actively look for wrong baselines,
proxy metrics treated as promotion criteria, missing stop conditions, unfair
comparisons, hidden assumptions, stale context, environment mismatches, and
commands whose artifacts would not answer the stated question.

Report every defect, gap, and risk you find. Do not soften findings. The user
wants problems surfaced now, before execution, when fixing them is cheap.

---

## User requirements (verbatim from the governing directive)

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

Decomposed into ten requirements (R1–R10) with two marked as hard constraints:

1. **R1**: Add Phase 0 — audit existing tests, establish coverage, write refactor contract, BEFORE refactor work
2. **R2**: Master program document with all details and phases
3. **R3**: Subplans for each phase
4. **R4**: All backgrounds (context sufficient for future agents/reviewers)
5. **R5**: Allowlist of tool/command permissions required
6. **R6**: Upfront approval requests for end-to-end execution
7. **R7 (HARD CONSTRAINT)**: No mid-execution choice questions to the user
8. **R8 (HARD CONSTRAINT)**: No mid-execution plan changes when difficulties arise
9. **R9**: Mandatory per-phase repair step: repair what is repairable, refresh plan for next phase
10. **R10**: Memo to Codex auditing the whole program against every user requirement (you are reading it)

The user further clarified: "We want a smooth execution without any clicking in
the middle of execution."

---

## Documents to review

All under `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild/docs/plans/`:

1. `ledh-while-loop-refactor-master-program-2026-08-30.md` (master program)
2. `ledh-while-loop-refactor-phase0-subplan-2026-08-30.md` (Phase 0 subplan)
3. `ledh-while-loop-refactor-phase1-subplan-2026-08-30.md` (Phase 1 subplan)
4. `ledh-while-loop-refactor-phase2-subplan-2026-08-30.md` (Phase 2 subplan)
5. `ledh-while-loop-refactor-phase3-subplan-2026-08-30.md` (Phase 3 subplan)
6. `ledh-while-loop-refactor-phase4-subplan-2026-08-30.md` (Phase 4 subplan)
7. `ledh-while-loop-refactor-codex-audit-memo-2026-08-30.md` (the earlier audit memo, now an input artifact listing R1–R10 with 12 audit questions)

Supporting scripts and diagnostics (evidence the program references):

- `scripts/run_phase_tests.sh` — single-entry wrapper for all test/diagnostic runs
- `docs/benchmarks/diagnose_graph_size_20260830.py` — graph-node baseline measurements
- `docs/benchmarks/diagnose_eval_time_20260830.py` — trace/warm-eval baseline
- `docs/benchmarks/diagnose_direction_cost_scaling_20260830.py` — CSE effectiveness measurement
- `docs/benchmarks/step1_true_surrogate_force.py` — the surrogate-force driver to be updated
- `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md` — λ=δ=0 correction

Implementation and test targets (on the worktree, NOT on main):

- `bayesfilter/highdim/ledh_canonical_score_tf.py` — single-cloud authority (lines 1–100 read; `flow_substeps` parameter at line 68)
- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` — the refactor target (read in full, 443 lines)
- `bayesfilter/highdim/ledh_canonical_batch_tf.py` — non-fused batch lane (95 lines; contains a kernel-side `substeps=` → `flow_substeps=` forwarding bug at line 75)
- `tests/highdim/test_ledh_canonical_batch_fused.py` — fused parity tests (lines 100–170 read; 4 test calls needing `substeps=` repair at lines 119, 136, 143, 161; line 109 is an AUTHORITY call that must stay `flow_substeps=`)
- `tests/highdim/test_ledh_canonical_batch.py` — non-fused parity tests (read in full, 140 lines; 5 test calls needing `substeps=` repair at lines 85, 104, 108, 125, 130; line 80 is an authority call that must stay `flow_substeps=`)

---

## Known defects already found (do not duplicate these)

The authoring agent found four errors in the documents during writing. **These
have not yet been corrected in the documents you will review**, so you will see
them. They are listed here so you do not spend time re-finding what is already
known:

1. **Phase 0 subplan overcounts fused-test repair**: lists 5 substitutions in
   `test_ledh_canonical_batch_fused.py` including line 111, but line 111 is an
   authority call that takes `flow_substeps=` correctly. Actual repair: 4
   substitutions at lines 119, 136, 143, 161.

2. **Phase 0 subplan undercounts non-fused-test repair**: lists 1 substitution
   at line 83, but reading the file showed 5 needed at lines 85, 104, 108, 125,
   130, and that line 80 (not 83) is an authority call that must stay
   `flow_substeps=`.

   **Corrected total: 10 substitutions across 3 files** (4 fused test + 5
   non-fused test + 1 kernel), not "7 defects."

3. **Master program API contract misstates the fused lane's score shape**: says
   "theta [B, P] → value [B], score [B, P]". The fused kernel returns `d_total`
   of shape `[B]` (one directional derivative per row, not P). `[B, P]`
   describes the non-fused lane. Phase 2's purpose is turning the `[B]`
   single-direction return into `[B, K]`.

4. **Master program success criterion 5 requires a measurement the test lane
   cannot make**: it requires "device memory usage at (B=6, horizon=50, N=252)
   ≤ 2× current". The test lane is CPU-only float64 by design
   (`CUDA_VISIBLE_DEVICES=-1` in every test and in the wrapper). GPU
   device-memory validation needs an escalated GPU run, which would trigger a
   mid-execution permission prompt — forbidden by R7. Criterion 5 should use
   host RSS / graph nodes / GraphDef bytes with GPU validation deferred as an
   explicitly recorded open item.

These will be corrected after your review, so the user receives accurate documents.

---

## Review focus areas

### Hard-constraint compliance (R7, R8)

**R7**: No mid-execution choice questions. Check every phase for:
- Branching on test results where both branches require user choice
- "Option A vs Option B" decision points that are not resolved by measurement
- Ambiguous pass conditions that might force asking which metric matters
- Missing tie-breaking rules in verification steps

**R8**: No mid-execution plan changes when difficulties arise. Check every phase for:
- Repair scopes that are too narrow (forcing a plan revision on the first obstacle)
- Stop conditions that are too aggressive (triggering on recoverable failures)
- Missing fallback branches for known risks
- Verification steps whose likely failure modes have no mapped repair

Both R7 and R8 are somewhat in tension with good engineering: you want to be
able to ask clarifying questions and respond to evidence. The user's constraint
is that *during execution of an approved program*, these must not happen. The
resolution is in the repair-scope / stop-condition split: if something is
repairable without asking, it is in the repair scope and execution continues;
if it is not, it is a stop condition and execution halts with a result note.
Check whether that split is drawn correctly in each phase.

### Coverage and contract (R1)

Phase 0 must establish what "good test coverage" means for this kernel and
measure the baseline before any refactor work. Does the Phase 0 subplan:
- Define a coverage metric and threshold?
- Measure baseline coverage before repair?
- Have pytest-cov installed or a plan to install it?
- Specify what the refactor contract is, and where it is recorded?

The authoring agent found that `pytest-cov` is NOT installed and that the
environment might not be `tftwogpu` (base interpreter instead). Is the install
step in the plan?

### Backgrounds (R4)

Does the master program carry enough context that a future agent with no
conversation history could execute Phases 1–4? Check for:
- Assumed knowledge not recorded in the program
- References to "the prior approach" without saying what it was
- Verification commands that depend on unwritten scripts
- Diagnostic baselines referenced but not recorded

### Allowlist and upfront approvals (R5, R6)

The allowlist entry is a single wrapper script `scripts/run_phase_tests.sh`.
Check:
- Does the wrapper actually exist?
- Does it cover all the commands the phases will run?
- Are the upfront approvals (A1–A5) specific enough that granting them once
  covers the whole program?
- Do any phase steps require a permission the wrapper does not cover?

The authoring agent noted that the allowlist JSON uses an invented schema and
needs rewriting against the real Claude Code settings format. Is that flagged?

### Phase repair policy (R9)

Each phase must have:
- A defined repair scope (what is repairable without asking)
- Stop conditions (what forces a halt)
- A refresh step before the next phase

Check whether the repair scopes are realistic. A repair scope of "trivial typos
only" on a phase that converts 300 lines of loops is not credible.

### Verification realism

For each phase, check:
- Are the pass conditions testable by the declared commands?
- Are the commands realistic for a CPU-only test lane?
- Are the rtol thresholds reasonable given float64 arithmetic?
- Do any verification steps silently require GPU, network, or external resources?

### Mathematical and implementation correctness

This is not a paper review, but spot-check:
- Does the Phase 2 design actually implement what it claims (multi-direction tangent)?
- Does the Phase 1 loop conversion preserve loop-carried state correctly?
- Are there any obvious rank or shape mismatches in the described transformations?
- Does the QR-based XLA det/trace workaround (mentioned in Phase 1) have a
  recorded rationale, and does the plan preserve it?

### Stale context and wrong baselines

The program references several diagnostics from `docs/benchmarks/`. Check:
- Are those files actually on the worktree?
- Do the baseline numbers in the program match what those files would produce?
- Does the program compare against the right thing (e.g., comparing the
  refactored fused lane against the non-refactored fused lane, not against the
  non-fused lane)?

The authoring agent noted that main and the worktree are disjoint tracks (only
3 files overlap). Does the program make any assumption that breaks if the
worktree has diverged?

### Governance proportionality

The global CLAUDE.md "Academic Research Governance And Proportionality" section
was adopted 2026-08-20 and removes legacy ceremony. Check whether the program:
- Over-specifies approval mechanics (hash-bound statements, one-use tokens, etc.)
- Requires mandatory review of every subplan beyond the one Codex review happening now
- Treats retry under the same campaign budget as requiring renewed approval

The user's request predates the proportionality policy, so there may be
ceremony inherited from older practice. If you find it, flag it as a governance
defect under current policy.

---

## Deliverable format

Produce a review note under `docs/plans/` with:

1. **Compliance table**: R1–R10, each marked SATISFIED / DEFECT / AT_RISK with
   one-line reasoning
2. **Defect register**: every problem found, with severity (BLOCKING /
   MAJOR / MINOR), location (which document, which section), and recommended fix
3. **Risk register**: things that are not defects but could go wrong, with
   likelihood and mitigation
4. **Hard-constraint analysis**: R7 and R8 specifically, with every potential
   mid-execution interruption point identified
5. **Readiness verdict**: APPROVE / APPROVE_WITH_CORRECTIONS / REJECT, with reasoning

"APPROVE" means the program can be executed as written after correcting the four
known errors listed above. "APPROVE_WITH_CORRECTIONS" means additional changes
are required first. "REJECT" means the program has a structural flaw that cannot
be fixed by editing the documents.

Be direct. The user wants problems found now, not discovered during execution.

---

## Context notes

- **Branch**: the refactor lives on `worktree-ledh-canonical-rebuild`, not main.
  All file paths are relative to that worktree. Main and the worktree are
  disjoint tracks; do not assume main's state.
- **Environment**: `tftwogpu` conda env, but the test lane is CPU-only by design
  (`CUDA_VISIBLE_DEVICES=-1`). GPU validation is explicitly deferred.
- **Backend**: TensorFlow/TFP only; NumPy allowed only in tests and diagnostics
  (CLAUDE.md backend rule).
- **No pfor**: TensorFlow pfor requires prior written approval (CLAUDE.md TF
  policy); this program does not have it and must not introduce pfor.
- **Execution tier**: this is a "serious local research campaign" under the
  CLAUDE.md execution-tier taxonomy. One concise experiment plan (the master
  program) with an evidence contract, total compute budget, versioned output
  root, and stop conditions. Plain-language user approval ("execute the
  program") is sufficient campaign authorization under current policy — no magic
  wording or manifest hash.
- **Prior session**: The user asked for this program in a prior session that ran
  out of context. The current session is continuing that work. This is why the
  summary is unusually detailed.

---

## Reading order

Suggested order for efficient review:

1. Master program — establishes what the whole thing is supposed to do
2. Phase 0 subplan — gates everything else
3. Phases 1–4 subplans in order — each depends on the prior
4. Earlier audit memo — contains the R1–R10 mapping and 12 audit questions; some of those questions are now answered by the subplans
5. Cross-check: do the five subplans collectively satisfy the master program's scope?
6. Cross-check: does the wrapper script cover all the commands?
7. Cross-check: are the upfront approvals sufficient?

---

## What this review is not

- **Not a mathematical proof**: you are checking whether the *plan* is coherent,
  not whether the tangent recursion is correct
- **Not an implementation review**: you are reviewing documents, not code
- **Not a decision on whether to do the refactor**: the user already decided
  that; you are checking whether the *program* is executable as written
- **Not a style review**: focus on R1–R10 compliance, correctness risks, and
  missing pieces

---

## Timeline

The user wants to execute this program once your review returns and any
necessary corrections land. Prioritize blocking and major defects over minor
polish.

---

Thank you. The user and I are waiting on your findings before proceeding.
