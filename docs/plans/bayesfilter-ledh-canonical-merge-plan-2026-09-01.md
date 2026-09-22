# LEDH Canonical Merge Plan: worktree-ledh-canonical-rebuild → main

**Date**: 2026-09-01  
**Authority**: User directive 2026-09-01 — "one authority of the LEDH algo"  
**Branch state**: worktree HEAD `39f0b20e` (2026-08-29), main HEAD `b230c986` (2026-09-01)  
**Status**: AWAITING_AUDIT

---

## Executive Summary

This plan merges the LEDH canonical Q-program (72 commits, 12 implementation modules, 18 test files) from `worktree-ledh-canonical-rebuild` into `main` to establish **one claim-bearing LEDH implementation**. The worktree holds the algorithm rebuilt per the August 2024 canonical contract: UKF per-particle covariance lifecycle, analytical recursive score, conformance-gated single-cloud and batch lanes. Main currently has the pre-rebuild RQMC/GenUT path, which the rebuild plan (P7) marked for deletion at completion.

**Merge is BLOCKED** pending two resolutions:
1. **Worktree test failures**: Phase 0 audit (2026-08-30) found 21 failures, 8 errors in the canonical test suite due to API drift (`flow_substeps` vs `substeps`) and missing fixtures. Merging known-broken tests into main violates the fail-closed policy.
2. **In-flight refactor**: `ledh-while-loop-refactor-master-program-2026-08-30.md` (status: AWAITING_OWNER_APPROVAL) proposes tf.while_loop conversion of the batch-fused lane for surrogate-force HMC. Merging mid-refactor would orphan that work or require re-baselining.

**Proposal**: Repair Phase 0 findings first, complete or shelve the while-loop refactor, then merge. The additive nature of the work (12 modules absent on main) means the technical merge is low-risk once the worktree is self-consistent.

---

## Motivation

Main currently violates the rebuild plan's "One Algorithm, Two Lanes, Zero Forks" principle. Two implementations exist:

| Aspect | main (pre-rebuild) | worktree (canonical) |
|--------|-------------------|---------------------|
| Covariance | Shared/identity placeholder | Per-particle UKF lifecycle |
| Flow | LEDH-PF-PF | LEDH-PF-PF |
| Score | Standard backward marks | Analytical recursive through UKF/flow/log-det/reset/correction |
| Conformance | Scattered | Contract registry + 18 dedicated test files |
| Claim-bearing | No (bootstrap era) | Yes (Q1-Q3 complete, conformance-stamped) |

The worktree is the intended single authority. Main's `ledh_pfpf_genut_initial_rqmc_tf.py` predates the rebuild and uses the shared-covariance approximation the canonical contract explicitly prohibits.

---

## Scope

### In Scope (Merge)

**12 implementation modules** (additive, no conflicts):
- `ledh_alg1_contract.py` — Algorithm 1 step registry and conformance contract
- `ledh_canonical_filter_tf.py` — Single-cloud value lane (semantic authority)
- `ledh_canonical_score_tf.py` — Single-cloud analytical score
- `ledh_canonical_batch_tf.py` — Batch value/score lane
- `ledh_canonical_batch_fused_tf.py` — Fused batch lane (NeuTra-eligible)
- `ledh_canonical_autodiff_oracle_tf.py` — Forward-mode autodiff oracle (gate only)
- `ledh_canonical_reset_score_tf.py` — Reset-stage score implementation
- `ledh_canonical_score_stages_tf.py` — Staged score helpers
- `ledh_canonical_models_tf.py` — Model callbacks (Austria, LGSSM, PP, KSC, SV)
- `ledh_canonical_neutra_targets_tf.py` — NeuTra target factory
- `ledh_flow_perparticle_tf.py` — Per-particle dual-state flow
- `ledh_ukf_lifecycle_tf.py` — UKF predict/update with per-particle covariance

**18 test files** (additive, no conflicts):
- `test_ledh_alg1_contract.py` (if exists, not in prior listing)
- `test_ledh_canonical_filter.py`
- `test_ledh_canonical_score_*.py` (6 files: full, recursion, stages, step, ukf_tangent, flow_perparticle)
- `test_ledh_canonical_batch.py`, `test_ledh_canonical_batch_fused.py`
- `test_ledh_canonical_governance.py`, `test_ledh_canonical_meta_governance.py`
- `test_ledh_canonical_model_fidelity.py`, `test_ledh_canonical_models.py`
- `test_ledh_canonical_neutra_target.py`
- `test_ledh_canonical_fisher_identity.py`
- `test_ledh_ukf_lifecycle.py`
- `vendored_reference_batch_adapters.py`

**4 fidelity fixes** (cherry-pick):
- `66afacd5` — Consecutive-seed pseudo-replication fix (SeedSequence hashing, independence gate)
- Infidelity #4 (Austria RK4 half-step), #5 (KSC mixture density), #6 (bootstrap comparator resampling)

**Class-B guards**:
- Nonfinite lifecycle/predicted-state veto (fail-closed)
- Runner memory-growth ordering fix (`b1db2962`)
- (PSD floor guard already on main from 2026-09-01 port)

**Documentation**:
- `docs/requests/ledh_score_discrepancy_audit_response_2026_08_29.md` (currently untracked, must commit before merge)
- Completion program, execution plans, Q2 calibration campaign, Q3 leaderboard report

### Out of Scope (Deferred or Separate)

**Retirement of pre-rebuild lane** (P7 deletion, owner-gated):
- `cubature_genut_batch_tf.py`, `cubature_genut_batch_adapters.py`, `cubature_genut_neutra_targets.py`
- 18 live consumers on main (4 test files, 14 benchmark/tuning scripts)
- Handled separately after merge to avoid forcing the deletion decision prematurely

**Main-side additions** (not conflicts, stay on main):
- C2/TT-engine work: `squared_tt_engine_*.py`, `retained_moments_tf.py`
- `neutra_global_mixing.py`
- `sqmc_tuning_scope.py`, `ledh_pfpf_genut_initialization_tf.py` (just ported 2026-09-01)

**While-loop refactor** (in-flight, approval pending):
- Phase 0-4 plan to convert `ledh_canonical_batch_fused_tf.py` to tf.while_loop for surrogate-force HMC
- Either complete pre-merge or rebase post-merge

---

## Pre-Merge Gates (BLOCKING)

### G0: Worktree Test Suite Repair (REQUIRED)

**Finding**: Phase 0 audit (2026-08-30) reports 21 failures + 8 errors in canonical test suite.

**Root causes**:
1. **API drift** (`flow_substeps` vs `substeps`): Single-cloud authority uses `flow_substeps`, batch lanes use `substeps`. Tests call batch lanes with `flow_substeps=`, causing `TypeError`. Affects 6 tests.
2. **Missing fixtures**: phase5/phase1 JSON fixtures absent, causing 8 collection errors.
3. **Other failures**: 15 additional test failures (mechanism TBD).

**Gate criterion**: All canonical tests (`-k canonical`) must pass or be explicitly marked xfail with recorded rationale before merge.

**Repair options**:
- **Option A**: Apply Phase 0 repairs from the refactor subplan (API unification, fixture regeneration).
- **Option B**: Fix only the 6 API-drift tests (minimal), defer fixture/other repairs as post-merge work with xfail markers.

**Blocker rationale**: Merging 21 known-broken tests into main's CI contradicts the fail-closed policy. The canonical suite is the contract's enforcement mechanism; if it doesn't pass, the algorithm's claim-bearing status is unverified.

### G1: Score-Lane Registry Resolution (REQUIRED)

**Finding**: `ledh_alg1_contract.py` ENTRY_POINTS registers `canonical_value_score_and_diagnostics`, which does not exist. The module exports `canonical_value_and_analytical_score`. Discovered 2026-09-01 during synchronization audit.

**Impact**: Any code resolving score entry points through the registry gets an AttributeError. The score-discrepancy audit response flags this as the first blocker to investigation resume.

**Gate criterion**: One-line fix applied, wiring test added (resolve all ENTRY_POINTS, assert callable exists).

**Repair**: Trivial (change 1 string), but must be done before merge since the registry is claim-bearing infrastructure.

### G2: Commit Untracked Work (REQUIRED)

**Finding**: 32 untracked files on worktree, including:
- `docs/requests/ledh_score_discrepancy_audit_response_2026_08_29.md` (the resume-point document)
- 3 while-loop refactor plans dated 2026-08-30
- 15+ investigation scripts, epsilon-schedule experiments

**Gate criterion**: Audit response and any keeper investigation artifacts committed to worktree branch before merge. Transient/exploratory scripts can stay untracked if documented as such.

**Rationale**: Losing the audit response during merge would orphan the investigation. Git tracks the resume state, not the filesystem.

### G3: While-Loop Refactor Decision (REQUIRED)

**Finding**: `ledh-while-loop-refactor-master-program-2026-08-30.md` status AWAITING_OWNER_APPROVAL. Targets `ledh_canonical_batch_fused_tf.py` (one of the 12 modules this merge brings in).

**Options**:
- **A. Complete first**: Execute Phase 0-4, land the while-loop changes on worktree, then merge.
- **B. Rebase post-merge**: Merge canonical as-is, rebase the refactor onto main afterward.
- **C. Shelve**: Defer the refactor to a future session, merge canonical now.

**Blocker rationale**: Merging mid-refactor orphans the in-flight work and forces re-baselining. The refactor's audit is scoped to the worktree's state; merging changes that scope.

---

## Merge Execution (Post-Gate)

### Stage A: Additive Merge (12 modules + 18 tests)

**Method**: `git merge --no-commit worktree-ledh-canonical-rebuild`, resolve conflicts, inspect diff.

**Expected conflicts**:
1. `CLAUDE.md`, `AGENTS.md` — editorial (both branches added rules)
2. `higher_moment_contract_e.py` — floor defaults (`1.0e-5` worktree vs `1.0e-6` main for `pairwise_floor`)
3. Possibly `bayesfilter/highdim/__init__.py` (exports)

**Resolution**:
- Docs: three-way manual merge, keep both rule additions
- Floor defaults: **Owner decision required** (Class C numerics change). Worktree's `1.0e-5` matches the PSD guard ported today; main's `1.0e-6` predates that. Recommend worktree value (aligned with relative-ridge derivation).
- Exports: add canonical modules to `__all__`

**Post-merge verification**:
- `git diff --cached` review
- `pytest tests/highdim/test_ledh_canonical_*.py -v` (canonical suite green)
- `pytest tests/highdim/test_higher_moment_contract_e.py` (PSD guard regression check)

### Stage B: Fidelity Fixes (Cherry-Pick)

**Commits**:
- `66afacd5` (seed pseudo-replication)
- Austria RK4, KSC density, bootstrap comparator fixes (3 commits, TBD SHAs)

**Method**: `git cherry-pick -x <SHA>`, resolve conflicts if any.

**Verification**: Run affected tests after each pick.

### Stage C: Documentation + Untracked Commit

**Actions**:
1. Commit audit response: `git add docs/requests/ledh_score_discrepancy_audit_response_2026_08_29.md`
2. Commit keeper investigation artifacts (owner choice on which to keep)
3. Update memory note: remove "do not merge to synchronize" directive, replace with "canonical LEDH is now on main, worktree archived post-merge"

### Stage D: Regression + CI

**Test surface**:
- Full canonical suite: `pytest tests/highdim/test_ledh_canonical_*.py`
- PSD guard regression: `pytest tests/highdim/test_higher_moment_contract_e.py tests/highdim/test_genut_*.py`
- RQMC smoke: `python docs/benchmarks/test_rqmc_trust_region_smoke.py`
- (Optional) Broader regression: `pytest tests/highdim/ -k "not zhao_cui" --ignore=tests/highdim/test_c2_*` (CPU, ~3 min)

**Success criterion**: All tests pass or are explicitly xfail-marked.

---

## Audit Scope

### Track 1: LEDH Contract Faithfulness (LEDH-Responsible Agent)

**Questions**:
1. Do the 12 canonical modules implement Algorithm 1 per the contract registry?
2. Does the analytical score differentiate the complete executed map (UKF + flow + log-det + reset + correction)?
3. Are the conformance gates (C-1 through C-9) sufficient to enforce the contract?
4. Does `ledh_canonical_batch_tf.py` preserve the pairwise and coordinate-cap capabilities that `ff074fa2` added to the pre-rebuild batch lane?
5. Are the fidelity fixes (seed pseudo-replication, Austria RK4, KSC density, bootstrap resampling) scientifically sound?
6. Are there any derivation errors, missing tangent terms, or incorrect chain-rule applications in the score recursion?

**Artifacts to review**:
- 12 implementation files
- 18 test files
- Contract registry (`ledh_alg1_contract.py`)
- Completion program (`bayesfilter-ledh-canonical-completion-program-2026-08-24.md`)
- Q2 calibration artifacts, Q3 leaderboard report

**Decision output**: APPROVE (merge) / REVISE (gate failures enumerated) / REJECT (fundamental flaw)

### Track 2: Main-Branch Impact Assessment (Independent Reviewer)

**Questions**:
1. Does merging canonical break any main-side consumers (RQMC, C2/TT-engine, existing benchmarks)?
2. Is the P7 deletion (retiring `cubature_genut_batch_tf.py` and 18 consumers) scientifically justified, or does it discard validated work?
3. Are there capability regressions — features main has that canonical loses?
4. Is the merge timing safe given the in-flight while-loop refactor and 21 test failures on worktree?
5. Should the floor-default divergence (`1.0e-5` vs `1.0e-6`) be resolved as a Class-C numerics decision before merge?

**Conflict-of-interest note**: The LEDH agent that built the canonical implementation cannot neutrally assess whether it should replace main. Track 2 provides independent validation.

**Artifacts to review**:
- This plan
- Diff summary: `git diff --stat main...worktree-ledh-canonical-rebuild`
- Consumer inventory (files importing deleted modules)
- Phase 0 audit findings (test failure breakdown)

**Decision output**: APPROVE / CONDITIONAL (gates or sequencing changes) / DEFER (repair worktree first)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|---------|-----------|
| Canonical test failures propagate to main CI | High (21 failures known) | High (CI red) | **G0 gate**: repair before merge |
| While-loop refactor orphaned by merge | Medium | Medium (rework) | **G3 gate**: complete or shelve first |
| Capability regression (pairwise/coordinate-cap) | Low (needs verification) | High (breaks consumers) | **Track 1 Q4**: parity check canonical batch vs ff074fa2 |
| Floor-default divergence causes numerical drift | Low | Medium (tuning artifacts invalidated) | Owner decision pre-merge; document in commit message |
| Audit response lost during merge | Low (if G2 gate enforced) | High (investigation orphaned) | **G2 gate**: commit untracked work first |

---

## Success Criteria

1. ✅ All pre-merge gates (G0-G3) green
2. ✅ Audit Track 1: APPROVE
3. ✅ Audit Track 2: APPROVE or CONDITIONAL with mitigations applied
4. ✅ Canonical test suite passes on main post-merge
5. ✅ RQMC smoke test green (trust-region wiring intact)
6. ✅ No regressions in PSD guard or higher-moment test surface

**Post-merge state**: One claim-bearing LEDH implementation on main (canonical). Pre-rebuild modules remain temporarily for backward compat; P7 deletion handled separately.

---

## Out-of-Scope (Explicitly Excluded)

1. **P7 deletion execution**: Retiring the pre-rebuild lane is owner-gated and deferred.
2. **Posterior correctness**: Merge establishes algorithmic correctness per contract; posterior validation is separate.
3. **HMC readiness**: While-loop refactor (if completed) enables surrogate-force HMC, but that's not a merge criterion.
4. **LGSSM score discrepancy resolution**: Audit response defines the investigation resume point; resolution is post-merge work.
5. **Tuning re-execution**: Existing tuning artifacts remain valid (same route, same controls). Fresh tuning is post-merge.

---

## Next Steps

1. **Owner approval on gates**: Confirm G0-G3 sequencing (repair first vs conditional merge).
2. **Audit assignment**: Designate Track 1 (LEDH agent) and Track 2 (independent) reviewers.
3. **G0 repair decision**: Apply Phase 0 fixes (Option A) or minimal API-drift fix (Option B)?
4. **G3 refactor decision**: Complete, rebase, or shelve the while-loop work?
5. **Floor-default decision**: `1.0e-5` (worktree/PSD-aligned) or `1.0e-6` (main/historical)?

**Execution hold**: No merge action until audit complete and all gates green.
