# Reset Memo: LEDH Score Wiring Repair

Date: 2026-07-10

## Current State

The active program is:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-master-program-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-gated-execution-runbook-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-execution-ledger-2026-07-10.md`

Current status: Phase 5 actual-SV local checks passed. The next required gate is
review of the Phase 5 result and Phase 6 generalized-SV subplan. Do not start
Phase 6 implementation until that review is completed or a recorded substitute
review agrees.

## Completed In This Segment

### Phase 4: Predator-Prey

Status: `PASSED`

Changed files:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase4-result-phase5-subplan-review-bundle-2026-07-10.md`

What changed:

- Predator-prey score defaults now use `float32` and TF32 enabled.
- `_coordinate_fd_score_diagnostic` now uses
  `_compact_value_and_score_from_components` as the score base.
- Finite differences use a value-only same-scalar objective.
- Score artifacts include explicit `score_precision`.
- Full-admission artifact construction rejects nested historical/manual
  relabeling and tiny-shape promotion.
- Historical reverse/manual score routes remain diagnostic-only.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: `70 passed, 2 warnings`.

Review:

- Claude review gate was attempted and rejected by execution policy as external
  repository data disclosure. No workaround was attempted.
- Fresh Codex substitute read-only review returned `VERDICT: AGREE`.

### Phase 5: Actual-SV

Status: `LOCAL_CHECKS_PASSED_REVIEW_PENDING`

Changed files:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase5-result-phase6-subplan-review-bundle-2026-07-10.md`

What changed:

- Actual-SV score defaults now use `float32` and TF32 enabled.
- `_coordinate_fd_score_diagnostic` now uses
  `_compact_value_and_score_from_components` as the score base.
- Finite differences use a value-only same-scalar objective.
- Score artifacts include explicit `score_precision`.
- Full-admission artifact construction rejects nested historical/manual
  relabeling and tiny-shape promotion.
- Historical reverse/manual score routes remain diagnostic-only.
- The transformed actual-SV target policy
  `transformed_actual_sv_log_y_square` is preserved.
- `claims_exact_native_actual_sv_likelihood` remains false.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: `70 passed, 2 warnings`.

Review status:

- Claude review gate for Phase 5 result plus Phase 6 subplan was attempted and
  rejected by execution policy as external repository data disclosure.
- No substitute review has been completed yet for Phase 5. This is the exact
  next step.

## Important Boundaries

- CPU-hidden local tests are wiring evidence only.
- No new trusted GPU `N=10000` score-memory run was performed in this segment.
- No full score admission was claimed for predator-prey or actual-SV.
- No leaderboard rebuild was performed.
- No HMC readiness, posterior correctness, exact native likelihood, or
  scientific-superiority claim was made.
- Claude calls are currently policy-blocked as external data disclosure. Do not
  attempt workarounds. Use a fresh Codex substitute read-only review unless
  policy changes.

## Exact Next Step

Run a fresh Codex substitute read-only review for:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase5-result-phase6-subplan-review-bundle-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `bayesfilter/highdim/ledh_score_contract.py`

If the substitute review returns `VERDICT: AGREE`, update the visible ledger
with a Phase 5 review entry and begin Phase 6. If it returns `VERDICT: REVISE`,
patch the fixable issue, rerun focused checks, and repeat review within the
five-round blocker limit.

## Phase 6 Preview

Phase 6 is generalized-SV. The current finding is that the compact route is
already the score path, but the module still defaults to `float64`/TF32 disabled
and needs the shared production precision/full-admission hardening. Preserve
the source-route prior-mean generalized-SV target; do not substitute KSC or
actual-SV target semantics.
