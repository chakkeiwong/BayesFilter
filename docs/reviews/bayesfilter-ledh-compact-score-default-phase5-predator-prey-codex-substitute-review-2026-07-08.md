# Codex Substitute Review: Phase 5 Predator-Prey Compact Score

Date: 2026-07-08

Status: `AGREE_WITH_LIMITATIONS`

## Reason For Substitute Review

Claude review was attempted but blocked by the local escalation reviewer as external data disclosure risk. Codex did not attempt to route around that block.

A fresh Codex read-only reviewer was spawned, but it did not return within the review wait window. This substitute review records the local read-only checks available before continuing the visible runbook.

## Scope Reviewed

- `docs/reviews/bayesfilter-ledh-compact-score-default-phase5-predator-prey-review-bundle-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-subplan-2026-07-08.md`
- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

## Findings

No material blocker found for closing Phase 5 as a tiny-only compact score gate and starting Phase 6.

Checks supporting this:

- The score artifact provenance is `compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`.
- The score artifact status is `tiny_score_diagnostic_not_admitted`.
- The score parameter order is `['r', 'K', 'a', 's', 'u', 'v']`.
- Static source inspection over compact default symbols found none of:
  - `records.append`
  - `reversed(records)`
  - `_transport_vjp_tf`
  - `GradientTape`
  - `ForwardAccumulator`
  - `stop_gradient`
- The historical `manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot` route still exists only as a diagnostic path and is explicitly tested as distinct from compact provenance.
- Phase 6 subplan names the generalized-SV row, raw-y target policy, active transformed parameter order, and forbids actual-SV/KSC substitution.

## Local Evidence Confirmed

Final focused checks passed:

```text
38 passed, 2 warnings
```

Tiny artifact fields:

```text
score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot
score_admission_status = tiny_score_diagnostic_not_admitted
max_abs_error = 1.4451367178480723e-06
max_rel_error = 7.776625302494132e-09
```

## Limitations

- This is not a Claude review.
- The fresh Codex reviewer had not returned within the wait window when this substitute review was written.
- This review does not prove full-row memory viability, HMC readiness, posterior correctness, source-faithfulness, or scientific superiority.
- A later returned reviewer finding should be handled before further material phase advancement if it identifies a concrete blocker.

## Verdict

VERDICT: AGREE
