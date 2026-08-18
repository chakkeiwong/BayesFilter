# Fable R2 Verdict: Revised BayesFilter Tuning Plan

Date: 2026-08-17
Review target:
`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
Request:
`docs/plans/bayesfilter-tuning-fable-coverage-audit-r2-request-2026-08-17.md`
Mode: bounded read-only review; no edits, commands, agents, or repository-wide
review

## Findings

### Seven repairs

1. **Run-level baselines and four-entry ledger: accepted.** Phase 0 makes
   focused execution mandatory, records 11 MacroFinance and 2 dsge_hmc
   failures, and separates them into four technical entries. The counts are
   internally consistent: `8 + 2 + 1 = 11`, `90 + 11 = 101`, and
   `24 + 2 = 26`. Each entry requires cited producer/consumer authorities,
   classification, a named repair phase, and a rerun record.
2. **Adjudicated repairs before Phases 4-6: accepted.** The plan explicitly
   states that preserving 90/11 or 24/2 cannot close a migration gate. Allowing
   an unresolved contract choice during structural extraction while blocking
   the affected migration gate is fail-closed and not a defect.
3. **Unfiltered mass commands: accepted.** The two dedicated mass modules run
   without the broader command's `-k` filter. The filtered command remains
   mechanics evidence and cannot close the five-part mass contract.
4. **Mismatched-SPD arm: accepted.** The plan requires a valid but
   geometrically mismatched SPD arm, arm-specific fresh seeds and epsilon/`L`
   retuning, target-preserving holdout validity as primary, and
   uncertainty-gated promotion language.
5. **NumPy mass debt: accepted.** The plan requires a TensorFlow/TFP canonical
   mass path in Phase 2 and restricts NumPy to an independent diagnostic or
   reference role outside both canonical routes.
6. **Non-Gaussian stress evidence: accepted.** Curved/varying-Hessian evidence,
   or an owner waiver with nonclaims, is required for numeric-default
   promotion and is correctly kept separate from the Phase 0-2 structural
   extraction gate.
7. **Staged/conditional mass, epsilon, and `L`: accepted.** The plan correctly
   distinguishes one-epsilon-then-`L` from per-`L` dual averaging, requires the
   policy in artifact identity, and invalidates prior epsilon/`L` evidence after
   a mass change.

### F1 rebuttal

Fable agrees with the rebuttal as operationalized. The four-entry ledger is
more precise than treating all 13 failures as three uniform BayesFilter drift
families. Entry 3 is explicitly not a timeout-policy drift. Requiring owner
direction only when inspected producer and consumer authorities leave a
material intended-contract choice unresolved is the correct boundary; routine
repair of a demonstrably brittle test does not need blanket owner sign-off.

Fable did not inspect the failing consumer files in this bounded R2 review, so
it classified the factual claim that the `0.25` assertion is brittle as not
independently checked in R2. It nevertheless found the prescribed structural
redaction assertion safe because it preserves the intended privacy check even
if the original failure classification is later revised.

### F3 rebuttal

Fable agrees that a valid fixed SPD mass does not by itself change the invariant
target of correctly MH-adjusted HMC. The mass affects efficiency and stable
step size. Requiring repair solely from mass-to-covariance distance would
incorrectly promote an efficiency diagnostic into a validity veto. Whether
fresh epsilon/`L` tuning compensates is empirical, and the plan correctly uses
the qualified term `may compensate` while requiring independent holdout
validity.

The exact repair condition remains an evidence-contract implementation item.
That is acceptable because the plan requires it to be predeclared rather than
silently defaulted.

## Blocking Defects

None found. Counts, ledger entries, gate wording, checklist items, and both
Codex rebuttals are internally consistent.

## Open Work

Fable classified these as open fail-closed implementation work rather than plan
defects:

- implement and run the posterior-oracle fixture and its predeclared repair,
  tolerance, and MCSE policies;
- add the missing MacroFinance robust-broad-grid behavior test;
- adjudicate the intended timeout identity and dsge_hmc grid-policy contracts;
- resolve or waive the MacroFinance full-suite collection blockers; and
- continue to omit, and explicitly report, the dsge_hmc archive path while its
  module-level segfault remains unresolved.

## Codex Verification Note

Fable's raw open-work list said that the two dedicated mass test files do not
yet exist. That statement is wrong: both
`tests/test_hmc_mass_matrix.py` and
`tests/test_hmc_windowed_mass_adaptation.py` exist in this checkout, and their
combined CPU-hidden `tfgpu` run passed 25 tests. This does not alter Fable's
acceptance of the command or the plan.

No posterior-correctness, convergence, superiority, GPU, production-readiness,
scientific-validity, or default-readiness conclusion is supported by this
review.

`AUDIT_VERDICT: AGREE`

`PLAN_VERDICT: AGREE`
