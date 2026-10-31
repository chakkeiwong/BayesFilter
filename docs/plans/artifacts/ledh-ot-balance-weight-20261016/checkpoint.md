# Active checkpoint

Superseded handoff: the subsequent mixture integration is complete. Resume
from `../ledh-mixture-guide-20261016/checkpoint.md`; the localization evidence
below remains preserved and does not instruct another localization run.

- Current stage COMPLETE: OT mass and pre-reset concentration localization.
  Owner correction: the identity/global/local density-mixture proposal and
  beta pilot optimizer already exist here and on local main. The recent
  guide experiment used another consumer. Next question: reconcile that
  experiment and manuscript with the existing mixture implementation.
  Evidence and bounded verification: `mixture-route-correction.md`.
  Verification complete: five existing mixture tests pass in the matching
  tftwogpu CPU/FP64 environment. First tf-gpu collection attempt failed on
  a custom-op ABI mismatch; both logs are preserved. No runtime change.
- Branch `ledh-refactor-with-policy-fix`; worktree
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`;
  HEAD `ea1018de1da55ed02a29a8d60f08a6827fce8a42`; preserve existing dirty work.
- Plan: `../../ledh-ot-balance-weight-localization-20261016.md`; skeptical
  audit passed for bounded localization. No default or claim-bearing tuning.
- Checked: shared reset performs Sinkhorn5 + balance32 identical updates;
  the frozen-input comparison identifies additive denominator1e-7 as a
  material cause of the observed mass residual.
- Previous evidence: `../ledh-gaussian-score-error-20261016/result.md`;
  255/280 rows, rejected column-TV case, finite-program derivative checks pass.
- CPU01/02 and floor-reference01 consumed49.276452 CPU seconds. All four
  balancing counts fail the same cloud's mass check (TV2.705e-4 at32,
  2.529e-4 at256); other17 clouds valid. Iteration-only repair rejected.
- CPU02 repairs only the independent checker's float32-rounded constant.
  All density identities agree to1.5e-14; consumer parity passes. Persistent
  first-step weighted variance0.0453 versus exact0.1957 remains unexplained
  by a density bug. The conditional proposal has about0.296 of the correct
  conditional variance in the continuous scalar flow calculation.
- Independent log-domain scaling matches the current transport to1e-15.
  Optional zero denominator perturbation in the SAME shared reset reduces
  maximum frozen-cloud column-TV to4.45e-11 at the same37 updates.
- Repair01: all18 frozen clouds and four fresh-noise full consumers pass
  mass/finite checks and the tested analytical finite-difference checks.
  However,13/17 previously valid clouds fail the predeclared1e-6 state
  non-harm tolerance (maximum4.67e-6). No drop-in/default promotion. The
  default remains1e-7; guards and epsilon1/Sinkhorn5 remain unchanged.
- Focused regression:29 pass, one pre-existing six-versus-seven return-value
  unpacking failure in the covariance-carry test, confirmed in HEAD.
- Terminal checks:29 passed/one known-failure deselected; full initial run
  retains that pre-existing failure.68-page PDF builds without warnings;
  final proof pages visually checked. MathDev direct algebra checks pass,
  but automated label/implementation audits are not fully verified.
- Consumption:161.073 measured CPU wall seconds plus untimed overhead;
  conservatively charge200/300 CPU seconds.100 CPU /400 GPU seconds remain.
  Subsequent route correction used18.22 CPU seconds; charge20 more, for a
  current220/300 CPU seconds and80 CPU /400 GPU seconds remaining.
  No GPU run or active process. See `result.md` and `validation.md`.
- Stage is UNTUNED FP64 CPU/XLA REFERENCE; not production/TF32 or model-score
  improvement evidence. Frozen-cloud/fresh-noise results are not new-data
  holdout evidence. Artifacts:repair01/result.json and reset-tests01.log.
- Next: use the completed existing-mixture call-chain check to define guidance
  composition and a prospective comparison using the intended consumer.
  The mixture local map already uses Q; the concentration and 1e-7 mass
  findings belong to the tested other route. TV alone does not prove score
  harm. Preserve defaults and the distinct reset/admission identities.
