# NeuTra Unfinished-Lanes Reset Memo, Revised Scope (2026-08-17)

## State

The active closeout scope is limited to Banana, KSC-UKF, and German credit.
The prior executed result is retained as historical provenance and is not an
active promotion record for this scope.

## Frozen Evidence

- Banana result: `docs/plans/artifacts/neutra-unfinished-closeout-banana-2026-08-17/result.json`
- KSC attempted root: `docs/plans/artifacts/neutra-unfinished-closeout-ksc-2026-08-17/`
- German repair result: `docs/plans/artifacts/neutra-unfinished-closeout-german-2026-08-17/result.json`
- German support result: `docs/plans/artifacts/neutra-unfinished-closeout-german-proposal-2026-08-17/result.json`
- Revised active plan: `docs/plans/bayesfilter-neutra-algorithm-full-validation-plan-2026-08-17.md`

## Operational Constraints

- Do not launch German HMC from the failed proposal audit.
- Do not infer the missing KSC candidate from a public summary.
- Do not promote Banana feature ratios to a posterior equality claim.
- GPU NeuTra runs must set and verify memory growth before TensorFlow device
  initialization and use XLA by default.
- `L=1` remains forbidden for HMC; use the shared sequential controller.

## Next Entry Points

1. Complete the KSC broad-grid handoff, then run sequential HMC.
2. Design a fresh German support-repair hypothesis and retain the failed audit
   as holdout evidence.
3. Execute the broader algorithm-validation ladder in the active full-
   validation plan before making any default-readiness claim.
