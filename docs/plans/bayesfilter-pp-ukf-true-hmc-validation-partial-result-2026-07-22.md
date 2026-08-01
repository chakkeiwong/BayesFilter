# PP-UKF True HMC Partial Result

Date: 2026-07-22

Plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Three warmup prefixes preserved; campaign incomplete | Target/recomposition and per-chunk finite/status/movement checks passed | No candidate veto supported; prior energy veto was wrong | No candidate reached minimum warmup or retained sampling | Continue all candidates under corrected controller | No candidate ranking, posterior convergence, superiority, or default claim |

## Candidate Results

| L | Role | Warmup | Retained | Acceptance | Energy-error vetoes | Target status | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 5 | independently tuned primary | 1,000 | 0 | 0.7500 | 5 explanatory extreme proposals | valid | unevaluated; warmup incomplete |
| 9 | independently tuned primary | 1,000 | 0 | 0.6365 | 19 explanatory extreme proposals | valid | unevaluated; warmup incomplete |
| 12 | inherited coverage from L=13 | 1,000 | 0 | 0.77675 | 11 explanatory extreme proposals | valid | unevaluated; warmup incomplete |

Correction: finite log acceptance below `-1000` is an explanatory extreme-tail
diagnostic, not an energy or divergence veto. Non-finite log acceptance remains
a hard health veto. Native TFP divergence remains unavailable and is not zero.

## Inference Status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | No candidate veto supported | Each prefix had finite target/state/status and movement; the prior energy classification was wrong |
| Statistically supported ranking | None | No retained draws and no uncertainty comparison |
| Descriptive-only differences | Acceptance and energy-veto counts | Descriptive candidate diagnostics only |
| Default readiness | Not evaluated | Seven candidates remain untested |
| Next evidence needed | Durable resume harness, then L=13,14,17,18,19,24,25 | Preserve fresh roots and reconcile without rerunning completed rows |

## Infrastructure Ledger

- `attempt-03`: failed before sampling because memory growth was configured
  after TensorFlow logical-device initialization. This was repaired.
- `attempt-04`: completed and preserved L=5,9,12, then ended without a terminal
  result at the managed process boundary.
- `attempt-05` and `attempt-06`: terminated before first candidate checkpoint;
  no scientific result was written.

Focused checks after the repairs: `9 passed`. The three candidate rows are
preserved in [attempt-04 progress](/home/chakwong/BayesFilter/docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-04/progress.json).
