# Fixed-Variant Value/Score Multimodel Campaign Result

Date: 2026-08-04

Status: `PASS_METHOD_A_MANUAL_ANALYTICAL_WITH_BLOCKED_METHOD_B_ROWS`

Anti-drift authority:
`docs/plans/bayesfilter-zhao-cui-fixed-variant-all-model-campaign-control-master-program-2026-08-06.md`

Plan:
`docs/plans/bayesfilter-fixed-variant-value-score-multimodel-plan-2026-08-04.md`

Reset memo:
`docs/plans/bayesfilter-fixed-variant-value-score-multimodel-reset-memo-2026-08-04.md`

Machine-readable result:
`docs/plans/bayesfilter-fixed-variant-value-score-multimodel-result-2026-08-04.json`

## Verdict

The cross-model fixed-variant campaign now has manual / analytical Method A score backends for all five Method A rows, including actual SV. The actual-SV row is now bound to the same active batch TT finite program as the value route, with a same-program manual score backend rather than autodiff. The transformed-SV helper route remains out of scope for this campaign because it is a different scalar family.

## Command actually run

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py
```

## Outcome table

### Method A — single-fitted frozen parent

| Model | Value status | Score status | Same-scalar check | Derivative backend | Value | Score norm |
|---|---|---|---|---|---:|---:|
| LGSSM | VALID | VALID | PASS (`max_abs=1.379701e-08`, FD `1e-5`) | `manual` | -775.8863878339255 | 206.87029671803433 |
| KSC SV | VALID | VALID | PASS (`max_abs=1.481198e-07`, FD `1e-5`) | `manual` | -2319.557779425765 | 72.5768196637414 |
| actual SV | VALID | VALID | PASS (`max_abs=1.952860e-10`, FD `1e-5`) | `manual` | -22.890871021866218 | 1.5231492068453724 |
| predator-prey | VALID | VALID | PASS (`max_abs=2.853935e-09`, FD `1e-5`) | `manual` | -109.08502517043668 | 13.75146979114478 |
| Austria SIR | VALID | VALID | PASS (`max_abs=1.989597e-07`, FD `1e-5`) | `manual` | -682.3480055392415 | 110.86484986266814 |

Interpretation: all Method A rows now satisfy the current campaign rule: the value route stays on the declared fixed-variant scalar and the score backend is manual / analytical for that same finite program. This remains a result about the route’s declared finite-program approximation, not a claim of exact physical likelihood or posterior correctness.

### Method B — tangent/interpolation extension

| Model row | Status | Reason code | Blocking detail |
|---|---|---|---|
| Austria SIR | VALID | `NONE` | Fresh current-source score/tangent artifact at `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260806/pilot-01-selected-current-closure/artifact`; runtime child now loads and executes. |
| Austria SIR T2 extension | VALID | `NONE` | Fresh current-source T1/T2 training-JVP chain at `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-training-jvp-20260806/attempt-01-current-closure`; runtime child now loads and executes. |
| LGSSM | BLOCKED | `B_SCALAR_COMPATIBLE_ROUTE_MISSING` | Existing persisted LGSSM frozen-transport artifacts change the scalar via transport pullback plus log-Jacobian, so they are not same-scalar-compatible Method B rows for this campaign. |
| KSC SV | BLOCKED | `B_ROUTE_MISSING` | No persisted Method B infrastructure yet. |
| actual SV | BLOCKED | `B_ROUTE_MISSING` | No persisted Method B infrastructure yet. |
| predator-prey | BLOCKED | `B_ROUTE_MISSING` | No persisted Method B infrastructure yet. |

## Engineering repairs completed during execution

The campaign runner and actual-SV route were repaired so the Method A actual-SV row now uses the same active batch TT finite program for both value and score, with a manual replayed score backend rather than autodiff.

Files changed in the actual-SV repair pass:

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
  - Added the same-program manual score backend for the active batch TT scalar.
  - Preserved `batched_fixed_tt_likelihood_value_trace(...)` as the value authority.
  - Replaced the runtime score path with explicit directional replay over the existing one-axis and two-axis fit steps.
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
  - Switched adapter metadata and runtime wiring to the same-program manual score backend.
  - Marked the actual-SV HMC-facing route as non-autodiff for this finite program.
- `tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py`
  - Added / updated same-scalar finite-difference checks for the active finite program.
- `tests/test_zhao_cui_actual_sv_neutra_target.py`
  - Updated adapter-level tests for the same-program manual backend metadata and finite-difference behavior.

These changes were sufficient to move actual SV from blocked autodiff-backed Method A status into same-program manual / analytical Method A admission without changing the value scalar.

## Decision table

| Decision | Primary criterion | Veto/blocked condition | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit actual SV under Method A | Same-program manual score matches centered finite differences for the active batch TT scalar | Any mismatch with the active scalar or fallback to autodiff-only score provenance | Whether wider probe coverage will reveal a stability issue | Keep the current row admitted and optionally extend the FD ladder | No exact physical-likelihood claim, no posterior claim, no HMC claim |
| Keep Method B separate and subordinate to Method A | Method B should remain model-by-model and contingent on Method A admission | Any attempt to treat Method B recovery as Method A completion | Whether Austria Method B stays live after future source changes | Preserve explicit Method B rows and blocked rows when rerunning the summary | No Method B promotion claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard execution success for the current runner state | PASS |
| Manual / analytical admission for Method A | PASS for all five Method A rows |
| Autodiff-backed Method A rows | None remain in the active summary |
| Austria Method B availability | Live for Austria SIR only |
| Cross-model A vs B comparison | Not concluded beyond the declared route evidence |
| Default-readiness | No |
| Next evidence needed | Optional wider FD ladder for actual SV or future Method B work |

## Important cautions

- This result does **not** upgrade any row into an exact physical-likelihood, posterior-correct, HMC-ready, source-faithful, default-ready, or production-ready claim.
- The Method A rows are baseline fixed-variant value/score routes only.
- The Method B rows are mixed: Austria SIR is live under fresh current-source artifacts, while all other models remain blocked because no persisted Method B route exists yet.
- Austria SIR Method B remains the only model with a concretely realized extension route in the current tree.
- The actual-SV manual backend is a same-program derivative of the active deterministic batch TT approximation, not a claim that the route equals the exact actual-SV likelihood.

## Recommended next step

The most sensible next step is:

1. keep Austria SIR as the first and currently only live same-scalar-compatible Method B model,
2. decide whether this campaign should stop there or continue into a new LGSSM Method B wrapper design phase, and
3. if stronger actual-SV evidence is desired, add a slightly wider same-scalar finite-difference ladder around the current campaign probe rather than changing the route family.
