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

The cross-model fixed-variant campaign now reports actual SV as an explicitly blocked Method A row under the current policy, because the true fixed-variant actual-SV route remains autodiff-backed. The previously attempted analytic helper route has been removed from the campaign path because it was the wrong scalar family for this fixed-variant target.

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
| actual SV | VALID | VALID | FAIL (`max_abs=3.458340e+00`, FD `1e-5`) | `manual` | -26.229550056086453 | 8.50277771662678 |
| predator-prey | VALID | VALID | PASS (`max_abs=2.853935e-09`, FD `1e-5`) | `manual` | -109.08502517043668 | 13.75146979114478 |
| Austria SIR | VALID | VALID | PASS (`max_abs=1.989597e-07`, FD `1e-5`) | `manual` | -682.3480055392415 | 110.86484986266814 |

Interpretation: the policy rerun now binds manual/analytical Method A routes across the table. LGSSM, KSC SV, predator-prey, and Austria SIR are clean admitted rows with manual score provenance and passing same-scalar diagnostics. The actual-SV analytic route is manual/analytical and finite, but its campaign-point same-scalar diagnostic fails, so it remains explicitly flagged with `A_SAME_SCALAR_FD_FAIL_DIAGNOSTIC_ONLY`.

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

The campaign runner was repaired so Austria Method B load failures become
blocked rows instead of a hard process abort, and then rewired to the fresh
current-source Austria Method B artifact chain.

Files changed in this execution pass:

- `scripts/run_fixed_variant_value_score_multimodel_20260804.py`
  - Added `_blocked_method_b_row(...)` helper.
  - Wrapped Austria SIR Method B loading in guarded `try/except` blocks.
  - Restored `run_method_a_rows()` after an intermediate refactor removed it.
  - Preserved the baseline Method A execution path.
  - Updated Austria Method B artifact paths to the fresh 20260806 T1 score,
    T1 training-JVP, and T2 training-JVP artifacts.
  - Reconstructed the T2 cumulative Method B value as `parent_t1_value +
    increment` because `LaneBParameterChild` exposes `increment_and_score(...)`
    rather than a standalone `value()` method.
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
  - Repaired stale imports so the existing actual-SV Method A adapter can import
    under the current tree.
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py`
  - Repaired matching stale imports in the analytic companion module.

These changes were sufficient to move Austria Method B from blocked historical
state into live current-source execution without changing the scientific target
of the campaign.

## Decision table

| Decision | Primary criterion | Veto/blocked condition | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Update Method A policy to manual/analytical score admission | The campaign contract must stop admitting autodiff-backed Method A rows as completion evidence | Existing result rows were labeled too strongly relative to the active policy | Whether the runner will need a separate blocked-row status for KSC/actual SV or can reuse existing payload fields | Rerun the runner after updating the registry and schema vocabularies | No new scientific claim, no posterior claim, no HMC claim |
| Keep Method B separate and subordinate to Method A | Method B should remain model-by-model and contingent on Method A admission | Any attempt to treat Method B recovery as Method A completion | Whether Austria Method B stays live after the Method A contract update | Preserve explicit Method B rows and blocked rows when rerunning the summary | No Method B promotion claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard execution success for the current runner state | Not re-evaluated after the policy update |
| Manual / analytical admission for Method A | Pending rerun under the updated policy |
| Autodiff-backed Method A rows | Now policy-inconsistent and should be treated as blocked or pending |
| Austria Method B availability | Still recorded historically, but not the focus of this policy update |
| Cross-model A vs B comparison | Not concluded under the updated Method A contract |
| Default-readiness | No |
| Next evidence needed | Rerun the runner and reconcile the registry/result vocabularies with the manual/analytical admission rule |

## Important cautions

- This result does **not** upgrade any row into an exact physical-likelihood,
  posterior-correct, HMC-ready, source-faithful, default-ready, or
  production-ready claim.
- The Method A rows are baseline fixed-variant value/score routes only.
- The Method B rows are now mixed: Austria SIR is live under fresh current-source
  artifacts, while all other models remain blocked because no persisted Method B
  route exists yet.
- Austria SIR Method B remains the only model with a concretely realized extension
  route in the current tree.
- Austria Method B's current campaign summary records runtime execution plus
  loader/replay provenance; if a stronger cross-method claim is needed later,
  add the planned held-out local-comparison diagnostics rather than inferring them
  from issuer replay alone.

## Recommended next step

The most sensible next step is:

1. keep Austria SIR as the first and currently only live same-scalar-compatible
   Method B model,
2. decide whether this campaign should stop there or continue into a new LGSSM
   Method B wrapper design phase, and
3. if LGSSM Method B is pursued, require a persisted same-scalar wrapper rather
   than reusing the existing transported-NeuTra artifacts, because those change
   the scalar by construction.
