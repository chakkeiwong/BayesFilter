# Structural AR2 consumer limitation confirmed

The current shared canonical DCTR batch endpoint rejects the AR2 model's
singular process covariance. Both diagnostic parameter rows returned
`value_valid=false`, `force_valid=false`, factorization failure code2 and
first failure at initialization (time-1). The positive-definite factorization
control passed. These are implementation-capability findings; no structural
surrogate-HMC chain was run.

The [scope audit](surrogate-hmc-structural-scope-audit-2026-09-22.md) preserves
the model law, source anchors and pre-run criterion. Its
[executable check](../../tests/test_surrogate_hmc_structural_scope.py) reaches
the actual shared batch endpoint with the fixture's F,Q,H,R and full canonical
controls. Two tests passed in11.645 wall seconds, consuming16.378755 CPU-worker
seconds and no GPU time. GPU devices were intentionally hidden. The
[manifest and source snapshot](artifacts/surrogate-hmc-all-models-20260922-structural-scope-audit-01/run-manifest.json)
and [complete log](artifacts/surrogate-hmc-all-models-20260922-structural-scope-audit-01/pytest.log)
preserve the execution.

The claimed AR2 transition has Q=diag(.09,0) and an exact lag coordinate.
The engine accepts additive Gaussian innovations only when its process
factorization succeeds. Thus the current consumer cannot evaluate this law;
adding process noise to the lag coordinate would compute a different model.
An exact Kalman reference remains possible but does not remove the consumer
limitation. The nonlinear structural posterior additionally requires a shared
innovation to drive the first state and its deterministic quadratic completion.
That capability has not been demonstrated in the canonical consumer.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Mark AR2 canonical consumer unsupported | Expected invalidity propagated through shared endpoint | Singular process factorization fails | A general innovation-space implementation has not been qualified | Preserve scope; implement and qualify general deterministic-completion support before tuning | No rejection of surrogate HMC, no posterior comparison |

| Inference status | Finding |
|---|---|
| Hard veto screen | Current canonical value/force route rejects the original process law |
| Statistically supported ranking | None; deterministic capability check |
| Descriptive-only differences | Timing is diagnostic only |
| Default-readiness | No structural HMC admission |
| Next evidence needed | Same-law general consumer, qualified posterior reference, scoped controls and public tuning |

Red team: a different eligible canonical consumer might already support this
law, but none was connected to the inspected batch path. A passing full-call
test of such a route would overturn the capability blocker. A singular
process covariance is legitimate for AR2; this finding concerns the current
implementation, not the mathematical validity of the model or Kalman filter.
