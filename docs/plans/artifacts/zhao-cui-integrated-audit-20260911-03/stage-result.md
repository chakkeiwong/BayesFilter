# Zhao-Cui Integrated Audit: Stage 03 Diagnostic Reconciliation

Date: 2026-09-11

Status: prior bounded CPU evidence reconciled to the current source; no new
runtime code or experiment was run.

## Evidence identity

The rank and finite-output diagnostic is preserved at
`docs/plans/artifacts/zhao-cui-audit-20260911-01/diagnostic-result-01.json`.
Its recorded hashes for `zhao_cui_algorithm2_preparation_tf.py`,
`zhao_cui_algorithm3_tf.py`, and `squared_tt_engine_v0_tf.py` match the
current checkout hashes recorded in the Stage 01 inventory. The focused
conditional/CDF/score/call-chain checks are preserved at
`docs/plans/artifacts/zhao-cui-audit-20260911-03/stage-result.md`; the source
files inspected for those checks are unchanged at the Stage 01 hashes.

This reconciliation is sufficient for the current audit question. It does not
promote those fixtures to scientific, posterior, HMC, GPU, or production
evidence.

## Reconciled findings

| Check | Recorded result | Audit meaning |
|---|---|---|
| Rank activation | `h(x,y)=1+0.25xy`, inherited effective rank one, RMS `0.234375`; active rank-two warm start RMS about `1.03e-10` | The initializer at `squared_tt_engine_v0_tf.py:136-148` can silently disable configured rank channels. This blocks a rank-two preparation claim until repaired and regression-tested. |
| Aggregate finite guard | Extreme finite-input fixture returned `valid=true`, `log_likelihood=-inf`, weight sums `2.0` | `zhao_cui_algorithm3_tf.py:171-200` does not include accumulated increments, weight sums, or ESS in the validity mask. This is a fail-closed defect. |
| Coupled conditional mechanics | Maximum fixture error `6.7e-16` | The tested conditional CDF, density, marginal, and affine Jacobian agree on the bounded fixture. This is mechanics evidence only. |
| Numerical proposal law | Cell-slope error `2.5e-11`; smooth-TT density gap `3.68e-3` | The grid/interpolation derivative matches the implemented piecewise law but differs from the smooth TT density. The manuscript wording must name the numerical law explicitly. |
| Frozen value/score parity | Value error `0`; score error about `2.0e-10` on the C2 mechanics fixture | The frozen finite-program recurrence is internally consistent on that fixture. It is not evidence for an adaptive TT total derivative or scientific filtering accuracy. |
| Consumer inventory | No non-test consumer for `compile_algorithm3`, `prepare_algorithm2_proposals`, or the sigma-point guide | Isolated primitives and tests do not close a claim-bearing production call chain. |

## Decision table

| Decision | Primary criterion | Veto/status | Next justified action | Not concluded |
|---|---|---|---|---|
| Continue audit | Source anchors and bounded mechanics remain coherent | Four implementation/evidence blockers remain | Design focused repairs and call-chain ownership | No rejection of the Zhao-Cui research direction |
| Admit current route as claim-bearing | Full source-anchored call chain, finite aggregates, active rank, named numerical law | Failed | Repair and add no-fire/negative-control checks before any claim run | No default, HMC, GPU, posterior, or leaderboard readiness |
| Preserve frozen score as diagnostic | Same frozen scalar value/score parity | Passes the tested fixture only | Retain as fixed-program reference; keep adaptive total derivative separate | Not an exact adaptive score |

## Exact next action

Prepare a repair proposal with four bounded components: (1) activate rank
channels in the initializer and add a healthy rank-two no-fire regression;
(2) guard finite increments, finite positive weight sums, normalized weights,
and ESS; (3) revise the numerical-CDF manuscript paragraph to define the
piecewise proposal density and smooth TT reference separately; and (4) trace or
build one generic claim-bearing consumer endpoint. Do not apply the runtime
repairs until their mathematical and source-boundary classifications are
written into the repair note.

