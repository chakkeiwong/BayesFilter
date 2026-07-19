# Canonical LGSSM TF32 Balance And Horizon Continuation Result

Date: 2026-07-18
Campaign ID: `canonical-lgssm-tf32-balance-horizon-continuation-20260718`
Status: `T2_PASS_T10_CLAIM_VETO_T50_NOT_RUN`

## Outcome

The pushed Contract E mass-accounting repair is correct for the identified
TF32 T=2 marginal failure. The fused state now computes row mass and its JVP
with explicit reductions of the coupling rather than passing the all-ones
payload channel through a TF32-eligible GEMM. This preserves the same finite
value/score program, one shared OT traversal, and the declared `K=N=1024`
chunk policy.

Commit: `1164c881b63dad6b0a0400045eda5d5202700ab3` (`origin/main`).

The earlier TF32 count-2 failure was reproduced before the repair:
`TV_col=1.777e-4` while `E_row` had already plateaued near `6.13e-4` after
three balance steps. Counts 2, 3, 5, and 8 all failed the unchanged
`TV_col <= 1e-4` design gate. A direct-coordinate squared-distance diagnostic
left `TV_col=1.765e-4`, falsifying the hypothesis that TF32 distance GEMMs
were the cause. The explicit mass-reduction repair then passed with count 2.

## Evidence Summary

| Node | Status | Evidence |
| --- | --- | --- |
| TF32 marginal design, `T=2,N=1024`, seeds `81600..81607`, count 2 | pass | max `TV_col=1.610e-5`, max `E_row=6.239e-4`, bitwise replay, one shared Sinkhorn/balance/transport state per step, zero diagnostic work |
| TF32 marginal audit, disjoint seeds `81620..81627`, count 2 | pass | max `TV_col=2.607e-5`, max `E_row=4.784e-4`, bitwise replay, same work counts |
| Same-count float64 T=2 reference, seeds `81500..81515` | hard-valid | `TV_col=1.785e-5`, `E_row=3.925e-4`, `StatelessWhile`, `K=N`, 8 GiB logical limit |
| TF32 T=2 claim, seeds `81500..81515` | pass | hard-valid, `TV_col=1.801e-5`, `E_row=3.973e-4`, warm `1.010 s`, peak `0.890 GB`, exact work counts |
| TF32 T=2 inactive witness | pass | hard-valid and zero Sinkhorn, balance, transport, marginal, and diagnostic work |
| TF32 T=10 one-seed resource witness | pass | hard-valid, `TV_col=1.072e-5`, `E_row=4.717e-4`, warm `0.630 s`, peak `0.105 GB`, within cap |
| TF32 T=10 16-seed claim | veto | finite/replayable/within cap with correct work, but `TV_col=2.869e-5` passes while max `E_row=0.020587 > 0.01`; chart/reset validity therefore fails |
| TF32 T=50 | not run | conditional supervisor stopped at the T=10 claim veto |

The T=2 same-program precision gate passed: source hashes, campaign/plan,
count, seeds, dtype identity, replay, score signs, and no-order-one drift all
match. Float32 minus float64 aggregate value drift was `0.005277`; score drift
was `[0.003217, 0.001162, 0.000001, -0.022606, -0.010301]`. These are
descriptive precision diagnostics, not a superiority claim.

## Decision Table

| Decision | Primary criterion | Veto status | Interpretation | Next justified action |
| --- | --- | --- | --- | --- |
| Promote mass-accounting repair for T=2 | fused value/JVP parity, direct marginals, one-solve work | pass | implementation defect is repaired at T=2 | retain repair and source identity |
| Admit TF32 count 2 for all horizons | all active resets pass marginal gates | veto at T=10 16-seed claim | count 2 is target-valid at T=2 but not horizon-universal | select against representative T=10 states before any retry |
| Promote T=10 route | 16-seed hard-valid claim node | veto | candidate fails row-marginal gate, not resource budget | diagnose failing seed/time and balance-state trajectory |
| Run T=50 | conditional on T=10 claim pass | not reached | no T=50 evidence exists | do not launch until T=10 passes |
| HMC/leaderboard readiness | outside this campaign | not evaluated | unsupported | separate reviewed program required |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | T=2 passes; T=10 16-seed claim fails only `E_row`/reset validity |
| Statistically supported ranking | none; no candidate comparison with uncertainty was run |
| Descriptive-only differences | Kalman value/score differences, runtime, allocator peak, and cross-precision drift |
| Default-readiness | not established beyond the repaired T=2 canonical implementation and its fixed chunk policy |
| Next evidence needed | per-seed/time T=10 marginal trajectories, representative-state balance selection, then fresh T=10 claim and only afterward T=50 |

## Attempt And Repair Ledger

1. Selection attempts 1 and 2 failed in the new harness before candidate
   execution due to dtype-rejecting TensorFlow conversions. They are harness
   failures, not numerical results.
2. Selection attempt 3 reached XLA and rejected all predeclared counts 2, 3,
   5, and 8 under the unchanged gates.
3. Direct-distance diagnostic attempt 1 failed in reporting because the
   functional result does not retain raw-column max history; it did not alter
   the route.
4. Direct-distance diagnostic attempt 2 failed before execution because
   imports initialized TensorFlow before the 8 GiB logical device cap. The
   import-order repair was localized and verified.
5. Direct-distance diagnostic attempt 3 completed and falsified the distance
   GEMM hypothesis (`TV_col=1.765e-4` at count 2).
6. Mass-reduction repair attempt 4 passed selection and disjoint audit.
7. Post-commit selection attempt 5 and same-count float64 reference completed
   with source identity bound to `1164c88`.
8. The conditional supervisor passed T=2, inactive zero-OT, and the T=10
   resource witness, then stopped at the T=10 claim veto. No T=50 launch was
   attempted.

## Post-Run Red Team

The strongest alternative explanation for the T=10 veto is not that the
algorithmic Contract E target is wrong, but that the count was selected from a
T=2 state distribution and transferred to later, more ill-conditioned states.
The one-seed T=10 witness supports this: it passed while the 16-seed batch
exposed `E_row=0.020587`. The result would be overturned by a representative
T=10 design/audit selection that passes with the same finite program and by a
fresh T=10 claim node passing every seed/time gate.

The weakest current evidence is the absence of per-seed/time row-error fields
in the terminal node schema and the lack of T=50 execution. Neither omission
is silently filled by inference. The next plan must add those diagnostics and
select the balance count over representative T=10 states without using Kalman
accuracy to choose it.

## Nonclaims

This result does not establish nonlinear-model validity, HMC readiness,
posterior correctness, statistical superiority, production admission,
leaderboard completeness, a universal balance count, or T=50 feasibility.
