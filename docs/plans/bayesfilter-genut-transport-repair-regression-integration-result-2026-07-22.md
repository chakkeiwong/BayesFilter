# GenUT Transport Repair Regression Result

Date: 2026-07-22
Plan: `docs/plans/bayesfilter-genut-transport-repair-regression-integration-plan-2026-07-22.md`
Comparison artifact: `docs/benchmarks/artifacts/genut_transport_repair_regression_20260722/comparison_attempt01/`

Integration base: concurrent authorized SGQF commit `ee346978741cf306167bb5a3e11c8aded506d593`
(`Complete SGQF high-dimensional leaderboard column`), already present on both
local `main` and `origin/main` before the GenUT integration commit.

## Result

The repaired candidate route completed the requested regression scope on the
RTX 4080 SUPER with FP32, TF32, XLA, and verified TensorFlow memory growth.
The repair uses the realized row-mass quotient, fixed terminal balancing, the
manual recursive total score, and fail-closed transition/likelihood/reset
validity. The runtime contains no autodiff or finite-difference score path.

Fresh structural tuning selected:

```text
epsilon=4, sinkhorn_steps=4, balance_steps=16, ridge=1e-6
```

The fresh structural claim used `N=1002`, `T=100`, and particle seeds
`2026072301..2026072308`. Every row was finite. The maximum reset residual was
`3.90e-6`, maximum transition residual `9.54e-7`, and maximum score-increment
sum relative residual `8.19e-7`. The repaired structural row is included as a
candidate but is not leaderboard-admitted.

Regression scopes:

- LGSSM: `N=1008`, `T=2,10,50`, 16 common historical claim seeds, scope-specific four-control tuning, and Kalman diagnostics.
- Fresh exact transformed SV: `N=1998`, `T=50`, 16 common claim seeds, fresh-DGP dense reference, and scope-specific tuning.
- Predator-prey: `N=1002`, `T=20`, 16 common claim seeds, scope-specific tuning, no exact score oracle.
- Actual Austria SIR: canonical fixed source-order SGQF value-only route, CPU-XLA/GPU-XLA parity; no free parameter score exists for this route.

The old reduced SIR/J=1 mechanics fixture and original iid-normal SV fixture
are excluded from comparison.

## Decision Table

| Decision | Status | Interpretation |
|---|---|---|
| Finite/device/reset screen | Pass | All repaired particle and structural claim rows are finite and on GPU/XLA; SIR CPU/GPU parity is exact in the tested value. |
| Regression preservation | Pass descriptively | Values and scores remain finite; paired deltas are reported in the comparison artifact. The repaired scalar is a new finite program, not an algebraic replay of the old scalar. |
| Structural candidate | Included, not admitted | Structural hard gates pass, but no independent nonlinear score authority establishes accuracy or leaderboard admission. |
| Default promotion | Not established | The run is a feasibility/regression result, not broad default-readiness evidence. |
| HMC readiness | Not established | Recursive score consistency and finite execution do not prove posterior correctness or HMC validity. |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the fresh repaired scopes. |
| Statistically supported ranking | None. Prior-versus-repaired differences compare different finite programs. |
| Descriptive-only differences | Means, sample SDs, 95% Student-t intervals, and paired common-seed deltas in `comparison.json`. |
| Default readiness | Not established. |
| Next evidence needed | Independent nonlinear score authority and broader model-specific claim campaigns. |

## Attempt Ledger

| Attempt | Classification | Repair |
|---|---|---|
| Structural tuning attempt 01 | Harness serialization failure when an invalid arm reached `statistics.variance` as `None` | Preserve invalid raw rows, assign no variance objective, and keep the arm ineligible. |
| Structural tuning attempt 02 | Pass | Fresh tuned structural candidate and eight-seed claim completed. |
| Model regression attempt 01 | Harness initialized GPU before memory-growth configuration; no model computation ran | Configure memory growth immediately after TensorFlow import. |
| Model regression attempt 02 | Pass | LGSSM horizon ladder, fresh SV, and predator-prey regressions completed. |
| LGSSM common-seed replay | Pass | Replayed frozen repaired tuning with historical `82220..82235` seeds for paired comparison. |
| Austria SIR GPU attempt 01 | Runner contract failure because CPU reference was omitted | Generate CPU-XLA reference first. |
| Austria SIR CPU attempt 01 and GPU attempt 02 | Pass | Exact tested CPU/GPU value parity. |

Post-merge validation on `ee346978` passed 63 focused GenUT/SGQF tests and 22
focused Zhao-Cui/transport tests, plus Python compilation and `git diff --check`.

## Post-run Red Team

The strongest alternative explanation is that terminal balancing changes the
finite objective enough to improve numerical validity while changing the
approximation bias. The current evidence supports that the repaired program is
finite and reproducible, not that it is closer to the exact nonlinear target.
The structural score intervals are very wide after conversion to physical
coordinates, so the structural score should not be treated as an accurate
parameter estimate. The next result that would overturn the feasibility
conclusion is a reproducible non-finite row or invalid reset under the fresh
scope; none occurred in this campaign.
