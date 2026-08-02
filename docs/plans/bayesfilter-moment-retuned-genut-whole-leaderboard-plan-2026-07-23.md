# Moment-Retuned GenUT Whole Leaderboard Plan

Date: 2026-07-23  
Status: `EXECUTING_BOUNDED_FEASIBILITY_CAMPAIGN`

## Research Intent

Produce a current, same-target comparison of the moment-retuned GenUT finite
filter against the available fixed SGQF and fixed-variant Zhao-Cui routes for
the model suite used for nonlinear-filter feasibility work. The target suite is
LGSSM, KSC-SV, exact transformed SV, generalized SV, additive-Gaussian
predator-prey, and the score-capable Austria SIR (`J=9`, state dimension
`d=18`, `T=20`). This is not an NAWM experiment and does not estimate an LGSSM
for its own sake.

The principal question is coverage and feasibility under one shared finite
value/recursive-score execution contract. Lower moment residuals are an
explanatory diagnostic; they are not a likelihood or score oracle.

## Evidence Contract

| Field | Contract |
|---|---|
| Candidate | Existing non-fused `transition -> likelihood -> OT -> Contract E -> higher-moment correction -> equal-weight` GenUT route |
| Candidate tuning | Scope-specific calibration/validation selection using normalized diagonal skewness/kurtosis residual, with conditional value/score replicate variance as a secondary diagnostic |
| Baseline | Same target observations, event order, parameter chart, and theta truth evaluated by the fixed SGQF route |
| Zhao-Cui comparator | Fixed-variant route only; retained-grid/multistate historical routes are excluded. Source classification is recorded per cell. |
| Primary pass criterion | Every executed cell is finite, target/hash matched, uses the declared score route, and satisfies the numerical/reset/increment gates |
| Score criterion | Runtime score is analytical/manual recursive or source route analytical score; no runtime autodiff and no runtime finite differences |
| Hard vetoes | Target mismatch, stale tuning identity, nonfinite value/score, invalid covariance/OT/reset, score-increment mismatch, wrong time order, forbidden NumPy/XLA path, or missing required artifact |
| Explanatory diagnostics | Dense/Kalman agreement where available, moment residuals, score differences, runtime, allocator bytes, and one-seed descriptive gaps |
| Statistical interpretation | 16 common particle seeds for stochastic GenUT cells; differences are descriptive unless paired uncertainty supports a claim |
| Nonclaims | No unbiasedness, exact nonlinear likelihood, score superiority, default/HMC readiness, source-faithful Zhao-Cui claim for extension routes, or NAWM conclusion |

## Frozen Row Crosswalk

The rows below are the active contract. Legacy `T=1000` source-scope artifacts
remain historical and are not concatenated into this table.

| Row | Target and event order | `N` | Required methods |
|---|---|---:|---|
| `lgssm_T50` | Dataset seed `81100`, physical theta, observe stationary `x0` before later transitions, `T=50` | 1008 | GenUT, SGQF, Zhao-Cui exact affine adapter |
| `ksc_sv_T10` | KSC seven-component transformed observation from seed `81101`, initial-observe prefix, `T=10` | 1008 | GenUT, SGQF, Zhao-Cui fixed branch |
| `exact_sv_T10` | Exact `log(y^2)` transformed observation from seed `81101`, initial-observe prefix, `T=10` | 1008 | GenUT, SGQF, Zhao-Cui fixed branch |
| `generalized_sv_T10` | Generated generalized-SV raw-y target seed `81105`, transition-before-observe, `T=10` | 1008 | GenUT, SGQF, Zhao-Cui fixed design TT |
| `predator_prey_T20` | Source-order RK4 target seed `81104`, `x0 -> transition_1..20 -> y1..y20` | 1008 | GenUT, SGQF, Zhao-Cui fixed-variant extension (classified, not source-faithful) |
| `austria_sir_T20` | Parameterized Austria SIR seed `81120`, `d=18`, `J=9`, `y1..y20`, log-scale theta | 1008 (`28 * 2d`) | GenUT, SGQF; Zhao-Cui blocked pending observed-data marginal score |

All GenUT rows use FP32 tensors, TF32 enabled, XLA enabled, GPU memory growth,
and `N=1008`, which is divisible by `2d` for every row (`d` is at most 18).

## Implementation And Repair Phases

1. **Contract and preflight.** Freeze row IDs, tensor hashes, time order,
   parameter order, score provenance, controls family, seed sets, and output
   root. Fail closed on a mismatch. Preserve all historical artifacts.
2. **Adapter repair.** Add a TensorFlow-only FP32 parameterized Austria SIR
   GenUT adapter with explicit RK4 state/parameter tangents and infectious-only
   observation score. Do not import the float64 diagnostic implementation into
   the XLA path and do not use the reduced two-state mechanics fixture.
3. **Scope tuning.** For each row, use disjoint calibration and validation
   trajectories and two particle seeds. Select controls before reading the 16
   claim seeds. The selected controls are frozen in a repository-issued scope
   identity and cannot be copied across rows.
   The high-dimensional Austria SIR scope includes the additional candidate
   epsilon values `4` and `8` and terminal Sinkhorn/balance count `16`; the
   `epsilon=2` candidate is retained as a diagnostic baseline but is not
   assumed to satisfy the marginal gate at `d=18`.
4. **Comparator execution.** Run SGQF and fixed-variant Zhao-Cui on the exact
   row target. Reuse an earlier result only when the target hash, event order,
   theta chart, route identity, and result kind match; otherwise execute again.
   Emit `blocked` for the Austria SIR Zhao-Cui observed-data score rather than
   using the local complete-data component.
5. **Claim execution.** Run 16 common particle seeds for each GenUT row and
   compute mean, sample SD, and paired 95% intervals where a reference exists.
   Record all raw rows, tuning candidates, identities, memory, device, and
   source hashes.
6. **Terminal audit.** Check target equality, numerical gates, score
   recomposition, tuning separation, route classification, and inference status.
   Produce JSON, Markdown, result note, and reset memo. A complete matrix does
   not imply that every method is promotion-ready.

## Skeptical Plan Audit

- **Wrong baseline risk:** the legacy `T=1000` rows and fixed value-only SIR
  row are excluded; only the crosswalk above is active.
- **Proxy risk:** moment residuals nominate controls but cannot establish value
  or score accuracy; dense/Kalman values remain diagnostics.
- **Missing-route risk:** Zhao-Cui Austria SIR is explicitly blocked because its
  available analytical score is local complete-data, not the observed-data
  filtering score. No demoted retained-grid route may fill it.
- **Unfair comparison risk:** each row binds observations, event order, theta
  chart, dtype, horizon, and `N`; inherited controls are not treated as tuned.
- **Variance risk:** one candidate can pass finite gates while having poor
  stochastic accuracy. Sixteen common seeds and paired intervals are recorded;
  no ranking is claimed from descriptive differences alone.
- **Execution risk:** GPU/XLA, memory growth, no-NumPy runtime, and score
  recomposition are hard gates. A failed candidate is preserved and repaired
  under the bounded campaign budget; it is not silently replaced.

Audit decision: `PASS_WITH_EXPLICIT_BLOCKED_SIR_ZHAO_CUI_CELL`.

## Budget And Stop Conditions

One bounded GPU campaign, 16 controls per newly tuned scope, two calibration
and two validation trajectories, two tuning seeds, and 16 claim seeds per
GenUT row. Stop on target corruption, repeated nonfinite/reset failures,
missing score identity, or exhausted campaign budget. Versioned attempt
directories are mandatory and prior evidence is never overwritten.

## Planned Artifacts

- `docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/`
- `docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-result-2026-07-23.md`
- `docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-reset-memo-2026-07-23.md`
