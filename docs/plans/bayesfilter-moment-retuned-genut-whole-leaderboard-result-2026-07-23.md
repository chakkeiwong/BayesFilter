# Moment-Retuned GenUT Whole Leaderboard Result

Date: 2026-07-23  
Plan: `docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-plan-2026-07-23.md`  
Final artifact: `docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json`  
Derived labeled analysis: `docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/analysis.json`

## Outcome

The current six-row same-target leaderboard was produced. GenUT and fixed SGQF
have finite value/score cells on all six rows. Fixed-variant Zhao-Cui has finite
value/score cells on five rows and one honest blocked cell: the score-capable
Austria SIR target has no implemented observed-data marginal Zhao-Cui score.
The available local complete-data score was not substituted.

The final run used FP32, TF32, XLA, GPU memory growth, `N=1008`, 16 GenUT
claim seeds, and per-scope offline tuning. It completed in `594.38` seconds with
`hard_valid=true`. TensorFlow peak live allocator use was `134,578,688` bytes
(about `128.3 MiB`); this is not the same as process reservation reported by
`nvidia-smi`.

## Leaderboard

GenUT entries are the mean over 16 particle seeds. SGQF and Zhao-Cui entries
are deterministic fixed-route values at the frozen target theta. Cross-method
differences are descriptive; this table does not support a method ranking.

| Model | Method | Value | Score |
|---|---|---:|---|
| LGSSM `T=50` | GenUT | -136.333493 | (5.795313, -4.049524, 0.239922, -1.983577, 5.537582) |
| LGSSM `T=50` | SGQF / exact affine | -136.075975 | (5.655446, -3.835057, 0.302362, -1.917176, 4.354276) |
| LGSSM `T=50` | Zhao-Cui affine adapter | -136.075975 | (5.655446, -3.835057, 0.302362, -1.917176, 4.354276) |
| KSC-SV `T=10` | GenUT | -19.953950 | (-0.694425, 0.607675) |
| KSC-SV `T=10` | SGQF | -19.950942 | (-0.692475, 0.609578) |
| KSC-SV `T=10` | Zhao-Cui | -19.956289 | (-0.705672, 0.635489) |
| Exact transformed SV `T=10` | GenUT | -19.994411 | (-0.698259, 0.567965) |
| Exact transformed SV `T=10` | SGQF | -19.737671 | (-0.532466, 0.745356) |
| Exact transformed SV `T=10` | Zhao-Cui | -19.995663 | (-0.707200, 0.590572) |
| Generalized SV `T=10` | GenUT | -16.017544 | (-0.122880, -0.152849, 0.022310) |
| Generalized SV `T=10` | SGQF | -16.019455 | (-0.122006, -0.153907, 0.022287) |
| Generalized SV `T=10` | Zhao-Cui | -16.019873 | (-0.125470, -0.154843, 0.022261) |
| Predator-prey `T=20` | GenUT | -102.739536 | (-27.775234, 0.077647, -0.087487, 1.042272, 18.367237, -23.650981) |
| Predator-prey `T=20` | SGQF | -102.622704 | (-27.641143, 0.084107, -0.084143, 0.855699, 17.525598, -22.634978) |
| Predator-prey `T=20` | Zhao-Cui fixed-variant extension | -102.419676 | (-22.676433, 0.138280, -0.083417, 0.245887, 17.605349, -22.815862) |
| Austria SIR `d=18`, `T=20` | GenUT | -683.363808 | (-865.923095, 170.885295, 114.981207) |
| Austria SIR `d=18`, `T=20` | SGQF | -682.348006 | (28.739453, -106.658857, 9.431176) |
| Austria SIR `d=18`, `T=20` | Zhao-Cui | blocked | observed-data analytical marginal score route absent |

Parameter order is respectively:

- LGSSM: `(phi1, phi2, phi3, q_scale, r_scale)`;
- both SV rows: `(z_gamma, log_beta)`;
- generalized SV: `(z_gamma, log_tau, mu_over_tau)`;
- predator-prey: `(r, K, a, s, u, v)`; and
- Austria SIR: `(log_kappa_scale, log_nu_scale, log_observation_noise_scale)`.

## Uncertainty And Interpretation

| Row | GenUT value 95% CI | Comparator interpretation |
|---|---:|---|
| LGSSM `T=50` | [-136.582858, -136.084128] | exact affine value is outside; finite GenUT value bias remains |
| KSC-SV `T=10` | [-19.979315, -19.928584] | SGQF and Zhao-Cui values and scores are inside the GenUT intervals |
| Exact transformed SV `T=10` | [-20.020429, -19.968392] | Zhao-Cui is inside; SGQF value and both score coordinates are outside |
| Generalized SV `T=10` | [-16.025832, -16.009256] | both comparator values and all score coordinates are inside |
| Predator-prey `T=20` | [-102.895313, -102.583759] | SGQF value is inside; several score coordinates and Zhao-Cui value are outside |
| Austria SIR `T=20` | [-683.703076, -683.024540] | SGQF value is outside; the extremely wide GenUT score intervals contain SGQF but are not evidence of agreement |

The Austria SIR GenUT score has very high Monte Carlo variance: sample SDs are
approximately `(3435.6, 1272.4, 302.0)`. The score means therefore should not
be interpreted as stable estimates. The value is much more stable, with sample
SD `0.637`.

## Repairs Made

1. Corrected the active LGSSM and predator-prey event-order contracts. Prior
   GenUT evidence using a different first-observation convention was not reused.
2. Rejected stale KSC/exact-SV comparator artifacts by tensor hash and freshly
   evaluated the current SGQF and fixed Zhao-Cui routes.
3. Added the TensorFlow-only parameterized Austria SIR GenUT adapter with
   explicit RK4 state/parameter tangents and analytical observation score.
4. Added repository-issued identity registrations for KSC-SV, generalized SV,
   and parameterized Austria SIR GenUT routes.
5. Repaired the tuning harness so nonfinite candidate arms become ineligible
   rather than breaking the variance diagnostic.
6. Diagnosed the SIR failure at `epsilon=2` as a real Sinkhorn quotient-column
   TV gate failure (`1.145e-4 > 1e-4`). The gate was not relaxed. Scope-specific
   tuning selected `epsilon=8`, `sinkhorn_steps=16`, and `balance_steps=16`,
   which passed all 16 untouched claim seeds.

## Attempts

| Attempt | Classification | Result |
|---|---|---|
| `attempt01` | harness failure | nonfinite candidate variance arithmetic; no scientific result |
| `attempt02` | candidate failure / repair trigger | five rows passed; SIR failed marginal validity under inherited controls |
| `attempt03_sir_repair` | localization diagnostic | 12/16 SIR seeds invalid at `epsilon=2`; quotient-column TV gate identified |
| `attempt04_sir_retuned` | SIR repair evidence | SIR passed 16/16 with scope-specific controls |
| `attempt05_final` | final claim artifact | all six GenUT and SGQF cells valid; five Zhao-Cui cells valid, one blocked |

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Publish current feasibility matrix | 17/18 required cells executed, explicit terminal blocker for the remaining cell | numerical/target gates clear for executed cells | nonlinear method gaps are mostly descriptive | use this artifact as the current comparison baseline | no method ranking |
| GenUT six-model viability | all six rows finite under scope tuning | clear | LGSSM bias and SIR score variance | target-specific replication and score-variance work | no default or HMC readiness |
| Whole Zhao-Cui coverage | fails at Austria SIR observed-data score | `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE` | requires an 18/36-D fixed-TTSIRT proposal/derivative architecture | separate Zhao-Cui implementation program | local complete-data score is not accepted |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Pass for 17 executed cells; one explicit Zhao-Cui SIR architecture blocker |
| Statistically supported ranking | None across methods |
| Descriptive-only differences | All cross-method value/score gaps |
| Default readiness | Not established |
| Next evidence needed | Zhao-Cui SIR observed-data route; replicated same-target uncertainty; GenUT SIR score-variance repair |

## Post-Run Red Team

The strongest alternative explanation for the apparently close scalar-model
rows is target simplicity and short `T=10`, not broad nonlinear accuracy. The
result that would overturn GenUT viability is a same-target replicated run with
stable numerical gates but persistent value/score disagreement beyond declared
uncertainty. The weakest evidence is Austria SIR score accuracy: it is finite
but too variable to support agreement or disagreement. Predator-prey Zhao-Cui
remains an `extension_or_invention`, not a source-faithful default.

