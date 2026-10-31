# Shared surrogate-HMC repair: consolidated evidence

The shared analytical score and sampler repairs have executable parity evidence. Full posterior sampling is incomplete. Short mechanics runs do not establish convergence, model-score accuracy, or economical inference. FP32 remains the default; SIR and range bearing currently need an optional FP64 recursion with FP32 pair kernels for the checked scopes.

This report preserves the exact source snapshot for every row. An earlier-snapshot result cannot issue a fresh tuning authority, even when a later repair has separate parity evidence.

| Model | Scope | Snapshot | Arithmetic score discrepancy | Mechanics | Warm seconds/transition | Acceptance |
|---|---|---|---:|---|---:|---|
| LGSSM | T50, state 3, parameter 4 | Current source | 0.00398714% | PASS_HMC_MECHANICS_ONLY | 51.315 | 1, 1, 1, 1 |
| KSC SV | T50, state 1, parameter 2 | Current source | 0.0777674% | PASS_HMC_MECHANICS_ONLY | 33.518 | 1, 1, 1, 1 |
| Predator-prey | T50, state 2, parameter 6 | Earlier source | 0.135718% | PASS_HMC_MECHANICS_ONLY | 116.911 | 0.5, 1, 1, 1 |
| SIR | T50, state 18, parameter 3 | Current source | 0.183521% | PASS_HMC_MECHANICS_ONLY | 63.340 | 0, 1, 0.5, 0.5 |
| Range bearing | T20, state 4, parameter 2 | Current source | 0.0235435% | PASS_HMC_MECHANICS_ONLY | 28.034 | 1, 1, 1, 1 |

| Model | Independent reference | Max absolute log-likelihood difference | Max scaled score difference |
|---|---|---:|---:|
| LGSSM | exact_Kalman_CPU_reference_exception | 0.0211268 | 3.834% |
| KSC SV | refined_KSC_grid_approximation | 0.0659808 | 7.83% |
| Predator-prey | Not established in this local selection | — | — |
| SIR | Not established in this local selection | — | — |
| Range bearing | Not established in this local selection | — | — |

These reference comparisons use two independent validation datasets and two local parameter rows per dataset. They measure local approximation differences, with no statistical ranking or posterior bias conclusion. Near-zero score components use an absolute unit scale.

Latest current-source attempts:

- LGSSM: HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN; docs/plans/artifacts/surrogate-hmc-guard-advance-lgssm-20261013-01/advance.json
- KSC SV: HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN; docs/plans/artifacts/surrogate-hmc-guard-advance-ksc-20261013-01/advance.json
- Predator-prey: TUNING_TIME_LIMIT_REACHED; docs/plans/artifacts/surrogate-hmc-guard-advance-predator-20261013-01/advance.json
  - Saved candidate0: valid=True, precision_pass=True, max scaled arithmetic discrepancy 0.0348313%. No final selection.
  - Saved candidate1: valid=True, precision_pass=True, max scaled arithmetic discrepancy 0.0351927%. No final selection.
- SIR: HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN; docs/plans/artifacts/surrogate-hmc-guard-advance-sir-20261013-01/advance.json
- Range bearing: HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN; docs/plans/artifacts/surrogate-hmc-guard-advance-range-20261013-01/advance.json

Arithmetic discrepancies use max(1, absolute FP64 component) as the denominator and compare the same finite filter. They are not discrepancies from a model oracle. Timing covers a warmed transition of all four chains at L=1, epsilon=0.002. Acceptance uses only two proposals per chain; these are descriptive observations.

The SIR pilot has one chain with no accepted proposal in its two-step warmed run. It passes the limited numerical mechanics criterion, but does not pass a claim of established movement or convergence. R-hat and posterior ESS are not available.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain verified engineering repairs | Paired value/score/state checks and focused tests | See linked per-repair results | Limited fixtures and local parameter coverage | Continue the master only within available compute | Statistical speed superiority |
| Keep viable finite-filter candidates exploratory | Own-scope precision and short numerical health | Rejected precision candidates remain rejected | Force accuracy against the actual MH endpoint and posterior region | Improve cost and independent reference coverage before long chains | Full posterior inference or canonical LEDH admission |
| Keep full posterior phase open | Adaptive warmup and cumulative sampling not completed | Required convergence/reference evidence absent | Current measured cost and explicit prior/target choice | Resume only under an adequate reviewed budget and target scope | Five-model HMC campaign completed |

| Inference status | Finding |
|---|---|
| Hard veto screen | See each numerical mechanics status; source mismatches prohibit reuse as current tuning authority |
| Statistically supported ranking | None |
| Descriptive differences | Timing, short acceptance, energy errors and three-pair compact-covariance timings |
| Default readiness | FP32 stays default; mixed precision remains an optional checked scope; no inference/default promotion |
| Next evidence needed | Practical cost, justified endpoint/prior, target-specific tuning, adaptive warmup, cumulative R-hat/ESS and posterior reference agreement |

The LGSSM MH endpoint is exact Kalman under the benchmark prior. KSC uses a fixed 1201-node grid whose refinement must be checked over the posterior region. Predator-prey and SIR use fixed bootstrap likelihood approximations; range bearing uses a fixed trajectory-importance approximation. These endpoint limitations are separate from the precision of the proposal force.

At the measured L=1 rate, even 2000 transitions per chain cost many hours. This is an illustrative projection before retained sampling, not a tuned-HMC forecast. Tuning may change step size, trajectory length and efficiency.

Post-run red team: local precision and a healthy two-step chain can coexist with a poor force, biased approximate endpoint, inadequate parameter coverage, or an unaffordable long run. Independent posterior comparisons and longer cumulative diagnostics could overturn the present viability judgment. Heuristic/practical dominance has not been established, so practical-method promotion is unsupported.

Budget charged: 35988.991 / 36000.000 seconds; ledger remainder 11.009 seconds. Running reservations: 0. The ledger remains authoritative.

Evidence paths:

- LGSSM: docs/plans/artifacts/surrogate-hmc-guard-advance-lgssm-20261013-01/lgssm/mechanics/lgssm/pilot.json; docs/plans/artifacts/surrogate-hmc-guard-advance-lgssm-20261013-01/lgssm/filter-tuning/lgssm/filter-selection.json
- KSC SV: docs/plans/artifacts/surrogate-hmc-guard-advance-ksc-20261013-01/ksc/mechanics/ksc/pilot.json; docs/plans/artifacts/surrogate-hmc-guard-advance-ksc-20261013-01/ksc/filter-tuning/ksc/filter-selection.json
- Predator-prey: docs/plans/artifacts/surrogate-hmc-fresh-scope-advance-20261012-01/predator_prey/mechanics/predator_prey/pilot.json; docs/plans/artifacts/surrogate-hmc-fresh-scope-advance-20261012-01/predator_prey/filter-tuning/predator_prey/filter-selection.json
- SIR: docs/plans/artifacts/surrogate-hmc-guard-advance-sir-20261013-01/sir_d18/mechanics/sir_d18/pilot.json; docs/plans/artifacts/surrogate-hmc-guard-advance-sir-20261013-01/sir_d18/filter-tuning/sir_d18/filter-selection.json
- Range bearing: docs/plans/artifacts/surrogate-hmc-guard-advance-range-20261013-01/range_bearing/mechanics/range_bearing/pilot.json; docs/plans/artifacts/surrogate-hmc-guard-advance-range-20261013-01/range_bearing/filter-tuning/range_bearing/filter-selection.json
- Engineering results in this directory: compact-covariance-result.md, force-reuse-result.md, precision-guard-result.md and component-cost-result.md.
- Full structured values and source differences: repair-evidence-summary.json.
