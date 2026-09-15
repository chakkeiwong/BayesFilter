# Complete SGQF–TT run: reference screens pass; heuristic comparison blocks promotion

The complete master executed SGQF, observation-informed coordinates, adjacent-state TT fitting, retained marginalization, conditional KR sampling, original-model particle correction, reference comparisons and frozen analytical scores for d=1,d=4 and all 20 observations. **The guided TT had descriptive conditional losses to simple proposals, so this configuration is not promoted.** All five proposals passed the predeclared exploratory reference screens. SGQF passed the requested rough-guide check.

Master: `docs/benchmarks/run_observation_aware_tt_complete.py`. Frozen plan: `docs/plans/observation-aware-tt-repair-complete-program-20260913.md`. Run root: `docs/benchmarks/artifacts/observation_aware_tt_complete_20260913/campaign-01/`; adjacent log: `campaign-01.log`. Exact commands, source hashes, configuration, environment and device details are in `run_manifest.json`. Section `sec:executed-sgqf-tt` of the updated attempt05 note contains the derivation and proofs.

## Design and results

The frozen C2 data uses model seed 52, observation seed 42, beta=.4, sigma=1 and the original scalar/correlated transition matrices; the JSON records source provenance. The TensorFlow implementation checks stationary covariance and starts with X0~N(0,P0), then y0. Each candidate uses 512 particles and four particle seeds (1101–1104, mapped to sampler keys in the driver). Each reference uses four independent 32768-particle runs. The scalar grid is refined from 801 to 1201 points over ±12 stationary standard deviations.

Degree 3, rank 3, four sweeps, 1024 independent rows per training/validation/audit split, L1 grid {0,1e-5,1e-3}, and 5% joint defensive Gaussian mass were fixed before execution. Validation selects L1; audit rows and heuristic comparisons do not tune anything. This construction is an extension of the author route, not author TT-cross. The SGQF Gaussian, retained TT density and corrected particle bank have separate roles and all carry past observations through time.

| d | Guided TT log evidence | MCSE | Reference log evidence | Reference MCSE | Maximum SGQF mean gap |
|---|---:|---:|---:|---:|---:|
| 1 | -27.50838495 | .03138108 | -27.54108213 | scalar grid | .02715048 |
| 4 | -66.81236915 | .06049544 | -66.73251140 | .02603993 | .02915635 |

All methods passed log-evidence and filtering-mean screens. Tolerances are 3.182446 combined MCSE + .15 for log evidence, and the analogous coordinate-wise MCSE allowance + .15 sigma for means. These are exploratory tolerances, not simultaneous confidence guarantees. MCSE describes replication variation of the log estimate; it does not remove finite-particle bias. SGQF gaps against the larger particle reference pass its one-sigma rough-guide bound; observation and history perturbations changed its estimates in both dimensions. The d4 reference uses independent draws but shares the particle-update kernel with the candidates; it cannot exclude a shared kernel defect. The scalar grid is a separate implementation.

Conditional RMSE uses the four-run ensemble filtering mean relative to the reference, pooled only within the stated regime. Near zero is max|y|/beta<=.5; large is max|y|/beta>=2. Regime counts (near-zero,ordinary,large) are (4,11,5) for d=1 and (0,7,13) for d=4.

| d | Proposal | Near-zero RMSE | Ordinary RMSE | Large RMSE |
|---|---|---:|---:|---:|
| 1 | Transition | .03343 | .02850 | .04493 |
| 1 | Stationary prior | .04827 | .02943 | .04817 |
| 1 | SGQF Gaussian | .03304 | .24130 | .01871 |
| 1 | Predictive TT | .02057 | .02147 | .02506 |
| 1 | Guided TT | .02650 | .02910 | .02099 |
| 4 | Transition | not observed | .04999 | .04483 |
| 4 | Stationary prior | not observed | .07736 | .09299 |
| 4 | SGQF Gaussian | not observed | .03866 | .04986 |
| 4 | Predictive TT | not observed | .05692 | .04572 |
| 4 | Guided TT | not observed | .03866 | .05116 |

These differences are descriptive. The frozen rule conservatively vetoes promotion on an observed conditional loss to any constructed simple proposal; it does not establish statistical significance. Guided TT has such losses in two scalar regimes and on large four-dimensional observations. No ranking is statistically established. Four-dimensional near-zero behavior is not checked because that regime did not occur.

## Verification and execution

Ten focused tests passed, covering nonuniform multivariate density integration, incomplete-Hermite-CDF derivative, suffix conditioning and upper ordering, retained marginal integration, observation/history response and executable wiring to the shared KR implementation. Log: `docs/benchmarks/artifacts/observation_aware_tt_complete_20260913/focused-tests-20260914.log`. The full three-step CPU/non-XLA and GPU/XLA smokes agree on 48 numerical decision values within 1.73e-15 (`gpu_cpu_parity.json` beside the run root).

Full-sequence analytical-score relative finite-difference errors are 5.1599e-10 (d=1) and 4.2184e-10 (d=4), for diagonal transition shift, log beta and log sigma. The target is the recorded finite particle log-evidence recursion with proposals, states and ancestry fixed. This does not differentiate retraining or validate an exact-likelihood/HMC score.

The call-chain review is `docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/implementation-status-audit.md`. All three executed Python files and the frozen plan match their launch hashes. The old scalar full-master PASS labels are withdrawn as correctness evidence because of wrong initialization, bounded support, dependent innovations and missing retained-TT propagation. Their files remain historical evidence.

MathDevMCP reports and document review are stored in `docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/mathdev-complete-20260914/`. Four bounded symbolic identities are verified equivalent: conditional Bayes/Jacobian cancellation, the conditional mixture density, centered covariance algebra and importance-density cancellation. The scoped density-to-code check matches every supplied term but explicitly does not certify function semantics. All four proposition audits are inconclusive: one statement yields no extractable obligation, and the remaining integral/gradient notation exceeds the bounded backend. There is no full MathDevMCP proof certificate. The note provides explicit proofs and the independent numerical tests provide implementation evidence; neither is silently upgraded to formal verification.

The RTX5080 GPU/XLA campaign completed in 99.574 seconds, with verified memory growth and TensorFlow allocator peak 39,741,440 bytes. Environment: `/home/chakwong/anaconda3/envs/tftwogpu`, TensorFlow 2.20.0-dev0+selfbuilt, float64. Repeated fitting, KR and particle-update kernels use stable signatures and XLA. SGQF rule construction, chart setup, target preparation and independent references are explicit setup/diagnostic exceptions outside those kernels. This is not an end-to-end XLA or performance claim.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept tested mechanics | Density/CDF/marginal/wiring and score checks pass | No observed numerical or harness failure | Finite test coverage | Retain executable/evidence | Correctness for arbitrary TT configurations |
| Accept tested SGQF guide | Rough-reference and history-response screens pass | No guide validity failure | One sequence; Gaussian closure | Keep optional guide | Exact filtering or retention of all higher moments |
| Withhold TT promotion | Reference screens pass | Conditional descriptive heuristic losses | Four seeds; one sequence | Disjoint calibration and untouched replicated validation | Superiority, default readiness or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No observed density, finite-value, guide, score or reference-screen failure; heuristic promotion veto active. |
| Statistically supported ranking | None. |
| Descriptive-only differences | All method rankings, conditional RMSE, MCSE differences and timing. |
| Default-readiness | Not established; false in decision objects. |
| Next evidence needed | Independent observation sequences, more paired particle replications, separately calibrated capacity choices, higher-dimensional checks and separate exact-score validation. |

The promotion failure does not invalidate the target, proposal identity, implementation or reference; it rejects promotion of this configuration. All planned numerical stages and comparator arms completed. No further retry is justified as infrastructure repair. One of three allowed full launches used 99.574 of 2400 numerical seconds. Unused time does not authorize tuning on the holdout comparison.

Post-run skeptical review: ordinary particle variation is the strongest alternative explanation for the conditional losses; limited TT capacity and observed fit residuals are also plausible. A replicated comparison on independent sequences could overturn the descriptive conclusion. Comparative benefit and higher-dimensional extrapolation are the weakest parts of the evidence. Passing broad reference screens, correct importance factors and normalized proposals does not establish small Monte Carlo error or bounded importance-weight variance.
