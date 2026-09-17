# LEDH graceful-failure execution result - 2026-09-18

The shared failure-containment repairs pass the exercised engineering checks. **LG1 did not pass calibration in any of the three permitted substantive attempts.** The campaign therefore stops at the master program's attempt ceiling, with posterior validation and expansion to the other model scopes blocked. This rejects the tested configurations; it does not reject the LEDH research direction.

The final focused suite passed **61 tests**. The frozen healthy full-feature fixture retains exactly the original value `-4.4698582934543625` and analytical score `[-0.39633162869729477]`. A separate combined run had 141 passes, one skip, and nine failures/setup errors caused by obsolete fixtures; the repaired cases all passed in the final focused suite. These are engineering results, not posterior or default-readiness evidence.

Work is isolated on branch `ledh-graceful-failure-execution-20260917` in `/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`, starting from `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`. The other checkout was not edited. The governing [master program](ledh-graceful-failure-revised-program-2026-09-17.md) remains unchanged; its SHA-256 is `3b1725f1f2114e764d306b9090e80c05411f36f590c6abc942f58f14048d92de`. All 25 source entries in the reviewed starting manifest matched before execution.

The [machine-readable decision](artifacts/ledh-graceful-failure-20260917T105313Z-01/decision.json), [campaign ledger](artifacts/ledh-graceful-failure-20260917T105313Z-01/campaign.json), and [run accounting](artifacts/ledh-graceful-failure-20260917T105313Z-01/run-accounting.json) preserve the individual decisions, commands, failed runs, and charges. Investigation accounting is complete at the declared stop condition. The scientific validation objective is not achieved.

## Implemented behavior and executable evidence

The common numerical guard checks the entire represented covariance, relative symmetry, factor finiteness, and positive factor diagonal while preserving leading batch dimensions. Rejected factors become finite inactive placeholders; their analytical tangents are zero. A failed tangent on a valid factor remains distinguishable from an invalid value. Healthy small-scale matrices are not rejected merely for having small absolute entries.

The canonical recursion carries sticky, separate value and force status across observations. Both batch wrappers call the shared recursion, preserve `-Infinity` for value failure, and return zero with an explicit flag for invalid force. They report the first failure's time, stage, and available particle index. The dual-parameter target preserves a valid exact finite-program value even when its separately configured force fails. Nested annealing uses a traced TensorFlow loop and now runs through the same path in both wrappers.

Analytical dependencies now include parameter-dependent initial moments, process and observation covariances, observations and Jacobians, and stationary initialization. The stationary covariance adapter checks a sufficient stability condition separately from positive definiteness and the Lyapunov residual. Its operator-norm stability check is conservative; it is not a general spectral-radius admission theorem.

The HMC consumer accepts typed value/force status. Numerical failures and true prior-support rejections remain distinct; rejected endpoints preserve the accepted state and cached potential. A failed force can use the existing deterministic zero extension with explicit telemetry. Tests check actual LEDH endpoints, momentum-reversed proposals, endpoint energy correction, cache preservation, invalid initialization, and archived status. Such mechanics do not bound the posterior mass excluded by numerical failures.

The public TensorFlow tuner now retains endpoint diagnostics from adaptation, candidate verification, and executed repairs. Old artifacts without that trace remain readable without invented telemetry. The registered public binding ran against real LG1 and returned finite state and diagnostic health, while correctly reporting `handoff_eligible=false` and rejecting construction of a retained sampling runner. Its current non-XLA diagnostic mode is an explicit exception, not GPU/XLA admission evidence.

| Check | Result and scope | Evidence |
|---|---|---|
| Healthy equivalence | Exact value/score equality on the saved current-source full-feature fixture; all invalid-factor assertions passed | [final comparison](artifacts/ledh-graceful-failure-20260917T105313Z-01/final-baseline-cpu-01/comparison.json) |
| Actual batch and dual consumers | GPU/XLA float32 with TF32 and float64 reference checks passed for mixed valid, value-invalid, and force-invalid rows | [GPU results](artifacts/ledh-graceful-failure-20260917T105313Z-01/gpu-validation.json) |
| First failure propagation | 144 injected cases across eight stages, three observation times, value/force modes, and three consumers passed | [stage-test log](artifacts/ledh-graceful-failure-20260917T105313Z-01/stage-propagation-cpu-01/command.log) |
| Matrix guard families | CPU and trusted GPU checks cover scale, conditioning, rotation, cancellation, asymmetric/nonfinite inputs, and leading batch dimensions | [strict CPU export](artifacts/ledh-graceful-failure-20260917T105313Z-01/strict-json/matrix-families-cpu-01/matrix-families.json), [strict GPU export](artifacts/ledh-graceful-failure-20260917T105313Z-01/strict-json/matrix-families-gpu-01/matrix-families.json) |
| Analytical score | Independent dependency tests and LG1 finite-difference ladders passed for the tested baseline scopes, including stationary initialization and prior/chart terms | [LG1 diagnostic](artifacts/ledh-graceful-failure-20260917T105313Z-01/lg1-calibration-diagnostic-01/diagnostic.json), [dependency log](artifacts/ledh-graceful-failure-20260917T105313Z-01/dependencies-cpu-01/command.log) |
| Exact block transport | `N=4096,K=2048` executes the prescribed 2-by-2 geometry; one-block parity, row masses, and float32/64 derivative ladders passed | [block checks](artifacts/ledh-graceful-failure-20260917T105313Z-01/block-transport-gpu-04/block-validation.json) |
| Real HMC and public API | Final 61-test suite passed; separate telemetry tests passed; real public LG1 binding remained diagnostic-only | [final test log](artifacts/ledh-graceful-failure-20260917T105313Z-01/corrected-regressions-cpu-01/command.log), [public binding](artifacts/ledh-graceful-failure-20260917T105313Z-01/public-tuner-cpu-02/public-binding.json) |

Stage injection proves propagation from the producer/status boundary, not that every possible numerical fault in each producer has been generated. Natural failures are also tested through matrix corruption, invalid covariances, nonfinite callbacks, parameter directions, and chart boundaries. The [endpoint inventory](artifacts/ledh-graceful-failure-20260917T105313Z-01/endpoints.md) identifies both executable coverage and untested boundaries. The operation inventory is static discovery, not proof that every external/native consumer is covered.

The exact-block implementation still returns full transport matrices and their tangents. Its output memory is quadratic. The row-mapped canonical wrappers also do not qualify as batch-native NeuTra training. Neither limitation is hidden by the new guards.

## LG1 calibration and the reason for stopping

The target is the declared stationary LG1 model: `F=phi`, `Q=1`, `H=1`, `R=0.5`, `T=10`, and a uniform physical prior on `(-0.95,0.95)`, with transition before observation. The executed quantity is the likelihood from the fixed finite LEDH program, together with its analytical derivative. The comparator is the exact Kalman likelihood and filtering moments for the same represented observations. These quantities are different; the Kalman comparisons measure their discrepancy, while finite differences test differentiation of the finite LEDH quantity itself.

Four calibration, two validation, and three claim datasets were generated using disjoint seeded streams and three CPU workers. Candidate nomination used calibration replicates 0 and 2. Validation and claim observations were not evaluated or used for selection. Passing the nomination screen would still have required the complete calibration design, scope-specific derivative checks, untouched validation, and valid tuning authority.

The unchanged screens require worst-time whitened mean and covariance errors at most `0.05` and centered log-likelihood-ratio discrepancy at most `0.02`, with valid value/force status. A candidate must satisfy every screened calibration group; an isolated passing dataset is insufficient.

| Substantive attempt | Frozen candidate family | Candidate configurations / dataset groups | Candidates passing all screened groups |
|---|---|---:|---:|
| 1 | Initial control curves, including the repaired `N=4096` exact-block continuation | 21 / 42 | 0 |
| 2 | Fixed Halton designs, particle counts 256/1024/4096 and declared flow variants | 9 / 18 | 0 |
| 3 | Same fixed-design ladder with the separately calibrated positive coordinate cap of 16 | 9 / 18 | 0 |

Full per-configuration maxima and identities are in the [calibration summary](artifacts/ledh-graceful-failure-20260917T105313Z-01/calibration-summary.json). For illustration, these final-attempt `N=4096` results show why the screen did not pass:

| Flow steps | Calibration replicate | Maximum mean error | Maximum covariance error | Centered log-ratio error | Screen |
|---:|---:|---:|---:|---:|---|
| 3 | 0 | 0.01121 | 0.01977 | 0.01023 | Pass |
| 3 | 2 | 0.09167 | 0.09367 | 0.13001 | Fail |
| 1 | 0 | 0.01783 | 0.02761 | 0.00375 | Pass |
| 1 | 2 | 0.21823 | 0.36824 | 0.06071 | Fail |

These are descriptive errors and hard screening decisions, not statistically supported rankings. All final-attempt values and forces were numerically valid. The failure is approximation of the reference, not a crash or sentinel failure.

A separate safety evaluation exposed an unjustified inherited choice: coordinate cap `0.98` moved a healthy standardized Gaussian cloud by as much as `2.3723` standard deviations and changed kurtosis from `2.9933` to `1.5879`, despite restoring the first two moments. It fails that non-harm question. The smallest tested positive cap passing the predeclared `1e-4` healthy-displacement criterion was 16, with displacement `4.53e-6` and finite, flagged stress behavior. This is narrow primitive evidence, not justification for a universal default. The campaign used it only in attempt 3 and did not change the global default. See the [protocol](artifacts/ledh-graceful-failure-20260917T105313Z-01/ledh_cap_protocol.md) and [measurements](artifacts/ledh-graceful-failure-20260917T105313Z-01/coordinate-cap-nonharm-01/cap-calibration.json).

An independent scalar calculation agreed with the flow coefficients, determinant, Gaussian transition/observation densities, and UKF covariance to scaled error below `6.91e-7` on the tested cases. The conditional mixture calculation still showed substantial finite-cloud discrepancy. This weakens the hypothesis of a simple scalar-flow algebra error; it does not prove every reset/proposal operation correct or identify the sole cause. See [localization](artifacts/ledh-graceful-failure-20260917T105313Z-01/lg1-localization-01/localization.json).

Master section 12.4 requires stopping the affected action at its attempt ceiling. The [attempt-3 protocol](artifacts/ledh-graceful-failure-20260917T105313Z-01/lg1-attempt3-protocol.md) predeclared exactly this stop if no scope passed. Remaining GPU time does not authorize a fourth scientific candidate. Independent engineering closure continued after that stop.

## Scope decisions and remaining work

| Required scope | Evidence-backed status | Concrete next requirement |
|---|---|---|
| LG1 | Engineering checks pass in their stated scope; all three calibration attempts fail; no posterior admission | A revised bounded repair program must first discriminate proposal-cloud quadrature and reset shape/coupling effects, then pass calibration and untouched validation before tuning authority or posterior execution |
| LG2 | Exact specification and generic linear parameter adapter exist; this full scope is not validated | Pass the LG1 prerequisite, then its own `T=20` derivative/calibration/tuning/reference checks |
| LG4 | Revised partially observed specification and adapter exist; this full scope is not validated | Its own `d=4,T=50` checks; do not reuse the rejected earlier LG4 identification claim |
| LG20 | Specification and adapter exist; capacity, filtering, and posterior work are not checked | Its own `d=20,T=120` feasibility forecast, references, calibration, and tuning |
| SV20 | Actual state-dependent-variance proposal conformance remains unresolved | Establish a canonical proposal for the zero-mean observation density; do not substitute a nonzero observation mean |
| SV40 | Same capability gap, plus a separate horizon/data/tuning scope | Resolve the shared proposal question and validate independently at `T=40` |
| PP20 | Corrected RK4 parameter/tangent adapter and model-domain validation remain unimplemented | Implement and check the declared two-parameter problem and its reference before inference |
| SIR3 | Fixed-parameter three-compartment state-filtering adapter remains unimplemented | Validate that declared state-filtering scope; no parameter-posterior claim applies |

The LG1-first dependency comes from the adopted master closeout sequence and Claude's reviewed recommendation. The dependent scopes above were not silently counted as passed, and no scope inherited another scope's tuning.

Several requirements remain independently necessary for promotion: a repository-issued LEDH tuning artifact bound to the actual scope; an eligible TensorFlow HMC admission route with its required convergence checks; complete source/native dependency identity; numerical-exclusion mass assessment; conditional heuristic comparisons; finite-target and model-reference posterior comparisons; and declared repeated capacity/timing evidence. The existing TensorFlow public tuner is diagnostic-only, while the legacy admitting tuner retains runtime NumPy and is not an eligible fallback under current policy. The present fixture hashes are provenance, not authority issued by a canonical admission factory.

The new driver is a bounded diagnostic launcher with explicit CPU/GPU choice, commands, attempt limits, logs, and resource accounting. It is not yet the complete admission driver proposed in master section 13.1: it does not enforce repository-issued route/tuning prerequisites for arbitrary commands. It must not be used as evidence that those missing scientific prerequisites were satisfied.

The inherited Sinkhorn cost floor, normalization safeguards, correction floor, ridge, damping, and trust controls still need their complete scope-specific justification and activation/non-harm evidence before promotion. The coordinate-cap result cannot justify these other controls. The final cap-16 configurations did not earn full downstream derivative/admission qualification because they failed earlier calibration.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain the engineering repairs for review | Actual containment, healthy equivalence, and consumer checks pass | No remaining failure in the exercised final checks | Coverage is finite; some native/external paths are not audited | Review the isolated change and preserve the focused regressions | Global numerical safety, posterior validity, or merge readiness |
| Reject the tested LG1 configurations | No configuration passes all calibration groups | Approximation screen fails; three-attempt ceiling reached | Proposal quadrature and reset/correction shape effects are not fully separated | Revise the bounded LG1 investigation before another substantive attempt | Rejection of LEDH as a method |
| Retain cap 16 as a safety hypothesis | Narrow healthy-displacement/stress screen passes | Downstream LG1 calibration still fails | Different dimensions/scopes may need different calibration | Evaluate only within an explicitly revised scope | Universal numerical default |
| Withhold HMC/posterior admission | No eligible tuning handoff or posterior comparison | Missing authorities, exclusion bounds, convergence and reference checks | Excluded mass and posterior discrepancy remain unknown | Repair the prerequisite chain after LG1 calibration | Correct posterior sampling or scientific superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | All tested LG1 candidates fail calibration; later evidence gates are missing |
| Statistically supported ranking | None; no candidate ranking is supported |
| Descriptive-only differences | Per-dataset errors, cap movements, and measured runtime differences |
| Default readiness | Not established; production direction remains GPU/TF32/XLA, with validation pending |
| Next evidence needed | A passing calibrated scope, untouched validation, proper tuning authority, numerical-exclusion bound, conditional heuristics, and multi-chain posterior/reference evidence |

## Resource and artifact accounting

There are 37 recorded runs, including failed checks and infrastructure repairs. Charged use is **1,309.3105 GPU seconds (0.364 hours)** and **3,187.8432 aggregate CPU-worker seconds (0.886 hours)**, within the 64/16-hour ceiling. CPU charge includes the conservative 900-second reserve for the earlier unmetered readiness build; measured child-process CPU time is 2,287.8432 seconds. GPU charge includes whole command wall time, including compilation. Routine static file inspection is not individually metered. Exactly three substantive LG1 attempts were consumed.

GPU runs used trusted access to `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, verified TensorFlow memory growth, and recorded TF32/XLA/dtype settings. CPU checks intentionally hid GPUs. No environment packages were installed. The ignored native library's checksum and changed Python syntax checks are preserved in the [source inventory](artifacts/ledh-graceful-failure-20260917T105313Z-01/final-source-inventory.json).

Failed commands remain in the [run accounting](artifacts/ledh-graceful-failure-20260917T105313Z-01/run-accounting.json), with fresh directories for retries. One attempted broader test collection exposed pre-existing `tests/highdim/test_genut_shape_lm_tf.py` imports of the deleted `cubature_genut_batch_tf` module. That historical fork was not recreated. Active shared-correction regressions passed, but the full repository suite is not claimed clean.

Eight raw diagnostic JSON files contained 70 nonstandard nonfinite tokens. Their originals remain preserved; [strict exports and checksums](artifacts/ledh-graceful-failure-20260917T105313Z-01/strict-json/index.json) encode each as `{"__nonfinite_float__":"NaN|Infinity|-Infinity"}`, with one of the three literal tags. The affected writers now emit standards-compliant JSON. Original run manifests identify the executable hashes used at launch; final-source hashes identify subsequent engineering repairs.

## Terminal skeptical review

The strongest alternative explanation for the calibration failure is that the finite proposal/reset cloud and its dependence structure are inadequate for these observations, rather than a failure of LEDH in principle. Agreement of low-order moments or of local flow algebra cannot resolve that explanation. The narrow cap test exposed one harmful setting but did not cure the final discrepancy.

A fresh, properly bounded candidate that passes complete calibration and untouched validation, with independently checked references and the required tuning authority, would overturn the present unready verdict for that scope. Posterior/default promotion would still require the later evidence gates. The weakest evidence here is the limited two-dataset nomination screen and the absence of prior-wide exclusion and posterior comparisons; no conclusion about those quantities is drawn from a finite score or healthy short HMC run.

This closeout is the executor's source/evidence review. No independent terminal Claude review was performed. The existing material plan review remains available, but it is not being represented as a review of these implementation results. The next substantive experiment requires a revised LG1 repair program because the authorized attempt allowance is exhausted.
