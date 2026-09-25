# Baseline execution and Phase 0D refresh

The seven Gaussian baseline consumers execute through the actual master CLI:
Kalman, UKF, prior SIS, adapted SIS, bootstrap SIR, adapted SIR, and canonical
LEDH. The last route uses the shared canonical executor, Contract E, both
higher-moment corrections, and analytical derivatives of all six model
parameters, including the initial law. This closes the small engineering
baseline, not the scientific comparison.

The protected execution checkout is
`/tmp/bayesfilter-younis-score-execution-20260914`, initially based on
`e7f2a88ecff49ced481b9615c0b1237b8cabe732`. Main-checkout canonical development
was concurrent; an early attempt encountered its temporary syntax error.
Numerical runs were moved to the isolated checkout. Each run's manifest binds
the numerical source closure, command, settings, seeds, environment, and timing.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close the 0C engineering baseline | All seven actual consumers complete | Finite values, score shape/dtype, analytical parity and reset validity pass | One tiny Gaussian scope | Build KDM consumers in 0D | Better model-score estimation |
| Retain scoped tuning machinery | Issued selection is consumed by a separate claim-plumbing run | Altered controls, N/T/dtype mismatch, leaked data and mechanics promotion rejected | Only mechanics selection tested | Extend to powered study criteria | Tuned numerical defaults |
| Retain resumable execution | Deliberate interruption resumes to seven completed rows in eight attempts | Failure remains in history; completed rows reused | Hard subprocess timeout remains to implement before long studies | Add watchdog before a serious sweep | Unlimited-runtime reliability |
| Retain XLA validity repair | Injected invalid final reset rejects; healthy parity unchanged | Invalidity reaches returned value/score even if XLA removes Assert | Other primitives retain their own checks | Keep shared repair in isolated revision | All possible numerical failures covered |

Current-source artifacts: `baseline-cpu-03` (7 rows, 19.106 s),
`baseline-gpu-02` (7 rows, 24.352 s), `tuning-cpu-02` (8 rows, 22.143 s),
`phase-0c-selection-02.json`, `claim-plumbing-cpu-02` (1 row, 17.022 s), and
`interrupt-cpu-01` (7 rows, 8 attempts, 25.212 s). CPU runs deliberately hide
GPU devices and use float64/XLA. GPU baseline uses float32/TF32/XLA on the
RTX 4080 SUPER selected by UUID, with verified memory growth. The original
primitive GPU probe selected an RTX 5080; it is identified correctly in its
own manifest. Three framework GPU launches have been used.

Checks: coordinator 13 passed; Gaussian plus coordinator 22 passed; full
canonical initial-law/invalid-reset checks 2 passed; reporting 3 passed;
deterministic FD 7 passed. `tuning-consumption-checks-01/checks.json` records
the seven positive/negative selection checks. These overlapping test runs
must not be summed as independent tests. Full logs are preserved here.

| Inference status | Finding |
|---|---|
| Hard veto screen | Early syntax and non-XLA empty-output failures were implementation/environment failures; invalid-reset propagation was repaired. Current tiny rows pass. |
| Statistically supported ranking | None. |
| Descriptive-only differences | The Gaussian UKF equals the exact Kalman reference to numerical precision; particle errors on this fixture are illustrative only. |
| Default readiness | Not established for any candidate or numerical control. |
| Next evidence needed | Nonlinear oracle-bearing cases, scope-specific calibration, independent datasets/particle replications, and paired uncertainty. |

The strongest alternative explanation for any apparent particle-method
difference is Monte Carlo fluctuation on a single observation dataset. An exact
Gaussian reference is a correctness test, not a hurdle requiring a particle
method to improve on an exact algorithm. Gaussian performance cannot select a
nonlinear default. This is the weakest part of the current scientific evidence.

## Next bounded implementation tranche

Phases 0D--0G remain within the existing 12 CPU-process-hour, 8 GPU-device-hour,
12-GPU-launch engineering allocation (2 GPU hours reserved for repair). No
large scientific Cartesian grid is launched under this refresh. The next
slice permits at most 120 tiny method/fixture rows, 3 repair attempts per
distinct failure before reassessment, and at most 3 further GPU integration
launches. Each numerical command remains capped by a harness or process
timeout. Stop this slice on source corruption, an unresolved mathematical
target mismatch, unavailable required density/normalizer, or budget exhaustion;
a candidate failure alone triggers its planned repair.

The immediate question is whether the integrated and resampling KDM consumers
can execute the full shared canonical recurrence, preserving their declared
finite targets and total analytical tangents. Repair `return_trace`, observation
factor callbacks and post-reset callbacks inside the existing TensorFlow time
loop. Carry trace tensors through the loop; do not collect loop tensors in
Python lists. Baselines are the unchanged canonical value/score and exact
frozen-mixture Gaussian identities. Zero-bandwidth atom recovery, positive
bandwidth total derivatives, anchor/replay equality, and density/sample-law
checks are pass criteria. Runtime and ESS are explanatory only. IWSG conditional
unbiasedness does not establish an unbiased marginal model score.

Defaults audit: one annealing stage and the existing tiny Contract-E controls
are mechanics fixtures, not defaults; full initialization derivatives are
mandatory; bandwidth is explicit and differentiated; fixed streams are paired
only when the two methods' laws permit it; signed quadrature weights cannot be
used as probabilities. Exact control variates require a justified center and
coefficient independence. Biased-estimator combinations require separate oracle
calibration and untouched evaluation. Missing centering blocks the exact-CV
claim, not a clearly labeled calibrated combination.

Skeptical audit passes for this engineering slice: actual canonical consumers
are the comparators, output target labels remain distinct, source closure is
isolated, initial-law and bandwidth terms are explicit, and no smoke metric
can promote a scientific candidate. A successful command alone will not close
0D; its consumer identities and negative checks must pass first.
