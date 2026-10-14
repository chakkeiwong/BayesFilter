# Lorenz matrix crossover result — 1 October 2026

The 32 selected author regressions expose two effects: tiny differences in
Gram-matrix construction and different solver behavior on the same matrix.
TensorFlow's centered-QR endpoint agrees with the independent centered-QR
reference in every selected regression. This localizes a source-comparison
problem; it does not yet explain the invalid precision on the failed full
TensorFlow trajectory in attempt 76.

Plan: `controlled-smc-lorenz-matrix-crossover-20261001.md`. Evidence root:
`artifacts/controlled-smc-kdm-replication-20260928/`.

| Attempt | Evidence | Result | Charged seconds |
|---|---|---|---:|
| 91, c1_lorenz_matrix_reference | Original Octave matrices from captured X and Value | 32/32 captured coefficient vectors reproduced exactly; 63/63 checks | 0.615087 CPU |
| 92, c2_lorenz_matrix_crossover | Shared GPU/XLA endpoints and four crossed matrix/RHS pairs | 22/22 checks; each graph traced once; exact design and endpoint/extracted-solve equality | 6.578016 GPU |
| 93, c1_lorenz_matrix_crosssolve | Octave backslash, explicit LU and Cholesky on identical inputs | 39/39 checks; all 128 Cholesky factorizations succeed | 0.314621 CPU |

On identical original Gram matrices and right-hand sides, TensorFlow's solve
and Octave backslash agree in only 9/32 rows under the unchanged componentwise
screen, 1e-7 + 1e-9 times the reference magnitude. Explicit Octave LU and
Cholesky also agree with backslash in only 9/32 rows. Therefore matrix rounding
alone cannot explain the difference. Gram and RHS entrywise relative
differences reach only 2.59e-15 and 3.24e-15 respectively, but these matrices are
severely ill conditioned: the original design's smallest/largest singular-value
ratio ranges from 1.50e-8 to 2.41e-8. Forming normal equations squares the
2-norm condition number in exact arithmetic.

The actual TensorFlow centered-QR endpoint agrees with the independent Octave
centered-QR reference in 32/32 rows; maximum coefficient discrepancy is
8.34e-10. The extracted TensorFlow normal-equation solve reproduces the actual
raw regression endpoint exactly. These executable checks rule out a changed
diagnostic solve or design-column ordering as the explanation in this sample.

Residual sums of squares are explanatory only. Across the selected rows their
maxima are 1.151 for author backslash, 5.701 for TensorFlow raw normal equations,
and 0.016535 for both independent and TensorFlow centered QR. This is no
efficacy ranking and no full-trajectory validity result.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Matrix crossover complete | Exact source-input replay and endpoint wiring pass | No missing/nonfinite mandatory data | Octave's automatic near-singularity fallback | Test forced fallback against ordinary backslash | Full replication or a repaired adaptive filter |
| Retain centered QR as a candidate | Independent coefficient parity passes | Failed-trajectory precision veto remains | Full backward recursion and new clouds | Replay the failing trajectory after identifying source solver semantics | Default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | All diagnostic validity checks pass; previous invalid-precision veto still blocks promotion |
| Statistically supported ranking | None; deterministic local comparisons only |
| Descriptive-only differences | Coefficients, matrix rounding and residual sums of squares on 32 selected regressions |
| Default-readiness | Not established |
| Next evidence needed | Solver-branch attribution, then full healthy and failed trajectory validation |

Post-run red team: Octave may silently change the effective rank of the Gram
solve. An unchanged MATLAB source file executed in Octave is not proof of
MATLAB numerical equivalence. Testing the documented fallback can overturn the
interpretation that the author and TensorFlow are executing equivalent raw
linear solves. This is the weakest part of the current source comparison.

Checkpoint after attempt 93: 93 attempts, ceiling 96; current allocation has
129.685 CPU and 3533.844 GPU seconds left. Inherited GPU balance is
116485.144 seconds (32.357 hours). No process is running. Next action: bounded
Octave solver-fallback diagnostic; preserve all previous evidence.

## Terminal attribution: attempts 94–95

The decisive result is that the author's original backward-fitting body also
produces an invalid Gaussian precision on the frozen TensorFlow failure cloud.
The replay first reproduces all 16 healthy author policies exactly. On the
failed row's third refit, both routes first fail at zero-based time 12:
the source body in Octave gives minimum precision eigenvalue -2077.614511;
the saved TensorFlow QR policy gives -1730.163394. The first two fixed-cloud
refits remain valid under both routes. Numerical values differ, but the
negative sign is not a marginal rounding discrepancy.

Attempt 94 completes the backend attribution. Of 32 original matrices,
15 ordinary solves are marked Singular by Octave and match its forced
minimum-norm branch bit for bit. The other 17 are marked Positive Definite.
The same partition holds in all four matrix/RHS arms: 60/128 exact fallback
matches, exactly the 60 cached Singular cases. This is selective fallback,
not evidence that every original regression uses a truncated solve.

The exact Octave 6.4 source implements the condition estimate check and
least-squares fallback at dMatrix.cc:1448–1461,1530–1567,1625–1628;
DGELSD is called in the least-squares routine. See the
[official solver-selection documentation](https://docs.octave.org/v6.4.0/Techniques-Used-for-Linear-Algebra.html)
and [matrix_type documentation](https://docs.octave.org/v6.4.0/Basic-Matrix-Functions.html).
These are Octave semantics; MATLAB equivalence remains unchecked.

| Attempt | Diagnostic completion | Scientific result | Charged seconds |
|---|---|---|---:|
| 94, c1_lorenz_matrix_fallback | Original coefficient replay and finite branch outputs pass | 15/32 original solves use the confirmed minimum-norm branch; 17/32 are positive-definite solves | 0.314717 CPU |
| 95, c1_lorenz_failed_source_replay | 6/6 checks; exact equality for all 16 healthy source refits | Original source also rejects the third frozen-input refit at time 12 | 10.129952 CPU |

| Frozen TF row 0 refit | Source precision minimum | Saved TF QR precision minimum | Source result |
|---|---:|---:|---|
| 1 | 109.603680 | 109.607245 | Valid |
| 2 | 117.248242 | 117.315442 | Valid |
| 3 | -2077.614511 | -1730.163394 | Invalid at time 12 |

The claimed target is a normalizable Gaussian twisted transition with precision
P = Q^-1 + 2 A. A finite Gaussian integral on all real state space requires
symmetric positive-definite P: along an eigenvector with negative eigenvalue,
the exponent -x' P x / 2 grows quadratically, so its integral diverges. The
actual unconstrained least-squares update here gives negative P. It is wrong
relative to a valid Gaussian-policy claim at that step. The implementation's
fail-closed rejection is therefore correct.

The paper explicitly conditions feasibility on the precision constraints in
section 6.2, equations 43–44, and mentions positive-definite projection in
footnote 4 (local text lines 1178–1185 and 1219–1221). The full-quadratic author
code at cSMC_Resample.m:213–215 leaves coefficients unconstrained; only its
diagonal branch applies a constraint. Thus this evidence identifies an
unprotected implementation regime. It does not refute the controlled-SMC
construction or the reported experiment on the authors' random trajectories.

Three explanations are now separated:

- Solver/backend differences explain the matched-regression discrepancies.
- Unconstrained full-quadratic fitting can produce an invalid Gaussian policy
  even in the original source replay. Changing QR back to author backslash
  does not rescue this fixed-cloud case.
- The earlier covariance safeguard remains rejected: near-machine floors
  produced very large coefficients and failed independent full-recursion
  checks (82–84). Identifying the original failure does not validate that repair.

No Python particle loop, retracing, NumPy runtime fallback, or XLA compilation
failure was found in these checked TensorFlow endpoints. Each checked graph
traces once and numerical kernels run in FP64 GPU/XLA. Independent references
deliberately use CPU Octave/NumPy; they are not runtime candidates.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close source/backend localization | Exact healthy replay and branch attribution pass | None invalidates diagnostic evidence | MATLAB's own backend was not run | Preserve Octave-specific qualification | Exact MATLAB reproduction |
| Reject unprotected failed-cloud refit | Source and TF have negative precision at the same time | Gaussian integrability fails | Frequency over independent complete trajectories | Calibrate a bounded full-covariance safeguard | Rejection of controlled SMC theory |
| Keep full empirical launch blocked | No independently validated safeguard yet | Earlier repairs remain rejected | A stable constrained update and its downstream behavior | Validate local arithmetic, full recursion, fresh seeds, then batching/cost | Figure 6 or either full replication completed |

| Inference status | Terminal finding |
|---|---|
| Hard veto screen | Negative precision reproduced by original source, with exact healthy controls |
| Statistically supported ranking | None |
| Descriptive-only differences | Eigenvalue magnitudes, coefficient errors and residuals on frozen inputs |
| Default-readiness | Unprotected failed refit and prior safeguards are not ready |
| Next evidence needed | Calibrated valid correction, independent full-recursion agreement, healthy non-harm and untouched-seed conditional filtering comparisons |

Terminal self-review: the strongest alternative explanation is that fixed
TensorFlow clouds and incoming policies differ from a complete native random
trajectory. That limits claims about failure frequency and published results,
but cannot make the observed source-produced negative precision integrable.
The harness's exact 16-fit healthy replay rules out the extraction as the
explanation for all differences. No numerical policy or production default
changed in this stage. No job remains running.

Next phase is implementation and calibration of a bounded SPD safeguard,
following the sequence added to the plan. It is not yet an executable repair.
All five diagnostic phases 91–95 are complete; neither full replication is.
