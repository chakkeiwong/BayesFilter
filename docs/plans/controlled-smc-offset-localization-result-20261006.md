# Controlled-SMC regression discrepancy: result, 6 October 2026

The approximately 1.1% likelihood discrepancy at Lorenz F=4.8801,
seed 20260930 is localized to the fitted policies at the 14 time steps where
the author runtime takes its SVD fallback. Restoring the author policies at
those times reduces the QR likelihood discrepancy from 1.09928% to 0.00349%,
removing 99.68% of the log-likelihood gap in this case. These are diagnostic
policy substitutions, not a repaired autonomous TensorFlow implementation.
Both replications remain incomplete.

The [plan](controlled-smc-offset-localization-20261006.md) and
[preceding perturbation result](controlled-smc-perturbation-result-20261006.md)
define the question and controls. No acceptance margin or numerical default
was relaxed, and no author method was changed.

## What the checked code computes

The author fits a quadratic design D to the backward target v by computing
G = D' D, h = D' v, and applying ordinary backslash to G beta = h
(.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz/cSMC_Resample.m:185–231,
particularly line 193). This implements the least-squares step of controlled
SMC, published paper Section 4.2, in the Lorenz quadratic family of Section 6.2.
The backward target includes the next fitted policy's Gaussian correction.

The declared Octave baseline conditionally selects Cholesky, LU, or DGELSD.
The inspected runtime and linked-library anchors are preserved in the
[solver result](controlled-smc-p0-solver-semantics-result-20261005.md).
The current TensorFlow source_normal_equations endpoint instead always calls
tf.linalg.solve on its Gram matrix; centered_qr solves the centered, scaled
design before transforming the coefficients back
(bayesfilter/replication/multivariate_controlled_smc_tf.py:312–351).
Neither endpoint implements Octave's conditional solver semantics.

For a full-rank design, all these formulas specify the same least-squares
solution in exact arithmetic. In the 2-norm, kappa(D' D) = kappa(D)^2, since
the Gram eigenvalues are the squared singular values of D. Forming and solving
the Gram system therefore exposes this case to much greater rounding error.
At time 28, the raw design condition is 7.20e7, its centered-design condition
is 7.97, and the source reciprocal condition estimate of the formed Gram
matrix is 7.81e-17. Finite-arithmetic equality between the solver routes is
wrong for these inspected inputs; equality of their exact-arithmetic
least-squares target does not remove the observed discrepancy.

## Controls and direct regression evidence

The saved stage-zero particle clouds agree to 7.11e-15 and all ancestors
match. Instrumenting the source to retain regression inputs leaves the
entire cSMC result and both final random streams bitwise unchanged. The trace
contains 101 regressions: 87 positive-definite solves and 14 SVD fallbacks at
times 16–28 inclusive and 86 (zero-based). All SVD solves retain rank 45.
There is no observed rank truncation to emulate in this case.

An independent adapter to Octave's actual system LAPACK reproduces all 101
coefficient vectors bit for bit and all branch decisions. Its shared library
is intentional for checking runtime arithmetic; this is not an independent
proof of least-squares accuracy. The GPU run then calls each actual TF
regression endpoint on those same source targets, before recursive target
differences can enter.

| Time-28 diagnostic | Author | TF raw Gram solve | TF centered QR |
|---|---:|---:|---:|
| RMS residual on the same source target | 0.0843063 | 0.0336513 | 0.00705612 |
| Maximum prediction difference from author | 0 | 0.418663 | 0.451600 |

Across all 101 targets, TF centered-QR predictions agree with an independent
centered-QR reference to 4.72e-11. Thus a translation error in that QR fit is
not supported by these checks. Its smaller regression residual does not
establish a more accurate likelihood, superior filter, or completed
replication. The source remains the declared comparator.

## Downstream policy intervention

All forwards use the same stage-one normal and uniform tapes, supplied data,
d=8, N=512, T=100, float64, and observation-normalization convention. The
source-policy control agrees in final log likelihood to 8.08e-11 with zero
ancestor mismatches. The saved conditional-QR forward is reproduced within
the predeclared 1e-6 control bound. Every diagnostic forward is valid.
Percentage differences are 100 * expm1(port logZ - author logZ).

| Policy supplied to the unchanged TF forward | Delta log likelihood | Likelihood difference | Ancestor mismatches |
|---|---:|---:|---:|
| Author policy control | -8.07e-11 | -8.07e-9% | 0 |
| Conditional QR policy | 0.0109328244 | +1.099281% | 2177 |
| Author policy, QR substituted at 14 SVD times | 0.0109031630 | +1.096282% | 2175 |
| QR policy, author substituted at 14 SVD times | 0.0000348520 | +0.003485% | 2 |
| Author policy, QR substituted at time 28 only | 0.0023274511 | +0.233016% | 428 |
| QR policy, author substituted at time 28 only | 0.0083837000 | +0.841894% | 1807 |

The reciprocal substitutions locate the relevant policy times; their effects
are not additive because particle trajectories and resampling change. The
substituted QR policies come from the complete backward fit, so the table
does not assign all error at each time to its own local solve in isolation.
The direct identical-target regressions independently demonstrate the local
solver discrepancy. The remaining 0.003485% lies below this case's earlier
0.0223% median author perturbation response, but the hybrid is not an
autonomous algorithm and this observation cannot qualify the port.

## Execution and review

CPU diagnostic stages offset-localization-20261006-01/02/03 used 4.730780
seconds in total, with CUDA deliberately hidden and NumPy restricted to
saved-data inspection and independent references. Their manifests retain
commands, input/output hashes, base commit, seeds and reference identity.

Attempt156_p0_offset_localization completed with exit 0, charging 24.227988
GPU wall seconds. Its GPU was the RTX 4080 SUPER selected by model name;
TensorFlow enumerated it as GPU:0. Memory growth was configured before
initialization; float64, TF32 off and XLA were recorded. Each regression
endpoint traced once; the forward endpoint traced once for all six policies.
Peak TensorFlow allocator usage was 52,579,072 bytes. The diagnostic driver
uses Python iteration to submit independent checks; the repeated numerical
regression and forward kernels execute in their existing stable TF graphs.
No XLA failure or repeated retracing occurred. No new NumPy runtime path or
numerical-core edit was introduced.

The exact GPU launch was:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py start --phase p0_offset_localization

The original launch stopped before numerical execution at the historical
155-attempt ceiling. Allocating attempts 156–157 within the existing balance
resolved that bookkeeping limit; it added no compute. Remaining charged
campaign balance is 849.218199 GPU seconds (14.15 minutes) and 151428.365656
CPU seconds before the separately recorded 4.730780-second CPU diagnostic.
No numerical job remains running. The whole replication, particularly KDM
training, is not funded or established by this small remaining GPU balance.

Evidence root: artifacts/controlled-smc-kdm-replication-20260928/.
The three CPU directories retain source traces and reference predictions;
attempt156_p0_offset_localization retains the frozen executable, all direct
fits, six forward traces, full log, summary and manifest. Base commit is
89065bc6354801cb368d5e163ba49fd9c4372d10, with the task's uncommitted sources
hashed in each manifest. Focused supervisor tests and final hash checks are
recorded in offset-localization-final-review-20261006-01/: 17 focused tests,
333 input/output/source hash checks, and five compile checks passed. The
CPU-only review used 1.164830 seconds and verified both worker and supervisor
had exited. The dispatcher and refreshed checkpoint report both replications
as incomplete.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Localization passed | Identical-target differences and reciprocal policy substitutions explain almost all of the selected gap | Input identities, unchanged-source/RNG controls, finite outputs and matched forward controls passed | One case; propagation among backward times; different numerical libraries | Test source conditional-solver compatibility on saved systems before autonomous fitting | No full replication, all-case root cause, or author-method repair |
| Existing ports remain unqualified | Original matched downstream disagreement persists in autonomous ports | Existing P0 mismatch remains; no new diagnostic crash | Downstream no-worse margins and untouched repetitions unfinished | Complete P0 endpoint/failure checks and independently calibrated P1 | QR is not promoted solely for its lower residual |

| Inference status | Finding |
|---|---|
| Hard veto screen | Diagnostic validity checks passed; autonomous port qualification remains open |
| Statistically supported ranking | None; one deterministic case with matched random inputs |
| Descriptive-only differences | Likelihood percentages, residuals and the 99.68% reduction under substitution |
| Default readiness | Not established; no solver default changed |
| Next evidence needed | Actual conditional TF solver/consumer checks, then matched untouched downstream repetitions under frozen margins |

Post-run skeptical review: the strongest alternative is broader backward
propagation or library rounding beyond the selected fallback operations.
The direct-target check and reciprocal swaps reduce that ambiguity for this
case but cannot settle all other cases. A failed input/control check would
invalidate this result; a fresh case with a different localization would
limit its generality. Full adaptive fitting is the weakest remaining link.
The result identifies a difference in executing the author's numerically
ill-conditioned least-squares formula; it is not evidence of a fundamental
problem with the controlled-SMC paper.

## Next executable work

Keep the verified source baseline. First test whether TF Cholesky/LU/SVD
operations can reproduce the conditional solve's evaluated policies on the
128 earlier systems and these 101 regressions. Check exact source matrices
before TF-reformed matrices, and retain branch, rank, prediction and
downstream differences separately. Reproducing a branch alone is insufficient:
the previous SciPy-versus-system LAPACK check already demonstrated that.
Do not add a ridge, changed rank floor or precision extension.

Only a surviving compatibility candidate should enter the complete backward
fit and matched forward checks. Exact coefficient identity remains a
localization instrument, not the final no-worse criterion. Before P1, derive
and freeze metric-specific downstream margins on calibration data, then use
fresh paired repetitions and confidence bounds. The 1e-8 sensitivity
experiment informs that calibration but cannot by itself excuse a systematic
solver change. Failed-trajectory parity and the separate KDM source/training
specification remain open.
