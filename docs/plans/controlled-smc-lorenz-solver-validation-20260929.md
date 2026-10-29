# Lorenz equivalent-solver repair validation

The authorized consumer diagnosis passed51 required checks, including the
actual source forward/replay endpoints, but failed all3 source fit coefficient
comparisons. On identical backward targets, raw GPU normal equations give
RSS849.090409 versus2.701200 for centered QR. Source Octave raw Gram reciprocal
conditions reach1.53e-17; standardized conditions exceed.0034. This triggers
the planned equivalent-solver repair, not rejection of controlled SMC.

The question is whether centered/scaled QR computes the intended unweighted
full-quadratic least-squares recursion accurately and yields a valid filter.
The mathematical baseline is the same source objective, including every cross
term and the difference of next-policy Gaussian integrals. The numerical
comparators are the preserved raw normal-equation results and an independent
CPU SVD least-squares reference on the same source clouds. No ridge, damping,
projection, coefficient clipping, target change or primary-metric tuning is used.

The CPU reference uses SciPy/NumPy only in an explicitly independent diagnostic
script, with GPU hidden and single-thread BLAS. It independently evaluates RK4,
Gaussian integrals, the backward recursion, full quadratic basis and original
coordinate coefficient conversion. SVD must report full rank. Preserve every
policy coefficient, singular-value ratio and residual; archive two reference
forward trajectories using the source round1 random inputs. The accepted
TensorFlow runtime never imports this reference module.

The GPU/XLA test consumes the frozen source fixture and independent reference.
Require QR policy coefficients, weights, normalizers, ESS and paths to agree
within1e-7 absolute plus1e-9 relative, with identical ancestry. The precision
guard must pass at every time. Also check fresh forward streams20261001/2 for
finite values and valid precisions, and verify stable single tracing. Previous
healthy exact-quadratic and Gaussian controls remain prerequisites; this is
not a repetition of the whole validation suite. Unhealthy raw normal equations
are a numerical diagnostic, not the correctness oracle for the repair.

Promotion is restricted to the centered-QR solver for this d8/H101/N512
consumer, with explicit solver identity. Primary criteria are independent SVD
agreement and a valid downstream forward run. Invalid precision, rank loss,
nonfinite outputs or failed agreement veto promotion and trigger localization.
Likelihood, ESS and runtime differences are descriptive and cannot select the
solver. There is no stochastic ranking, Figure6/Table1 replication, general
default, or HMC claim. No heuristic performance claim is made by this repair.

The mean/std coordinate transform is an invertible basis change whose purpose
is to avoid cancellation from concentrated particles around nonzero means.
Its failure modes are zero spread, rank loss and conversion cancellation;
the diagnostics above expose them. SVD rank tolerance is LAPACK's documented
machine precision criterion, a reference convenience checked against the
reported smallest/largest singular values; it is not a regularization knob.
No SVD truncation is accepted. Retain the raw normal-equation route explicitly
as a source-numerics comparison, never silently as a fallback.

Skeptical audit: PASS for this bounded numerical repair. This addresses the
observed failure with an unchanged scientific objective; it does not tune on
likelihood or relax the previous parity thresholds. An independent solver and
full downstream replay prevent coefficient-only or training-residual-only
evidence from being mistaken for consumer correctness.

Implement dispatcher phases c1_lorenz_solver_reference (CPU,60 seconds per
attempt,120 aggregate,max2) and c2_lorenz_solver_validation (GPU,120 seconds per
attempt,240 aggregate,max2). At most4 attempts fit the existing30-attempt cap.
The total campaign allocation stays CPU300/GPU18000 seconds, within the inherited
budget. Use unique campaign attempt directories, source/input hashes and the
existing manifest/accounting. Command prefix remains the same dispatcher;
GPU execution is escalated and records verified memory growth, FP64 and XLA.

Afterward update the repair result and master with separate source-forward,
source-numerics and intended-objective solver verdicts. Full empirical Lorenz,
RBF/NNLS and KDM learning evidence remain separate work.

Attempt26 failed at the independent forward array allocation because Octave
metadata dimensions serialize as JSON floats. Explicit integer conversion at
this diagnostic boundary fixes the harness without changing values. A focused
101.0/512.0/8.0 allocation/range check and syntax check pass. Charge2.970996434
CPU seconds; retry in a fresh directory under the same criteria and cap.

Attempt27 independent SVD passed, full-rank singular ratios at least.10869.
Attempt28 QR fit is valid but its original-coordinate linear coefficients fail
1e-7+1e-9 relative agreement (maximum difference2.94686e-6). The diagnostic
previously exited at that check; the second GPU attempt records all coefficient
and downstream comparisons before deciding. All thresholds and the promotion
rule remain unchanged. Invalid precisions still stop immediately. This is
localization, not permission to ignore a failing coefficient comparison.
