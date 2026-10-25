# Final report: nonlinear structural Model B score diagnostics

Date: 26 October 2026. Disposition: diagnostic investigation complete and
closed at the owner's request.

## Conclusion

The completed diagnostics do not establish an error in the analytical
differentiation at the tested parameter point. They identify a concrete
source of approximation error: the particle reset changes the next nonlinear
predictive density and its score even after moment correction. This explains
much of the observed short-horizon discrepancy on the inspected particle
design. A longer-horizon persistence-score discrepancy remains unresolved
because the independent reference also has unmeasured finite-particle bias.

The findings do not justify a broad rewrite of the derivative implementation.
They support a narrower conclusion: the finite-program derivative checks
well, while the accuracy of its approximation to the original model remains
partly unsettled. The investigation closes with that distinction explicit.
No new production default, reset method, or HMC result is promoted.

## Model and question

The model under investigation is

\[
 m_t=\rho m_{t-1}+\sigma\varepsilon_t,\qquad
 k_t=0.55k_{t-1}+\beta\tanh(m_t),\qquad
 y_t=m_t+k_t+0.30\eta_t,
\]

with independent standard normal innovations and the campaign's
parameter-independent initial Gaussian law. The inspected parameter point
is \((\rho,\sigma,\beta)=(0.70,0.25,0.80)\). All scores in this report
use \(q=(\operatorname{atanh}\rho,\log\sigma,\log\beta)\), rather than
the physical parameter coordinates. Observations are held fixed. Production
calculations retain FP32 and 4,032 particles.

For a fixed particle design \(u\), the program returns a finite-particle
log likelihood \(\ell_N(q;u)\) and its analytical score. The first question
is whether that score agrees with \(\nabla_q\ell_N(q;u)\). The second is
whether either quantity agrees sufficiently with the original model's
likelihood \(\ell(q)\) and score \(\nabla_q\ell(q)\). These are separate
questions; a correct derivative of an approximation does not settle its bias.

## What the diagnostics establish

The code trace follows the structural consumer into the shared batched
TensorFlow recurrence, including conditional innovation proposals, importance
weights, analytical tangents, transport, reset, and moment correction.
Deterministic completion is enforced conditional on each parent. The checked
filtering recurrence uses TensorFlow loops; Python orchestrates diagnostic
runs rather than the production observation loop. The numerical GPU consumer
requests XLA, while reference integration and diagnostic bookkeeping execute
outside that compiled kernel.

At T1 and T2, the FP64 CPU checks agree with finite differences to about
4e-11 at the smallest tested step. The corresponding FP64/FP32 likelihood
differences are at most 1.03e-6 and score differences at most 1.19e-7. These
are joint precision/backend checks, not an isolated test of precision: the
FP64 diagnostic uses a CPU graph and the uniform reset policy. Agreement at
these horizons and this parameter point does not certify T50 or the whole
posterior region.

In FP32, excessively small finite-difference steps amplify subtraction error.
For T50, the maximum component discrepancy is 0.00130 at step 0.01, but
0.364 at step 0.00003. The complete step ladder is retained. Every retained
T50 baseline row completed all 50 steps, every recorded validity check passed,
and no conditional FP64 reset fallback fired. The ladder does not demonstrate
a persistent large derivative defect.

### Reset contribution at T2

The reset comparison starts from the same weighted particles and tangents,
then compares the next predictive density before and after the complete
production reset. Independent Gauss-Hermite integration handles the next
innovation. The comparison therefore measures the effect of the reset,
including its moment correction, on that prediction.

| Quantity | LEDH minus refined T2 reference | Post-reset minus pre-reset prediction |
|---|---:|---:|
| Log likelihood | +0.014424 | +0.014979 |
| Score for atanh(rho) | +0.036151 | +0.030208 |
| Score for log(sigma) | +0.000826 | -0.006396 |
| Score for log(beta) | +0.012735 | +0.006954 |

The T2 model reference is refined path quadrature. Increasing its order from
28 to 32 changes the log likelihood by 5.47e-9 and the largest score component
by approximately 1.23e-7. It is a well-refined numerical reference, not a
closed-form answer or a rigorous error bound. Separately, increasing the reset
diagnostic's conditional integration from 32 to 64 nodes leaves its reported
log-density effect unchanged and changes score effects by at most 8.94e-8.

For this dataset and particle design, reset explains much of the likelihood
and persistence-score gap. It does not explain every score component in the
same direction. Matching means and covariances cannot in general preserve an
expectation of the nonlinear next-observation density or its derivative.
The saved additive decomposition separates the reset contribution from the
remaining pre-reset and next-step integration differences.

### The remaining T50 question

Across four independent particle designs, the mean persistence score is
-2.351205. The existing bootstrap/Fisher reference reports -2.125355, giving
a difference of -0.225850. Combining their estimated replication standard
errors in quadrature gives 0.041124. The gap is therefore large relative to
the measured random variation; dismissing it as Monte Carlo noise is
unsupported. Four LEDH replications do not justify a calibrated significance
claim, and neither standard error measures finite-particle bias.

The reference's complete-data score formulas pass the source and dependency
audit. Their particle approximation is still subject to normalized-weight
bias and loss of historical path diversity. It is not an exact T50 oracle.
Reset bias, reference bias, or both remain possible explanations. The mean
likelihood difference is only +0.002295, illustrating why close likelihood
values at one point do not establish close derivatives.

## Additional issues and repairs

The source always performs a reset before accepting the final likelihood
increment. A failed terminal reset can therefore invalidate an otherwise
completed likelihood even when no terminal cloud is requested. This path
did not fail in the retained runs. A likelihood-only terminal path would
need its own consumer and parity tests before changing production behavior.

Fresh target instances create fresh graph closures during design replication,
adding compilation cost. This does not imply that a frozen HMC target
retraces at every sampling step. The reference also lacks ancestor-diversity
and time-resolved smoothing-score uncertainty diagnostics. These are useful
follow-up engineering items; neither was measured as the cause of the T50 gap.

The new diagnostic runner initially discarded most of its computed T50 step
ladder and inherited an FP32 label in an FP64 manifest. Those reporting
defects are repaired. The full ladder, perturbations, and baseline traces are
now retained; metadata corrections preserve the original evidence. The
runner refuses reused output directories and records future launch outcomes.
Production filtering and HMC were not changed or rerun.

## Close-out decision

| Decision | Primary criterion status | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Close this diagnostic investigation | Reviewed diagnostic stages completed | Successful runs finite and valid | Long-horizon model-score accuracy | Archive report, code, and results | Exact-model posterior certified |
| Retain derivative implementation | Tight T1/T2 FP64 consistency | No demonstrated mismatch at tested point | Other points and T50 FP64 untested | No broad derivative rewrite | Global derivative proof |
| Record reset as a T2 contributor | Effect survives quadrature refinement | Reset and density checks pass | Single dataset/design attribution | Replicate only in a future accuracy study | Sole cause of the T50 gap |
| Leave T50 attribution open | Gap exceeds estimated random variation | Reference formulas checked | Bias of both approximations | Refine reference and assess inferential materiality if reopened | Gap is harmless or purely random |

| Inference status | Final assessment |
|---|---|
| Hard veto screen | Successful diagnostic results are finite and valid; failed harness attempts are excluded from scientific comparisons |
| Statistically supported ranking | None; no competing reset or filter is promoted |
| Descriptive-only differences | Four-design summaries and single-design reset attribution |
| Default readiness | No change to FP32, N=4032, production controls, or the HMC target |
| Next evidence needed | Replicated reset attribution, a stronger T50 reference, and effects on parameter estimates or credible intervals |

There is no unfinished sampling stage or active run in this work. No further
experiment is scheduled. If the accuracy question is reopened, its useful
endpoint is inferential materiality: whether the approximation meaningfully
changes parameter estimates or credible intervals. Neither a score gap alone
nor convergence of chains sampling the finite target settles that question.

## Verification and reproducibility

The focused structural tests passed: 19 tests with two warnings. Final Python
compilation, CLI help, JSON parsing, and script-checksum checks also passed.
Successful diagnostic process wall times total 368.56 seconds for GPU-enabled
runs and 529.94 seconds for CPU-only runs. These include compilation and do
not measure GPU utilization. Early failed attempts lack complete timing/log
records; conservative interval accounting and that limitation are preserved
in the execution record. The bounded diagnostic budget was not exhausted.

The investigation used base commit
`a54add9e78678c5bd5a36ba89164586bd97dab34` on the isolated
`structural-score-diagnostic-20261026` branch. The executable diagnostic,
reviewed plan, successful results, preserved failed-attempt evidence, and
explicit metadata corrections accompany this report.
The original T2/T50 observations and particle designs are preserved in a
compressed input archive, with byte-level checksums and extraction
instructions, so reproduction does not depend on the temporary source
worktree surviving.

- [Reviewed diagnostic plan](structural-score-diagnostic-2026-10-26.md)
- [Detailed results and execution limits](artifacts/structural-score-diagnostic-20261026-01/result.md)
- [Call-chain audit and reference-score derivation](artifacts/structural-score-diagnostic-20261026-01/source-audit.md)
- [Machine-readable summary](artifacts/structural-score-diagnostic-20261026-01/summary.json)
- [T2 decomposition](artifacts/structural-score-diagnostic-20261026-01/t2-decomposition.json)
- [Diagnostic runner](../../scripts/run_structural_score_diagnostics.py)
- [Portable input archive instructions](artifacts/structural-score-diagnostic-20261026-01/README.md)

Closing this investigation records what the measurements establish and the
remaining uncertainty. It does not turn the unresolved T50 comparison into
an accuracy pass.
