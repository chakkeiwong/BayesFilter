# Completion checks for the marginal-weight experiment

This completes the already authorized plan. It uses the remainder of its
21,600-second elapsed GPU budget, including all previous attempts; it creates no
new budget and changes neither the filter nor the LG/KSC statistical criteria.
Roots `-01` and `-02` preserve the requested N=256 sequence. A fresh `-03` root
will exercise the plan's optional N=4096 range-bearing refinement, with the same
calibration/validation/claim separation, points and 32 claim designs. This is
needed because the original range-bearing discrepancy occurred at N=4096.
Calibration is repeated for this distinct particle-count scope. If the remaining
budget cannot accommodate this stage, report it as incomplete, not as a failure
of the estimator. Larger N is a refinement hypothesis, not a presumed repair.

An additional diagnostic script will complete three checks: (1) the actual
mixture kernel on N=4096, where exact tiling crosses block boundaries, against
an independently reduced two-component mixture and finite differences; (2)
selected N=256 FP64 evaluations on identical FP32-rounded claim inputs, to
localize precision effects; and (3) three simple conditional comparators:
prior-trajectory importance sampling (no observation-driven proposal or
resampling), the already measured bootstrap PF (resampling only), and Gaussian
filtering (exact Kalman for LG, moment-matched observation noise for KSC, EKF
with wrapped bearing residual for M13). The first two use N=256 and 32 seeds;
the last is deterministic. These are falsification checks, never calibration
targets. Oracle distance remains the accuracy measure. The exact Kalman filter
is also the LG reference; it necessarily has zero reference error and motivates
using Kalman on this tractable model rather than a general particle method.
These added checks cannot retroactively strengthen the predeclared LG/KSC test.

The T20 range-bearing particle reference is approximate. Before describing an
accuracy gain, compare its N=32768/65536 refinement and replicated MCSE with
the observed filter differences. Require finite positive uncertainty estimates
and distinguish reference uncertainty from variability of the 32 candidate
designs. A material refinement disagreement triggers further reference work
within budget. A stable particle ladder still does not prove absence of
particle bias. T1 quadrature is the stronger local authority.

Skeptical review: these commands answer the remaining questions about tiling,
precision, simple alternatives and the original N=4096 setting. None changes
the target, runtime formula, controls after holdout inspection, materiality
margin, or default. Invalid candidates remain failed evidence; the program
continues to the next predeclared diagnostic. A failed mathematical identity,
invalid reference, or resource-budget exhaustion is a continuation veto until
repaired. Outputs go in a new `supplement` directory with commands, source
hashes, environment, wall time, validity and all observed errors. The final
result will separate numerical correctness, non-deterioration, M13 accuracy,
and the still-unestablished surrogate-HMC/posterior claim.
