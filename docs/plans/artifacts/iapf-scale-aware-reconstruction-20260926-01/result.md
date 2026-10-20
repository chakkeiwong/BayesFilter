# iAPF scale-aware reconstruction result — 26 September 2026

The bounded diagnostic is complete. The numerical stopping problem is repaired
and checked, but **no tested reconstruction produced a complete backward sweep
on the paper's five-dimensional model**. All four sweeps failed at time 98.
No fitted adaptive iAPF pilot qualified. Both scientific replications remain
incomplete.

The claimed target is Equation (15), with a diagonal Gaussian and free positive
scale lambda. The implementation analytically profiles lambda and reports the
mean squared residual, L/N. For fixed fitting sample count this differs from the
paper's sum only by a positive constant; it has the same minimizers. Local
residual scaling is held fixed during each step and acceptance test. It does
not replace the objective with shape error.

## What was checked and what failed

| Stage | Result | Meaning |
| --- | --- | --- |
| Analytic gradient versus finite differences | Relative error 4.51e-11 on the Gaussian control; below 1e-5 on all actual-input checks | Checked derivatives agree for these inputs |
| Fixed residual-scale invariance | Parameter difference 2.78e-16 | Numerical rescaling preserves the tested fitted result |
| Profiled versus direct joint objective | Relative difference 1.03e-15 | Profiling computes the same tested objective |
| Exact-twist filter versus Kalman | Log-likelihood difference 1.42e-14 | Filter mechanics pass this control |
| Perturbed exact Gaussian recovery | 6/8 cases passed; d=20 and d=80 failed at perturbation 0.20 | Broad solver qualification failed; d=5 passed both perturbations |
| Actual backward sweeps | 0/4 completed; each had two stationary fits before failing at time 98 | The checked d=5 solver does not yet fit the actual recursion reliably |
| Same-start iteration repair | Four 400-step failures remained failures after 800 steps | More iterations did not repair these cases |
| Initial-damping repair | All 12 real-input fits failed across damping 1, 1000, 1000000; all 6 corresponding d=5 recovery controls passed | These more cautious starts did not rescue the actual fits |
| Terminal provenance and wiring audit | 25 source, 35 output and 16 input hash checks passed; data/cloud/target replay exact | Saved failures belong to the stated executable recursion |

The first control attempt was FAILED under the original all-dimension gate.
That artifact remains unchanged. Before further execution, the reviewed plan
separated broad qualification from permission to run the already-planned d=5
diagnostic. The d=5 numerical criterion was not relaxed. The second control
attempt explicitly reports PASS_D5_ONLY. The subsequent actual-input failures
prevent promotion even within d=5.

At time 100, the target is an exact diagonal Gaussian: the QR start already
solves it, so those four stationary results test preservation. At time 99 the
solver took 7–8 iterations; relative gradients were at most 7.64e-7 and the
checked minimum curvature eigenvalues were positive. At time 98 all four
same-start 800-step retries retained relative gradients between 12.92 and
246.17, against the 1e-6 criterion. The stronger-damping experiments produced
six iteration-cap and six no-descent-step failures. No failed fit was silently
used as a twist.

## Why a tiny loss was misleading

For density values p and target values y,

    lambda* = (p'y)/(y'y)
    L* = ||p||^2 - (p'y)^2/(y'y).

If p tends to zero on the fitting cloud, lambda* tends to zero and the raw loss
also tends to zero. A fit can therefore reduce Equation (15) by moving density
away from the observed cloud. This is a mathematical escape route of the
unrestricted objective; it is not a TensorFlow or XLA effect.

The actual failures demonstrate that route. Some variances grew from roughly
0.84–0.92 to 943–277532, and mean coordinates moved by 1222–20705. Maximum
log-density on each cloud fell below -800. Log mean-squared loss fell from
about -20 to -29 initially to below -1609, while relative shape error worsened
from 0.0015–0.0197 to 0.0084–0.155. Direct FP64 density evaluations at those
endpoints would underflow; the rescaled diagnostic still detects the large
relative gradient. See [failure anatomy](attempt06/failure_anatomy.csv).

The derivatives and original-loss descent checks passed. This weakens the
explanation that the observed failures came from a derivative implementation
error or an absolute stopping tolerance alone. It does not prove that no
useful finite local minimum exists, that another local optimizer could not
reach one, or that the author's implementation followed this path. The
author's optimizer, basin selection and positive-floor formula remain unknown.

## Floor protection and comparator evidence

Both tested floor formulas give their intended minimum base-component mass:
delta=1/N^2 or 1/N. On a healthy exact one-dimensional Gaussian at the
maximizing parent, second moments were 1.000000000002 and 1.0000011345,
respectively, within the bounds 1/(1-delta). Away from that parent, the bound
must use the actual mixture probability, which may greatly exceed delta.

With an intentionally underdispersed variance 0.02, the corresponding
one-dimensional second moments were approximately 197287 and 355.3. They are
finite but large. Thus the floor supplies local tail protection without
establishing a good whole proposal. These deterministic reference checks do not
select a floor or establish high-dimensional performance.

The fresh diagnostic dataset used d=5,T=100,N=1000 and seed 9263001, with
eight filter draws per label. Five observations exceeded the Kalman
chi-square 95% innovation threshold; 95 did not.

| Comparator | Total log-likelihood error RMSE | Ordinary-innovation incremental MSE | Large-innovation incremental MSE |
| --- | ---: | ---: | ---: |
| Bootstrap filter | 2.74146 | 0.0344362 | 0.175057 |
| Observation-only twist | 0.290525 | 0.000483529 | 0.00298691 |
| Fully adapted APF | 0.290525 | 0.000483529 | 0.00298691 |
| Exact Kalman authority | 0 | 0 | 0 |

These figures are descriptive, from one dataset and eight draws; there is no
predeclared uncertainty analysis supporting a stochastic ranking. The
observation-only twist and fully adapted APF coincide here. With C=I and R=I,
the observation twist is proportional to g_t, giving q_t proportional to f_t*g_t,
the exact one-step conditional Gaussian. Their total values agree for all
eight common seeds and incremental discrepancies are roundoff-sized. Therefore
the three executed labels represent only two distinct particle strategies,
plus the exact Kalman authority. The initial plan overstated their independent
breadth; preserve that limitation instead of counting duplicate strategies as
extra evidence. No iAPF heuristic-dominance verdict is available because no
eligible fitted iAPF arm exists.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the stopping/observability repair for this reference diagnostic | Derivative, rescaling, descent and recovery controls pass in d=5 | No checked harness/derivative veto | Applicability outside tested inputs | Retain relative gradients and trajectories | General optimizer validity or TF readiness |
| Reject current d=5 reconstruction for adaptive execution | Four complete backward sweeps required; zero completed | Fit stationarity failed; bounded retries failed | Local basin, starting rule and author's implementation | Source recovery or a reviewed finite-branch hypothesis | iAPF research direction rejected |
| Retain both floors as unselected hypotheses | Local protection controls pass | No downstream fitted comparison exists | Adequacy and original formula | Assess protection separately from fitting and full-filter quality | Floor superiority or source fidelity |
| Keep full replication open | No fitted adaptive or paper-scale experiment | Missing claim-bearing evidence | Solver/source choices, higher dimensions, uncertainty, port parity | Resolve useful finite fits before paper-scale compute | Replication/default/HMC readiness |

| Inference category | Status |
| --- | --- |
| Hard veto screen | Current real-input fits failed the declared stationarity criterion; no adaptive pilot admitted |
| Statistically supported ranking | None |
| Descriptive-only differences | All stochastic comparator errors, tails and timings |
| Default readiness | Not established; independent R/CPU reference only |
| Next evidence needed | Source-justified finite local fitting, complete adaptive runs, multiple independent datasets/seeds and uncertainty, then TF/GPU/XLA parity |

## Execution and review

The reviewed plan is [iapf-scale-aware-reconstruction-20260926.md](../../iapf-scale-aware-reconstruction-20260926.md).
Six launches: controls (failed broad recovery), controls with explicit scope
reporting, real backward sweeps, bounded damping repair, baseline-only pilot
phase, and terminal audit. Every source snapshot, exact command, seed, R version,
CPU/GPU choice, wall time and output is preserved in each attempt's manifest.
The runner uses immutable source copies. GPU devices were deliberately hidden.
There were no package changes or modifications to the historical experiment
outputs. The new scripts remain independent reference diagnostics.

CPU wall charge: **41.725506 seconds**, GPU charge **0**. Combined CPU charge is
**131.661771 / 1800 seconds**, leaving **1668.338229 seconds**. The six-launch
cap is reached. Approximately 36.57 authorized GPU hours remain, unchanged.
Do not rerun the completed campaign or spend GPU time on the same unresolved
fit merely because budget remains.

Self-review and executable terminal audit were performed; no independent
reviewer is claimed. Post-run red-team findings: the broad control gate needed
an explicit scope correction; two baseline labels were equivalent; damping
failures do not exclude alternative local basins; eight stochastic draws
cannot support a scientific ranking. The strongest alternative explanation is
that a different source-consistent initialization or local fitting strategy
finds a useful finite solution. A reproducible stationary, positive-curvature
solution on these saved inputs, followed by full independent backward sweeps
and adaptive filters, would overturn the current candidate rejection.

The next scientific question is how the original implementation selected a
useful finite Eq15 solution. Recover the author's optimizer/domain choices if
possible; otherwise state and review a finite-branch reconstruction explicitly.
Bounds, penalties, shape normalization, or constraints on lambda would change
the optimization problem and must not be inserted silently.
