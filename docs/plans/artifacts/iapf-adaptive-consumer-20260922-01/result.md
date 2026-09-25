# Multivariate adaptive iAPF consumer comparison

The actual local adaptive consumer agrees with an independent R reconstruction
on all ten CPU and ten GPU FP64/XLA fixtures. The adapter's scalar restriction
has been removed for linear models; existing nonlinear scalar restrictions,
claim/tuning verification and convergence checks remain. The shared general
filter and fitter execute through the public consumer, with no reduced fork.
This closes the tested local-consumer boundary, not the paper-replication gap.

The final evidence contains 32 preflight checks, 850 CPU/R checks and 850
GPU/R checks, all passing. Twenty-two regression tests pass, including actual
public consumer calls at d2/o1 and d5/o3 and retained rejection boundaries.
The preflight independently verifies objective derivatives and exact known
Gaussian recovery at d2/d5 before testing the backward fitting recursion.
Fixtures cover (d,o)=(1,1),(2,1),(2,2),(5,3),(5,5), two seeds each, T4 and
initial N64. Final N ranges from 128 to 512, and stopping occurs after 4–11
offline evaluations. These are deterministic implementation comparisons.

| Maximum absolute difference from independently reconstructed R | CPU FP64/XLA | GPU FP64/XLA |
|---|---:|---:|
| Log likelihood | 2.57e-12 | 2.11e-12 |
| Particle coordinates | 1.88e-11 | 1.55e-11 |
| Fitted center | 3.69e-10 | 3.04e-10 |
| Fitted covariance | 3.10e-12 | 2.56e-12 |
| Log floor | 1.33e-12 | 1.10e-12 |
| Frozen-guide analytical score vs label-stable R finite differences | 7.04e-9 | 8.16e-9 |

R receives the realized random draws and observations, constructs its own
clouds, recursively fits its own guides, and independently decides particle
doubling and stopping. It computes Gaussian proposals in precision form and
the profiled objective by direct density algebra. This avoids hiding a fitting
or transition error by feeding TF coefficients back into a reference filter.
Both reference and actual paths preserve every ancestor and mixture decision;
the optional numerical trace is bitwise neutral for all captured executions.

One original derivative comparison failed because finite perturbations crossed
resampling boundaries. The d5/o5/seed41 final minimum choice margin was 4.56e-7;
steps 1e-5 and 5e-6 changed labels and produced an apparent score error of .111.
The original 960-check table, including seven failures, remains in attempt03.
The reviewed repair selects a finite-difference pair by label stability alone,
using the fixed ladder 1e-5, 1e-6, 1e-7, 1e-8 and half-step partners. At 1e-6
and 5e-7 all labels stay fixed, and the discrepancy is 7.04e-9. Original
acceptance tolerances were unchanged. Attempt04 rechecks the original TF
evidence in R; it is not a new untouched stochastic run. This illustrates why
a fixed-label analytical derivative must be tested within its smooth region.

The compared local program uses X0 particle initialization, a six-parameter
linear model, every-step resampling, bounded diagonal density fitting and a
declared peak-density floor. The paper's study uses a different linear model,
X1 initialization and adaptive resampling with retained weights. Original
author solver and floor identity remain unknown. Neither successful conformance
nor an absolute-gradient stopping flag establishes a statistically good guide.
The next discriminating check is a known-target dimensional ladder for the
local fitting tolerance and initialization, keeping the density objective
separate from relative-shape or log-regression alternatives.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain removal of scalar restriction for local linear consumer | Full independent CPU/R and GPU/R comparisons pass | No implementation veto on tested fixtures | Dimensions beyond five, longer horizons and other precision settings | Diagnose dimension-dependent fitting/stopping before larger studies | Arbitrary-dimensional robustness or production/default readiness |
| Retain repaired shared transition | Both devices agree in FP64; public call chain checked | Prior strict TF32 failures remain | TF32 distributional accuracy | Revalidate affected TF32 evidence before reuse | TF32 veto cleared |
| Accept tested frozen-guide finite-program derivative | Label-stable independent finite differences pass | Initial unstable comparison preserved and explained | Expectation/model-score identity | Separate marginal-score study | Unbiased marginal score or HMC validity |
| Keep paper replication open | Same local program now checked across languages | Paper identity prerequisites remain unmet | Original initialized solver/floor | Known-target fitting diagnosis, then explicit paper reconstruction hypotheses | Original-author implementation reproduced |

| Inference status | Finding |
|---|---|
| Hard veto screen | Final conformance passes; original unstable finite differences preserved; TF32 promotion veto unchanged |
| Statistically supported ranking | None; this is deterministic conformance evidence |
| Descriptive-only differences | Runtime, iteration count, fitted residuals and boundary activity do not rank methods |
| Default-readiness | Not established; local capability enabled, no numerical backend or canonical LEDH policy changed |
| Next evidence needed | Dimension-dependent fit validity, author-choice reconstruction and downstream likelihood/model-score evidence |

Execution used the existing tftwogpu environment, TensorFlow
2.20.0-dev0+selfbuilt, CUDA12.8.1/cuDNN9. GPU runs used the authorized RTX4080
SUPER UUID, escalated access, XLA, FP64, TF32 disabled and verified memory
growth before device initialization. CPU runs explicitly hid GPUs and used
one BLAS/TF thread. Exact commands, seeds, actual inputs/outputs, environment,
R session information and source snapshots are saved per attempt. Data are
synthetic from the exact consumer and preserved with each run.

The five launches and regression tests consumed 148.95 CPU worker seconds and
93.86 GPU process seconds, including R time during GPU comparison; all attempts
are charged. The new campaign balance before final verification is 45.887 CPU
hours and 47.891 GPU hours. `budget.json` contains exact accounting.

Terminal skeptical review: the strongest alternative explanation is that a
matched bounded local optimizer can agree across languages while fitting a
poor guide. The result is deliberately limited to that program's implementation.
The independent formulas, own-guide R recursion, full label histories and public
consumer tests answer the call-chain question; they do not answer high-dimensional
statistical adequacy. Failure on a longer or higher-dimensional known-target
check would restrict applicability without overturning these saved comparisons.
The weakest extrapolation would be treating d5 conformance as paper-scale d80
validity; that claim is explicitly withheld and motivates the next check.
