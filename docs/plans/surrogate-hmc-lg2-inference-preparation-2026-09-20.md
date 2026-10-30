# LG2 posterior-reference preparation

The research question is whether imperfect full canonical DCTR proposal fields
support endpoint-corrected HMC parameter inference on LG2. This stage prepares
the independent model-posterior reference needed for that test. It runs no HMC
and selects no DCTR settings.

The target is the existing `LG2` fixture: two states, two observations, T=20,
`F=a*[[.7,.2],[.1,.6]]`, `Q=exp(2*b)*[[1,.3],[.3,.8]]`, H=I, R=.5I,
independent uniform priors a in (.5,1.2), b in (-1,1), and the parameter-dependent
stationary initial law. Synthetic truth is (1,0), with transition before each
observation. Freeze attempt1, replicate0 and repository-issued seeds for each
of calibration, validation and claim. Generate each tiny dataset on CPU as an
explicit reference exception. Round observations once to FP32 and preserve
those exact values for both the later DCTR consumer and the FP64 reference.

The checked matrix Kalman reference is the comparator; its density and total
gradient have passed an independent joint-Gaussian calculation. Integrate the
claim posterior over the full two-dimensional prior rectangle on trapezoidal
grids with32,64,128,256 intervals per axis. Use the finest grid to freeze means,
standard deviations and each marginal's .1/.5/.9 quantiles. Report thirteen
functions: two standardized parameter means, two centered second moments,
their standardized cross moment, six marginal CDF values, and the two exact
conditional probabilities Pr(y21,j>0 | theta,y1:20). Integrating the predictive
functions concerns parameter uncertainty in the exact model, not DCTR state
prediction. Frozen equivalence margins are .1 for the five moment functions,
.02 for the six CDF values and .03 for the two predictive probabilities.

The reference passes this numerical screen only when every value and
normalization is finite, every empirical allowance is at most one tenth of
its function's margin, and the last change is at most half the preceding
change plus1e-5. The allowance is twice the maximum of the last two changes,
with all grids evaluating the same finest-grid-defined functions. Integrate
CDF cuts within cells using the piecewise-linear marginal density, avoiding
an indicator-at-grid-node discretization. These are empirical convergence
checks, not rigorous quadrature error bounds. A failed reference blocks the
affected posterior comparison and triggers refinement within the remaining
budget; it does not reject surrogate HMC.

Defaults and risks: fixture initialization, timing and priors are fixed target
choices; the derivative includes stationary covariance and process scale.
FP64 is an independent-reference exception. TensorFlow/XLA is used with
stable signatures; no NumPy or pfor. Grid sizes and tolerances are numerical
hypotheses checked by the refinement ladder. Synthetic uniform priors do not
establish a scientifically appropriate prior for real applications. A tiny
Gaussian scope cannot certify nonlinear models. Keep all data partitions
separate and never tune DCTR controls on the claim posterior error.

The model baseline ladder remains exact-gradient HMC, imperfect-force HMC
against the exact endpoint, and imperfect-force HMC against a qualified finite
endpoint. The cheap adversaries are the reference itself, zero-force/random-
walk Metropolis, and a simple Gaussian posterior approximation. For later
conditional comparison, freeze central and marginal-tail situations from this
reference. Their evaluation is NOT_RUN at this stage and continues to veto
efficiency/default promotion.

Skeptical audit: a correct Kalman likelihood with the wrong initial covariance
or unrounded observations would answer a different question. The numerical
test and frozen data identity address these risks. A posterior grid agreeing
with itself may miss local structure; refinement allowances remain empirical,
and later sampler/reference discrepancies require investigation. Posterior
means, losses, acceptance and runtime alone are not promotion criteria.

Budget: P3 LG2 first-scope allocation remains2 GPU hours/.5 CPU-worker hours,
within the whole63 GPU/15 CPU-worker-hour campaign. This reference stage plus
its new focused tests may consume at most300 CPU-worker seconds (in addition
to the18.979316-second earlier implementation check), with at most300 seconds
per subprocess. GPU devices are deliberately hidden before TensorFlow import.
No GPU time or extra inference attempt is allocated by this preparation.
Preserve unique versioned roots, command logs, seeds, source/data hashes,
TensorFlow tensors, reference report, run manifest and incremental accounting.

Execute `scripts/prepare_surrogate_hmc_linear_reference.py --model LG2
--output <new-root>` with the tftwogpu Python and the master's CPU watchdog.
The next stage must separately qualify the actual full-DCTR consumer, calibrate
scope-specific controls and measure costs before public tuning or HMC.
