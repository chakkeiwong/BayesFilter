# LG2 independent posterior reference

LG2's model-posterior reference passed the predeclared four-grid numerical
screen. The main research question remains the testing of surrogate HMC with
an imperfect full-DCTR force; no LG2 sampler comparison has run yet.

The target uses the existing stationary two-state, two-parameter LG2 model,
T=20, independent uniform parameter priors and transition before observation.
Attempt1/replicate0 calibration, validation and claim data have separate seeds
and checksums. Observations are rounded once to FP32 and promoted exactly for
the FP64 reference, so later consumers can use identical data.

The claimed reference quantity is the parameter posterior under the complete
Kalman likelihood, including parameter-dependent stationary covariance. Its
numerical calculation uses the full prior rectangle and trapezoidal grids with
32,64,128,256 intervals per axis. Same-law joint-Gaussian density, gradient and
predictive checks pass; analytic linear-density and uniform-rectangle tests
also pass. All three focused tests passed before the actual integration.

On the claim data, posterior means are approximately(0.7587243,-0.0750715),
standard deviations(0.1573859,0.1918190), and standardized cross moment
-0.2249872. The exact-model predictive positive-observation probabilities are
0.5189722 and0.5318472. These are reference summaries, not evidence that any
sampler recovers them. Thirteen reference functions and their margins are
frozen in the result: five moment functions, six marginal CDF values and two
predictive probabilities. The largest empirical integration allowance is
0.000855451 for the second parameter's upper-tail CDF, below its0.002 screen.
Every refinement and precision check passes. Allowances are empirical, not
rigorous quadrature bounds.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Reference available for LG2 comparisons | Four-grid precision/refinement screen passes | No nonfinite grid value, invalid normalization or unresolved function | Refinement can miss finer numerical structure | Qualify actual full-DCTR consumer, calibrate controls and forecast complete HMC | Sampler accuracy, convergence, efficiency, finite-DCTR posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical reference and independent oracle tests pass |
| Statistically supported ranking | None; no method comparison has run |
| Descriptive-only differences | Posterior summaries and timing |
| Default-readiness | Not established; independent FP64 CPU reference exception |
| Next evidence needed | Scope-specific DCTR tuning, GPU consumer checks, public HMC verification and retained reference comparisons; heuristic comparisons |

The strongest alternative explanation is a later mismatch between this target
and the DCTR consumer's initialization, observations or chart. Consumer wiring
and identity checks remain necessary. A materially different finer-grid or
independent reference would reopen qualification. No candidate was selected
using claim posterior error.

Evidence root:
`artifacts/surrogate-hmc-all-models-20260920-lg2-reference-01`. Its
`reference/run-manifest.json` records the actual command, git revision,
environment, CPU-only choice, XLA, seeds, sources, plan and report paths.
`reference/claim/reference.json` records all functions, thresholds, refinement
values and serialized-tensor checksums. The source snapshot and full command
log are preserved. The reference consumed13.312587 CPU-worker seconds; the
preceding three tests consumed14.371981. Together with the earlier reference
checks, LG2 preparation has consumed46.663884 CPU-worker seconds and no GPU
time. This fits the first-scope P3 allocation and does not consume an HMC
inference attempt.
