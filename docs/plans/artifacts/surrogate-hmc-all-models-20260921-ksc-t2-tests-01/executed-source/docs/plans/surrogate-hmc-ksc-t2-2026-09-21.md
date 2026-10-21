# Surrogate HMC for the two-observation KSC model

The question is whether endpoint-corrected HMC with the full canonical DCTR
analytical proposal field samples the declared parameter posterior accurately.
This is the next M03 scope in the authorized all-model program. Qualification
of a likelihood or a consumer is preparation; the intended result is a retained
HMC comparison. The KSC (Kim–Shephard–Chib) mixture observation law and exact
transformed SV remain separate model obligations.

## Target and independent reference

Use the seven-component observation law already implemented in
`bayesfilter/highdim/sv_mixture_cut4.py::KSCMixtureTransformedSVSSM`, with the
weights, shifted means and variances returned by
`ksc_1998_log_chi_square_mixture`. This is a test of that repository model,
not a new claim about the paper or the accuracy of its approximation to SV.
Copy the independent Uniform(0.1,0.9) priors and two-probit chart from
`bayesfilter/testing/ksc_ukf_neutra_target_tf.py` (prior/chart definitions at
lines618–643). Infer physical persistence gamma and observation scale beta;
the state innovation standard deviation is fixed at1.

Let s=1/(1-gamma²). Under stationary initialization, the two observed latent
states have covariance [[s,gamma*s],[gamma*s,s]]. This holds both for an
observation at stationary time zero and for a transition from a stationary
initial state before the first observation. The full canonical consumer uses
the latter convention; it must retain parameter-dependent initial covariance.
For component labels i,j, the two observations z have mean
2log(beta)+(mu_i,mu_j) and covariance
[[s+v_i,gamma*s],[gamma*s,s+v_j]]. Therefore the model likelihood is exactly
the sum of49 bivariate Gaussian densities with weights w_i*w_j. This follows
by integrating the Gaussian latent pair conditionally on the two component
labels; there is no moment collapse or latent-grid truncation.

Check this closed-form calculation against an independently arranged exact
Gaussian-mixture Kalman recursion, the repository observation density and
finite-difference derivatives. For prediction, condition x2 in each of the49
Gaussian components, propagate x3=gamma*x2+epsilon, then integrate the seven
new observation components. Posterior predictive functionals are
Pr(z3<=-2|theta,z1,z2) and Pr(z3<=0|theta,z1,z2).

The physical uniform density and the chart Jacobian cancel their width factors:
the unconstrained log posterior is log L(.1+.8 Phi(q)) + log Normal(q;0,I).
Add the prior/Jacobian once. The reference is FP64 TensorFlow/XLA with a stable
batch signature. DCTR remains the FP32/TF32 full canonical analytical engine;
its score is explicitly a deterministic position-only proposal field.

## Data and separation

Generate two observations directly from the declared KSC transformed-observation
law, once rounded to FP32, with independent fixed seeds for calibration,
validation and claim. Nominal truth is(gamma,beta)=(.5,.5); stress truth is
(.88,.12). Freeze all seeds before qualification. These are new synthetic
instances, not a prefix of the T1000 dataset. No raw-return transformation is
applied to observations already generated in transformed coordinates. When raw
returns are used in a later scope, their existing log(y²+1e-8) preprocessing
must be preserved and declared separately.

Calibration/validation data and fixed particle banks select DCTR numerical
controls. Generate the untouched nominal claim data only after selection;
never use a failed claim to select controls. Claim-specific deterministic
posterior integration may define precision allowances and functionals, but
cannot tune the sampler or DCTR controls.

## Evidence and interpretation

| Item | Role and rule |
|---|---|
| B/R | Primary comparison: exact49-component endpoint with full DCTR proposal force versus independent same-law posterior integration. |
| C/D | Separate primary comparison: frozen finite DCTR endpoint/force versus independent integration of that same finite endpoint. Run after its own reference and target checks. |
| D/R and C/R | Preserve model approximation and total inference error separately. D/R disagreement cannot veto continuation of C/D. |
| Posterior equivalence | Frozen master functionals: two standardized means, two centered second moments, cross moment (margin.1), six marginal CDFs at reference q10/q50/q90 (margin.02), two predictive probabilities (margin.03). Use simultaneous uncertainty and the existing modern mixing/MCSE rules. |
| Reference precision | Every retained functional's numerical allowance must be <=one tenth its equivalence margin; require refinement contraction. A disagreement or inadequate precision blocks that comparison only. |
| Numerical validity | Nonfinite accepted states/targets, invalid accounting, changed sources/data, wrong prior/chart, missing canonical stages, failed reversibility or an unqualified endpoint are continuation vetoes for the affected computation. Preserve failures; do not relabel them as support. |
| Score disagreement, acceptance, loss, runtime | Explanatory or nomination diagnostics, never evidence of posterior accuracy or convergence. |
| Failed tuning, mixing or equivalence | Reject or leave the candidate inconclusive. Use only a predeclared repair within remaining attempts/budget; do not reject the research direction. |

Use the public `tune_hmc_kernel` with the repository-issued typed position-only
runner binding. A mechanics wrapper cannot issue tuning authority. Reuse the
master's frozen tuning and retained-look policy rather than inventing a shorter
claim after observing costs. Each arm gets its own exact target identity,
controls, tuning artifact, independent reference and retained history.

Construct the cheap adversary set before interpreting usefulness: prior-center
plug-in prediction (no fitting), prior predictive integration (no data learning),
and the existing moment-collapsed Gaussian-mixture UKF posterior (cheap
classical approximation). Compare posterior/predictive error conditionally on
nominal and stress situations; report any supported heuristic dominance as a
usefulness veto. One nominal short scope cannot establish practical superiority,
default readiness, broad model coverage or KSC accuracy for actual SV.

## Defaults, audit and budget

| Choice | Provenance, rationale, risk and earliest check |
|---|---|
| T2 | Smallest temporal KSC case with a49-term exact reference; scope convenience, not a representative long series. Exact mixture recursion verifies timing/covariance. |
| Existing prior/chart | Existing KSC adapter, not transferred from an unrelated model. Check prior/Jacobian identity and parameter boundaries. FP32 chart saturation must remain a flagged numerical failure. |
| N1024 and full canonical controls | Current all-model initial-scope hypothesis. Exact K=N chunk policy, Contract E–Chol, UKF lifecycle, GenUT and both caps required. Direct engine parity verifies the actual call chain. |
| Protection values | Shared four-multiplier non-harm curve on disjoint nominal/stress calibration and validation. First stable adjacent pair under the existing1e-3 rule; inherited values are unselected hypotheses. All computed moment/status diagnostics are retained. |
| TF32/XLA/GPU | Repository execution policy. Record verified growth, dtype, compilation cost and allocator use; run GPU commands escalated. |
| New fixed seeds | Declared reproducibility choices, not statistical replication. Preserve every bank and data hash. Tiny fixed-bank generation is an explicit CPU diagnostic exception, with no learned training. |
| Reference integration | Refine physical posterior quadrature and every functional. Exact likelihood arithmetic does not by itself certify posterior integration or finite-DCTR integration. |

Initial M03 family allocation:7200 GPU seconds and1800 CPU-worker seconds,
including compilation, tests, failures, reference work, tuning and analysis.
Global charges before this scope:9410.299116664915 GPU and10951.767909163998 CPU
seconds, against226800/54000. Reserve at most60 CPU seconds for focused reference
tests,120 CPU/180 GPU for consumer qualification,250 CPU/350 GPU for controls,
and150 CPU/250 GPU for posterior/reference preparation. The remainder is for
actual tuning, retained comparisons and analysis. Stage allocations may be
redistributed within the unchanged family and campaign caps with a recorded
reason; no unrecorded extra allowance. Measure complete B/C transitions early
and reserve the full next look before launching. No job exceeds four hours;
each command also carries a smaller stage-specific wall/CPU ceiling.

At most three substantive candidates per new scope. The first exact-reference
and consumer checks are not inference attempts; failed computations still
consume budget. One serious GPU scope at a time, GPU1
GPU-68251639-fe82-8f81-3ccc-2953c32e805b. Every launch uses a fresh root
`docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-*`, preserved
source snapshots, exact command/environment/seeds and accounting. CPU tests
set CUDA_VISIBLE_DEVICES=-1. Numerical entry point:
`scripts/prepare_surrogate_hmc_ksc.py`; retained execution must be wired through
the public master once its prerequisites are checked.

Skeptical audit before implementation: a moment-collapsed UKF baseline would
be wrong for the exact mixture target, so the independent49-component reference
is required. Source time-zero and transition-first conventions agree here only
because the initial law is stationary; test its parameter dependence explicitly.
Generating transformed observations avoids silently changing an offset. Tests,
control curves and short transitions cannot answer the posterior question.
The M09 non-additive stochastic-completion consumer is currently missing, so
it is not a ready alternative to this additive-Gaussian-state family. Do not
approximate its process law simply to fit the current canonical callback API.

Pre-mortem: the run can pass mechanics while sampling the wrong prior, mixture
mean shift or endpoint. Independent source-density, recursion and chart checks
catch those first. It can fail because of finite-chart saturation or poor
canonical controls; record that as a numerical/tuning failure, not evidence
against imperfect-force HMC. It can exhaust the family budget during preparation;
the next complete-look forecast must be recorded before any inference launch.
