# Actual-SV20 independent reference preparation

The research goal is testing surrogate HMC with an imperfect full canonical
DCTR proposal field. This stage prepares an independent actual-SV likelihood
reference. It does not run HMC, select DCTR controls or claim posterior accuracy.
The poor calibration of the existing MCSE veto remains a separate repair item.

## Target and evidence contract

Scope M02/SV20, attempt 1: x0 is stationary Normal(0,1/(1-gamma²)),
x_t=gamma*x_(t-1)+epsilon_t, and y_t=beta*exp(x_t/2)*eta_t, with independent
standard-normal innovations and transition before observation. Infer gamma
and beta with independent Uniform(.1,.9) priors. Preserve the repository's
chart `gamma,beta=.1+.8*Phi(z)` from
`bayesfilter/testing/exact_sv_sgqf_neutra_target_tf.py:110`. Prior plus chart
Jacobian is exactly the two standard-normal log densities in z; include it
once. This is the actual observed-return law, not the KSC mixture approximation.
No Zhao-Cui algorithm or source-faithfulness claim is involved.

The independent reference integrates a scalar-state density on a fixed
trapezoidal grid. Initial mass is the stationary Gaussian density times grid
weights. Each transition uses the Gaussian density times destination weights.
Neither initial mass nor transition rows are renormalized to hide domain loss.
After each observation normalize filtering weights and accumulate the removed
log normalization. Emit predicted domain mass loss and analytical derivatives
of this finite quadrature. Numerical integration approximates the model; it is
not declared an exact likelihood or a posterior reference yet.

For transformed v=log(y²)-2log(beta)-x, the log density of log(eta²) is
`v/2-exp(v)/2-log(2*pi)/2`. The change of variables sums both signs of y, giving
`log p(y|x)=log p(log(y²)|x)-log(abs(y))`. This difference is independent of
parameters. Check both representations using identical nonzero raw data.
Since x0 is stationary, x1 has the same stationary marginal; integrating the
initial transition analytically preserves the declared observation ordering.

Analytical derivative checks include stationary initialization:
`d_gamma log p(x1)=gamma*x1²-gamma/(1-gamma²)`;
`d_gamma log p(x'|x)=(x'-gamma*x)*x`;
`d_beta log p(y|x)=(y²*exp(-x)/beta²-1)/beta`.
Propagate derivatives through every prediction and normalized update. Compare
against central differences of the same finite grid, not against a DCTR score.

Freeze calibration, validation and claim datasets separately, generated at
(gamma,beta,sigma)=(.6,.4,1), using seeds [20260920,11100+j],
[20260920,11200+j], [20260920,11300+j], j=0,1,2 for initial/process/observation
noise. Raw observations are rounded to FP32 once and promoted to FP64 for the
reference. No log floor is allowed; a zero or nonfinite observation is a veto.
Preserve all generated arrays and hashes. Reference qualification uses
calibration data only; other partitions are frozen for later scoped work.

The twelve physical probes are the nine Cartesian combinations of gamma,beta
in {.1,.5,.9}, truth (.6,.4), and (.2,.8),(.8,.2). Use four-row batches and
state grids (intervals,half-width)=(128,12),(256,12),(512,12),(512,16),
(1024,16),(1280,20). The last domain comparison keeps spacing 1/32 fixed.
Pass when the final within-domain refinement and expanded-domain comparison
both change log likelihood by at most 1e-6 and physical score by at most
1e-5*(1+abs(finer score)), with final weighted predicted mass loss <=1e-9.
These are empirical probe checks, not uniform or rigorous quadrature bounds.

Independent authorities: direct one/two-time Gaussian integration with
Gauss-Hermite nodes, raw/transformed density change of variables, and central
differences. These are constructed from the law and evaluated at interior and
boundary probes. They do not replace later heuristic-dominance comparisons
for HMC. Required checks also include row independence, replay, chart/prior
algebra and finite normalized filtering weights. Wrong law, missing initial
dependence, mismatched data, numerical invalidity or failed independent checks
veto reference qualification. Coarse-grid failure triggers the planned finer
grids; only final failure stops this stage.

## Default audit, budget and next decision

Uniform priors/probit chart are preserved model-specific choices. FP64/GPU/XLA
is an explicit independent-reference precision exception, not a change to
FP32/TF32 DCTR. Stationarity and sigma=1 are the existing synthetic law; test
their density/derivative consequences. Trapezoidal integration is a transparent
reference hypothesis; smooth Gaussian kernels and log-normal observations
motivate it, while domain and mesh refinement expose truncation/aliasing.
Probe points cover corners and interiors but cannot certify the whole prior.
Tolerances are small relative to contemplated posterior margins, yet remain
empirical qualification choices. No current posterior margin is changed.

Pre-mortem: an internally consistent reference could use the KSC law, omit the
stationary derivative, silently normalize a truncated transition, or agree at
selected probes while failing elsewhere. Independent law/derivative tests and
explicit domain-loss reporting address the first three; a later full-prior
posterior refinement is still required for the fourth. The skeptical audit
passes this bounded preparation, not inference readiness.

Initial family allocation remains 2 GPU hours / .5 CPU-worker hours. This
stage uses at most 450 GPU seconds and 700 CPU-worker seconds, including one
localized infrastructure retry, plus at most 60 CPU seconds for focused tests.
Charge P3 and the unchanged 63/15-hour campaign. Stop on resource exhaustion;
preserve failures and do not expand the grid after seeing the outcome.
Use escalated GPU access and verified memory growth before device creation.
CPU tests explicitly hide GPUs. The executable master will expose this
independent reference job; a failed LG1 branch cannot cancel it.
The independent short-horizon Hermite calculation is a one-off CPU/eager
reference exception. Repeated filtering kernels run on GPU with XLA.

Exact command (trusted GPU1; master enforces stage and cumulative budgets):

```bash
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage P3-SV20-reference --scope SV20 --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01
```

The worker returns zero only when all likelihood checks pass. The master
returns two after a completed preparation because all-model HMC testing is
still incomplete; this is distinct from a worker failure.

Artifacts: `docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01`;
result: `docs/plans/surrogate-hmc-sv20-reference-result-2026-09-20.md`.
After qualification, the next work is full-prior posterior reference refinement
and a separately qualified actual-SV DCTR consumer with correct stationary
initialization. No older KSC factory is presumed eligible.
