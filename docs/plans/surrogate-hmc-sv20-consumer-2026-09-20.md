# Actual-SV full DCTR consumer qualification — 2026-09-20

The campaign question is whether imperfect full canonical DCTR proposal fields
support endpoint-corrected HMC parameter inference. This stage checks whether
the actual SV20 consumer computes the declared model and reaches the shared
canonical implementation. It qualifies mechanics only, before SV-specific
control calibration and public HMC tuning. Previous LG1/C01 decisions stay
closed. A force/endpoint discrepancy is explanatory, never a continuation veto.

## Target, derivation and evidence contract

Use the frozen calibration data from
`artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference`.
The target has x0~N(0,1/(1-gamma^2)), xt=gamma*xprev+epsilon with unit process
variance, and yt|xt~N(0,beta^2*exp(xt)). Gamma and beta independently have
Uniform(.1,.9) priors and theta=.1+.8*Phi(z). Transition precedes observation.
The endpoint uses the independently qualified FP64 state quadrature on 1024
intervals over [-16,16], with the same once-rounded raw observations.
This is a numerical model reference, not an analytically exact likelihood.

The DCTR adapter uses log(y^2) for its flow geometry. For eta~N(0,1),
log(eta^2) has mean psi(1/2)+log(2) and variance trigamma(1/2)=pi^2/2, from
the first two derivatives of log E[(eta^2)^s]=s*log(2)+log Gamma(s+1/2)
-log Gamma(1/2). Thus the moment observation map is x+2log(beta)+mu,
Jacobian 1 and covariance pi^2/2. The canonical flow uses these exact moments;
the actual weight density remains the raw Gaussian return density. With
v=log(y^2)-2log(beta)-x, log p(y|x)=-log(beta)-x/2-exp(v)/2-log(2pi)/2
and its directional derivative is (exp(v)-1)*(dx/2+dbeta/beta).
No KSC mixture replaces this law. A zero raw return cannot use this flow
coordinate and must be rejected explicitly, with no log floor or clipping.

All initial-state dependencies enter the existing stationary callbacks:
P0=1/(1-gamma^2), dP0=2*gamma*dgamma/(1-gamma^2)^2. Initial particles are
fixed standard Gaussian base draws transformed by the shared initialization.
The adapter supplies analytical tangents; autodiff/finite differences are
independent test authorities only. The chart adds its Jacobian exactly once;
the uniform density cancels the width, giving log phi(z) and derivative -z.

Primary qualification requires actual endpoint-to-canonical-call parity,
deterministic replay, batch permutation and independence, callback derivative
checks, raw-observation density agreement, stationary initialization, prior/chart
agreement, correct force sign/status, endpoint independence, involution and
per-chain energy/cache checks. Failures of these checks, source/data drift,
nonfinite endpoint, a numerics-invalid artifact or budget overrun veto this
stage. An unhealthy original DCTR field is a repair trigger; record its status
and any fallback rather than interpreting fallback-only mechanics as a healthy
force. Acceptance and force/reference differences are explanatory only.

Artifacts: a versioned consumer manifest, exact source snapshots, all original
statuses, typed public-tuner binding, timing/allocator diagnostics, check report,
run manifest and accounting. No reference qualification, mechanics test or
short transition may claim posterior recovery, convergence or default readiness.

## Default audit, budget and skeptical audit

| Choice | Provenance and status | Justification / risk | Early check |
|---|---|---|---|
| Full Contract E / GenUT / pairwise / coordinate cap, N1024, K1024 | Current canonical engine and chunk policy | Avoids lane forks; fixed sample approximation remains imperfect | Actual call parity and status |
| Protection multiplier 1, ten substeps and finite transport iterations | Existing post-reset controls, warm-start hypothesis only | Bounded mechanics starting point, not transferred tuning | Original-force health; mandatory SV-specific non-harm curves before inference |
| Ridge/damping/trust limits from that hypothesis | Class C evaluation candidate, not a default | Existing numerical protection rationale motivates testing; no SV justification yet | This stage cannot admit controls; dedicated SV calibration remains owed |
| FP32/TF32 force, FP64 integrator and endpoint | Repository GPU policy / independent reference exception | FP32 discontinuities can affect numerical reversibility | Measured round trip <=32 FP32 ulps on scaled inputs |
| Four positions, two steps, epsilon .02, identity mass | Mechanics convenience | Tests wiring only, not tuning or coverage | No convergence claim; public tuner required later |
| IID fixed draws / partition seed | Baseline hypothesis | Finite-design sensitivity | Manifest hashes, replay and later D/R comparison |

At stage entry campaign usage is 8440.075283 GPU / 8906.644331 CPU-worker
seconds. SV-family usage is about 163.284049 GPU / 218.620107 CPU-worker
seconds, including preparation tests. Allocate at most 600 GPU / 450 CPU-worker
seconds to consumer qualification across one execution and one localized
infrastructure retry, plus 60 CPU-worker seconds for focused tests. All charges
remain within the existing family 7200 GPU / 1800 CPU-worker seconds and
campaign 226800 GPU / 54000 CPU-worker seconds. GPU1 only, memory growth,
XLA, max peak allocated memory 4 GiB. Stop before exceeding allocations.

Pre-mortem: the old KSC adapter or a logistic chart could yield finite but wrong
results; direct raw-density and chart tests catch these. Omitted stationary
derivatives could falsely validate a force; test callbacks independently.
All force failures replaced by zero could make mechanics pass vacuously;
separate original-force health from guarded-kernel validity. A successful run
cannot establish HMC recovery without tuning, calibrated MCSE checks, retained
comparisons and uncertainty. Heuristic adversaries (prior-only, bootstrap PF,
moment filter) and conditional scenarios will be constructed at the inference
stage; their absence here prevents promotion.

Skeptical pre-execution audit: the endpoint comparison uses the actual raw
return law and stationary initial law, with the established prior/probit chart.
The consumer calls the shared complete canonical filter without reduced flow
or reset. Independent reference precision is qualified. The warm-start controls
are explicitly unselected; this stage's criteria are engineering criteria only.
No material flaw in this bounded qualification plan remains.

Expected command after focused tests (escalated, GPU1):
`python scripts/run_surrogate_hmc_program.py --stage P3-SV20-qualify --scope SV20 --reference-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference --posterior-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-posterior-01/P3-SV20-posterior --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-consumer-01`.
Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` with
`CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `BAYESFILTER_PRELOAD_CUSTOM_OP=0` and two
intra-op / one inter-op TF threads. Save the actual command in the manifest.
