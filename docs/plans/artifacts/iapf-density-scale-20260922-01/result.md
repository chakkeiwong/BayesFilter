# Known-target density-scale diagnosis

The actual TensorFlow fitter can report numerical convergence without learning an exactly representable Gaussian. On both CPU and GPU, all eight shifted d40/d80 cases stop at iteration zero. Fixed objective scaling starts optimization but does not reliably recover the target. The frozen independent R reference recovers all 30 targets to numerical precision.

This is an implementation/stopping diagnosis, not a paper replication or a claim that the original authors' initialized local solver fails. All runs use FP64/XLA; CPU intentionally hides GPUs, and the GPU uses verified memory growth on the RTX 4080 SUPER.

## Mechanism

Let p be the fitted Gaussian density evaluated on N cloud points and b the positive target values. Profiling lambda gives lambda=(p^T b)/(b^T b), residual r=p-lambda b and L=||r||^2/N. Orthogonal projection gives ||r||<=||p||. For parameter eta_j, write s_ij=partial log p_i/partial eta_j. The envelope derivative is

    partial_j L = 2 r^T (p s_j)/N,
    |partial_j L| <= 2 max_i |s_ij| mean_i(p_i^2).

The bound holds for every target b. At the cloud-moment initialization, p_i=exp(-||z_i||^2/2), with mean/log-SD scores z and z^2-1. If the bound is below the absolute tolerance, the current stopping rule cannot distinguish any target. All computed bounds pass on both devices. For d80/seed71 the bound is 4.98e-23 versus tolerance 1e-7; the narrow-target actual gradient is 3.00e-24. This is tiny nonzero arithmetic, not floating-point underflow or a wrong derivative.

A second problem survives fixed scaling. On a finite set of points, increasing variance or moving density away can shrink p and the profiled residual together. Thus decreasing this density loss need not improve Gaussian shape. The bounded optimizer can move toward its box boundary or stop at a low-density point. At d5/seed71/narrow, it moves to an active boundary: loss 4.10e-7, shape residual .599 and KL 5.273. The exact representable target has zero profile loss. This shows failure of this initialized local procedure; it does not prove every initialization or local optimizer fails.

## Conditional observations

Ranges below are the two CPU seeds, descriptive only. GPU values agree to below 1e-10. KL is D_KL(target || fitted Gaussian).

| d | target | baseline KL range | fixed-scale KL range | baseline zero-step failures |
|---|---|---:|---:|---:|
| 5 | shifted_narrow | 5.27347–5.29956 | 5.27349–5.29959 | 0/2 |
| 5 | shifted_broad | 1.2981e-10–2.28377e-10 | 2.17755e-10–3.13548e-10 | 0/2 |
| 10 | shifted_narrow | 4.52869–4.53799 | 5.30552–5.44782 | 0/2 |
| 10 | shifted_broad | 2.45627e-06–3.95654e-06 | 7.86196e-07–1.08336e-06 | 0/2 |
| 20 | shifted_narrow | 2.36022–2.47196 | 4.40246–4.95928 | 0/2 |
| 20 | shifted_broad | 0.624999–0.626019 | 0.439914–0.563809 | 0/2 |
| 40 | shifted_narrow | 2.43829–2.50612 | 4.82476–5.36638 | 2/2 |
| 40 | shifted_broad | 4.2012–4.29133 | 1.21142–1.28375 | 2/2 |
| 80 | shifted_narrow | 5.04138–5.06167 | 6.15296–7.22912 | 2/2 |
| 80 | shifted_broad | 8.3045–8.72272 | 4.53515–6.06519 | 2/2 |

The ten healthy fits are unchanged by fixed scaling to the predeclared 1e-10 tolerance. The R reference's largest center error is 1.079e-15 and variance error 2.665e-15. Its QR initialization already recovers the target; scaling and optimizer choices also differ, so the R/TF comparison does not yet isolate initialization. The exact oracle and QR recover every regime. Some learned narrow-target fits are worse than the unlearned cloud-moment Gaussian, which vetoes promotion on these fixtures.

## Decision and inference

| decision | primary criterion | veto status | main uncertainty | next action | limits |
|---|---|---|---|---|---|
| Diagnose uninformative stopping | 8 cases/device satisfy the predeclared criterion | No arithmetic/source/nonfinite continuation veto | General non-Gaussian prevalence unknown | Isolate initialization/scaling factorial | No marginal-score or filter-accuracy inference |
| Reject fixed scaling as a sufficient repair | Known targets remain inaccurate | Known-oracle/QR heuristic veto | Local basin vs stopping scale | Test QR initialization with the same objective | It remains viable as a numerical scaling component |

| inference status | finding |
|---|---|
| Hard veto screen | Poor known-target recovery blocks promotion; no continuation veto |
| Statistically supported ranking | None; two clouds are diagnostic coverage |
| Descriptive differences | KL, shape, objective and iteration ranges above |
| Default readiness | No default change and no paper/LEDH/HMC admission |
| Next evidence needed | Factorial initialization/scaling; nonexact targets; downstream heldout filtering |

Terminal skeptical review: the same-input GPU comparison, bound and exact Gaussian oracle distinguish this from compiler error, representability failure and literal underflow. The strongest alternative explanation is the baseline's poor starting point/local basin. A successful target-aware initialization would support that explanation, not invalidate the zero-step diagnosis. Weakest evidence: only diagonal Gaussian targets and two clouds; no general filtering benefit is established. All source snapshots and failed-fit outputs are retained. Commands/environment/times are in the launch records and summaries.
