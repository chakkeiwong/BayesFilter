# KSC finite-target surrogate-HMC result

The independent C arm completed public tuning and 2,000 retained draws per
chain across four chains. **C/D and C/R are INCONCLUSIVE.** All twelve tuning
verification checks and all eight retained chunk health checks passed. The
final Monte Carlo uncertainty screen failed for beta's median-CDF functional.
Separately integrated finite-target evidence gives **D/R EQUIVALENT** under
the declared margins and empirical quadrature allowances. This does not
certify the sampled posterior or validate surrogate HMC across all models.

The goal is testing whether endpoint-corrected HMC using the full canonical
analytical DCTR position-only proposal force produces reliable parameter
inference. C/D tests sampling of its declared finite target; D/R measures that
target's approximation to the model posterior; C/R combines those questions.
The earlier [B/R experiment](surrogate-hmc-ksc-t2-result-2026-09-21.md) remains
closed and INCONCLUSIVE. Its samples, tuning and looks were not reused for C.

## Target and actual computation

This is the KSC Gaussian-mixture observation model at T=2, N=K=1024, with
stationary AR(1) initialization, innovation scale 1, and independent
Uniform(0.1,0.9) priors for gamma and beta. The chart is
theta=0.1+0.8 Phi(q). The exact model sums all 49 mixture paths; it is distinct
from the actual-SV transformed observation law. The finite endpoint is
`ksc_t2_dctr_physical_fp32_probit_fp64_v1`: full canonical FP32 DCTR physical
log density and prior, promoted to FP64 with the FP64 chart Jacobian. Boundary
rounding uses the declared nearest interior FP32 value. This defines a finite
numerical target; it is not the exact model posterior.

C binds this endpoint independently of its immutable full canonical analytical
DCTR force. The force need not equal the endpoint gradient. Twenty-seven
endpoint checks passed, including direct callable parity, deterministic replay,
reversal, status handling and binding/cache independence. These executable
checks support the actual consumer call chain.

Independent D quadrature integrated the full proper prior on grids with
16, 32, 64 and 128 intervals. All 16,641 finest-grid evaluations were valid;
all thirteen precision and contraction checks passed. Its 13 functionals
match R's scaling, cuts and margins: five standardized moments, six marginal
CDF values and two predictive probabilities. Prediction uses the exact
model's conditional prediction averaged over D's parameter posterior, not
DCTR filter prediction. Numerical allowances are empirical refinement
estimates, not rigorous global error bounds.

## Inference and uncertainty

C independently used the public typed position-field tuner, archived 250
discarded warmup and 1,000 fresh verification transitions per chain, then
retained eight chunks of 250 draws. The only posterior looks were at 1,000
and 2,000 draws per chain. The selected step size was 1.2002106920 with one
leapfrog step and an independently adapted metric. No numerical, endpoint,
force-status, energy, movement or divergence veto fired during retention.

At the final look the maximum parameter R-hat was 1.000690, minimum bulk ESS
2944.70 and minimum tail ESS 3572.76. These are descriptive diagnostics.
The beta median-CDF functional's largest-batch lag correlation was −0.388813,
outside the frozen ±0.3 limit. Other R-hat, ESS and batch-size sensitivity
checks passed. Nine of thirteen intervals lay inside the equivalence margins
in each comparison, and none established a discrepancy. Because uncertainty
qualification failed, those positions do not establish equivalence. At the
first look beta's second-moment functional failed the same screen.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C/D as INCONCLUSIVE | Joint equivalence not established | MCSE qualification failed; numerical chunks healthy | Batch dependence and interval width | Preserve closed looks; continue other scopes | Neither posterior recovery nor biased sampling established |
| Record D/R EQUIVALENT | All 13 intervals inside margins | Reference precision/contraction passed | Empirical numerical allowances; short horizon | Keep finite-target evidence separate | General DCTR likelihood accuracy |
| Close C/R as INCONCLUSIVE | Joint equivalence not established | Same MCSE veto as C/D | Sampling and approximation error | No retrospective rescore or extra look | End-to-end posterior accuracy |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No numerical failure; failed MCSE qualification vetoes posterior equivalence |
| Statistically supported ranking | None |
| Descriptive-only differences | ESS, R-hat, interval positions, kernel choices and timings |
| Default-readiness | Not established; all-model validation remains false |
| Next evidence needed | Prospective uncertainty-qualified replication, longer horizons and conditional heuristic comparisons |

The prior-center prediction, prior predictive and moment-collapsed mixture-UKF
heuristic comparisons remain NOT_RUN. No superiority or usefulness claim is
supported. Engineering checks, numerical execution and scientific inference
remain separate: the first two passed for this scope; the chain comparisons
did not establish equivalence.

## Execution and review

The [prospective plan](surrogate-hmc-ksc-finite-target-2026-09-21.md) records
exact commands, assumptions, seeds, stop conditions and the complete forecast.
The [C manifest](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-finite-inference-01/run-manifest.json)
and [report](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-finite-inference-01/report.json)
preserve 89 executed source files, tuning, every chunk and both uncertainty
assessments. Exit code was zero. Execution used tftwogpu, escalated RTX4080SUPER
GPU1 access, verified memory growth, XLA and TF32 for DCTR, and FP64 HMC
integration. Allocator peak was 18,499,328 bytes. C cost 1,769.407618 CPU-worker
seconds and 1,856.154848 GPU seconds, within its 3,100 / 3,300 reservation.
D reference construction cost 476.793396 / 494.054665 seconds; endpoint
qualification cost 118.217901 / 122.684598 seconds.

Post-run review verified archived source and data checksums, reference grids,
tuning bindings, sample checksums and continuation order. Master reporting
merges independent C evidence with preserved B/R evidence. Seventy-one focused
CPU regressions passed, including corrupted archives, invalid uncertainty,
repeated closed looks and C/D passing while D/R fails. Reporting-only P0
authorizes no new sampling.

The strongest alternative explanation for the inconclusive result is a noisy
finite-batch diagnostic and conservative multiplicity-adjusted intervals,
rather than sampler bias. This does not justify relaxing a frozen screen after
seeing results. Prospectively qualified uncertainty and fresh independent
evidence could resolve the question. The weakest evidence is the short horizon
and one nominal dataset, with heuristic comparisons absent. This result does
not reject surrogate HMC as a research direction.
