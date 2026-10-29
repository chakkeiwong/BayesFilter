# SIR NeuTra surrogate-HMC terminal result

The run finished normally at 03:57:44 Hong Kong on 2026-10-23. It reached
the declared cumulative R-hat stopping rule with 2,000 retained draws per
chain, after 2,000 warm-up transitions per chain. Four chains contribute
8,000 inference draws; warm-up is excluded. All post-warm-up draws are
inference draws, with no additional validation-sampling phase.

The sampler reports `VERIFIED`; the corrected master reports
`NEUTRA_POSTERIOR_CRITERIA_MET`. The owner clarified on 2026-10-23 that ESS
is diagnostic, not a convergence criterion. The previous `INCOMPLETE` label
improperly used a precision threshold to block completion. The master now
requires valid completed sampling and cumulative parameter-chart R-hat <=1.01;
ESS and MCSE remain reported as precision diagnostics only. The original
assessment is preserved in [reassessment 001](reassessments/001.json).
No new sampling was needed or launched for this reporting correction.

## Retained diagnostics

These are the SIR parameter-chart coordinates q after applying the learned
map, not the latent NeuTra coordinates z. R-hat is the maximum of the
rank-normalized split and folded rank-normalized split statistics.

| Coordinate | R-hat | Bulk ESS | Tail ESS | Mean MCSE / posterior SD |
|---|---:|---:|---:|---:|
| q1 | 1.009603 | 412.28 | 199.29 | 0.05317 |
| q2 | 1.003647 | 595.13 | 1129.09 | 0.04134 |
| q3 | 1.007047 | 475.48 | 769.04 | 0.04599 |

The first coordinate's tail minimum comes from its upper 95% indicator ESS;
its lower 5% ESS is 1885.08. This describes less effective information about
the upper tail and does not change convergence classification. The latent-coordinate
R-hat used by the sequential controller also passed, with maximum 1.009824.

Observed acceptance by chain is 79.30%, 79.15%, 76.45%, and 75.90%
(pooled 77.70%). Tuning targeted 70%; retained acceptance is descriptive,
not a convergence criterion. Recorded divergences and force fallbacks are
both zero. All finite-state, endpoint-validity, chain-movement and warm-up
readiness checks passed. Maximum absolute retained energy error is 6.44571;
finite energy error remains explanatory only.

## Target, execution and evidence

The experiment tests surrogate HMC: the frozen LEDH force drives proposals,
and the independent fixed bootstrap-filter endpoint determines the MH
correction. The recorded endpoint is
`independent_fixed_bootstrap_4x4032_approximate_posterior` with the scientific
SIR prior. These diagnostics concern that fixed numerical posterior; they
do not establish equality to the exact SIR posterior. Posterior/reference
agreement remains `not_checked`.

Scope: SIR T50, N4032, ordinary FP32 filter, TF32 disabled, GPU/XLA; learned
map and HMC coordinates are FP64. GPU memory growth was configured and
verified before initialization. The frozen canonical IAF and target binding
were reused throughout. Worker execution took about 9.21 hours, including
kernel tuning, warm-up and retained sampling. All previous attempts remain
charged: 8.4580 of the 24 HMC hours and 8.5678 of the 16 training hours remain.

The [HMC result](hmc-result.json) contains the terminal diagnostics and
assessment. The [worker manifest](019-hmc-neutra/manifest.json) records the
command, Python environment, Git commit, source hashes, GPU/memory policy,
plan, job and result paths. The [job](019-hmc-neutra/job.json) binds the frozen
map and tuned filter; the [sampler manifest](019-hmc-neutra/public-tuning/manifest.json)
records seeds, target signature, stopping configuration and retained tensors.
The [terminal phase](next-phase.json) records the remaining budgets.

## Decision

| Decision item | Finding |
|---|---|
| Decision | Sampling completed; the declared convergence diagnostic passed. |
| Primary criterion | Valid completed sampling and all three cumulative parameter-chart R-hat values <=1.01. |
| Veto diagnostics | No recorded numerical/endpoint/movement failure; no divergence or fallback. No continuation veto fired. |
| Main uncertainty | Upper-tail Monte Carlo precision and unchecked exact-posterior agreement. |
| Next justified action | Retain the inference samples; report ESS and MCSE diagnostically. No further sampling is required by the master. |
| What is not concluded | Exact-posterior accuracy, mathematical proof of convergence, Gaussianity, superiority or default readiness. |

| Inference status | Finding |
|---|---|
| Hard veto screen | Passed the recorded health checks. |
| Statistically supported ranking | None; this is a single-method run. |
| Descriptive-only differences | Acceptance, runtime and the earlier provisional whitening diagnostics do not rank methods. |
| Default readiness | Not established. |
| Next evidence needed | None for the declared completion rule; independent posterior agreement would be needed for an accuracy claim. |

Terminal audit: the saved outputs support process completion, sampler
completion and the declared R-hat criterion. ESS and MCSE have no pass/fail
role. All 35 focused CPU tests passed in 15.05 seconds; the real reassessment
command preserved sampler outputs and charged budgets without importing
TensorFlow or launching a worker. The strongest alternative explanation for
apparently satisfactory mixing is slow exploration of the upper tail or an
unvisited region; R-hat and finite-run diagnostics cannot rule it out. Longer
ordinary sampling could change the tail diagnosis. The weakest accuracy
evidence is the missing comparison with an independent posterior reference.
The reporting criterion was corrected after execution. The R-hat threshold,
target, map, sampler, recorded runtime sources and prior numerical evidence
remain unchanged.
