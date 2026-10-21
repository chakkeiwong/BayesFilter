# Fresh 70%-target range-bearing run: completed

The requested rerun finished on 20 October 2026 at 03:49 Hong Kong time
(19 October 19:49 UTC). It satisfied the predeclared cumulative sampling rule
after 2,000 warm-up transitions and 1,000 retained draws per chain across four
chains. All 4,000 retained draws are inference samples. There was no extra
discarded validation phase. The recorded worker time was 25,883.80 seconds
(7 hours 11 minutes), within the eight-hour allocation; 2,916.20 seconds remain
unspent. The master stopped because the sampling criterion passed.

| Diagnostic | Completed 70%-target run |
| --- | --- |
| Actual acceptance by chain | 70.2%, 68.0%, 68.4%, 68.7% |
| Maximum modern R-hat | 1.003620 |
| Bulk ESS, two parameters | 1,111; 2,221 |
| Tail ESS, two parameters | 1,688; 1,910 |
| Posterior mean MCSE in q coordinates | 0.011735; 0.004695 |
| Non-finite arithmetic divergences | 0 |
| Retained deterministic force fallbacks | 1 in 4,000 chain transitions |
| Maximum absolute finite energy error | 291,250.50; warning only |

Every recorded chunk passed finite-state, endpoint, movement and archive checks.
Large finite energy errors remain explanatory warnings under the owner's rule;
the MH step is unchanged. The saved `VERIFIED` label means the declared sampler
rule passed. Posterior agreement with an independent reference remains not
checked, and default readiness is not established.

| Decision | Primary criterion | Validity checks | Main uncertainty | Next action | Limits |
| --- | --- | --- | --- | --- | --- |
| Requested rerun complete | Cumulative modern R-hat <=1.01 passed | Recorded chunk health checks passed; finite energy/fallback warnings preserved | Independent posterior agreement and replicated efficiency | Preserve this completed result; SIR continues under the existing master and budget | No exact-posterior or efficiency superiority claim |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No recorded arithmetic, endpoint, movement or artifact veto |
| Statistically supported ranking | None; both completed acceptance settings remain viable under the sampling rule |
| Descriptive-only differences | New observed acceptance 68.0-70.2%, compared with 93.8-94.9% before; new R-hat 1.00362 and bulk ESS 1,111-2,221 after 1,000 retained draws per chain |
| Default-readiness | Not established; reference-posterior agreement is not checked |
| Next evidence needed | Independent reference comparison for approximation accuracy; replicated, hardware-matched runs for an efficiency ranking. Neither is an extra stopping condition for this completed run. |

Terminal evidence: `run/range_bearing/inference/hmc.json`,
`run/range_bearing/inference/public-tuning/fresh-progress.json`,
`run/master.json`, and `budget.json`. The master refreshed its phase plan to
`SELECTED_MODELS_HMC_SAMPLING_COMPLETE`. SIR remains a separate ongoing part of
the existing five-model programme. Trusted process inspection on 20 October
confirmed its controller and worker live, with 4,750 warm-up transitions per
chain and zero retained draws at the latest checkpoint.

Post-run review: actual acceptance is close to the requested 70% and cumulative
R-hat meets the predeclared threshold. This supports completion of this rerun.
The strongest remaining alternative explanation for any apparent efficiency
gain is Monte Carlo variation plus different hardware and retained counts.
The old run used an RTX 5080 and the new run an RTX 4080 SUPER. A replicated,
matched comparison could overturn a descriptive efficiency interpretation;
none is asserted here. The weakest scientific evidence is the unchecked
agreement between the finite particle posterior and an independent reference.

## Preserved launch record

The requested rerun started on 19 October 2026 at 12:23 UTC (20:23 Hong Kong).
The actual worker config requests acceptance 0.70 and has no resume manifest.
It retains T=20, N=4032, K=2016, FP32, TF32 off, the literature prior, the same
prepared data and the selected 16-flow-step filter. GPU/XLA and memory growth
are recorded in `run/range_bearing/inference/manifest.json`. Master PID 628890
and worker PID 628893 were confirmed live. SIR continues on the other GPU.

`manifest.json` records the exact command, Git/source/selection identities,
environment, seeds, plan and final output paths. `preflight.json` records the
successful CPU-only scope and command check. The existing focused configuration
suite passed 24 tests before this launch. `budget.json` caps this newly requested
attempt at eight GPU-worker hours; the previous SIR ledger is unchanged.

| Decision | Primary criterion | Validity checks | Main uncertainty | Next action | Limits |
| --- | --- | --- | --- | --- | --- |
| Run started; continue automatically | Sampling criterion pending | Preflight and startup passed; numerical run pending | Achieved acceptance and cumulative mixing | Complete fresh tuning, adaptive warm-up and cumulative sampling | No success or efficiency ranking yet |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No startup veto; sampling health remains pending |
| Statistically supported ranking | None |
| Descriptive-only differences | Comparison pending; acceptance and ESS will be reported with counts |
| Default-readiness | Not established by this run |
| Next evidence needed | Nomination, retained diagnostics and final result; replication for ranking |

The preserved 95%-target baseline had acceptance 93.8-94.9% and maximum modern
R-hat 1.00746. The new run uses an RTX 4080 SUPER while the baseline used an RTX
5080; raw runtime differences would confound hardware and tuning. A fresh
70%-target run may still realize a different acceptance rate, and satisfying
the sampling rule does not establish posterior agreement with an oracle.
The strongest early alternative explanation for high acceptance will be
limited adaptation or the selected trajectory, which must be checked in the
nomination artifact rather than inferred from the target alone.

The launch-only statements and pending checks above describe the start of the
run; the terminal findings at the beginning of this note supersede them.
