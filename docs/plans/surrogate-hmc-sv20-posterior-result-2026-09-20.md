# SV20 posterior reference result — 2026-09-20

The independent numerical posterior reference passes all predeclared checks.
This establishes a comparison reference for testing surrogate HMC on actual
SV20; no SV20 HMC chain has yet run. The target is the stationary Gaussian
log-volatility model with raw Gaussian returns, independent Uniform(.1,.9)
priors for gamma and beta, and the probit parameter chart.

The command in `surrogate-hmc-sv20-posterior-2026-09-20.md` ran through the
master program. Evidence is in
`artifacts/surrogate-hmc-all-models-20260920-sv20-posterior-01`, especially
`P3-SV20-posterior/report.json`, the source snapshots, six archived quadrature
grids and the run manifest. The worker completed successfully; the master
correctly reports all-model implementation incomplete.

All thirteen posterior/predictive functions passed both contraction and
precision criteria. Empirical numerical allowances range from 1.14e-5 to
1.35e-3, each below one tenth of its declared equivalence margin. Both latent
refinements changed full-prior log likelihoods by at most 1.43e-14. Domain-mass
checks passed. These allowances are empirical refinement estimates, not
rigorous quadrature error bounds. Descriptive posterior means are gamma
0.45418 and beta 0.59912, with standard deviations 0.21178 and 0.14193.

The measured forecast was 475.33 GPU seconds / 444.20 CPU-worker seconds.
Actual master charges were 152.853661 GPU seconds / 142.927857 CPU-worker
seconds, including process overhead. TensorFlow peak allocated memory was
848,696,064 bytes. The run used GPU1, FP64 XLA, TF32 disabled and verified
memory growth. Fifty-one preceding numerical/orchestration tests passed.
Cumulative campaign use is 8,440.075283 GPU seconds and 8,906.644331
CPU-worker seconds under the unchanged 63/15-hour caps.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Qualify model posterior reference | All 13 refinement checks pass | Full-prior latent and domain checks pass | Empirical rather than rigorous error allowance | Qualify actual-SV full DCTR consumer | Surrogate-HMC correctness, efficiency or all-model coverage |

| Inference status | Finding |
|---|---|
| Hard veto screen | No reference invalidity detected under declared checks |
| Statistically supported ranking | None; no sampler comparison ran |
| Descriptive-only differences | Numerical grid changes and cost |
| Default-readiness | Not evaluated |
| Next evidence needed | Actual consumer wiring, SV-specific control selection, public HMC tuning and retained comparisons |

Post-run red team: correlated quadrature errors could survive refinement.
Independent short-horizon checks and analytical-score finite differences from
the preceding likelihood stage reduce this risk, but they are not an exact
long-horizon oracle. A finer independent reference violating the recorded
allowances would overturn this qualification. Heuristic dominance is not run.
The separate MCSE diagnostic repair remains necessary before new inferential
decisions; closed LG1 and C01 results remain unchanged.
