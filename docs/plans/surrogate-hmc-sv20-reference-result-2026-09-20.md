# Actual-SV20 likelihood reference result

The actual stochastic-volatility likelihood passed all nine predeclared
qualification checks on the twelve parameter probes. This prepares an
independent comparator for testing surrogate HMC; no SV posterior or DCTR
sampler has yet been qualified.

The target is the stationary latent AR(1), with sigma=1 and raw returns
`y_t | x_t ~ Normal(0, beta^2 exp(x_t))`. The implementation integrates this
law by finite scalar-state quadrature and differentiates the same finite
program analytically, including stationary initialization. The physical prior
is independent Uniform(.1,.9), and the two probit coordinates include the
prior and Jacobian exactly once. The quadrature is an empirical approximation
to the continuous-state likelihood, not an exact likelihood certificate.

The [plan](surrogate-hmc-sv20-reference-2026-09-20.md) contains the exact command,
defaults, derivations, thresholds and budget. The master launched
`P3-SV20-reference --scope SV20` on GPU1 with FP64, XLA and verified memory
growth. Full source snapshots, data, logs, manifests and numerical arrays are
preserved in
[the run directory](artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01).
The [worker report](artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference/report.json)
records the individual checks. Worker exit 0 and master exit 2 mean successful
reference preparation with incomplete all-model HMC testing.

| Check | Observed | Predeclared limit |
|---|---:|---:|
| Final same-domain mesh change, log likelihood | 3.55e-15 | 1e-6 |
| Final domain expansion, log likelihood | 7.11e-15 | 1e-6 |
| Mesh / domain relative score change | 2.98e-15 / 1.74e-15 | 1e-5 |
| Maximum predicted lost domain mass | 2.22e-16 | 1e-9 |
| Independent T1 / T2 integral difference | 2.89e-15 / 9.77e-15 | 1e-6 |
| Independent Hermite refinement, largest change | 2.19e-11 | 1e-6 |
| Total score finite-difference relative error | 1.12e-9 | 1e-5 |

Replay, row independence and the chart/prior chain rule also passed. Twelve
corner and interior probes constrain the error at those points; they do not
qualify integration throughout the parameter prior. The next reference stage
must check posterior integration and latent-grid refinement across that
domain. The actual-SV DCTR consumer also needs its own qualification; the
older KSC factory computes a different observation law.

Six CPU numerical tests and 47 master/restart regressions passed. Earlier
regression attempts exposed an incremental-only accounting-reader bug and
two obsolete job-list expectations; both were repaired before execution.
The tests consumed 45.635 CPU-worker seconds of their 60-second allowance.
The reference job and master used 10.430 GPU seconds and 14.596 CPU-worker
seconds; worker wall time was 9.327 seconds. Peak TensorFlow allocator use
was 135,203,072 bytes. These timings are descriptive. Cumulative campaign use
is 8,287.222 GPU seconds and 8,748.255 CPU-worker seconds, within 63/15 hours.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Qualify likelihood at the declared probes | Mesh, domain and independent-integral checks pass | Finite output, mass, derivative, replay and row checks pass | Full-prior integration and untested datasets | Qualify posterior reference and actual-SV consumer separately | Posterior recovery or surrogate-HMC validity |

| Inference status | Finding |
|---|---|
| Hard veto screen | No declared reference veto fired |
| Statistically supported ranking | None; no sampler comparison ran |
| Descriptive-only differences | Timing and allocator usage |
| Default readiness | Not established; FP64 independent reference exception |
| Next evidence needed | Full-prior posterior refinement, scoped DCTR qualification and actual inference |

Post-run red team: a common likelihood error could survive grid refinement.
Independent raw/transformed-density tests and T1/T2 Hermite integration reduce
that risk, while finite-difference checks test the finite-program score.
They do not prove long-horizon integration uniformly accurate. A failure of
full-prior or claim-data refinement would restrict this qualification to its
tested probes and prevent posterior comparisons until repaired.
