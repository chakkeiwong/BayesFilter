# LG2 exact-reference preparation result

The two-parameter LG2 Kalman endpoint passed a CPU/FP64/XLA comparison with an
independent 40-dimensional joint Gaussian calculation. The target is the
existing stationary LG2 fixture, including both transition scale and process
scale. Both calculations include the parameter dependence of the stationary
initial covariance and the bounded-coordinate Jacobian. This is preparation
for testing surrogate HMC; no LG2 HMC chain or posterior comparison has run.

Three deterministic probes cover the nominal parameter and two opposite
boundary-near parameter combinations. Log-density agreement passed absolute
tolerance 1e-9; gradients passed `atol=rtol=1e-8`. Batch permutation and a
single-row call agreed. The runtime reference solves the stationary Lyapunov
equation; the independent oracle sums 1,025 covariance-series terms and builds
the entire observation covariance. There is no shared Kalman or Lyapunov
implementation in the oracle.

The first check failed at compilation because XLA requires a bounded reverse
loop accumulator. Setting `maximum_iterations` to the known observation
horizon repaired this engineering failure without changing the target or
numerical formula. The second check passed. Both attempts are preserved under
`artifacts/surrogate-hmc-all-models-20260920-lg2-reference-check-{01,02}`; their
manifests record commands, source hashes, environment, CPU-only status, timing
and logs. They consumed 18.979316 CPU-worker seconds of the 120-second
preparation allowance and no GPU time. GPUs were intentionally hidden before
TensorFlow import.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Reference implementation passes these probes | Same-law density and gradient agreement passed | No nonfinite value or batch coupling at the probes | Three probes do not qualify a posterior integral or the GPU consumer | Freeze LG2 data partitions and functionals; qualify the independent posterior reference and full-DCTR consumer within P3 | LG2 surrogate-HMC accuracy, mixing, efficiency or all-model coverage |

| Inference status | Finding |
|---|---|
| Hard veto screen | Initial compilation failure repaired; numerical probes pass |
| Statistically supported ranking | N/A: deterministic implementation check |
| Descriptive-only differences | Timing is diagnostic, not GPU performance evidence |
| Default-readiness | Not established; FP64 exact-reference exception |
| Next evidence needed | Scope-specific DCTR calibration, consumer checks, reference integration, public HMC tuning and retained posterior comparisons |

The strongest alternative explanation is a shared modeling mistake in the
fixture interpretation. The independent covariance-series oracle checks the
stationary law and observation covariance, but the later DCTR consumer must
still be checked against that same initialization, timing, prior and chart.
Finite-posterior accuracy and heuristic comparisons remain untested.
