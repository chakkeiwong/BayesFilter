# q=20 SSL-LSTM NeuTra Sequential Real-Run Plan

Date: 2026-07-22  
Tier: serious local GPU/XLA research campaign  
Status: `READY_FOR_GPU_SMOKE_CAP_AND_GPU_LANE_PENDING`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Main question | Do the two independently trained q=20 `(32,32)` NeuTra charts support sequential fixed-kernel HMC that passes modern multi-chain warm-up and retained-sample diagnostics? |
| Exact inputs | The distinct `seed-a` and `seed-b` admitted frozen transport receipts under `docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/arch-32x32/`. |
| Candidate mechanism | TensorFlow/TFP four-chain identity-mass HMC in each frozen NeuTra chart, using the canary kernels only as starting hypotheses. |
| Warm-up promotion | After at least 2,000 warm-up transitions per chain, the latest 1,000-transition window has finite maximum rank-normalized split/folded R-hat `<=1.05`. |
| Retained promotion | At a cumulative retained checkpoint, every parameter has maximum rank/folded R-hat `<=1.01`, bulk ESS `>=400`, and tail ESS `>=400`. |
| Hard vetoes | Nonfinite state, target value, score, log-acceptance ratio, or target-status field; nonzero target status; false `valid_pre_regularized_score`; any chain never moving in a chunk; exposed positive native divergence; archive failure; GPU memory-policy failure; resource cap. |
| Explanatory only | Acceptance probability/rate, runtime, allocator bytes, continuous R-hat/ESS values before a pass, and chart differences. Native divergence unavailability is not zero divergences. |
| Repair trigger | A chart that reaches the cap without passing is a fixed-kernel/transport repair trigger. No in-run step-size or transport retraining is allowed. |
| Continuation veto | Invalid target/status instrumentation, broken archive lineage, source drift, GPU policy failure, or exhausted campaign cap. One chart failing does not reject the other chart or the NeuTra direction. |
| Artifact | A versioned output root containing separate warm-up and retained TensorFlow tensor chunks, per-checkpoint diagnostics, exact manifest, and result note. |
| Nonclaims | No posterior oracle, stationarity proof, model adequacy proof, predictive validation, architecture ranking, transport ranking, or default-readiness claim. |

## Sequential Policy

The controller policy identifier is `bayesfilter_neutra_sequential_hmc_v1`.
Warm-up draws are archived but never included in posterior estimates.

- Warm-up chunk: 500 transitions per chain.
- Minimum warm-up: 2,000 per chain.
- Warm-up readiness window: latest 1,000 per chain.
- Maximum warm-up: 10,000 per chain.
- Retained chunk: 500 transitions per chain.
- Minimum retained checkpoint: 1,000 per chain.
- Maximum retained: 10,000 per chain.
- Four chains, TensorFlow/TFP, XLA enabled, GPU memory growth verified before
  logical-device initialization.

The 2,000/1,000/10,000 counts and R-hat thresholds are repository policy. The
500-transition cadence is a convenience choice for stopping granularity. ESS
`400` is an inherited confirmation threshold and is treated as a reviewed
starting criterion, not a universal theorem.

## Kernel Hypotheses

| Chart | Step size | Leapfrog steps | Provenance/status |
| --- | ---: | ---: | --- |
| A | `0.5656854249492381` | 4 | Canary-nominated hypothesis; 64-draw confirmation was finite/moving but had two chain acceptances above the old canary band. |
| B | `0.5656854249492381` | 4 | Canary-nominated hypothesis; 16-draw midpoint was finite/moving but chain acceptance estimates were dispersed. |

Acceptance is not a convergence criterion in the real run. These kernels are
not described as previously admitted or frozen by the failed canary gate.

## Default And Assumption Audit

| Choice | Provenance/status | Why used | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| Two charts | Independent training seeds | Checks sensitivity to learned coordinates | Two seeds do not estimate broad training uncertainty; report separately |
| Identity mass | Canary HMC configuration | NeuTra is intended to simplify geometry | Residual anisotropy; R-hat/ESS failure triggers later metric/transport repair |
| Four starts | Existing q=20 HMC starts | Preserves continuity with canaries | Poor coverage; warm-up R-hat and latest-window check expose persistent separation |
| Fixed kernel through run | Required for interpretable retained evidence | Prevents adapting on retained samples | Bad kernel wastes budget; sequential warm-up stops or reaches cap |
| Target-status instrumentation | New reporting method over the same UKF value/score calculation | Required hard-veto telemetry | If instrumentation changes value/score, parity tests veto execution |
| GPU/XLA/TF32 | Repository default and completed q=20 canaries | Same execution class as target evidence | Device/source drift; manifest and preflight fail closed |

## Compute Budget

Measured chart-A confirmation cost was
`2.158400593840876 seconds/transition-leapfrog`. The policy-minimum two-chart
workload is

`2 * (2000 warm-up + 1000 retained) * 4 leapfrogs * 2.158400593840876`

`= 51,801.61425218102 seconds = 14.3893 GPU-hours`.

Applying the existing 1.5 non-preemptive/contention margin and rounding upward
to 300 seconds gives a proposed campaign cap of `77,700 seconds`
(`21.5833 GPU-hours`). This is a cap, not expected use; sequential stopping
returns unused budget. It covers both charts together and does not authorize
training, tuning search, or another candidate family.

The long launch is pending explicit user approval because this cap materially
expands compute beyond the completed `23,400 s` tuning campaign. Implementation,
unit tests, CPU-hidden contract smokes, and a tiny GPU mechanics smoke remain
inside routine preparation.

## Skeptical Pre-Execution Audit

- Wrong baseline: checked; both charts use the same q=20 target/data and
  independently trained `(32,32)` transports.
- Proxy promotion: repaired; canary acceptance bands nominate kernels only.
  Real promotion uses modern R-hat and ESS on retained draws.
- Missing stop: repaired; warm-up and retained counts are each capped at
  10,000 per chain, with a cumulative wall-clock cap.
- Unfair comparison: checked; charts are replications, not ranked candidates.
- Hidden assumptions: recorded above, including identity mass, cadence, starts,
  ESS threshold, and fixed kernel.
- Stale context: checked; the historical retained harness is ineligible because
  it uses fixed burn-in and lacks the shared sequential policy.
- Environment mismatch: the new launcher must use the repository GPU memory
  helper before logical-device initialization, XLA, TF32, and the q=20 target.
- Artifact adequacy: every warm-up and retained chunk is archived separately;
  retained diagnostics never include warm-up draws.
- Misleading pass pre-mortem: R-hat/ESS can pass without posterior correctness.
  The result will state that no posterior oracle exists and downstream
  predictive validation remains separate.
- Misleading failure pre-mortem: a bad fixed kernel can fail even if the
  transport idea is viable. A cap failure is classified as kernel/transport
  repair evidence, not research-direction rejection.

Audit decision: `PASS_FOR_IMPLEMENTATION_AND_SMOKE`; long execution remains
conditional on the proposed campaign cap.

## Energy-Error And Coordinate Diagnostic Contract

The installed TensorFlow Probability `MetropolisHastings` implementation forms
the HMC log acceptance ratio as the proposed target log probability minus the
accepted target log probability plus the inner HMC kinetic correction.  The
inner HMC correction is the initial kinetic energy minus the proposed kinetic
energy.  Therefore, for this identity-mass kernel, the finite quantity recorded
by the controller satisfies

`Delta H = H_proposed - H_initial = -log_accept_ratio`.

The controller records this identity as `delta_h_equals_negative_log_accept_ratio`
and applies the existing reviewed absolute energy-error hard limit `1000.0`.
This is an energy-error check, not native divergence telemetry.  TFP 0.25 does
not expose a native divergence flag on this route, so the manifest retains
`native_divergence_status = not_exposed_by_kernel`; it never converts that
absence into a zero-divergence claim.

Admission is evaluated in two coordinate systems.  The first is the latent
NeuTra/HMC coordinate `z`; the second is the mapped model-parameter coordinate
`theta = T(z)`.  Both coordinate systems must pass the same finite rank/folded
R-hat and bulk/tail ESS screens.  This does not establish posterior correctness
or chart equivalence, but it prevents a chart from passing solely because its
latent coordinates look mixed while the mapped parameters remain separated.

## Active NeuTra-HMC Route Ledger

| Route | Classification | Status/nonclaim |
| --- | --- | --- |
| `docs/benchmarks/run_ssl_lstm_q20_neutra_sequential_real_2026_07_22.py` | `bayesfilter_neutra_sequential_hmc_v1` | Active claim-bearing route; uses the shared sequential controller |
| `docs/benchmarks/run_ssl_lstm_neutra_complexity_retained_hmc_2026_07_19.py` | historical fixed-burn-in/checkpoint route | Historical only; ineligible for new convergence or posterior claims |
| `docs/benchmarks/run_ssl_lstm_neutra_complexity_predictive_validation_2026_07_19.py` | downstream historical consumer with an embedded fixed-archive HMC extension | Historical only; cannot acquire new HMC evidence |
| `docs/benchmarks/run_ssl_lstm_neutra_phase7_retained_admission_2026_07_17.py` | historical fixed-burn-in/checkpoint route | Historical only; ineligible for new convergence or posterior claims |

| `docs/benchmarks/run_ssl_lstm_neutra_complexity_hmc_tuning_2026_07_19.py` | historical fixed-kernel tuning route | Tuning evidence only; not a claim-bearing sequential sampler |
| `docs/benchmarks/run_ssl_lstm_neutra_hmc_complexity_canary_2026_07_19.py` | historical canary route | Canary mechanics only; not convergence or posterior evidence |
| `docs/benchmarks/run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py` | historical transformed-HMC tuning route | Tuning evidence only; not a claim-bearing sequential sampler |
| `docs/benchmarks/benchmark_ssl_lstm_complexity_hmc_budget_rate_2026_07_19.py` | historical timing-only route | Benchmark only; no sampler claim |
| `docs/benchmarks/benchmark_ssl_lstm_hmc_chain_topology_2026_07_19.py` | historical chain-topology benchmark | Benchmark only; no convergence or posterior claim |

The new launcher has a source scan test requiring the shared controller.  The
historical routes remain preserved for reproducibility, but they are not
silently promoted by this campaign.

## Preparation Result, 2026-07-22

Implemented:

- TensorFlow/TFP-only shared sequential controller with separate warm-up and
  retained shards, cumulative stopping, target-status checks, movement checks,
  exact energy-error telemetry, and prospective budget refusal.
- q=20 launcher binding both exact `(32,32)` frozen payloads and the fixed
  canary kernel hypothesis.
- R-hat and ESS checks in both NeuTra coordinates and mapped model parameters.
- Serious run-manifest fields, GPU memory-growth fail-closed configuration,
  versioned output-root enforcement, and route-ledger discovery coverage.

Focused verification:

- `14 passed` for the shared controller and q=20 launcher contract suite.
- Scalar target-status value/score parity: passed.
- Batched target-status value/score parity: passed.
- CPU-hidden contract smoke: passed.
- Python compilation, `git diff --check`, and active-route NumPy source scan:
  passed.

The GPU mechanics smoke was not launched because both GPUs were already under
heavy compute utilization by existing processes (`94%` and `98%` at the
preflight observation).  No process was stopped and no competing workload was
started.  The long campaign remains unlaunched.  Launch conditions are:

1. an available GPU lane; and
2. explicit approval of the proposed cumulative `77,700 s` (`21.5833`
   GPU-hour) cap.
