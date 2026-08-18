# NeuTra Banana Predictive-Equivalence Diagnostic Result (2026-08-16)

## Outcome

The frozen seed-15, `L=10` banana HMC candidate completed the target-specific
output-law diagnostic in `10.82 s` on GPU 0. The runner used 4 chains, 1,024
retained draws per chain, float64/XLA, TF32 disabled, and the repository GPU
memory-growth policy. The candidate was compared with independent exact
analytic banana draws at two fixed archive offsets. SHA-256 checks passed for
all artifacts.

The finite-sample screen did **not** pass: the candidate's 99% moving-block
bootstrap upper MMD interval exceeded the exact-vs-exact calibration envelope
at both offsets for all tested block lengths. This is evidence that the
candidate is detectably different under this diagnostic, not a formal proof
that the two probability laws are unequal. Conversely, it does not establish
posterior failure in every metric or invalidate the HMC kernel by itself.

## Evidence contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-banana-predictive-equivalence-plan-2026-08-16.md` |
| Artifact root | `docs/plans/artifacts/neutra-banana-predictive-equivalence-2026-08-16-r2/` |
| Candidate | Confirmation archive, seed `15`, 6,000 training updates, frozen identity z mass, `L=10`, step `0.7709722545680272` |
| Sample unit | Raw 16-dimensional model-coordinate banana draw; no artificial time horizon |
| Statistic | Biased multi-bandwidth RBF MMD, bandwidths `(2,4,8)`, per-bandwidth and equal-mixture summaries |
| Dependence treatment | Four fixed HMC chains, within-chain moving block lengths `32,64,128`; 256 bootstrap replicates |
| Calibration | 32 independent exact-vs-exact banks per offset, identical layout and bootstrap contract |
| Offsets | `0` and `1024` from the 5,000-draw retained archive |
| Runtime | `10.8 s` (same fixed campaign, fresh reproducibility-complete rerun) |
| Provenance | GPU 0, `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth, float64, XLA, TF32 disabled |

## Primary results

The calibration envelope is the empirical 99th percentile across the 32
exact-vs-exact upper intervals. Values below are MMD squared, not a p-value.

| Offset | Block | Candidate point | Candidate upper 99% | Exact-control envelope | Ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.00040056 | 0.00100853 | 0.00090562 | 1.11 |
| 0 | 64 | 0.00040056 | 0.00101816 | 0.00090063 | 1.13 |
| 0 | 128 | 0.00040056 | 0.00104374 | 0.00089455 | 1.17 |
| 1024 | 32 | 0.00041271 | 0.00105177 | 0.00086444 | 1.22 |
| 1024 | 64 | 0.00041271 | 0.00111092 | 0.00085564 | 1.30 |
| 1024 | 128 | 0.00041271 | 0.00105642 | 0.00084730 | 1.25 |

The MMD point statistic was independently checked against
`bayesfilter.inference.predictive_equivalence.fixed_rbf_mmd` on an independent
fixture; the difference was `1.1e-16`.

## Explanatory results

Coordinate summaries were not grossly displaced. For offset 0, the first four
candidate coordinate means were `(-0.00599,-0.01449,-0.03020,-0.03027)` and
variances `(0.9945,1.2666,1.0117,1.0056)`. For offset 1024 they were
`(0.01136,0.00668,-0.02222,-0.00004)` and
`(1.0410,1.2883,1.0431,1.0301)`. The banana's second coordinate has analytic
variance `1 + 2*0.35^2 = 1.245`; the observed values are close at this sample
size. These are explanatory summaries only and do not override the MMD screen.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Candidate output-law screen | Candidate upper MMD below exact-control envelope at both offsets and all blocks | **Failed screen**, no hard infrastructure veto | 32 calibration banks and 256 bootstrap replicates are descriptive; finite kernel grid may miss or emphasize features | Preserve candidate as HMC-viable but run a discriminating diagnostic separating dependence/finite-sample effects from a real law discrepancy before changing training | Formal equality or universal HMC failure |
| Harness validity | Finite samples, exact-vs-exact calibration, independent implementation tie-out, hashes | **Passed** | No formal calibration theorem for this custom envelope | Keep harness for the next banana diagnostic | Scientific correctness |
| HMC confirmation interaction | Prior L=10 exact-law and convergence gates passed | **Passed previously** | Exact-law screens are limited moments; native divergence telemetry unavailable | Treat predictive failure as a repair trigger, not as proof that the L=10 kernel is invalid | Posterior correctness or SSL-LSTM transfer |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | No infrastructure or numerical veto; predictive screen itself failed |
| Statistically supported ranking | None; there was no candidate comparison or uncertainty-supported ranking |
| Descriptive-only differences | MMD point/upper intervals, coordinate moments, block sensitivity, runtime |
| Default-readiness | Not supported |
| Next evidence needed | A predeclared diagnostic using longer retained chains and an independent exact-vs-exact replication to determine whether the 10-30% envelope excess is sampling variability, block dependence, or a genuine learned-transport law error |

## Interpretation and red-team note

The strongest alternative explanation is not a training failure: the candidate
has passed the earlier exact-law moment screens and its discrepancy is modest,
while the custom finite-sample envelope is itself estimated from only 32 banks.
The second alternative is that the retained-chain dependence is not adequately
represented by block lengths 32-128. A third is a genuine small distributional
error in the learned transport/HMC dynamics that the coordinate screens did not
see. The current result cannot distinguish these explanations.

The result therefore triggers a targeted follow-up, not a new default or a
training rewrite. The next run should increase the retained archive used for
the diagnostic (without retuning the candidate), repeat exact-vs-exact controls
with a larger calibration bank, and add an independent random archive window.
If the candidate remains above the control envelope, inspect tail and
nonlinear-feature discrepancies before changing the learned transport.

## Nonclaims

- No formal equality p-value was computed.
- Passing the earlier HMC screens plus this failed finite screen does not prove or disprove posterior correctness in general.
- No universal `L=10` or NeuTra default is promoted.
- No SSL-LSTM transfer, multimodal claim, superiority claim, or production readiness is supported.
