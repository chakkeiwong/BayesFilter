# Shared component-cost result, 2026-10-12

All five first-observation staged computations pass comparison with the actual
fused factory: value, all parameter scores, particles and particle tangents are
finite and within the1% screen; the observed differences are much smaller.
N4032/K2016,B2,GPU/XLA,TF32off and memory growth were used. LGSSM,KSC and
predator-prey use uniform FP32; SIR/range use optional FP64 recursion with
centered FP32 pair kernels. Each stage traced once. Means below use three
warmed evaluations with rotated order.

| Model | Proposal/observation seconds | Densities/weights seconds | Transport/reset seconds | Fused seconds | Staged/fused |
|---|---|---|---|---|---|
| lgssm | 0.0835 (11.3%) | 0.1701 (23.0%) | 0.4858 (65.7%) | 0.7189 | 1.028 |
| ksc | 0.0262 (7.1%) | 0.0051 (1.4%) | 0.3367 (91.5%) | 0.3228 | 1.140 |
| predator_prey | 0.0423 (9.7%) | 0.0669 (15.3%) | 0.3274 (75.0%) | 0.4213 | 1.036 |
| sir_d18 | 0.7616 (41.9%) | 0.5770 (31.8%) | 0.4771 (26.3%) | 1.6255 | 1.117 |
| range_bearing | 0.0927 (17.5%) | 0.1601 (30.1%) | 0.2786 (52.4%) | 0.5263 | 1.010 |

Fractions describe the staged first observation only. Proposal cost includes
initialization, and stages materialize intermediates that the full graph may
fuse away. The staged/fused discrepancy, especially14% for KSC, limits exact
attribution. These are diagnostic timings under possibly competing GPU load,
not statistically supported speed rankings or whole-horizon fractions.

Transport dominates the measured LGSSM/KSC/predator steps. SIR has substantial
proposal and density cost as well. Inspection of build_maps reveals repeated
covariance construction and checked eigendecomposition after broadcasting
shared component matrices to every particle. The next bounded repair computes
and validates each distinct covariance before broadcasting its outputs.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept instrumentation for localization | All five staged/fused parity screens pass | No invalid or nonfinite rows; one trace per graph | Staging and first-observation coverage | Test compact covariance construction, then transport reduction reuse if budget permits | Whole-horizon attribution or posterior validity |
| Continue HMC mechanics on frozen selections | Precision candidates available | Endpoint and sampler health still to be checked | Four-chain nonlinear behavior and cost | SIR/range exploratory mechanics | Convergence from two transitions |

| Inference status | Finding |
|---|---|
| Hard veto screen | No component parity failure |
| Statistically supported ranking | None |
| Descriptive-only differences | Component means/fractions and fusion ratios |
| Default-readiness | No default change from instrumentation |
| Next evidence needed | Full-scope paired correctness/cost for any repair; eventual tuned cumulative HMC |

Alternative: a covariance calculation visible in source may already be removed
by XLA in the fused graph. A compact rewrite with no measured full-scope gain
would weaken this proposed cost explanation. Therefore preserve actual source
comparators and time the full step as well as the component.

Per-model command, environment, source snapshots, GPU/memory policy and wall
time are in sibling surrogate-hmc-component-{lgssm,ksc,predator,sir,range}-20261012-01
directories under each model/result.json; exact inputs, controls, hashes and
measurements are in component-cost.json. All five attempts used197.4371s
combined, charged to the cumulative ledger. Before mechanics9873.0972s remained.
