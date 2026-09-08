# Phase 4B: Full-Mixture IWSG Resampling Reference

Date: 2026-09-08  
Status: `IMPLEMENTATION AUTHORIZED; CORRECTNESS GATES BEFORE CAMPAIGN`  
Scope: diagnostic research route only; no canonical, HMC, DSGE, or default promotion

## Research intent

The scientific question is whether the full marginalized kernel-mixture
resampling mechanism of Younis and Sudderth can reduce error in estimating the
exact model score when it is combined with the complete LEDH--OT--GenUT
Contract-E dual-cap filter. The canonical comparator is the analytical total
derivative of the unchanged finite Contract-E program. The candidate is not a
second implementation of that derivative: positive-bandwidth resampling
changes the finite particle program.

The targets are therefore separate:

- `ATOM-FINITE`: the unchanged canonical finite value and its exact analytical
  total derivative for a fixed stream;
- `RESKDM-FINITE`: Contract-E and both GenUT correction caps followed by
  fixed-anchor, full-mixture IWSG resampling, with its own exact analytical
  total derivative; and
- the exact Kalman value and score on the linear-Gaussian fixture, which judge
  model-score error but do not make either finite score exact at finite N.

The candidate succeeds only if its separately labelled score has lower paired
error against the exact model score. It must never be reported as computing
the unchanged `ATOM-FINITE` derivative.

## Source and extension boundary

The local primary sources are:

- `docs/papers/differentiable/Differentiable and stable long-range tracking of multiple posterior modes Younis(23).pdf`, Section 2.2, Eq. (4), Section 4, Eqs. (14)--(15), and Appendix B.1; and
- `docs/papers/differentiable/Learning to be smooth An end-to-end differentiable particle smoother Younis(24).pdf`, Sections 2.1 and 2.3, Eqs. (4), (6)--(7), and the computational-requirements discussion.

The source-supported operation samples from the complete continuous mixture,
holds the sample location fixed in the IWSG derivative, evaluates the
marginalized mixture density rather than the sampled component density, and
incurs all-pairs work during gradient computation. Stratified component
selection is used in the 2024 paper.

The following are BayesFilter extensions, not claims from those papers:

- retaining Contract-E and the GenUT dual-cap reset before KDM resampling;
- carrying particle-local UKF covariance marks through the resampler;
- using the resulting finite program as a candidate estimator of a generative
  state-space-model score; and
- using a common bandwidth proportional to process covariance in the first
  linear-Gaussian experiment.

The exact source MDPF would replace discrete resampling. Replacing Contract-E
would remove part of the algorithm named in the research question. Phase 4B
therefore tests the explicitly combined program, not source MDPF fidelity.

## Mathematical program

After each canonical Contract-E and dual-cap reset, let its outputs be
locations `c_i`, covariance marks `P_i`, and uniform component weights
`alpha_i=1/N`. For common positive-definite bandwidth `B`, define

~~~text
m_theta(z_j) = sum_i alpha_i k_B(z_j - c_i)
q_0(z_j)     = m_theta0(z_j)
r_j(theta)   = m_theta(z_j) / q_0(z_j)
w_j(theta)   = r_j(theta) / sum_l r_l(theta).
~~~

At the anchor, stratified component uniforms and Gaussian noises generate
`z_j`; the component label is recorded but marginalized in every density and
gradient evaluation. During replay, `z_j` and `q_0(z_j)` are fixed. The full
component responsibility is

~~~text
gamma_ji = alpha_i k_B(z_j-c_i) / m_theta(z_j).
~~~

Every cloud, component-weight, and bandwidth tangent enters `d log m`. The
normalized outgoing log-weight tangent is

~~~text
d log w_j = d log m_theta(z_j) - sum_l w_l d log m_theta(z_l).
~~~

Those log weights and their tangents are incoming terms in the next PF--PF
normalizer. This is essential: the current canonical executor hard-codes the
uniform incoming term because Contract-E normally resets it. Phase 4B must
generalize that internal recurrence while proving bit-level or declared-
tolerance parity for every canonical caller.

Younis particles do not carry UKF covariance marks. Phase 4B defines the mark
as the conditional mean of the augmented component mark given the fixed sample:

~~~text
Pbar_j   = sum_i gamma_ji P_i
d Pbar_j = sum_i d gamma_ji P_i + sum_i gamma_ji d P_i.
~~~

This Rao--Blackwellized mark makes the downstream input a deterministic
function of the marginalized mixture. It adds no between-component scatter,
because `P_i` is an algorithmic mark rather than the covariance of the KDM
location. This choice is an extension and must be ablated against fixed-label
mark carry before any scientific promotion.

The analytical derivative to be implemented is the total derivative of the
fixed-anchor replay scalar `L_resKDM^N(theta; xi, z0, q0)`. It is not the total
derivative of the random sampling map, is not `d L_0^N/d theta`, and is not the
joint likelihood-ratio identity for an arbitrary nonlinear function of all N
draws. The Younis papers do not prove it unbiased for the generative model
score; the Kalman experiment tests that empirical question.

## Research question guardian

| Item | Predeclared role |
|---|---|
| Main question | Does `RESKDM-FINITE` reduce paired exact-model score error relative to `ATOM-FINITE`? |
| Candidate mechanism | Full-marginal IWSG resampling may propagate information through a smooth density rather than only through deterministic reset locations. |
| Expected failure mode | Positive bandwidth adds bias, normalized IWSG weights become unstable, or invented covariance-mark transport harms the next UKF step. |
| Promotion criterion | On untouched validation paths, the paired 95% interval for mean squared-score-error difference (`candidate - canonical`) is below zero and its upper endpoint is at most `-0.10 * canonical MSE`. |
| Promotion veto | Any heuristic comparator has lower conditional MSE, or candidate bias magnitude is larger without a compensating, statistically supported MSE reduction. |
| Continuation veto | Wrong target label, analytical/finite-difference failure, anchor replay failure, omitted all-pairs term, invalid covariance/support, non-finite output, broken canonical parity, or exhausted campaign budget. |
| Repair trigger | A finite and reproducible candidate fails the promotion criterion without firing a continuation veto. Diagnose bandwidth, weight dispersion, mark carry, and horizon dependence on fresh calibration data. |
| Explanatory only | ESS, responsibility entropy, cap activity, runtime, peak memory, and point estimates without uncertainty. |
| Must not be concluded | Passing does not establish equality to `ATOM-FINITE`, unbiasedness, DSGE support, HMC validity, production readiness, or a new default. |

## Heuristic adversary set

The first linear-Gaussian campaign constructs and evaluates these comparators
on exactly the same observation paths:

1. Exact Kalman score: certifying oracle, not a heuristic.
2. Canonical Contract-E LEDH analytical score: the required baseline.
3. Bootstrap particle-filter likelihood score with ordinary fixed-stream
   resampling: the simplest particle-filter score comparator.
4. Phase 4A observation-convolution score at its calibration-selected
   bandwidth: a previously implemented KDM candidate.
5. Canonical score plus a calibration-frozen no-feedback KDM control variate,
   only if a conditional-zero-mean construction and coefficient are derived;
   otherwise this row is explicitly `not implemented`, not approximated.

Results are reported separately for each `(N,T)` cell and for low versus high
absolute Kalman-score paths. An unconditional average cannot hide a failure in
one of those salient groups. Losing to any implemented cheap comparator in a
salient group blocks promotion.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| Contract-E plus both GenUT caps | Repository owner policy | A reduced lane silently omits the requested algorithm | Endpoint call-chain and cap-activity test | Frozen baseline |
| Post-reset KDM insertion | Research question plus source/reset incompatibility | Defines a new scalar and is mislabelled canonical | Route identity and value-difference test | Explicit extension |
| Full marginalized mixture | Younis 2023 Eqs. (14)--(15) | Selected-component shortcut biases/misses gradients | Permutation and all-component perturbation tests | Required source operation |
| Fixed sample/proposal during replay | Younis IWSG | Re-anchoring changes the finite scalar | Byte/equality replay and denominator-stop tests | Required source operation |
| Stratified labels | Younis 2024 Eq. (4) and Section 2.1 | Incorrect strata or endpoint uniforms | Deterministic boundary fixtures | Required source operation |
| Common SPD `B=rho^2 Q` | Phase 4A warm start, not a source default | Wrong scale or anisotropy rejects the idea for tuning reasons | Fresh calibration grid and Cholesky margin | Hypothesis |
| Responsibility-mean covariance mark | Local augmented-mark derivation | Nonlinear UKF use makes it inferior to selected-label carry | Mark finite differences and prespecified ablation | Extension/hypothesis |
| Float64, no TF32 correctness authority | Current endpoint evidence | Does not establish production throughput | Float32/no-TF32 parity after correctness | Reference authority |
| XLA enabled | Repository policy | Unsupported operation or excessive compile cost | Tiny GPU XLA smoke with timing/memory | Required candidate default |
| All `N x N` terms retained | Younis source complexity | Memory cost grows quadratically | Pair counter and peak allocator bytes | Required reference |

No bandwidth, mark policy, or Phase 4A setting transfers as a promoted default.
Each is a calibration hypothesis.

## Implementation sequence

### 1. Complete local resampling primitive

Extend the existing TensorFlow mixture core only enough to expose the already
computed per-component log-density tangents. Implement a fixed-shape primitive
that returns:

- full marginalized log density and tangent;
- every `N x N` responsibility and responsibility tangent;
- fixed-proposal log ratios;
- normalized IWSG log weights and tangents;
- Rao--Blackwellized covariance marks and complete tangents;
- finite/SPD/normalization validity; and
- the exact pair count.

No selected-component density, sparse neighbor search, low-rank feature map,
stopped cloud tangent, stopped covariance tangent, NumPy runtime computation,
pfor, or sample-wise Python loop is allowed.

### 2. Generalize the shared recurrence without changing canonical behavior

Add a repository-internal post-reset transform hook to
`_value_and_analytical_score_impl`. Carry incoming log weights and their
tangents as explicit recurrent state. The canonical wrapper supplies no hook
and continues to set the same uniform constant after Contract-E. The Phase 4B
hook supplies KDM states, covariance marks, log weights, and tangents.

Executable parity must compare the canonical endpoint before and after this
change for value, score, reset states, covariance carry, and trace. A wiring
test must prove the next PF--PF logits change when only the outgoing KDM
log-weight tangent is perturbed. Merely finding the hook in source is not
evidence.

### 3. Implement anchor and replay endpoints

The anchor endpoint consumes fixed stratified uniforms and Gaussian noises,
runs the full sequential combined program, stops every generated KDM sample
location, and records samples, component indices, proposal log densities, and
all resampling diagnostics. The replay endpoint consumes that fixed bank and
must not regenerate or re-anchor it. Both endpoints call the shared canonical
executor with `reset_policy=contract_e`, diagonal correction steps, pairwise
correction steps, and a positive coordinate cap.

The public diagnostic factory has an explicit stable TensorFlow signature and
defaults to `jit_compile=True`. It is not added to the canonical registry.

### 4. Correctness tests

Run focused float64 tests with GPU hidden unless an XLA GPU is specifically
required:

- all dependency blocks jointly and separately against centered finite
  differences;
- responsibility rows and normalized outgoing weights sum to one, with zero
  tangent sums;
- component permutation equivariance;
- perturbing any nonselected component changes a sample's mixture tangent;
- anchor ratios equal one and anchor outgoing weights are uniform;
- replay at the anchor exactly reproduces the anchor finite program;
- `T=2` finite differences of the complete replay scalar agree with the
  analytical score;
- state, weight, and covariance paths each affect the second normalizer under
  isolated ablations;
- canonical endpoint parity with no hook;
- invalid SPD, proposal, normalization, shape, or non-finite inputs fail
  closed; and
- source scans reject NumPy runtime imports, pfor, `tf.vectorized_map`,
  `tf.map_fn`, selected-component density, and reduced reset settings.

Autodiff may be used only as a diagnostic oracle; the implemented score path
remains analytical.

### 5. GPU/XLA implementation smoke

After CPU reference gates pass, run one escalated managed-session GPU smoke at
`D=2, N=8, T=2`, float64/no-TF32 and one float32/no-TF32 parity arm. The
two-dimensional fixture is required so the pairwise GenUT correction is an
executed operation rather than a vacuous configuration flag. Enable and
verify memory growth before device initialization. Record compile-plus-first
call, warm-call time, allocator peak, device, XLA, dtype, TF32 state, route ID,
source hashes, and anchor/replay residuals in a new versioned directory.

This smoke answers compilation and call-chain questions only. It is not used
to select bandwidth or interpret score quality.

### 6. Bounded calibration, power, and untouched validation

Only after all implementation gates pass, create a campaign amendment with the
exact command and source hashes. The initial scopes are `(N,T)=(32,5)` and
`(64,20)` in the scalar stationary LGSSM. Calibration and validation path seeds
are disjoint; particle streams are paired across every comparator.

Calibration evaluates `rho in {0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6}`
on at least 40 paths per scope. The selected bandwidth and covariance-mark
policy are frozen before validation. A separate 30-path variance pilot on
unused seeds estimates the paired standard deviation. For practical effect
`delta = 0.10 * canonical calibration MSE`, the validation count is

~~~text
n = ceil((1.96 + 0.84)^2 * sigma_pair^2 / delta^2),
~~~

rounded up to the next multiple of 20, with minimum 100 and maximum 500 per
scope. If the cap is insufficient for 80% nominal power, the scope is reported
underpowered and cannot promote. Validation uses a paired bootstrap interval
and reports mean error, variance, MSE, MCSE, and intervals without ranking from
point estimates alone.

Maximum campaign budget: two scopes, 40 calibration paths per bandwidth, 30
power-pilot paths, 500 validation paths, two implementation retries, 45 GPU
minutes total, and a unique output directory per attempt. Stop on a
continuation veto or budget exhaustion. A viable but nonpromoting candidate
triggers diagnosis, not rejection of the whole research direction.

## Skeptical pre-execution audit

The plan was checked against the required failure modes before implementation:

- Wrong baseline: avoided by calling the registered single-cloud canonical
  endpoint with explicit Contract-E and both caps, plus the independent Kalman
  oracle. The reduced fused batch lane is excluded.
- Proxy promotion: ESS, entropy, cap activity, runtime, and smoke parity are
  explanatory or engineering checks, never score-quality promotion criteria.
- Missing stop conditions: target, derivative, replay, support, finiteness,
  canonical-parity, source-operation, and budget vetoes are explicit.
- Unfair comparison: all methods share observation paths and particle streams;
  positive bandwidth gets its own calibration and is not compared using a
  setting selected on validation data.
- Hidden assumption: bandwidth geometry, covariance-mark transport, fixed
  anchors, dtype, TF32, and all-pairs complexity are recorded as hypotheses or
  requirements above.
- Stale context: the authoritative Phase 4A result is Attempt 05 on current
  source hashes; the older Attempt 03 wording “powered” is retired.
- Environment mismatch: CPU reference tests explicitly hide CUDA; all GPU/XLA
  checks use escalated device access and memory growth.
- Artifact mismatch: the implementation smoke cannot answer the scientific
  question; the later paired campaign is the only score-quality evidence.
- Call-chain omission: incoming log weights and their tangents are named
  recurrent state and tested at the next normalizer, not inferred from a local
  mixture helper.
- Nonlinear IWSG overclaim: the result is the total derivative of the declared
  fixed-anchor finite replay program only. No joint-expectation unbiasedness or
  equality to `ATOM-FINITE` is claimed.

The audit passes for Steps 1--5. Step 6 is conditional on a separate campaign
amendment after the implementation artifacts exist; no long score-quality run
is launched from this plan alone.

## Required result record

The implementation result must report, in separate ledgers:

| Ledger | Required conclusion |
|---|---|
| Engineering correctness | Local primitive, anchor/replay, recurrent wiring, canonical parity, and XLA status |
| Numerical validity | Finite-difference errors, SPD/normalization margins, dtype/TF32 differences, and non-finite vetoes |
| Scientific interpretation | Not evaluated until the powered paired campaign; no score-improvement language from tests or smoke |

It must also list every discrepancy between this plan, the LaTeX equations,
and the executed call chain. An empty discrepancy list requires executable
evidence; prose inspection alone is recorded as `not checked`.
