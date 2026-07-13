# Phase 0 Result: Source And Identifiability Inventory

Date: 2026-07-08

## Decision

`PASS_PHASE0_LOWER_TRIANGULAR_FIRST`

The first serious multidimensional LGSSM NeuTra-HMC target should use a
lower-triangular stationary transition matrix with fixed observation matrix
`H = I`, diagonal positive `Q/R`, fixed coordinate order, fixed zero offsets
for the first rung, and stationary initial covariance `P_inf` from the
discrete Lyapunov equation.

Block-lower-triangular dynamics remain a later extension. They should not be
the first implementation target because they introduce extra within-block
degrees of freedom before the simpler triangular benchmark has passed
stationarity, score, synthetic recovery, and HMC gates.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which source-anchored constrained multidimensional LGSSM target should BayesFilter implement first? |
| Baseline/comparator | MARSS constrained state-space form, VAR stationarity parameterization literature, statsmodels stationary VAR transform, and local stationary/Lyapunov utilities. |
| Primary criterion | A first target recommendation with supported and unsupported claims separated. |
| Veto diagnostics | Unsupported identifiability claim, missing stationary initial law, hidden dense-`A` assumption, source mismatch, or treating stationarity as full identifiability. |
| Result | Pass for `lower_triangular_first`; no implementation or runtime gate is passed. |

## Source-Support Ledger

| Source | Inspected Support | Supports | Does Not Support |
| --- | --- | --- | --- |
| Holmes (2013), MARSS EM derivation, arXiv:1302.3919 | Abstract/source summary describes constrained MARSS equations `x_t = B x_{t-1} + u + w_t`, `y_t = Z x_t + a + v_t`, and matrix constraints on `B,u,Q,Z,a,R`. | It supports using a constrained linear Gaussian state-space form and treating fixed/zero/shared matrix elements as first-class model restrictions. | It does not prove the proposed BayesFilter triangular model is globally identifiable, HMC-ready, or posterior-convergent. |
| Heaps (2020), stationary VAR prior, arXiv:2004.09455 | Technical text states VAR stationarity is tied to roots of the characteristic equation and describes partial-autocorrelation-matrix reparameterizations for stationary VARs; it also discusses HMC via Stan. | It supports the need to enforce stationarity by construction and shows that unconstrained dense stationary VAR parameterizations are nontrivial. | It does not directly define the BayesFilter LGSSM parameter contract or prove identifiability with latent states. |
| statsmodels `constrain_stationary_multivariate` docs | The API maps unconstrained optimizer parameters to stationary VAR coefficient matrices using Ansley-Kohn lineage. | It supports treating stationary multivariate transforms as a standard implementation pattern and a possible future comparator. | It is not the first BayesFilter implementation target because it is a dense VAR transform, not a deliberately coordinate-anchored triangular LGSSM benchmark. |
| Auger-Methe et al. (2015), linear Gaussian SSM estimability problems | The paper reports that even simple linear Gaussian SSMs can have parameter/state-estimation problems, especially when measurement error dominates process variation. | It supports keeping synthetic recoverability and HMC diagnostics separate from formal stationarity and model specification. | It does not imply the triangular benchmark will fail; it is a caution against overclaiming. |
| Local `bayesfilter/linear/stationary_lgssm_derivatives_tf.py` | Read locally. It implements continuous-time-to-discrete stationary LGSSM construction, continuous Lyapunov solve, Frechet derivative of matrix exponential, diffusion-factor derivatives, and first-derivative coverage metadata without runtime generic Jacobian construction. | It supports reusing audited stationary covariance/derivative infrastructure and reinforces that stationary initial covariance must be part of the model object. | It is continuous-time primitive based; Phase 1/3 must decide whether to reuse it directly or add a discrete triangular Lyapunov helper. |
| Local `tests/test_stationary_lgssm_derivatives_tf.py` | Read locally. Tests check Frechet derivatives, Lyapunov residuals, finite-difference agreement, derivative coverage, and absence of runtime `GradientTape`/generic `jacobian` in the stationary module. | It supports a no-runtime-`GradientTape` policy for admitted derivative infrastructure and gives a regression base for Phase 3. | It does not test the proposed triangular parameterization, synthetic recovery, XLA compilation, NeuTra, or HMC. |

## Design Classification

### Stationarity

For a discrete-time lower-triangular transition matrix `A`, the eigenvalues are
the diagonal entries. If Phase 1 maps each diagonal parameter into
`(-rho_max, rho_max)` with `rho_max < 1`, then stationarity of `A` is enforced
by construction.

The initial law must be stationary:

```text
x_0 ~ N(0, P_inf)
P_inf = A P_inf A' + Q
```

This is required. A diffuse, arbitrary, or omitted initial covariance is a
Phase 1 veto.

### Coordinate Anchoring

The first target fixes:

```text
y_t = x_t + eps_t
eps_t ~ N(0, R)
R = diag(r_1^2, ..., r_d^2)
```

With `H = I`, fixed observed coordinate order, diagonal `Q/R`, and fixed zero
offsets, the obvious latent similarity transformations, permutations, and sign
flips are removed from the benchmark definition.

This is a design restriction, not a theorem that every parameter is globally
identifiable from finite data.

### Synthetic Recoverability

Synthetic recoverability remains empirical and must be tested later. Phase 2
must choose truth values with:

- distinct diagonal dynamics away from zero and away from the stationarity
  boundary;
- nonzero but bounded lower-triangular couplings;
- process and observation variances separated enough to avoid the known
  process-noise/measurement-noise ridge as the first rung;
- enough time points for posterior concentration diagnostics;
- fixed seeds and hashes.

### HMC Evidence

No HMC evidence exists from Phase 0. Later phases must still demonstrate:

- exact Kalman value/score correctness;
- `jit_compile=True` compile success;
- GPU-only NeuTra training;
- frozen transport integrity;
- CPU-hidden multicore HMC sampling;
- per-parameter R-hat/ESS and truth/reference diagnostics.

## Recommendation

Proceed to Phase 1 with `lower_triangular_first`.

The Phase 1 contract should specify:

- `state_dim = observation_dim`, initially `3` or `4`;
- `H = I`;
- zero offsets and zero stationary mean for the first rung;
- `A` lower triangular with diagonal entries mapped to
  `(-rho_max, rho_max)`, with a recommended first value `rho_max <= 0.85`;
- bounded lower entries, with explicit parameter names and finite transforms;
- `Q` and `R` diagonal positive via log-scale parameters;
- `P_inf` computed by a discrete Lyapunov solve;
- fixed parameter order and stable target signature fields;
- nonclaims preserving that this is an audited synthetic benchmark, not a
  broad LGSSM identifiability proof.

## Local Checks

- Read `bayesfilter/linear/stationary_lgssm_derivatives_tf.py`.
- Read `tests/test_stationary_lgssm_derivatives_tf.py`.
- Searched local BayesFilter code/tests/docs for stationary LGSSM, Lyapunov,
  initial covariance, score, XLA, `GradientTape`, and HMC references.
- Confirmed Phase 0 made no algorithmic code edits and ran no training/HMC.
- Launch review: Claude review gate was denied by sandbox external-disclosure
  policy; same-foreground Codex substitute review returned `VERDICT: AGREE`.

## Decision Table

| Decision | Primary Criterion Status | Veto Diagnostic Status | Main Uncertainty | Next Justified Action | Not Concluded |
| --- | --- | --- | --- | --- | --- |
| Use `lower_triangular_first` | Met | No Phase 0 veto triggered | Practical parameter recoverability is not yet tested | Write Phase 1 model contract | No implementation correctness, HMC readiness, global identifiability, product/default readiness, or scientific validity |

## Plain-Language Gate

Claimed target: choose a source-aware first multidimensional LGSSM target.

Computed/inspected quantity: source/context inventory plus local stationary
code/test inventory.

Verdict: `correct` for beginning Phase 1 contract work on a lower-triangular
benchmark; `not checked` for implementation, XLA, NeuTra, HMC, and posterior
recovery.

What remains unproved: global identifiability, finite-sample recoverability,
score correctness, compile readiness, NeuTra usefulness, and HMC convergence.

## Next Phase Handoff

Phase 1 may begin. It must convert this recommendation into a precise model
contract and must stop if `P_inf`, coordinate order, parameter transforms,
truth template, or nonclaims remain ambiguous.
