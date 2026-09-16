# HMC pipeline validation: established methods and the BayesFilter gaps

Date: 2026-09-15. Status: source review and proposed test specification; the
additional suite below has not been implemented or executed. This note extends
the [overall repair result](bayesfilter-hmc-overall-repair-result-2026-09-15.md).
It does not change numerical defaults or tuning admission requirements.

The subsequent [infrastructure design](bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md)
integrates these methods into shared target/reference definitions, distinct
experimental designs, public procedure adapters, independent assessment
engines, and coverage/power reporting. It supersedes this note's implementation
ordering; the source survey and coverage findings below remain its basis.

The existing tests establish specific engineering repairs and some numerical
properties. They do not establish systematic coverage of difficult posterior
geometries, statistical calibration of the complete pipeline, or the power of
the tests to detect deliberately incorrect implementations. The user's concern
is correct. Increasing the count of similar Gaussian or mocked tests would not
close these gaps.

There are established complementary approaches in the literature and in Stan,
PyMC, and R. There is no single test among the inspected sources that certifies
an arbitrary HMC tuning pipeline. Dynare's convergence and run-length diagnostics
answer a related but different question: whether a particular simulation appears
adequate for estimation.

## What the existing evidence actually covers

The repair result records 277 passing tests followed by 114 overlapping tests
after the final changes. Those counts combine unit, contract, and integration
checks; they are not counts of independently validated inference problems.

- [Automatic-search regressions](../../tests/test_hmc_automatic_search_regression.py)
  exercise automatic preparation and broad search on isotropic and anisotropic
  Gaussian targets. Their assertions concern completion, verified members, and
  candidate records. They do not assess retained posterior accuracy. The
  recorded executions precede the latest diagnostic repair.
- The latest [posterior precision example](../examples/hmc_posterior_precision.py)
  exercises actual tuning, verification, member construction, discarded warmup,
  and retained mean/quantile assessment on a two-dimensional Gaussian. It
  supplies a precomputed mass and one initial pair, so it is not an automatic
  geometry-plus-broad-search integration test.
- [Neural-force tests](../../tests/test_neural_force_hmc.py) include momentum-flip
  involution, volume preservation, a wrong-energy counterexample, and numerical
  health cases. These are useful mechanism tests for that route. They do not
  establish coverage of every public ordinary or fixed-transport path.
- The [latest repair tests](../../tests/test_hmc_overall_repair.py) include
  checkpoint, budget, seed, precision, and missed-mode counterexamples. Some
  isolate control flow with mocked diagnostics. The recorded stationary AR(1)
  precision calibration checks fixed-size estimates, not coverage after the
  actual sequential stopping rule.

The evidence reviewed here does not establish a current full-pipeline campaign
on nonlinear, heavy-tailed, multimodal, hierarchical, or MacroFinance targets.
Registry coverage and test names containing “all models” do not supply that
numerical evidence.

## Established methods and their proper use

| Method and source | What it checks | What BayesFilter should adopt |
| --- | --- | --- |
| Stan integrator and adaptation unit tests [S1–S3] | Individual leapfrog updates, energy error, a two-dimensional volume check, dual-averaging state, and short/incompatible warmup schedules | Independent small mathematical oracles for each active transition and preparation route |
| PyMC HMC and quadratic-potential tests [S4–S5] | Forward/backward integration, covariance/precision conventions, momentum covariance, adaptation windows, singular covariance, and freezing step size after tuning | Check geometry, transition mechanics, and the adaptation-to-retention boundary independently |
| Gandy–Scott MCMC tests; R `mcunit` [S6–S7] | Whether a fixed transition preserves the intended distribution, using generative-model identities and a reversible rank construction | Statistical unit tests of frozen kernels with a declared false-rejection rate and measured detection power |
| Simulation-based calibration, SBC [S8–S10] | Repeatedly simulate parameters and data, fit the model, and compare the generating value with posterior draws | Test the complete procedure from model construction and automatic preparation through posterior output |
| `posteriordb` [S11] | Comparisons with analytical expectations or documented reference posterior draws across models | Check posterior means, squares, dependence, quantiles, and scientific quantities against references with uncertainty |
| Dynare convergence diagnostics [S12] | Geweke beginning/end comparisons, Brooks–Gelman between-chain diagnostics, and optional Raftery–Lewis run-length estimates | Useful comparison for posterior reporting, not a substitute for implementation validation |

Stan's source test named `symplecticness` measures area preservation in a
two-dimensional example. That test is not a general proof of symplecticity.
Likewise, copying a library's numerical constants does not establish suitable
tolerances, power, or warmup lengths for BayesFilter.

### Frozen-kernel tests need a separate mixing test

Gandy–Scott Algorithm 1 draws parameters and data from the generative model,
applies a fixed number of posterior transitions, and compares the resulting
joint parameter/data distribution with independent generative draws. Algorithm
2 places the generating parameter at a uniformly random position in a path and
runs a reversible kernel outward in both directions. Proposition 2.3 gives a
uniform anchor rank, with an ordinal ranking that handles ties. The special
construction permits dependent transitions; an ordinary rank of the first
state in a Markov chain does not have this guarantee.

Apply these tests to a frozen kernel. If its settings are prepared from the
simulated data, preparation randomness and starting states must be independent
of the test's reference state conditional on those data. Applying the theorem
to an arbitrary adapting trajectory would require a separate argument.

Invariance alone does not establish exploration. The identity kernel
`K(x, A) = 1_A(x)` preserves every distribution while never moving. Therefore
an invariance test must be accompanied by movement, recurrence, and posterior
accuracy checks. Passing it does not establish effective tuning.

For standard exact-score HMC, test the score against a separate derivative
oracle as well. A wrong proposal force can still preserve the intended density
when the proposal remains reversible and volume preserving and receives the
correct full-energy Metropolis correction. A score mutation must fail the score
oracle; it need not fail every invariance test. The registry's position-field
branch remains conditional mechanics evidence and is not promoted to exact-score
authority by this review.

### SBC must test data use and dependence

For each independent replication, draw `theta_star ~ prior`, simulate
`y ~ likelihood(theta_star)`, run the complete inference procedure, and form
the rank of a declared test quantity at `theta_star` among its posterior values.
Talts et al.'s rank theorem assumes independent posterior draws; Appendix B
uses their conditional exchangeability with the generating value.

MCMC autocorrelation can distort the usual SBC ranks. Use a justified treatment
of dependence and verify the rank procedure on exact reference draws. Thinning
based on estimated ESS is an approximation to examine, not a proof of
independence. Do not apply iid rank-histogram or KS thresholds blindly to
correlated output. Gandy–Scott's random-position construction is a different
test, not a license to ignore dependence in ordinary full-pipeline SBC.

Modrák et al. demonstrate an especially relevant failure: inference that ignores
the data and returns prior draws can pass every SBC quantity depending only on
parameters. Indeed, the generating parameter and returned draws are then iid
from the prior unconditionally, giving uniform parameter ranks despite incorrect
conditional inference. Include the joint log likelihood, selected pointwise
likelihoods when data omission is plausible, parameter products/differences, and
scientifically relevant predictions. Their Section 4.3 also shows how splitting
results by observed-data summaries can expose cancelling errors. Such strata
must be predeclared and their reference calibration checked.

Use a proper generative prior and an independently checked simulator. SBC cannot
detect a shared error that changes both simulation and fitting to the same wrong
model. Retain failed fits and their reasons in the replication denominator;
success-only SBC can conceal precisely the difficult cases under investigation.
Passing finitely many quantities at finite replication count is bounded evidence,
not a proof for every dataset or scientific question.

## A test matrix organized by failure mechanism

| Target or controlled situation | Mechanism under test | Independent check or expected outcome |
| --- | --- | --- |
| Exact Gaussian/quadratic | Mass versus inverse mass, momentum generation, leapfrog and Metropolis energy | Closed-form energy, score, leapfrog matrix, moments and covariance |
| Rotated, ill-conditioned Gaussian | Geometry preparation, whitening, reconstruction, chain batching | Known covariance and affine change-of-variables identities; transformed and original posterior agreement |
| Gaussian resonance | Perfect acceptance with no movement or a two-cycle | Derived recurrence below; movement/recurrence failure must remain visible |
| Banana distribution built from a known transform | Spatially changing curvature and nonlinear Jacobian/score | Independent transformed iid draws and explicit change-of-variables density |
| Centered/noncentered funnel or eight schools | Warmup transients, varying curvature, parameterization sensitivity | Direct funnel draws or a checked model-coordinate reference; both parameterizations represent the same target |
| Student-t and Cauchy | Tail exploration and invalid moment assumptions | Exact CDF/quantiles; mean/variance accuracy claims only when the required moments exist |
| Separated Gaussian mixture | Missed modes despite favorable local diagnostics | Known mode weights, CDF and cross-mode functionals; include starts confined to one mode |
| Positive, bounded and simplex parameters | Support, transformations and omitted Jacobians | Gamma/Beta/Dirichlet reference distributions and independently evaluated transformed densities |
| Conjugate normal and Beta-binomial generative models | Full data/parameter pipeline and SBC harness | Analytical conditional posterior and iid reference ranks, including ignored-data mutations |
| Linear Gaussian state-space model | Filtering likelihood/score plus HMC parameter inference | Exact Kalman likelihood and independent score checks; unknown-parameter posterior needs its own reference |
| Nonlinear state-space and representative MacroFinance model | Actual consumer data, model-coordinate quantities and target lineage | Matched-density reference or bounded numerical benchmark; distinguish filtering approximation from HMC sampling error |

These are proposed fixtures, not a statement that all are already tested. The
small matrix should be selected for distinct failure mechanisms before adding
many similar models. A funnel or mixture that cannot be adequately sampled under
the stated budget should produce an honest failure or unresolved assessment.
Relaxing its criteria until it passes would defeat the test.

There is a useful exact corner case for the user's broad-L procedure. For
`U(q)=q^2/2` and unit mass, a kick-drift-kick leapfrog step is

```text
p_half = p - epsilon*q/2
q_new  = q + epsilon*p_half
p_new  = p_half - epsilon*q_new/2

A(epsilon) = [[1-epsilon^2/2,              epsilon],
              [-epsilon*(1-epsilon^2/4),  1-epsilon^2/2]]
```

Substitution gives `det(A)=1`. At `epsilon=sqrt(2)`, `A^2=-I` and `A^4=I`.
Thus two leapfrog steps propose `q_new=-q`; four return to the original state.
Both preserve the endpoint Hamiltonian exactly in exact arithmetic, yielding
acceptance one, even with freshly drawn momentum at each HMC transition. One
case cycles and the other does not move. A check of squared position also
reveals the two-cycle. This is a local algebraic derivation, not an empirical
result or a numeric default borrowed from another library. Floating-point tests
need tolerances derived for their dtype and operation count. Near-resonance
tests should also inspect movement; the current acceptance upper bound might
already reject the exact example, which alone would not test recurrence logic.

## BayesFilter-specific mechanism checks

Use the [interface reference](../reference/hmc-tuning-interface.md) and
`HMC_TUNING_INTERFACE_CAPABILITIES` to enumerate eligible public routes. The
ordinary and frozen-transport tuners share the controller, but have distinct
preparation and target-coordinate obligations. Test their actual numerical
bindings as well as controller logic.

A deterministic evaluator with an independently specified event transcript is
appropriate for exhaustive lifecycle checks: several surviving L families,
multiple epsilons at the same L, all-member retention, individual fresh
verification, candidate-specific repair, nonmonotone acceptance, inconclusive
evidence, exact budget boundaries, repair reserves, interruptions, resume,
candidate-local errors, and shared invalidity. It should assert the required
observable sequence and final set, not reimplement the controller's selection
logic as its oracle. Numerical integration tests must then exercise these
mechanisms with real transitions.

In particular, high, missing, or failed R-hat must not change the tuning decision,
repair schedule, or retained candidate set. ESS and MCSE also remain outside
tuning qualification. Their posterior-assessment behavior is tested separately.
No first passing candidate may terminate the declared funded cohort.

The final integration tests must call the public tuner with automatic geometry
preparation, per-L proposals, measurement, individual verification and repairs;
export/reload actual verified members; then run discarded warmup and retained
assessment using the same target and kernel. For the small acceptance fixtures,
exercise every verified member. For expensive targets, predeclare which members
receive posterior assessment and leave the others without posterior-validation
claims. Do not rank or discard members using descriptive diagnostics.

At the warmup boundary, verify that adaptation has stopped, tuning/warmup draws
are excluded from estimates, all phase streams are distinct, and restart
preserves the specified sequence without repeating or losing draws. Compare
chunked and uninterrupted execution under the same numerical and seed contract;
CPU/GPU comparisons require numerical/statistical tolerances, not universal
bitwise equality.

## The tests themselves need validation

Introduce controlled test-only mutations: omit a nonlinear Jacobian, reverse
the MH energy difference, inconsistently invert the metric, perturb the score,
ignore data, duplicate streams, leak warmup draws, discard an eligible candidate,
transfer epsilon qualification across L, and lose/repeat a restart chunk.
Assign each mutation to the oracle that should detect it. Some defects require
deterministic contract checks rather than a stochastic distribution test.

For statistical tests, measure false rejection on correct implementations and
power against the named incorrect ones, with uncertainty. A test that never
rejects either is not useful. A fixed seed makes a failure reproducible; it does
not calibrate false rejection. Predeclare test quantities, multiplicity handling,
replication count and computation budget. Derive the allocation from the size
of defect to detect and tolerated error probabilities, instead of inventing a
universal number of fits.

The Gandy–Scott sequential wrapper has specific assumptions, including suitable
stage p-values and independent stage vectors. Repeatedly rerunning an ordinary
test until it passes does not inherit its guarantee. The inspected R code also
uses numerical/asymptotic p-value calculations; distinguish exact null rank
identities from exact finite-sample calibration of a particular implementation.

For burn-in and precision, compare stationary reference starts with deliberately
dispersed, remote, and single-mode starts. Across independent full runs, assess
the distribution of estimates at the actual stopping time, failure-to-complete
rate, reference error, and interval coverage where intervals are claimed.
Compare that with fixed-length assessment. Stationary AR(1) arithmetic checks
do not validate detection of transients, missed modes, or coverage after
repeated looks. Lugsail estimates precision under its assumptions; it does not
determine whether burn-in removed bias.

Also calibrate the acceptance screen itself near its declared boundaries. Use
an independent reference estimate with quantified uncertainty under the same
start and measurement protocol, and measure false qualification, false
rejection, and inconclusive rates across replications. Include temporally
correlated acceptance decisions and the actual evidence-extension rule. This
would assess the operational screen; it must not silently turn its current
compatibility intervals into a claim of nominal sequential confidence coverage.

## Recommended order and skeptical review

1. Build the mechanism/oracle/mutation inventory and extend deterministic
   transition, geometry, lifecycle, and adaptation-boundary tests.
2. Add frozen-kernel statistical tests and validate the statistical harness
   against exact samplers and named mutations.
3. Run full automatic-path integration on analytical and transformed targets,
   then conjugate-model SBC with data-dependent quantities.
4. Add transient, heavy-tail and multimodal assessment tests, including actual
   stopping-rule calibration.
5. Expand to a curated `posteriordb` subset and the actual MacroFinance consumer.
   Match data, priors, transformations and Jacobians before comparing with Stan
   or PyMC. Compare posterior quantities with reference Monte Carlo uncertainty;
   NUTS and fixed-L HMC need not choose the same epsilon or L.

These stages are a recommendation, not an executable research campaign yet.
Before stochastic execution, record target definitions, measured pilot costs,
test error/power requirements, total budget, environment, seeds, versioned
output root and stop conditions in a bounded experiment plan. Routine
deterministic checks can remain in CI; larger replicated checks belong in an
explicit extended suite. Runtime tests should use the TF/TFP implementations;
independent test oracles may use diagnostic/reference numerical tools. Preserve
the GPU/XLA default and label any small CPU/non-XLA reference exceptions.

The skeptical review found and corrected five possible design errors in this
recommendation: treating invariance as mixing; parameter-only SBC as detection
of ignored data; iid rank thresholds for dependent MCMC; prepared-pair examples
as full automatic-pipeline coverage; and fixed-size MCSE calibration as
validation of stopping. It also checked reference uncertainty, hard-target
failure classification, the distinction between gradient correctness and
Metropolis-corrected invariance, and moment assumptions for heavy tails.
No replication count, test alpha, or model-scale acceptance criterion is promoted
without a subsequent power/budget rationale. Source review is sufficient for
this recommendation; it is not execution evidence for the proposed tests.

## Sources inspected and provenance

Local paper, manual and source copies are under
`.localresources/papers/hmc_validation_standards_20260915/`. The retrieval JSONs
record URLs, file hashes, and failures. Stan sources below are pinned to commit
`aa74058458dec5fe2c33d44dcf672143a4ddc346`; other moving-branch downloads are
identified by their local SHA-256 records. Source functions were inspected;
external library test suites were not run.

- **S1:** Stan [`expl_leapfrog_test.cpp`](https://github.com/stan-dev/stan/blob/aa74058458dec5fe2c33d44dcf672143a4ddc346/src/test/unit/mcmc/hmc/integrators/expl_leapfrog_test.cpp)
  and [`expl_leapfrog2_test.cpp`](https://github.com/stan-dev/stan/blob/aa74058458dec5fe2c33d44dcf672143a4ddc346/src/test/unit/mcmc/hmc/integrators/expl_leapfrog2_test.cpp),
  individual updates, `energy_conservation` and `symplecticness`.
- **S2:** Stan [`stepsize_adaptation_test.cpp`](https://github.com/stan-dev/stan/blob/aa74058458dec5fe2c33d44dcf672143a4ddc346/src/test/unit/mcmc/stepsize_adaptation_test.cpp),
  `learn_stepsize` tests.
- **S3:** Stan [`windowed_adaptation_test.cpp`](https://github.com/stan-dev/stan/blob/aa74058458dec5fe2c33d44dcf672143a4ddc346/src/test/unit/mcmc/windowed_adaptation_test.cpp),
  `set_window_params1/2/3`.
- **S4:** PyMC [`test_hmc.py`](https://github.com/pymc-devs/pymc/blob/main/tests/step_methods/hmc/test_hmc.py),
  `test_leapfrog_reversible` and `test_nuts_tuning`.
- **S5:** PyMC [`test_quadpotential.py`](https://github.com/pymc-devs/pymc/blob/main/tests/step_methods/hmc/test_quadpotential.py),
  energy/velocity, covariance/precision equivalence, random momentum, weighted
  covariance, update-window and noninvertibility tests.
- **S6:** Gandy and Scott, [*Unit Testing for MCMC and other Monte Carlo Methods*](https://arxiv.org/abs/2001.06465),
  inspected version September 2021: Sections 2.1–2.2, Algorithms 1–2,
  Proposition 2.3; Section 3, Algorithm 3 and Theorem 3.1; Section 4.1 mutation
  example and appendix proofs of Proposition 2.3 and Theorem 3.1.
- **S7:** R [`mcunit`](https://github.com/cran/mcunit), inspected
  [`expect_invariant.R`](https://github.com/cran/mcunit/blob/master/R/expect_invariant.R),
  `expect_mcmc_reversible` and `expect_mcmc`. This is a bounded source inspection,
  not an audit of every package function or its sequential wrapper.
- **S8:** Talts et al., [*Validating Bayesian Inference Algorithms with Simulation-Based Calibration*](https://sites.stat.columbia.edu/gelman/research/unpublished/sbc.pdf),
  Sections 4.1, 5.1 and 6.2, Algorithms 1–2, and Appendix B. The inspected
  author-hosted copy was retrieved on the date of this note.
- **S9:** Modrák et al., [*Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity*](https://arxiv.org/abs/2211.02383),
  version 3: Sections 1.1–1.2, 3.1–3.4, 4.3, 6.1–6.3, and the appendix proof
  of Theorem 4.
- **S10:** R [`SBC`](https://github.com/hyunjimoon/SBC), README and diagnostic
  aggregation code inspected at tree `3a8e632fd4d9e117073761d07502c730cbcac85b`.
  Its rank implementation was not audited here.
- **S11:** Magnusson et al. (2025), [*posteriordb: Testing, Benchmarking and Developing Bayesian Inference Algorithms*](https://proceedings.mlr.press/v258/magnusson25a.html),
  Sections 2.1–2.3, 3.3–3.4 and 4.1; repository
  [reference definitions](https://github.com/stan-dev/posteriordb/blob/master/doc/REFERENCE_POSTERIOR_DEFINITION.md).
  References have uncertainty and are not available for every model. This note
  does not adopt numeric reference-quality thresholds without a separate check.
- **S12:** Dynare [model-file manual](https://www.dynare.org/manual/the-model-file.html),
  Sections 4.16.3 (Algorithms) and 4.16.3.13 (Convergence diagnostics), including
  `geweke_interval` and `raftery_lewis_diagnostics`. Raftery–Lewis uses a
  first-order Markov approximation for run-length estimates; it is not a
  universal proof of sufficient burn-in.

An initial guessed SBC identifier, arXiv `1805.09294`, retrieved an unrelated
Lueckmann et al. paper. Files named `sbc-1805.09294.*` are explicitly excluded;
the SBC claims above use the checked Talts author copy and Modrák paper.
