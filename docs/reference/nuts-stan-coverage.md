# Stan 2.37 sampler transfer: source and test coverage

The implementation is [stan_nuts.py](../benchmarks/stan_nuts.py). The
[guide](nuts-direct-stan.md) describes its generic TensorFlow interface;
the [execution record](../plans/bayesfilter-full-stan-nuts-port-result-2026-10-27.md)
records the actual test and benchmark outcomes. Source fidelity means agreement
with the pinned Euclidean multinomial NUTS sampler on the supported inputs.
It does not mean that the Stan model compiler, every CLI convention, or its
seeded random stream has been transferred.

Native paths in this table are relative to
`.localresources/cmdstan/cmdstan-2.37.0/stan/src/stan/`. Tests build the installed
headers rather than a rewritten reference. The release archive SHA-256 is
`635e2e1cf9c4774c9019001325354d3610bae2adc4ed5af2fed87872dfc671fc`.
Build records retain the source/header hashes actually used.

| Behavior | Native source anchor | Translation | Evidence and limits |
| --- | --- | --- | --- |
| Tree expansion, direction, early return, candidate cache | `mcmc/hmc/nuts/base_nuts.hpp`, `transition`, `build_tree` | `Kernel.transition`, `Kernel.subtree` | Actual native transitions with supplied momenta/uniforms: five targets, unit/diagonal/dense metrics, both directions, invalid proposals and depth limits |
| Inner normalized and outer biased proposal selection | Same, the two `log_sum_weight` comparisons | Source-ordered log weights and conditional draws | Native selected position/momentum/value/gradient and random-consumption equality; this is stronger than matching marginal moments |
| Three strict momentum U-turn tests | Same, `compute_criterion` and the two extended-rho checks at each merge | `criteria`, `criterion` | Native transition fixtures include zero-momentum and exact direction ties; source inspection includes Stan's `base_nuts_test.cpp` rho/endpoint tests |
| Leapfrog | `mcmc/hmc/integrators/expl_leapfrog.hpp` | `leapfrog` | Native header integrator runs inside the transition oracle; retained cache/replay checks |
| Unit, diagonal and dense inverse metrics | `mcmc/hmc/hamiltonians/{unit,diag,dense}_e_metric.hpp` | `velocity`, `energy`, `factor`, `momentum` | Kinetic/velocity algebra inspected; native transitions test supplied physical momentum. Normal generation uses TF; dense factor application is the same upper-triangular solve |
| Divergence and acceptance/work accounting | `base_nuts.hpp`, leaf calculation and final transition summary | Strict energy threshold, all evaluated leaves counted | Two exact threshold fixtures, invalid subtrees and native telemetry comparisons |
| Step-size jitter | `mcmc/hmc/base_hmc.hpp`, `sample_stepsize` | `transition` | Native supplied-uniform cases with jitter .17; nominal/realized values separate |
| Initial and metric-boundary step search | Same, `init_stepsize` | `find_epsilon` | Actual native search with scripted probe energies: grow, shrink, tie, initial above cap, cap failure and NaN energy; preserves repeated first probe |
| Dual averaging and completion | `mcmc/stepsize_adaptation.hpp`; `nuts/adapt_*_e_nuts.hpp` | Shared `dual_update`, warm-up completion | Native arithmetic/restart tests and replay of actual direct-port warm-up positions/acceptances |
| Window schedule and covariance shrinkage | `mcmc/{windowed,var,covar}_adaptation.hpp` | `schedule`, warm-up Welford updates | Native short/zero/custom/default schedules and covariance replay, including singleton windows; pinned short-budget no-restart behavior retained |
| Initial and reset adaptation order | `services/sample/hmc_nuts_*_adapt.hpp`; `services/util/run_adaptive_sampler.hpp` | Initial mu before search; boundary mu after search | Source audit and native replay; boundary searched epsilon supplied to isolate adaptation from its separately tested search |
| Initialization retries and validity | `services/util/initialize.hpp`; `io/random_var_context.hpp` | `initialize` | Native flat-parameter service oracle for supplied, partial, random and zero starts, failure/retry counts and finite-gradient-sum overflow; no named constrained-variable parity |
| Initial metric checks | `services/util/validate_{diag,dense}_inv_metric.hpp`; Stan Math symmetry/positivity checks | `valid_metric` | Source inspection, finite/positive/SPD local tests; TF Cholesky and native Eigen validation can differ at numerical boundaries |
| Disabled adaptation, warm-up zero, thinning | `services/util/{run_sampler,run_adaptive_sampler,generate_transitions}.hpp` | Static phase branches and zero-based save counters | Source-inspected service order; local lifecycle tests compare thinned draws with the unthinned stream |
| Retained continuation and batching | Stan per-chain state semantics; TF-specific execution design | `FrozenState`, per-chain keys/counters and masks | Exact same-backend chunk continuation and scalar/batch isolation tests; duplicate batch keys rejected |
| Entire numerical lifecycle compiled | TF/XLA implementation requirement | Three stable-signature enclosing functions | Optimized HLO for initialization, warm-up and sampling; CPU/GPU unit/diagonal/dense checks, no Python callback or pfor; graph growth test at depths 3, 10, 20 |

## Deliberate interface differences

The callback takes an unconstrained FP64 matrix and returns one density per
independent row. The caller supplies parameter transforms and Jacobians.
Numerical domain failures are nonfinite tensor values. Python exceptions,
Stan named parameter contexts, generated quantities, interrupts and CSV
writers are outside the compiled sampler.

The sampler uses TensorFlow Philox and TensorFlow normal transforms.
Common-variate transition equality does not establish equality of chains
started from the same integer seed in Stan. Distinct stream keys are required
across chains and fits; the second seed word is a counter. Counter overflow
invalidates the result. Retained chunk continuation preserves the random state;
thinning restarts its save index per call, as the service does per phase. To
preserve a global thinning pattern across chunks, use chunk lengths divisible
by `thin`, or retain unthinned output and thin after concatenation.

This API validates numerical options explicitly. In particular it rejects
jitter=1, whereas CmdStan accepts that CLI value and the native sampler setter
ignores it; it limits depth to 30 for int32 work counters. It permits delta
between zero and one as the native sampler setter does; the CmdStan 2.37 CLI
uses the narrower lower bound .1. Buffers/counts must fit signed int32, while
native buffer arguments use unsigned integers. Invalid metrics/nonfinite
retained states return a failure flag, checked by `require_valid`, rather than
reproducing native log messages or exception types.

Zero-dimensional targets use a documented local constant-evaluation convention.
Stan's exact zero-parameter service behavior is not reproduced. The positive-
dimension Euclidean sampler is the transfer target. Riemannian/SoftAbs NUTS,
other Stan inference algorithms, Stan Math's full distribution library and
bitwise CPU/GPU/native agreement remain outside this work.

## Review conclusion

Native oracles, source inspection, local invariant tests and posterior
comparisons answer different questions. None alone proves universal sampler
correctness. The source matrix supports the declared Euclidean transition and
adaptation implementation; the result note determines which tested targets
passed the posterior screens and what performance claims the experiment can
support. No repository production default or retired public NUTS route is
changed by this standalone implementation.
