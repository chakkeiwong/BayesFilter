# Closure master implementation review — 2026-09-22

This review concerns the closure driver, its numerical workers, the wrapped
observation-law repair and their connection to the existing master. The goal
is testing surrogate HMC: full canonical analytical DCTR supplies a fixed,
deterministic proposal force; separately declared endpoint values determine
Metropolis acceptance. A diagnostic repair is not a successful posterior test.
The first executable stages are implemented. The full structural/generative
consumer, remaining posterior references and conditional cost comparisons are
still unfinished. The driver must expose those gaps, not claim to close them.

## Mathematical and scientific checks

An imperfect force can be used in this experiment without being the endpoint
log-density gradient. For position-only force F, a momentum kick
p <- p + epsilon F(q) and a position drift q <- q + epsilon M^{-1}p each
have triangular Jacobian with determinant one. Symmetric kick–drift–kick
composition reverses when momentum is negated. With fixed settings and a
reversible, volume-preserving proposal, the acceptance probability is
min(1, exp(log pi(q') - K(p') - log pi(q) + K(p))). The endpoint pi, not F,
specifies the invariant target. This argument requires the actual consumer to
meet those conditions; it does not establish mixing, accurate finite-filter
likelihoods or efficiency. The closure driver does not change that consumer.

The four comparisons remain distinct: B/R compares model-endpoint chains to
the model posterior; C/D compares finite-endpoint chains to the finite
posterior; D/R measures the finite approximation; C/R combines both errors.
Archived formal verdicts are preserved. LG1's owner-accepted practical result
is a separate field and cannot be converted into a formal equivalence claim
or silently reopened by the closure driver.

For the range/bearing observation law, let v = R[0,0], c = R[1,0],
beta = c/v and s² = R[1,1] - c²/v. The wrapped joint density is
N(r;0,v) times the sum over integer k of
N(a - beta r + 2 pi k;0,s²). Conditioning handles correlated noise correctly;
a nearest-angle unwrapped Gaussian does not normalize on the circle for broad
noise. The implementation sums aliases after wrapping the conditional angular
residual. Its finite truncation omits at most the declared tail mass, rather
than being an exactly normalized replacement for the infinite law. For the
specified maximum conditional standard deviation pi, four aliases on each
side bound omitted mass by 2.2572e-19. The analytical directional derivative
includes residual and covariance dependence (including beta and s²), away
from wrapping cut points. Invalid covariance/domain inputs are flagged without
changing an accepted covariance by a ridge. This does not qualify a complete
M13 marginal likelihood or the canonical M13 recursive score.

The uncertainty experiment uses normalized trapezoid weights on the existing
KSC T2 reference grid to define an explicitly finite probability law. Its 13
function expectations are computed from that law. IID draws and stationary
refresh chains therefore have known diagnostic truth. The latter have
transition P = rho I + (1-rho) Pi and are stationary because each first draw
comes from Pi. This construction exercises dependence without pretending to
be a new HMC posterior run. Chain offsets, drift and excessive persistence are
rejection controls. Continuous-posterior integration error is excluded from
this experiment, and must be included in a future posterior comparison.

Coverage uses both predeclared looks and all 13 functions, with multiplicity
accounted for in the interval critical value. Group-level binomial intervals
measure uncertainty in failure rates. The first dependent groups did not meet
the declared upper-bound screen. One larger confirmation was specified before
new draws, with disjoint seeds and identical method, margins and thresholds;
its results are not pooled with the initial run. This avoids changing the rule
or rescoring archived posterior chains in response to a borderline outcome.
Passing this diagnostic cannot promote a runtime default or all-model validity.

## Call-chain and execution review

The script dispatches `--stage closure` to
`bayesfilter.inference.surrogate_hmc_closure_program.main`. That driver always
runs its review dependency before the selected numerical jobs. Regression
failures block dependent jobs; a scientific candidate's negative result is
preserved independently of process completion. Source snapshots, hashes,
commands, devices, seeds, logs and incremental accounting are retained in
fresh directories. A dry run does not execute numerical jobs.

The wrapped worker calls the actual range/bearing fixture observation callback,
which resolves to the new TensorFlow kernel. Independent unit tests compare
its correlated density to a bivariate alias calculation and its analytical
tangent to finite differences, and exercise periodicity and invalid inputs.
The GPU worker checks numerical unit mass across narrow and broad noise.

The AR2 worker calls `canonical_batch_fused_value_score` with Q = diag(.09,0).
It confirms the factorization failure in the actual consumer; a separate
structural/Kalman fixture is not substituted for that consumer. Supporting AR2
still requires innovation-space proposals, weights on the correct measure,
deterministic completion, and the full canonical reset/covariance/analytical
score path. Adding noise to the lag coordinate would change the model.

The uncertainty worker calls the existing diagnostic implementations on actual
posterior function types. The generator and batch/randomization kernels use
TensorFlow/XLA. Tie-aware rank diagnostics retain an explicitly recorded
non-XLA diagnostic graph. GPU workers verify memory growth before their GPU
probe. CPU reference workers deliberately hide GPUs. No pfor or NumPy runtime
path was introduced.

## Findings and repairs

- Fixed tensor serialization before the first numerical launch.
- Replaced an inheritable parent CPU limit with a process-local timer, while
  retaining child CPU and wall-time watchdogs.
- Preserved setup failures and charges, required fresh output paths and included
  every failed closure attempt in subsequent reservations.
- Expanded review to include the existing master and dependency tests. This
  exposed stale SV20 history isolation, allocation and status-schema assertions.
  Fixed the tests without changing archived results or the documented budget.
- Corrected the attempt limiter so successful or repaired routine review runs
  do not consume numerical experiment attempts. Reviews still must pass and
  all work remains charged against the original reservation. An executable
  test verifies the numerical attempt limit is retained.
- Final integration review found that the original master did not discover
  closure accounting and the closure driver missed interleaved original-master
  jobs. Both now use shared receipt discovery. Tests exercise the original
  master entry point, reject unfinished receipts, avoid nested double charging,
  and distinguish campaign charges from this pass's reservation.
- Execution 05 exposed tests reading the live campaign while their own review
  was unfinished. Isolated those fixtures; retained the real unfinished-receipt
  guard and the independent receipt-wiring tests. No inference rule changed.
- Reconciled the opening cost statement against 161 receipts. Its excess,
  214.712703 CPU and 79.294813 GPU seconds, stays reserved and is explicitly
  labeled unverified bookkeeping, not measured additional computation.

Final executable review: **70 tests passed** in execution 06, with no numerical
jobs rerun. All six attempts preserved unchanged source snapshots. Executions
02, 03 and 05 failed at tests and launched no numerical experiments. Execution
01 preserves the first numerical pass; execution 04 preserves the 2,048 fresh
confirmation replications. Both dependence groups passed the unchanged scoped
criteria. Both master budget readers agree on the final charged balance.
See [the result note](surrogate-hmc-closure-master-result-2026-09-22.md).

Review verdict: the bounded diagnostic and repair driver passed this review.
The complete all-model surrogate-HMC program remains **incomplete**. A static
coverage entry or a passed diagnostic is not an implemented missing consumer,
qualified posterior reference or completed model-level experiment. Claude
review remains deferred as requested.

## Remaining limitations

The readiness table is a conservative report of prerequisites for a new
prospective scope. It is not an implemented inference scheduler that can repair
missing model mathematics. Most unqualified model variants still need concrete
scope definitions, independent references and consumer qualification. Existing
LG2 preparation is preserved, but its forecast exceeds its family allocation
and a prospective precision design is missing. New retained HMC stages require
their scope-specific mathematical, reference and precision prerequisites; the
existing owner authorization does not supply that missing scientific evidence.

The strongest alternative explanation for passing the uncertainty experiment
is that this finite KSC law and simple dependence family omit difficulties of
actual HMC chains. A failure on a separately specified target/function/dependence
scope would overturn transfer to that scope. The wrapped-law checks are strong
local evidence, but an integrated M13 posterior check remains absent. There is
no statistically supported sampler ranking or efficiency claim.
