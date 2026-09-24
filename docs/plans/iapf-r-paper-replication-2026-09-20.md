# Independent R iAPF reference and first paper replication

2026-09-20. The owner requests a complete, noncanonical R reference, tested
against the paper before it is used to diagnose BayesFilter. The owner selected
the first linear-Gaussian study, starting with bounded runs and scaling toward
published settings. This is a new reference campaign, not a reopening of the
exhausted nonlinear-score allocation.

Current stage result: implementation and bounded study finished; fitted paper
replication failed and remains unresolved. The [result note](artifacts/iapf-r-paper-replication-20260920-01/result.md)
is authoritative for execution status. Exact-twist controls pass through d80;
renewed d5/d10 fitted batches hit variance boundaries. No experiments are
running. Seven launches used 248.38/1800 worker seconds; one slot remains.
Resolve the fitting prescription using saved failures before any larger run.

## Research intent and evidence contract

Question: can an independent R implementation of Guarniero, Johansen and Lee's
equations (5)--(6), (15)--(16) and Algorithms 3--5 reproduce the first Section 5.2
likelihood experiment? Implement from the equations in a separate diagnostic
module; preserve the public R source unchanged. R is an owner-requested
independent reference, never a canonical TensorFlow runtime or a score/HMC
implementation.

The first mathematical criteria are Gaussian product/mixture identities,
telescoping path density, exact-twist agreement with Kalman, both adaptive
resampling branches, profiled equation-(15) loss and derivative, a positive
equation-(16) floor, zero-based stopping and a fresh final run. These are
required before the experimental ladder. Unit fixtures must exercise the
actual reference call chain; a separately coded formula is an oracle, not a
substitute for execution.

For empirical replication use the paper's model: m=0, initial/process/observation
covariances I, observation matrix I, A[i,j]=0.42^(abs(i-j)+1), T=100, dimensions
5,10,20,40,80. Published particle counts are iAPF N0=1000, BPF N=10000,
fully adapted APF N=5000; k=5, tau=.5, ESS fraction=.5. The first study uses
1000 replicates per dimension. The bounded ladder first measures capacity and
then uses 32 independent repeats at feasible dimensions; increase toward 1000
only within the remaining budget. A reduced-repeat result is explicitly a
preliminary replication, never a completed reproduction of the paper's tables.

Baseline/adversary set, constructed from this model: (1) BPF uses the transition
as proposal, (2) fully adapted APF uses the exactly available one-step
observation update and ancestor lookahead, (3) sequential importance sampling
uses unmodified transition proposals without resampling, and (4) Kalman
provides an exact likelihood. The Kalman method dominates particle methods on
this tractable model; the experiment tests reproduction of the particle-method
comparison, not promotion of iAPF over Kalman. Report every method separately
by dimension, and prefix likelihood errors in ordinary and large-innovation
time steps (Kalman innovation quadratic form above the fixed chi-square .9
quantile). Heuristic losses must be stated, not hidden in aggregate metrics.

Primary empirical quantities are likelihood ratio to Kalman, its mean and SD,
and paired bootstrap intervals for variance ratios between particle methods.
The exact Gaussian likelihood is the authority. An equivalence interval for
the mean ratio entirely inside [.9,1.1] is a bounded-run accuracy screen, not
proof of unbiasedness. Variance ordering requires its 95% bootstrap interval
to exclude 1; otherwise report descriptive differences only. Also report the
published SD and resampling counts as external descriptive benchmarks. Their
conditional data differ, so equality with a rounded table entry is not a
correctness test. Average N, iteration counts, floor mixture probabilities,
fit convergence and timings are explanatory diagnostics. No selected-fit,
score, nonlinear case-1900, production/default or HMC claim follows.

Nonfinite arithmetic, failed covariance factorization, corrupted artifacts,
wrong reference identities or silently changed objectives are continuation
vetoes. A numerical fitting failure triggers a recorded local solver repair
and fresh pilot, not rejection of iAPF. A failed statistical or heuristic screen
vetoes promotion but does not by itself veto the planned diagnostic follow-up.
Iteration/particle/time caps must return an explicit incomplete status, never
silently supply the last adaptation likelihood as a valid final answer.

## Published details and reconstruction choices

Primary source: `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`,
especially Section 5.1 pp.18--19 and Section 5.2 pp.20--22; inspect algorithms
and proofs as well as the experimental description. The public comparator's
four departures are preserved in the preceding conformance report. Its authorship
is unverified. The original data, seed, optimizer settings and exact positive
function c(N,m,Sigma) are not specified in the inspected paper. Therefore this
is an independently reconstructed implementation and statistical replication
on freshly generated data, not a byte-for-byte reconstruction of the authors'
program or data.

| Choice | Provenance and justification | Failure risk and early diagnostic | Status |
|---|---|---|---|
| Gaussian plus positive constant | Equation (16); sample the corresponding exact two-component Gaussian mixture | Adding the floor only to weights would target the wrong filter; independent mixture density and path identity tests | Paper-required family |
| Floor density at chi-square tail probability N^-2 | Explicit reconstruction c=peak_density*exp(-qchisq(1-N^-2,d)/2); N Gaussian draws have expected 1/N points beyond that contour | Floor may affect high-dimensional proposals; record component probabilities and compare ideal-twist/no-floor authority | Hypothesis, not recovered author setting |
| Profile lambda in equation (15) | Analytic least-squares optimum b/c | Confusing absolute and relative loss; direct objective and finite-difference gradient tests | Derived exact equivalence |
| Fixed scaling of the fit loss | Prevent high-dimensional density units from defeating optimizer tolerances; scaling fixed per fit | Parameter-dependent scaling changes the minimizer; invariance tests | Numerical reconstruction |
| Diagonal log-quadratic initialization, L-BFGS-B analytic gradient | Paper specifies numerical minimization and diagonal covariance but no solver; initialize near Gaussian shape | Local minima, ill-conditioned log regression and absolute-loss broadening; log fit diagnostics and exact Gaussian recovery | Reconstruction, not paper default |
| Doubling allowed at zero-based l=k, stopping first at l=k+1 | First complete history window in Algorithm 4; the paper does not define negative history indices | One early doubling can differ from the implementation behind the published average N=1000 at d<=20; retain exact controller tests and report N histories | Literal earliest-window interpretation, author implementation not recovered |
| Newly generated data; independent method seeds | No original data/seed found; fixed seed manifests and CSV hashes | Mistaking data differences for failed replication; exact Kalman tie-out | Replication limitation |
| 32 repeats before 1000 | Bounded capacity and precision ladder | Uncertain tails/rankings; bootstrap intervals, no full-replication claim | Preliminary rung |
| Base R, CPU, at most two workers | Independent reference requested; no new packages required | BLAS oversubscription; set OMP/BLAS threads to one | Reference-only exception |

Implementation audit before tests: the fully adapted comparator must use the
paper's ESS threshold, not automatic resampling (the latter would force 99
resamplings and contradict Table 2's comparator). Implement its lookahead and
carried-weight bookkeeping separately. Observation-only twists provide a
parity check of that same algorithm, not a distinct heuristic. Twisted prefix
normalizers include a future potential, so conditional prefix comparisons must
remove that potential via the terminal test function in Proposition 1. Variance
parameter bounds are the numerical interval [machine epsilon,1/epsilon];
contact is an explicit failed fit, not a scientific choice of variance range.

The absolute loss in equation (15) can approach zero as density amplitude
vanishes while lambda approaches zero. A numerical optimizer reporting success
does not therefore prove useful shape fitting. Preserve absolute and relative
residual, gradient, parameter range, density scale, convergence and boundary
diagnostics. Do not silently replace the published objective to obtain a better
benchmark. Any reconstruction change must be recorded and piloted before fresh
replication observations are examined.

## Execution and budget

1. Build a standalone base-R reference with Gaussian transitions (including a
   callable transition mean), arbitrary observation log-density, exact floor
   mixture proposals, stable log weights, backward fitting, and Algorithm 4.
   Add linear-Gaussian model, Kalman and fully adapted reference comparators.
2. Run focused R/Python conformance tests and an independent exact-likelihood
   diagnostic, including T=1 and controller/error paths.
3. Pilot the full T=100,N0=1000,d=5 scope on a separate data seed. Freeze source
   and reconstruction controls before fresh-data repeated runs. Local solver
   failures are repairable within this budget; preserve each attempt.
4. Run repeated comparisons, starting at d=5 and moving up the published ladder
   as capacity permits. No adaptation on replication data; a failure produces
   a new repair stage with fresh data if needed. Preserve per-repeat results.
5. Save a decision/inference table, uncertainty intervals, source hashes,
   commands, environment, CPU choice, seeds, elapsed time, and exact next step.
   Update the master program and active checkpoint.

Initial allocation: 600 seconds of focused mechanics checks plus 1800 seconds
of summed R-worker wall time for pilots and experiments, at most eight pilot or
comparison launches, two simultaneous workers, 300 seconds per worker, at most
20 outer iterations and N<=16000. Before a launch reserve its full timeout in
the remaining budget; never overrun by starting unbudgeted workers. Resource
caps are diagnostic bounds, not new stopping rules claimed from the paper.
Output root: `docs/plans/artifacts/iapf-r-paper-replication-20260920-01/`;
every launch uses a fresh directory. Commands use `Rscript --vanilla` and
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, with
`CUDA_VISIBLE_DEVICES=-1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

### First bounded-run finding and repair protocol

The d=5 pilot completed. On fresh data, repeats 101--104 completed but repeat
105 stopped with `vanishing-density fit`; the d=10 pilot completed after 20
adaptation estimates. Preserve all three attempts. Successful repeats from the
interrupted batch cannot establish replication: failure may select which
estimates are observed. Before another experiment, replay only the failed fit
under the mechanics allocation and save its inputs, optimizer parameters and
unscaled log densities. Distinguish diagnostic underflow from an optimizer
escaping through the absolute-loss degeneracy. A stable diagnostic may be
repaired without changing equation (15); a constrained or normalized objective
would be a separate extension and cannot silently replace the paper target.
Any solver reconstruction change needs a new pilot and fresh replication data.
Skeptical audit: this replay answers the observed failure directly and avoids
reporting a success-only variance estimate as a paper replication.

The replay localized the failure to backward time 44 of the first adaptation:
the largest scaled log density was -434.5, squared densities underflowed, and
the fitted mean/variances escaped far from initialization. Before selecting a
repair, compare three base-R local solvers on this identical saved fitting
problem: BFGS, Nelder--Mead and `nlm` with scaled maximum step one. For `nlm`,
one unit is the Euclidean norm in coordinates measured in initial standard
deviations and log variances; it limits each iteration, not the final parameter
space. Keep equation (15), its fixed scaling, and analytic derivative intact.
This is a numerical reconstruction hypothesis because the author solver is
unspecified. Diagnose finite covariance, residual relative to density, objective
decrease, gradient and convergence on the saved failure and known-Gaussian
fixtures; final likelihood performance must not select the solver. Any recovered
local minimum is not a certificate of a global minimizer of (15).

All three profiled solvers escaped to densities with poor relative shape fit.
The next focused check optimizes the paper's joint variables `(m, log Sigma,
log lambda)` directly using BFGS and damped Gauss--Newton. Profiling is an exact
elimination for the objective, but need not preserve a local solver's trajectory.
The damping/trust-step choices are numerical reconstruction, justified by the
local quadratic model's actual/predicted decrease and scale measured in initial
standard deviations. This check cannot change the loss or use replication
likelihoods to tune the solver. Failure to obtain a useful stationary fit would
leave the paper replication unresolved; it would not prove iAPF itself wrong.

The profiled and joint solver probes both showed the escape direction; changing
solvers is not justified by these probes. Keep the original L-BFGS-B numerical
reconstruction. Repair only `relative_residual`: compute the density vector
after subtracting its own maximum log density for this scale-invariant
diagnostic. Keep the objective, gradient, optimizer, floor and filter unchanged.
Record log loss and a loss-underflow flag. A finite but poorly fitted Gaussian
is a valid positive-floor proposal; fit degeneracy explains poor performance
and prevents an optimization-accuracy claim but is not a filter-validity veto.
This corrects the overly strict diagnostic: a `0/0` reporting expression is
not evidence that the Gaussian itself is invalid. The saved pre-repair failure
remains visible. Recheck formula equivalence on healthy inputs, the exact
Gaussian fixture and the formerly failing seed, then use a fresh pilot and
fresh replication data. Skeptical audit: no normalized loss replaces (15), no
bad fit is removed from the comparison, and filter validity remains guarded.

### Boundary failures and discriminating oracle check

Renewed batches stopped at a floating-point variance boundary: d5 repeat202
after one complete repeat, and d10 repeat203 after two complete repeats. These
are failures of the numerical fitting reconstruction. Preserve the failed
fit inputs/parameters via structured R error conditions and replay those seeds
under the mechanics allocation; this adds observability without changing the
solver. Do not expand to 1000 repeats or rank success-only samples.

Use one of the two remaining experimental launch slots for a separate exact
Gaussian-twist check: d=5,10,20,40,80, T=100, N=1000, three independent particle
seeds per dimension and newly generated data seed 58000000+d. Compare both the
direct fully adapted APF and its observation-twist equivalent on matched seeds
(N=5000), and compare the full-covariance optimal-twist APF to exact Kalman.
The optimal-twist control deliberately bypasses diagonal fitting and floor
reconstruction; it is a mathematical oracle, not an iAPF paper-table result.
Pass: relative log-likelihood error <=1e-10*(1+abs(Kalman log likelihood)),
constant terminal weights to 1e-9 and matched fully-adapted likelihoods and
resampling counts. This tests Algorithms 1/5, Gaussian products and likelihood
accounting at paper dimensions. It cannot clear the failed fitting procedure.
Reserve 300 worker seconds within the same 1800-second/8-launch allocation.
Skeptical audit: the source of the twist and its non-replication role are
explicit; a successful oracle cannot be relabeled as a fitted-iAPF result.

Skeptical audit: PASS for staged execution. Directly patching the public code
would still leave the positive-floor proposal mixture and stopping rule
unchecked. A separate equation-based implementation and call-chain tests answer
that risk. Full replication is not yet established: omitted author settings,
different simulated data and limited repeat counts are explicit. The paper's
absolute fitting loss has a real degeneracy risk, so solver success alone must
not authorize a scientific conclusion. The current diagnostic target does not
require a canonical GPU or TensorFlow implementation, and no package mutation
or original particle-campaign extension is needed.
