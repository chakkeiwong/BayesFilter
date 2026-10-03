# Symmetric score regression as a diagnostic iAPF fitting repair

Status: COMPLETE. 446 terminal checks pass; see
[results](artifacts/iapf-score-regression-20260922-01/result.md). The candidate
fails the conditional heuristic screen at dimensions 5 and 10.
Skeptical plan review PASS, after requiring full cloud whitening,
independent R checks, and a fresh downstream comparison. Authorized renewed48+48
campaign; phase cap one CPU hour and one GPU process hour, six launches <=600s
each including repairs. Root: artifacts/iapf-score-regression-20260922-01/.
This is an explicit local extension, not GJL Eq15 or a guess about author code.
The source paper Eq15 fits density values with a free target amplitude; Eq16
adds a positive floor. We keep the existing normalized Gaussian plus floor and
shared finite particle consumer, and replace only the diagnostic fitting rule.
Localized repair: attempt03 failed before data generation because the recursive
density factory rejects zero optimization steps. The QR-only comparator now
uses a diagnostic recursive wrapper around the existing bounded fit primitive
with max_steps=0. Target, data, method, budget and comparisons are unchanged.
Attempt04 is the fresh downstream retry; attempt03 and its5.001s are preserved.

## Question and derivation
Can a fitting objective using all coordinate gradients avoid vanishing density
energy, squared-target concentration and omitted-cross-term aliasing, and give
useful likelihood estimates on fresh data? For Gaussian beta,
log beta(z)=c+a'z-z'Pz/2, so grad log beta=a-Pz. With full empirical whitening
z=L^-1(x-m), C=LL', E z=0 and E zz'=I. Transform the physical target score as
h_z=L' h_x. The symmetric least-squares solution is
 a=mean(h_z), P=-sym(mean((h_z-a)z')).
This follows by expanding mean||h_z-a+Pz||²: the quadratic term is tr(P²),
so the orthogonal projection onto symmetric matrices is the minimizer.
For an exact Gaussian, full-rank clouds recover its full precision and mean;
only N>=d+1 is needed, since each observation supplies d score components.
Transform back: c=m+L P^-1 a, V=L P^-1 L'. The diagonal Gaussian minimizing
KL(N(c,V)||N(mu,D)) has mu=c and D=diag(V), by differentiation of Gaussian KL.
We use that diagonal covariance in the candidate consumer. This projection is
exact for the recovered Gaussian; non-Gaussian targets remain approximations.

For the local linear Gaussian model and fitted next guide,
 beta_t(x)=g_t(x)[N(c_next;A x,Q+V_next)+exp(log_floor_next)].
 Its score is H'R^-1(y-Hx)+w(x)A'(Q+V_next)^-1(c_next-Ax), where
 w=sigmoid(log N(c_next;A x,Q+V_next)-log_floor_next).
At the terminal time the second term is zero. Check this gradient independently
by finite differences and in R. All target evaluations use the stated fitted
next guide; no exact oracle coefficient enters the candidate fitter.

## Evidence contract and research intent
Baseline ladder: shared bootstrap filter; current-observation Gaussian guide;
full backward Gaussian oracle; existing bounded Eq15 fit from diagonal QR with
initial-peak scaling; diagonal log-quadratic regression without optimization;
then the new symmetric score fit with diagonal projection.
All learned guides receive exactly the same independent bootstrap pilot cloud
and one backward fitting pass. This isolates the fitter; adaptive iteration,
particle growth and paper replication remain subsequent questions.
Primary engineering pass: exact Gaussian recovery and independent R parity
<=1e-8, target score finite-difference error <=1e-6, exact Kalman agreement <=1e-9,
shared consumer/oracle identities <=1e-8, one trace per fixed configuration.
Primary downstream quantity: per-data-set mean squared log likelihood error
against exact Kalman, averaging eight independent final particle streams. Score
outputs are checked for finiteness only: frozen-guide finite-program scores are
not assumed to be unbiased model scores.
Candidate non-SPD/rank/finite failure is a promotion veto and retained negative
result. Density-fit nonconvergence is likewise a candidate veto, not an excuse
to omit it. Harness, model/score identity, R parity, source or artifact failure
is a continuation veto and repair trigger. A likelihood/heuristic loss never
invalidates the experiment and triggers the next discriminating repair.
Explanatory quantities: full-oracle Gaussian KL, whitening/precision margins,
score residual and symmetry, fit time, positivity, and floor effects. None
substitutes for downstream accuracy.
No default, original-paper replication, nonlinear extension, model-score,
LEDH, TF32 or HMC claim. A successful diagnostic can nominate a later adaptive
comparison only. No fitting-quality promotion from exact-Gaussian fixtures.

## Frozen design, defaults and sanity checks
- Gaussian controls: d=2,5,10,40,80, N256, deterministic seeds94121+d. Off-diagonal
  covariance constructed as .6^|i-j| plus .3I. Provenance: full-rank analytic
  control exposing the known scalar-quadratic rank gap. Guarded covariance and
  precision factorizations test the mechanism. No ridge/damping: exact SPD
  controls justify no alteration; indefinite candidates fail closed.
- Fresh data: d=2,5,10, T8, N256; seeds92101:92104 in a new phase namespace.
  Four data sets per dimension are a bounded diagnostic convenience, not enough
  for a superiority claim. Eight final particle replicates each, disjoint from
  pilot streams; frozen before outcomes. Shared streams for paired comparisons.
  The six-parameter local model uses the prior declared fit theta; this is not
  the different paper-study model. Same target/data for every arm.
- One backward pass is an explicitly scoped component comparison, not Algorithm4
  convergence. Original Eq15 controls: mean bound4, sd bounds.2..4,2000 steps,
 30 backtracks,tolerance1e-7, QR initial,initial_peak; all inherited hypotheses
  with existing diagnostics. Diagonal-QR-only baseline uses no descent. The new
  fit has no coordinate box or iterative optimizer: its full SPD inverse and
  diagonal projection define the estimator, and this difference is explicit.
- Candidate floor ratio.01 is the unchanged local hypothesis, already known to
  lose mixture influence at high d. It stays fixed to isolate fitting. Oracle
  and current-observation heuristics use peak-1000 as the Gaussian limit. This
  creates a diagnostic ceiling, not equal-floor algorithm ranking. Bootstrap
  uses the shared constant-twist route. No claim that this floor is author-chosen.
- Full empirical whitening changes the regression metric and is justified by
  the symmetric least-squares derivation above. Rank/SPD guard is dimension and
  scale aware (64*eps*max(N,d) times largest eigenvalue); guard only rejects.
  Invalid branches may use finite placeholders solely to record rejection;
  they never supply an accepted guide. No silent numerical regularization.
- Heuristic set is constructed from this problem: bootstrap needs no guide,
  current-observation Gaussian uses the exact one-step likelihood, full backward
  oracle uses all future information. Evaluate separately at each dimension.
  Any observed conditional mean loss to a heuristic is a promotion veto/headline.
  Equal-particle comparisons do not establish equal compute efficiency.

## Execution and uncertainty
Harness docs/benchmarks/diagnose_iapf_score_regression.py; modes gpu_preflight,
cpu_reference, gpu_downstream, cpu_results, with unique --attempt values.
Use /home/chakwong/anaconda3/envs/tftwogpu/bin/python; GPU4080 UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, escalated FP64 XLA, TF32off, growth
verified before initialization. CPU R reference deliberately hides GPU.
Preserve sources, commands, Git, seeds, inputs, failed cases, environment, timing
and result. No new package/environment changes. Read only frozen R references.
No stochastic ranking will be claimed from four data sets. Report all per-seed
MSEs, conditional means, and descriptive paired differences. Avoid inferential
intervals whose four-cluster precision could look conclusive. Any ranking needs
fresh multi-data replication with predeclared uncertainty and multiplicity.

Pre-mortem: exact Gaussian recovery can hide non-Gaussian floor approximation;
low local residual can hide bad global geometry; one pass can hide adaptive
instability; different fit costs can hide inefficiency; oracle Gaussian limit
can expose combined floor/family losses rather than the fit alone. Fresh actual
likelihoods, rejection records, paired baseline ladder and explicit scope prevent
those errors from being interpreted as success.
Skeptical audit: no material unresolved baseline or arithmetic ambiguity remains.
The phase does not select a default or retune on these data. If the candidate
fails, separate objective failure, SPD/rank failure, and downstream/floor limits
before planning the next experiment; do not repeat the failed objective blindly.
