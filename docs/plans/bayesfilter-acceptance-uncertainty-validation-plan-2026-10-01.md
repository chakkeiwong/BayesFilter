# Acceptance uncertainty and temporal-screen validation plan

Date: 2026-10-01

Current checkpoint: all six development phases are complete. The tested
interval rule failed statistical promotion; public tuning admission remains
unchanged. The [execution result](bayesfilter-acceptance-uncertainty-validation-result-2026-10-01.md)
records the final calibration, tests, and next repair. The executable
specification below supersedes the initial early/late contrast sketch.

## Research question

Can BayesFilter distinguish finite Monte Carlo variation in a fixed HMC
kernel's mean Metropolis probability from a practically material temporal or
chain-level discrepancy, while returning an explicit inconclusive result when
the trace contains too little effective information?

The target estimand is the mean Metropolis probability

\[
 a_{it}=\min(1,\exp(\ell_{it})),\qquad
 \mu= m^{-1}\sum_{i=1}^m E(a_{it}),
\]

where each chain is post-warmup, independently seeded, and run with one fixed
kernel. Binary accept/reject frequency remains a separate explanatory
quantity. The procedure does not estimate posterior convergence and does not
use R-hat as a tuning gate.

## Baseline and proposed mechanism

The baseline is the current `HMCAcceptancePolicy` v5 raw block crossing
screen, with the already present v6 paired-contrast route retained as an
experimental historical comparison. The proposed mechanism is an independent,
TensorFlow implementation of chain-preserving long-run variance and MCSE for
the acceptance-probability trace, plus predeclared temporal contrasts. It will
report batch sizes, complete-batch counts, long-run variance, MCSE, effective
information, contrast covariance, and an explicit insufficient-information
status. No default tuning policy changes in this plan.

## Evidence contract

* **Primary question:** does the uncertainty calculation have calibrated
  coverage and useful detection behavior for the declared stationary,
  dependent, drifting, and short-trace classes?
* **Comparator:** the current raw block screen and the independent analytic
  reference calculations for synthetic traces.
* **Promotion criterion for this development phase:** the implementation agrees
  with independent formulas on valid fixtures, fails closed on invalid or
  under-batched traces, and produces a versioned calibration report. Passing
  this phase does not promote a new production default.
* **Vetoes:** nonfinite input, invalid shapes, fewer than the required complete
  batches, nonpositive/nonfinite long-run variance, missing chain boundaries,
  and failed health telemetry remain hard evidence failures.
* **Explanatory diagnostics:** R-hat, binary acceptance rate, raw block ranges,
  and estimator sensitivity are reported but do not admit or reject a kernel by
  themselves.
* **Nonclaims:** the phase does not establish posterior convergence, burn-in
  sufficiency, stationary coverage for arbitrary HMC targets, sampler
  superiority, or the correctness of the BGS candidate.
* **Artifacts:** this plan, the new module and tests, a calibration result
  under `docs/plans/artifacts/acceptance-uncertainty-validation-20261001/`,
  and a result note recording limitations and the migration decision.

## Work phases

1. **Repository audit.** Confirm the current v5/v6 policy semantics, identify
   reusable MCSE arithmetic, and record dirty-worktree boundaries. Preserve all
   unrelated changes.
2. **Estimator implementation.** Add a standalone TensorFlow/TFP acceptance
   uncertainty module. It will preserve chain axes, use complete contiguous
   batches, expose ordinary and lugsail estimates, and return unavailable
   evidence instead of clipping invalid estimates. Preserve multivariate
   covariance between quantities in the shared primitive; construct all
   within-chain window contrasts from simultaneous marginal intervals as
   specified below. Keep this module opt-in and separate from the v5 default
   decision path.
3. **Reference and property tests.** Use an independent test-only reference
   implementation for IID and AR(1)-style traces. Test permutation,
   serialization, boundary, short-trace, constant-trace, near-boundary, and
   numerical-invalidity behavior.
4. **Behavioral calibration.** Run bounded deterministic replications for
   stationary IID, strongly autocorrelated stationary, genuine drift,
   one-chain heterogeneity, block-boundary changes, oscillatory traces, and
   invalid/stuck cases. Report empirical coverage, false conflict, missed
   material discrepancy, and inconclusive rates with replication uncertainty.
5. **BGS replay and integration checks.** Replay saved summaries/traces as
   explanatory regression cases without changing their historical decisions.
   Verify that R-hat fields cannot alter the acceptance result and that fresh
   verification remains a separate requirement.
6. **Guide and migration note.** Document the new estimator as experimental,
   state its assumptions, and explicitly leave v5/v6 artifacts under their
   original semantics. A later plan may promote a new policy only after
   calibration review.

## Skeptical plan audit before execution

The initial proposal had four material risks. First, reusing the four-block
Student-t calculation would still treat too few dependent summaries as
independent; the repair is to compute uncertainty from the transition-level
acceptance trace. Second, a finite MCSE cannot detect initialization bias or
adaptation drift; the repair is to require a fixed-kernel post-warmup input and
keep readiness/health checks separate. Third, a lugsail estimate is not
automatically calibrated for very persistent or short chains; the repair is to
return an explicit unavailable status and validate persistence and batch-size
sensitivity. Fourth, replaying the BGS measurements alone cannot calibrate a
general rule; the repair is a synthetic replication campaign with independent
reference arithmetic. These constraints pass the audit, so implementation may
begin. No new default, threshold, or historical decision is authorized by this
plan.

## Assumption and number provenance

The lugsail form and square-root batch-size convention are literature baselines
already used by `HMCPrecisionPolicy`; they are implementation hypotheses here,
not coverage guarantees. Minimum complete-batch counts are operational
insufficiency floors and remain unvalidated as evidence of dependence
resolution. Effective-information estimates are reported without an asserted
sufficiency threshold. The practical temporal
discrepancy is supplied by the caller in this experimental layer; it is not
silently identified with the acceptance-band width.

## Stop conditions

Stop the implementation phase on a TensorFlow shape/finite-value failure,
inability to preserve chain boundaries, or a test showing that an invalid or
under-batched trace can be promoted. Stop the calibration phase if the
artifact is incomplete, the independent reference cannot be reproduced, or
the estimator's coverage and false-decision behavior cannot be interpreted
under the declared simulation class. A failed synthetic case is evidence for
repair, not evidence against HMC tuning as a whole.

## Planned review record

Review this file against the current implementation before running any
calibration command. The review must confirm that the proposed module is
opt-in, does not silently reinterpret v5/v6 artifacts, does not use R-hat as a
tuning gate, and that every stochastic result has seeds, a manifest, and a
bounded replication count.

## Executable specification and second skeptical review

The review identified two errors in the earlier discussion. Seed independence
does not make four non-identically initialized chain means a calibrated t
sample. A fixed kernel alone also does not justify a stationary estimand:
finite-start expectations can differ from stationary expectations. The new
report must state local stationarity/mixing as assumptions, never facts proved
by its screen. Opposing drifts can cancel in a pooled contrast; all temporal
contrasts must therefore be evaluated **within each chain**.

The first experimental implementation uses four contiguous windows and all six
contrasts within each chain. It constructs simultaneous *marginal* intervals
for window means and whole-chain means, then subtracts interval endpoints.
If all marginal intervals cover, every such difference interval covers,
irrespective of covariance between adjacent windows or comparisons. This
conservative construction avoids estimating adjacent-window independence or
pretending that four blocks are independent replications. It reports covariance
between quantities in the shared batch-means primitive; it does **not** pretend
to estimate cross-window covariance. Bonferroni allocation bounds the family
error only **if the marginal intervals are valid**. The approximate t reference
with batch-count-minus-one degrees of freedom is a development hypothesis,
especially for lugsail, not a finite-sample theorem.

Three contrast outcomes are required: interval wholly beyond a supplied
tolerance (supported material difference), wholly within the tolerance
(supported compatibility at the checked windows), or unresolved. Failure to
reject a difference cannot grant compatibility. A constant observed trace is
insufficient to infer zero process variance; analytically deterministic
acceptance requires separate evidence, outside this estimator.

The experiment compares plain batch means and lugsail with the raw v5 and
paired v6 screens, and an IID-MCSE comparator that deliberately ignores
dependence. A known-covariance oracle supplies finite-count mean variance for
bounded two-state Markov traces: a_t = mu + d S_t, with stationary symmetric
S_t in {-1,+1} and correlation rho^k. Then
Var(mean) = d^2[n + 2 sum_{k=1}^{n-1}(n-k)rho^k]/n^2.
This is derived by expanding the covariance of the sum. The oracle interval
is itself a normal approximation, not an exact coverage theorem.

### Fixed development allocation and criteria

* CPU reference/diagnostic execution only, with GPU deliberately hidden before
  TensorFlow import; XLA is on for the new numerical kernels. No learned map
  training or live BGS work. Maximum two hours wall time, four TensorFlow CPU
  threads (eight CPU-hours upper allocation), and three calibration attempts
  including repairs. Existing unrelated campaigns are untouched.
* Seed 20261001 is a reproducibility choice, not a favorable-seed selection.
  256 independent replications per cell give at most about 0.031 binomial
  standard error (derived worst case sqrt(.25/256)). Every rate has a Wilson
  95% interval. These intervals are descriptive pointwise intervals.
* Lengths 512 and 8192, persistence 0, .8, .95, .995 and -.8; additional near
  boundary, transient, opposing-drift, single-chain drift, oscillation and
  permanent-chain-offset fixtures. The lengths stress the BGS allocation and
  a longer alternative; neither is a recommended universal tuning count.
* Explicit experimental policy: base batch size floor(sqrt(window length)),
  sensitivity batch twice as long, minimum 8 batches, family alpha .10,
  temporal tolerance .10 and chain tolerance .10; r=3,c=.5 for lugsail.
  These are development hypotheses. Alpha matches the legacy working interval;
  the tolerances express a ten-percentage-point material change for this
  validation, not a newly justified universal tolerance. Eight batches is an
  intentionally small stress baseline. Per-look/candidate allocation is
  explicit; tests also cover multiple declared looks/candidates.
* Engineering pass: independent arithmetic and codec tests, all health veto
  mutations, unchanged historical admission, and actual fixed-kernel model
  trace integration must pass. No requirement that a difficult model qualify.
* Statistical default-promotion screen: for every stationary cell the lower
  Wilson bound for joint marginal coverage (unavailable counts as not
  delivered) must reach .90, upper bound for false conflicts must be <=.05;
  for long IID material-drift cells the lower bound for detection must reach
  .80; long IID in-band compatibility delivery must reach .80. These are
  predeclared development requirements, not proven error guarantees. A failure
  keeps the estimator experimental and is recorded; it does not relax the
  criterion or trigger a larger HMC campaign. Boundary-near and short/highly
  dependent cells must not be hidden by conditioning on available intervals.
* Fixed-kernel model integration: Gaussian, actual Kalman LGSSM, nonlinear
  sigma-point SSM, mixture, noncentered funnel, and a partially whitened funnel.
  Tiny CPU allocations check traces, shapes and health wiring, not posterior
  accuracy or default readiness. Existing controller tests check independent
  verification, seed reuse, exact-pair identity, and reporting-only R-hat.

Development completes with honest calibration results even if default
promotion fails. Promotion failure is not an execution failure: the intended
output is a tested experimental diagnostic and a recorded decision about its
limits. Public tuning admission is not wired to an unvalidated diagnostic.
The second skeptical audit passes on that explicit, bounded basis.

Commands: `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 python -m pytest ...` and
`python -m bayesfilter.testing.acceptance_uncertainty_validation --output <new-root>`
with the same environment. The result manifest records exact expanded commands,
source checksums, Git revision, versions, seeds and wall time. The local copy of
Vats--Flegal (arXiv:1809.04541), sections 2 and 4.1--4.3, equation (7), and
Appendix B was inspected for the estimator and its stationary/mixing assumptions;
the operational interval and decision rule are local experimental choices.

The scalar TFP t critical value is evaluated in a cached non-XLA graph at
configuration time; the covariance and trace decision graphs default to XLA.
This exception isolates inverse-beta special-function compilation from the
repeated trace kernels and carries no non-XLA sampler fallback.

The declared search stress uses 128 independent searches, eight candidates
per search and three looks (512, 2048, 8192), plus one fresh verification
bank. This is a convenience allocation within the same two-hour cap. It tests
bounded multiplicity accounting, false compatibility outside the wider band,
and fresh-stream separation; it does not establish optional-stopping guarantees.
The policy records the look/candidate allowance, while the existing controller
remains responsible for actual work and seed budgets. Overrunning a declared
allowance has no calibrated interpretation. The stress report includes that
limitation explicitly. No numerical policy will be fitted to these results.

## Phase refresh after run-01

Run-01 completed in 30.95 wall seconds. All eleven BGS replays matched, but the
default-promotion screen failed: at rho=.995 and n=8192 the lugsail screen
reported false conflicts in 37/256 replications; plain batch means in 63/256.
At n=512, the larger sensitivity batch leaves fewer than eight complete batches
per window, so every result was insufficient. At n=8192 lugsail also produced
nonpositive window variance estimates in some IID replications. These findings
invalidate default promotion of the tested interval rule, not the raw HMC
transitions or the long-run variance identity.

The next bounded diagnosis uses fresh seed coordinates 810 and 811, still 256
replications and the same rho=.995, n=8192 process. Compare explicit base batch
sizes 16, 64, and 128 with their doubled sizes; these span below and around the
derived correlation scale 1/(1-rho)=200. Also compare two overlapping but
contiguous windows displaced by 1024 draws from a longer stationary series.
That displacement is an eighth-window sensitivity choice, not an independent
replication. This diagnosis tests bandwidth sensitivity and identifies evidence
needed next. It cannot promote the most favorable batch or revise the original
criteria after observing them. Run-02 uses a new output root and retains run-01.
This diagnosis remains inside the two-hour wall allocation; no GPU work is needed.

Run-02 completed in 39.70 seconds. At the longest diagnosed base batch (128),
false conflicts fell but joint coverage remained inadequate. Neither estimator
is promoted. The final engineering review also requires explicit `fixed_kernel`
declaration, exact-duplicate-chain detection, and the name
`variance_estimate_available` rather than `information_available`: finite
variance arithmetic is not adequate effective information. Run-03 preserves
source snapshots and validates these final changes. The original search stress
with rho=.95 generated no naive false nominations, so it failed to exercise
that corner. Replace that stress by the already declared extreme rho=.995,
using fresh seed coordinates 710/711; add a viable IID bank at .70 with
720/721 to ensure selection/fresh verification is actually exercised. Do not
change statistical promotion criteria or choose a favorable batch from run-02.

## Final checkpoint and next phase

Run-03 is complete. Its [manifest](artifacts/acceptance-uncertainty-validation-20261001/run-03/manifest.json)
and [result](artifacts/acceptance-uncertainty-validation-20261001/run-03/result.json)
preserve executed source snapshots and the final calibration. All six
development phases have been executed, with 171 final checks passing and
eleven historical BGS cases reproduced. Three calibration attempts consumed
about 120 wall seconds; no GPU budget was used. The guide's tuning chapter
excerpt was rendered and inspected.

Statistical promotion fails: the tested lugsail rule reports 37/256 conflicts
under strongly persistent stationary traces and delivers joint coverage in
only 20/256 cases. Ordinary batch means and the larger-bandwidth diagnosis
also fail the declared criteria. The v6 comparison misses all opposing-drift
cases. The experimental report cannot issue a tuning artifact, and these
findings do not establish BGS candidate validity or posterior convergence.

The next task is to resolve the estimand and dependence-resolution strategy,
then specify fresh held-out calibration before admission integration. Possible
independent trial replication must explicitly target finite-start,
finite-horizon expectations; it cannot silently replace a stationary target.
The [result note](bayesfilter-acceptance-uncertainty-validation-result-2026-10-01.md)
and [master program](bayesfilter-hmc-repair-master-program-2026-09-16.md)
retain this statistical gap as open. No numerical run remains active for this
development phase.
