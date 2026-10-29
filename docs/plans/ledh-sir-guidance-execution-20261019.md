# SIR backward-guidance comparison — 19 October 2026

Completed: [result](artifacts/ledh-sir-guidance-20261019/result.md). All 96
keys evaluated, 95 valid, one independently reproduced global reset failure.
Four primary paired intervals show conditional harm. Next execute the
[independent-beta repair](ledh-sir-beta-repair-20261019.md); no OT retuning.

## Authorized continuation

### Localized global-comparator failure

Attempt 02 contains 59 accepted rows and one failed global row, not 60 valid
rows: data 26100832, seed 261019405, beta (.1,.9,0). At observation 4, the
relative reset covariance error is .988952 and the minimum factorization
margin is 6.24946e-17. The guard correctly rejected it. The runner preserved
the failure but continued to the next row; this violated its stated immediate
stop rule and is repaired before resumption. Attempt 03 detected the invalid
retained row and stopped, spending 79.886878 seconds.

First allocate 180 GPU-process seconds to exact-seed reproduction with the
unchanged consumer and settings (`scripts/diagnose_sir_global_reset.py`).
Pass means the same failed step and trace within 1e-9, not numerical success.
No tolerance, ridge, OT, guide or model change is authorized by this check.
If reproduced and isolated to this arm/key, preserve it as failed evidence
and continue independent evaluations in attempt 04 (3,600 seconds), explicitly
quarantining the failed key. No valid score or RMSE may be imputed for it.
Any new invalid key stops that attempt for the same bounded investigation.
All original keys remain accounted for; full-design numerical success and
promotion remain false if any row fails. Primary paired contrasts on other
arms can be reported only with their own eight valid pairs. This is a
documented localized repair of execution scope, not removal of a scientific
veto or a claim that the failing comparator is successful.

The owner has authorized twelve hours for this campaign. The total ceiling is
43,200 GPU-process seconds, including preflight and all earlier attempts; this
supersedes the original 3,000-second ceiling and the pending 5,400-second request.
The scientific question, OT(1,5), six arms, two data sets, eight paired seeds,
reference and decision rules are unchanged. Attempt 02 retains its existing
2,700-second process cap. A fresh attempt 03 will import its checksum-verified
completed rows and frozen guides and evaluate only missing keys, with a
3,600-second cap. Original files are preserved. Repeated smoke checks verify
the continuation; imported rows retain their original provenance and timing.

Skeptical continuation audit: additional time does not authorize selection on
the comparison scores, a reduced seed design, a different baseline or a claim
of canonical LEDH improvement. Continuation must reject duplicate/unexpected
keys, different data or guides, changed numerical implementation, and invalid
rows. A complete result triggers the master's repair-and-refresh pass; any
subsequent experiment needs its own bounded allocation and evidence contract
within this ceiling, not renewed procedural approval.

### Localized diagnostic repair

Attempt 02 passed all 18 fine-step derivative and nine Richardson checks.
The maximum errors are 1.0189e-5 and 1.8003e-9 respectively. The second-order
coarse-step pattern was truncation, not evidence of a missing derivative.
Post-run reporting now independently checks the exact 96 paired keys, frozen
guide checksums, fixed mixture/OT settings, finite scores/values/traces and
complete derivative evidence. Six synthetic reporting regressions passed.

Full T50 evaluation costs about 41 seconds per row after compilation. This
exceeds the planning estimate. The owner has now authorized twelve hours;
the authorization and per-attempt allocation above supersede the earlier
3,000-second ceiling. Retain all methods and eight paired seeds;
an incomplete run cannot satisfy the predeclared primary criterion.

Attempt 01 fitted both guides but stopped before evaluation because the
quadratic-guided finite difference at steps 1e-4 and 5e-5 differed from the
analytical score by relative 1.6303e-4 and 4.0755e-5. The error decreases
fourfold, consistent with central-difference truncation, not evidence yet of
a derivative implementation defect. Keep these checks; add steps 2.5e-5 and
1.25e-5 and require both to meet the unchanged 2e-5 threshold. Also require
their Richardson extrapolation to meet 2e-6. This repairs the reference
diagnostic, not the evaluated method or a score-selection criterion.
Attempt 02 reuses the exact checksum-verified frozen guide arrays and fitting
seeds; no refitting or score-driven selection. Attempt 01 consumed 178.5001
seconds. Reserve 50 seconds conservatively for preflight; attempt 02 is
capped at 2,700 seconds. The 3,000-second campaign bound remains unchanged.

Implementation freeze: 512 training and 256 disjoint validation locations per
message, 32 centers from the training design, 200 positive multiplicative NNLS
updates. Select bandwidth by message validation MSE only; activate a kernel
message only when it improves on the constant training-mean message on that
validation set. This may leave some times unguided and must be reported.
Evaluation seeds are 261019301–308 and 261019401–408; independent fit seeds are
261019101 and 261019102. All score validation uses frozen guide coefficients.

Focused CPU checks: 11 passed in 16.83 seconds, including the integration,
indexing and projection checks and existing full-consumer derivative checks.
GPU SIR checks additionally require three arms, three parameter coordinates
and two finite-difference step sizes before the comparison starts.
The final combined fitter/consumer/reporting regression passed all 17 tests
in 17.99 seconds. Including reporting-test attempts, CPU test time is
35.44 seconds; logs retain the repaired test-fixture syntax failure.

Reference source audit: Zhao–Cui paper Eq. (26)/Algorithm 4 and author
`source/models/full_sol.m:139`–`203` were inspected. The author code samples
backward paths and computes target/proposal log weights, but its displayed
`lml=mean(w_temp)` is not a log-mean-exp likelihood estimator. The saved local
reference uses the corrected log-mean-exp calculation and a fitted parameter
score. It is an approximate numerical reference, not an exact oracle or an
unaltered original-author likelihood-score implementation.

## Question and corrected scope

Does frozen backward information from the replicated controlled-SMC fitter,
represented as a quadratic or positive Gaussian-kernel mixture, reduce the
SIR likelihood-score error? OT is fixed at epsilon 1 and five Sinkhorn steps
in every arm. The October 18 iteration experiment answered the wrong question
and supplies neither a baseline nor tuning evidence here. The owner's request
to refresh and execute authorizes this corrected local campaign.

Checkout: `ledh-refactor-with-policy-fix`, worktree
`ledh-iapf-execution-20260922`, base `ea1018de1`. Preserve unrelated changes.
The original score master remains authoritative. This is its SIR transfer
stage; the closed source-replication and KSC campaigns are not resumed.

## Call chain and scientific limits

Use `MultivariateGaussianControlledSMC.forward/fit/regress` for the actual
offline backward fit, then `FrozenQuadraticBackwardGuide` or
`FrozenMixtureBackwardGuide` through `covariance_proposal_tf.make_filter`.
The latter is the existing three-density identity/global/local consumer in
the master program. Preserve its exact mixture-density weight, common reset,
and analytical recursive score. Test that guide coefficients reach this call
chain, terminal guidance is zero and finite differences agree with the total
analytical derivative while coefficients stay frozen.

This consumer is an optional finite-program candidate, not the canonical
UKF/dual-cap/Contract-E admitted route. Do not issue canonical/default/HMC
claims. A result can answer whether this particular proposal augmentation
helps, not whether the canonical LEDH algorithm has been improved. A later
canonical transfer still requires conformance and scope-specific tuning.

The replicated KDM neural learner has no SIR training adapter. The new
positive-kernel representation is the manuscript's local KDM-inspired message
approximation, not a source-faithful run of that learner. Do not conflate them.

## Research intent and evidence contract

Target: `canonical_sir_d18_x0_then_transition_observe_v1`, theta zero, T50,
18 state coordinates, nine observations, three score coordinates. Use the
two saved October data sets 26100831 and 26100832 and their separately
computed Zhao–Cui rank-20 numerical scores. These data have been inspected:
the comparison is diagnostic, not a new untouched confirmation.

Primary comparison: paired per-run squared Euclidean score error, guided
minus unguided, separately for each data set. Eight fresh particle seeds per
data set; all arms use identical initial/noise/uniform draws within a pair.
Report component means, MCSE and RMSE, paired intervals and reference
uncertainty. Four primary contrasts (two guide families by two data sets)
use 98.75% paired bootstrap intervals, 20,000 resamples, seed 26101999.
Only a complete eight-pair design supports these conditional intervals.
Negative upper endpoints in both data sets are required to support a
conditional gain for that family. This does not generalize across fits or
data sets; no default promotion follows.

Constructed heuristic adversaries, all at OT(1,5): physical prior proposal
(no observation map), defensive global (.1,.9,0; shared covariance), defensive
local (.1,0,.9; particle-specific covariance). Plain and guided main arms use
equal density probabilities (1/3,1/3,1/3), a neutral fixed hypothesis rather
than the inherited beta tuned in another scope. Evaluate every adversary
separately on both data sets and every parameter. Any observed mean-squared
error loss to an adversary vetoes promotion, even if the primary contrast is
favorable. Heuristic differences without intervals remain descriptive.

Continuation vetoes: corrupted data/reference, wrong target/time indexing,
failed shared-consumer wiring, nonfinite or invalid evaluated result, failed
analytical/finite-difference parity, or exhausted total budget. An ineligible
fit rejects that candidate; it does not invalidate the plain baseline or the
other representation. Expected poor approximation or heuristic loss is not
a reason to stop another valid arm. Localized infrastructure failures may be
repaired within the remaining budget with new attempt directories.

Explanatory diagnostics: pilot ESS and cSMC precision margins, fit conditioning,
heldout message error, PSD projection size, kernel weight concentration,
likelihood, filter ESS/weights, moment and OT residuals, runtime. They do not
substitute for score accuracy. Roundoff near 1e-13 is not a veto.

## Fitting design and derivation

Include the actual Gaussian x0 as time zero with observation potential one,
then perform all 50 transition-observe steps. A Gaussian approximation to x1
must not replace the nonlinear pushforward of x0. Run one controlled-SMC
backward update from a zero policy with 512 particles and centered QR, then
an independent forward pilot under that fitted policy. Keep the source
precision validity check; reject invalid policies without silently clipping.

For the policy psi at the next time, compute the future-only message exactly
under the frozen pilot transition: h(x) = integral N(z; f_star(x), Q_star)
exp(-z' A z - b' z - c) dz. Let S=(Q_star^-1+2A)^-1 and
v=Q_star^-1 f_star(x)-b. Then log h(x) equals
(-log|Q_star|-log|Q_star^-1+2A|)/2 - c
+ v'Sv/2 - f_star(x)'Q_star^-1 f_star(x)/2.
This is Gaussian completion of the square, implemented by the replicated
`multivariate_twist`. At the last observation h=1. No current observation
factor is exported as future guidance; no evaluation ancestor tracing is used.

At each nonterminal time, use an independent Gaussian design centered on the
weighted fitted pilot cloud. Training and validation draws are disjoint.
Fit -log h by the replicated centered quadratic regressor. If its information
is indefinite, the declared consumer approximation projects it to positive
information while preserving the fitted gradient at the design mean. Record
the projection and heldout error. This is a proposal approximation, not an
exact identity for nonlinear SIR. A roundoff-sized positive eigenvalue floor
is explicit and recorded. No model density is modified.

For the kernel representation, fit positive RBF coefficients to the same
future message on independent training points. Use 32 frozen centers and
full design covariance times bandwidth squared, candidates 0.5, 1, 2.
Nonnegative least squares uses bounded multiplicative updates; choose the
bandwidth only by validation relative message error, never by score/reference
agreement. Normalize the coefficients into mixture logits; the discarded
common normalization does not affect spatial guidance. Record heldout error
against a constant-message baseline and coefficient concentration. This
target-specific protocol is an initial bounded diagnostic, not trained KDM
quality or population-optimal fitting.

## Defaults, pre-mortem and skeptical audit

N256 is a bounded pilot scope (K=N exactly), not a promoted particle count;
it permits eight paired repetitions with all simple adversaries. All main
arms use the same N and the full T50 horizon. FP64 GPU/XLA with TF32 off is
an explicit reference exception to the production FP32 direction. Memory
growth is mandatory; pin the known GPU and disable CUDA command buffers
after the recorded October 18 native failures, retaining XLA.

Flow16 and zero optional moment correction are inherited controls of this
particular mixture consumer, held fixed to isolate guidance. Their adequacy
is unproved; complete finite derivatives and moment diagnostics are checked.
They are not a claim to implement canonical dual-cap correction. Equal beta,
one fit pass, 512 fit particles, 32 kernels and the bandwidth grid are tested
hypotheses. High-dimensional coverage, indefinite message curvature, a noisy
pilot or a poor kernel basis can fail independently of the guidance idea.
Fit validity/heldout diagnostics precede the evaluation ladder and preserve
these alternative explanations. No hyperparameter is selected on test scores.

Skeptical preflight PASS for this diagnostic scope: the actual enhancement is
varied, OT is fixed, both data sets have matched fresh baselines, the source
reference is approximate, and the consumer/learner limitations are explicit.
The old plan failed this audit by substituting iteration count for guidance.
Provisional mathematical correctness concerns the proposal density correction
and derivative of its finite likelihood estimate; equality to the model's
true score is empirical and is not asserted.

## Execution, budget and artifacts

Implement the SIR adapter and executable wiring/indexing/derivative checks;
run focused CPU-only tests, then a GPU/XLA fit/consumer smoke, then the full
paired comparison. No external messaging, package changes or git publication.
Total GPU-process ceiling 43,200 seconds under the owner's twelve-hour
authorization. Initial comparison allocation: attempts 01 and 02 as recorded,
then 3,600 seconds for continuation attempt 03; localized retries remain within
the total ceiling. CPU checks/analysis allocation 300 seconds. Stop each worker with a supervisor
deadline. The closed 18 October spending remains separate and is not erased.

Environment `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. Commands:
`python -m pytest -q tests/highdim/test_sir_backward_guidance.py` (explicit CPU),
then `python scripts/run_ledh_sir_guidance.py --output <fresh-attempt> --seconds
<remaining-budget>` with trusted GPU access, memory growth and recorded XLA
settings. Record exact command, git/source hashes, data/reference hashes,
seeds, controls, fit artifacts, per-row scores/diagnostics and time spent.
Root: `docs/plans/artifacts/ledh-sir-guidance-20261019/`.

After execution, write decision/inference-status tables and strongest
alternative explanations, update this plan, both masters and the active
checkpoint, and report the numerical results. Preserve failed candidates and
all prior attempts. A partial or failed run must not be called improvement.
