# SGQF and TT fitting: clarification from saved evidence

Date: 2026-09-14. This note answers the owner's question about reliable fitting
and the SGQF comparison. It inspects existing code and results only; no new
experiment, method change or promotion criterion is introduced. It supplements
phase 7 of the [master program](observation-aware-tt-repair-complete-program-20260913.md).

## What SGQF supplies

The current driver has a standalone `sgqf_gaussian` particle-proposal arm:
`docs/benchmarks/run_observation_aware_tt_complete.py:115–157`. It draws from the
SGQF filtering Gaussian at each time, evaluates that Gaussian density, and
applies the original model's importance correction. This is an importance-
corrected particle filter using SGQF proposals, not the raw Gaussian filter's
reported moments. The proposal does not condition on the individual ancestor.
The pair arm also uses this Gaussian directly at t=0.

For the TT arm, SGQF supplies current/previous marginal coordinate charts and,
after t=0, a correlated Gaussian component of the training-row distribution.
The latter uses a Gaussian backward conditional with explicit importance
weights. These uses do not make the fitted TT equal to the SGQF proposal.
The call chain is `build_pair_tt_path` -> `pair_fit_from_log_target` ->
`pairtt.fit_pair_features` -> `compiled_pair_fitter`; see
`bayesfilter/highdim/observation_guided_tt_tf.py:32–204` and
`bayesfilter/highdim/pair_block_tt_tf.py:145–256`.

The three inspected implementation files' SHA-256 values still match the
completed campaign's [manifest](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/run_manifest.json).
The recorded campaign/replay supplies execution evidence for these consumers;
this inspection adds no new executable parity check.

## How the regression uses the available joint Gaussian guide

Owner follow-up, 2026-09-15: the earlier explanation about two marginals was
incomplete. In this model the known linear-Gaussian transition and the incoming
SGQF Gaussian determine a backward conditional. Combining that conditional with
the updated SGQF Gaussian gives a full approximate two-time joint, including
cross-time covariance. `joint_sgqf_row_sampler` already constructs this joint.
SGQF does not need to fail for the subsequent TT regression to introduce error.

The implemented SGQF update retains a Gaussian approximation from quadrature
means/covariances, not the unrestricted non-Gaussian filtering density. Write
its incoming Gaussian as r(z), transition as f(x|z), and predictive Gaussian
as a(x) = integral r(z) f(x|z) dz. The backward conditional is

    B(z|x) = r(z) f(x|z) / a(x).

For r = N(m,P), f = N(Az,Q), the code uses S = A P A' + Q,
K = P A' S^{-1}, and B = N(m + K(x-Am), P-K S K'). It combines B with
the current SGQF filtering Gaussian q(x), giving q_joint(z,x) = q(x) B(z|x).
The current observation depends only on x, so it does not further change B.

This makes the owner's point precise. Under the same incoming Gaussian r, the
exact one-step updated marginal is p(x) = g(x) a(x) / Z and its exact joint
is p(x) B(z|x). Consequently,

    integral |p(x) B(z|x) - q(x) B(z|x)| dz dx
      = integral |p(x)-q(x)| [integral B(z|x) dz] dx
      = integral |p(x)-q(x)| dx.

Thus replacing this exact marginal by q does not add a separate joint error
in total variation. Exact marginal agreement would imply exact joint agreement
under that incoming-law assumption. This algebra is a local derivation, not a
claim that the measured SGQF marginal or the recursive TT incoming law is exact.

The implementation distinction is that the correlated joint supplies the
guided component of regression rows, while the TT reference measure remains
the product of separate marginal Gaussians. The importance weights return the
sampled objective to that product reference. TT therefore still represents the
Gaussian cross-time dependence in its fitted coefficients; it does not merely
fit a residual relative to q_joint. Its finite representation and unfinished
optimization can introduce error even when q_joint is accurate. This is a
possible mechanism for loss of accuracy, not yet an isolated causal finding.

At t>0 the regression target uses the joint density proportional to
`g_t(x) f(x|z) pi_hat_TT_previous(z)`, where the incoming density is the retained
TT approximation. It fits its square-root ratio against the product of the
two SGQF marginal Gaussians, in their separate standardized coordinates u,v.
The previous joint-target audit is
[recorded here](observation-tt-first-transition-root-cause-20260914-result.md).
Accurate marginal means/covariances do not eliminate cross-time dependence:
even two exactly Gaussian marginals can have a strongly correlated joint.
This describes the TT product reference; it does not mean SGQF lacks an
available joint approximation or establish that its approximation is poor.
Pairing addresses representation of that dependence; it does not solve the
coefficient optimization or certify the recursive incoming approximation.

For fitted amplitude h, define c(v) as the standard-Gaussian integral of
h(u,v)^2 over u. The actual conditional proposal is

    q_TT(x|z) = q_SGQF(x) * (h(u,v)^2 + tau) / (c(v) + tau).

This follows directly from `sample_pair_conditional`'s log density at
`bayesfilter/highdim/pair_block_tt_tf.py:104–125`, followed by the current affine
chart. A correction constant in u recovers q_SGQF algebraically. That special
case has not been newly tested through the endpoint in this inspection.
An inaccurate nonconstant correction can worsen the Gaussian proposal.
Positive tau ensures support; it does not guarantee SGQF-level accuracy.

Each fit currently starts from generic stateless random coefficients plus a
constant component (`initial=None` at the consumer), not a fitted SGQF joint
warm start. It executes four alternating sweeps with 128 proximal-gradient
steps per core. These are fixed work limits, not convergence stopping rules.
The recorded subproblems can be badly conditioned. In diagnostic-02 the chosen
d4 pair fit's maximum capped Gram condition was about 3.78e7 and KKT residual
0.010057; doubling the work did not establish convergence. This is evidence
of an unresolved optimization problem, not proof that conditioning alone is
its cause. Initialization, capacity, row coverage and the recursive target
remain competing explanations requiring controlled checks.

## Scope of the observed fitting problem

The common Gaussian first-transition audit improved from 0.442461 for grouped
scalar TT to 0.116883 for the chosen pair fit. Initialization and coefficient
count confound attribution to pairing alone. Preserve this favorable observed
result; a downstream promotion veto does not erase it.

Reading `campaign-02/d{1,4}/tt_pair_block_fit_t*.json` gives the following saved
relative amplitude errors over t=1–19 (zero-based time):

| Scope | Median audit error | Maximum audit error | Interpretation |
| --- | ---: | ---: | --- |
| d1 | 0.032652 | 0.109675 | Generally small errors in this run; the d1 t=0 and t=1 stationarity residuals are around machine precision. Do not describe every scope as optimizer failure. |
| d4 | 0.143162 | 1.307943 | Some late recursive fits fail badly; the first-transition improvement is insufficient to establish reliable fitting over the sequence. |

At [d4 t=18](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/d4/tt_pair_block_fit_t18.json),
training/validation/audit errors are 0.374702/1.361372/1.307943 and the KKT
residual is 0.101954. These are square-root-amplitude diagnostics, not filtering
mean errors. They demonstrate a poor fitted realization and a large split gap;
they do not identify the causal share of optimization, capacity or row coverage.
Earlier retained-fit errors enter later fitting targets, which may propagate
distortion; causal accumulation has not been isolated.

## Is the resulting filter worse than the SGQF proposal?

The saved conditional filtering-mean RMSE comparison is mixed:

| Dimension | Observation regime | SGQF proposal | Pair TT proposal |
| --- | --- | ---: | ---: |
| 1 | Near zero | 0.079116 | 0.016192 |
| 1 | Ordinary | 0.050992 | 0.024908 |
| 1 | Large | 0.018675 | 0.019715 |
| 4 | Near zero | 0.045774 | 0.027447 |
| 4 | Ordinary | 0.044386 | 0.034611 |
| 4 | Large | 0.047386 | 0.048544 |

Source: [campaign result](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/result.json),
`dimensions[d].methods[m].conditional_mean_errors`. Four particle seeds share
one observation sequence; d4 near-zero has one time point. No statistical
ranking is supported. All six methods pass the declared mean/evidence screens.
The large-observation comparisons retain the existing heuristic promotion veto.
The data do not support a blanket assertion that pair TT is worse than SGQF.

A like-for-like SGQF-only joint-amplitude error is not supplied by this
diagnostic's comparator set. Its TT audit errors cannot be compared directly
with SGQF filtering-mean error. The current evidence therefore does not decide
whether the fitted joint density is closer to its target than an explicitly
defined SGQF-only joint approximation.

## Consequence for phase 7

The already required separation of optimization, capacity and recursive targets
should explicitly identify the Gaussian comparison at each level: marginal
proposal, Gaussian joint approximation, and TT correction. Define a common
target/metric before comparing density fits; include an executable check of
the Gaussian-recovery case. Examine the failed d4 recursive targets while
preserving d1 as a known-good diagnostic. These are protocol design questions
under A03, not permission to change the baseline, method or thresholds during
execution. Any proposed initialization/solver/proposal change must be stated
in the phase-7 protocol and reviewed before numerical execution. Heuristic
comparisons remain evaluation vetoes, not training or tuning targets.

Compare the already implemented correlated SGQF joint with the TT approximation
on the same target; separately distinguish Gaussian incoming-law error from
error against a retained-TT incoming target. Do not explain a TT failure as
SGQF failure without that evidence. Replacing the TT product reference by a
coupled joint chart would change the representation and its marginal/conditional
consumers; coupled charts are outside A03's continuation scope. Such a change
would need an explicit amendment and review, rather than being silently folded
into this clarification. No such change has been made here.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue the planned fitting investigation; retain observed progress | Large reduction versus original TT on the common first-transition audit; recursive d4 reliability unresolved | Existing filtering promotion veto remains; this inspection found no new continuation veto | Optimizer versus capacity/coverage/recursive effects; no matched Gaussian joint-error comparison | Specify these distinctions in phase 7 and review the executable protocol | Blanket TT inferiority, converged d4 fitting, statistical superiority, or a changed promotion rule |

| Inference category | Status |
| --- | --- |
| Hard veto screen | Corrected sampler passed its tested validity checks; prior heuristic promotion veto unchanged. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Fitting errors and all conditional RMSE comparisons above. |
| Default-readiness | Not established. |
| Next evidence needed | Controlled fitting diagnostics on frozen targets plus untouched independent-sequence evaluation under the reviewed protocol. |
