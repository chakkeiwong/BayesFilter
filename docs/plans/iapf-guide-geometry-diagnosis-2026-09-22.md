# Exact backward guides and fitted-guide geometry

Status: COMPLETE. Independent R checks and all 342 actual-consumer evaluations
passed. [Terminal findings](artifacts/iapf-guide-geometry-20260922-01/result.md)
separate fitted-guide error from diagonal-family restriction and initial Monte
Carlo integration. No fitted candidate or default was promoted.
Prior: `artifacts/iapf-positive-floor-20260922-01/result.md`.

## Scientific question and derivation

Floor reduction restores Gaussian proposals but leaves conditional heuristic
losses. Are the saved guides misplaced or poorly scaled, does the diagonal
family itself impose a large error, or does the consumer fail even with exact
guides? The first priority is fitted observations; heldout observations are an
explicit guide-transfer diagnosis and not a properly refitted method.

For the existing local linear model x_t=A x_(t-1)+noise(Q), y_t=H x_t+noise(R),
define psi*_t(x)=p(y_t:T|x_t=x). GJL Proposition 2 and equations 7–8 give this
optimal guide and the constant-weight property with an exactly twisted initial
law. Section 5.1 instead fits a diagonal Gaussian plus positive floor. The
local tractable paper copy and frozen `reference_iapf_paper.R::iapf_exact_twists`
(lines 458–473) are the mathematical/source anchors for this diagnostic.

Up to a multiplicative constant, psi*_t=N(x;c_t,V_t), where
V_t^-1=H'R^-1 H + A'(Q+V_(t+1))^-1 A and
V_t^-1 c_t=H'R^-1 y_t + A'(Q+V_(t+1))^-1 c_(t+1).
Omit future terms at T. The existing frozen R implementation computes this
recursion independently; use it unchanged. Map the local X0 convention to the
R X1 convention using m1=A m0 and P1=A P0 A'+Q for its Kalman check.

The local consumer samples X0 from its prior and estimates Fpsi_1(X0) before
resampling, rather than sampling an exactly twisted initial law. Therefore an
exact-guide run need not equal Kalman at finite N. For normalized guides, let
a_t=log g_t(c_t)+log Fpsi_(t+1)(c_t)-log N(c_t;c_t,V_t), omitting the future
term at T. Its predicted finite initial-sample value is
logmeanexp_i N(c_1;A X0_i,Q+V_1) + sum_t a_t.
The marginal value integrates X0 analytically, replacing the logmeanexp by
log N(c_1;A m0,A P0 A'+Q+V_1). Verify both identities. Confusing this local
initial integration error with a broken twist would use the wrong baseline.

## Evidence contract and research-intent ledger

Use the same 19 completed guides and saved actual random arrays as the preceding
phase, with all five adaptive failures still rejected. At d2/d5/d10, seeds81/82,
T4 and each realized count, compare these explicit diagnostic guides:

1. Saved fitted Gaussian (the actual baseline coefficients).
2. Exact backward mean and diagonal of its covariance. This minimizes
   KL[N(c*,V*) || N(c,V diagonal)], not the paper's Equation-15 fitting loss.
   It is an oracle family comparator, not a proposed fitted algorithm.
3. Full exact backward Gaussian (conditional oracle).

Cross each with the old .01 peak floor, the frozen R N^-2 tail rule, and a
negligible positive floor exp(-1000) times peak. The last is an explicit oracle
limit, not a deployable tail-protection setting. Keep it in log units. The
existing floor study already supplies the tail-rule provenance and uncertainty;
do not select a new rule on log-likelihood error.

Primary checks, predeclared:
- Independent R exact Kalman value versus the saved TF value <=1e-9.
- R exact-message identities evaluated at zero, center and +/-coordinate
  directions <=1e-9; covariance positive definite. Preserve means, covariance,
  constants and commands in versioned outputs.
- Actual TF full-guide/negligible-floor consumer value versus independent
  initial-sample prediction <=1e-9, Gaussian probabilities within 1e-12 of one,
  all post-initial ESS/N within 1e-9 of one. Initial ESS need not be N.
- Saved fitted guide with original floor exactly replays previous values/scores
  <=1e-9; the matching R-tail arm also matches previous floor replay <=1e-9.
- Source/input identities, finite outputs, CDF validity and single graph trace
  per scope. No alternate runtime filter is implemented.

Explanatory diagnostics: Gaussian KL (mean and covariance contributions),
best-diagonal KL floor, guide-conditioned proposal probabilities, ESS by time,
initial-integration error and conditional log-likelihood error. Large excess KL
identifies mismatch beyond the family limit but is not the optimizer's objective
or a promotion criterion. Saved likelihood/particle data do not establish a
stochastic ranking. The original floor still receives tail-protection rationale;
reducing it is a Class C diagnostic with no default adoption.

Constructed heuristic adversaries: saved matched-count BPF, constant-guide
consumer, one-step optimal proposal and exact Kalman. Compare conditionally on
dimension, observation regime, seed, fitting controls, count and time. A loss
remains a promotion veto, not a rejection of iAPF. Do not claim the full oracle
must dominate every stochastic draw: residual initial integration is explicit.

Continuation veto: incorrect oracle math, failed source/data identity, wrong
consumer replay, nonfinite accepted output, missing required evidence or budget
exhaustion. A geometry mismatch or heuristic loss triggers a fitting/initial-law
repair investigation, not a stop. Local harness repairs may retry under the
unchanged budget. No paper replication, default readiness, model-score, TF32,
canonical LEDH or HMC conclusions are allowed.

## Default audit and skeptical review

Every substantive new choice is an oracle/diagnostic: normalized backward
Gaussians have source/derivation support; the KL projection has a stated target
different from Equation 15; the negligible floor tests the constant-weight
identity, with its machine-realized probability checked. No parameter is selected
on likelihood. The old fitted sample/caps are preserved solely for mechanism
isolation. Survivor bias and intentional heldout transfer limit interpretation.
The exact oracle requires full-column-rank H; verify positive definite precision
in these square fixtures and fail if this assumption fails. Use FP64/XLA after
the verified floor normalizer checks; TF32 remains vetoed.

Skeptical review PASS: the constant-weight prediction accounts for X0 sampling;
the R model initialization matches local timing; whole-covariance and diagonal
comparators separate family restriction from fitted coefficients; the oracle
cannot be mistaken for a learned deployable method. Required evidence uses the
shared real consumer. No arbitrary optimizer/count expansion or default change.

## Execution and budget

Root: `artifacts/iapf-guide-geometry-20260922-01/`. Remaining campaign balance
about 45.853 CPU / 47.837 GPU hours, exact transfer in budget.json. Phase cap:
one CPU hour / fifteen GPU minutes; at most six launches, three local repairs;
300 seconds each. Commands use tftwogpu Python and
`docs/benchmarks/diagnose_iapf_guide_geometry.py --attempt <unique>
--device cpu|gpu --mode reference|consumer`. CPU reference hides GPU; GPU uses
UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b with escalation and verified memory
growth. Base R only, unchanged independent reference. Archive exact commands,
source/input snapshots, timings, conditional results, decision/inference tables,
terminal skeptical review and remaining budget. No production edits planned.
