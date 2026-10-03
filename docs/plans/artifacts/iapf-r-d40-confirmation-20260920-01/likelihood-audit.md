# Why fitting and stopping need not bias the final likelihood

2026-09-20. This audit concerns the executed independent R reference, not
canonical LEDH, KDM or an original-author implementation. Source anchors are
the local GJL2017 accepted manuscript, equations5--6, Proposition1,
Algorithms3--5, Section5.1 and the Proposition3 appendix. The PDF SHA256 is
41d751f9698c17550477563ed2bde3ca89ca9918d868ab496b44cb6f7304c975.
The text is `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt`.
Relevant text ranges: initial twisted definitions and Proposition1, lines
210--316; Algorithms3--4,654--688; fitting and adaptive resampling,909--1005;
Proposition3,506--546 and its appendix starting1918. The appendix studies
asymptotic variance; it does not certify the accuracy of32 realizations.

## The likelihood target and the quantity returned

For fixed observations y[1:T], the target is the original state-space model's
marginal likelihood Z, not its logarithm. Write its transition density as
f_t and observation density as g_t. For fixed positive guides psi_t, define

    B_0 = integral mu(x_1) psi_1(x_1) dx_1,
    B_t(x_t) = integral f_(t+1)(x_t,x_(t+1)) psi_(t+1)(x_(t+1)) dx_(t+1),
    B_T = 1.

The twisted initial and transition densities and their incremental weights are

    q_1(x_1) = mu(x_1) psi_1(x_1) / B_0,
    q_t(x_t|x_(t-1)) = f_t(x_(t-1),x_t) psi_t(x_t) / B_(t-1)(x_(t-1)),
    G_1(x_1) = B_0 g_1(x_1) B_1(x_1) / psi_1(x_1),
    G_t(x_t) = g_t(x_t) B_t(x_t) / psi_t(x_t), t>1.

Multiplication cancels each psi_t and the adjacent B_t:

    q_1 G_1 product_(t=2:T)(q_t G_t)
      = mu(x_1) product_(t=2:T) f_t(x_(t-1),x_t)
          product_(t=1:T) g_t(x_t).

Integrating gives the paper's Proposition1: the twisted normalizing constant
is exactly Z. A poor guide changes the sampling distribution and weight
variance; it does not change this identity when the proposal and all
corrections are evaluated consistently.

For the actual guide psi=N(m,V)+c, Gaussian multiplication gives
B=N(m;a,Q+V)+c. The proposal samples the Gaussian product component with
probability N(m;a,Q+V)/B and the original transition with probability c/B.
`reference_iapf_paper.R:49` implements the integral; `:58` constructs both
components; `:77` draws the mixture. `iapf_apf` at`:108` uses the same guide
in proposal and weight, adds B_0 at time1, and sets the terminal future factor
to1. This is the required mathematical quantity in exact arithmetic.

## Why adaptive resampling preserves the expectation

Let C be the product of completed resampling-block mean weights, and W_i the
current unnormalized weights. The unnormalized particle estimate of a test
function h is C times mean_i(W_i h(X_i)). At the next step, if resampling is
skipped, conditional expectation is

    C/N sum_i W_i q_t(G_t h)(X_i).

If resampling is selected, set C'=C mean(W), choose ancestors independently
with probabilities W_i/sum(W), and reset their previous weights to1. The
conditional expectation is again

    C' sum_i [W_i/sum(W)] q_t(G_t h)(X_i)
      = C/N sum_i W_i q_t(G_t h)(X_i).

The branch may depend on the current particle system, including its ESS:
both branches have the same conditional expectation. Induction from time1
therefore gives E[Z_hat | fixed psi,N]=Z for h=1 at timeT. The positive
floor and finite fixed Gaussian parameters keep the relevant finite-horizon
Gaussian-model integrals well defined. This derivation assumes exact proposal
draws, arithmetic and unbiased multinomial resampling.

The executed call chain is Python bounded driver -> captured
`replicate_iapf_paper_linear.R` -> `iapf_iterate` -> `iapf_apf` ->
`iapf_proposal`/`iapf_draw_proposal` and the matching integrals and weights.
The runner uses `value$final`, and the report computes exp(logZ_hat-logZ_Kalman).
At each resampling boundary the filter accumulates the prior mean weight and
resets previous weights; otherwise it retains them. Existing executable
conformance checks reconstruct actual weights for both branches and verify
path telescoping with nonideal guides and nonzero floors. They also check
proposal normalization against numerical integration.

## Why the final draw matters

Let H contain the entire fitting history, the stopping decision, final guides
and final particle count. Algorithm4, step3 performs a fresh APF after the
stopping rule fires. With independent subsequent random draws,

    E[Z_hat_final | H] = Z,
    E[Z_hat_final] = Z.

Conditioning on any completed-training event measurable from H also preserves
this argument. Selecting successful final estimates after drawing them would
not. Neither unbiased log-likelihood estimates nor exact finite-sample mean
ratios follow: generally E[log Z_hat] differs from log Z.

The actual controller at `reference_iapf_paper.R:362` follows the fresh-draw
step and returns that result. Its R conformance test now injects a deliberately
different post-stop value, verifying the returned value as well as the call
count. A mutation that still draws but returns the stopping estimate is
rejected. Eleven focused pytest cases pass, including this new mutation.
No filtering code, data, seeds or numerical settings changed during the runs.
The strengthened tests were added after source capture and are preserved
separately in the audit evidence; the earlier captured test files remain intact.

This establishes the identity for the idealized checked algorithm and rules
out the tested implementation mistakes. It does not prove exact unbiasedness
of finite-precision R calculations, pseudorandom streams, or arbitrary input
models. A bootstrap interval excluding1 can occur even for an unbiased
estimator. The completed [confirmation](result.md) did not reproduce the
earlier upward discrepancy; both new intervals include1. That empirical
finding is compatible with this derivation, but does not prove its assumptions
hold exactly in finite-precision computation.

## What still prevents an original-paper replication claim

Section5.1 explicitly fits a diagonal Gaussian using equation15 and then adds
a positive constant. The successful optional R route instead fits a quadratic
to log targets. It is a different objective. The written equation15 permits
an amplitude escape: for V=s^2 I, densities vanish as s grows; profiling lambda
gives loss p'p-(p'y)^2/(y'y), bounded above by p'p and hence tending to0.
Local optimizer behavior, initialization and constraints therefore matter.
The paper does not identify a numerical solver or bounds that resolve this
issue, nor its positive-floor formula or original data seeds. Its diagonal
covariance restriction alone does not prevent the escape.

The locally pinned public comparator
`.localresources/code/sempreteamo-iapf-a8811439/iapf.R:217` computes
sum(y-p/lambda)^2 with lambda=(p'y)/(y'y), a different objective from15.
Its L-BFGS-B calls at231--235 use nonnegative variance bounds and different
terminal/nonterminal initial means. It is not established as original-author
code, so those choices cannot be substituted as author settings. Two web
queries for official/source provenance returned upstream HTTP502 errors on
this turn; no new external provenance was recovered.

The remaining gaps are therefore substantive: original objective/solver
reconciliation, author floor and early doubling convention, author-code/data
provenance, the five-dimension1000-repeat study, and eventual TensorFlow parity.
More repetitions of the current optional method can validate that method,
but cannot turn it into an equation15 implementation or establish LEDH/KDM
correctness. The useful present conclusion is narrower: the checked proposal,
importance correction and fresh final evaluation do not require the fitted
guide to equal the ideal guide in order to target Z correctly.
