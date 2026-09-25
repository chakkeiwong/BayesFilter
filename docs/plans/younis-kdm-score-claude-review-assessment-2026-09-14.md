# Assessment of Claude's score-program review

Date: 2026-09-14. Question: which findings in the
[Claude reply](younis-kdm-score-master-program-claude-review-reply-2026-09-14.md)
should inform the current research program?

The review is useful as a checklist, but its mathematical approval is
unsupported. Two of its central assertions are wrong. Several observations
also describe a different state from the current plan or working tree. Retain
the useful checks, correct the mathematical errors, and use the
[current handoff](younis-kdm-score-master-program-claude-review-handoff-2026-09-14.md)
for the remaining questions. This assessment does not approve a research
campaign or reject the KDM research direction.

## Findings worth retaining

- **Initial-law derivatives:** explicitly exercise parameter-dependent initial
  means and covariances. The current master already requires initial
  dependencies in Phase 1, lines 438–441. The inspected scalar score executor
  initializes `d_states` and `d_covariances` to zero, lines 221–224. That is
  appropriate for fixed initial inputs; it does not supply the missing
  sensitivities when initialization depends on the differentiated parameter.
  First check the derivative of the same finite program. A finite-particle
  estimate need not equal the exact Kalman model score at every realization.
- **Covariance observability:** retain the actual per-particle covariance
  lifecycle in diagnostic outputs and check which proposal consumes it.
  Covariance calibration can explain score error; it cannot certify score
  accuracy by itself.
- **Control-variate centering:** test the expectation of the proposed control
  under its declared sampling law, including coefficient fitting and data
  separation. For fixed coefficient matrix B and correctly known center c,
  E[S0 - B(C-c)] = E[S0]. Such a control preserves the baseline's bias. An
  inaccurately centered control can change that bias.
- **Explicit budgets:** specify total compute and attempt limits in the actual
  execution plan. The review's stronger assertion that no particle, horizon,
  or replication bounds exist is wrong for the current master: lines 300–310
  specify particle and horizon ladders and a replication maximum. These do
  not alone specify a total campaign budget.

## Correction 1: ratio bias is an expectation mismatch

The reply, lines 50 and 115, proposes distinguishing D_hat/Z_hat from the
derivative of log Z_hat as a witness of ratio bias. If Z_hat(theta; xi) is
positive and differentiable for fixed random input xi, and
D_hat = d Z_hat/d theta, the chain rule gives exactly

\[
  \frac{\widehat D}{\widehat Z}
  = \frac{d}{d\theta}\log\widehat Z.
\]

There is no gap to detect between those two quantities. The bias question is
whether E[D_hat/Z_hat] equals (dZ/dtheta)/Z, where Z is the model normalizer.
Unbiased estimates of Z and its derivative do not imply equality of these
ratios. The current master states this correctly at lines 258–265.

A complete counterexample uses Z(theta)=theta and
Z_hat(theta)=theta+X, with P(X=a)=P(X=-a)=1/2 and theta>a>0.
Then D_hat=1 is both the exact derivative of Z_hat and an unbiased estimate
of Z'. Nevertheless,

\[
  E\!\left[\frac{\widehat D}{\widehat Z}\right]
  =\frac12\left(\frac1{\theta+a}+\frac1{\theta-a}\right)
  =\frac{\theta}{\theta^2-a^2}
  \ne \frac1\theta
  =\frac{d}{d\theta}\log Z.
\]

This is a suitable ratio-bias witness. If D_hat is constructed separately and
is not the derivative of the same Z_hat, the chain-rule identity need not
apply; that is a distinct estimator definition, not evidence against it.

## Correction 2: restoring moments does not preserve a distribution

The reply, lines 20 and 54, calls the OT reset measure-preserving and attributes
the accumulated likelihood Jacobian to that reset. Neither assertion is
supported by the inspected code.

For a simple counterexample, an empirical distribution with two distinct atoms
and weights 1/4 and 3/4 cannot be represented exactly by two equally weighted
atoms: every singleton mass of the latter is 0, 1/2, or 1. Matching its mean
and covariance cannot change this fact. Determinism does not imply equality
of the input and output distributions.

The Contract E implementation computes source and transported moments,
injects a residual design, and applies a ridged affine transformation
(`ledh_contract_e_reset_tf.py`, lines 60–85). These operations do not prove
equality of the filtering distributions. In the inspected scalar executor,
the log determinant used in the importance weight comes from
`_flow_substeps_with_tangent`, lines 320 and 347. It is a flow determinant.
The Contract E reset occurs later, starting at line 392. Its dependence on
the source cloud, weights, and transport belongs in subsequent total
derivatives; this does not justify inventing an additional reset determinant
in the likelihood. A density correction requires its own sampling-law and
change-of-variables derivation.

## Evidence criteria and current execution status

The review calls variance reduction the primary criterion at line 183.
For biased score estimates this is insufficient: squared error decomposes
into squared bias plus the trace of the covariance. The current master's
paired oracle-error criterion, lines 267–294, is the appropriate criterion
to retain. Equal-particle and equal-compute comparisons should be reported
as separate comparisons when their resource constraints differ.

The reply records commit `5cc59cfa`; this assessment inspected branch
`surrogate-hmc` at `5836f0344293f1c4af85abba23689ba83d34d9af`, including
uncommitted changes in the canonical score executor. A commit identifier
alone does not identify the untracked master document that was reviewed.

There is a concrete incompatibility visible in the current source:

| Consumer or executor | Inspected behavior |
| --- | --- |
| Integrated KDM, `ledh_younis_kdm_integrated_tf.py:522` | Requests `return_trace=True` and an observation-factor callback. |
| Resampling KDM, `ledh_younis_kdm_resampling_tf.py:443` | Requests `return_trace=True` and a post-reset callback. |
| Shared scalar executor, `ledh_canonical_score_tf.py:179` | Explicitly raises for those trace/callback options. |

This is a static finding in the current working tree, not proof about the
checkout Claude actually inspected. **Executable call-chain verification:
not checked in this assessment.** The review's existing approval cannot
establish present readiness; the source conflict needs resolution and a
focused endpoint regression before those routes can supply new evidence.
No concurrent implementation changes were modified here.

The current handoff also identifies questions about SGQF integration,
twisting/iAPF, finite-difference smoothness and noisy error tests, and the
placement of tuning before quality claims. The reply does not resolve those
questions with checked derivations or current execution artifacts. Its
paper-section and bibliography assertions were not independently re-audited
in this assessment and must not be treated as newly verified source support.

## Decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not established |
| --- | --- | --- | --- | --- | --- |
| Retain the review as an issue list; revise its approval verdict | Current oracle-error criterion retained; no new empirical comparison | Two mathematical errors require correction; current source has a trace/callback conflict | Runtime behavior and remaining source/theory questions have not been rechecked | Carry the corrected findings into the existing handoff; resolve prerequisites before experimental use | Score improvement, present implementation readiness, or completion of a mathematical audit |

Validation for this assessment consists of the derivations above, selected
plan/source inspection, and document checks. No GPU job, experiment, Claude
invocation, or MathDevMCP audit was run. The older recovery and manuscript
work is already completed, as recorded in
[the September 12 continuation note](younis-score-session-continuation-2026-09-12.md).
