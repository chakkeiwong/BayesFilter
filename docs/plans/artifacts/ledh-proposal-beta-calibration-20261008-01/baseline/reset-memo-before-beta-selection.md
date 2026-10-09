# Active SIR diagnosis checkpoint

Question: why do baseline and tuned SIR value/scores fail relative to the
TT/bootstrap reference? Stage: bounded local diagnosis complete, no runtime
repair promoted. Branch `sqmc-development`, base `209223fdd`. Only the new
diagnostic script, plan, result note, evidence directory and this memo are dirty.

Checked: two original N=1008/T=10 GPU FP64/XLA designs replay value and first
score exactly. Actual marginal-flow integration errors relative to exact
incoming-cloud Gaussian integration sum to -279.298 and -158.886. The same
shared flow and marginal correction with covariance Q instead of accumulated
UKF P gives local shadow errors +0.000196 and +0.003450, without propagating
the shadow into the next step. The failure is proposal coverage caused by
covariance mismatch, greatly amplified in the early SIR growth regime.

Results: `docs/benchmarks/ledh-sir-proposal-diagnosis-20261007.md`.
Plan: `docs/plans/ledh-sir-proposal-diagnosis-20261007.md`.
Evidence: `docs/plans/artifacts/ledh-sir-proposal-diagnosis-20261007-01/`.
Three successful diagnostic attempts used 114.12 worker seconds; the three
attempt diagnostic budget is closed. No further run remains in this plan.

Follow-up clarification: the local UKF covariance update was not shown wrong.
P_j describes a carried Gaussian uncertainty, whereas Q conditions on a fixed
ancestor. The full posterior covariance also contains between-ancestor
dispersion. P_j != Q does not itself invalidate importance sampling; measured
proposal inefficiency is the finding. See the added clarification in the result
note. No new experiment or runtime change was made.

AR(1) clarification: the whole predictive cloud keeps phi^2 P_0+Q, equal to
stationary P_0 when initialized at Q/(1-phi^2). Never replace that total by Q.
The crucial additional detail is ancestor-specific flow centring: a common flow
using the full predictive mean and covariance can be correct even when P != Q.
The present code passes each transition anchor as its flow's prior mean. The
result note derives the resulting excess contraction of between-ancestor
spread in a scalar example. The user's full-cloud moment-matching instruction
was not shown wrong. No new filtering experiment was run.

Previous action, now completed: explain the AR(1) example and flow-centre distinction. A subsequent
repair must define the proposal covariance policy
explicitly, preserve the shared analytical score and covariance lifecycle,
and validate full recursive T=50 value/all scores plus protected model scopes.
Local shadow accuracy and prior finite-difference parity do not establish
repaired full likelihoods, scores, or default readiness.


Discussion update: changing the ancestor-specific flow covariance to Q makes the
auxiliary UKF covariance recursion irrelevant to accepted particle/value updates
in the proposed single-stage pseudocode, apart from validity checks. Runtime is
unchanged. Current Contract E targets weighted children's moments; nonzero ridges
and later moment repair mean final covariance agreement must be measured.

Second-order ETPF preserves importance-weighted moments, not necessarily the true
posterior moments. Inspected local Acevedo/de Wiljes/Reich paper: Definition 3.1,
equations (11)-(13), Section 6 algorithm, Section 7.3 limitation. For tested SIR,
Gaussian transitions and linear Gaussian observations yield exact posterior
mixture moments conditional on the incoming discrete ancestor distribution:
alpha_j proportional to omega_j N(y; H m_j, H Q H' + R), mu_j=m_j+K(y-H m_j),
C=Q-K H Q, P=C+sum alpha_j(mu_j-mubar)(mu_j-mubar)'. This provides a direct local
covariance check. Using these as new reset targets is a further candidate change,
not an implemented or promoted default. No new experiment was run.

Current task completed: the self-contained proposal and prior MathDevMCP audit
are now integrated into docs/main.tex as Chapter 27, printed pages 245–264
(PDF pages 265–284). docs/main.pdf (642 pages) and the standalone 19-page PDF
both compile. A shared body preserves all 49 equations and three algorithms.

Current evidence:
- docs/chapters/ch32c3_ukf_covariance_guided_particle_proposal.tex (wrapper).
- docs/chapters/ledh_covariance_proposal_body.tex (shared mathematical text).
- docs/plans/ledh-covariance-monograph-integration-results-20261008.md.
- docs/plans/artifacts/ledh-covariance-monograph-integration-20261008-01/.
- docs/plans/ledh-covariance-latex-mathdev-audit-results-20261008.md.

Exact reversible content preservation, all new references and citations, actual
compiler input chains, and rendered pages passed. The standalone is warning-free;
new chapter warnings concern spacing only; existing other-chapter layout and
hyperref warnings remain. Prior MathDevMCP coverage and limits are unchanged:
49 initial labels plus four revised labels, nine bounded symbolic checks proved,
two inputs not encodable, and incomplete formalization of the general proofs.

The combined filter remains an unimplemented, unpromoted proposal. Its optional
zero-start higher-moment optimizer can stall on a symmetric cloud; boundedness
does not guarantee feasible or accurate higher moments. No runtime edit or new
experiment was made. Branch sqmc-development, base 209223fdd; unrelated diagnostics
preserved. The documentation build budget is complete; old diagnostic budget
remains closed. Human readability feedback is pending.

Next action: deliver the integrated chapter. If implementation is requested,
start with the simpler conditional-covariance repair under a bounded plan and
protected cross-model value/all-score comparisons. Do not promote the composite
merely because the derivation or document compiles.
