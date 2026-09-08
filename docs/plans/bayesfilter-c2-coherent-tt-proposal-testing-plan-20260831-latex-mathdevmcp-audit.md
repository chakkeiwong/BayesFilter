# Actionable Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `11d8622befa67e4d00d51b0f425442e09969a6ada143f2b977dc67a1d21ada34`
Coverage: `partial_coverage`; selected `30`; distinct issues `6`; open `5`; actionable proposals `2`; resolved by context `1`.

Detailed evidence pointer: `source_reports`; forensic rendering: `forensic_markdown`.

## Issue Ledger

### `eq:branch-target/formalization-and-source-role`

- Status: `resolved_by_existing_context`
- Roles: `['definition']`
- Location: `attempt05_n4_failure_analysis.tex > What the Engine Actually Computes > Frozen pre-update state and branch target > eq:branch-target > line 195`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:als/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > What the Engine Actually Computes > The implemented ALS objective > eq:als > line 233`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:q-physical/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['definition']`
- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > The retained marginal as a proposal > eq:q-physical > line 670`
- Existing context support:
  - `attempt05_n4_failure_analysis.tex:653-685`: Fix the observations $y_{0:T-1}$ and an offline reference parameter $\theta_\star$. For $t\geq1$, after the direct C2 fit at time $t$, let $u\in\mathbb R^n$ denote the current whitened coordinate and \begin{equation}\label{eq:proposal-map} x=m_t+L_tu, \qquad L_t\text{ lower triangular with positive diagonal}. \end{equation} Let $V_t(u)\in\mathbb R^{1\times r_t}$ be the row vector obtained by contracting the retained current-state TT cores, and let $E_t\in\mathbb R^{r_t\times r_t}$ be the positive-semidefinite retained suffix Gram. With $\eta_n$ the standard-normal density and $t_{\nu,n}$ the product Student-$t$ density used by the defensive route, define \begin{align} h_t(u)&=V_t(u)E_tV_t(u)^\mathsf T,\\ Z_{H,t}&=\int h_t(u)\eta_n(u)\,du,\\ \widetilde q_t(u)&=h_t(u)\eta_n(u)+\tau_t^{\rm abs}t_{\nu,n}(u),\\ Z_{q,t}&=Z_{H,t}+\tau_t^{\rm abs},\label{eq:q-normalizer}\\ q_t(x)&=\frac{\widetilde q_t(L_t^{-1}(x-m_t))}  {Z_{q,t}\|\det L_t\|}.\label{eq:q-physical} \end{align} Here $n\geq1$, $u,x,m_t\in\mathbb R^n$, and $L_t\in\mathbb R^{n\times n}$. The lower-triangular positive-diagonal condition makes $L_t$ nonsingular, so $L_t^{-1}$ and the strictly positive Jacobian determinant in equation~\eqref{eq:q-physical} are defined. Proposal density $q_t$ is with respect to $n$-dimensional Lebesgue measure in physical coordinates (with the corresponding standard-reference measure in $u$). When the configured defensive law is the Gaussian reference rather than a Student-$t$, replace $t_{\nu,n}$ by $\eta_n$. In the implementation, $\tau_t^{\rm abs}$ is the absolute retained floor mass, not the dimensionless ratio used to report $\tau_t$. The fitted Gram gives $Z_{H,t}$ exactly for the stored coefficients. Because both reference densities integrate to one, equation~\eqref{eq:q-normalizer} follows without an empirical row-count normalization.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:hermite-antiderivative/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Exact Gaussian--Hermite conditional CDFs > eq:hermite-antiderivative > line 755`
- Unresolved obligations: `['obligation_2', 'obligation_3']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:initial-gamma-score/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Manual score for the C2 SV parameterization > eq:initial-gamma-score > line 997`
- Unresolved obligations: `['obligation_1', 'square_matrix_required']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:transition-gamma-score/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Manual score for the C2 SV parameterization > eq:transition-gamma-score > line 1014`
- Existing context support:
  - `attempt05_n4_failure_analysis.tex:955-980`: For the first C2 test, parameterize \begin{equation} \theta=(\gamma,\xi),\qquad \beta=e^\xi,\qquad A_\gamma=C+\gamma I_n, \end{equation} where the fixture seed fixes $C$, $\sigma=1$, and $Q=\sigma^2I_n\succ0$. Restrict $\gamma$ to the open stability domain $\rho(A_\gamma)<1$. On this domain the stationary covariance is uniquely positive definite and satisfies \begin{equation}\label{eq:lyapunov} P_\gamma=A_\gamma P_\gamma A_\gamma^\mathsf T+Q. \end{equation} Since $\partial_\gamma A_\gamma=I_n$, differentiation gives \[ \dot P_\gamma =P_\gamma A_\gamma^\mathsf T  +A_\gamma\dot P_\gamma A_\gamma^\mathsf T  +A_\gamma P_\gamma, \] and hence the second Lyapunov equation \begin{equation}\label{eq:lyapunov-dot} \dot P_\gamma-A_\gamma\dot P_\gamma A_\gamma^\mathsf T =P_\gamma A_\gamma^\mathsf T+A_\gamma P_\gamma. \end{equation} The associated linear operator is invertible because $\rho(A_\gamma\otimes A_\gamma)<1$, so $\dot P_\gamma$ is unique.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

## Non-Claims

- Context closure means the document states the scoped exposition condition; it is not a proof certificate.
- Candidate patches are bounded human-review text and do not establish source-specific truth.
- This focused audit does not certify general readability, pedagogy, or whole-document correctness.
