# Actionable Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `808883a1112ed4ac5ea89840fc3580c7f2f9c3d6d234756139339f6da838d0dc`
Coverage: `partial_coverage`; selected `30`; distinct issues `9`; open `9`; actionable proposals `3`; resolved by context `0`.

Detailed evidence pointer: `source_reports`; forensic rendering: `forensic_markdown`.

## Issue Ledger

### `eq:actual-joint-hint/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'estimator_objective', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-joint-hint > line 348`
- Unresolved obligations: `['obligation_1', 'obligation_2', 'obligation_3']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:actual-previous-map/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-previous-map > line 360`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:actual-old-reexpression/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-old-reexpression > line 368`
- Existing context support:
  - `attempt05_n4_failure_analysis.tex:355-379`: For a row $u_i=(u_i^c,u_i^p)\in\mathbb R^{2n}$, the engine makes the physical pair \begin{align}   x_{t,i}&=m_t^c+L_{cc}u_i^c,   \label{eq:actual-current-map}\\   x_{t-1,i}&=m_t^p+L_{pc}u_i^c+L_{pp}u_i^p.   \label{eq:actual-previous-map} \end{align} These two lines define the current and previous physical coordinates for each reference row. The previous state in this row is then re-expressed in the \emph{old} whitened coordinates: \begin{equation}   u_i^{\rm old}=(L_{t-1}^{\rm TT})^{-1}        (x_{t-1,i}-m_{t-1}^{\rm TT}).   \label{eq:actual-old-reexpression} \end{equation} Here $L_{t-1}^{\rm TT}\in\mathbb R^{n\times n}$ is lower triangular with strictly positive diagonal, because the engine stores the Cholesky factor of the previous hint covariance in the retained coordinate map.  It is therefore invertible (nonsingular), so the inverse in this definition exists for every row. This solve is performed for every row.  There is no truncation to a box and no containment test; the old retained density is evaluated at the exact re-expressed point.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:actual-transition-scale/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-transition-scale > line 400`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:actual-gh-slope/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['definition', 'approximation_linearization', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > How the current GH9 hint is produced > eq:actual-gh-slope > line 477`
- Existing context support:
  - `attempt05_n4_failure_analysis.tex:355-379`: For a row $u_i=(u_i^c,u_i^p)\in\mathbb R^{2n}$, the engine makes the physical pair \begin{align}   x_{t,i}&=m_t^c+L_{cc}u_i^c,   \label{eq:actual-current-map}\\   x_{t-1,i}&=m_t^p+L_{pc}u_i^c+L_{pp}u_i^p.   \label{eq:actual-previous-map} \end{align} These two lines define the current and previous physical coordinates for each reference row. The previous state in this row is then re-expressed in the \emph{old} whitened coordinates: \begin{equation}   u_i^{\rm old}=(L_{t-1}^{\rm TT})^{-1}        (x_{t-1,i}-m_{t-1}^{\rm TT}).   \label{eq:actual-old-reexpression} \end{equation} Here $L_{t-1}^{\rm TT}\in\mathbb R^{n\times n}$ is lower triangular with strictly positive diagonal, because the engine stores the Cholesky factor of the previous hint covariance in the retained coordinate map.  It is therefore invertible (nonsingular), so the inverse in this definition exists for every row. This solve is performed for every row.  There is no truncation to a box and no containment test; the old retained density is evaluated at the exact re-expressed point.
  - `attempt05_n4_failure_analysis.tex:447-504`: The word \emph{frozen} refers to the runtime interface, not to a constant mean over time.  During fixture preparation, the GH factory maintains a Gaussian moment state.  For a prior $N(m^-,P^-)$, write $P^-=L^-{L^-}^{\mathsf T}$ and let $(z_k,\omega_k)$ be the tensor-product Hermite nodes and normalized weights.  It forms \begin{equation}   x_k=m^-+L^-z_k,\qquad   \widetilde\omega_k=\omega_k\,g_t(y_t\mid x_k),\qquad   \bar\omega_k=\frac{\widetilde\omega_k}{\sum_\ell\widetilde\omega_\ell}.   \label{eq:actual-gh-nodes} \end{equation} The following equations define the finite GH9 Gaussian projection after the observation; they are a quadrature moment approximation, not an exact nonlinear posterior: \begin{align}   m_t^c&=\sum_k\bar\omega_kx_k,&   P_{cc,t}&=\sum_k\bar\omega_k     (x_k-m_t^c)(x_k-m_t^c)^\mathsf T.   \label{eq:actual-gh-update} \end{align} At $t=0$, $m^-=0$ and $P^-=P_0$.  At later times the factory first predicts \begin{equation}   m_t^-=A\,m_{t-1}^c,\qquad   P_t^-=A\,P_{t-1}^cA^\mathsf T+Q,\qquad   C_{pc,t}^-=P_{t-1}^cA^\mathsf T,   \label{eq:actual-gh-prediction} \end{equation} then applies~\eqref{eq:actual-gh-update}.  To supply the joint hint required by the TT engine, it uses the Gaussian conditional slope \begin{equation}   G_t=C_{pc,t}^-{P_t^-}^{-1}   \label{eq:actual-gh-slope} \end{equation} In this fixture $P_t^-\in\mathbb R^{n\times n}$ and $P_t^-\succ0$ because $Q\succ0$.  Hence $P_t^-$ is invertible (nonsingular), and the inverse in \eqref{eq:actual-gh-slope} is defined; the display is the conditional regression coefficient, not an additional fitted parameter.  It then sets \begin{align}   m_t^p&=m_{t-1}^c+G_t(m_t^c-m_t^-),&   P_{pp,t}&=P_{t-1}^c-G_tP_t^-G_t^\mathsf T                     +G_tP_{cc,t}G_t^\mathsf T,\\   P_{pc,t}&=G_tP_{cc,t},&   \bar P_t&=   \begin{bmatrix}P_{cc,t}&P_{pc,t}^\mathsf T\\                   P_{pc,t}&P_{pp,t}\end{bmatrix}.   \label{eq:actual-gh-joint} \end{align} Here the lower-left block $P_{pc,t}$ is $\operatorname{Cov}(X_{t-1},X_t\mid y_{0:t})$, so the displayed block matrix has the stated $(X_t,X_{t-1})$ ordering. The pair $(\bar m_t,\bar P_t)$ is serialized in the fixture.  At runtime the engine only reads that pair and takes its Cholesky factor.  Thus this is a recursive \emph{GH Gaussian projection} performed outside the TT fit, not a UKF and not a moment contraction of $H_{t-1}$.  The use of a log-sum-exp weight normalization in the implementation is algebraically equivalent to the normalized weights in~\eqref{eq:actual-gh-nodes}; the fixture uses the tensor-product rule with nine nodes per state axis.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:actual-gh-joint/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'approximation_linearization', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > How the current GH9 hint is produced > eq:actual-gh-joint > line 492`
- Unresolved obligations: `['obligation_4']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:actual-ukf-points/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'source_reported_result', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-points > line 549`
- Unresolved obligations: `['obligation_4', 'obligation_5', 'obligation_6']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:actual-ukf-update/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['definition', 'source_reported_result', 'local_derived_claim']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-update > line 573`
- Existing context support:
  - `attempt05_n4_failure_analysis.tex:544-599`: For comparison, a standard unscented transform for a $d$-dimensional Gaussian $X\sim N(m,P)$ chooses $P=SS^\mathsf T$ and $\lambda=\alpha^2(d+\kappa)-d$, $c=d+\lambda$.  Its $2d+1$ sigma points and weights are defined by the following standard UKF rule: \begin{align}   \chi_0&=m,&   \chi_i&=m+\sqrt c\,S_{:i},&   \chi_{i+d}&=m-\sqrt c\,S_{:i},   \label{eq:actual-ukf-points}\\   W_0^m&=\lambda/c,&   W_0^c&=\lambda/c+1-\alpha^2+\beta_{\rm UKF},&   W_i^m=W_i^c&=1/(2c)\quad(i>0).   \label{eq:actual-ukf-weights} \end{align} Here $\beta_{\rm UKF}$ is the usual unscented-transform tuning parameter; it is unrelated to the C2 observation scale $\beta=0.4$. For a nonlinear observation $z=h(x)+\nu$, propagate $\zeta_i=h(\chi_i)$ and form \begin{align}   \bar z&=\sum_iW_i^m\zeta_i,&   P_{zz}&=\sum_iW_i^c(\zeta_i-\bar z)(\zeta_i-\bar z)^\mathsf T+R,&   P_{xz}&=\sum_iW_i^c(\chi_i-m)(\zeta_i-\bar z)^\mathsf T.   \label{eq:actual-ukf-moments} \end{align} Assume the observation-noise covariance satisfies $R\succ0$.  Then the innovation covariance $P_{zz}\in\mathbb R^{d_z\times d_z}$ is positive definite and therefore invertible (nonsingular), so the following inverse is defined: \begin{equation}   K=P_{xz}P_{zz}^{-1},   \qquad m^+=m+K(y-\bar z),   \qquad P^+=P-KP_{zz}K^\mathsf T.   \label{eq:actual-ukf-update} \end{equation} The measurement update is the definition of the UKF Gaussian projection, not an exact posterior update for a nonlinear observation. For the adjacent pair needed by the TT engine, augment the sigma-point calculation with the previous-state block.  If $P_{pz}$ is its cross-covariance with the transformed observation, set $K_c=P_{cz}P_{zz}^{-1}$ and $K_p=P_{pz}P_{zz}^{-1}$.  The joint update is \begin{align}   m_c^+&=m_c^-+K_c(y-\bar z),&   m_p^+&=m_p^-+K_p(y-\bar z),\\   P_{cc}^+&=P_{cc}^--K_cP_{zz}K_c^\mathsf T,&   P_{pp}^+&=P_{pp}^--K_pP_{zz}K_p^\mathsf T,&   P_{pc}^+&=P_{pc}^--K_pP_{zz}K_c^\mathsf T.   \label{eq:actual-ukf-joint-update} \end{align} This is the block form of the same Gaussian conditioning update applied to the augmented state $(X_t,X_{t-1})$; it is not an exact nonlinear Bayes rule. The updated blocks are assembled in the $(X_t,X_{t-1})$ order before taking the Cholesky factor passed to the TT routine. These equations explain exactly what it would mean to insert a UKF into the moment-provider slot: sigma points would be propagated through the transition and observation, their weighted moments would be factored, and the resulting joint mean/Cholesky would be passed to the TT engine.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:actual-initial-retention/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['estimator_objective']`
- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > The initial step $t=0$ > eq:actual-initial-retention > line 318`
- Unresolved obligations: `['obligation_2']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

## Non-Claims

- Context closure means the document states the scoped exposition condition; it is not a proof certificate.
- Candidate patches are bounded human-review text and do not establish source-specific truth.
- This focused audit does not certify general readability, pedagogy, or whole-document correctness.
