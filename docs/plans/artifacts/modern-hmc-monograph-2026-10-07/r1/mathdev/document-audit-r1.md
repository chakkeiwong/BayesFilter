# Actionable Math Document Rigor Audit

Source: `ch26g_modern_hmc_methods.tex`
Source SHA-256: `853ded03984ef291830b4ea1da86eb001cb47fadaf22ca5e9108eac6f473e161`
Coverage: `partial_coverage`; selected `30`; distinct issues `11`; open `10`; actionable proposals `6`; resolved by context `1`.

Detailed evidence pointer: `source_reports`; forensic rendering: `forensic_markdown`.

## Issue Ledger

### `eq:bf-modern-marginal-likelihood/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['definition', 'approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > The posterior, its coordinates and its computational cost > Marginal parameters and latent trajectories are different targets > eq:bf-modern-marginal-likelihood > line 52`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:bf-modern-parameter-target/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > The posterior, its coordinates and its computational cost > Marginal parameters and latent trajectories are different targets > eq:bf-modern-parameter-target > line 56`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:bf-modern-prediction-error/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > The posterior, its coordinates and its computational cost > Marginal parameters and latent trajectories are different targets > eq:bf-modern-prediction-error > line 71`
- Unresolved obligations: `['obligation_1', 'obligation_2']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:bf-modern-transformed-target/determinant-domain`

- Status: `resolved_by_existing_context`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > The posterior, its coordinates and its computational cost > Constraints and a frozen transport > eq:bf-modern-transformed-target > line 101`
- Existing context support:
  - `ch26g_modern_hmc_methods.tex:93-120`: Suppose the model chart $\vartheta=C(q)$ maps $q\in\R^d$ bijectively to the admissible parameter space. A further frozen transport $q=F(z)$ may improve geometry. Assume the maps are differentiable with nonsingular Jacobians on the domain of use. With $J_C$ and $J_F$ denoting those Jacobians, change of variables gives \begin{align}  \log\widetilde\pi_q(q)  &=\log p(C(q))+\ell(C(q))+\log\|\det J_C(q)\|,\\  \log\widetilde\pi_z(z)  &=\log\widetilde\pi_q(F(z))+\log\|\det J_F(z)\|.  \label{eq:bf-modern-transformed-target} \end{align} The tilde indicates that a normalization constant is unnecessary. The score must differentiate the same expression. The chain rule and Jacobi's determinant formula imply \begin{align}  \nabla_z\log\widetilde\pi_z(z)  &=J_F(z)^\top\nabla_q\log\widetilde\pi_q(F(z))       +\nabla_z\log\|\det J_F(z)\|,\\  \partial_{z_j}\log\|\det J_F(z)\|  &=\tr\{J_F(z)^{-1}\partial_{z_j}J_F(z)\}.  \label{eq:bf-modern-transformed-score} \end{align} To obtain the second line, use $d\det J=\det J\,\tr(J^{-1}dJ)$ and divide by $\det J$; its sign is locally constant when $J$ is nonsingular. Freezing the learned coefficients of $F$ removes adaptation during sampling. It does not remove the derivative through $F(z)$ or the log determinant.
- Boundary: The determinant-domain diagnostic is scoped exposition evidence, not a proof of a log-determinant identity.

### `eq:bf-modern-transformed-score/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > The posterior, its coordinates and its computational cost > Constraints and a frozen transport > eq:bf-modern-transformed-score > line 112`
- Existing context support:
  - `ch26g_modern_hmc_methods.tex:93-120`: Suppose the model chart $\vartheta=C(q)$ maps $q\in\R^d$ bijectively to the admissible parameter space. A further frozen transport $q=F(z)$ may improve geometry. Assume the maps are differentiable with nonsingular Jacobians on the domain of use. With $J_C$ and $J_F$ denoting those Jacobians, change of variables gives \begin{align}  \log\widetilde\pi_q(q)  &=\log p(C(q))+\ell(C(q))+\log\|\det J_C(q)\|,\\  \log\widetilde\pi_z(z)  &=\log\widetilde\pi_q(F(z))+\log\|\det J_F(z)\|.  \label{eq:bf-modern-transformed-target} \end{align} The tilde indicates that a normalization constant is unnecessary. The score must differentiate the same expression. The chain rule and Jacobi's determinant formula imply \begin{align}  \nabla_z\log\widetilde\pi_z(z)  &=J_F(z)^\top\nabla_q\log\widetilde\pi_q(F(z))       +\nabla_z\log\|\det J_F(z)\|,\\  \partial_{z_j}\log\|\det J_F(z)\|  &=\tr\{J_F(z)^{-1}\partial_{z_j}J_F(z)\}.  \label{eq:bf-modern-transformed-score} \end{align} To obtain the second line, use $d\det J=\det J\,\tr(J^{-1}dJ)$ and divide by $\det J$; its sign is locally constant when $J$ is nonsingular. Freezing the learned coefficients of $F$ removes adaptation during sampling. It does not remove the derivative through $F(z)$ or the log determinant.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:bf-modern-hamiltonian/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > Corrected HMC and the scope of its guarantee > Hamiltonian invariance > eq:bf-modern-hamiltonian > line 138`
- Existing context support:
  - `ch26g_modern_hmc_methods.tex:93-120`: Suppose the model chart $\vartheta=C(q)$ maps $q\in\R^d$ bijectively to the admissible parameter space. A further frozen transport $q=F(z)$ may improve geometry. Assume the maps are differentiable with nonsingular Jacobians on the domain of use. With $J_C$ and $J_F$ denoting those Jacobians, change of variables gives \begin{align}  \log\widetilde\pi_q(q)  &=\log p(C(q))+\ell(C(q))+\log\|\det J_C(q)\|,\\  \log\widetilde\pi_z(z)  &=\log\widetilde\pi_q(F(z))+\log\|\det J_F(z)\|.  \label{eq:bf-modern-transformed-target} \end{align} The tilde indicates that a normalization constant is unnecessary. The score must differentiate the same expression. The chain rule and Jacobi's determinant formula imply \begin{align}  \nabla_z\log\widetilde\pi_z(z)  &=J_F(z)^\top\nabla_q\log\widetilde\pi_q(F(z))       +\nabla_z\log\|\det J_F(z)\|,\\  \partial_{z_j}\log\|\det J_F(z)\|  &=\tr\{J_F(z)^{-1}\partial_{z_j}J_F(z)\}.  \label{eq:bf-modern-transformed-score} \end{align} To obtain the second line, use $d\det J=\det J\,\tr(J^{-1}dJ)$ and divide by $\det J$; its sign is locally constant when $J$ is nonsingular. Freezing the learned coefficients of $F$ removes adaptation during sampling. It does not remove the derivative through $F(z)$ or the log determinant.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:bf-modern-hamilton-equations/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > Corrected HMC and the scope of its guarantee > Hamiltonian invariance > eq:bf-modern-hamilton-equations > line 143`
- Existing context support:
  - `ch26g_modern_hmc_methods.tex:93-120`: Suppose the model chart $\vartheta=C(q)$ maps $q\in\R^d$ bijectively to the admissible parameter space. A further frozen transport $q=F(z)$ may improve geometry. Assume the maps are differentiable with nonsingular Jacobians on the domain of use. With $J_C$ and $J_F$ denoting those Jacobians, change of variables gives \begin{align}  \log\widetilde\pi_q(q)  &=\log p(C(q))+\ell(C(q))+\log\|\det J_C(q)\|,\\  \log\widetilde\pi_z(z)  &=\log\widetilde\pi_q(F(z))+\log\|\det J_F(z)\|.  \label{eq:bf-modern-transformed-target} \end{align} The tilde indicates that a normalization constant is unnecessary. The score must differentiate the same expression. The chain rule and Jacobi's determinant formula imply \begin{align}  \nabla_z\log\widetilde\pi_z(z)  &=J_F(z)^\top\nabla_q\log\widetilde\pi_q(F(z))       +\nabla_z\log\|\det J_F(z)\|,\\  \partial_{z_j}\log\|\det J_F(z)\|  &=\tr\{J_F(z)^{-1}\partial_{z_j}J_F(z)\}.  \label{eq:bf-modern-transformed-score} \end{align} To obtain the second line, use $d\det J=\det J\,\tr(J^{-1}dJ)$ and divide by $\det J$; its sign is locally constant when $J$ is nonsingular. Freezing the learned coefficients of $F$ removes adaptation during sampling. It does not remove the derivative through $F(z)$ or the log determinant.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:bf-modern-energy-conservation/matrix-domain-and-invertibility`

- Status: `partially_resolved`
- Roles: `['approximation_linearization']`
- Location: `ch26g_modern_hmc_methods.tex > Corrected HMC and the scope of its guarantee > Hamiltonian invariance > eq:bf-modern-energy-conservation > line 148`
- Existing context support:
  - `ch26g_modern_hmc_methods.tex:93-120`: Suppose the model chart $\vartheta=C(q)$ maps $q\in\R^d$ bijectively to the admissible parameter space. A further frozen transport $q=F(z)$ may improve geometry. Assume the maps are differentiable with nonsingular Jacobians on the domain of use. With $J_C$ and $J_F$ denoting those Jacobians, change of variables gives \begin{align}  \log\widetilde\pi_q(q)  &=\log p(C(q))+\ell(C(q))+\log\|\det J_C(q)\|,\\  \log\widetilde\pi_z(z)  &=\log\widetilde\pi_q(F(z))+\log\|\det J_F(z)\|.  \label{eq:bf-modern-transformed-target} \end{align} The tilde indicates that a normalization constant is unnecessary. The score must differentiate the same expression. The chain rule and Jacobi's determinant formula imply \begin{align}  \nabla_z\log\widetilde\pi_z(z)  &=J_F(z)^\top\nabla_q\log\widetilde\pi_q(F(z))       +\nabla_z\log\|\det J_F(z)\|,\\  \partial_{z_j}\log\|\det J_F(z)\|  &=\tr\{J_F(z)^{-1}\partial_{z_j}J_F(z)\}.  \label{eq:bf-modern-transformed-score} \end{align} To obtain the second line, use $d\det J=\det J\,\tr(J^{-1}dJ)$ and divide by $\det J$; its sign is locally constant when $J$ is nonsingular. Freezing the learned coefficients of $F$ removes adaptation during sampling. It does not remove the derivative through $F(z)$ or the log determinant.
- Unresolved obligations: `['dimension_contract']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:bf-modern-random-length-correlations/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['unknown']`
- Location: `ch26g_modern_hmc_methods.tex > Trajectory length, resonance and learned adaptation > Random lengths and NUTS > eq:bf-modern-random-length-correlations > line 309`
- Unresolved obligations: `['obligation_4', 'obligation_5']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:bf-modern-snaper-derivative/matrix-domain-and-invertibility`

- Status: `unresolved`
- Roles: `['statistical_estimator', 'equilibrium_condition', 'approximation_linearization', 'local_derived_claim']`
- Location: `ch26g_modern_hmc_methods.tex > Trajectory length, resonance and learned adaptation > SNAPER, endpoint derivatives and freezing > eq:bf-modern-snaper-derivative > line 409`
- Unresolved obligations: `['dimension_contract', 'invertibility']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

### `eq:bf-modern-langevin/matrix-domain-and-invertibility`

- Status: `unresolved`
- Roles: `['approximation_linearization', 'local_derived_claim']`
- Location: `ch26g_modern_hmc_methods.tex > Partial refreshment, MALT and MEADS > The Ornstein--Uhlenbeck refreshment > eq:bf-modern-langevin > line 457`
- Unresolved obligations: `['dimension_contract', 'invertibility']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

## Non-Claims

- Context closure means the document states the scoped exposition condition; it is not a proof certificate.
- Candidate patches are bounded human-review text and do not establish source-specific truth.
- This focused audit does not certify general readability, pedagogy, or whole-document correctness.
