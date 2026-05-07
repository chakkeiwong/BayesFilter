# Reset memo: BayesFilter monograph consolidation

## Date
2026-05-02

## Context
The user wants a new BayesFilter monograph under `~/BayesFilter/docs` that
consolidates filtering, HMC, analytic Kalman derivatives, SVD-filter numerical
lessons, and industrial-scale state-space inference material now scattered
across:

- `~/python/docs/monograph.tex`
- `~/python/docs/chapters/*.tex`
- `~/latex/CIP_monograph/main.tex`
- `~/latex/CIP_monograph/chapters/*.tex`
- `~/MacroFinance/analytic_kalman_derivatives.tex`
- related implementation notes, tests, and reset memos

The original documents are source/provenance material and should not be edited
as part of this monograph consolidation. Work should happen inside
`~/BayesFilter`.

## Initial repo state
- Repo: `/home/chakwong/BayesFilter`
- Branch: `main`
- Initial commit before this work: `a09e8d1 first commit`
- Initial tracked file: `README.md`
- Existing untracked file before this execution pass:
  `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`

## User request
1. Update the reset memo.
2. Audit the plan as another developer.
3. Execute each phase using a plan/execute/test/audit/tidy/update-memo cycle.
4. After each phase, update the memo with results, interpretation, and whether
   the next phase is justified.
5. Continue without human intervention if no major issue blocks the next phase.
6. Commit modified files after the whole plan finishes.
7. Update the reset memo on completion.
8. Provide a detailed result summary and next-phase hypotheses.

## Execution policy
- Do not edit the original source documents in `~/python`, `~/latex`, or
  `~/MacroFinance`.
- Use `MathDevMCP` where useful for source discovery, derivation/code auditing,
  and notation/shape review.
- Use `ResearchAssistant` where useful for literature workspace setup and
  claim-support planning.
- Stop content migration until the BayesFilter skeleton builds and the initial
  source map exists.

## Current status
This memo starts the execution pass. The plan audit and Phase 0 scaffold are
next.

## 2026-05-07 update: analytic-derivative chapter completion pass started

User requested a full analytic-derivative documentation completion pass with the
same cycle used by earlier monograph work:

```
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Scope for this pass:
- complete the analytic-derivative chapters:
  `docs/chapters/ch09_kalman_score.tex`,
  `docs/chapters/ch10_kalman_hessian.tex`,
  `docs/chapters/ch11_structural_derivatives.tex`,
  `docs/chapters/ch12_factor_derivatives.tex`,
  `docs/chapters/ch13_custom_gradient_wrappers.tex`, and
  `docs/chapters/ch14_derivative_validation.tex`;
- keep the already-added SVD sigma-point score/Hessian material in
  `docs/chapters/ch18_svd_sigma_point.tex` aligned with those chapters;
- update `docs/source_map.yml` and this reset memo with phase results;
- compile and audit after each phase;
- commit the modified files after the whole pass completes.

Initial plan audit, pretending to be a separate developer:
- The proposed order is justified.  The score and Hessian chapters must define
  the backend-neutral innovation derivative contract before factor,
  structural, custom-gradient, and validation chapters depend on it.
- The plan would be incomplete without explicit first- and second-order update
  recursions for the Kalman gain, filtered mean, and filtered covariance.  Those
  recursions are needed so Chapter 10 is not merely a likelihood-Hessian formula
  detached from the filtering scan.
- The plan would also be incomplete without a clear distinction between
  covariance-form derivatives, solve-form likelihood derivatives, triangular
  factor derivatives, and spectral factor derivatives.  Those are related but
  not interchangeable.
- SVD/eigen derivative formulas may be stated on smooth simple-spectrum
  branches, but the chapter must not claim production HMC certification without
  finite-difference, branch, gap, and fallback evidence.
- Missing-data masks, fixed quadrature rules, transformed parameters, and
  initial-state derivatives must be named explicitly as derivative contracts.
- MathDevMCP can support provenance checks for local labels, but manual
  derivation and numerical parity remain the primary evidence for the formulas.
  ResearchAssistant has no local summaries for the SVD/eigen derivative papers,
  so this pass will rely on committed bibliography entries and source-map
  records rather than unsupported literature claims.
- No blocker prevents Phase 1.  The pass should continue without human
  intervention unless compilation fails, the source-map update exposes a
  provenance contradiction, or a formula cannot be reconciled with the
  MacroFinance derivative note.

### Phase AD-1: source-map and provenance update

Phase plan:
- promote the MacroFinance analytic derivative source-map entries from generic
  candidate status to drafted derivative-chapter provenance where the formulas
  are now being consolidated;
- add this 2026-05-07 analytic-derivative completion pass as a BayesFilter
  source-map item;
- record the key MacroFinance equation labels that support score, Hessian,
  factor, and structural derivative chapters;
- parse the YAML and compile the monograph before proceeding.

Execution:
- Updated `docs/source_map.yml` with a new
  `bayesfilter_analytic_derivative_completion_pass` source item.
- Promoted the MacroFinance analytic derivative spine and related code-reference
  entries for covariance-form, solve-form, QR square-root, and TensorFlow/XLA
  derivative paths to `drafted`.
- Added an explicit label list for the derivative formulas being consolidated:
  `eq:dm_pred_note`, `eq:dP_pred_note`, `eq:dv_note`, `eq:dS_note`,
  `eq:score_note`, `eq:solve_score_proved`, `eq:first_w_derivative`,
  `eq:second_w_derivative`, `eq:solve_hessian_proved`,
  `eq:first_cholesky_derivative`, `eq:second_cholesky_derivative`,
  `eq:first_qr_r_derivative`, `eq:second_qr_r_derivative`,
  `eq:structural_map_g`, `eq:structural_first_derivative_contract`, and
  `eq:structural_second_derivative_contract`.

Tests:
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map_yaml_ok')"`
  passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported the
  PDF up to date.
- `git diff --check` passed.

Audit:
- The provenance layer is now explicit enough to support the formula-expansion
  phases.
- No source-map contradiction was found.
- ResearchAssistant still has no local paper summaries for the SVD/eigen
  derivative literature, so this pass must continue to phrase spectral formulas
  as source-backed/bibliography-backed mathematical contracts plus validation
  targets, not as literature-certified production readiness.
- Next phase remains justified: Chapter 9 can now be expanded against the
  MacroFinance score labels and solve-form source labels.

### Phase AD-2: Kalman score chapter

Phase plan:
- expand Chapter 9 from a compact score formula into a complete first-order
  derivative scan;
- include prediction, innovation, score, gain, filtered mean, and filtered
  covariance derivatives;
- include the missing-data mask convention because masked observations change
  the effective observation equation but not the derivative algebra;
- compile, scan references, check whitespace, and audit for overclaiming.

Execution:
- Updated `docs/chapters/ch09_kalman_score.tex`.
- Added prediction derivative equations for
  `\dot m_{t|t-1}` and `\dot P_{t|t-1}` with MacroFinance source labels
  `eq:dm_pred_note` and `eq:dP_pred_note`.
- Added labeled innovation derivatives for `\dot v_t` and `\dot S_t`.
- Kept both covariance-form and solve-form score equations.
- Added covariance-form and solve-form gain derivative equations.
- Added filtered mean and filtered covariance derivative propagation.
- Added an explicit mask/time-varying observation block convention, including
  zero likelihood/score contribution and identity update for all-missing times.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed after
  the expected reruns for new labels.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches after the final rerun.
- `git diff --check` passed.

Audit:
- Chapter 9 now defines the first-order filtering scan, not only the final score
  contraction.
- The chapter remains honest about evidence: MathDevMCP AST checks are
  provenance support, while MacroFinance finite-difference/autodiff/backend
  parity tests remain the stronger validation evidence.
- The missing-data convention is now explicit and consistent with Chapter 7.
- Next phase remains justified: Chapter 10 can build on Chapter 9 by adding the
  second-order prediction, innovation, gain, update, and solve-Hessian contract.

### Phase AD-3: Kalman Hessian chapter

Phase plan:
- expand Chapter 10 from the compact solve-Hessian formula into a complete
  second-order filtering scan;
- include second-order prediction, innovation, gain, filtered mean, and
  filtered covariance derivatives;
- make the solve-form Hessian bridge explicit so linear, square-root, and
  sigma-point backends can share the same likelihood-Hessian contraction once
  they supply matching innovation derivatives;
- compile, scan references, check whitespace, and audit the evidence language.

Execution:
- Updated `docs/chapters/ch10_kalman_hessian.tex`.
- Added second-order prediction equations for
  `\ddot m_{t|t-1}` and `\ddot P_{t|t-1}`.
- Added second-order innovation equations for `\ddot v_t` and `\ddot S_t`.
- Expanded the solve-form Hessian into explicit solved-vector,
  log-determinant, innovation-linear, and quadratic pieces.
- Added second-order precision, gain, filtered mean, and filtered covariance
  recursions for covariance-form audit parity.
- Added sign-convention and Hessian-ready backend contract sections linking the
  chapter to observed information and the later SVD sigma-point bridge.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported the
  PDF up to date after the chapter edits had been compiled.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches.
- `git diff --check` passed.

Audit:
- Chapter 10 now supplies the second-order counterpart of the Chapter 9 score
  scan, rather than only a terminal Hessian contraction.
- The chapter explicitly separates the model-specific derivative objects from
  the backend-neutral solve-form likelihood Hessian.
- Evidence language remains cautious: the formulas are source-backed contracts
  and validation targets, while production readiness still requires numerical
  parity on each backend.
- Next phase remains justified: Chapter 12 can now state factor-derivative
  contracts as ways to reproduce the covariance, score, and Hessian objects
  defined in Chapters 9 and 10.

### Phase AD-4: factor-derivative chapter

Phase plan:
- expand Chapter 12 from a short factor warning into a mathematical factor
  contract;
- add universal first- and second-order reconstruction equations for
  differentiated factors;
- restate the Cholesky and QR first-/second-derivative formulas under explicit
  branch conventions;
- connect factor derivatives back to Kalman prediction, innovation, update,
  score, and Hessian parity;
- compile, scan references, check whitespace, and audit for hidden
  regularization claims.

Execution:
- Updated `docs/chapters/ch12_factor_derivatives.tex`.
- Added the universal reconstruction identities
  `\dot A_\star=\dot C C^\top+C\dot C^\top` and
  `\ddot A_\star=\ddot C C^\top+C\ddot C^\top+\dot C_i\dot C_j^\top+\dot C_j\dot C_i^\top`.
- Added lower-triangular Cholesky derivative formulas tied to source labels
  `eq:first_cholesky_derivative` and `eq:second_cholesky_derivative`.
- Added thin-QR derivative formulas tied to source labels
  `eq:first_qr_r_derivative` and `eq:second_qr_r_derivative`.
- Added Kalman stack examples for prediction, innovation, and Joseph filtered
  covariance factors.
- Expanded the spectral-factor section to emphasize that SVD/eigen factors
  reconstruct the implemented covariance `P_+`, not automatically the
  pre-regularized covariance `P`.
- Added backend parity gates covering value reconstruction, derivative
  reconstruction, solve parity, branch reporting, and cross-backend parity.

Tests:
- MathDevMCP label lookups for the Cholesky and QR source equations found the
  expected MacroFinance neighborhoods.
- ResearchAssistant privacy status confirmed the local read-only/offline
  workflow remains in effect.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed after
  the expected reruns for new labels.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches after the final rerun.
- `git diff --check` passed.

Audit:
- Chapter 12 now gives mathematical checks for what a differentiated factor
  means, rather than relying on verbal stability claims.
- The Cholesky and QR formulas are clearly limited to smooth fixed-rank,
  fixed-sign branches.
- The spectral section now makes the implemented/floored covariance target
  explicit, which prevents mistaking numerical regularization for the original
  model law.
- Next phase remains justified: Chapter 11 should define how structural models
  provide the reduced-form derivatives consumed by Chapters 9, 10, and 12.

### Phase AD-5: structural-derivative chapter

Phase plan:
- expand Chapter 11 from a short provider checklist into the upstream
  structural derivative contract for the filtering scan;
- include raw/transformed parameter chain rules;
- add initial-state derivative handling, including stationary initialization
  equations;
- distinguish structural zeros from unavailable derivative entries;
- connect observation masks and provider metadata to the score/Hessian chapters;
- compile, scan references, check whitespace, and audit for ambiguous parameter
  coordinate claims.

Execution:
- Updated `docs/chapters/ch11_structural_derivatives.tex`.
- Added the structural provider map
  `g(\psi)=ve(c,F,Q,a,H,R,m_0,P_0)`, extending the MacroFinance
  `(b,F,Q,a,H,R)` source map with initial moments.
- Added first- and second-order provider contracts for all reduced-form blocks.
- Added transformed-parameter chain rules for first and second derivatives.
- Added fixed, stationary, and estimated initial-condition policies, including
  stationary mean and covariance derivative equations.
- Added sections on structural zeros, unavailable blocks, observation masks,
  shape metadata, parameter units, and provider validation.

Tests:
- MathDevMCP label lookups found the MacroFinance structural map and first- and
  second-derivative provider contract labels.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed after
  the expected reruns for new labels.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches after the final rerun.
- `git diff --check` passed.

Audit:
- Chapter 11 now defines the derivative objects that enter Chapters 9, 10, and
  12, including the parameter coordinate system in which they are valid.
- The initial-state policy is now explicit, which closes a common gap in
  filtering derivative derivations.
- The mask convention is tied to the score chapter and keeps masks as data-side
  selections rather than parameters.
- Next phase remains justified: custom-gradient wrappers and validation can now
  be written against explicit structural, innovation, Hessian, and factor
  contracts.

### Phase AD-6: custom-gradient and validation chapters

Phase plan:
- expand Chapter 13 so custom gradients are tied to the exact same scalar,
  branch, masks, parameter transforms, and regularization policy as the value
  path;
- expand Chapter 14 into a concrete validation ladder with comparison targets,
  tolerances, backend-specific gates, and required artifacts;
- keep HMC claims separated from deterministic value/gradient/Hessian
  certification;
- compile, scan references, check whitespace, and audit for surrogate-gradient
  ambiguity.

Execution:
- Updated `docs/chapters/ch13_custom_gradient_wrappers.tex`.
- Added the same-scalar contract
  `g_i(\psi)=\partial\mathcal{L}(\psi)/\partial\psi_i`.
- Added diagnostic requirements for structural provider, masks, covariance or
  factor backend, jitter, PSD projection, spectral floors, pivots, rank
  truncation, sign/order corrections, and finite status.
- Added a separate Hessian/observed-information path for mass matrices and
  diagnostics, distinct from the HMC hot path.
- Added static-shape and HMC safety label sections.
- Updated `docs/chapters/ch14_derivative_validation.tex`.
- Added explicit score/Hessian comparison targets, a detailed validation ladder,
  finite-difference step-size guidance, backend-specific failure modes, and a
  validation artifact schema.

Tests:
- ResearchAssistant local search returned no paper-summary hits for the exact
  Kalman/SVD derivative-validation cluster, so the chapter remains grounded in
  local MacroFinance evidence and BayesFilter contracts.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed after
  the expected reruns for new labels.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches after the final rerun.
- `git diff --check` passed.

Audit:
- The custom-gradient chapter now rules out the major HMC risk: evaluating one
  scalar while returning the derivative of a nearby surrogate or hidden
  regularized branch.
- The validation chapter now says exactly what is compared and why backend
  parity only applies when backends target the same law.
- HMC smoke tests are explicitly demoted to sampler-usability evidence unless
  deterministic value/gradient/Hessian checks have already passed.
- Next phase remains justified: the SVD sigma-point bridge can now be audited
  against the completed Chapters 9--14 and the full monograph can be checked.

### Phase AD-7: SVD bridge and whole-document audit

Phase plan:
- audit Chapter 18 against the completed analytic-derivative stack in
  Chapters 9--14;
- add a small bridge only if the SVD score/Hessian discussion does not clearly
  connect structural derivatives, factor derivatives, custom gradients, and
  validation;
- run the full document compile, warning scan, source-map parse, whitespace
  check, and git-status audit;
- decide whether final commit remains justified.

Execution:
- Reviewed `docs/chapters/ch18_svd_sigma_point.tex`.
- Added a bridge paragraph stating how the SVD sigma-point derivative recursion
  depends on Chapter 11 structural derivatives, Chapter 12 factor
  reconstruction, Chapter 13 same-scalar custom gradients, and Chapter 14
  validation gates.
- Confirmed the chapter already states that the SVD/UKF likelihood score and
  Hessian are the Kalman solve-form expressions applied to sigma-point
  innovation moments, and that spectral factors reconstruct the implemented
  `P_+`.

Tests:
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map_yaml_ok')"`
  passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches.
- `git diff --check` passed.
- `git status --short` showed the analytic-derivative chapter files and main
  reset memo/source map as modified, plus unrelated/untracked plan/PDF/template
  files that should not be included in this commit.

Audit:
- The SVD bridge is now tied to the whole derivative stack rather than only to a
  local SVD formula.
- No compilation, reference, YAML, or whitespace blocker remains.
- The next step remains justified: update this reset memo with final
  interpretation, stage only the relevant analytic-derivative pass files, and
  commit them.

### Final completion: analytic-derivative documentation pass

Final result:
- The analytic-derivative chapter set is now complete at the documentation
  contract level.  Chapters 9--10 define first- and second-order Kalman
  prediction, innovation, score, gain, update, and solve-Hessian recursions.
  Chapter 11 defines the structural provider layer.  Chapter 12 defines
  triangular and spectral factor derivative contracts.  Chapters 13--14 define
  same-scalar custom-gradient and validation requirements.  Chapter 18 now
  points the SVD sigma-point Hessian discussion back to the whole stack.
- The pass deliberately does not claim production HMC readiness for every
  backend.  It defines the mathematical and diagnostic obligations that must be
  tested before such a claim is valid.

Final tests:
- Full monograph compile:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Warning scan:
  `rg -n "undefined|Citation.*undefined|Reference.*undefined|multiply defined|LaTeX Warning" docs/main.log`
  returned no matches.
- Source-map parse:
  `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map_yaml_ok')"`
  passed.
- Whitespace audit:
  `git diff --check` passed.

Final audit:
- The plan order was justified and no phase exposed a blocker requiring human
  direction.
- ResearchAssistant remains local/offline and has no local summary hits for the
  exact SVD/eigen derivative-validation cluster; spectral derivative claims
  remain framed as mathematical contracts and validation targets.
- MathDevMCP label lookups support provenance for the MacroFinance score,
  Hessian, Cholesky, QR, and structural-provider labels, but manual derivation
  and numerical parity remain the proof obligations.
- Unrelated SGU plan/provenance work and pre-existing structural-SVD memo
  changes were observed in the working tree.  The analytic-derivative commit
  should stage only the files in this pass and leave those unrelated changes
  untouched.

Recommended next hypotheses:
- H1: On small affine Gaussian systems, the completed SVD sigma-point derivative
  recursion should match the exact Kalman score and Hessian to finite-difference
  tolerance when the spectral factor reconstructs the same covariance branch.
- H2: On small nonlinear systems, the analytic sigma-point score/Hessian should
  match finite differences of the same implemented SVD sigma-point likelihood,
  with mismatches concentrated near spectral-gap, floor, or branch events.
- H3: The factor reconstruction residuals in Chapter 12 should predict
  likelihood-gradient discrepancies better than raw finite-gradient status near
  singular covariance regimes.
- H4: Same-scalar custom-gradient diagnostics should catch every intentional
  jitter, floor, PSD projection, pivot, rank truncation, sign/order correction,
  or fallback that changes the derivative target before HMC is allowed to run.
- H5: Stationary initial-condition derivatives are a likely remaining source of
  score/Hessian mismatch in models that currently validate only fixed initial
  moments.
- H6: Masked mixed-frequency panels should pass derivative parity when masks
  are treated as data-side row selections, but should fail or change target if
  dynamic shape-changing mask implementations are used inside compiled scans.
- H7: Production promotion should proceed backend by backend: covariance/solve
  first, Cholesky/QR factor derivatives second, SVD/spectral sigma-point
  derivatives only after gap/floor/branch validation artifacts exist.

### CUT higher-order sigma-point addendum

Follow-up request:
- document the experimental conjugate unscented transform filter in
  `/home/chakwong/python/src/dsge_hmc/filters/CUTSRUKF.py`;
- cite the original CUT papers;
- answer whether CUT can be combined with SVD for numerical stability;
- answer whether analytic gradients and Hessians can be derived.

Execution:
- Added CUT references to `docs/references.bib`.
- Added Section~`Conjugate Unscented Transform Rules` to
  `docs/chapters/ch16_sigma_point_filters.tex`.
- Added Section~`CUT With Square-Root Backends` to
  `docs/chapters/ch17_square_root_sigma_point.tex`.
- Added Section~`CUT-SVD Score and Hessian` to
  `docs/chapters/ch18_svd_sigma_point.tex`.

Interpretation:
- CUT is a higher-order sigma-point rule, not a covariance factorization.
  The experimental implementation uses CUT4-G points with a square-root QR
  backend and rank-deficient process-noise factors.
- CUT can be combined with SVD by using an SVD/eigen factor
  `C=U Lambda_+^{1/2}` to place the CUT points.  This can improve value-side
  robustness for singular or nearly singular covariances, but it does not change
  CUT's quadrature order and it inherits SVD branch/gap/floor derivative risks.
- Analytic gradients and Hessians are derivable by replacing the UKF offsets in
  the SVD sigma-point derivative recursion with fixed CUT offsets and weights.
  The cost scales with the CUT point count, for CUT4-G `2n+2^n`, and therefore
  is most plausible on low-dimensional declared stochastic blocks.

Next phase justified?
- Yes for documentation and small validation fixtures.
- No for production HMC promotion until CUT moment tests, CUT-vs-UT nonlinear
  benchmarks, factor reconstruction, SVD branch diagnostics, and same-scalar
  gradient/Hessian finite-difference checks exist.

## 2026-05-03 update: DSGE structural deterministic dynamics warning

User raised a serious modeling-design issue: nonlinear DSGE filters must
separate shock-driven exogenous states from endogenous/predetermined or
accounting states that are deterministic conditional on lagged states and
current shocks.  The source DSGE SVD sigma-point adapter separates pruned
first-order and second-order components, but it does not expose an explicit
exogenous/endogenous state partition inside the first-order block.  This is
mostly harmless for the smallest NK model whose state vector is only
exogenous shocks, but it is a structural filtering bug/risk for Rotemberg NK,
SGU, EZ, and NAWM-class models when using nonlinear sigma-point or particle
filters.

Documentation action:
- Added a dedicated monograph chapter:
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`.
- Added the chapter to `docs/main.tex` after the SVD sigma-point chapter.
- Added a state-space contract cross-reference in
  `docs/chapters/ch02_state_space_contracts.tex`.
- Added bibliography entries for Herbst--Schorfheide, An--Schorfheide,
  Kim--Kim--Schaumburg--Sims, Andreasen--Fernandez-Villaverde--Rubio-Ramirez,
  Gordon--Salmond--Smith, and Doucet--de Freitas--Gordon.
- Updated `docs/source_map.yml` with provenance for this chapter.

Interpretation:
- This is not a minor numerical patch issue.  It is a release-gating design
  contract issue for BayesFilter DSGE backends.
- Next implementation work should add explicit structural DSGE transition
  metadata and tests before any nonlinear DSGE filter is promoted as an HMC
  target.

Follow-up planning action:
- Added `docs/plans/dsge-structural-filtering-refactor-plan-2026-05-03.md`.
- The plan requires re-deriving the DSGE filtering equations with explicit
  exogenous/endogenous state separation before any filter rewrite.
- The plan then audits all filtering paths, introduces a structural filtering
  contract, rewrites or gates unsafe nonlinear DSGE paths, and adds tests for
  metadata, manifold preservation, linear recovery, toy nonlinear DSGE
  behavior, Rotemberg NK, SGU, XLA, gradients, and HMC smoke gates.

## 2026-05-04 update: explicit structural UKF versus full-state UKF contrast

User asked for a summary of the recent Chapter 18b changes and noted a real
remaining weakness: the structural UKF example still did not contrast sharply
enough with what a reader would treat as a normal generic full-state UKF.

### Focused UKF contrast pass

Phase plan:
- Add explicit sections that contrast the structural UKF with a naive or
a generic additive-noise full-state UKF pattern.
- Make the difference operational, not only conceptual.
- Keep the new material consistent with Chapters 16, 20, and 32.

Execution:
- Added a new section:
  - `Structural UKF Versus a Generic Full-State UKF Pattern`
- Added a compact comparison table that contrasts:
  - integration space;
  - independent new randomness;
  - deterministic-block treatment;
  - support preservation;
  - target law;
  - metadata label;
  - when acceptable.
- Added a new subsection:
  - `Algorithmic Contrast with a Naive Full-State UKF`
  explaining step by step how the two constructions differ operationally.
- Added a new subsection:
  - `What a Naive Full-State UKF Would Do Differently`
  tied directly to the existing toy model.
- Reused the chapter’s existing artificial-variance comparison so it now reads
  as an explicit target-law contrast rather than only a closing remark.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted search in Chapter 18b confirmed explicit use of:
  `UKF`, `approximation`, `support`, `stochastic`, and `deterministic`.

Audit:
- The chapter now answers the concrete reader questions that were still too easy
  to miss after the previous pass:
  - what the sigma-point object is in the structural UKF;
  - what a naive full-state UKF would do differently;
  - why that changes the target law;
  - when the full-state version is only an approximation.
- The new material does not claim that all full-state UKFs are invalid.  It
  states instead that, for this mixed structural class, they require an
  approximation label when they enlarge the stochastic space.
- No further immediate chapter pass is required for this specific UKF contrast
  issue.

## 2026-05-04 update: why the naive full-state UKF update fails

User then asked for one more missing piece: not just the contrast, but an
explicit section explaining why the naive full-state UKF update equation fails
once exogenous and endogenous states are separated structurally.

### Focused naive-UKF-failure pass

Phase plan:
- Add one focused section explaining why the failure is semantic, not merely a
  different sigma-point convention.
- Tie the failure directly to the separation of exogenous/stochastic and
  endogenous/deterministic-completion states.
- Make clear that the algebraic UKF update formula itself is not invalid; the
  predicted moments and cross covariances are wrong when built from the wrong
  target law.

Execution:
- Added a new subsection:
  - `Why the Naive Full-State UKF Update Fails in Mixed Structural Models`
- The new subsection explains the failure at three levels:
  - prediction-law failure;
  - cross-covariance failure;
  - update-law failure.
- It now states explicitly that, after a naive full-state additive covariance is
  imposed, the UKF gain and covariance update are computed for an altered
  Gaussian approximation rather than for the intended structural transition.
- It also states the key conclusion directly: the linear algebra is fine, but
  the target law is wrong.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted search confirmed the new section includes the intended failure
  language around prediction-law failure, cross-covariance failure, update-law
  failure, and `the target law is wrong`.

Audit:
- This directly closes the last major exposition gap in the UKF part of Chapter
  18b.
- The chapter now distinguishes clearly among:
  - structural UKF construction;
  - naive full-state UKF approximation;
  - and the specific reason the naive update fails for mixed structural models.
- No further clarification is immediately necessary for this point unless you
  want an even more formal derivation against a textbook additive-noise UKF
  notation.

## 2026-05-05 update: assumptions-and-small-lemmas refactor for the formal nonlinear-measurement section

User then asked whether some minor issues could be declared as explicit
assumptions so that the remaining proof obligations become easier for
MathDevMCP to inspect.  This was the right move.

### Focused assumptions-and-lemmas pass

Phase plan:
- separate accepted assumptions from proved identities;
- split the formal section into smaller theorem-like obligations;
- move interpretation-level conclusions out of the heaviest proof burden;
- re-run bounded MathDevMCP audits on the smaller formal labels.

Execution:
- Replaced the single larger formal block with:
  - an `Accepted assumptions` block;
  - smaller proposition-level obligations for:
    - latent predictive pushforward;
    - deterministic-completion identity;
    - observation predictive pushforward;
    - naive full-state perturbation changes the latent law;
  - a concluding proposition on failure propagation and the observation-side-only
    error regime.
- Added and reused explicit labels so the proof now has narrower anchors:
  - `ass:bf-exp-affine-factorization`
  - `eq:bf-exp-affine-latent-map`
  - `eq:bf-exp-affine-pushforward`
  - `eq:bf-exp-affine-structural-identity`
  - `eq:bf-exp-affine-observation-map`
  - `eq:bf-exp-affine-composed-map`
  - `eq:bf-exp-affine-observation-pushforward`
- Converted the unsupported `lemma` environment into supported `proposition`
  environments so the monograph build remains compatible with the current
  preamble.
- Preserved the chapter’s concrete `(m_t,k_t,\varepsilon_t)` notation throughout.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Re-ran bounded MathDevMCP audits on the smaller formal labels.

MathDevMCP audit result:
- the smaller-label audits still remain on the audit boundary, but they now
  isolate more precisely where the extraction problem lies;
- the result is still not backend certification, but the formal section is now
  cleaner and more modular for human review.

Audit:
- This refactor makes the chapter more honest mathematically: assumptions are now
  declared explicitly, exact identities are proved separately, and the failure
  interpretation is derived in a lighter final step.
- It also gives a much clearer path for any future attempt at stronger
  MathDevMCP certification.
- Remaining warning hotspots are cosmetic LaTeX line-break and heading-width
  issues, not conceptual blockers.


User then asked for both remaining follow-ups to be done together:
- refactor the proposition/proof so MathDevMCP has a better chance to audit it;
- do a chapter-local LaTeX polish pass for the recent warning hotspots.

### Focused auditability-and-polish pass

Phase plan:
- split the proposition/proof into smaller labeled obligations;
- add stable labels for the augmented variable, augmented Gaussian law, and
  structural identity;
- shorten or sanitize local headings and reduce comparison-table warning load;
- re-run the bounded MathDevMCP audit and record the new boundary honestly.

Execution:
- Renamed the subsection heading to plain text:
  - `Why the Sigma-Point Variable Uses Pre-Transition Uncertainty`
- Shortened the proposition title to:
  - `Predictive pushforward and sigma-point space`
- Refactored the proposition into explicit numbered obligations and labeled
  equations:
  - `eq:bf-structural-ukf-prop-m`
  - `eq:bf-structural-ukf-prop-k`
  - `eq:bf-structural-ukf-prop-state`
  - `eq:bf-structural-ukf-prop-map`
  - `eq:bf-structural-ukf-prop-factorization`
  - `eq:bf-structural-ukf-prop-pushforward`
  - `eq:bf-structural-ukf-prop-deterministic-completion`
- Added stable labels for the later UKF exposition:
  - `eq:bf-ukf-augmented-variable`
  - `eq:bf-ukf-augmented-law`
  - `eq:bf-ukf-structural-identity`
- Replaced the old math-mode comparison array with a text-mode tabular using
  shorter cell text and a shorter right-column header.
- Re-ran the bounded MathDevMCP audit on
  `prop:bf-structural-ukf-pushforward`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Bounded MathDevMCP audit still returned `unverified`, but the obligation count
  tightened from the earlier larger diagnostic bundle to a smaller set:
  - total obligations: 3
  - unverified: 2
  - inconclusive: 1
- The previous `source_label_missing` issue disappeared after the refactor.

Audit:
- The proposition is now more readable and more structurally auditable.
- MathDevMCP still does not certify the proof, but the result is cleaner: the
  remaining boundary is now primarily `manual_formalization_required` rather than
  a larger mix of ambiguity and missing-source issues.
- The local hyperref warning from raw math in the subsection heading was removed
  by the plain-text heading rewrite.
- Remaining warning hotspots are now mostly cosmetic table/line-break issues and
  do not undermine the logic of the chapter.


User then asked for a formal proposition and proof explaining why the structural
UKF uses `(x_{t-1},\varepsilon_t)` as the sigma-point variable even though the
filtered state remains `x_t`, and explicitly asked that the proof be audited by
MathDevMCP if possible.

### Focused proposition-and-proof pass

Phase plan:
- add a bounded proposition and proof before the worked UKF example;
- keep the proof in the toy-model notation `(m_t,k_t,\varepsilon_t)`;
- state formally that the predictive law of `x_t` is the pushforward of the
  joint law of `(x_{t-1},\varepsilon_t)`;
- use MathDevMCP for a bounded derivation audit and record its verification
  boundary honestly.

Execution:
- Added a new subsection:
  - `Why the Sigma-Point Variable is $(x_{t-1},\varepsilon_t)$`
- Added Proposition `prop:bf-structural-ukf-pushforward` stating that under the
  structural transition, the one-step predictive law of `x_t` is the pushforward
  of the joint law of `(x_{t-1},\varepsilon_t)` under the structural map.
- Added a proof using a bounded test function `\varphi` and the conditional
  expectation identity
  \[
    \E[\varphi(x_t)\mid y_{1:t-1}]
    =
    \iint
    \varphi(F(x_{t-1},\varepsilon_t))
    p(\varepsilon_t)
    p(x_{t-1}\mid y_{1:t-1})
    \, d\varepsilon_t \, dx_{t-1}.
  \]
- Added an interpretation paragraph making explicit that:
  - the filtered state remains `x_t`;
  - `(x_{t-1},\varepsilon_t)` is the sigma-point / integration variable because
    it generates the predictive law;
  - omitting `\varepsilon_t` misses genuine current-shock uncertainty, while
    directly perturbing `k_t` changes the law.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Ran a bounded MathDevMCP audit on
  `prop:bf-structural-ukf-pushforward`.

MathDevMCP audit result:
- status: `unverified`
- reason: at least one obligation remains unverified or diagnostic-only.
- high-priority actions reported by the tool include:
  - human formalization/review for typed obligations;
  - splitting or rewriting an ambiguous derivation row.

Audit:
- The proposition materially improves the chapter by separating the filtered
  state `x_t` from the sigma-point variable `(x_{t-1},\varepsilon_t)` in a
  formally stated way.
- MathDevMCP did not certify the proof.  Its result is still useful: it marks a
  verification boundary and indicates that the proof is currently at the level
  of human-readable mathematical derivation rather than machine-verified
  obligation closure.
- This is acceptable for the chapter as long as the text does not overclaim that
  the proposition has been backend-certified.  It should be presented as a
  bounded chapter-level derivation with explicit audit limitation.


While I was finishing the derivation pass, the user raised one more important
clarity gap: the numerical illustration gave structural values and a naive
likelihood delta, but it still did not follow through with the two different
computed update objects in a way that visibly tied the numbers back to the three
failure claims.

### Focused numerical tie-back pass

Phase plan:
- compute and display the naive full-state predictive/update quantities beside
  the structural ones;
- tie those numbers directly to prediction-law failure, cross-covariance
  failure, and update-law failure.

Execution:
- Expanded the numerical-contrast paragraph after the artificial-variance
  example so it now writes the naive full-state quantities explicitly:
  - `\tilde P_{xx,t}`
  - `\tilde S_t`
  - `\tilde P_{xz,t}`
  - `\tilde K_t`
  - `\tilde x_{t|t}`
- Tied those numbers directly to the three failures:
  - prediction-law failure via the changed predictive covariance;
  - cross-covariance failure via the enlarged second component of
    `\tilde P_{xz,t}`;
  - update-law failure via the changed gain and posterior correction.
- Verified the naive-gain and posterior-mean numbers numerically before writing
  them into the chapter.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Independent numeric check confirmed:
  - `\tilde K_t = (0.42259, 0.19408)^\top`
  - `\tilde x_{t|t} = (0.08019, 0.14707)^\top`

Audit:
- The numerical illustration now does what the prose claims: it shows two
  different computed update objects, not only two different likelihood values.
- This makes the three failures visible both algebraically and numerically.


User then asked for the three named failures to be derived explicitly rather
than only described in words.

### Focused derivation pass

Phase plan:
- derive the structural predictive law explicitly in the toy model’s notation;
- show why the naive full-state sigma-point cloud represents a different
  predictive law;
- derive explicitly why the cross covariance changes;
- derive explicitly why the same algebraic update formula then updates the wrong
  approximate model.

Execution:
- Expanded the UKF-failure passage so it now derives:
  - the structural predictive law from
    `m_t = \rho m_{t-1} + \sigma \varepsilon_t` and
    `k_t = \phi k_{t-1} + \gamma m_t^2`;
  - the structural identity
    `k_t - \phi k_{t-1} - \gamma m_t^2 = 0` as the support condition for the
    intended one-step law;
  - the prediction-law failure via propagated naive points
    `\tilde x_t^{(j)} = (\tilde m_t^{(j)}, \tilde k_t^{(j)})` that in general do
    not satisfy that identity;
  - the cross-covariance failure through the difference between
    `P_{xz,t}^{\mathrm{struct}}` and `P_{xz,t}^{\mathrm{naive}}`, with explicit
    explanation that the naive construction introduces covariance terms coming
    from artificial perturbations of the deterministic-completion coordinate;
  - the update-law failure by showing that once
    `(\hat x_{t|t-1}, S_t, P_{xz,t})` come from the altered predictive law, the
    gain `K_t` itself changes and the standard UKF update is then updating the
    wrong approximate model.
- Connected the update-law derivation directly to the existing numerical
  comparison where the artificial variance changes `S_t` and the likelihood.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- A targeted file-content check confirmed the chapter now contains:
  - `Prediction-law failure`
  - `Cross-covariance failure`
  - `Update-law failure`
  - `P_{xz,t}^{\mathrm{struct}}`
  - `P_{xz,t}^{\mathrm{naive}}`
  - `k_t - \phi k_{t-1} - \gamma m_t^2`

Audit:
- The chapter now proves, within the toy-model notation, why the three failures
  are genuine failures and not only labels.
- The remaining limitation is still one of scope, not clarity: this is a bounded
  derivation for the toy model and the associated structural-design lesson, not
  a universal theorem about every possible UKF construction.


User then pointed out another real exposition gap: in the worked UKF example,
the augmented variable includes the current shock `\varepsilon_t`, but the text
did not yet explain clearly enough why this is necessary and how it differs from
a naive post-transition full-state augmentation.

### Focused epsilon-augmentation clarification pass

Phase plan:
- Explain why the augmentation with `\varepsilon_t` is not arbitrary padding.
- Tie the augmentation directly to the predictive law and to the distinction
  between genuine new uncertainty and deterministic completion.
- Contrast omission of `\varepsilon_t` with direct perturbation of `k_t`.

Execution:
- Expanded the paragraph before the augmented UKF variable so it now explains:
  - the predictive law is generated by the joint uncertainty in
    `(m_{t-1},k_{t-1},\varepsilon_t)`;
  - once those are fixed, both `m_t` and `k_t` are fixed by the structural map;
  - `\varepsilon_t` is therefore included because it is a genuine pre-transition
    uncertainty variable;
  - omitting `\varepsilon_t` would miss current shock uncertainty;
  - directly perturbing `k_t` instead would define a different transition law.
- Added an explicit remark that standard additive-noise UKF presentations also
  augment with process noise for this same structural reason: sigma points must
  live on the variables that generate the one-step uncertainty.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted search confirmed the revised paragraph now explicitly addresses:
  - `pad the state arbitrarily`;
  - `pre-transition uncertainty variables`;
  - omission of `\varepsilon_t`;
  - and direct perturbation of `k_t`.

Audit:
- The UKF example now motivates the augmented variable much more clearly.
- The reader no longer has to infer why `\varepsilon_t` appears in the sigma-point
  object.
- This closes another substantive readability gap in the example.


User then asked to make the contrast even sharper by writing the naive and
structural UKF equations more explicitly.

### Focused sharpened-contrast pass

Phase plan:
- Add side-by-side structural versus naive UKF prediction/update equations.
- Make the wrong object in the naive cross covariance explicit.
- Add one simple structural identity test that distinguishes the two
  constructions point by point.

Execution:
- Added a `Side-by-side prediction/update contrast` paragraph in Chapter 18b.
- Wrote the structural sigma-point propagation and the structural
  cross-covariance formula `P_{xz,t}^{\mathrm{struct}}` explicitly.
- Wrote the naive full-state additive approximation and the altered
  cross-covariance formula `P_{xz,t}^{\mathrm{naive}}` explicitly.
- Added a direct explanation that once the altered `P_{xz,t}` is used in the
  standard UKF gain/update equations, the update is correcting the wrong
  approximate model.
- Added a `Structural identity test` based on
  `k_t - \phi k_{t-1} - \gamma m_t^2 = 0` to show exactly where the structural
  and naive propagated sigma points part company.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted search confirmed the presence of the sharpened UKF contrast material,
  including the structural identity test and the `wrong transition law` /
  `target law is wrong` explanation.

Audit:
- The UKF part of the chapter is now materially sharper.
- A careful reader can now see not only that the naive full-state UKF is a
  different approximation, but exactly where the wrong prediction and
  cross-covariance objects enter the update.
- This is probably sufficient unless you want a separate appendix-level formal
  proposition.

## 2026-05-04 update: thorough Chapter 18b audit and cleanup pass

User asked for a thorough audit/cleanup pass on Chapter 18b after the recent
incremental additions.

### Focused cleanup pass

Phase plan:
- preserve the stronger mathematics;
- reduce duplicated UKF contrast/failure material;
- smooth transitions;
- standardize terminology where possible;
- reduce overlap with generic policy chapters.

Execution:
- Folded the earlier standalone UKF contrast section into the worked-example
  section so the chapter now reaches the toy model through one continuous
  progression instead of separate stitched blocks.
- Removed the repeated standalone `Why the Naive Full-State UKF Update Fails`
  and `What a Naive Full-State UKF Would Do Differently` subsection structure as
  separate exposition blocks, while preserving their strongest content:
  - semantic failure explanation;
  - wrong-target-law diagnosis;
  - side-by-side prediction/update contrast;
  - structural identity test.
- Kept the comparison table, numerical illustration, and off-manifold/full-state
  approximation contrast, but made them read as one sequence instead of repeated
  variants of the same point.
- Preserved the chapter’s concrete `(m_t,k_t)` notation throughout the UKF
  discussion.
- Left the DSGE-specific structural tests and policy sections in place, but the
  UKF part now relies less on repeated restatement and more on a single clean
  argument.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted phrase searches showed reduced duplication in the UKF portion.
- Internal grep confirmed that the removed UKF subsection labels no longer
  remain in the file.

Audit:
- The chapter now reads more like one developing argument and less like several
  accreted notes.
- The mathematical content was preserved while the strongest duplication was
  reduced.
- The biggest remaining issues are cosmetic LaTeX table/line-break warnings, not
  structural clarity or notation drift.


User pointed out a real notation bug: in the naive-UKF-failure section I had
momentarily switched from the chapter’s concrete example notation `(m_t,k_t)` to
the generic contract notation `(s_t,d_t)`.  That weakened readability in exactly
the place where the chapter is trying to be most concrete.

### Focused notation-fix pass

Phase plan:
- Restore consistency with the toy example notation.
- Keep the generic BayesFilter structural language in the surrounding prose, but
  write the failure subsection itself in the chapter’s explicit `(m_t,k_t)`
  notation.

Execution:
- Rewrote the opening of the `Why the Naive Full-State UKF Update Fails in Mixed
  Structural Models` subsection so it now consistently uses:
  - exogenous state `m_t`
  - endogenous state `k_t`
  - transition maps `T_m` and `T_k`
- Removed the temporary switch to `s_t`, `d_t`, and `\operatorname{pack}(s_t,d_t)`
  in that subsection.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.

Audit:
- The notation is now locally consistent with the worked UKF example.
- The subsection remains connected conceptually to the generic structural-state
  contract, but it no longer forces the reader to translate notation midstream.


User requested a full autonomous execution pass for
`docs/chapters/ch18b_structural_deterministic_dynamics.tex`, including another
full read-through, tool-assisted checking where useful, phase-by-phase reset
memo updates, final provenance closure, commit, and a detailed completion
summary.

### Phase C18b-0: baseline and evidence check

Phase plan:
- Re-read Chapter 18b and the governing structural-contract and API chapters.
- Re-read nearby sigma-point, particle, filter-choice, and production-checklist
  chapters for consistency constraints.
- Re-check whether MathDevMCP and research-assistant materially help.
- Incorporate the independent plan-audit findings into the executable Chapter
  18b plan before prose edits begin.

Execution:
- Re-read:
  - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`
  - `docs/chapters/ch02_state_space_contracts.tex`
  - `docs/chapters/ch04_bayesfilter_api.tex`
  - `docs/chapters/ch16_sigma_point_filters.tex`
  - `docs/chapters/ch19_particle_filters.tex`
  - `docs/chapters/ch32_production_checklist.tex`
- Re-checked the local-tool role:
  - MathDevMCP is useful for bounded equation/derivation audits and assumption
    surfacing, but it is not sufficient to certify the whole conceptual DSGE
    filtering argument on its own.
  - research-assistant is useful only if strengthened literature-backed claims
    require section-level citation support; it is not the primary tool for this
    chapter’s internal derivation cleanup.
- Performed an independent plan audit and found the prior chapter plan was
  mathematically good but under-specified for BayesFilter workflow discipline.
- Tightened
  `docs/plans/ch18b-structural-deterministic-dynamics-revision-plan-2026-05-04.md`
  to add:
  - contract/notation reconciliation;
  - nearby-chapter consistency checks;
  - provenance classification;
  - semantic claim audits;
  - required-tests section audit;
  - explicit phase-by-phase reset-memo obligations.
- Parsed `docs/source_map.yml`; YAML remains valid.

Tests:
- `python -c "import yaml; yaml.safe_load(...)"` returned `source_map yaml ok`.

Audit:
- The rewrite pass remains justified.
- No blocking provenance or cross-chapter inconsistency has been found yet.
- The main issue at this phase was execution-plan rigor, not chapter invalidity.
- Phase C18b-1 remains justified.

### Phase C18b-1: contract and notation reconciliation

Phase plan:
- Align Chapter 18b vocabulary with the structural state-partition and API
  metadata chapters before deeper mathematical edits.
- Make Chapter 18b explicitly use BayesFilter contract language without losing
  DSGE readability.
- Add only bounded cross-references and terminology cleanup at this phase.

Execution:
- Updated the chapter opening to connect the DSGE `m_t`/`k_t` split to the
  structural partition language from
  `docs/chapters/ch02_state_space_contracts.tex`.
- Reframed the endogenous block as a deterministic-completion block and the
  exogenous block as the declared stochastic block.
- Added cross-references to:
  - Chapter~\ref{ch:bf-state-space-contracts}
  - Chapter~\ref{ch:bf-api-design}
- Tightened the linear-to-nonlinear transition warning so it refers explicitly
  to integration-space metadata rather than only state dimension.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- A targeted terminology search in Chapter 18b confirmed visible use of:
  `stochastic`, `deterministic-completion`, `approximation`, and
  `integration space`.

Audit:
- Contract ambiguity was reduced without changing the chapter’s core claim.
- Chapter 18b now reads more clearly as a downstream application of the generic
  BayesFilter contract, not as a DSGE-only exception.
- No broader restructuring was required.
- Phase C18b-2 remains justified.

### Phase C18b-2: predictive law and inherited randomness

Phase plan:
- Make the nonlinear target law explicit.
- Clarify that deterministic completion means no independent innovation, not
  zero one-step predictive variance.
- Reuse Chapter 2’s lag-stack lesson by reference where that avoids redundant
  generic exposition.

Execution:
- Added an explicit predictive-law display in Chapter 18b that writes the
  nonlinear one-step predictive distribution as the pushforward of the joint law
  over lagged state and current stochastic input.
- Clarified in prose that ``deterministic'' means fixed once
  `(x_{t-1},\varepsilon_t)` is given, not zero predictive variance conditional
  only on the lagged filtering state.
- Added a short inherited-randomness clarification tied to the Chapter 2 AR(2)
  lag-stack example and stated explicitly that `\Var(k_t\mid x_{t-1}) > 0`
  can hold even without an independent innovation in `k_t`.
- Used MathDevMCP in a bounded way on revised labels; it returned `unverified`,
  which is treated as an evidence limit rather than silent support.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted semantic search in Chapter 18b confirmed explicit coverage of
  `variance`, `innovation`, `deterministic`, and `noise`.
- Bounded MathDevMCP audit on `eq:bf-m-transition` and
  `eq:bf-ukf-example-m` returned `STATUS unverified`.

Audit:
- The main conceptual ambiguity about deterministic completion versus inherited
  randomness is now addressed directly in the chapter.
- No stronger literature claim was introduced; the new material is best
  interpreted as local BayesFilter derivation and contract restatement.
- MathDevMCP abstention does not block the rewrite because the chapter change is
  explanatory rather than a new theorem-strength claim.
- Phase C18b-3 remains justified.

### Phase C18b-3: linear exactness versus nonlinear approximation boundary

Phase plan:
- Tighten the linear Kalman section so it states precisely when the collapsed
  representation is acceptable.
- Separate exact degenerate conditional Gaussian laws from numerical
  regularization and from model-changing artificial noise injection.
- Audit wording such as `exact`, `singular`, `regularization`, `noise`, and
  `correct` for overclaim.

Execution:
- Rewrote the linear Kalman section to say that singular or nearly singular
  `Q` is acceptable when the collapsed linear representation preserves the same
  conditional law induced by the structural transition.
- Clarified that the key point is not that structural determinism disappears,
  but that the exact linear-Gaussian path still encodes the same one-step
  conditional Gaussian law.
- Added an explicit three-way distinction in the degenerate-transition section:
  - exact degenerate transition law;
  - numerical regularization for stable evaluation;
  - altered transition law with new stochastic degrees of freedom.
- Kept the SVD warning separate: value-side robustness does not repair the
  wrong target law.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Semantic search on `exact`, `singular`, `regularization`, `noise`, and
  `correct` in Chapter 18b found the intended tightened phrasing only.

Audit:
- The chapter now distinguishes structural exactness from numerical expedients
  more sharply.
- No blocking source-discipline problem appeared; the revised claims remain a
  conservative restatement of standard linear-Gaussian and BayesFilter contract
  logic.
- The remaining issue is expository cleanup in the nonlinear sigma-point,
  particle, and UKF sections, not a derivational blocker.
- Phase C18b-4 remains justified.

### Phase C18b-4: nonlinear sigma-point, particle, and UKF cleanup

Phase plan:
- State the nonlinear integration-space choice explicitly.
- Replace overly geometric wording with support-based wording where needed.
- Add a compact particle-filter propagation/weighting formula.
- Upgrade the UKF example so it teaches the reusable structural-design rule
  without overclaiming a universal UKF theorem.

Execution:
- Rewrote the nonlinear-filter section to identify the integration-space choice
  as the filter-metadata decision.
- Replaced the strongest `off-manifold` wording with language about the
  `model-implied support of the structural transition` and tied the warning to
  the approximation boundary in Chapter 16.
- Added a compact particle-filter propagation/weighting formula in structural
  notation and clarified that deterministic coordinates should be completed
  after sampling the declared stochastic block.
- Added a preface to the UKF example stating the BayesFilter design lesson:
  sigma points belong in the declared pre-transition uncertainty variables.
- Added a closing UKF summary paragraph stating that this is a structural-design
  rule and approximation-label boundary for mixed structural models, not a
  universal theorem about every UKF formulation.
- Checked consistency against Chapters 16, 18, 19, and 20.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted semantic search in Chapter 18b confirmed intended use of
  `support`, `proposal`, and `approximation` and reduced reliance on
  `manifold` wording.

Audit:
- Chapter 18b is now better aligned with the sigma-point, particle, and
  filter-choice chapters.
- The remaining `off-manifold` comparison in the toy approximation paragraph is
  tolerable because it is clearly labeled as an approximation contrast rather
  than the main explanatory device.
- No nearby-chapter consistency blocker was found.
- Phase C18b-5 remains justified.

### Phase C18b-5: required tests, misunderstandings, and policy language

Phase plan:
- Audit the chapter’s strongest policy section against the production-checklist
  and nonlinear-backend chapters.
- Replace informal claim language with BayesFilter claim-discipline language
  where appropriate.
- Add only a compact misunderstandings subsection that adds value without
  duplicating Chapter 2 or Chapter 32.

Execution:
- Rewrote the required-tests introduction so it now ties Chapter 18b claims to
  the Chapter 32 labels `value-valid`, `gradient-valid`, `sampler-usable`,
  `converged`, and `production-ready`.
- Renamed the former `Manifold test` to `Constraint-support test` to match the
  chapter’s revised support-based language.
- Added a compact `Common Misunderstandings` subsection that clarifies:
  - deterministic completion versus predictive variance;
  - singular covariance versus modeling error in the exact linear case;
  - regularization versus new stochastic coordinates;
  - approximation labeling for enlarged stochastic spaces.
- Tightened the closing policy paragraph so failing the listed tests is framed
  as a limit on the strongest BayesFilter claim labels, not as a blanket claim
  that the backend is useless.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted claim-language search in Chapter 18b showed the intended new uses of
  `converged`, `production-ready`, and `support`, with no new `guarantee` or
  `must always` overclaim.

Audit:
- The chapter’s policy outputs are now more consistent with the formal
  production-checklist discipline.
- No text now implies that the listed tests already exist as completed evidence;
  they remain gate requirements.
- The compact misunderstandings subsection adds value without becoming a long
  FAQ.
- Phase C18b-6 remains justified.

### Phase C18b-6: provenance, overlap audit, and closure

Phase plan:
- Update provenance so another agent can understand the revised Chapter 18b
  support basis after a reboot.
- Audit overlap with Chapters 2, 16, 19, 20, and 32.
- Run final mechanical checks before commit preparation.

Execution:
- Updated `docs/source_map.yml` so the Chapter 18b rationale now records the
  revised BayesFilter emphasis on:
  - predictive-law statement in structural-transition form;
  - deterministic completion versus zero predictive variance;
  - alignment with contract and claim-discipline chapters.
- Added Chapter 18b provenance pointers in the source-map duplicate-group entry
  to the local revision plan, structural-state-partition plan, and nearby
  BayesFilter contract/policy chapters used in this pass.
- Audited overlap against Chapters 2, 16, 19, 20, and 32.  The current chapter
  now reads as a focused DSGE structural-filtering warning and worked example,
  while those neighboring chapters remain the generic contract/policy sources.
- Ran final `git diff --check` and final LaTeX build before commit review.

Tests:
- `python -c "import yaml; yaml.safe_load(...)"` had already passed earlier in
  the pass and the source-map structure remains valid.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported all
  targets up to date.

Audit:
- Provenance is now explicit enough for another agent to continue after a
  reboot.
- No duplication or support mismatch was found that would block commit
  preparation.
- The plan is complete enough to move to git review and commit.
- Phase C18b-7 remains justified.

### Phase C18b completion summary

Results:
- Tightened the chapter-specific execution plan.
- Revised Chapter 18b to make the target law, inherited-randomness distinction,
  exact-linear-versus-approximate boundary, nonlinear integration-space rule,
  UKF lesson, and policy/claim discipline substantially clearer.
- Updated `docs/source_map.yml` with revised provenance and support basis for
  Chapter 18b.

Interpretation:
- The chapter is still a policy chapter, not a formal theorem chapter, but it
  now states its mathematical content and approximation boundary much more
  cleanly.
- MathDevMCP remained useful only as a bounded audit tool; its `unverified`
  outputs mark an evidence boundary, not a contradiction.
- research-assistant was not needed because the pass did not expand literature
  claims beyond the current source-backed scope.

What remains unresolved or intentionally approximate:
- The chapter still does not prove a general sigma-point theorem; it states a
  BayesFilter structural-design rule for mixed structural models.
- The required-tests section remains a release-gating policy list, not evidence
  that those tests have already been implemented for every DSGE target.

Next-phase hypotheses to test:
1. A BayesFilter structural sigma-point backend that integrates only over the
   declared stochastic block should recover exact Kalman likelihoods on linear
   degenerate-transition test cases up to declared approximation tolerance.
2. The same structural contract should cover AR(p) lag stacks, DSGE
   predetermined-state transitions, and MacroFinance affine/auxiliary-state
   constructions without inventing model-specific filter forks.
3. Explicit filter metadata for integration space and approximation label should
   prevent accidental promotion of model-changing nonlinear approximations into
   HMC targets.
4. Case-study adapters for Rotemberg NK, SGU, and later EZ/NAWM targets should
   be able to document structural timing cleanly enough to satisfy the Chapter
   18b release gate without relying on dense artificial transition noise.


User reframed the documentation goal as a monograph on Bayesian estimation for
structural state-space models.  This is the right scope: SVD sigma-point
filtering and DSGE HMC are important chapters, but the common object is the
filtering and derivative infrastructure for structural state-space likelihoods
with stochastic, deterministic, auxiliary, lag, and accounting state blocks.

New planning artifacts:
- Added canonical BayesFilter implementation plan:
  `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`.
- Added bounded writing/execution plan:
  `docs/plans/bayesfilter-structural-ssm-monograph-consolidation-pass-plan-2026-05-04.md`.
- Updated the DSGE-side plan in `/home/chakwong/python/docs/plans` so it is an
  adapter/integration handoff, not the generic filter implementation source of
  truth.

Independent plan audit:
- The pass plan is sensible only if interpreted as a first consolidation pass,
  not a finished manuscript.
- The main missing risk in prior plans was source-of-truth ambiguity between
  BayesFilter and `dsge_hmc`.  The new plan fixes that: BayesFilter owns
  structural filtering; DSGE and MacroFinance own structural maps and adapters.
- The plan correctly postpones final derivative, SVD-gradient, and HMC
  convergence claims.
- No phase should edit original source projects.

### Phase S0: hygiene baseline

Phase plan:
- Record dirty state.
- Parse `docs/source_map.yml`.
- Run `git diff --check`.
- Build the monograph with `latexmk`.

Execution:
- `git status --short` showed existing modified files:
  `docs/chapters/ch02_state_space_contracts.tex`, `docs/main.tex`,
  this reset memo, `docs/references.bib`, and `docs/source_map.yml`.
- Existing untracked files:
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`,
  `docs/plans/bayesfilter-structural-ssm-monograph-consolidation-pass-plan-2026-05-04.md`,
  `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`,
  `docs/plans/dsge-structural-filtering-refactor-plan-2026-05-03.md`, and
  `docs/plans/svd-sigma-point-derivative-tool-field-test-2026-05-03.md`.
- `python -c "import yaml; ..."` returned `source_map yaml ok`.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `docs/` reported all targets up to date.

Audit:
- Phase S0 passes.
- Dirty state is understood and contains BayesFilter documentation work only.
- Phase S1 remains justified.

### Phase S1: title and framing

Phase plan:
- Reframe the monograph title around Bayesian estimation for structural
  state-space models.
- Update the introduction so HMC-safe filtering is the production discipline,
  not the whole subject.
- State the central design thesis: model projects own structural maps, while
  BayesFilter owns reusable filters and derivative/diagnostic contracts.

Execution:
- Updated `docs/main.tex` title to:
  `Bayesian Estimation for Structural State-Space Models`.
- Updated `docs/chapters/ch01_introduction.tex` to define structural state
  roles such as stochastic shocks, deterministic lags, accounting identities,
  endogenous completions, auxiliary variables, and measurement-only summaries.
- Added a `Central Design Thesis` section.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  successfully and produced an 85-page PDF.
- A targeted claim search found only the intended cautionary phrase
  `HMC-safe` in the introduction.

Audit:
- The first page now matches the intended monograph scope.
- No new HMC convergence or derivative-certification claim was introduced.
- Phase S2 remains justified because Part I now needs to make the structural
  partition contract precise.

### Phase S2: structural contracts and API

Phase plan:
- Extend the state-space contract chapter with a structural state partition.
- Add AR(p)/lag-stack material as the simplest generic example of a degenerate
  transition.
- Extend the API chapter with a structural model protocol, filter run metadata,
  and a client adapter boundary.

Execution:
- Updated `docs/chapters/ch02_state_space_contracts.tex` with:
  - definition of a structural state partition;
  - the rule that partition metadata is not inferred only from a covariance
    matrix;
  - AR(2) companion-form/lag-stack example with singular transition
    covariance;
  - a structural nonlinear transition contract;
  - approximation-labeling assumption.
- Updated `docs/chapters/ch04_bayesfilter_api.tex` with:
  - five core objects including a structural state partition;
  - structural model protocol;
  - filter run metadata;
  - strict adapter boundary between BayesFilter and client projects.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  successfully and produced an 88-page PDF.
- Targeted claim search found only expected uses of `exact` for exact linear
  Kalman filtering and no unsupported convergence/production claim.

Audit:
- Part I now gives implementation agents a stable contract vocabulary for
  BayesFilter structural filtering.
- The AR(2) example confirms that the endogenous/deterministic-state issue is
  generic and not DSGE-only.
- Only minor LaTeX overfull/underfull warnings remain; they are not blockers.
- Phase S3 remains justified.

### Phase S3: provenance and reset updates

Phase plan:
- Update `docs/source_map.yml` with the new BayesFilter structural-state
  partition core plan and this monograph consolidation pass.
- Keep provenance explicit: the new contract prose is based on local planning
  artifacts and documented source-project audits, not on a new literature
  claim.
- Update this reset memo so a rebooted agent can identify the canonical next
  work.

Execution:
- Added source-map entries for:
  - `bayesfilter_structural_state_partition_core_plan`;
  - `bayesfilter_structural_ssm_monograph_pass`.
- Added chapter mappings from those plans to:
  - `docs/chapters/ch01_introduction.tex`;
  - `docs/chapters/ch02_state_space_contracts.tex`;
  - `docs/chapters/ch04_bayesfilter_api.tex`;
  - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`;
  - `docs/chapters/ch32_production_checklist.tex`.
- Re-affirmed that BayesFilter is the generic structural filtering source of
  truth, while `dsge_hmc` and MacroFinance should provide adapters and
  structural maps.

Tests:
- `python -c "import yaml; yaml.safe_load(...)"` returned
  `source_map yaml ok`.

Audit:
- Provenance is now explicit enough for another agent to continue after a
  reboot.
- This phase did not add new mathematical claims beyond local planning and
  source-code audit provenance.
- Phase S4 remains justified.

### Phase S4: final build, audit, and commit preparation

Phase plan:
- Run final source-map parse, whitespace check, LaTeX build, and risky-claim
  search.
- Stage only BayesFilter documentation files related to the structural
  state-space monograph pass.
- Commit after this reset memo records completion status.

Execution and tests:
- `python -c "import yaml; yaml.safe_load(...)"` returned
  `source_map yaml ok`.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported
  `All targets (main.pdf) are up-to-date`.
- Risky-claim search over the touched chapters found one expected cautionary
  use of `production-ready` in
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`, where the text
  says no future BayesFilter report should claim nonlinear DSGE filtering is
  correct or production-ready unless the listed tests pass.

Audit:
- The monograph now has a clearer organizing thesis: Bayesian estimation for
  structural state-space models.
- Part I now defines structural state partitions, degenerate transitions,
  approximation labels, structural model protocols, run metadata, and client
  adapter boundaries.
- The work remains a first consolidation pass, not a completed manuscript.
- Remaining warnings are typographic LaTeX overfull/underfull warnings and are
  not blockers.

Interpretation:
- The pass is complete and ready to commit.
- Next writing work should proceed to the exact linear Gaussian likelihood
  spine and analytic Kalman derivative consolidation.

## 2026-05-04 update: exact linear Gaussian spine writing pass

User requested the next documentation continuation pass with a detailed plan,
reset memo updates, independent audit, phase-by-phase execution, tests, audit,
commit, and summary.  This pass is scoped to the value-side exact linear
Gaussian likelihood spine.  It deliberately defers analytic score/Hessian
formulas to the next pass.

Planning artifact:
- Added
  `docs/plans/bayesfilter-linear-gaussian-spine-writing-plan-2026-05-04.md`.

Independent plan audit:
- The plan is sensible because it keeps the pass value-side and
  contract-focused.
- It would be unsafe to finalize Kalman score/Hessian formulas here without a
  MacroFinance derivation/code audit, so those formulas remain deferred.
- The pass should include singular process covariance, missing-data, backend
  diagnostics, and large-scale validation policy because those are prerequisites
  for structural partitions and later HMC claims.

### Phase L0: hygiene and source inventory

Phase plan:
- Record working tree state.
- Parse `docs/source_map.yml`.
- Run `git diff --check`.
- Identify source material for linear Gaussian, Kalman, and mixed-frequency
  prose.

Execution:
- `git status --short` showed one pre-existing modified file:
  `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`,
  plus the newly added linear-Gaussian spine plan.
- `python -c "import yaml; ..."` returned `source_map yaml ok`.
- `git diff --check` passed.
- Source inventory used targeted `rg` over:
  - `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`;
  - `/home/chakwong/python/docs/chapters/ch08_kalman_filter.tex`;
  - `/home/chakwong/latex/CIP_monograph/chapters/ch16_kalman_filter.tex`;
  - `/home/chakwong/latex/CIP_monograph/chapters/ch18_mixed_frequency.tex`.

Audit:
- MacroFinance analytic Kalman note is the primary source for exact likelihood
  and later derivative layering.
- CIP Kalman and mixed-frequency chapters provide useful exposition and
  missing-data/mixed-frequency policy but must not be copied wholesale.
- The pre-existing edit to the structural partition core plan is related and
  should be preserved in this commit if final checks pass.
- Phase L1 remains justified.

### Phase L1: prediction-error decomposition chapter

Phase plan:
- Make Chapter 5 the exact value-side likelihood reference.
- Clarify singular process covariance, innovation regularity, missing
  observations, Joseph/covariance update semantics, and initialization policy.
- Keep score/Hessian formulas deferred.

Execution:
- Updated `docs/chapters/ch05_prediction_error_decomposition.tex` with:
  - semidefinite `Q` policy for companion-form lag stacks, mixed-frequency
    accumulators, and deterministic structural coordinates;
  - innovation regularity assumption;
  - Joseph-form covariance update as a numerical representation of the same
    likelihood;
  - all-missing time-step convention;
  - value-oracle/reference role for later backend and derivative chapters.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Targeted search for `Hessian|score|gradient|converged|production-ready|
  certified|guarantee` found only derivative deferrals or consistency
  requirements, not formulas or success claims.

Audit:
- Chapter 5 now provides a clearer exact likelihood value contract.
- The chapter still defers derivative recursions.
- Phase L2 remains justified.

### Phase L2: stable linear filtering chapter

Phase plan:
- Make Chapter 6 a numerical backend policy for exact linear Gaussian
  likelihoods.
- Distinguish covariance-form reference algebra from solve/square-root
  production evaluation.
- Preserve a clear line between exact linear SVD/spectral backends and
  nonlinear SVD sigma-point filters.

Execution:
- Updated `docs/chapters/ch06_stable_linear_filtering.tex` with:
  - covariance form as reference algebra;
  - solve-form contract based on factorization, triangular solves, log
    determinants, and residual diagnostics;
  - Cholesky, QR, and spectral backend roles;
  - explicit warning that SVD-factored exact LGSSM success does not certify
    nonlinear SVD sigma-point filtering;
  - backend agreement test checklist;
  - diagnostic record policy for HMC reports.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Risky-word search found only intended exact-linear terminology and warnings
  against automatic safety/certification.

Audit:
- Chapter 6 now explains how exact likelihood semantics are preserved across
  numerical backends.
- The chapter avoids implying that spectral fallback proves gradient safety.
- Phase L3 remains justified.

### Phase L3: missing data and mixed frequency chapter

Phase plan:
- Make Chapter 7 a precise contract for row selection, mask patterns,
  all-missing time steps, mixed-frequency augmentation, and static-shape
  compiled policies.
- Treat missingness and mixed-frequency alignment as likelihood semantics, not
  as preprocessing outside the model.
- Keep derivative formulas deferred.

Execution:
- Updated `docs/chapters/ch07_missing_data_mixed_frequency.tex` with:
  - observed-row notation for per-period innovations and innovation
    covariances;
  - all-missing likelihood contribution convention;
  - mixed-frequency state-augmentation policy for deterministic cumulators,
    publication-lag states, and running averages;
  - singular deterministic-row covariance policy;
  - diagnostic record requirements for masks, all-missing steps, compiled
    grouping, handling policy, and regularization;
  - explicit derivative deferral.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Targeted risky-claim search found only intended policy language and
  derivative deferrals.

Audit:
- Chapter 7 now treats sparse and mixed-frequency panels as part of the
  likelihood contract.
- It avoids claiming derivative or compiled-backend certification.
- Phase L4 remains justified.

### Phase L4: large-scale LGSSM chapter

Phase plan:
- Make Chapter 8 the bridge from exact textbook filtering to industrial-scale
  validation.
- Add scale metadata, backend registry, validation ladder, stress scenarios,
  HMC readiness criteria, and derivative-consolidation hypotheses.
- Avoid claiming that current evidence solves every NAWM-scale model.

Execution:
- Updated `docs/chapters/ch08_large_scale_lgssm.tex` with:
  - backend registry requirements for value backend, derivative backend,
    initialization, missing-data handling, static-shape assumptions, and
    fallback labels;
  - value-to-HMC validation ladder;
  - stress scenarios for singular `Q`, nearly singular innovations, long
    panels, ragged panels, boundary draws, and spectral telemetry;
  - explicit HMC-readiness criteria;
  - hypotheses for the next MacroFinance analytic Kalman derivative
    consolidation pass.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Risky-claim search over Chapters 5--8 returned no matches for unsupported
  guarantee, convergence, production, industrial, certification, or
  always-type claims.

Audit:
- Chapter 8 now states what must be tested before large-scale HMC experiments.
- It treats MacroFinance large-scale evidence as infrastructure evidence, not
  as blanket NAWM readiness.
- The derivative material is framed as hypotheses to test, not completed
  certification.
- Phase L5 remains justified.

### Phase L5: provenance, final audit, tidy, and commit

Phase plan:
- Record the exact-linear spine pass in `docs/source_map.yml`.
- Run final mechanical checks.
- Audit the changed files for unsupported claims and generated artifacts.
- Commit only the intended documentation and plan files.

Execution:
- Updated `docs/source_map.yml` with:
  - source id `bayesfilter_linear_gaussian_spine_plan`;
  - chapter mapping from the plan to Chapters 5--8.
- Preserved the related pre-existing edit to
  `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
  because it strengthens the math-audit, code-audit, and reuse rules needed by
  the next implementation agent.

Final tests:
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8')); print('source_map yaml ok')"` passed.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed and
  reported `main.pdf` up to date after the Chapter 8 build.
- Risky-claim scan over Chapters 5--8 returned no matches for unsupported
  guarantee, convergence, production, industrial, certification, unbiased, or
  always-type claims.

Audit:
- No BayesFilter code was changed.
- No source-project files under `/home/chakwong/python`,
  `/home/chakwong/latex/CIP_monograph`, or `/home/chakwong/MacroFinance` were
  modified.
- The pass did not introduce final Kalman score/Hessian formulas.
- The pass did not claim industrial readiness for NAWM, DSGE, or any backend
  family.
- Generated PDFs and LaTeX byproducts remain ignored and are not intended for
  staging.

Interpretation:
- The exact linear Gaussian value contract is now clearer and can be used as
  the reference likelihood for later derivative, nonlinear-filter, and HMC
  validation work.
- The next phase remains justified: consolidate and audit the MacroFinance
  analytic Kalman score/Hessian derivations against this value contract before
  promoting derivative backends to HMC use.

Completion summary:
- Added the detailed execution plan
  `docs/plans/bayesfilter-linear-gaussian-spine-writing-plan-2026-05-04.md`.
- Strengthened Chapter 5 as the exact prediction-error likelihood reference,
  including singular `Q`, innovation regularity, Joseph-form semantics,
  all-missing observations, and reference-role language.
- Strengthened Chapter 6 as the exact linear numerical backend policy,
  including solve form, square-root/spectral roles, backend agreement tests,
  and HMC diagnostic metadata.
- Strengthened Chapter 7 as the missing-data and mixed-frequency likelihood
  contract, including row-selection notation, deterministic auxiliary states,
  all-missing steps, compiled mask policy, and derivative deferral.
- Strengthened Chapter 8 as the large-scale LGSSM validation bridge, including
  backend registry, validation ladder, stress scenarios, HMC readiness, and
  explicit derivative-consolidation hypotheses.
- Updated `docs/source_map.yml` and this reset memo.

Next hypotheses to test:
- H1: The Chapter 5 prediction-error likelihood is exactly the value contract
  differentiated in the MacroFinance analytic Kalman note after notation
  reconciliation.
- H2: Analytic score recursions can match autodiff or finite differences on
  small dense and structurally singular-`Q` LGSSM cases without changing the
  value contract.
- H3: Analytic Hessian recursions can match finite-difference-on-gradient tests
  on small cases before being used for mass-matrix or curvature construction.
- H4: Solve-form derivative implementations can avoid explicit innovation
  covariance inverses while preserving value/gradient/Hessian parity.
- H5: The backend registry and diagnostics in Chapter 8 are sufficient to make
  later HMC failures attributable to value algebra, derivative path,
  posterior geometry, or compilation mode.

## Plan audit

Plan audited:
- `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`

Audit outcome:
- The plan is directionally correct and should create a new BayesFilter
  monograph without changing original source documents.
- The major risk is uncontrolled content migration: hundreds of pages could be
  copied before notation, bibliography, provenance, and build policies are
  settled.

Amendments made:
- Added an independent audit addendum to the plan.
- Required original source docs to remain read-only.
- Added label namespace policy for BayesFilter labels.
- Added bibliography and build-artifact policy.
- Clarified that Phase 2 literature setup is a gate for literature-heavy
  chapters.
- Clarified that Phases 3--7 should not run as content migration until Phase 0
  and Phase 1 pass, and Phase 2 is either passed or explicitly scoped down.

Interpretation:
- Phase 0 is justified.
- Phase 1 is justified after Phase 0 builds.
- Phase 2 setup is justified after Phase 1.
- Full content migration is not yet justified.

## Phase 0: Repository and build scaffold

Phase plan:
- Create a standalone BayesFilter LaTeX book skeleton under
  `/home/chakwong/BayesFilter/docs`.
- Keep original source documents read-only.
- Commit LaTeX sources and plans only; treat generated build byproducts as
  local artifacts.

Execution results:
- Created `docs/main.tex`, `docs/preamble.tex`, `docs/references.bib`, and
  `docs/source_map.yml`.
- Created placeholder chapters `docs/chapters/ch01_*.tex` through
  `docs/chapters/ch32_*.tex`.
- Created placeholder appendices `docs/appendices/app_a_*.tex` through
  `docs/appendices/app_g_*.tex`.
- Added `.gitignore` entries for LaTeX byproducts.
- Built the skeleton from `/home/chakwong/BayesFilter/docs` with:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

Test results:
- Skeleton PDF build succeeded.
- Generated `docs/main.pdf` was 83K and 51 pages.
- The only expected warning was an empty bibliography, because the skeleton has
  no citations yet.
- LaTeX byproducts are ignored by Git. `docs/main.pdf` is also ignored as a
  generated local artifact unless a later release phase explicitly requests a
  committed PDF.

Audit:
- Phase 0 satisfies the build-scaffold pass criteria.
- No source content was migrated, beyond short placeholders.
- Original source projects were not edited.

Interpretation:
- The scaffold is healthy enough to support source indexing.
- Phase 1 is justified.

## Phase 1: Source index and provenance map

Phase plan:
- Build a machine-readable source map before importing any prose.
- Classify each major source chapter as `candidate`, `superseded`, or
  `excluded`.
- Identify duplicate groups and sources whose claims require a literature gate.

Execution results:
- Replaced the initial `docs/source_map.yml` stub with a detailed provenance
  map covering:
  - DSGE/HMC monograph source chapters and SVD reset notes.
  - CIP monograph source chapters and appendices.
  - MacroFinance analytic Kalman note, bibliography, implementation files, and
    tests.
- Recorded destination chapters for reusable material.
- Marked application-specific economics chapters as excluded or case-study
  only.
- Marked transport, particle-filter, nonlinear-filtering, and quantitative
  production claims as literature/audit gated.
- Built temporary MathDevMCP LaTeX indexes:
  - `/home/chakwong/python/docs`: 3615 blocks, 1065 labels.
  - `/home/chakwong/latex/CIP_monograph`: 2607 blocks, 1016 labels.
  - `/home/chakwong/MacroFinance`: 225 blocks, 140 labels.
- Ran targeted MathDevMCP searches for:
  - Kalman likelihood, score, and Hessian material.
  - Mixed-frequency nonlinear filtering and HMC.
  - SVD sigma-point/eigen-gap material.
  - Particle-filter, differentiable resampling, and pseudo-marginal HMC
    material.
  - XLA, `GradientTape`, HMC, and finite-gradient policy.

Test results:
- `docs/source_map.yml` parses successfully with `yaml.safe_load`.
- The LaTeX skeleton remains buildable with `latexmk`.
- The source map now includes phase gates showing Phase 1 passed, Phase 2
  pending, and content migration blocked until literature/provenance gates are
  satisfied.

Audit:
- The map avoids wholesale import of original chapters.
- It distinguishes generic BayesFilter material from downstream economics.
- It explicitly identifies duplicate source groups:
  - Kalman likelihood basics.
  - HMC foundations and diagnostics.
  - Nonlinear Gaussian filters.
  - Particle-filter frontier.
- Bibliography sources are listed but not merged, preserving the key-conflict
  audit requirement.

Interpretation:
- Phase 1 satisfies its pass criteria.
- Phase 2 literature workspace setup is justified.
- Full content migration remains blocked because literature-heavy claims and
  advanced nonlinear-filter/HMC claims do not yet have approved local source
  summaries.

## Phase 2: Literature workspace setup

Phase plan:
- Create a dedicated ResearchAssistant workspace.
- Confirm local tool readiness.
- Search for existing approved summaries.
- Create a durable seed list under BayesFilter so the phase can continue after
  reboot even if the temporary workspace is removed.

Execution results:
- Initialized ResearchAssistant workspace:
  `/tmp/ra-bayesfilter-monograph`.
- `doctor` reported status `ok`, offline mode enabled, providers disabled, and
  local PDF parsing tools available (`pdftotext`, `markitdown`,
  `marker_single`, and `magic-pdf`).
- `find --query "Hamiltonian Monte Carlo Kalman filter differentiable
  filtering"` returned no existing local hits.
- `find --query "Kalman"` returned no existing local hits.
- `review-list` returned no approved reviews.
- Added durable seed/audit checklist:
  `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`.
- Inspected source bibliography sizes:
  - `/home/chakwong/python/docs/references.bib`: 1459 lines.
  - `/home/chakwong/latex/CIP_monograph/references.bib`: 3984 lines.
  - `/home/chakwong/MacroFinance/analytic_kalman_derivatives.bib`: 73 lines.
- Confirmed many relevant candidate keys exist locally, but they are not yet
  reviewed or merged.

Test results:
- ResearchAssistant workspace initialization passed.
- ResearchAssistant `doctor` passed.
- Local reviewed-summary search found no approved source material.

Audit:
- Phase 2 setup is useful and reproducible.
- Phase 2 full pass criteria are not met because there are no approved local
  summaries or citation-neighborhood notes.
- The seed list preserves the next literature task without falsely treating
  raw `.bib` entries as evidence.

Interpretation:
- Phase 2 is setup-only, not passed as a literature gate.
- Phases 3--7 content migration are not justified in this autonomous pass.
- The next justified phase is Phase 2B: literature ingestion, review, and
  claim-support audit, or a deliberately scoped internal-only drafting pass
  that avoids external literature claims.

## Stop decision before Phases 3--7

Decision:
- Stop autonomous content migration before Phases 3--7.

Reason:
- The plan's audited gate requires approved literature summaries or an explicit
  scoping decision before drafting literature-heavy chapters.
- Phase 2 created the tooling workspace and durable seed list, but did not
  produce reviewed summaries.
- Proceeding into the linear/nonlinear/HMC chapters now would risk turning
  BayesFilter into a copy-paste archive of source monographs and unsupported
  claims.

What is still safe without additional approval:
- Continue Phase 2B literature ingestion and claim-support review.
- Draft purely internal scaffolding material: notation conventions,
  source-map appendix, implementation-interface placeholders, and reset-memo
  templates, provided no unsupported external claims are introduced.

What is not justified yet:
- Drafting full Parts III--VI.
- Claiming any SVD sigma-point filter path is HMC-safe at large scale.
- Importing particle-filter, pseudo-marginal HMC, transport-map, NeuTra, or
  production-threshold claims from source chapters without source review.

## Final validation before commit

Commands:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  from `/home/chakwong/BayesFilter/docs`.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml'));
  print('source_map yaml ok')"` from `/home/chakwong/BayesFilter`.
- `git status --short --ignored` from `/home/chakwong/BayesFilter`.

Results:
- LaTeX skeleton is up to date and buildable.
- `docs/source_map.yml` parses successfully.
- LaTeX byproducts and `docs/main.pdf` are ignored as generated artifacts.

Completion state:
- Completed: plan audit, Phase 0, Phase 1, Phase 2 setup.
- Stopped by design before content migration.
- Initial scaffold commit for this work was `6bafcc5`, then amended during
  reset-memo finalization.

## Commit checkpoint

Scaffold commit before the final reset-memo correction:
- `373bb8e` (`Scaffold BayesFilter monograph`)

Note:
- A later commit may update only this reset memo. Use
  `git log --oneline -2` to inspect the exact final repository state.

Committed files:
- `.gitignore`
- `docs/main.tex`
- `docs/preamble.tex`
- `docs/references.bib`
- `docs/source_map.yml`
- `docs/chapters/*.tex`
- `docs/appendices/*.tex`
- `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
- `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`

Generated but intentionally uncommitted:
- `docs/main.pdf`
- LaTeX build byproducts (`.aux`, `.log`, `.toc`, `.bbl`, `.blg`,
  `.fdb_latexmk`, `.fls`, `.out`)

## Phase 2B: User decision and conservative literature gate

User decision on 2026-05-02:
- Proceed with Phase 2B literature ingestion and review.
- Permission granted to fetch and ingest papers.
- Use a conservative publication/peer-review standard.
- Prefer detailed derivations and MathDevMCP verification wherever possible.
- Commit and push completed work.
- Do not commit generated PDFs.

Phase 2B plan:
- Create a persistent but Git-ignored ResearchAssistant workspace under
  `/home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph`.
- Fetch/ingest a bounded first core paper set covering:
  Kalman filtering, analytic Kalman derivatives, sigma-point filters,
  SVD/eigen differentiability risk, particle/pseudo-marginal methods,
  HMC/NUTS/geometry, and transport/surrogate methods.
- Write durable committed claim-support notes under `docs/plans`.
- Keep source PDFs, extraction artifacts, and RA caches uncommitted.
- Update `docs/source_map.yml` gate status based on the evidence obtained.

## Phase 2B: Literature ingestion and conservative claim gate

Phase plan:
- Ingest a bounded first set of core papers with ResearchAssistant.
- Prefer exact arXiv source packages over fuzzy PDF metadata.
- Reject metadata/download mismatches explicitly.
- Use MathDevMCP on local derivation material wherever it can give useful
  diagnostics.
- Write a durable committed result note and update the source-map phase gates.

Execution results:
- Persistent ignored workspace used:
  `/home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph`.
- Exact arXiv source packages fetched and inspected:
  - Betancourt HMC, arXiv `1701.02434`, paper id
    `betancourt2017conceptual_hmc`.
  - Hoffman-Gelman NUTS, arXiv `1111.4246`, paper id
    `hoffman_gelman2014_nuts`.
  - Hoffman et al. NeuTra, arXiv `1903.03704`, paper id
    `hoffman2019neutra`.
  - Ionescu et al. Matrix Backprop, arXiv `1509.07838`, paper id
    `ionescu2015_matrix_backprop`.
  - Kitagawa Kalman score/Hessian, arXiv `2011.09638`, paper id
    `kitagawa2020_kalman_score_hessian`.
  - Corenflos et al. differentiable PF via OT, arXiv `2102.07850`, paper id
    `corenflos2021differentiable_pf_ot`.
  - Jacob-Chopin-Robert PMCMC discussion/commentary, arXiv `0911.0985`,
    paper id `pmcmc_discussion_comments_candidate`.
- Rejected mismatches:
  - Betancourt HMC PDF queries returned a rank-normalized R-hat paper.
  - PMCMC title and DOI queries returned *Sequential Monte Carlo Methods in
    Practice*, not Andrieu-Doucet-Holenstein.
  - Julier-Uhlmann UKF query returned an ensemble Kalman/unscented-transform
    paper.
  - Pseudo-marginal candidate arXiv `1011.0419` was unrelated.
  - Kitagawa PDF metadata was low confidence, so exact arXiv source is the
    authority.
- Added durable result note:
  `docs/plans/bayesfilter-phase2b-literature-gate-result-2026-05-02.md`.
- Updated `docs/source_map.yml` so Phase 2 literature is `partial_review` and
  content migration is only `partially_unblocked`.

MathDevMCP results:
- `search-latex` found the MacroFinance solve-form likelihood, score, and
  Hessian material, including `eq:solve_score_proved` and
  `eq:first_w_derivative`.
- `audit-derivation-label` on those labels returned `inconclusive`, not
  verified, because the matrix-calculus obligations were outside the bounded
  backend.
- `audit-kalman-recursion` extracted a rich AST operation graph from
  `filters/solve_differentiated_kalman.py`, including Cholesky solves, trace-like
  operations, log determinant via Cholesky diagonals, Kalman updates, gradients,
  and Hessians. The strict required-operation query reported a mismatch because
  the tool records `np.linalg.solve` as `inverse_or_solve` and did not detect
  explicit shape/covariance guards.

Test results:
- Literature artifacts are under `.research/`, which is Git-ignored.
- Generated PDF and LaTeX byproducts remain ignored.
- `docs/source_map.yml` parses successfully with `yaml.safe_load`.
- The LaTeX skeleton remains buildable with `latexmk`; the current generated
  PDF was already up to date.

Audit:
- Phase 2B is useful but partial.
- HMC, NUTS, and SVD/eigen differentiation-risk foundations are now supported
  well enough for careful drafting.
- The NUTS section must explicitly record the strategic implementation decision:
  TFP NUTS is a useful reference and historical lesson, but BayesFilter should
  not use it as the production backend because prior work exposed practical
  issues with full-pipeline XLA compilation, nested-kernel opacity,
  adaptation/control-flow debugging, and model-specific failure handling.
- 2026-05-03 update: the NUTS decision should be documented with a tiny
  reproducible Gaussian timing benchmark, not just project memory. Benchmark
  script: `docs/benchmarks/benchmark_tfp_nuts_gaussian.py`. Result note:
  `docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.md`. The benchmark is
  intentionally simple so that any observed TFP NUTS overhead is not blamed on
  DSGE or SVD-filter complexity.
- Recorded benchmark result: TFP NUTS did XLA-compile on the Gaussian target,
  so the claim is not "NUTS cannot compile." The measured issue is that even on
  this toy target NUTS was materially slower than fixed-step HMC: warm per-draw
  ratios were about 23.4x in eager mode, 24.3x in graph mode, and 14.6x in XLA
  mode. This supports treating TFP NUTS as a diagnostic/reference backend, not
  as the default fix hypothesis for BayesFilter filtering-target failures.
- Quick rerun validation: `python docs/benchmarks/benchmark_tfp_nuts_gaussian.py
  --num-results 4 --num-burnin-steps 2 --repeats 2 --output
  /tmp/tfp_nuts_gaussian_smoke.json` completed successfully. The small default
  profile is suitable as a fast agent guardrail; the fuller JSON under
  `docs/benchmarks/` remains the recorded evidence artifact.
- Analytic Kalman score/Hessian material is promising but not certified.
- PMCMC/pseudo-marginal and Julier-Uhlmann sigma-point primary-source support
  remain open.
- No RA paper downloads or generated PDFs should be committed.

Interpretation:
- The next phase is justified, but it should not be broad content migration.
- The next phase should be a focused analytic-gradient/filter-contract audit:
  formalize BayesFilter notation, cross-check Kitagawa and MacroFinance, add or
  require shape/SPD/finite-value guards, and define the custom-gradient policy
  before industrial-scale SVD/HMC claims are drafted.

Next hypotheses:
- H1: Analytic/custom-gradient filtering will be more robust for HMC than raw
  tape gradients through spectral decompositions when singular values or
  eigenvalues are close.
- H2: A solve-form or square-root linear Gaussian backend can be certified first
  and used as the regression oracle for nonlinear filters.
- H3: SVD sigma-point filtering requires gradient-path safeguards or custom
  derivatives before it is safe for NAWM-scale HMC.
- H4: NeuTra and related transport maps should be treated as geometry
  accelerators, not as correctness substitutes.

Final pre-commit status:
- Phase 2B result note, source-map gate update, `.gitignore`, and this reset
  memo are ready to commit.
- `.research/` and `docs/main.pdf` remain intentionally uncommitted.

## 2026-05-03 Writing Continuation Plan

User request:
- With the NUTS implementation decision benchmarked and documented, create a
  plan to continue writing the BayesFilter monograph.

Execution result:
- Added
  `docs/plans/bayesfilter-monograph-writing-continuation-plan-2026-05-03.md`.

Plan summary:
- Continue from the current scaffold by executing:
  - W0 hygiene and baseline checkpoint.
  - W1 notation, contracts, and reader map.
  - W2 linear Gaussian likelihood spine.
  - W3 analytic derivatives and custom-gradient policy.
  - W4 HMC-safe filtering policy, including the settled TFP NUTS benchmark.
  - W5 nonlinear filtering and SVD sigma-point lessons.
  - W6 transport, surrogates, and geometry.
  - W7 industrial case studies.
  - W8 literature and citation completion.
  - W9 release-quality audit.

Interpretation:
- The immediate next writing action should be W0 followed by W1 only. This
  gives the monograph stable notation, source-backed scope, and API contracts
  before moving into derivative-heavy or SVD/HMC-heavy chapters.
- The plan preserves the consolidation boundary: original documents remain
  read-only, and no new material should be drafted without explicit provenance
  from source docs, reviewed literature, code, tests, benchmarks, or recorded
  project decisions.

## 2026-05-03 Autonomous Writing Pass Started

User request:
- Update this reset memo.
- Audit the writing continuation plan as another developer.
- Execute each phase with a plan/execute/test/audit/tidy/update-memo cycle.
- After each phase, record results, interpretation, and whether the next phase
  is justified.
- Pay special attention to math and derivation audits against original tests.
- Commit the modified files when the pass finishes.

Baseline state before execution:
- Current HEAD: `799352a Record BayesFilter literature gate`.
- Uncommitted intended work already present:
  - TFP NUTS benchmark files under `docs/benchmarks/`.
  - NUTS implementation-position updates in Chapter 21, source map, Phase 2B
    note, and this memo.
  - Writing continuation plan under `docs/plans/`.
- Ignored generated artifacts remain:
  - `.research/`
  - LaTeX byproducts.
  - `docs/main.pdf`.

Plan audit:
- The continuation plan is sound if interpreted as a first conservative
  consolidation pass.
- The audit addendum now states that W2 can include exact likelihood formulas,
  W3 must keep derivation claims tied to source labels/code/tests, W4--W7 should
  draft policy and case-study structure rather than final production claims, and
  W8 may remain a blocker register.
- Proceeding to W0 is justified.

### W0: Hygiene and Baseline Checkpoint

Phase plan:
- Verify source-map YAML.
- Verify benchmark JSON.
- Build the monograph.
- Check diff hygiene.
- Record dirty state before drafting.

Execution and test results:
- `docs/source_map.yml` parsed successfully with `yaml.safe_load`.
- `docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.json` validated:
  benchmark id present, six result rows, all rows status `ok`.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- `git diff --check` passed.

Audit:
- W0 pass criteria are met.
- The current dirty state is understood and consists of intended BayesFilter
  documentation/benchmark work plus ignored generated artifacts.

Interpretation:
- W1 is justified.

### W1: Notation, Contracts, and Reader Map

Phase plan:
- Draft Part I scope and contracts without introducing new derivations.
- Establish notation, shape conventions, target contracts, backend roles, and
  source-map discipline.
- Keep original source documents read-only.

Execution results:
- Drafted `docs/chapters/ch01_introduction.tex`.
- Drafted `docs/chapters/ch02_state_space_contracts.tex`.
- Drafted `docs/chapters/ch03_hmc_target_requirements.tex`.
- Drafted `docs/chapters/ch04_bayesfilter_api.tex`.
- Drafted `docs/appendices/app_a_notation.tex`.
- Drafted `docs/appendices/app_f_source_map.tex`.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W1 files returned no matches.
- Targeted label-namespace search over W1 files returned no violations.
- `git diff --check` passed.

Audit:
- W1 is infrastructure prose and definitions only; no unsupported derivation was
  introduced.
- DSGE, CIP, AFNS, and NAWM material is framed as downstream application
  context, not BayesFilter core.
- The target chapter explicitly requires value, derivative, finite-failure, and
  compilation policies before HMC use.

Interpretation:
- W2 is justified.

### W2: Linear Gaussian Likelihood Spine

Phase plan:
- Draft the value-only exact linear Gaussian likelihood spine.
- Use the MacroFinance analytic derivative note as the authoritative source for
  formulas and MacroFinance tests as implementation evidence.
- Avoid writing score/Hessian formulas before W3.

Execution results:
- Drafted `docs/chapters/ch05_prediction_error_decomposition.tex`.
- Drafted `docs/chapters/ch06_stable_linear_filtering.tex`.
- Drafted `docs/chapters/ch07_missing_data_mixed_frequency.tex`.
- Drafted `docs/chapters/ch08_large_scale_lgssm.tex`.

Source/test comparison:
- The LGSSM state, observation, Kalman recursion, innovation covariance, and
  log-likelihood formulas were consolidated from
  `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`, especially the
  linear Gaussian state-space and prediction-error decomposition sections.
- Backend ladder and large-scale diagnostics were cross-checked against
  `/home/chakwong/MacroFinance/tests/test_large_scale_lgssm_backend_ladder.py`.
- Missing-data policy was cross-checked against
  `/home/chakwong/MacroFinance/tests/test_large_scale_lgssm_missing_data_policy.py`.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W2 files returned no matches.
- `git diff --check` passed.

Audit:
- W2 includes exact likelihood formulas but no new derivative formulas.
- Mentions of score, Hessian, and gradient are restricted to forward references
  and validation/readiness criteria.
- The missing-data chapter states that unsupported masks must fail explicitly,
  matching the MacroFinance test policy.

Interpretation:
- W3 is justified, but derivative drafting must remain source-label and
  test-backed.

### W3: Analytic Derivatives and Custom Gradient Policy

Phase plan:
- Draft analytic derivative chapters conservatively.
- Include only source-backed derivative formulas.
- Use MathDevMCP and MacroFinance tests to distinguish provenance,
  structural-code evidence, and numerical parity evidence.

Execution results:
- Drafted `docs/chapters/ch09_kalman_score.tex`.
- Drafted `docs/chapters/ch10_kalman_hessian.tex`.
- Drafted `docs/chapters/ch11_structural_derivatives.tex`.
- Drafted `docs/chapters/ch12_factor_derivatives.tex`.
- Drafted `docs/chapters/ch13_custom_gradient_wrappers.tex`.
- Drafted `docs/chapters/ch14_derivative_validation.tex`.
- Drafted `docs/appendices/app_b_matrix_calculus.tex`.
- Drafted `docs/appendices/app_c_factor_derivative_proofs.tex`.
- Drafted `docs/appendices/app_d_mathdevmcp_workflows.tex`.

Math/source audit:
- Extracted source context for:
  - `eq:score_note`.
  - `eq:solve_score_proved`.
  - `eq:solve_hessian_proved`.
- Ran `MathDevMCP audit-kalman-recursion` on
  `/home/chakwong/MacroFinance/filters/solve_differentiated_kalman.py`.
- The MathDevMCP result found solve/cholesky, prediction/update,
  quadratic-form, gradient, and Hessian-recursion structure, but returned
  `mismatch` for strict required operations `logdet` and `trace`, and reported
  missing explicit shape/covariance guards.

Original test comparison:
- Cross-checked W3 evidence statements against
  `/home/chakwong/MacroFinance/tests/test_generic_lgssm_autodiff_validation.py`.
- The tests compare solve, square-root, QR square-root, TensorFlow autodiff,
  finite-difference, eager, and compiled paths on controlled LGSSM cases.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W3 files returned no matches.
- Risky-claim search found only cautionary language around proof,
  certification, and production readiness.
- `git diff --check` passed.

Audit:
- W3 does not claim peer-review-ready derivations.
- The score formula is source-backed by `eq:score_note` and
  `eq:solve_score_proved`.
- The Hessian chapter records `eq:solve_hessian_proved` as a source-backed
  contract and explicitly distinguishes it from a fully certified proof.
- The missing shape/covariance guard finding is preserved as a
  production-readiness gap.

Interpretation:
- W4 is justified.

### W4: HMC-Safe Filtering Policy

Phase plan:
- Draft HMC, mass-matrix, boundary, JIT, and diagnostics policy.
- Use the Phase 2B HMC/NUTS literature gate and the TFP NUTS Gaussian benchmark.
- Preserve the distinction between HMC/NUTS mathematical foundations and the
  project-specific implementation decision not to use TFP NUTS as production
  default.

Execution results:
- Expanded `docs/chapters/ch21_hmc_for_state_space.tex`.
- Drafted `docs/chapters/ch22_mass_matrices.tex`.
- Drafted `docs/chapters/ch23_boundary_gradients.tex`.
- Drafted `docs/chapters/ch24_xla_jit.tex`.
- Drafted `docs/chapters/ch25_diagnostics.tex`.

Source/test comparison:
- NUTS implementation stance was cross-checked against
  `docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.md`.
- XLA/JIT policy was cross-checked against
  `/home/chakwong/python/docs/chapters/ch16_xla_ops.tex`.
- Mass-matrix and HMC-diagnostic helper claims were cross-checked against
  MacroFinance HMC regression tests.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W4 files returned no matches.
- Search for NUTS-as-universal-solution patterns returned no bad matches.
- `git diff --check` passed.

Audit:
- The chapter states that TFP NUTS can XLA-compile on the toy benchmark, but
  remains too heavy and opaque to be the default production fix.
- The JIT chapter distinguishes full-pipeline compilation from an outer
  `tf.function` wrapper.
- Diagnostics are framed as warning/evidence, not convergence proof.

Interpretation:
- W5 is justified, but particle-filter and UKF claims must keep blocked-source
  status where the literature gate is incomplete.

### W5: Nonlinear Filtering and SVD Sigma-Point Lessons

Phase plan:
- Draft EKF, sigma-point, square-root sigma-point, SVD sigma-point, particle
  filter, and filter-choice chapters as a conservative policy pass.
- Use the source-project nonlinear-filter chapters only as motivation and
  implementation provenance.
- Anchor SVD claims in the 2026-04-25 SVD math/code audit and the 2026-04-27
  SVD-HMC gap-closure plan.
- Preserve the Phase 2B gates: Julier-Uhlmann UKF primary support and PMCMC /
  pseudo-marginal support remain blocked.

Execution results:
- Drafted `docs/chapters/ch15_ekf.tex`.
- Drafted `docs/chapters/ch16_sigma_point_filters.tex`.
- Drafted `docs/chapters/ch17_square_root_sigma_point.tex`.
- Drafted `docs/chapters/ch18_svd_sigma_point.tex`.
- Drafted `docs/chapters/ch19_particle_filters.tex`.
- Drafted `docs/chapters/ch20_filter_choice.tex`.

Source/test comparison:
- EKF and sigma-point material was cross-checked against
  `/home/chakwong/latex/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
  and `/home/chakwong/python/docs/chapters/ch09_sr_ukf.tex`, but rewritten as
  BayesFilter approximate-likelihood policy rather than imported as a final
  DSGE result.
- SVD material was cross-checked against
  `/home/chakwong/python/docs/chapters/ch09b_svd_filters.tex`,
  `/home/chakwong/python/docs/plans/svd-filter-math-code-audit-result-2026-04-25.md`,
  and `/home/chakwong/python/docs/plans/svd-hmc-gap-closure-plan-2026-04-27.md`.
- Particle-filter text was limited to the Phase 2B accepted differentiable-PF
  frontier status and avoided PMCMC correctness claims because the primary
  PMCMC gate remains blocked.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W5 files returned no matches.
- `git diff --check` passed.
- The W5 risky-claim scan found only intended cautionary uses of terms such as
  exact, convergence, unbiased, industrial, safe, certify, and proof.

Audit:
- W5 does not claim that EKF, UKF, CKF, SVD, or particle filters are exact
  nonlinear likelihoods in general.
- The SVD chapter explicitly distinguishes value-side robustness from HMC
  gradient robustness.
- The SVD chapter records that close or repeated spectral values are a
  derivative-risk mechanism and requires eigen/singular-gap telemetry before
  SVD-HMC production claims.
- The filter-choice chapter recommends analytic/custom derivative paths for
  NAWM-sized or industrial targets and treats raw tape gradients through
  spectral decompositions as diagnostic/small-case tools unless stress evidence
  survives the intended regime.

Interpretation:
- W5 pass criteria are met.
- W6 is justified because transport and surrogate methods can now be drafted as
  downstream geometry aids whose validity depends on the target and derivative
  contracts already written.

### W6: Transport, Surrogates, and Geometry

Phase plan:
- Draft `docs/chapters/ch26_transport_surrogates.tex` as a geometry and
  surrogate-policy chapter.
- Use the Phase 2B NeuTra gate only for bounded transport support.
- Make correction requirements explicit whenever a surrogate changes the target.

Execution results:
- Drafted transported-target notation with the Jacobian-corrected potential.
- Recorded NeuTra as a geometry accelerator, not a correctness guarantee.
- Added a surrogate-status register distinguishing diagnostic, corrected, and
  target-defining surrogates.
- Added transport diagnostics for finite transformed targets, log-Jacobian
  checks, tail probes, reconstruction residuals, and sampler performance.

Source/test comparison:
- Cross-checked transport-map formulas and warnings against
  `/home/chakwong/python/docs/chapters/ch16_transport_foundations.tex`.
- Cross-checked NeuTra/reverse-KL positioning against
  `/home/chakwong/python/docs/chapters/ch18_transport_training.tex` and the
  Phase 2B accepted NeuTra source status.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over Chapter 26 returned no matches.
- `git diff --check` passed.
- The targeted risky-claim scan found only intended cautions: transport is not
  a correctness substitute and should not be proposed as a fix for filter-layer
  derivative or compilation failures.

Audit:
- W6 does not market NeuTra or transport maps as a universal HMC solution.
- The chapter states that exact transported HMC requires evaluating the
  Jacobian-corrected transformed target.
- Surrogates are not allowed to silently change the sampled posterior.

Interpretation:
- W6 pass criteria are met.
- W7 is justified because the core contracts, nonlinear-filter cautions, and
  transport policy now exist and can be applied to case-study structure without
  claiming completed industrial validation.

### W7: Industrial Case Studies and Production Checklist

Phase plan:
- Draft the LGSSM, nonlinear SSM, NK SVD, CIP/AFNS, NAWM, and production
  checklist chapters as evidence maps and readiness checklists.
- Preserve project boundaries: BayesFilter gets filtering lessons, not
  application economics.
- Do not claim completed NK, DSGE, or NAWM convergence.

Execution results:
- Drafted `docs/chapters/ch27_lgssm_validation.tex`.
- Drafted `docs/chapters/ch28_nonlinear_ssm_validation.tex`.
- Drafted `docs/chapters/ch29_nk_svd_case_study.tex`.
- Drafted `docs/chapters/ch30_cip_afns_case_study.tex`.
- Drafted `docs/chapters/ch31_nawm_design_target.tex`.
- Drafted `docs/chapters/ch32_production_checklist.tex`.

Source/test comparison:
- LGSSM validation evidence was cross-checked against MacroFinance Kalman
  backend, missing-data, QR/square-root, large-scale, and HMC diagnostic tests.
- Nonlinear SSM and NK SVD ladders were cross-checked against
  `/home/chakwong/python/docs/plans/hmc-svd-filter-convergence-validation-plan-2026-04-26.md`
  and
  `/home/chakwong/python/docs/plans/nk-hmc-svd-postrefactor-reset-memo-2026-04-26.md`.
- CIP/AFNS scope was cross-checked against the CIP monograph AFNS and
  state-space material, but only filtering interface lessons were imported.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W7 files returned no matches.
- `git diff --check` passed.
- The W7 risky-claim scan found only intended contexts: exact LGSSM,
  blocked NK convergence, NAWM design target, and production-readiness label
  definitions.

Audit:
- W7 treats LGSSM as the exact regression oracle.
- W7 treats nonlinear SSM as the required bridge before DSGE/NK.
- W7 explicitly states that NK SVD has useful production-XLA/smoke evidence but
  is not a converged case study.
- W7 treats NAWM as a design target and hypothesis source, not completed
  evidence.
- The production checklist enforces narrow claim labels rather than hiding
  gaps.

Interpretation:
- W7 pass criteria are met.
- W8 is justified as a literature/workflow and blocker-register pass, not as a
  claim-completion pass.

### W8: Literature Workflow and Blocker Register

Phase plan:
- Do not pretend the bibliography gate is complete.
- Draft the ResearchAssistant workflow appendix.
- Add a durable blocker register for remaining primary-source and
  derivation/code-audit gaps.

Execution results:
- Drafted `docs/appendices/app_e_researchassistant_workflows.tex`.
- Added
  `docs/plans/bayesfilter-literature-blocker-register-2026-05-03.md`.

Source/provenance comparison:
- The appendix and blocker register were cross-checked against:
  - `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`.
  - `docs/plans/bayesfilter-phase2b-literature-gate-result-2026-05-02.md`.
  - `docs/source_map.yml`.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted placeholder search over W8 files returned no matches.
- `git diff --check` passed.

Audit:
- No new bibliography entries were merged in W8 because the remaining
  primary-source gates are not closed.
- The blocker register preserves unresolved UKF, PMCMC/pseudo-marginal,
  square-root filtering bibliography, analytic Kalman derivative certification,
  and industrial SVD-HMC safety gaps.
- The ResearchAssistant workflow states that raw `.bib` entries, downloaded
  PDFs, and search results are not claim support.

Interpretation:
- W8 pass criteria are met as a blocker-register phase.
- W9 release-quality audit is justified.

### W9: Release-Quality Audit

Phase plan:
- Run the full monograph build.
- Parse `docs/source_map.yml`.
- Validate the TFP NUTS Gaussian benchmark JSON.
- Run `git diff --check`.
- Search for unresolved placeholders and risky claims.
- Audit label namespace discipline.
- Confirm generated PDFs and LaTeX byproducts are ignored, not staged.

Execution results:
- Filled the remaining appendix placeholder in
  `docs/appendices/app_g_experiment_templates.tex` with reusable experiment
  templates for filter references, HMC smoke runs, medium recovery, and strict
  convergence.
- Removed the temporary empty `.codex_write_check` directory created only to
  verify write access under the sandbox.

Test results:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- `docs/source_map.yml` parsed successfully with `yaml.safe_load`.
- `docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.json` validated
  against its actual schema:
  - benchmark `tfp_hmc_nuts_gaussian`;
  - six result rows;
  - samplers `{hmc, nuts}`;
  - modes `{eager, graph, xla}`;
  - all statuses `ok`.
- `git diff --check` passed.
- Placeholder search over `docs` found no content placeholders. Remaining
  matches are in plan text that quotes the audit command itself or describes
  Phase 0 placeholder policy.
- Risky-claim search found intended cautionary or definitional contexts:
  blocked NK convergence, exact LGSSM statements, production-readiness label
  definitions, literature blockers, and SVD-HMC safety warnings.
- Label audit found BayesFilter namespaced labels. The only nonstandard prefix
  is `asm:bf-shape-value-contract`, which is still namespaced and corresponds
  to the assumption environment.

Audit:
- No generated PDF or LaTeX byproduct is staged.
- `.research/` remains ignored.
- The monograph still has expected bibliography warnings because citations are
  deliberately gated and most references have not yet been inserted into prose.
- One small overfull hbox remains in Chapter 24 around `tf.function`; the build
  succeeds and the warning is cosmetic.
- The current pass is a conservative first writing pass, not a final
  peer-review certification.

Interpretation:
- W9 pass criteria are met.
- The modified files are ready to commit.

## Completion summary for this writing pass

Completed W0--W9 as a first conservative consolidation pass:
- W0 baseline, build, YAML, benchmark, and diff hygiene.
- W1 notation, state-space contracts, target contracts, API, and source-map
  appendix.
- W2 exact linear Gaussian likelihood spine.
- W3 analytic derivative and custom-gradient policy with honest audit limits.
- W4 HMC, mass matrix, boundary, XLA/JIT, diagnostics, and TFP NUTS decision.
- W5 nonlinear filters, SVD sigma-point lessons, particle-filter caution, and
  filter-choice policy.
- W6 transport, NeuTra, and surrogate geometry policy.
- W7 LGSSM, nonlinear SSM, NK SVD, CIP/AFNS, NAWM, and production checklist
  case-study structure.
- W8 ResearchAssistant workflow and literature blocker register.
- W9 release-quality audit and final placeholder cleanup.

Main remaining work:
- close primary-source blockers for UKF, PMCMC/pseudo-marginal methods, and
  square-root filtering literature;
- complete a dedicated derivation/code audit before marking Kalman score and
  Hessian formulas peer-review ready;
- run strict LGSSM and nonlinear SSM HMC recovery;
- clear the NK SVD clean stress gate before any NK convergence claim;
- add model-specific production-XLA, clean-stress, and convergence gates before
  EZ, SGU, Rotemberg, or NAWM are promoted.

Next hypotheses to test:
- H1: analytic/custom filtering gradients will be more robust than raw tape
  gradients through spectral decompositions for high-dimensional HMC targets.
- H2: exact or controlled LGSSM backends should serve as the regression oracle
  for every nonlinear BayesFilter backend.
- H3: SVD sigma-point filters are useful robust approximate value backends, but
  production HMC use requires eigen/singular-gap telemetry and derivative
  validation.
- H4: transport and NeuTra-like maps can improve geometry only after the target
  and gradient contracts are correct.

## 2026-05-03 update: SVD derivative tool field test

User asked whether the old MathDevMCP Kalman Hessian agent guide should be
deleted and whether the rewritten MathDevMCP and ResearchAssistant tools can
help with the analytical gradient/Hessian derivation for the SVD sigma-point
filter.

Cleanup action:
- Deleted stale file:
  `/home/chakwong/MathDevMCP/docs/kalman-hessian-agent-guide.md`.
- A reference sweep found no active current-doc dependency on that guide.  One
  historical reset-memo mention remains in MathDevMCP as provenance only.

Tool status:
- `codex mcp list` shows both `mathdevmcp` and `research-assistant` enabled.
- MathDevMCP benchmark gate passed 41/41 cases with failed count 0.
- MathDevMCP doctor is usable: LaTeXML, Pandoc, Sage, SymPy, and LeanDojo
  import are available; direct Lean executable version check failed because it
  attempted a download.
- ResearchAssistant doctor under the BayesFilter research workspace reports
  status ok, offline/provider-disabled mode, and local PDF/source inspection
  readiness.

Field test:
- Added durable report:
  `docs/plans/svd-sigma-point-derivative-tool-field-test-2026-05-03.md`.
- Used a scratch LaTeX fixture under `/tmp/bayesfilter_svd_field_test` with
  labeled equations for the one-step sigma-point Gaussian likelihood,
  innovation covariance, score differential, solve derivative, and mixed
  Hessian.
- ResearchAssistant successfully retrieved Matrix Backprop source labels,
  including `prop:svd`, `eqn:svd_dS`, `eqn:svd_dV`, `eqn:dLdX`, and `svd_K`.
  The `svd_K` source equation exposes the spectral-gap denominator
  `1 / (sigma_i^2 - sigma_j^2)`, supporting the SVD/eigen derivative risk
  warning.
- MathDevMCP `search-latex` and `extract-latex-neighborhood` found and
  extracted the scratch labels cleanly.
- MathDevMCP `audit-derivation-v2-label` correctly refused to certify the
  score differential and mixed Hessian, returning `unverified` with actions for
  SPD/invertibility, conformability, trace-square constraints, split
  derivation rows, solve residuals, and conditioning diagnostics.
- MathDevMCP `derive-label-step` is not yet reliable for matrix differential
  identities.  It reported `mismatch` on the standard inverse-solve
  differential pattern because current tokenization does not preserve products
  such as `dS S^{-1}` well enough.

Interpretation:
- The user's skepticism about AI-token math derivations is justified.
- Current tools are useful as guardrails: source lookup, provenance, missing
  assumption detection, abstention, and audit packaging.
- Current tools are not sufficient to certify the analytical gradient or
  Hessian of an SVD sigma-point filter.
- SVD sigma-point derivative work must be done as small labeled derivation
  steps with human mathematical review, MathDevMCP abstention audits, and
  independent numerical parity tests.

Planning action:
- Updated
  `docs/plans/dsge-structural-filtering-refactor-plan-2026-05-03.md` with a
  derivative-audit gate requiring ResearchAssistant primary-source lookup,
  MathDevMCP v2 audit, and finite-difference/autodiff/JVP/VJP/compiled/stress
  tests before SVD sigma-point gradients are used inside HMC.

Next justified work:
- Improve MathDevMCP matrix-differential parsing and obligation extraction
  before treating it as a serious checker for SVD Hessians.
- Write the BayesFilter SVD derivative derivation as a proof-obligation ladder:
  likelihood differential, solve differential, innovation covariance
  derivative, sigma-point moment derivative, spectral-factor derivative or
  non-spectral custom-gradient policy, mixed Hessian terms, and numerical
  parity tests.

## 2026-05-04 update: six-step structural filtering closure plan

User asked for an explicit plan covering the six remaining workstreams:

1. audit before implementation;
2. analytic Kalman derivatives;
3. structural filtering implementation;
4. SVD sigma-point structural filter;
5. validation ladder;
6. literature and monograph completion.

Planning artifact:
- Added
  `docs/plans/bayesfilter-six-step-structural-filtering-closure-plan-2026-05-04.md`.

Current baseline discovered during this pass:
- Latest committed documentation commit is
  `466de70 Strengthen BayesFilter linear Gaussian likelihood spine`.
- The working tree contains an untracked candidate BayesFilter package and
  tests:
  - `bayesfilter/structural.py`;
  - `bayesfilter/filters/kalman.py`;
  - `bayesfilter/filters/sigma_points.py`;
  - `bayesfilter/filters/particles.py`;
  - `bayesfilter/testing/structural_fixtures.py`;
  - `tests/test_*.py`.
- The working tree also contains
  `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`.

Phase C0: baseline and audit of existing work

Phase plan:
- Inspect dirty/untracked state.
- Read the untracked structural source/code audit note.
- Read the candidate BayesFilter package and tests.
- Run the candidate tests.

Execution:
- `git status --short` showed:
  - modified `.gitignore`;
  - untracked `bayesfilter/`;
  - untracked `tests/`;
  - untracked
    `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`.
- Candidate package files were inspected and found to implement:
  - structural state partition metadata;
  - structural filter configuration validation;
  - filter run metadata;
  - covariance-form Kalman reference;
  - structural SVD/cubature sigma-point reference;
  - particle-filter fail-closed placeholder;
  - AR(2) and nonlinear accumulation fixtures.
- `pytest -q` passed with 15 tests.  The only warning was a sandbox-related
  inability to write `.pytest_cache`.

Audit:
- The untracked package appears coherent and directly related to the structural
  filtering plan.
- It should be treated as current candidate implementation state, not ignored
  or duplicated by a later agent.
- It is still an early reference implementation, not a derivative-safe or
  HMC-ready backend.
- The six-step closure plan should therefore distinguish "already candidate
  implemented" from "still blocked".

Interpretation:
- A master closure plan is justified and can reference the candidate package.
- The next phase remains justified: write and audit a durable plan.

Phase C1: write and audit master closure plan

Phase plan:
- Write a detailed plan under `docs/plans`.
- Include motivation, implementation instructions, tests, pass gates, stop
  rules, current baseline, and execution order.
- Audit the plan as another developer and apply modifications.

Execution:
- Added
  `docs/plans/bayesfilter-six-step-structural-filtering-closure-plan-2026-05-04.md`.
- The plan covers:
  - Workstream A: audit before implementation;
  - Workstream B: analytic Kalman derivatives;
  - Workstream C: structural filtering implementation;
  - Workstream D: SVD sigma-point structural filter;
  - Workstream E: validation ladder;
  - Workstream F: literature and monograph completion.

Independent audit:
- The plan is sensible because it preserves the exact LGSSM likelihood as the
  oracle and blocks derivative/HMC promotion until parity tests pass.
- The plan explicitly accounts for the candidate package and tests already
  present in the working tree, so a later agent should not duplicate them.
- The plan separates common BayesFilter infrastructure from DSGE and
  MacroFinance client logic.
- The plan keeps particle filtering fail-closed until a separate audit exists.
- The plan uses ResearchAssistant and MathDevMCP as provenance and obligation
  tools, not automatic proof engines.

Audit modifications applied:
- Added a current-baseline section.
- Added package-metadata task before external package use.
- Added model-family-specific validation labels.
- Added global stop rules for artificial noise on deterministic coordinates,
  unlabeled full-state integration, and derivative parity failures.

Interpretation:
- The plan is ready to serve as the next-session master handoff.
- Final mechanical checks and commit are justified.

Phase C2: final checks, tidy, and commit

Phase plan:
- Run package tests and documentation checks.
- Confirm risky-claim hits are explanatory or historical only.
- Commit the coherent handoff set, including the candidate structural core
  because it passes tests and is now part of the documented current baseline.

Execution and tests:
- `pytest -q` passed: 15 tests passed.  The only warning was a sandbox
  `.pytest_cache` write warning.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8')); print('source_map yaml ok')"` passed.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported the
  monograph up to date.
- Risky-claim scan over the new plan, reset memo, and source/code audit found
  only historical reset-memo cautions and the audit command text itself.

Audit:
- No original source project was edited.
- The candidate package is dependency-light NumPy reference code and tests,
  not client-specific DSGE or MacroFinance logic.
- Particle filtering remains fail-closed.
- Analytic derivatives and HMC promotion remain explicitly blocked behind
  future gates.
- The plan does not claim NK/NAWM convergence or SVD-gradient safety.

Interpretation:
- The six-step master plan is complete.
- The current candidate structural core is safe to commit as an early tested
  baseline, not as a final HMC-ready backend.
- The next phase should start with Workstream A from the new plan: ratify the
  source/code audit and then decide whether to keep extending the candidate
  package or adjust it before derivative and adapter work.

Completion summary:
- Added the six-step structural filtering closure plan.
- Recorded and tested the candidate BayesFilter structural core.
- Committed the source/code audit note as the current audit baseline.
- Preserved fail-closed particle-filter semantics.
- Preserved derivative/HMC gates.

Next hypotheses to test:
- H1: The candidate structural metadata contract is sufficient to represent
  AR(p), mixed-frequency auxiliaries, and DSGE-style stochastic/deterministic
  state partitions without inferring roles from covariance matrices alone.
- H2: The covariance-form Kalman reference can serve as the exact value oracle
  for singular-`Q` LGSSM tests while richer solve/square-root backends are
  developed.
- H3: The structural SVD/cubature sigma-point backend recovers exact LGSSM
  likelihoods in linear cases and remains a clearly labeled Gaussian-closure
  approximation in nonlinear cases.
- H4: MacroFinance analytic score/Hessian formulas can be reconciled with the
  Chapter 5 likelihood without changing the value contract.
- H5: DSGE Rotemberg, SGU, EZ, and NAWM adapters will remain blocked until each
  model supplies explicit structural maps and deterministic-completion tests.

## 2026-05-04 update: structural state partition core implementation pass

User requested execution of the canonical structural state partition core plan
with a phase-by-phase cycle: plan, audit math, audit code, execute, test, audit,
tidy, update reset memo, continue unless the next phase is not justified, then
commit.

Canonical plan:
- `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`

New audit artifact:
- `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`

### Phase 0: repository and dependency inventory

Execution:
- BayesFilter initially had no `bayesfilter/` package and no `tests/` tree.
- Client scans identified candidate DSGE paths in
  `/home/chakwong/python/src/dsge_hmc/filters`, model adapters, and tests.
- MacroFinance scans identified reusable LGSSM dataclasses and Kalman,
  differentiated Kalman, QR/square-root, SVD, masked, and TensorFlow paths.

Audit:
- Phase 0 passes.
- Candidate code paths and derivation sources are recorded in the audit note.
- Phase 1 remained justified.

### Phase 1: mathematical source audit and derivation reconciliation

Execution:
- Read relevant DSGE and MacroFinance monograph/code context.
- Used ResearchAssistant and MathDevMCP, but both were limited for this pass:
  local ResearchAssistant searches had no reviewed matching summaries, and
  MathDevMCP label extraction/search on the large monograph roots errored.
- Recorded those tool limitations explicitly instead of treating them as proof.

Audit:
- Exact/collapsed LGSSM Kalman likelihood is sufficiently supported for a
  BayesFilter-local reference backend.
- Structural nonlinear DSGE completion is not sufficiently derived for
  Rotemberg, SGU, EZ, or NAWM adapters yet.
- Particle-filter semantics remain blocked.
- Phase 2 remained justified for code audit and local-core reuse decisions.

### Phase 2: code audit and migration decision

Execution:
- Classified candidate paths in the audit note.
- Decision: implement BayesFilter-local metadata, exact covariance-form Kalman
  reference, AR(2)/toy nonlinear fixtures, and an approximate structural
  sigma-point backend.
- Decision: do not copy DSGE TensorFlow/MKL SVD code or MacroFinance
  differentiated Kalman code in this phase.

Audit:
- Phase 2 passes for local BayesFilter core implementation.
- Client adapter pilots and HMC gates are not yet justified as correctness
  claims.
- Phase 3/4/5/6/7 local implementation work remained justified.

### Phases 3--7: local contracts, fixtures, SVD sigma-point, and degenerate Kalman

Execution:
- Added package skeleton:
  - `bayesfilter/__init__.py`
  - `bayesfilter/structural.py`
  - `bayesfilter/filters/kalman.py`
  - `bayesfilter/filters/sigma_points.py`
  - `bayesfilter/filters/particles.py`
  - `bayesfilter/testing/structural_fixtures.py`
- Added tests:
  - `tests/test_structural_partition.py`
  - `tests/test_filter_metadata.py`
  - `tests/test_degenerate_kalman.py`
  - `tests/test_structural_ar_p.py`
  - `tests/test_structural_sigma_points.py`
  - `tests/test_derivative_validation_smoke.py`
- Added `tests/conftest.py` for local import path.
- Added `__pycache__/` and `*.py[cod]` to `.gitignore`.
- Updated `docs/source_map.yml` with the audit and local implementation status.

Test:
- `pytest -q tests/test_structural_partition.py tests/test_filter_metadata.py tests/test_degenerate_kalman.py tests/test_structural_ar_p.py tests/test_structural_sigma_points.py tests/test_derivative_validation_smoke.py`
  passed: 15 tests.

Audit:
- Structural partition validation rejects overlaps, missing coverage, and
  unlabeled mixed full-state approximation.
- AR(2) exact Kalman likelihood with singular process covariance is finite.
- Structural sigma-point filtering recovers the linear AR(2) Kalman likelihood
  and preserves the lag-shift identity pointwise.
- Nonlinear toy structural sigma likelihood is finite and close to a dense
  quadrature reference for a one-step case.
- Particle filtering fails closed through an explicit placeholder.
- The sigma-point backend is intentionally labeled as an approximate Gaussian
  closure, not a certified nonlinear exact likelihood.

Interpretation:
- BayesFilter now has a tested local structural contract and toy backend
  scaffold.
- This is enough to support follow-on MacroFinance and DSGE adapter planning.
- It is not enough to promote DSGE mixed-state nonlinear filtering, SVD/eigen
  gradients, or particle-filter HMC claims.

### Phase 8: MacroFinance adapter pilot gate

Gate decision:
- Stop before implementing a MacroFinance adapter in this pass.

Reason:
- MacroFinance derivative migration requires a dedicated derivation/code audit
  of score, Hessian, QR/square-root factor derivatives, SVD fallback telemetry,
  TensorFlow parity, and reference fixtures.
- The next justified MacroFinance step is an adapter design/audit note plus a
  minimal wrapper around the already generic LGSSM dataclasses, not wholesale
  code import.

### Phase 9: DSGE adapter pilot gate

Gate decision:
- Stop before implementing a DSGE adapter in this pass.

Reason:
- SmallNK may be all-exogenous enough for a low-risk adapter, but Rotemberg,
  SGU, EZ, and NAWM-style models need explicit exogenous/endogenous structural
  maps and deterministic completion tests before nonlinear BayesFilter routing
  is justified.
- The existing DSGE SVD sigma-point default adapter remains a candidate source
  and regression reference, not a BayesFilter structural backend.

### Phase 10: HMC readiness gate

Gate decision:
- Not run.

Reason:
- HMC readiness requires validated gradients, eager/compiled parity, and
  model-specific structural adapters.  The current pass deliberately provides
  only value-side local references and finite-difference smoke coverage.

Next hypotheses:
- H1: The BayesFilter `StatePartition` contract is sufficient to express AR(p)
  lag stacks, mixed-frequency accumulators, and DSGE exogenous/endogenous
  timing without inferring structure from `Q` alone.
- H2: A MacroFinance LGSSM adapter can wrap existing generic dataclasses and
  exact Kalman references with minimal code movement if derivative migration is
  kept as a separate audited phase.
- H3: A SmallNK structural adapter should recover existing exact Kalman and SVD
  generic-SSM behavior, while Rotemberg/SGU should fail closed until structural
  completion maps are declared.
- H4: SVD/eigen sigma-point gradients will need a custom-gradient or
  non-spectral derivative policy before production HMC promotion.
- H5: Particle filters should enter BayesFilter only after a proposal,
  resampling, estimator-variance, and target-correction audit.

## 2026-05-04 update: MacroFinance SSH pull and Phase 8 plan

User reported that `~/MacroFinance` was updated and requested a `git pull`
plus a plan for Phase 8 of the structural state partition core plan.

Pull result:
- Initial HTTPS pull failed because GitHub credentials were unavailable in the
  noninteractive environment.
- User clarified that SSH is set up on this machine.
- SSH authentication to GitHub succeeded.
- MacroFinance was fast-forwarded over SSH:
  `22f496e Record analytic HMC validation pilots` ->
  `e23c31e Document one-country HMC remaining test gates`.
- `HEAD`, `origin/main`, and `origin/HEAD` now point to `e23c31e`.

New plan artifact:
- `docs/plans/bayesfilter-phase8-macrofinance-adapter-plan-2026-05-04.md`

Source map:
- Added `bayesfilter_phase8_macrofinance_adapter_plan` to
  `docs/source_map.yml`.

MacroFinance code audit summary from the local snapshot:
- `domain/types.py` exposes `LinearGaussianStateSpace`,
  `LinearGaussianStateSpaceDerivatives`, and `RunConfig`.
- `HMCConfig` now includes chain count, target-XLA choice, full-chain-XLA
  choice, and optional latent initial scale.
- `inference/posterior_adapter.py` exposes `DifferentiableStateSpaceProvider`
  and `PosteriorAdapter` protocols.
- `one_country_derivative_provider.py` is the lowest-risk Phase 8 pilot target
  because it builds a restricted four-parameter AFNS LGSSM with analytical
  derivative tensors and existing parity tests.
- `one_country_tf_derivative_provider.py` now has an explicit
  `initial_mean_policy` and respects first- versus second-order derivative
  requests.
- `filters/tf_differentiated_kalman.py` now separates graph-native
  value-plus-score work from value-plus-score-plus-Hessian work.
- `inference/hmc.py` contains `OneCountryAnalyticThetaNoisePosteriorAdapter`,
  with `log_prob_and_grad`, `log_prob_grad_hessian`,
  `negative_log_prob_and_gradient`, and `negative_log_prob_hessian`.
  Value/gradient paths no longer call Hessian work.
- The updated MacroFinance checkout includes target-XLA HMC chain-policy tests,
  theta2 geometry diagnostics, and JSONL validation harnesses for one-country
  analytic HMC. These are references for BayesFilter readiness gates, not code
  to port wholesale.
- Large-scale and cross-currency providers are useful follow-on targets but
  include masking, production-readiness, and identification policies too broad
  for the first adapter slice.

Phase 8 decision:
- The latest-code gate now passes at `e23c31e`.
- The next justified action is still a value-only BayesFilter wrapper for a
  MacroFinance-shaped LGSSM object.
- After the value wrapper passes, the derivative bridge should preserve
  MacroFinance's split between value-plus-score and Hessian diagnostics instead
  of collapsing every gradient path into second-order work.
- Do not copy MacroFinance financial model construction, differentiated
  Kalman implementations, QR/square-root factor derivative code, SVD fallback
  code, TensorFlow HMC plumbing, or production cross-currency readiness logic
  into BayesFilter during the first slice.

Next hypotheses:
- H8.1: The one-country restricted AFNS derivative provider is sufficient as
  the first MacroFinance adapter target because it is small, deterministic,
  analytically differentiated, and already has local regression tests.
- H8.2: BayesFilter can match MacroFinance value likelihoods by converting only
  LGSSM dataclass fields and preserving initial-state and jitter conventions.
- H8.3: MacroFinance score/Hessian outputs can later be exposed through a
  BayesFilter-facing derivative bridge without migrating derivative recursion
  code; the bridge should preserve the value-plus-score versus Hessian workload
  split.
- H8.4: Large-scale and cross-currency providers should wait until the
  one-country value and derivative bridges pass.

Validation after SSH pull:
- BayesFilter checks passed:
  - YAML parse for `docs/source_map.yml`;
  - `git diff --check`;
  - `pytest -q tests`: 15 passed.
- MacroFinance focused one-country checks passed:
  - `pytest -q tests/test_one_country_analytic_hmc_adapter.py
    tests/test_one_country_hmc_analytic_gradient_hessian.py`: 7 passed.
- MacroFinance theta2 geometry subprocess checks were attempted together with
  the focused subset.  The four failures are path/configuration failures before
  model code executes:
  `tests/test_one_country_theta2_geometry_diagnostics.py` hard-codes
  `/home/chakwong/python/MacroFinance`, while the active checkout is
  `/home/chakwong/MacroFinance`.
- Interpretation: the path issue should be fixed or handled upstream in
  MacroFinance before treating theta2 geometry tests as Phase 8 gate evidence.
  It does not block the value-only BayesFilter adapter plan.

## 2026-05-04 update: Phase 8 MacroFinance adapter execution

Canonical plan:
- `docs/plans/bayesfilter-phase8-macrofinance-adapter-plan-2026-05-04.md`

Independent audit artifact:
- `docs/plans/bayesfilter-phase8-macrofinance-adapter-audit-2026-05-04.md`

Audit conclusion before execution:
- No blocking issue was found.
- The audit added explicit requirements for pure BayesFilter adapter tests,
  optional MacroFinance integration tests, separate first- and second-order
  derivative workloads, HMC-conformance metadata, and reset-memo updates after
  each phase.

### Phase 8.0: latest-code gate

Plan:
- Confirm the active MacroFinance checkout is the pulled SSH state and matches
  remote HEAD.

Execute:
- Verified `/home/chakwong/MacroFinance` on `main`.
- Verified `HEAD`, `origin/main`, and `origin/HEAD` at
  `e23c31e Document one-country HMC remaining test gates`.
- Verified SSH remote HEAD:
  `e23c31e531bdd7cb286af33c4caf154687cee634`.

Test:
- `git status --short --branch` in MacroFinance reports `main...origin/main`
  plus only the deliberate local test-path fixes.
- `git log -1 --oneline --decorate` reports `e23c31e`.
- SSH `git ls-remote` reports the same commit.

Audit:
- Latest-code gate passes.
- The configured MacroFinance `origin` is still HTTPS, but the SSH path works
  and remote-tracking refs have been refreshed.  This is not a blocker.

Interpretation:
- Phase 8.1 remains justified.

### Phase 8.1: MacroFinance contract and derivative audit

Plan:
- Audit the one-country provider and derivative tensors before adding
  BayesFilter adapter code.
- Treat MacroFinance tests as regression references, not as permission to copy
  derivative recursions.

Execute:
- Added contract audit:
  `docs/plans/bayesfilter-phase8-macrofinance-contract-audit-2026-05-04.md`.
- Recorded the parameter order
  `[theta_0, theta_1, theta_2, log_measurement_error_std]`.
- Recorded LGSSM field compatibility, jitter convention, initial-mean policy,
  derivative tensor shapes, active derivative blocks, and the split between
  value-plus-score and value-plus-score-plus-Hessian workloads.

Test:
- MacroFinance focused gate passed:
  `pytest -q tests/test_one_country_analytic_hmc_adapter.py
  tests/test_one_country_hmc_analytic_gradient_hessian.py
  tests/test_one_country_theta2_geometry_diagnostics.py
  tests/test_filter_conventions.py`
  produced 17 passed.
- MacroFinance `git diff --check` passed.

Audit:
- The one-country provider remains the smallest safe pilot.
- First-order and second-order analytical workloads are separable and should
  remain separable in BayesFilter metadata.
- No MacroFinance finance-construction or derivative-recursion code should be
  copied.

Interpretation:
- Phase 8.2 remains justified.

### Phase 8.2: value-only BayesFilter wrapper

Plan:
- Add an optional BayesFilter adapter module for MacroFinance-shaped LGSSM
  objects.
- Keep MacroFinance optional and avoid importing it at BayesFilter package
  import time.
- Prove value parity against the one-country MacroFinance reference.

Execute:
- Added `bayesfilter/adapters/__init__.py`.
- Added `bayesfilter/adapters/macrofinance.py`.
- Added `tests/test_macrofinance_adapter.py`.
- Implemented `macrofinance_lgssm_to_bayesfilter`.
- Implemented `evaluate_macrofinance_provider_likelihood`.
- Added `MacroFinanceLikelihoodResult` metadata.

Test:
- `pytest -q tests/test_macrofinance_adapter.py`: 3 passed.
- `pytest -q tests`: 18 passed.
- Import smoke:
  `python -c "import bayesfilter; import bayesfilter.adapters.macrofinance as m; print(m.__name__)"`
  passed.
- The optional integration test imported `/home/chakwong/MacroFinance` and
  matched the one-country MacroFinance differentiated-Kalman value at the same
  fixture/reference point to tight tolerance.

Audit:
- No MacroFinance financial-model construction was copied.
- No MacroFinance Kalman or derivative recursion was copied.
- The adapter result is exact LGSSM value-only and preserves BayesFilter
  metadata.
- The optional integration test skips when MacroFinance is unavailable.

Interpretation:
- Phase 8.3 remains justified.

### Phase 8.3: analytical derivative bridge, no migration

Plan:
- Add a narrow derivative bridge that delegates value/score/Hessian evaluation
  to MacroFinance-compatible backends.
- Preserve the separation between first-order value-plus-score and second-order
  Hessian workloads.

Execute:
- Added `MacroFinanceDerivativeResult`.
- Added `evaluate_macrofinance_provider_derivatives`.
- Exported the derivative bridge from `bayesfilter.adapters`.
- Added pure BayesFilter tests proving first-order calls request only
  `order=1` and return no Hessian, while second-order calls request `order=2`
  and record a Hessian.
- Added optional MacroFinance integration parity against
  `filters.differentiated_kalman.differentiated_kalman_loglik`.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 5 passed.

Audit:
- The derivative bridge is metadata normalization and delegation only.
- No derivative recursion was copied into BayesFilter.
- First-order and second-order workloads remain separate.
- The one-country score and Hessian match the MacroFinance NumPy reference.

Interpretation:
- Phase 8.4 remains justified as an HMC-readiness conformance gate.

### Phase 8.4: HMC-readiness conformance gate

Plan:
- Add no BayesFilter sampler.
- Add a finite-operation conformance check for posterior-like adapters that a
  future HMC driver could consume.
- Explicitly avoid convergence claims.

Execute:
- Added `MacroFinanceHMCReadinessResult`.
- Added `evaluate_macrofinance_hmc_readiness`.
- Added pure BayesFilter conformance tests using a fake posterior adapter.
- Added optional MacroFinance conformance test using
  `OneCountryAnalyticThetaNoisePosteriorAdapter`.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 7 passed.

Audit:
- The readiness result checks finite value, score, negative objective,
  negative gradient, and negative Hessian.
- The readiness label is `hmc_contract_ready_smoke`.
- The convergence claim is explicitly `not_claimed`.
- No sampler was added to BayesFilter.

Interpretation:
- Phase 8.5 remains justified as a deferral/readiness audit for larger
  MacroFinance providers.

### Phase 8.5: large-scale and cross-currency deferral audit

Plan:
- Document why large-scale and cross-currency MacroFinance providers are not
  the first adapter targets.
- Record follow-on hypotheses and candidate tests.

Execute:
- Added
  `docs/plans/bayesfilter-phase8-macrofinance-deferral-audit-2026-05-04.md`.
- Updated `docs/source_map.yml` for the Phase 8 audit artifacts.

Test:
- No new provider code was required for this phase.
- Deferral note was source-backed by local code reads of:
  `large_scale_lgssm_derivative_provider.py`,
  `cross_currency_structural_derivative_provider.py`, and
  `production_cross_currency_derivative_provider.py`.

Audit:
- Large-scale LGSSM should wait for explicit mask-policy and parameter-unit
  metadata.
- Cross-currency structural providers should wait for coverage-matrix and
  finite-difference oracle provenance metadata.
- Production cross-currency providers should wait for blocker-table,
  derivative-coverage, and identification-evidence metadata, and should fail
  closed when final-provider readiness is false.

Interpretation:
- Phase 8 implementation is complete.
- Final validation, tidy, reset-memo completion note, and commits are now
  justified.

### Phase 8 completion validation

BayesFilter validation:
- `pytest -q tests`: 22 passed.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` passed.
- `git diff --check` passed.

MacroFinance validation for the companion path-fix patch:
- `pytest -q tests/test_one_country_theta2_geometry_diagnostics.py
  tests/test_demo_hmc.py tests/test_demo_cross_currency.py
  tests/test_perf_tfp_analytic_filter_speed.py
  tests/test_perf_one_country_analytic_hmc_validation.py`:
  17 passed.
- Earlier focused MacroFinance gate:
  `pytest -q tests/test_one_country_analytic_hmc_adapter.py
  tests/test_one_country_hmc_analytic_gradient_hessian.py
  tests/test_one_country_theta2_geometry_diagnostics.py
  tests/test_filter_conventions.py`:
  17 passed.
- MacroFinance `git diff --check` passed.

Completion interpretation:
- Phase 8 is complete for the one-country MacroFinance pilot.
- BayesFilter now has optional MacroFinance-shaped LGSSM value conversion,
  delegated derivative result normalization, and finite HMC-target conformance
  checks without sampler migration or convergence claims.
- MacroFinance path-fix tests remove stale `/home/chakwong/python/MacroFinance`
  assumptions from executable tests and make the active checkout portable.

Next justified work:
- Phase 8 follow-on A: add mask-policy and parameter-unit metadata before
  adapting `LargeScaleLGSSMDerivativeProvider`.
- Phase 8 follow-on B: add derivative coverage and finite-difference oracle
  provenance before adapting `CrossCurrencyStructuralDerivativeProvider`.
- Phase 8 follow-on C: add blocker-table, identification-evidence, and
  production-readiness metadata before adapting production cross-currency
  providers.
- Phase 10: use the one-country adapter as a candidate HMC target only after
  value, score, Hessian, eager/compiled parity, and target-readiness gates are
  explicitly rerun in the BayesFilter context.

## 2026-05-04 update: Phase 8 follow-on metadata gap closure

Closure plan:
- `docs/plans/bayesfilter-phase8-followon-gap-closure-plan-2026-05-04.md`

Independent audit:
- `docs/plans/bayesfilter-phase8-followon-gap-closure-audit-2026-05-04.md`

Audit conclusion:
- No blocking issue.
- Execution must keep metadata immutable, preserve fail-closed readiness, avoid
  convergence claims, keep MacroFinance optional, and avoid staging the
  pre-existing dirty `ch18b` chapter edit.

### Phase C0: reset memo setup and latest-state check

Plan:
- Confirm current repo state and start the reset-memo execution trail.

Execute:
- BayesFilter latest commit before this pass:
  `a2b058e Add MacroFinance adapter pilot`.
- MacroFinance latest commit before this pass:
  `0e81988 Derive MacroFinance test roots from file paths`.
- BayesFilter has a pre-existing unstaged chapter edit:
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`.

Test:
- `git status --short --branch` inspected in both repos.
- `git log -1 --oneline` inspected in both repos.

Audit:
- The only BayesFilter dirty file outside this pass is the chapter edit.
- MacroFinance is ahead by the committed path-fix patch and has no uncommitted
  changes.

Interpretation:
- Phase C1 remains justified.

## 2026-05-04 update: structural filtering six-gap execution pass

Closure plan:
- `docs/plans/bayesfilter-structural-filtering-six-gap-execution-plan-2026-05-04.md`

Independent audit:
- `docs/plans/bayesfilter-structural-filtering-six-gap-execution-audit-2026-05-04.md`

Audit conclusion:
- No blocking issue.
- Execute in BayesFilter first.
- Do not edit DSGE or MacroFinance internals in this pass.
- Keep HMC convergence, SVD-gradient, and production-readiness claims blocked
  until their explicit gates pass.
- Scope staging carefully because unrelated MacroFinance follow-on work is
  already dirty in the working tree.

### Phase S0: baseline validation and scoped-state check

Plan:
- Record the latest dirty state.
- Confirm the existing test suite passes before edits.
- Continue only if unrelated dirty files can be kept out of this pass.

Execute:
- Current dirty files before this pass included:
  - `bayesfilter/adapters/macrofinance.py`;
  - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`;
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - untracked Phase 8 follow-on plan/audit files.
- Added this six-gap plan and independent audit.

Test:
- `pytest -q`: 23 passed, with two TensorFlow Probability deprecation warnings
  and one sandbox pytest-cache write warning.

Audit:
- Baseline tests pass.
- The dirty MacroFinance adapter belongs to a separate Phase 8 follow-on pass
  and should not be staged for this structural-filtering closure unless
  explicitly required.

Interpretation:
- Phase S1 remains justified.

### Phase S1: doctrine and reusable algorithm

Plan:
- Add a compact reusable structural-filtering algorithm to Chapter 18b.
- State that filters integrate over declared pre-transition uncertainty and
  complete deterministic coordinates through the model.

Execute:
- Added Section `Reusable BayesFilter Structural Filtering Step` to
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`.
- Added the `structural_filter_step` algorithm as a backend-neutral contract
  covering Kalman, UKF/CKF/SVD sigma-point, and particle-filter patterns.

Test:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Targeted grep confirmed `structural_filter_step` and the new section label.
- `git diff --check` passed.

Audit:
- The new section does not claim that every full-state filter is invalid.
- It states that mixed-model full-state integration is an approximation unless
  it reproduces the structural pushforward law.

Interpretation:
- Phase S2 remains justified.

### Phase S2: DSGE connection and BayesFilter API contract

Plan:
- Strengthen the implementation contract so future agents know what metadata
  and diagnostics a structural filter must expose.

Execute:
- Added `Structural Filter Implementation Requirements` to
  `docs/chapters/ch04_bayesfilter_api.tex`.
- Added requirements for fail-closed partition validation, visible integration
  space, deterministic identity diagnostics, approximation-label propagation,
  reference-oracle tests, and separate derivative/compilation gates.

Test:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Targeted grep confirmed `deterministic identity diagnostics` and
  `structural_filter_step`.
- `git diff --check` passed.

Audit:
- The API contract keeps DSGE and MacroFinance economics in adapters while
  keeping generic validation in BayesFilter.
- It does not promote HMC, SVD-gradient, or production claims.

Interpretation:
- Phase S3 remains justified.

### Phase S3: executable UKF worked-example regression

Plan:
- Convert the Chapter 18b structural UKF worked example into an executable
  numerical oracle.
- Add a pointwise deterministic identity test for the propagated sigma points.

Execute:
- Added `WorkedStructuralUKFModel` in
  `bayesfilter/testing/structural_fixtures.py`.
- Added `UnscentedRule` in `bayesfilter/filters/sigma_points.py` and exported
  it from `bayesfilter/filters/__init__.py`.
- Added tests in `tests/test_structural_sigma_points.py` that reproduce the
  chapter's predicted mean, covariance, observation variance, cross covariance,
  gain, posterior mean/covariance, log likelihood, and off-manifold
  artificial-noise contrast.
- Added a deterministic residual test for
  `k_t - phi k_{t-1} - gamma m_t^2 = 0`.

Test:
- First focused test run failed because the test used the existing cubature
  rule, while the chapter example uses a central-point UKF rule with
  `lambda=0` and central covariance weight 2.
- After adding `UnscentedRule`, `PYTHONDONTWRITEBYTECODE=1 pytest -q
  tests/test_structural_sigma_points.py` passed: 4 passed, with one sandbox
  pytest-cache warning.

Audit:
- The initial failure was a useful audit result, not a model failure: it
  prevented conflating cubature and UKF weights.
- The structural SVD/cubature backend was not changed; `UnscentedRule` is a
  reference rule for the worked UKF example and future tests.

Interpretation:
- Phase S4 remains justified.

### Phase S4: implementation contract audit and tidy

Plan:
- Audit the implementation against the six-gap contract and avoid broad
  refactors.

Execute:
- Confirmed the existing `StatePartition`, `StructuralFilterConfig`, and
  `validate_filter_config` paths already fail closed for missing mixed-model
  metadata and unlabeled full-state mixed integration.
- Confirmed `StructuralSVDSigmaPointFilter` still integrates over
  `(previous_state, innovation)` and records the structural metadata.
- Kept the only new code to the small UKF reference rule and worked-example
  fixture/tests.

Test:
- Focused structural sigma-point tests passed after the UKF-rule correction.

Audit:
- The pass stays adapter-neutral.
- No DSGE or MacroFinance model logic was copied into BayesFilter core.
- SVD/eigen derivative and HMC promotion remain blocked.

Interpretation:
- Phase S5 final validation and scoped commit are now justified.

### Phase S5: final validation, tidy, and commit

Plan:
- Run the full validation ladder for this pass.
- Fix any real build or test failures.
- Commit only the scoped structural-filtering files.

Execute:
- Ran full BayesFilter tests.
- Parsed `docs/source_map.yml`.
- Ran whitespace/diff validation.
- Ran a forced LaTeX rebuild to avoid trusting stale `latexmk` state.
- The forced rebuild exposed a real documentation infrastructure bug:
  Chapter 18b used `lemma`, but `docs/preamble.tex` did not define a lemma
  theorem environment.
- Added `\newtheorem{lemma}{Lemma}[chapter]` to `docs/preamble.tex`.

Test:
- `PYTHONDONTWRITEBYTECODE=1 pytest -q`: 43 passed, with two TensorFlow
  Probability deprecation warnings and one sandbox pytest-cache write warning.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8')); print('source_map yaml ok')"` passed.
- `git diff --check` passed.
- `latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex` passed
  after the lemma-environment fix.
- Final log grep found no `LaTeX Error`, no undefined citations, no undefined
  references, and no stale label-rerun warning.  Remaining LaTeX messages are
  layout warnings.

Audit:
- The pass is scoped to BayesFilter structural filtering docs/tests and small
  reference-rule support.
- The new `UnscentedRule` is a reference sigma-point rule for the worked UKF
  example; it does not change the structural SVD/cubature backend's default
  behavior.
- The executable UKF test now guards the exact chapter numbers and the
  deterministic-completion identity.
- The implementation remains adapter-neutral and dependency-light.
- HMC convergence, SVD/eigen gradient safety, and production-readiness claims
  remain explicitly unclaimed.

Completion interpretation:
- The six immediate gaps are closed at the documentation/contract/regression
  level:
  1. doctrine is stated in Chapter 18b;
  2. `structural_filter_step` gives a reusable algorithm;
  3. DSGE timing is connected to the BayesFilter partition contract;
  4. Chapter 4 now states implementation requirements;
  5. tests execute the worked UKF numerical example and identity check;
  6. BayesFilter remains the generic implementation home before client adapters.

Next hypotheses to test:
- H-S1: The structural UKF example's off-manifold likelihood distortion
  generalizes to larger DSGE mixed-state filters when deterministic-completion
  coordinates are noised directly.
- H-S2: AR(p), DSGE, and MacroFinance lag/accounting states can share one
  `StatePartition` contract without model-specific filter forks.
- H-S3: The current NumPy structural SVD/cubature backend should be extended
  with explicit deterministic-residual diagnostics before any DSGE case-study
  claim.
- H-S4: Analytic score/Hessian work must treat structural correctness and
  spectral-gradient safety as separate proof obligations.
- H-S5: NK/Rotemberg/SGU/EZ/NAWM HMC experiments should wait until value,
  derivative, compiled/eager, and sampler gates pass on structural test cases.

### Phase C1: generic metadata primitives

Plan:
- Add immutable metadata primitives for units, masks, derivative coverage,
  finite-difference oracles, production readiness, identification evidence,
  sparse backend policy, and HMC target gates.

Execute:
- Added dependency-free dataclasses to `bayesfilter/adapters/macrofinance.py`.
- Exported them from `bayesfilter/adapters/__init__.py`.
- Added pure tests for immutability, tuple coercion, and fail-closed readiness
  semantics.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 8 passed.

Audit:
- New result objects are frozen dataclasses.
- Readiness metadata has explicit `final_ready` and error fields.
- HMC gate metadata distinguishes target readiness from convergence claims.

Interpretation:
- Phase C2 remains justified.

### Phase C2: large-scale LGSSM mask/unit metadata

Plan:
- Add provider metadata extractors for parameter units and observation-mask
  policy.
- Keep the extraction value-only and metadata-only; do not run large-scale
  likelihood, derivative, or HMC workloads.

Execute:
- Added `extract_parameter_unit_metadata`.
- Added `extract_observation_mask_metadata`.
- Exported both helpers from `bayesfilter.adapters`.
- Added a pure fake-provider test covering finite unit extraction and
  finite-observation mask fallback.
- Added an optional MacroFinance integration test against
  `LargeScaleLGSSMDerivativeProvider` using the current
  `baseline_10x3x5` helper scenario.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 10 passed, with two
  TensorFlow Probability deprecation warnings from the optional MacroFinance
  import path.

Audit:
- The first optional test attempt used a stale scenario name,
  `small_dense_smoke`; the current MacroFinance registry exports
  `baseline_10x3x5`, `long_panel_10x3x5`, and stress scenarios.
- The corrected test checks actual provider names, unit vector shape,
  finite units, observation-mask shape, and all-observed dense-panel policy.
- No large-scale likelihood or derivative workload was run.

Interpretation:
- Phase C3 remains justified.

### Phase C3: cross-currency coverage and oracle provenance

Plan:
- Normalize derivative coverage rows from MacroFinance-like cross-currency
  structural providers.
- Record finite-difference oracle availability and step provenance without
  treating the oracle as the production derivative backend.

Execute:
- Added `extract_derivative_coverage_metadata`.
- Added `extract_finite_difference_oracle_metadata`.
- Added recursive metadata freezing so dataclass, namedtuple, mapping, list,
  set, and NumPy-array row fields are stored as immutable scalar/tuple
  structures.
- Exported both helpers from `bayesfilter.adapters`.
- Added pure fake-provider coverage/oracle tests.
- Added optional MacroFinance integration against
  `CrossCurrencyStructuralDerivativeProvider.from_synthetic_fixture(n_steps=4)`.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py -k "cross_currency_metadata"`:
  2 passed, 15 deselected.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 17 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- Coverage metadata includes real provider block names such as
  `physical_dynamics` and `measurement_error_blocks`.
- Oracle provenance records availability, finite-difference step, and oracle
  class name, but does not run finite differences and does not claim production
  readiness.
- Row metadata is tuple-frozen to avoid mutable list/array payloads.

Interpretation:
- Phase C4 remains justified.

### Phase C4: production readiness metadata

Plan:
- Normalize production blocker table, blocker summary, identification-evidence
  rows, sparse-backend policy rows, and final ten-country readiness validation.
- Fail closed when blockers exist or `validate_final_ten_country_ready` raises.

Execute:
- Added `extract_readiness_blocker_metadata`.
- Added `extract_identification_evidence_metadata`.
- Added `extract_sparse_backend_policy_metadata`.
- Exported the helpers from `bayesfilter.adapters`.
- Added a pure fake-production-provider test proving blocker preservation,
  fail-closed final readiness, identification row freezing, and sparse-policy
  row normalization.
- Added optional MacroFinance integration against
  `ProductionCrossCurrencyDerivativeProvider.from_synthetic_fixture(n_steps=4)`.

Test:
- `pytest -q tests/test_macrofinance_adapter.py -k "production"`:
  2 passed, 15 deselected.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 17 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- The optional production test preserves MacroFinance blockers and
  readiness-blocker table rows exactly.
- Final readiness is `False` because the provider raises
  `requires 10 countries`.
- Blocked derivative rows remain visible as `blocked_final_provider`.
- Identification evidence does not claim `Identified`.
- Sparse backend policy keeps `masked_covariance_reference` visible.

Interpretation:
- Phase C5 remains justified.

### Phase C5: BayesFilter HMC target gates

Plan:
- Add explicit target-readiness gates for finite value, finite score, finite
  negative Hessian, negative-Hessian symmetry, and optional eager/compiled
  value/score parity.
- Preserve the no-sampler and no-convergence-claim boundary.

Execute:
- Added `evaluate_macrofinance_hmc_gate`.
- Exported the helper from `bayesfilter.adapters`.
- Added pure fake-posterior tests for passing parity and failing parity.
- Added optional MacroFinance integration against
  `OneCountryAnalyticThetaNoisePosteriorAdapter`.

Test:
- `pytest -q tests/test_macrofinance_adapter.py -k "hmc_gate"`:
  3 passed, 14 deselected, with two TensorFlow Probability deprecation
  warnings.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 17 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- `target_ready` is true only when finite value/score/Hessian, Hessian
  symmetry, and optional parity gates pass.
- A parity failure records `eager_compiled_parity=False` and
  `target_ready=False`.
- The result carries `convergence_claim="not_claimed"`; no sampler was run.

Interpretation:
- Final validation is justified.

### Phase 8 follow-on completion validation

BayesFilter validation:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 17 passed, with two
  TensorFlow Probability deprecation warnings from the optional MacroFinance
  import path.
- `pytest -q tests`: 32 passed, with the same two TensorFlow Probability
  deprecation warnings.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` passed.
- `git diff --check` passed.

Completion interpretation:
- The large-scale metadata gap is closed at the adapter metadata layer:
  BayesFilter now records parameter units and observation-mask policy without
  running or endorsing large-scale likelihood/HMC workloads.
- The cross-currency provenance gap is closed at the metadata layer:
  BayesFilter now records derivative coverage and finite-difference oracle
  provenance without treating finite differences as production derivatives.
- The production-readiness metadata gap is closed at the adapter metadata
  layer: blockers, blocker tables, identification evidence, sparse backend
  policy, and final-readiness failures are preserved and fail closed.
- The BayesFilter-context HMC target gate is closed for target readiness:
  finite value/score/Hessian, Hessian symmetry, and optional eager/compiled
  parity are explicit, while convergence remains `not_claimed`.
- The pre-existing dirty chapter edit
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex` remains
  outside this pass and must not be included in the Phase 8 follow-on commit.

Next hypotheses to test:
- H-C6: Large-scale LGSSM likelihood adaptation is justified only after
  masked Kalman derivative slicing is implemented or every large-scale adapter
  fixture explicitly reports `all_observed=True`.
- H-C7: Cross-currency structural derivative adaptation is justified when the
  derivative coverage rows cover every provider parameter and when finite
  difference oracle checks pass on a bounded blockwise sample.
- H-C8: Production provider exposure is justified only when
  `final_ready=True`, no blocker rows remain, and identification evidence rows
  can claim final data rather than fixture-only or weak identification.
- H-C9: HMC sampler integration is justified only after target gates pass
  across eager/compiled backends and are followed by actual chain diagnostics
  for ESS, split R-hat, divergences, acceptance, and energy behavior.

## 2026-05-04 update: Phase 8 executable gate closure

Closure plan:
- `docs/plans/bayesfilter-phase8-gate-closure-plan-2026-05-04.md`

Independent audit:
- `docs/plans/bayesfilter-phase8-gate-closure-audit-2026-05-04.md`

Audit conclusion:
- No blocking issue.
- Execution must keep gate result objects immutable, fail closed on missing
  support/evidence, avoid importing MacroFinance implementation logic, avoid
  convergence overclaims, and avoid staging the pre-existing dirty `ch18b`
  chapter edit.

### Phase G0: setup and latest-state check

Plan:
- Record current repository state and start the executable-gate reset trail.

Execute:
- BayesFilter latest commit before this pass:
  `09d2b53 Close MacroFinance metadata adapter gaps`.
- BayesFilter has a pre-existing unstaged chapter edit:
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`.

Test:
- `git status --short --branch` inspected.
- `git log -5 --oneline` inspected.

Audit:
- The new plan converts the prior reset-memo hypotheses H-C6--H-C9 into
  executable gate phases G1--G4.
- No blocker was found before implementation.

Interpretation:
- Phase G1 remains justified.

### Phase G1: large-scale adaptation gate

Plan:
- Add an executable large-scale adaptation gate that combines observation-mask
  metadata with explicit masked-derivative support through the requested order.
- Keep the gate metadata-only; do not run large-scale likelihood inside
  BayesFilter.

Execute:
- Added `LargeScaleAdaptationGateResult`.
- Added `evaluate_large_scale_adaptation_gate`.
- Exported the result and helper from `bayesfilter.adapters`.
- Added pure tests for dense ready, sparse blocked, and sparse ready when
  masked derivative support through order 2 is declared.
- Added optional MacroFinance integration using the dense
  `baseline_10x3x5` provider and the sparse
  `sparse_panel_masked_or_documented_pending` mask.

Test:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py -k "large_scale_gate"`:
  2 passed, 24 deselected, with two TensorFlow Probability deprecation
  warnings.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 26 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- Dense all-observed panels pass without requiring masked derivatives.
- Sparse masks fail closed unless `masked_derivative_order_supported >= 2`.
- The gate records the blocker instead of silently imputing, dropping, or
  treating sparse panels as dense.

Interpretation:
- Phase G2 remains justified.

### Phase G2: cross-currency coverage/oracle gate

Plan:
- Add an executable cross-currency derivative gate that requires coverage rows
  to cover every provider parameter.
- Allow a bounded caller-supplied oracle check and record the maximum absolute
  discrepancy against a tolerance.
- Keep oracle execution outside BayesFilter implementation ownership.

Execute:
- Added `CrossCurrencyDerivativeGateResult`.
- Added `evaluate_cross_currency_derivative_gate`.
- Exported the result and helper from `bayesfilter.adapters`.
- Added pure tests for complete coverage, missing parameter coverage, passing
  oracle discrepancy, and failing oracle discrepancy.
- Added optional MacroFinance integration using
  `CrossCurrencyStructuralDerivativeProvider.from_synthetic_fixture(n_steps=4)`
  with a bounded two-direction oracle comparison.

Test:
- `pytest -q tests/test_macrofinance_adapter.py -k
  "cross_currency_derivative_gate or optional_cross_currency_gate"`:
  3 passed, 23 deselected.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 26 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- Missing provider parameters set `coverage_complete=False` and
  `adaptation_ready=False`.
- Oracle discrepancies larger than tolerance set `oracle_passed=False` and
  `adaptation_ready=False`.
- The optional MacroFinance gate covers all provider parameters and passes the
  bounded oracle discrepancy check without importing oracle logic into
  BayesFilter.

Interpretation:
- Phase G3 remains justified.

### Phase G3: production exposure gate

Plan:
- Add an executable production exposure gate that combines final readiness,
  readiness blockers, identification evidence, and sparse backend policy.
- Require final readiness, no blockers, nonempty sparse policy, and
  final-data `Identified` evidence.

Execute:
- Added `ProductionExposureGateResult`.
- Added `evaluate_production_exposure_gate`.
- Exported the result and helper from `bayesfilter.adapters`.
- Added pure tests for a blocked fake provider and a final-ready fake provider.
- Added optional MacroFinance integration proving the current production
  scaffold remains blocked.

Test:
- `pytest -q tests/test_macrofinance_adapter.py -k "production_exposure_gate"`:
  2 passed, 24 deselected.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 26 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- The current MacroFinance production scaffold remains `exposure_ready=False`.
- Final readiness failures and provider blockers are preserved as blockers.
- Weak, fixture, synthetic, blocked, or not-final identification text prevents
  final identification readiness.
- A pure final-ready fake can pass only when it has no blockers, final readiness
  succeeds, sparse policy exists, and identification rows are `Identified`
  without fixture/blocker language.

Interpretation:
- Phase G4 remains justified.

### Phase G4: HMC diagnostics gate

Plan:
- Add an executable HMC diagnostics gate layered on top of target readiness.
- Require target readiness, finite diagnostics, no divergences, acceptable
  acceptance rate, bounded split R-hat, and minimum ESS.
- Do not infer convergence from finite target operations alone.

Execute:
- Added `MacroFinanceHMCDiagnosticGateResult`.
- Added `evaluate_macrofinance_hmc_diagnostic_gate`.
- Exported the result and helper from `bayesfilter.adapters`.
- Added pure tests for passing diagnostics, bad acceptance, divergences, bad
  split R-hat, low ESS, and target-not-ready behavior.

Test:
- `pytest -q tests/test_macrofinance_adapter.py -k "hmc_diagnostic_gate"`:
  2 passed, 24 deselected.
- Broader focused adapter suite after the implementation:
  `pytest -q tests/test_macrofinance_adapter.py`: 26 passed, with two
  TensorFlow Probability deprecation warnings.

Audit:
- Diagnostics cannot pass if the target gate is not ready.
- Divergences, high split R-hat, low ESS, nonfinite diagnostics, or acceptance
  outside thresholds all produce explicit blockers.
- Passing diagnostics set `convergence_claim="diagnostics_thresholds_passed"`;
  failing diagnostics keep `convergence_claim="not_claimed"`.
- No sampler was added or run by BayesFilter.

Interpretation:
- Final validation is justified.

### Phase 8 executable gate closure validation

BayesFilter validation:
- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py` passed.
- `pytest -q tests/test_macrofinance_adapter.py`: 26 passed, with two
  TensorFlow Probability deprecation warnings from the optional MacroFinance
  import path.
- `pytest -q tests`: 41 passed, with the same two TensorFlow Probability
  deprecation warnings.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` passed.
- `git diff --check` passed.

Completion interpretation:
- H-C6 is closed as an executable large-scale gate: dense panels pass, sparse
  panels require declared masked derivative support through the requested order.
- H-C7 is closed as an executable cross-currency gate: provider parameter
  coverage is checked directly and bounded oracle discrepancies can be recorded
  before adaptation readiness is true.
- H-C8 is closed as an executable production exposure gate: current scaffold
  providers remain blocked, and passing requires final readiness, no blockers,
  sparse policy, and final-data `Identified` evidence.
- H-C9 is closed as an executable HMC diagnostics gate: sampler readiness is
  layered on top of target readiness and requires acceptance, divergence,
  split-R-hat, and ESS thresholds.
- The pre-existing dirty chapter edit
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex` remains outside
  this pass.

Next hypotheses to test:
- H-G5: BayesFilter can safely add a large-scale value/derivative adapter only
  after the G1 gate is supplied with real provider-level
  `masked_derivative_order_supported` metadata rather than a caller override.
- H-G6: Cross-currency structural adaptation should require oracle checks over
  a deterministic blockwise sample that includes dynamics, covariance,
  observation loadings, and measurement error, not only the two-direction smoke
  used in this gate pass.
- H-G7: Production exposure should remain blocked until a real final calibrated
  ten-country provider returns `final_ready=True` and identification rows avoid
  fixture, synthetic, weak, blocked, and not-final qualifiers.
- H-G8: HMC sampler integration should run the diagnostic gate on actual
  MacroFinance chain output and then compare diagnostic sensitivity across
  covariance, QR, and any future XLA-safe target backends.

## 2026-05-05 update: dependency-aware original-plan remaining-gap closure pass

User asked to update the reset memo, audit the remaining-gap roadmap as if by
another developer, and execute every roadmap phase using the cycle:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

The execution target was
`docs/plans/bayesfilter-original-plan-remaining-gap-closure-roadmap-2026-05-05.md`.

### R0: workspace hygiene and scope separation

Plan:
- Record the dirty workspace.
- Classify current dirty files and avoid staging unrelated work.
- Treat the reset memo as a shared file requiring hunk-level staging.

Execute:
- Recorded `git status --short --branch`.
- Confirmed the current pass owns the new remaining-gap roadmap, source-map
  registrations, BayesFilter adapter/filter/backend additions, new tests, and
  this memo section.
- Confirmed the reset memo already contained an unrelated formal
  nonlinear-measurement refactor note before this pass.

Test:
- `git status --short --branch` showed scoped current-pass files and the
  pre-existing reset-memo change.

Audit:
- The pass can proceed if commits stage only the new remaining-gap closure
  hunks and files.

Interpretation:
- R1 remains justified.

### Independent roadmap audit

Plan:
- Review the roadmap as if by a separate developer.
- Check for missing original-plan gaps, dependency mistakes, and overclaim risk.

Execute:
- Added
  `docs/plans/bayesfilter-original-plan-remaining-gap-closure-audit-2026-05-05.md`.
- Added
  `docs/plans/bayesfilter-original-plan-remaining-gap-blocker-register-2026-05-05.md`.

Test:
- Source-map YAML parse is part of final validation.

Audit:
- The roadmap covers DSGE adapters, particle semantics, factor/derivative
  backends, SVD/eigen derivative certification, MacroFinance provider evidence,
  HMC diagnostics, and release docs.
- The main correction is interpretive: BayesFilter can close local gates, but
  cannot honestly promote client DSGE/MacroFinance implementations without
  client-owned evidence.

Interpretation:
- R1 remains justified.

### R1: provenance and control-layer consolidation

Plan:
- Make the remaining-gap control layer explicit.
- Register new plan/audit/control artifacts in `docs/source_map.yml`.
- Preserve conservative labels.

Execute:
- Registered the roadmap, audit, blocker register, particle semantics audit,
  and backend/derivative certification audit in `docs/source_map.yml`.
- Added a closed-vs-blocked register with owner, evidence, required gate,
  allowed label, and promotion blocker for each remaining gap.

Test:
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` is included in final validation.

Audit:
- Every remaining gap now has a dependency and promotion gate.
- Claim labels distinguish `adapter_ready`, `monte_carlo_value_only`,
  `target_candidate`, `diagnostics_thresholds_passed`, and `not_claimed`.

Interpretation:
- R2 remains justified as a BayesFilter adapter gate.  Full DSGE client
  implementation remains blocked pending client-repo metadata work.

### R2: DSGE adapter pilot

Plan:
- Avoid copying DSGE economics into BayesFilter.
- Add a fail-closed BayesFilter gate that accepts explicit DSGE structural
  metadata and blocks missing or incomplete mixed-model metadata.

Execute:
- Added `bayesfilter/adapters/dsge.py`.
- Added `DSGEStructuralAdapterGateResult`.
- Added `dsge_structural_adapter_gate`.
- Exported the gate from `bayesfilter.adapters`.
- Added `tests/test_dsge_adapter_gate.py`.

Test:
- Focused suite including `tests/test_dsge_adapter_gate.py` passed.

Audit:
- Missing DSGE structural metadata fails closed.
- Explicit toy structural metadata passes.
- Mixed DSGE metadata without a deterministic completion map is blocked.
- SmallNK/Rotemberg/SGU client implementation was not claimed, because
  `/home/chakwong/python` is outside this BayesFilter pass.

Interpretation:
- R3 remains justified for BayesFilter toy structural particles.
- Client-facing DSGE adapter promotion remains blocked until the DSGE repo
  supplies and tests model-specific maps.

### R3: particle-filter semantics before implementation

Plan:
- Write the structural particle semantics audit.
- Implement only an innovation-space bootstrap particle reference path.
- Preserve deterministic-completion coordinates pointwise.
- Reject artificial deterministic-coordinate noise unless labeled as an
  approximation.

Execute:
- Added
  `docs/plans/bayesfilter-particle-filter-structural-semantics-audit-2026-05-05.md`.
- Replaced the fail-closed particle placeholder with:
  - `ParticleFilterConfig`;
  - `ParticleFilterResult`;
  - `particle_filter_log_likelihood`.
- Exported particle filter symbols from `bayesfilter.filters`.
- Added `tests/test_structural_particles.py`.
- Updated `tests/test_filter_metadata.py` to check audited particle metadata.

Test:
- Focused particle and metadata tests passed.

Audit:
- The implementation samples innovations, propagates each particle through the
  model transition, and records `monte_carlo_value_only`.
- AR(2) deterministic lag identity is preserved pointwise.
- Unlabeled deterministic-coordinate noise fails closed.
- Labeled artificial deterministic noise is allowed only as
  `declared_approximation`.

Interpretation:
- R4 remains justified.
- Differentiable PF and PMCMC/HMC claims remain blocked behind separate source
  and estimator audits.

### R4: factor-backend and derivative-hook audit

Plan:
- Add a BayesFilter-local backend classification gate.
- Keep value, derivative, compiled, approximation, and HMC statuses separate.

Execute:
- Added `bayesfilter/backends.py`.
- Added `FactorBackendAuditResult` and `audit_factor_backend`.
- Exported backend gates from `bayesfilter`.
- Added `tests/test_backend_readiness.py`.

Test:
- Focused backend tests passed.

Audit:
- Value-only covariance Kalman remains exact for LGSSM values but HMC-blocked.
- A fake exact/derivative/compiled backend can become `target_candidate`.
- Unlabeled approximations are blocked.

Interpretation:
- R5 remains justified.

### R5: SVD/eigen derivative certification

Plan:
- Add a spectral derivative gate with explicit gap telemetry and numerical
  evidence requirements.
- Do not certify client SVD/eigen paths without backend-specific tests.

Execute:
- Added `SpectralDerivativeCertificationResult`.
- Added `certify_spectral_derivative_region`.
- Added
  `docs/plans/bayesfilter-factor-backend-derivative-certification-audit-2026-05-05.md`.

Test:
- Focused backend tests passed.

Audit:
- Large-gap values with declared finite-difference and JVP/VJP checks can pass.
- Near-repeated spectral values produce `spectral_gap_too_small` and block HMC
  eligibility.
- Missing finite-difference or JVP/VJP evidence blocks certification.

Interpretation:
- R6 remains justified for MacroFinance provider-evidence gates.
- SVD/eigen HMC promotion remains blocked for real client paths until
  backend-specific stress tests pass.

### R6: MacroFinance expanded-provider evidence

Plan:
- Use existing MacroFinance gates but add tests for provider-owned masked
  support and blockwise oracle metadata.
- Do not claim final production readiness unless a provider supplies final
  evidence.

Execute:
- Added focused tests proving `evaluate_large_scale_adaptation_gate` can consume
  provider-owned `masked_derivative_order_supported`.
- Added focused tests proving `evaluate_cross_currency_derivative_gate` can
  consume a blockwise oracle result with dynamics, transition covariance,
  observation loadings, and measurement-error block names.

Test:
- Focused MacroFinance adapter tests passed.

Audit:
- BayesFilter now consumes expanded-provider evidence without owning
  MacroFinance financial-model logic.
- The current production exposure rule still requires final readiness, no
  blockers, sparse policy, and final-data `Identified` evidence.

Interpretation:
- R7 remains justified only as a diagnostics gate over supplied chain output.
- MacroFinance production promotion remains blocked until real final provider
  evidence passes the gates.

### R7: HMC sampler readiness

Plan:
- Reuse the existing target and diagnostic gates.
- Confirm no convergence is inferred from target readiness alone.

Execute:
- Kept `evaluate_macrofinance_hmc_gate` and
  `evaluate_macrofinance_hmc_diagnostic_gate` as the sampler-readiness boundary.
- No sampler was added or run by BayesFilter.

Test:
- Focused MacroFinance HMC target/diagnostic tests passed.

Audit:
- `target_ready` is distinct from `convergence_claim`.
- Diagnostics require target readiness, acceptance bounds, zero divergences,
  split R-hat, and ESS.
- Passing synthetic diagnostics set
  `convergence_claim="diagnostics_thresholds_passed"` only for the supplied
  diagnostic object, not for a real production run.

Interpretation:
- R8 remains justified as release-quality validation with conservative blocked
  labels.

### R8: release-quality documentation and literature gates

Plan:
- Keep docs/source-map/reset-memo in agreement with implementation state.
- Run stale-claim and syntax validations.

Execute:
- Registered all new plan/audit artifacts in `docs/source_map.yml`.
- This memo section records each phase result and next-phase justification.
- The blocker register keeps client promotion and production/HMC claims
  conservative.

Test:
- Final validation is recorded below.

Audit:
- Documentation artifacts now distinguish BayesFilter-owned gates from client
  repository blockers.
- No final production or convergence claim is introduced.

Interpretation:
- The BayesFilter-owned remaining-gap closure pass is complete.
- Client implementation/promotions remain as next-phase work.

### Final validation for remaining-gap closure pass

Planned validation:
- focused tests for new gates and affected adapter paths;
- full BayesFilter test suite;
- source-map YAML parse;
- `git diff --check`;
- Python compile check for new modules.

Results:
- `python -m py_compile bayesfilter/__init__.py
  bayesfilter/adapters/__init__.py bayesfilter/adapters/dsge.py
  bayesfilter/backends.py bayesfilter/filters/__init__.py
  bayesfilter/filters/particles.py tests/test_backend_readiness.py
  tests/test_dsge_adapter_gate.py tests/test_structural_particles.py` passed.
- `pytest -q tests/test_dsge_adapter_gate.py tests/test_structural_particles.py
  tests/test_backend_readiness.py tests/test_filter_metadata.py
  tests/test_macrofinance_adapter.py -q` passed, with the same two TensorFlow
  Probability deprecation warnings from the optional MacroFinance import path.
- `pytest -q` passed: 55 passed, with the same two TensorFlow Probability
  deprecation warnings.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` passed.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in
  `docs/` passed; `main.pdf` was already up to date.
- Stale-claim search for convergence/production labels found only label
  policies, blockers, or caution text, not a new promotion claim.

Final interpretation:
- BayesFilter now has executable gates or conservative blockers for every
  remaining original-plan gap.
- The original plan is not fully production-promoted, because DSGE client
  adapters, MacroFinance final-provider evidence, spectral client derivative
  certification, and real HMC chain diagnostics remain separate next-phase
  blockers.

## 2026-05-05 update: six next-issue closure pass

Context:
- User asked for a detailed plan using `docs/plans/templates` where possible,
  a reset-memo update, an independent audit, phase-by-phase execution, final
  commit, and next-phase hypotheses.
- The six issues were interpreted as the six hypotheses left by the previous
  closure pass: DSGE adapter evidence, particle Monte Carlo convergence,
  provider-owned MacroFinance masked metadata, cross-currency blockwise oracle
  evidence, SVD/eigen derivative policy, and HMC backend diagnostic comparison.

### S0: hygiene, plan, and audit

Plan:
- Use the local experiment-plan/result template headings.
- Keep pre-existing unrelated reset-memo and roadmap edits out of the scoped
  commit.

Execute:
- Added `docs/plans/bayesfilter-six-next-issue-closure-plan-2026-05-05.md`.
- Added `docs/plans/bayesfilter-six-next-issue-closure-audit-2026-05-05.md`.

Test:
- Manual audit confirmed all six issues are represented.

Audit:
- The plan avoids client-repo economics, production data, and real sampler
  claims.

Interpretation:
- S1 remains justified.

### S1: DSGE adapter evidence gate

Plan:
- Prove that SmallNK-like all-stochastic metadata passes while mixed
  Rotemberg/SGU-like metadata without completion fails closed.

Execute:
- Added `metadata_regime` to `DSGEStructuralAdapterGateResult`.
- Added all-stochastic and Rotemberg-like mixed-metadata tests.

Test:
- Focused six-issue test suite passed after implementation.

Audit:
- All-stochastic metadata now records `metadata_regime="all_stochastic"` and
  passes without a deterministic completion map.
- Mixed metadata records `metadata_regime="mixed_structural"` and remains
  blocked without a deterministic completion map.

Interpretation:
- S2 remains justified.
- DSGE client implementation is still blocked until `/home/chakwong/python`
  exposes real model adapters.

### S2: particle Monte Carlo convergence evidence

Plan:
- Add deterministic AR(2) evidence that increasing particle count improves
  likelihood error on average over a small seed panel.

Execute:
- Added a longer AR(2) particle-vs-Kalman seed-panel test.
- Kept identity diagnostics at every particle count.

Test:
- Focused structural particle tests passed.

Audit:
- The higher particle count has less than half the mean absolute likelihood
  error of the lower particle count under the fixed seed panel.
- The test does not claim a formal convergence rate or differentiability.

Interpretation:
- S3 remains justified.

### S3: provider-owned MacroFinance masked metadata

Plan:
- Distinguish provider-owned masked derivative metadata from caller overrides.
- Block caller overrides in production mode.

Execute:
- Added `masked_support_source` and `production_mode` to
  `LargeScaleAdaptationGateResult`.
- Added `production_mode` to `evaluate_large_scale_adaptation_gate`.
- Added tests for provider-owned support and production-mode caller override
  blocking.

Test:
- Focused MacroFinance adapter tests passed.

Audit:
- Caller overrides can still support non-production smoke tests.
- Sparse production readiness requires provider-owned
  `masked_derivative_order_supported`.

Interpretation:
- S4 remains justified.

### S4: cross-currency blockwise oracle evidence

Plan:
- Require named oracle blocks instead of accepting only an aggregate
  discrepancy.

Execute:
- Added required/checked/missing oracle block fields to
  `CrossCurrencyDerivativeGateResult`.
- Added `required_oracle_blocks` to `evaluate_cross_currency_derivative_gate`.
- Added tests for complete blockwise oracle evidence and missing-block
  fail-closed behavior.

Test:
- Focused MacroFinance adapter tests passed.

Audit:
- Dynamics, transition covariance, observation loadings, and measurement error
  can be required explicitly.
- A tiny aggregate oracle discrepancy no longer passes if required block names
  are absent.

Interpretation:
- S5 remains justified.

### S5: spectral derivative policy gate

Plan:
- Block small-gap spectral derivative paths by default.
- Allow explicitly declared non-spectral custom-gradient policy only with
  finite-difference and JVP/VJP checks.

Execute:
- Added `derivative_policy` to `SpectralDerivativeCertificationResult`.
- Added `derivative_policy` to `certify_spectral_derivative_region`.
- Added non-spectral small-gap certification test.

Test:
- Focused backend readiness tests passed.

Audit:
- Default spectral policy remains blocked near small gaps.
- `non_spectral_custom_gradient` can pass small-gap telemetry only when both
  numerical checks are declared, with warning label
  `small_gap_non_spectral_policy`.

Interpretation:
- S6 remains justified.

### S6: HMC backend diagnostic comparison

Plan:
- Compare supplied HMC diagnostics across named backends.
- Fail closed when any backend fails.

Execute:
- Added `MacroFinanceHMCBackendComparisonResult`.
- Added `compare_macrofinance_hmc_backend_diagnostics`.
- Exported comparison helper from `bayesfilter.adapters`.
- Added tests for all-pass and one-backend-fails cases.

Test:
- Focused MacroFinance adapter tests passed after correcting the comparison
  summary to use observed ESS and split-R-hat values rather than threshold
  fields.

Audit:
- All supplied backends must pass target/diagnostic gates before the comparison
  result is ready.
- A failing backend preserves `convergence_claim="not_claimed"`.
- The helper consumes diagnostics only; it does not run a sampler.

Interpretation:
- Final validation remains justified.

### Six next-issue closure validation

Results:
- Initial `PYTHONDONTWRITEBYTECODE=1 python -m py_compile ...` attempted to
  write under the read-only in-tree `__pycache__` path and failed with
  `Errno 30`; rerunning the same syntax check with
  `PYTHONPYCACHEPREFIX=/tmp/bayesfilter_pycache` passed.
- `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_dsge_adapter_gate.py
  tests/test_structural_particles.py tests/test_backend_readiness.py
  tests/test_filter_metadata.py tests/test_macrofinance_adapter.py -q -p
  no:cacheprovider` passed: 48 focused tests, with two TensorFlow Probability
  deprecation warnings from the optional MacroFinance import path.
- Final `pytest -q` passed: 63 tests, with the same two TensorFlow Probability
  deprecation warnings.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8')); print('source_map yaml ok')"` passed.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in `docs/`
  passed with `main.pdf` already up to date.
- Stale-claim search over `docs`, `bayesfilter`, and `tests` found label
  policies, blocker text, and tests asserting `not_claimed`/`blocked`, but no
  new unsupported convergence or production-readiness claim from this pass.

Final interpretation:
- The six BayesFilter-owned next issues are closed as executable gates and
  tests.
- Client promotion remains blocked for real DSGE adapters, real MacroFinance
  production providers, real non-spectral derivative implementations, and real
  HMC chain output.

## 2026-05-05 update: DSGE minimal-impact adapter stabilization handoff

Context:
- User approved planning for the DSGE client-repo issue with minimal effect on
  `/home/chakwong/python`, including a detailed handoff for another agent
  operating in that project.
- A read-only preflight first suggested DSGE metadata was missing, but a later
  verification against current `/home/chakwong/python` HEAD found commit
  `8645623 Expose DSGE structural metadata for BayesFilter`.

Plan:
- Treat the DSGE task as stabilization and evidence collection, not a broad
  implementation from scratch.
- Keep the adapter surface minimal: model-local metadata fields, a lightweight
  helper module, deterministic-completion bridges, and legacy full-state SVD
  guards.
- Do not move DSGE equilibrium logic, priors, parameter transforms, solvers, or
  HMC samplers into BayesFilter.

Execute:
- Added
  `docs/plans/bayesfilter-dsge-minimal-impact-adapter-stabilization-plan-2026-05-05.md`.
- Added
  `docs/plans/bayesfilter-dsge-minimal-impact-adapter-stabilization-audit-2026-05-05.md`.
- Registered both files in `docs/source_map.yml`.
- Read `/home/chakwong/python/src/dsge_hmc/models/structural_metadata.py`,
  SmallNK, Rotemberg, SGU, and
  `/home/chakwong/python/tests/contracts/test_structural_dsge_partition.py`.

Test:
- Read-only import preflight against `/home/chakwong/python/src` found:
  - `SmallNKEstimable` exposes all-stochastic BayesFilter metadata;
  - `RotembergNKEstimable` exposes mixed metadata plus a deterministic
    completion bridge;
  - `SGUEstimable` exposes mixed metadata plus a deterministic completion
    bridge.
- The import preflight produced TensorFlow/CUDA and Matplotlib cache warnings;
  these are environment noise, not adapter failures.

Audit:
- The handoff plan correctly avoids broad DSGE refactors.
- EZ and other unaudited DSGE models must remain fail-closed until a timing and
  structural-role audit exists.
- Legacy full-state SVD paths for mixed DSGE models must remain blocked by
  default or require explicit approximation labels.

Interpretation:
- The next justified work is for a `/home/chakwong/python` agent to run focused
  contract tests around commit `8645623`, fix only local metadata/guard
  regressions, and report the final commit hash and test output.
- BayesFilter should not proceed to DSGE structural particle or HMC promotion
  until the DSGE-focused test phase and BayesFilter gate confirmation are
  recorded.

## 2026-05-05 update: structural SVD six-phase validation closure

Context:
- User asked for six concrete phases motivated by the structural SVD handoff
  plan from another agent, followed by phase-by-phase execution, audit,
  reset-memo updates, and commit.
- The handoff was directionally valid but partially stale: BayesFilter already
  has structural partition/config/metadata, the structural SVD sigma-point
  reference backend, structural particle semantics, degenerate Kalman tests,
  and derivative policy gates.

### V0: hygiene and handoff reconciliation

Plan:
- Classify the handoff requests as already implemented, needing validation,
  BayesFilter-local, client-owned, or stale.
- Keep untracked assets out of scope unless explicitly requested.

Execute:
- Added
  `docs/plans/bayesfilter-structural-svd-six-phase-validation-plan-2026-05-05.md`.
- Added
  `docs/plans/bayesfilter-structural-svd-six-phase-validation-audit-2026-05-05.md`.
- Recorded current status: untracked `.claude/`, Julier PDF, the untracked
  structural SVD handoff, the new validation plan/audit, and
  `docs/plans/templates/`; no tracked BayesFilter code file was dirty.

Test:
- `git status --short --branch` completed.
- `git log -5 --oneline --decorate` completed.

Audit:
- The untracked Julier PDF and templates are not needed for this validation
  pass and should remain unstaged.

Interpretation:
- V1 remains justified.

### V1: BayesFilter structural core validation

Plan:
- Validate existing structural partition, sigma-point, AR(p), metadata, and
  particle semantics before considering code changes.

Execute:
- Ran the focused BayesFilter structural-core test group.

Test:
- `pytest -q tests/test_structural_partition.py
  tests/test_structural_sigma_points.py tests/test_structural_ar_p.py
  tests/test_filter_metadata.py tests/test_structural_particles.py` passed:
  19 tests.

Audit:
- Current structural SVD reference behavior already records approximation
  metadata and preserves deterministic completion in toy tests.
- No BayesFilter structural SVD rewrite is justified by this phase.

Interpretation:
- V2 remains justified.

### V2: exact Kalman and DSGE metadata validation

Plan:
- Validate exact/degenerate Kalman, derivative-smoke gates, BayesFilter DSGE
  adapter gates, and `/home/chakwong/python` structural metadata contracts.

Execute:
- Ran the BayesFilter exact/derivative/DSGE gate subset.
- Ran the focused DSGE client metadata contract test with BayesFilter on
  `PYTHONPATH`.
- Ran a read-only gate probe on SmallNK, Rotemberg, SGU, and EZ.

Test:
- `pytest -q tests/test_degenerate_kalman.py
  tests/test_derivative_validation_smoke.py tests/test_dsge_adapter_gate.py`
  passed: 8 tests.
- In `/home/chakwong/python`,
  `PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q
  tests/contracts/test_structural_dsge_partition.py` passed: 11 tests, with
  two TensorFlow Probability deprecation warnings and one pytest cache warning
  caused by the read-only client checkout.
- The read-only gate probe returned:
  - SmallNK: `adapter_ready=True`, `metadata_regime="all_stochastic"`;
  - Rotemberg: `adapter_ready=True`, `metadata_regime="mixed_structural"`;
  - SGU: `adapter_ready=True`, `metadata_regime="mixed_structural"`;
  - EZ: `adapter_ready=False`, `metadata_regime="missing"`.

Audit:
- DSGE metadata is now adapter-ready for SmallNK, Rotemberg, and SGU.
- This is not nonlinear structural promotion for Rotemberg or SGU.
- EZ correctly remains fail-closed pending timing/partition audit.

Interpretation:
- V3 remains justified.

### V3: blocker/register and source-map reconciliation

Plan:
- Update the blocker register and source map to reflect validation evidence
  without overpromoting DSGE structural filtering or HMC.

Execute:
- Updated
  `docs/plans/bayesfilter-original-plan-remaining-gap-blocker-register-2026-05-05.md`
  so the DSGE adapter pilot is now
  `client_metadata_ready_for_structural_tests` rather than
  `blocked_pending_client_metadata`.
- Kept explicit blockers for Rotemberg second-order/pruned identity evidence,
  SGU deterministic residual evidence, EZ timing/partition audit,
  derivative/JIT/HMC gates, and MacroFinance final-provider evidence.
- Registered the six-phase plan and audit in `docs/source_map.yml`.

Test:
- Source-map parse and final validation are deferred to V4.

Audit:
- The register no longer understates DSGE metadata progress.
- It also does not claim `filter-correct`, `sampler-usable`, or `converged`
  for DSGE targets.

Interpretation:
- V4 remains justified.

### V4: final validation

Plan:
- Run full repo validation and documentation/source-map checks after V0--V3.

Execute:
- Ran source-map YAML parse, whitespace check, stale-claim scan, full pytest,
  and LaTeX build.

Test:
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
  encoding='utf-8'))"` passed.
- `git diff --check` passed.
- Stale-claim scan over `docs/plans`, `docs/chapters`, `bayesfilter`, and
  `tests` found label policies, blocker text, and tests asserting
  `not_claimed`, but no new unsupported convergence or production-readiness
  claim from this pass.
- `pytest -q` passed: 63 tests, with two TensorFlow Probability deprecation
  warnings from the optional MacroFinance import path.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in `docs/`
  passed with `main.pdf` already up to date.

Audit:
- TensorFlow Probability warnings are deprecation warnings only.
- The client-repo pytest cache warning in V2 was caused by the read-only
  `/home/chakwong/python` checkout and did not affect assertions.

Interpretation:
- V5 remains justified.

### V5: commit and next hypotheses

Plan:
- Commit only scoped validation and reconciliation artifacts.
- Leave untracked handoff/assets/templates out of the commit unless explicitly
  requested.

Execute:
- Prepared scoped files for staging:
  - six-phase validation plan and audit;
  - blocker register update;
  - source-map registrations;
  - this reset-memo section.

Audit:
- The final commit should not include the Julier PDF, `.claude/`, the untracked
  handoff input, or `docs/plans/templates/`.

Final interpretation:
- BayesFilter structural core, exact/degenerate Kalman separation, and DSGE
  metadata adapter-readiness are validated.
- The next substantive phases are model-specific Rotemberg/SGU/EZ residual and
  timing evidence, then derivative/JIT gates, then HMC diagnostics.

## 2026-05-05 update: structural SVD 12-phase implementation handoff

Context:
- User asked for a detailed 12-phase plan with motivation, implementation
  instructions, reset-memo updates, independent audit, phase-by-phase execution,
  commit, and final summary.
- The immediately preceding user policy was that this session should be
  reserved for documentation only.
- Therefore this pass executes the documentation/audit/reset-memo phases and
  stops before backend implementation.  The plan itself requires a mathematical
  source audit before code changes.

Phase plan:
- Turn the final-goal and remaining-gap summary into a concrete 12-phase
  implementation roadmap.
- Write the plan in BayesFilter, where another coding agent will naturally look.
- Audit the plan as a separate developer.
- Update this reset memo with results and the stop decision.
- Commit only scoped planning artifacts, leaving unrelated untracked assets and
  other-agent dirty files alone.

Execution:
- Added
  `docs/plans/bayesfilter-structural-svd-12-phase-implementation-plan-2026-05-05.md`.
- Added
  `docs/plans/bayesfilter-structural-svd-12-phase-implementation-audit-2026-05-05.md`.
- Reused the previously moved handoff:
  `docs/plans/bayesfilter-structural-svd-code-implementation-handoff-plan-2026-05-05.md`.
- Did not edit BayesFilter backend code.
- Did not edit DSGE or MacroFinance code.

Test:
- Planned validation for this documentation-only pass:
  - `git diff --check`;
  - read-back/claim scan over the new plan and audit;
  - `git status --short --branch` to verify scoped staging.

Audit:
- The 12-phase plan covers:
  1. mathematical source audit;
  2. code reuse and migration audit;
  3. BayesFilter structural sigma-point core;
  4. exact Kalman and degenerate linear spine;
  5. generic structural fixtures;
  6. MacroFinance adapter and analytic derivative spine;
  7. DSGE adapter integration;
  8. model-specific DSGE completion evidence;
  9. derivative and Hessian safety gate;
  10. JIT and static-shape production gate;
  11. HMC validation ladder;
  12. documentation, provenance, and release gate.
- The independent audit approves the plan as a roadmap and explicitly blocks
  code implementation in this documentation-only session.
- Running Phase 1 as an actual mathematical source audit is justified next, but
  it should be its own BayesFilter execution pass with source retrieval and
  provenance artifacts.

Interpretation:
- The next phase is justified, but not inside this documentation-only session.
- The next coding/research agent should start with Phase 1 of the new plan:
  write the mathematical source audit before backend implementation.


## 2026-05-06 update: refined Proposition 19.1 assumptions around T_m and T_k

User raised a useful mathematical refinement about the assumptions behind the
first structural UKF proposition.  The core pushforward result does not require
full invertibility of `T_m`, but a stronger assumption is needed if the chapter
wants to say that one may replace `\varepsilon_t` by the current exogenous state
`m_t` as an equivalent sigma-point variable.  No comparable invertibility
assumption is needed on `T_k`.

### Focused assumption-refinement pass

Phase plan:
- keep the pushforward result under minimal assumptions;
- add a stronger assumption only for the optional `m_t`-instead-of-`\varepsilon_t`
  reparameterization claim;
- make explicit that no analogous invertibility restriction is needed on `T_k`.

Execution:
- Expanded the accepted assumptions block before Proposition 19.1.
- Added an explicit statement that no invertibility of `T_k` is required so long
  as it is a measurable deterministic completion map.
- Added a separate stronger assumption for the optional reparameterization of the
  sigma-point variable from `\varepsilon_t` to `m_t`: for fixed `m_{t-1}`, the
  map `\varepsilon_t \mapsto T_m(m_{t-1},\varepsilon_t)` should be injective or
  invertible onto its image with measurable inverse.
- Added a clarifying paragraph after the formal statements stating that this
  extra assumption is only needed for the reparameterization claim, not for the
  core pushforward identities themselves.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.

Audit:
- Proposition 19.1 is now more precise mathematically.
- It cleanly separates what is needed for the core law-based result from what is
  only needed for the stronger claim that sigma points may be parameterized by
  `m_t` instead of `\varepsilon_t`.
- This reduces the chance that a careful reader thinks the main pushforward
  result depends on an unnecessarily strong invertibility assumption.


## 2026-05-06 update: broad Chapter 18b audit and restructuring plan creation

User asked for a broad audit of the whole chapter, including restructuring,
literature/citation strengthening, degeneracy/SVD distinctions, algorithmic
explicitness, theorem/assumption hygiene, and worked-example ordering -- not
just another local repair.

### Broad planning-and-audit pass

Phase plan:
- audit the entire chapter for duplicated arguments, unclear flow, weak
  literature support, mixed evidence classes, and algorithmic ambiguity;
- review the Julier (1996) source PDF and nearby BayesFilter chapters to map
  what each literature/source class actually supports;
- write an explicit execution plan under `docs/plans` that another agent could
  use to perform a full rewrite.

Execution:
- Performed a broad chapter audit and identified the main structural weaknesses:
  - accumulated patch-like growth rather than one spine;
  - weak literature mapping;
  - under-specified algorithm sections for implementation use;
  - repeated doctrine in multiple places;
  - blurred distinctions between structural-law error, quadrature error,
    numerical factorization issues, and derivative/HMC issues.
- Re-read the Julier (1996) PDF and confirmed it is the right spine for:
  - unscented-transform mechanics;
  - sigma-point construction;
  - additive-noise augmentation lineage;
  - approximation-order framing.
- Wrote the explicit plan:
  - `docs/plans/ch18b-restructuring-and-literature-strengthening-plan-2026-05-05.md`
- Tightened that plan after an independent-style audit by adding:
  - a mandatory Phase 0 inventory/pruning/classification pass;
  - UKF-variant discipline;
  - theorem-budget / assumption-hygiene rules;
  - stronger reset-memo and provenance obligations;
  - explicit example-order and anti-duplication checks.
- Added a bibliography entry for the local Julier 1996 source as
  `julier1996general` in `docs/references.bib` so the planned literature rewrite
  has a stable cite key available.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- The new bibliography entry was added without breaking the monograph build.

Audit:
- The chapter now has a concrete, chapter-level rewrite plan rather than a
  series of local fixes.
- The plan is broad enough to cover the reviewers' main criticisms and several
  adjacent issues that would otherwise resurface during implementation.
- The next justified phase is execution of the restructuring plan itself.


## 2026-05-06 update: Phase 0 inventory, pruning, and claim classification for Chapter 18b rewrite

User requested that the broad Chapter 18b restructuring plan now be executed
phase by phase, with an explicit `plan -> execute -> test -> audit -> tidy ->
update reset memo` cycle and continuation unless a real blocker appears.

### Phase 0: inventory, pruning, and claim classification

Phase plan:
- inventory the current chapter structure, including sections, subsection
  sequence, theorem-like blocks, algorithm blocks, comparison tables, and
  worked examples;
- classify repeated or overloaded parts before any large rewrite begins;
- decide whether the next rewrite phase remains justified.

Execution:
- Inspected the full current chapter and enumerated the main structural blocks:
  - production warning;
  - structural split;
  - linear-Kalman-versus-nonlinear distinction;
  - structural filtering step algorithm;
  - degenerate-transition discussion;
  - UKF comparison, formal statement, and worked nonlinear-state example;
  - degenerate linear-transition + nonlinear-measurement formal section and
    worked example;
  - pruning / adapter implications;
  - source-project failure mode;
  - required tests;
  - common misunderstandings;
  - final policy rule.
- Classified the main current weaknesses:
  - duplicated UKF doctrine across prose, formal statements, and examples;
  - theorem/proposition overgrowth in the nonlinear-measurement section;
  - literature support too implicit relative to chapter ambition;
  - useful but dispersed distinction between structural degeneracy, numerical
    regularization, and SVD/spectral derivative issues;
  - algorithm sections improved but still not yet fully code-generation-ready in
    a literature-grounded form.
- Classified the major claim types now present in the chapter as a mix of:
  - exact derivation;
  - accepted assumption;
  - source-backed literature claim;
  - BayesFilter implementation policy;
  - toy numerical illustration;
  - source-project audit evidence.
- Concluded that the chapter is ready for a real rewrite, not another patch.

Tests:
- Verified the current chapter structure programmatically by listing sections,
  subsection headings, proposition blocks, and framed algorithm blocks.
- No build-breaking issue was introduced in this inventory phase.

Audit:
- Phase 0 passes.
- The chapter now has a clear pruning/rewrite target list.
- No blocker was discovered that would make the restructuring pass unjustified.
- Phase 1 rewrite remains justified.


### Phase 1A: rebuild the UKF spine around literature and explicit algorithms

Phase plan:
- remove the duplicated introductory UKF explanation that had accumulated around
  the worked example;
- introduce a clearer UKF narrative spine that starts from the original
  unscented-transform pattern and then derives the standard additive-noise and
  structural UKF algorithms explicitly;
- preserve the existing detailed formal and numerical content for later phases.

Execution:
- Replaced the old `Worked Structural UKF Example` opening block as the first
  UKF-facing entry point.
- Added a new section:
  - `Standard UKF, Structural UKF, and the Predictive Law`
- Added a new subsection:
  - `The Original Unscented-Transform Pattern`
  grounded explicitly in `julier1996general` and `julier1997new`.
- Kept the explicit `Standard Additive-Noise UKF Algorithm` and
  `Structural UKF Algorithm` sections, but moved them under the new UKF section
  so the chapter now introduces the literature mechanics before the structural
  contrast.
- Removed the duplicated high-level UKF contrast prose that previously sat in
  front of the worked example and repeated points now stated in the algorithm and
  proposition sections.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.

Audit:
- Phase 1A passes.
- The UKF half of the chapter now begins from the literature mechanism rather
  than from the local worked example, which is a much better pedagogical entry
  point.
- No blocker was found; the next rewrite chunk remains justified.


### Phase 1D: stabilize the rewritten section and clean malformed insertions

Phase plan:
- clean malformed scripted insertions that broke the build during the rewrite;
- restore build stability before continuing further restructuring;
- update the reset memo with what was fixed and whether the next phase remains
  justified.

Execution:
- Found and repaired remaining malformed literal backslash-n insertions in the
  `What Structural Correctness Does and Does Not Guarantee` section.
- Re-ran the monograph build from a clean `latexmk -C` state to ensure the
  rewritten chapter no longer depended on stale auxiliary files.
- Confirmed that the rewritten UKF and nonlinear-measurement sections now compile
  successfully in the current chapter state.

Tests:
- `latexmk -C`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Build completed successfully and produced an updated PDF.

Audit:
- Phase 1D passes.
- The remaining issues are now ordinary chapter-level warning cleanup and the
  continuation of the larger restructuring, not malformed-text blockers.
- The next rewrite phase remains justified.


### Phase 1E: tighten chapter-local readability in the new doctrinal sections

Phase plan:
- reduce local readability problems that arose from the rewritten doctrinal
  sections while preserving the new distinctions;
- keep the chapter buildable before moving on to the remaining restructuring
  phases.

Execution:
- Reflowed the `What Structural Correctness Does and Does Not Guarantee` section
  into shorter lines and clearer list items.
- Tightened the `Structural Degeneracy Versus Numerical Degeneracy` table by
  shortening headers and cell text while preserving the doctrinal distinction.
- Preserved the explicit statement that an SVD sigma-point backend may solve a
  factorization problem without solving a structural-law problem.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.

Audit:
- Phase 1E passes.
- The chapter remains in a stable build state after the doctrinal rewrite.
- The remaining work is now the continuation of the larger chapter restructure,
  not local emergency cleanup.


### Phase 1F: restructure the late chapter around adapters, source lesson, and validation

Phase plan:
- tighten the late chapter so it no longer reads as a sequence of disconnected
  postscript sections;
- align the pruning/adapters discussion, source-project lesson, and validation
  gates with the new doctrinal spine;
- reduce repeated policy restatement at the very end.

Execution:
- Renamed and reframed the late sections so the chapter now ends with:
  - `Pruned DSGE and Adapter Implications`
  - `Source-Project Lesson`
  - `Validation Gates and Final Policy Rule`
- Shortened the adapter section so it now emphasizes the two bookkeeping axes:
  perturbation order and structural timing.
- Tightened the source-project lesson into a bounded concrete warning rather than
  a longer project-specific postmortem.
- Folded the old `Required Tests` and `Common Misunderstandings` material into a
  cleaner validation-gate ending with a single final policy rule.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.

Audit:
- Phase 1F passes.
- The chapter ending now fits the earlier restructuring spine much better and is
  less repetitive.
- The remaining work is primarily literature-strengthening, final global audit,
  and a last pass for coherence rather than major structural surgery.


## 2026-05-06 update: reviewer-response Chapter 18b rewrite execution

User requested a full execution pass against
`docs/plans/ch18b-reviewer-response-rewrite-plan-2026-05-06.md`, with the
reset memo updated before work starts, then after every phase in a
plan/execute/test/audit/tidy/update-memo cycle.

### Pre-flight plan audit as an independent developer

Phase plan:
- inspect the new reviewer-response plan, current Chapter 18b, bibliography,
  local PDFs, and git state;
- audit whether the plan misses reviewer concerns or contains unsafe execution
  assumptions;
- decide whether the rewrite remains justified before editing the chapter.

Execution:
- Confirmed the plan directly maps to the reviewer issues:
  - code-generation-grade algorithms;
  - original/naive UKF review;
  - standard-versus-structural UKF comparison;
  - UT approximation-order proof;
  - notation definitions for predictive law, pushforward, and sigma-point
    placement;
  - proof hygiene;
  - readability and flow;
  - literature/SVD answer;
  - `\phi=0` edge case.
- Rechecked the current chapter structure and found that the plan's theorem
  hygiene concern is real: the chapter still contains six `proposition`
  environments, with five in the example-specific exp-affine section.
- Rechecked local source availability:
  - Julier--Uhlmann (1996) local PDF is readable with `pdftotext`;
  - van der Merwe / Wan local PDF is readable with `pdftotext`, but its
    bibliographic identity needs care because the local file is a short paper
    while the stable dissertation metadata is 2004.
- Verified via OHSU metadata that the dissertation
  `Sigma-point kalman filters for probabilistic inference in dynamic
  state-space models` is a 2004 Ph.D. dissertation by Rudolph van der Merwe;
  the local paper-style PDF is still useful for the SPKF/UKF notation cited in
  this chapter.
- Checked git state and found pre-existing untracked files:
  - the two local PDFs;
  - `docs/plans/bayesfilter-structural-svd-six-blocker-closure-plan-2026-05-06.md`;
  - `docs/plans/templates/`;
  - the newly created reviewer-response plan.

Test:
- No source files were edited in this pre-flight phase.
- Label-reference search found that the chapter-level label is used outside the
  chapter, while most section/proposition labels are chapter-local or plan-local.

Audit:
- The plan is justified and should be executed.
- Missing point found and added to the execution interpretation: add a stable
  bibliography entry for the van der Merwe dissertation/short source before
  relying on it in Chapter 18b.
- The rewrite should preserve the chapter label and important equation labels
  where practical, but it may downgrade example-specific proposition labels to
  paragraph/remark anchors because the plan explicitly requires this.
- The next phase is justified: definitions, notation, and literature-source
  setup.

### Phase A: definitions, source setup, and notation

Phase plan:
- add stable bibliography support for the van der Merwe sigma-point source;
- introduce formal notation for structural state, predictive law, pushforward,
  and sigma-point placement before the chapter relies on those terms;
- tighten the opening warning so it states what the chapter proves and what
  remains an approximation boundary.

Execution:
- Added `vandermerwe2004sigma` to `docs/references.bib` as a Ph.D. dissertation
  entry for Rudolph van der Merwe's
  `Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic
  State-Space Models`.
- Rewrote the opening production warning to state the chapter's law-level
  result: mixed structural prediction targets the pushforward of lagged-state
  and innovation laws through the structural transition.
- Added Definition `def:bf-structural-transition-notation`, defining:
  - `x_t=(m_t,k_t)`;
  - stochastic block `m_t`;
  - deterministic-completion block `k_t`;
  - structural map `F_\theta`;
  - filtering/predictive laws `\pi_{t-1|t-1}^x`, `\pi_t^a`,
    `\pi_{t|t-1}^x`;
  - pushforward `(F_\theta)_#\pi_t^a`;
  - the operational meaning of placing sigma points on a variable.
- Preserved the existing core transition equation labels:
  - `eq:bf-m-transition`;
  - `eq:bf-k-transition`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted search confirmed the new definitions and bibliography key are
  present.

Audit:
- Phase A passes.
- The chapter now defines the terms that reviewers found too abstract before it
  uses them heavily.
- Remaining issue: BibTeX still warns that `julier1996general` is an article
  with no journal; this is not a correctness blocker, but Phase B should tidy it
  by converting the entry to a technical report.
- The next phase remains justified: literature-grounded UKF algorithm rewrite.

### Phase B: literature-grounded UKF algorithms

Phase plan:
- expand the original unscented-transform review using Julier--Uhlmann and
  van der Merwe notation;
- make the standard additive-noise UKF and structural UKF algorithms parallel
  and implementation-ready;
- add an exact comparison table before the formal proposition;
- tidy the Julier 1996 bibliography warning.

Execution:
- Converted `julier1996general` from an `@article` with no journal to a
  `@techreport` entry.
- Expanded the original unscented-transform subsection with:
  - scaled sigma-point locations;
  - scaled UKF weights;
  - transformed mean and covariance formulas;
  - a source-backed distinction between UT mechanics and the structural-law
    question.
- Added `vandermerwe2004sigma` citation support to the SPKF/UKF discussion.
  Public OHSU metadata confirms the dissertation title, author, 2004 date,
  content type, school, and DOI `10.6083/M4Z60KZ5`; the local PDF remains the
  working notation source for this pass.
- Rewrote the standard additive-noise UKF algorithm so it now specifies:
  inputs, augmented variable, sigma-point generation, state propagation,
  observation propagation, innovation moments, Gaussian update/log likelihood,
  and returned metadata.
- Rewrote the structural UKF algorithm in parallel form, specifying:
  inputs, pre-transition variable, stochastic-block propagation,
  deterministic completion, observation moments, update/log likelihood, and
  returned metadata.
- Added a direct comparison table distinguishing filtered object, sigma-point
  variable, direct noise, deterministic-coordinate treatment, target law,
  failure mode, and SVD/square-root role.

Tests:
- Initial build failed because literal metadata text with an underscore was
  inserted in LaTeX text.  This was repaired by using the existing `\code{...}`
  macro.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` then
  succeeded.
- Targeted log search found no fatal errors, missing-dollar errors, undefined
  citations, undefined references, or BibTeX warnings after rerun.

Audit:
- Phase B passes.
- The two UKF algorithms are now explicit enough for a coding agent to identify
  the random variable, point cloud, propagation map, moment formulas, update
  equations, and metadata boundary.
- The comparison now states the exact difference: the update equations are
  shared, while the sigma-point variable and predictive cloud differ.
- The next phase remains justified: formal proposition cleanup and
  example-specific theorem demotion.

### Phase C: formal statements and proof hygiene

Phase plan:
- keep a small theorem budget in the main UKF section;
- preserve one pushforward proposition and add one moment-accuracy proposition;
- demote example-specific theorem-like blocks into worked identities,
  derivations, and diagnostics;
- use MathDevMCP to audit labels/provenance and record what the tool can and
  cannot certify.

Execution:
- Kept the main pushforward statement as
  `prop:bf-structural-ukf-pushforward`.
- Added `prop:bf-structural-ukf-ut-accuracy`, now stated as a local Taylor
  expansion result:
  - the structural UKF transformed mean matches the true transformed mean
    through the quadratic Hessian term;
  - the structural UKF covariance matches the leading
    `J P_a J^\top` term;
  - both claims are explicitly for the correct pre-transition input law
    `(x_{t-1},\varepsilon_t)`.
- Demoted the example-specific proposition ladder into paragraph-level worked
  derivations and diagnostics while preserving labels for cross-reference
  stability:
  - `lem:bf-exp-affine-latent-pushforward`;
  - `lem:bf-exp-affine-deterministic-completion`;
  - `lem:bf-exp-affine-observation-pushforward`;
  - `lem:bf-exp-affine-naive-full-state-support-violation`;
  - `prop:bf-exp-affine-law-mismatch-propagation`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted log search found no fatal errors, missing-dollar errors, undefined
  citations, undefined references, or BibTeX warnings after the final rerun.
- Theorem count search shows only two `proposition` environments remain in
  Chapter 18b, both in the main UKF section.

Audit:
- Phase C passes the plan criteria.
- MathDevMCP `typed_obligation_label` returned `consistent` for both main
  proposition labels, with usable file/line provenance.
- MathDevMCP `audit_derivation_v2_label` did not provide a backend proof
  certificate.  It routed the obligations to human review / manual
  formalization, which is expected for measure-pushforward notation and
  Taylor-order claims.  Interpretation: MathDevMCP is useful here as a
  structured audit and provenance tool, not as a complete formal verifier for
  this chapter without further formalization.
- The next phase remains justified: examples still need direct `\phi=0`
  treatment and numerical recomputation.

### Phase D: worked examples and reviewer edge case

Phase plan:
- give the nonlinear toy model its own worked-example heading;
- answer the reviewer question about `\phi=0` directly;
- recompute the displayed numbers in the nonlinear-state example and the
  degenerate-linear-transition/nonlinear-measurement example;
- keep the second example focused on the distinction between latent-law error
  and observation-side quadrature error.

Execution:
- Added `Worked Example A: Nonlinear Structural Transition`.
- Added a direct `Reviewer edge case: \phi=0` paragraph:
  - when `\phi=0`, `k_t=\gamma m_t^2`;
  - the lagged-`k` channel disappears, but current-shock randomness remains
    whenever `\gamma\neq0` and `m_t` is nondegenerate;
  - the support becomes the manifold `{(m,k): k=\gamma m^2}`;
  - collapse to a point requires a further degeneracy such as `\gamma=0` or
    degenerate `m_t`.
- Recomputed and inserted the `\phi=0` UKF moments:
  - `P_{xx,t}(2,2)=0.04247`;
  - `S_t=0.56807`;
  - `P_{xz,t}=(0.27560,0.04247)^\top`.
- Renamed the later section to
  `Worked Example B: Degenerate Linear Transition with Nonlinear Measurement`.
- Recomputed the second worked example from the stated transition and
  measurement parameters.  This corrected stale values in:
  - the latent point table;
  - `P_{xx,t}`;
  - `\hat z_t`, `S_t`, and `P_{xz,t}`;
  - structural and naive gains/posterior means.
- Made the naive nonlinear-measurement comparison reproducible by defining the
  artificial four-dimensional sigma-point variable
  `\tilde a_t=(x_{t-1},\varepsilon_t,\delta_t)`,
  with `\delta_t\sim N(0,0.04)`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Targeted log search found no fatal errors, missing-dollar errors, undefined
  citations, undefined references, or BibTeX warnings after the final rerun.
- Stale-number sweep found none of the old second-example values.
- Targeted search confirmed the `\phi=0` edge-case paragraph, manifold
  explanation, and recomputed values are present.

Audit:
- Phase D passes.
- Interpretation of the edge case: the reviewer is right that setting
  `\phi=0` removes the lagged endogenous-state propagation channel, but it does
  not collapse the predictive law unless the remaining stochastic map is also
  degenerate.
- The second example now better supports the intended claim: even when the
  latent transition is linear, a direct perturbation of a deterministic
  coordinate changes the latent law before observation-side quadrature is
  considered.
- The next phase remains justified: the chapter still needs a stronger
  literature map, direct SVD answer, and tighter adapter/validation closure.

### Phase E: literature, SVD objection, adapter, and validation close

Phase plan:
- add a source-backed literature map that ties each cited source class to the
  specific claim it supports;
- answer the reviewer's SVD sigma-point objection directly;
- tighten adapter-facing implications and validation gates so implementation
  work has an auditable target.

Execution:
- Added `sec:bf-structural-literature-map`, separating:
  - state-space likelihood / Bayesian DSGE sources;
  - pruned nonlinear DSGE state-space sources;
  - Julier--Uhlmann / van der Merwe unscented and sigma-point sources;
  - particle-filter and factor-backend sources.
- Added `ionescu2015matrixbackprop` to `docs/references.bib` for the
  spectral-gap derivative-risk claim.
- Strengthened the early SVD answer:
  - SVD can be the right factorization backend for a structural sigma-point
    rule;
  - it solves the reviewer's degeneracy concern only if the sigma points are
    placed on the correct structural integration variable and deterministic
    coordinates are completed pointwise;
  - if a backend first invents a full-rank post-transition covariance for
    deterministic-completion coordinates, SVD merely represents the altered
    law more stably.
- Renamed and expanded the degeneracy section to
  `Structural Degeneracy, Numerical Degeneracy, and SVD`.
- Added the explicit diagnostic question:
  `Which law is being factored and propagated?`
- Connected SVD/eigenspectral HMC promotion to spectral-gap, finite-gradient,
  and compiled-parity evidence.
- Tightened adapter implications:
  - dimensions/names for filtered, innovation, and deterministic blocks;
  - pointwise structural map from `(x_{t-1},\varepsilon_t)` to `x_t`;
  - deterministic-identity residuals on propagated points;
  - observation map after structural completion;
  - approximation label for legacy full-state paths.
- Tightened validation gates with:
  - `SVD/factorization test`;
  - `Derivative-promotion test`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- Initial LaTeX passes reported the new citation as undefined until BibTeX and
  reruns completed; the final targeted log search found no fatal errors,
  undefined citations, undefined references, rerun warnings, or BibTeX
  warnings.
- Targeted search confirmed the new literature map, SVD diagnostic question,
  Matrix Backprop citation key, factorization gate, and derivative-promotion
  gate.

Audit:
- Phase E passes.
- The SVD answer is now more precise than a rejection: SVD/square-root methods
  address factorization and numerical degeneracy; structural correctness is
  about the law being factorized and propagated.
- The chapter now ties the structural UKF and structural SVD story back to the
  rest of the monograph instead of letting Chapter 18b drift as an isolated
  warning.
- The next phase remains justified: perform final global audit, tidy scoped
  files, update memo, and commit.

### Phase F: final audit, tidy, memo, and commit

Phase plan:
- run final LaTeX and targeted log checks;
- run theorem-count, stale-number, citation-key, and reviewer-concern sweeps;
- inspect git scope and avoid staging unrelated user artifacts;
- update this reset memo with results, interpretation, and next hypotheses;
- commit the scoped rewrite files.

Execution:
- Added short optional section titles for long Chapter 18b section names so the
  running PDF headers are less likely to overflow.
- Left unrelated untracked artifacts unstaged:
  - local Julier and van der Merwe PDFs in `docs/`;
  - `docs/plans/templates/`.
- Included the new reviewer-response plan in the commit scope because this
  reset memo points to it as the controlling execution plan.

Tests:
- Final `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  succeeded from `docs/`.
- Final targeted searches confirmed:
  - only two `proposition` environments remain in Chapter 18b;
  - the literature map, `\phi=0` edge case, SVD diagnostic question, and new
    validation gates are present;
  - stale second-example numerical values are absent;
  - no `TODO`, `FIXME`, reviewer complaint residue, or placeholder `??`
    remains in the chapter/plan/memo surfaces checked.
- Final log search found no fatal errors, missing-dollar errors, undefined
  citations, undefined references, BibTeX warnings, citation-rerun warnings, or
  cross-reference rerun warnings.
- Residual LaTeX warnings are cosmetic overfull/underfull boxes, including a
  few compact Chapter 18b table/long-term lines and older chapters.

Audit:
- Phase F passes.
- The rewrite plan has been executed through all phases.  No blocking result
  was found that would make the next phase unjustified.
- MathDevMCP helped as an audit/provenance tool, but the proposition proofs
  remain human mathematical arguments rather than machine-certified formal
  proofs.
- ResearchAssistant currently has no local summary records for the Julier /
  van der Merwe sources, so the rewrite used direct PDF metadata/text checks
  plus existing monograph bibliography sources.

Completion interpretation:
- Chapter 18b now answers the reviewer concerns directly:
  - it defines predictive law, pushforward, and sigma-point placement;
  - it presents the standard additive-noise UKF and structural UKF in
    implementation-grade steps;
  - it compares the two algorithms explicitly;
  - it states and proves a local second-order UT inheritance result for the
    correct structural input law;
  - it answers the `\phi=0` objection;
  - it distinguishes structural-law correctness, quadrature accuracy,
    factorization/SVD robustness, and derivative/HMC readiness.

Next hypotheses to test:
- H1: A small executable Chapter 18b regression script can reproduce every
  displayed number in both worked examples and should be added before the next
  review draft.
- H2: A structural SVD sigma-point backend that factors the covariance of
  `(x_{t-1},\varepsilon_t)` and completes `k_t` pointwise will match the
  chapter's structural UKF/cubature examples, while a legacy full-state
  perturbed backend will fail the deterministic-identity residual test.
- H3: In mixed DSGE adapters, metadata based only on zero rows of a shock-impact
  matrix will misclassify at least one endogenous/predetermined coordinate; a
  model-supplied structural map and pointwise residual test should catch it.
- H4: SVD/eigenspectral value paths can pass finite likelihood tests while
  failing derivative promotion near small spectral gaps; spectral-gap telemetry
  and finite-gradient stress tests should remain mandatory before HMC claims.
- H5: Rotemberg/SGU/EZ-style pruned examples need separate first-order and
  second-order/pruned deterministic-completion residual gates; first-order
  structural success should not be promoted to pruned-order correctness without
  those tests.
