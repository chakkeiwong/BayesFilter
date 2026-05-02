# Plan: Continue BayesFilter Monograph Writing

## Date
2026-05-03

## Purpose

Continue writing the BayesFilter monograph after the scaffold, source map,
Phase 2B literature gate, and TFP NUTS Gaussian benchmark. This plan is for
consolidation and source-backed drafting only. It must not turn BayesFilter into
a copy-paste archive of the source projects, and it must not introduce new
technical claims that are not tied to local source documents, reviewed
literature, code, tests, or explicitly recorded project decisions.

## Current State

Repository:
- `/home/chakwong/BayesFilter`

Already available:
- Buildable LaTeX book skeleton under `docs/`.
- Detailed provenance map: `docs/source_map.yml`.
- Consolidation plan:
  `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`.
- Reset memo:
  `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.
- Phase 2B literature-gate result:
  `docs/plans/bayesfilter-phase2b-literature-gate-result-2026-05-02.md`.
- Literature seed list:
  `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`.
- TFP HMC/NUTS Gaussian benchmark:
  `docs/benchmarks/benchmark_tfp_nuts_gaussian.py`.

Important settled point:
- TFP NUTS is not the default BayesFilter production fix hypothesis. The
  Gaussian benchmark showed that TFP NUTS can XLA-compile on a toy target, but
  even there it is materially heavier than fixed-step HMC. The monograph may
  discuss NUTS foundations and use TFP NUTS as a reference/diagnostic backend,
  but the production path should remain a small, inspectable, filter-aware HMC
  layer unless later evidence changes this decision.

## Non-Negotiable Writing Rules

1. Original source projects are read-only:
   - `~/python/docs`
   - `~/latex/CIP_monograph`
   - `~/MacroFinance`
2. Write only under `~/BayesFilter/docs`.
3. Every drafted section must cite its provenance in either:
   - `docs/source_map.yml`,
   - a `docs/plans/*result*.md` note,
   - a local code/test reference,
   - a reviewed literature note.
4. Do not copy source chapters wholesale. Rewrite into BayesFilter notation and
   BayesFilter scope.
5. Do not commit generated PDFs or LaTeX byproducts.
6. Keep labels namespaced with BayesFilter-style prefixes:
   - `ch:bf-*`, `sec:bf-*`, `eq:bf-*`, `alg:bf-*`, `tab:bf-*`.
7. Do not claim that any filter is HMC-safe unless the chapter states:
   - log-target contract,
   - derivative policy,
   - finite-value and failure policy,
   - shape/static-compilation policy,
   - validation diagnostics.
8. Treat smoke tests as engineering checks, not convergence proof.

## Writing Strategy

Write the monograph from the inside out:

1. Establish notation and state-space contracts.
2. Draft the exact linear Gaussian likelihood spine.
3. Draft analytic derivative policy and validation before nonlinear filters.
4. Draft HMC requirements and implementation constraints after the derivative
   policy exists.
5. Draft nonlinear/SVD/particle material only after the exact and analytic
   derivative spine is clear.
6. Add case studies last, and keep project-specific economics in the source
   projects.

This order minimizes unsupported claims. It also gives BayesFilter a reusable
industrial core before discussing more fragile nonlinear or DSGE paths.

## Phase W0: Hygiene and Baseline Checkpoint

Goal:
- Start writing from a clean, reproducible state.

Tasks:
- Inspect `git status --short --ignored`.
- Confirm current uncommitted NUTS benchmark/doc changes are either committed
  or intentionally carried into the next writing branch.
- Build the monograph with `latexmk`.
- Parse `docs/source_map.yml`.
- Run `git diff --check`.

Commands:

```bash
cd /home/chakwong/BayesFilter
git status --short --ignored
python - <<'PY'
import yaml
with open('docs/source_map.yml', encoding='utf-8') as f:
    yaml.safe_load(f)
print('source_map yaml ok')
PY
git diff --check
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Pass criteria:
- Build succeeds.
- Source map parses.
- No whitespace errors.
- Dirty state is understood and recorded.

Memo update:
- Append the checkpoint result to
  `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.

## Phase W1: Notation, Contracts, and Reader Map

Primary chapters:
- `docs/chapters/ch01_introduction.tex`
- `docs/chapters/ch02_state_space_contracts.tex`
- `docs/chapters/ch03_hmc_target_requirements.tex`
- `docs/chapters/ch04_bayesfilter_api.tex`
- `docs/appendices/app_a_notation.tex`
- `docs/appendices/app_f_source_map.tex`

Source inputs:
- `docs/source_map.yml`
- `~/python/docs/monograph.tex`
- `~/latex/CIP_monograph/main.tex`
- state-space and HMC interface material listed in `docs/source_map.yml`

Drafting tasks:
- Explain BayesFilter scope: reusable filtering and HMC-safe likelihood layer.
- Define the generic state-space interface:
  - latent transition,
  - observation map,
  - parameter transform,
  - data mask/missing-data interface,
  - filter backend,
  - derivative provider.
- Define the target contract:
  - log prior,
  - log likelihood,
  - posterior target,
  - finite value on valid parameter region,
  - explicit invalid-state return policy.
- Define notation and shape conventions once.
- Add a reader map explaining why linear Gaussian and analytic derivative
  chapters come before nonlinear/SVD chapters.

Audit questions:
- Does every symbol have a single meaning?
- Are DSGE/CIP examples clearly marked as downstream applications?
- Does the API chapter avoid promising implementation features not yet present?

Pass criteria:
- Part I builds.
- Labels use BayesFilter prefixes.
- Source map updated from `candidate` to `drafted` only for sections actually
  consolidated.

## Phase W2: Linear Gaussian Likelihood Spine

Primary chapters:
- `docs/chapters/ch05_prediction_error_decomposition.tex`
- `docs/chapters/ch06_stable_linear_filtering.tex`
- `docs/chapters/ch07_missing_data_mixed_frequency.tex`
- `docs/chapters/ch08_large_scale_lgssm.tex`

Source inputs:
- `~/MacroFinance/analytic_kalman_derivatives.tex`
- `~/python/docs/chapters/ch08_kalman_filter.tex`
- `~/latex/CIP_monograph/chapters/ch16_kalman_filter.tex`
- `~/latex/CIP_monograph/chapters/ch18_mixed_frequency.tex`
- MacroFinance differentiated Kalman implementations and tests listed in
  `docs/source_map.yml`

Drafting tasks:
- Write exact prediction-error decomposition in BayesFilter notation.
- Distinguish covariance, Cholesky, QR, and solve-form implementations.
- Write missing-data and mixed-frequency handling as contracts, not as
  application-specific model prose.
- Add complexity and static-shape discussion for large LGSSMs.
- Explicitly separate mathematical likelihood from implementation backend.

Audit questions:
- Are all likelihood formulas compatible with the notation in Phase W1?
- Are SPD, rank, finite-value, and shape assumptions explicit?
- Are missing-data masks handled without changing the latent model definition?

Pass criteria:
- Linear Gaussian chapters build.
- No Hessian/score claims are introduced before Phase W3.
- Code references are added to `docs/source_map.yml`.

## Phase W3: Analytic Derivatives and Custom Gradient Policy

Primary chapters:
- `docs/chapters/ch09_kalman_score.tex`
- `docs/chapters/ch10_kalman_hessian.tex`
- `docs/chapters/ch11_structural_derivatives.tex`
- `docs/chapters/ch12_factor_derivatives.tex`
- `docs/chapters/ch13_custom_gradient_wrappers.tex`
- `docs/chapters/ch14_derivative_validation.tex`
- `docs/appendices/app_b_matrix_calculus.tex`
- `docs/appendices/app_c_factor_derivative_proofs.tex`
- `docs/appendices/app_d_mathdevmcp_workflows.tex`

Source inputs:
- `~/MacroFinance/analytic_kalman_derivatives.tex`
- `~/MacroFinance/filters/differentiated_kalman.py`
- `~/MacroFinance/filters/solve_differentiated_kalman.py`
- `~/MacroFinance/filters/qr_sqrt_differentiated_kalman.py`
- `~/MacroFinance/filters/tf_qr_sqrt_differentiated_kalman.py`
- MacroFinance derivative and large-scale LGSSM tests.
- Kitagawa Kalman score/Hessian source from Phase 2B.

Drafting tasks:
- Consolidate score recursions conservatively.
- Consolidate Hessian/observed-information recursions only where derivation and
  code parity can be checked.
- Define structural derivatives from parameter transforms to state-space
  matrices.
- Explain why custom gradients must return the derivative of the same log
  target being sampled.
- Define validation ladder:
  - finite differences,
  - AD parity on small models,
  - shape checks,
  - SPD and finite-value checks,
  - stress tests,
  - performance benchmarks.

MathDevMCP tasks:
- Search and index relevant labels in MacroFinance.
- Audit derivation labels where supported.
- Audit Kalman recursion code structure.
- Record inconclusive tool results honestly; do not upgrade them to proof.

Audit questions:
- Does each derivative formula identify the differentiated object?
- Are solve/Cholesky/QR variants kept distinct?
- Is the custom-gradient policy compatible with HMC detailed balance?

Pass criteria:
- No analytic derivative formula is marked final unless derivation/code parity
  has a recorded audit result.
- Validation chapter contains an explicit checklist for peer-review readiness.

## Phase W4: HMC-Safe Filtering Policy

Primary chapters:
- `docs/chapters/ch21_hmc_for_state_space.tex`
- `docs/chapters/ch22_mass_matrices.tex`
- `docs/chapters/ch23_boundary_gradients.tex`
- `docs/chapters/ch24_xla_jit.tex`
- `docs/chapters/ch25_diagnostics.tex`

Source inputs:
- Betancourt HMC source from Phase 2B.
- Hoffman-Gelman NUTS source from Phase 2B.
- TFP NUTS Gaussian benchmark under `docs/benchmarks/`.
- `~/python/docs/chapters/ch13_mass_matrix.tex`
- `~/python/docs/chapters/ch14_hmc.tex`
- `~/python/docs/chapters/ch15_advanced_hmc.tex`
- `~/python/docs/chapters/ch16_xla_ops.tex`
- CIP HMC and TensorFlow/TFP implementation chapters listed in the source map.

Drafting tasks:
- Write HMC requirements in filtering-target terms:
  - differentiability where HMC moves,
  - finite log target,
  - finite gradients,
  - static shapes for compiled kernels,
  - explicit failure returns.
- Document NUTS as an algorithm and diagnostic reference.
- Document the TFP NUTS implementation decision with the Gaussian benchmark.
- Explain mass matrix choices from analytic score/Hessian and MAP curvature.
- Define boundary and invalid-parameter behavior.
- Define diagnostics:
  - divergences,
  - E-BFMI,
  - acceptance rate,
  - tree depth if NUTS is used diagnostically,
  - condition numbers,
  - eigen/singular gap telemetry,
  - finite-gradient failure counts.

Audit questions:
- Does the chapter avoid saying NUTS solves convergence by itself?
- Does every backend mentioned have a derivative and failure policy?
- Are diagnostic thresholds presented as warning signs, not guarantees?

Pass criteria:
- The NUTS decision is clear enough that future agents do not reopen it as a
  default fix without new evidence.
- XLA/JIT discussion distinguishes full pipeline compilation from partial
  tracing.

## Phase W5: Nonlinear Filtering and SVD Sigma-Point Lessons

Primary chapters:
- `docs/chapters/ch15_ekf.tex`
- `docs/chapters/ch16_sigma_point_filters.tex`
- `docs/chapters/ch17_square_root_sigma_point.tex`
- `docs/chapters/ch18_svd_sigma_point.tex`
- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch20_filter_choice.tex`

Source inputs:
- `~/python/docs/chapters/ch09_sr_ukf.tex`
- `~/python/docs/chapters/ch09b_svd_filters.tex`
- `~/latex/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
- `~/latex/CIP_monograph/chapters/ch26_differentiable_pf.tex`
- `~/latex/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`
- `~/latex/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`
- Ionescu matrix backprop source from Phase 2B.
- Corenflos differentiable particle-filter source from Phase 2B.

Still-gated inputs:
- Julier-Uhlmann UKF primary support.
- PMCMC/pseudo-marginal primary support.

Drafting tasks:
- Present EKF/UKF/CKF/SVD filters as approximate likelihood backends.
- Define when a nonlinear filter is for likelihood evaluation, diagnostics, or
  HMC sampling.
- Document spectral derivative risk: repeated or close singular/eigen values
  can make raw autodiff fragile.
- Require eigen-gap/singular-gap telemetry for SVD-filter HMC experiments.
- Position particle filters carefully:
  - exact latent simulation and unbiased likelihood estimators are separate
    from differentiable approximate resampling.
  - pseudo-marginal HMC claims remain blocked until primary support is added.
- Write a filter-choice decision table by dimension, nonlinearity, derivative
  policy, and production risk.

Audit questions:
- Is the SVD filter described as robust evaluation infrastructure rather than a
  default industrial HMC gradient path?
- Are approximate likelihoods clearly labeled?
- Are particle-filter claims source-gated?

Pass criteria:
- No industrial-scale SVD-HMC safety claim appears.
- SVD/HMC chapters point back to analytic/custom-gradient policy.

## Phase W6: Transport, Surrogates, and Geometry

Primary chapters:
- `docs/chapters/ch26_transport_surrogates.tex`

Related chapters:
- `docs/chapters/ch22_mass_matrices.tex`
- `docs/chapters/ch25_diagnostics.tex`

Source inputs:
- NeuTra source from Phase 2B.
- Transport chapters in `~/python/docs`.
- CIP transport/surrogate chapters listed in `docs/source_map.yml`.

Drafting tasks:
- Explain transport and surrogate methods as geometry accelerators.
- State clearly that they are not correctness substitutes for a valid target
  and gradient.
- Define delayed-acceptance or correction requirements where an approximate
  surrogate changes the sampled target.
- Connect transport failures to diagnostics rather than promising generic
  convergence improvement.

Audit questions:
- Does the text avoid marketing NeuTra/surrogates as a universal fix?
- Are approximation and correction policies explicit?

Pass criteria:
- Transport chapter remains downstream of the exact target/gradient chapters.

## Phase W7: Industrial Case Studies

Primary chapters:
- `docs/chapters/ch27_lgssm_validation.tex`
- `docs/chapters/ch28_nonlinear_ssm_validation.tex`
- `docs/chapters/ch29_nk_svd_case_study.tex`
- `docs/chapters/ch30_cip_afns_case_study.tex`
- `docs/chapters/ch31_nawm_design_target.tex`
- `docs/chapters/ch32_production_checklist.tex`

Source inputs:
- LGSSM tests and benchmarks in MacroFinance.
- NK SVD reset memos and plans in `~/python/docs/plans`.
- CIP/AFNS validation material in `~/latex/CIP_monograph`.
- NAWM requirements as design target, not completed evidence.

Drafting tasks:
- Use each case study to test BayesFilter policies.
- Keep application economics compact.
- For NK SVD, describe the debugging lesson and the remaining gradient-risk
  policy.
- For CIP/AFNS, emphasize analytic LGSSM derivative validation.
- For NAWM, write a design target and readiness checklist, not a success claim.
- End with a production checklist covering:
  - source-backed formulas,
  - derivative policy,
  - JIT policy,
  - finite failure handling,
  - diagnostics,
  - benchmark reproducibility,
  - reset memo and provenance requirements.

Audit questions:
- Does each case study answer what BayesFilter learns from it?
- Are large-scale claims framed as requirements unless evidence exists?

Pass criteria:
- No downstream project is treated as part of the BayesFilter core.
- Production checklist is explicit enough to guide future implementation work.

## Phase W8: Literature and Citation Completion

Goal:
- Convert partially reviewed literature gates into auditable citations.

Tasks:
- Continue ResearchAssistant ingestion for:
  - Julier-Uhlmann UKF primary source.
  - Andrieu-Doucet-Holenstein PMCMC primary source.
  - pseudo-marginal MCMC efficiency/foundations.
  - square-root Kalman filtering sources.
  - large-scale DSGE/NAWM filtering sources if they are used for claims.
- Merge bibliography entries conservatively into `docs/references.bib`.
- Add citations only where the source has been checked.
- Record rejected metadata mismatches in result notes.

Pass criteria:
- Every literature-heavy paragraph has a citation or is marked as a local
  project decision/implementation note.
- `references.bib` contains no duplicate or stale keys from wholesale imports.

## Phase W9: Release-Quality Audit

Goal:
- Prepare the monograph for serious internal review.

Tasks:
- Full `latexmk` build.
- `git diff --check`.
- YAML parse of `docs/source_map.yml`.
- Search for unresolved placeholders:

```bash
rg -n "Placeholder|TODO|FIXME|blocked|unsupported|citation needed" docs
```

- Search for risky claims:

```bash
rg -n "converge|guarantee|exact|unbiased|industrial|safe" docs/chapters docs/appendices
```

- Audit labels for namespace discipline.
- Verify no generated PDF is staged unless explicitly requested.
- Update reset memo with:
  - chapters drafted,
  - sources used,
  - audits run,
  - blockers,
  - next hypotheses.

Pass criteria:
- Build clean.
- Known blockers are documented.
- No unsupported high-stakes claims remain hidden in prose.

## Recommended Immediate Next Action

Execute Phase W0, then Phase W1 only.

Reason:
- Phase W1 gives the monograph a stable language and prevents later chapters
  from diverging in notation and scope.
- W2 and W3 can then consolidate the MacroFinance analytic derivative spine
  without mixing it prematurely with nonlinear/SVD claims.
- W4 can use the now-settled NUTS benchmark to close the recurring
  implementation-decision loop.

## Stop Rules

Stop and ask for direction if:
- A source chapter conflicts materially with the BayesFilter notation contract.
- A section requires a literature source that is still blocked.
- A derivation audit contradicts the source implementation.
- The LaTeX build fails in a way that requires preamble restructuring.
- A chapter would need to make a production-readiness claim unsupported by
  tests, benchmarks, or derivation/code audits.

## Suggested Commit Cadence

Use small phase commits:

1. Commit W0 hygiene and any benchmark/provenance files not yet committed.
2. Commit W1 notation/contracts.
3. Commit W2 linear Gaussian likelihood.
4. Commit W3 analytic derivatives and validation policy.
5. Commit W4 HMC/JIT/diagnostics policy.
6. Commit W5 nonlinear/SVD filter chapters.
7. Commit W6 transport/surrogates.
8. Commit W7 case studies and production checklist.
9. Commit W8/W9 bibliography and release audit.

Each commit should include the relevant reset-memo update and avoid generated
PDFs unless explicitly requested.

## Independent Audit Before Execution

Date:
- 2026-05-03

Audit stance:
- Treat this as a second developer taking ownership of the writing plan.
- The plan is directionally sound, but executing all phases in one autonomous
  pass is safe only if each phase is interpreted as a bounded first writing
  pass with explicit gates, not as a final peer-review-quality derivation pass.

Required modifications:

1. Phase W1 should be fully drafted first.
   - Later chapters must use its notation and target-contract language.

2. Phase W2 may include exact linear-Gaussian likelihood formulas.
   - These formulas are supported by the MacroFinance analytic derivative note
     and by the older Kalman chapters.
   - It must not introduce score/Hessian formulas beyond forward references.

3. Phase W3 must be conservative.
   - It may state derivative contracts, crosswalks, and validation ladders.
   - It may quote or restate only bounded formulas whose source labels and code
     parity are recorded.
   - Any MathDevMCP result of `inconclusive` must remain inconclusive in the
     text.

4. Phases W4--W7 should draft implementation policy, diagnostics, and case-study
   structure rather than pretending all production evidence already exists.
   - The TFP NUTS benchmark settles only the implementation-default decision; it
     does not prove anything about HMC convergence.
   - SVD sigma-point material must remain framed as a risk-aware approximate
     filtering path until gradient policy and industrial-scale tests exist.

5. Phase W8 is allowed to remain a blocker register.
   - PMCMC, pseudo-marginal, Julier-Uhlmann UKF, and some square-root filtering
     claims still need primary support before final citation prose is written.

6. Phase W9 should be a mechanical and claim-risk audit.
   - Search for placeholders and risky words.
   - Do not remove honest blocker language just to make the text look complete.

Execution decision:
- Proceed through W0--W9 as a first consolidation pass.
- Stop only if a derivation/code audit contradicts the source implementation,
  if LaTeX fails in a structural way, or if a chapter would require unsupported
  production-readiness claims.
