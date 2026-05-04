# Experiment plan: six next-issue closure pass

## Date

2026-05-05

## Question

Can BayesFilter turn the six post-closure hypotheses into executable gates,
tests, and conservative documentation without falsely claiming client-repo
DSGE/MacroFinance production promotion or HMC convergence?

The six issues are:

1. DSGE adapter evidence for SmallNK-like all-stochastic metadata while mixed
   Rotemberg/SGU-like metadata fails closed without deterministic completion.
2. Particle-filter Monte Carlo convergence evidence on longer AR(p)-style
   panels.
3. MacroFinance sparse large-scale readiness using provider-owned masked
   derivative metadata rather than production caller overrides.
4. Cross-currency readiness requiring blockwise oracle evidence for dynamics,
   transition covariance, observation loadings, and measurement error.
5. SVD/eigen derivative promotion blocking near small spectral gaps unless a
   non-spectral derivative policy is explicitly declared and tested.
6. HMC sampler readiness requiring real diagnostics and backend sensitivity
   comparison across candidate backends.

## Mechanism Being Tested

This pass tests BayesFilter control-layer machinery rather than client
economics:

- DSGE metadata is structural adapter evidence, not DSGE model logic.
- Particle likelihood evidence is Monte Carlo value-only evidence, not
  differentiable PF or PMCMC certification.
- MacroFinance readiness consumes provider-owned metadata and oracle summaries,
  not BayesFilter-owned financial derivative recursions.
- Spectral derivative certification is a gate over telemetry and declared
  derivative policy.
- HMC readiness remains a diagnostic gate over supplied chain diagnostics.

## Scope

- Variant: BayesFilter-owned gate hardening and tests.
- Objective: close executable BayesFilter gates for the six issues while
  preserving client-promotion blockers.
- Seed(s): deterministic NumPy seeds for particle tests.
- Training steps: none.
- HMC/MCMC settings: diagnostic-gate inputs only; no sampler is run.
- XLA/JIT mode: compile parity remains metadata supplied by client gates.
- Expected runtime: under one minute for focused tests; a few minutes for full
  suite and LaTeX no-op build.

## Success Criteria

- A plan, independent audit, reset-memo phase log, and source-map registration
  exist.
- DSGE adapter tests distinguish all-stochastic metadata from mixed metadata
  lacking completion maps.
- Particle tests demonstrate lower AR(2) likelihood error at higher particle
  count under fixed seeds and preserve deterministic identities.
- Large-scale MacroFinance gates report whether masked support is provider-owned
  or caller-supplied, and production mode blocks caller overrides.
- Cross-currency gates can require and validate named oracle blocks.
- Spectral certification exposes derivative policy and blocks small-gap
  spectral paths while allowing explicitly non-spectral policies with numerical
  checks.
- HMC diagnostic comparison fails closed when any backend fails and reports
  backend sensitivity when all supplied diagnostics pass.
- No new production-ready or convergence claim is introduced.

## Diagnostics

Primary:
- `pytest -q tests/test_dsge_adapter_gate.py`
- `pytest -q tests/test_structural_particles.py`
- `pytest -q tests/test_macrofinance_adapter.py`
- `pytest -q tests/test_backend_readiness.py`

Secondary:
- `pytest -q`
- source-map YAML parse
- `git diff --check`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

Sanity checks:
- stale-claim search for `converged`, `production-ready`, `sampler-usable`,
  `target_candidate`, `not_claimed`, and `blocked`;
- `git status --short --branch` before staging;
- stage reset-memo hunks explicitly because the file has pre-existing unrelated
  edits.

## Expected Failure Modes

- A mixed DSGE-like object passes without a completion map.
- A particle convergence test is too noisy or too slow.
- Caller-supplied masked derivative support is accidentally accepted in
  production mode.
- Cross-currency oracle output reports an aggregate discrepancy but omits one
  required block.
- Small spectral gaps are treated as HMC eligible under the default spectral
  derivative policy.
- HMC backend comparison hides a failing backend behind an average diagnostic.

## What Would Change Our Mind

- If the particle convergence test is unstable under deterministic seeds, keep
  only non-regression bounds and record convergence as an experiment result
  rather than a unit-test gate.
- If provider-owned metadata cannot be represented without breaking existing
  MacroFinance tests, keep the old gate and record a client-repo blocker.
- If backend comparison requires actual sampler objects, stop at a diagnostic
  evidence contract and do not add sampler code.

## Command

```bash
pytest -q tests/test_dsge_adapter_gate.py tests/test_structural_particles.py \
  tests/test_macrofinance_adapter.py tests/test_backend_readiness.py
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Interpretation Rule

- If all focused gates pass and full validation passes, commit the BayesFilter
  gate-hardening pass.
- If a BayesFilter-owned gate fails because the code is incomplete, fix it and
  rerun the focused phase.
- If a phase requires client-repo economics, real production data, or real HMC
  chain output, keep the phase blocked and record the precise next client-owned
  evidence needed.

## Phase S0: hygiene, plan, and audit

Motivation:
- This pass touches reset memo, source map, gates, and tests.  The worktree
  already has unrelated edits, so hygiene is a prerequisite for commit
  correctness.

Implementation instructions:
- Record dirty files.
- Do not stage `docs/plans/templates` or unrelated reset-memo/roadmap edits
  unless explicitly changed by this pass.
- Write the independent audit before code changes.

Exit gate:
- Plan and audit exist, and the next phase has no ambiguous write set.

## Phase S1: DSGE adapter evidence gate

Motivation:
- The previous pass added a fail-closed gate.  The next issue is proving the
  intended claim boundary: all-stochastic SmallNK-like metadata can pass, while
  mixed Rotemberg/SGU-like metadata cannot pass without completion maps.

Implementation instructions:
- Add tests for an all-stochastic SmallNK-like object.
- Add tests for mixed Rotemberg/SGU-like metadata missing completion.
- If needed, add a gate field that records `metadata_regime`.

Exit gate:
- All-stochastic metadata passes.
- Mixed metadata without completion fails closed.

## Phase S2: particle Monte Carlo convergence evidence

Motivation:
- A bootstrap particle filter can be value-useful only if its AR(2) likelihood
  behavior improves with particle count under a fixed experiment design.

Implementation instructions:
- Add a deterministic two-level particle count test on a longer AR(2) panel.
- Compare absolute error against the exact Kalman value.
- Keep the threshold loose enough to test convergence direction without making
  the suite flaky.

Exit gate:
- Higher particle count has lower error than lower particle count, and identity
  diagnostics remain near zero.

## Phase S3: provider-owned MacroFinance masked metadata

Motivation:
- Sparse production panels must not depend on caller overrides.  Production
  readiness should require provider-owned support metadata.

Implementation instructions:
- Add a `masked_support_source` result field.
- Add `production_mode` to the large-scale adaptation gate.
- In production mode, block sparse readiness when support was caller-supplied.

Exit gate:
- Provider-owned support passes.
- Caller override can pass non-production smoke but fails production mode.

## Phase S4: cross-currency blockwise oracle evidence

Motivation:
- A single aggregate oracle discrepancy can hide a missing derivative block.
  Cross-currency readiness needs named block coverage.

Implementation instructions:
- Add required oracle block support to the cross-currency gate.
- Record checked and missing oracle blocks in the result.
- Require dynamics, transition covariance, observation loadings, and
  measurement error in tests.

Exit gate:
- Complete blockwise oracle evidence passes.
- Missing blockwise evidence blocks readiness even when aggregate discrepancy is
  small.

## Phase S5: spectral derivative policy gate

Motivation:
- SVD/eigen derivatives are unsafe near repeated values unless the derivative
  path avoids spectral differentiation or downgrades the claim.

Implementation instructions:
- Add a `derivative_policy` argument to spectral certification.
- Keep `spectral` as the default and block small gaps.
- Allow `non_spectral_custom_gradient` only when finite-difference and JVP/VJP
  checks are declared.

Exit gate:
- Small-gap spectral path is blocked.
- Small-gap non-spectral checked path can be HMC eligible with an explicit
  warning label.

## Phase S6: HMC backend diagnostic comparison

Motivation:
- A single target/diagnostic gate is not enough for backend promotion.  Real
  HMC readiness needs backend sensitivity comparison.

Implementation instructions:
- Add a comparison helper that evaluates HMC diagnostic gates for named
  backends.
- Fail closed when any backend fails.
- Record min ESS, max split R-hat, acceptance range, and blockers by backend.

Exit gate:
- Passing backends produce comparison-ready metadata.
- A single failing backend blocks comparison readiness and keeps convergence
  `not_claimed`.

## Final Phase: release validation and commit

Motivation:
- The pass is only useful if docs, reset memo, tests, and source map agree.

Implementation instructions:
- Run focused tests, full tests, YAML parse, `git diff --check`, and LaTeX.
- Update the reset memo with phase results and next hypotheses.
- Stage only current-pass files/hunks.
- Commit the scoped pass.

Exit gate:
- Commit exists and any unrelated dirty files remain unstaged.
