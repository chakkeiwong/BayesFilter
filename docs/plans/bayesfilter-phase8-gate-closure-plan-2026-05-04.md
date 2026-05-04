# Plan: Phase 8 executable gate closure

## Scope

This pass closes the hypotheses recorded after the Phase 8 follow-on metadata
closure.  The previous pass made metadata first-class.  This pass makes the
remaining go/no-go criteria executable so BayesFilter can decide whether a
MacroFinance-shaped provider is ready for the next adapter layer.

Remaining gaps:

1. Large-scale LGSSM likelihood adaptation needs a dense/all-observed gate or
   explicit masked-derivative support through order 2.
2. Cross-currency structural adaptation needs full derivative-coverage and a
   bounded finite-difference oracle check.
3. Production provider exposure needs a fail-closed exposure gate that requires
   final readiness, no blockers, and final identification claims.
4. HMC sampler integration needs chain-diagnostic gates after target readiness,
   not merely finite target operations.

## Non-goals

- Do not copy MacroFinance likelihood, derivative, production, TensorFlow, or
  sampler implementations into BayesFilter.
- Do not mark fixture-only or weakly identified providers as production-ready.
- Do not run a BayesFilter HMC sampler.
- Do not claim HMC convergence from target-readiness checks.
- Do not stage the pre-existing dirty `ch18b` chapter edit.

## Required Cycle

Each phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

## Phase G0: setup and latest-state check

Actions:

1. Record repo status and prior commits.
2. Record this plan and its independent audit.
3. Confirm known unrelated dirty state.

Tests:

- `git status --short --branch`
- `git log -1 --oneline`

Gate:

- Continue if the only unrelated dirty file remains the known chapter edit.

## Phase G1: large-scale adaptation gate

Actions:

1. Add a result object and helper that combines observation-mask metadata with
   explicit masked-derivative support metadata.
2. Mark likelihood adaptation ready only when either all observations are dense
   or masked derivatives are supported through the requested derivative order.
3. Keep this as a gate only; do not run large-scale likelihood in BayesFilter.

Tests:

- Pure fake-provider tests for dense ready, sparse blocked, and sparse ready
  when masked derivative support is declared.
- Optional MacroFinance integration against dense and sparse large-scale
  scenario metadata.

Gate:

- Continue if sparse panels fail closed unless order-2 masked derivative support
  is explicitly present.

## Phase G2: cross-currency coverage/oracle gate

Actions:

1. Add a result object and helper that verifies derivative coverage rows cover
   every provider parameter name.
2. Optionally run a bounded finite-difference oracle check supplied by the
   caller and record maximum absolute discrepancies.
3. Do not promote coverage metadata into production readiness.

Tests:

- Pure fake-provider tests for complete/incomplete coverage and passing/failing
  oracle checks.
- Optional MacroFinance integration using a bounded blockwise oracle check on
  `CrossCurrencyStructuralDerivativeProvider`.

Gate:

- Continue if incomplete coverage or failing oracle discrepancies set
  `adaptation_ready=False`.

## Phase G3: production exposure gate

Actions:

1. Add a result object and helper that combines blocker metadata,
   identification evidence, and sparse-backend policy into a production exposure
   decision.
2. Require `final_ready=True`, no blockers, nonempty identification evidence,
   and final identification trust status.
3. Preserve blocker and identification rows when exposure is false.

Tests:

- Pure fake-provider tests for blocked and final-ready providers.
- Optional MacroFinance production scaffold integration proving it remains
  blocked.

Gate:

- Continue if scaffold/fixture providers stay blocked and final-ready fakes can
  pass only with final identification evidence.

## Phase G4: HMC diagnostics gate

Actions:

1. Add diagnostic metadata and a helper that evaluates HMC chain diagnostics
   after target readiness.
2. Require target readiness plus finite ESS, finite split R-hat, no divergences,
   acceptable R-hat, minimum ESS, and bounded acceptance rate.
3. Preserve `convergence_claim="not_claimed"` unless diagnostic thresholds pass.

Tests:

- Pure tests for passing diagnostics, target-not-ready, bad R-hat, low ESS,
  divergence, and bad acceptance.
- Optional MacroFinance-style test can use synthetic diagnostic objects; no
  sampler run is required.

Gate:

- Final validation is justified if diagnostic gates do not claim convergence
  from target readiness alone.

## Final Validation

- `python -m py_compile bayesfilter/adapters/macrofinance.py
  tests/test_macrofinance_adapter.py`
- `pytest -q tests/test_macrofinance_adapter.py`
- `pytest -q tests`
- YAML parse for `docs/source_map.yml`
- `git diff --check`

## Hypotheses

- H-G1: A dense-or-masked-order gate can prevent large-scale likelihood
  adaptation from silently accepting sparse panels without derivative support.
- H-G2: Cross-currency structural adaptation can be made auditable by requiring
  parameter coverage and bounded oracle discrepancies before readiness.
- H-G3: Production provider exposure can fail closed using provider metadata
  alone, without importing final-provider implementation details.
- H-G4: HMC sampler readiness must be a separate chain-diagnostic gate layered
  on top of target readiness, not a property of finite target operations.
