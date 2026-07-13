# Phase 4 Subplan: XLA Value/Score Gate

Date: 2026-07-09

## Phase Objective

Verify that the target value/score path needed by HMC can be compiled and
executed with XLA/JIT enabled, without runtime `GradientTape`.

## Entry Conditions Inherited From Previous Phase

- Phase 3 fixture artifact exists and passes deterministic generation checks.
- Target adapter binds the fixture and prior config.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-result-2026-07-09.md`
- Compile timing/metadata JSON under `docs/benchmarks/artifacts/`.

## Required Checks / Tests / Reviews

- Target-path value/score function compiles with `jit_compile=True`.
- Metadata records compile time and HLO/module size when available.
- Static/dynamic code scan confirms no runtime `GradientTape` in target path.
- Focused pytest for adapter shape/value/score contract.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the LGSSM target path admissible for XLA-only HMC tuning? |
| Baseline/comparator | Phase 3 target fixture and existing TensorFlow/TFP implementation. |
| Primary pass criterion | XLA/JIT compile and finite value/score at configured probe points. |
| Veto diagnostics | JIT disabled/fallback, runtime GradientTape, nonfinite value/score, shape mismatch. |
| Explanatory diagnostics | Compile wall time, module size, probe values. |
| Not concluded | No HMC readiness or posterior recovery claim. |

## Forbidden Claims / Actions

- Do not run non-JIT fallback.
- Do not treat compile success as convergence or tuning success.
- Do not use `GradientTape` except diagnostic/test-only paths clearly outside target path.

## Exact Next-Phase Handoff Conditions

- Phase 5 can call geometry/mass tools against the compiled value/score adapter.

## Stop Conditions

- XLA compile fails and there is no reviewed target-path repair.
- Any required target value/score is nonfinite at prior mean or configured probes.
