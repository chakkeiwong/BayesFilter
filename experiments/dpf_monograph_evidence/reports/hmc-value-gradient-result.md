# IE7 HMC value-gradient result

## Outcome

- Master-program exit label: `ie_phase_passed`
- Local exit label: `ie7_hmc_value_gradient_passed`
- Phase status: `pass`

## Skeptical audit before execution

- Same-scalar audit: accept/reject and differentiated scalar values came from the same target wrapper.
- Gradient audit: stable-window central finite differences match the closed-form gradient within tolerance.
- Runtime-boundary audit: no HMC chain, tuning, adaptation, posterior summary, DPF-HMC target, DSGE target, or MacroFinance target was run.
- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.

## Research-intent ledger

- Main question: does the fixed scalar target preserve the same-scalar value-gradient contract before any HMC-facing interpretation?
- Promotion criterion: same scalar, finite-difference stable window, eager repeatability, compiled/eager status reporting, leapfrog reversibility, and bounded energy smoke pass.
- Promotion veto: missing target identity, same-callable proof, finite-difference ladder, seed policy, no-chain assertion, CPU manifest, source support, tolerance object, or exact non-implication text.
- Continuation veto: same-scalar failure, nondeterministic target behavior, or prohibited HMC execution.
- Repair trigger: same-scalar mismatch, gradient mismatch, repeatability failure, reversibility failure, energy-drift failure, nondeterminism, or prohibited HMC execution.

## Evidence contract

| Diagnostic | Comparator | Status | Source support |
| --- | --- | --- | --- |
| `hmc_value_gradient` | `fixed_scalar_same_callable_finite_difference_reference` | `pass` | `bibliography_spine_only` |

## Residual summary

- same-scalar residual: `0.000e+00`
- finite-difference stable-window residual: `1.083e-07`
- eager value repeat residual: `0.000e+00`
- eager gradient repeat residual: `0.000e+00`
- compiled status: `not_available` (`No compiled/autodiff backend is imported in the clean-room CPU-only IE7 fixture.`)
- leapfrog position reversibility residual: `0.000e+00`
- leapfrog momentum reversibility residual: `5.551e-17`
- forward energy-drift smoke residual: `3.116e-05`

## Decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `hmc_value_gradient` | `pass` | `pass` | `not_triggered` | `not_triggered` | `none` |

## Coverage semantics

- IE3 through IE5 diagnostics are carried forward as `passed`.
- IE6 learned-map residual remains `deferred`.
- IE7 `hmc_value_gradient` is `passed`.
- IE8 remains `missing`.

## Artifact list

- `experiments/dpf_monograph_evidence/reports/outputs/hmc_value_gradient.json`
- `experiments/dpf_monograph_evidence/reports/hmc-value-gradient-result.md`
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-result-2026-05-16.md`

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_hmc_value_gradient`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- python: `3.13.13` / numpy `2.1.3`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- seed policy: `deterministic_no_rng_fixed_scalar_target`

## Required non-implication text

IE7 fixed-scalar value-gradient diagnostics validate only same-scalar, finite-difference, repeatability, and fixed-target leapfrog/energy-smoke checks on a deterministic clean-room fixture. They do not validate HMC correctness, DPF-HMC correctness, posterior/reference agreement, tuning readiness beyond controlled-fixture eligibility, production bayesfilter code, banking use, model-risk use, or production readiness.

## Post-run red-team note

A same-scalar fixture pass still does not validate any real DPF-HMC posterior, HMC tuning, production code path, or banking/model-risk use.

## Next-phase justification

IE7 passed the fixed-scalar gate, so IE8 may summarize controlled-fixture evidence. IE8 still must not run or claim real DPF-HMC, DSGE, MacroFinance, posterior, banking, model-risk, or production validation.
