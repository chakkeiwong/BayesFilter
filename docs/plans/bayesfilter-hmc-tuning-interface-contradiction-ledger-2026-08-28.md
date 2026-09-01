# BayesFilter HMC Tuning Interface Contradiction Ledger

Date: 2026-08-28

Implementation baseline: `1a284ec2d09b7776b7e44fecd211e9f8e7a3ade3`

Status: `ACTIVE_IMPLEMENTATION_LEDGER`

This ledger records the implementation-start audit for
`bayesfilter-hmc-tuning-interface-documentation-and-verification-plan-2026-08-27.md`.
It is an interface audit, not sampler evidence.

| Statement or observed use | Classification at implementation start | Evidence and required disposition |
| --- | --- | --- |
| `tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel` are the only active artifact-authority tuning routes. | `correct` | The route registry and inventory check agree. Preserve exactly these two active routes unless the neural-force design gate fails. |
| The monograph explains how to choose and call those public interfaces. | `wrong relative to the documentation target` | Chapter 21 describes mechanics but names neither public API nor its configuration. Add the planned chapter, reference guide, and executable examples. |
| Calling any exported function whose name contains `tuning` performs canonical tuning. | `wrong` | Historical, diagnostic, stage-helper, and runner symbols have no artifact authority. Generate role and capability tables from the checked registry. |
| The ordinary tuner owns mass adaptation, step-size tuning, leapfrog-count selection, screening, fresh verification, and repair. | `implemented but incompletely enforced` | Internal stages share a runner hook, but the public route cannot bind it and the final classifier ignores a failed sequential R-hat verdict. Add typed binding and repair admission before documenting the full guarantee. |
| A sequential verifier R-hat failure or cap hit can coexist with a passed public handoff. | `wrong relative to the declared fresh-verification target` | The verifier returns `passed=False`, while `_classify_phase7_acceptance_evidence_verification` only consumes acceptance evidence. A failed or missing R-hat gate must prevent handoff. |
| Ordinary canonical tuning currently has a bulk/tail ESS admission gate. | `unsupported` | The ordinary tuner does not consume an ESS gate. Document ESS as disabled for tuning admission; do not select a new threshold in this work. Retained posterior ESS is a separate assessment. |
| `tune_fixed_transport_hmc_kernel` is a generic arbitrary-force tuner. | `wrong` | It requires a genuine frozen transport, transformed value/score adapter, transport manifest identity, and fixed identity mass in `z`. Keep arbitrary position-force mechanics out of this route. |
| `run_full_chain_neural_force_hmc` tunes mass and `L` when called directly. | `wrong` | The low-level runner receives fixed `L`, uses the supplied mass coordinates, and may dual-average epsilon only. Its identity fallback is mechanics-only. |
| An arbitrary frozen position-only force has a canonical public tuning route. | `not implemented at the baseline` | Test the preferred typed binding through `tune_hmc_kernel`. Until the end-to-end contract is green, downstream direct-runner output remains diagnostic. |
| MacroFinance consumers predominantly call the ordinary public tuner. | `correct for the inspected current symbol inventory; not a full consumer certification` | Many current source routes import `tune_hmc_kernel`. Downstream pin and contract checks remain owned by MacroFinance. |
| The cited dsge_hmc BGS four-transition direct neural-force run was a serious tuning rung. | `wrong` | It was fixed-mass, fixed-`L=1`, short epsilon adaptation and bypassed candidate screening and fresh verification. Migrate only after BayesFilter publishes and pins a supported binding. |

## Numeric Policy

The ordinary sequential fresh-verification threshold `1.01` is inherited from
the implemented verifier configuration and already labeled as a fixed-kernel
verification gate. This work restores its consumption; it does not calibrate or
promote a new threshold. ESS remains disabled because no reviewed ordinary-tuner
threshold is selected by this plan.

## Nonclaims

Resolving these contradictions does not establish target correctness,
posterior convergence, sampler quality, performance, GPU/XLA readiness, or a
downstream repository migration. The focused tests establish only the checked
interface and handoff behavior.
