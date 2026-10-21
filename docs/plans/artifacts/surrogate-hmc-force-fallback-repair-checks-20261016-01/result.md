# Surrogate-HMC force fallback repair: engineering result

The controller now records deterministic proposal-force fallbacks as warnings
for the bound endpoint-corrected position-field runner. The likelihood, score,
priors, proposal kernel and endpoint calculation are unchanged. MH still uses
the independent endpoint density. Other runner semantics retain their previous
fallback veto. Numerical endpoint failures and the remaining sampler health
checks still stop a run.

The saved range-bearing failure had only `no_force_fallback=False`, with zero
retained draws. It is preserved rather than reclassified as a completed run.
The owner requested the repaired retry on 2026-10-16. Its place is after the
running predator–prey and queued SIR, using at most 22712.3556 worker seconds
remaining in the original eight-hour range allocation and the shared total
216000-second/six-attempt limits.

## Checks and interpretation

The final focused suite passed 62 tests in 31.15 seconds. It ran on CPU with
GPU devices intentionally hidden, including the actual TensorFlow/XLA Gaussian
proposal and MH kernels. Two dependency deprecation warnings were emitted.
The initial run had 60 passes and two stale test failures: those tests still
required resume-only access to the already authorized 20000-sample maximum and
limited a subsequent retained segment to 10000. Their expectations now match
the existing cumulative policy; no runtime sample cap changed in this repair.
Both logs are preserved as `tests.log` and `tests-final.log`.

Invalid and nonfinite deterministic forces produced exactly the same tested
states, acceptance decisions and energies as an explicit zero field, while
recording fallbacks. Reversing the piecewise zero-extended proposal recovered
the initial phase-space point within 1e-13 in the fixture. Fresh chunks, replay,
resume, aggregate checks and retained continuation use the resolved policy and
preserve warning counts. The aggregate loader verifies archived diagnostics.
Endpoint failure, nonfinite arithmetic, divergence, excessive energy error and
immobile chains still failed health checks. Queue tests covered waiting for
the parent supervisor, settled worker charges, unique output, source changes,
remaining time and attempt limits. No GPU posterior claim follows from these
controller tests.

All three frozen preparations passed the repository preflight after the repair.
Their filter source closures exclude the changed controller modules; the new
worker manifest includes them. The already running PP process has its original
loaded controller. Its startup worker manifest preserves that provenance;
older terminal manifests hash disk files at finalization and must not be used
to assert that this active worker loaded the new controller.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Execute the authorized queued retry | Kernel/controller/queue regression suite passed | Endpoint and arithmetic guards retained; source and budget guards tested | Range fallback frequency and resulting mixing | Queue the existing range-bearing master after PP/SIR | Posterior convergence, exact nonlinear likelihood or improved efficiency |

| Inference status | Finding |
|---|---|
| Hard veto screen | Fallback-only stop repaired; all other tested failures still veto |
| Statistically supported ranking | None; this is an engineering repair, not a method comparison |
| Descriptive-only differences | CPU test timings; no new range-bearing sampling result yet |
| Default-readiness | Warning policy is restricted to the bound deterministic endpoint-corrected force runner |
| Next evidence needed | Repaired range sampling diagnostics under the existing endpoint, controls and compute cap |

The strongest alternative explanation for another failure is poor exploration
under a frequently zero force, or a separate endpoint/numerical failure. The
repair permits testing that possibility; it does not promise convergence.
A nondeterministic/history-dependent field or a faulty MH endpoint would
invalidate the mathematical justification. The current checks do not establish
global floating-point reversibility across all range-bearing trajectories.

Plan: `../../surrogate-hmc-force-fallback-repair-2026-10-16.md`.
Run commands, source identities and CPU status are recorded in `manifest.json`.
Live execution receipt:
`../surrogate-hmc-60h-continuation-20261016-01/queued-range-bearing.json`.

## Queue launch receipt

The exact planned enqueue command ran with escalated permissions at
2026-10-16 08:34:59 UTC and returned `QUEUED`, PID1485735, with a maximum
22712.355625355995 worker seconds. At08:35UTC the detached process was alive
and `WAITING_FOR_CURRENT_QUEUE`. The original supervisor1119837 and PP
master1194791/worker1194793 remained alive. The budget ledger still contained
only the completed first range attempt and the active PP reservation; waiting
did not reserve or charge another worker. The range retry has not sampled yet.
