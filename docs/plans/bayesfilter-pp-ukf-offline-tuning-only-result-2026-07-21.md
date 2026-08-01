# PP-UKF Offline Tuning-Only Result

Date: 2026-07-21
Plan: docs/plans/bayesfilter-pp-ukf-offline-tuning-only-plan-2026-07-21.md
Status: ROUTE_REPAIRED_PP_UKF_RUNTIME_DIAGNOSTIC_PENDING

## Command And Artifacts

The trusted GPU preflight passed:

- NVIDIA GeForce RTX 4080 SUPER, CUDA 13.1.
- TensorFlow GPU device visible.
- Memory-growth policy enabled before logical-device initialization.
- Full-device preallocation disabled.
- XLA service initialized and compiled a cluster.

The executed tuning-only command used commit 0fff464ab456b72a010007e552c1e2d761624afe, the PP-UKF target signature
d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5, and
the frozen transport SHA-256
b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221.

Fresh artifacts are under:

docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-01/PP-UKF/

The terminal authorities are result.json, run_manifest.json, and the
public tuning artifact. No sampling artifacts were created.

## Result

| Decision | Status |
| --- | --- |
| Target and frozen transport validation | passed |
| GPU/memory-growth/XLA preflight | passed |
| Bootstrap screens | finite diagnostics produced for all six rounds |
| Public fixed-identity tuning handoff | blocked before Phase 7 runtime |
| Tuning-only result | TUNING_FAILED |
| Sequential HMC launched | no |
| Scientific or posterior claim | none |

The hard veto is:

    phase7_runtime_error
    UnsupportedHMCAlgorithmRoute:
    HMC algorithm route blocked: operational_windowed_warmup_xla_not_validated
    (windowed_mass: operational_interleaved_windowed_warmup_v2)

The public tuner selected the operational top-level route
operational_paired_fixed_trajectory_selection_v3, which maps to the
operational windowed warmup route. The route contract intentionally rejects
that warmup route when use_xla=True; the route is not validated for XLA.

## Interpretation

This result does not invalidate the PP-UKF target, frozen transport, GPU
environment, or fixed-identity mass implementation. The bootstrap rounds
produced finite target, acceptance, and runtime diagnostics before the route
veto. The result is a route-contract infrastructure blocker, not evidence
against the PP-UKF scientific direction or a failed tuning candidate.

The plan did not authorize either of the tempting workarounds:

- use_xla=False would be a non-default debug/reference exception and would
  not answer the requested serious GPU/XLA tuning question.
- A legacy segmented/joint-grid route would change the selected algorithm and
  is non-promoting under the route contract.

## Post-Run Red Team

Strongest alternative explanation: the route contract is stale relative to the
newly merged operational warmup implementation, and a reviewed XLA validation
artifact may be missing rather than the implementation being numerically
invalid.

Result that would overturn this classification: a focused, reviewed route
validation artifact and tests proving the operational windowed warmup route is
safe under TensorFlow/XLA with the required target-status, metric-lineage,
epsilon, and fixed-identity invariants.

Weakest evidence: no Phase 7 candidate or final handoff was constructed, so
this run cannot say whether PP-UKF tuning would pass after the route blocker is
resolved.

## Next Justified Action

Do not launch PP-UKF claim/HMC sampling. The operational windowed warmup route
blocker was repaired and validated in the separate focused artifact
`docs/plans/artifacts/bayesfilter-operational-windowed-warmup-xla-route-validation-20260721/`.
That validation covered TensorFlow/XLA mechanics and route-contract identity
only; it did not establish PP-UKF tuning or HMC validity.

A fresh same-contract PP-UKF tuning-only root was then launched at
`docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-02/PP-UKF/`.
Bootstrap completed with finite diagnostics, but the attempt was interrupted
while the first operational window searched for a reasonable epsilon. No
`result.json`, `run_manifest.json`, tuning handoff, or sampling artifact was
written. The durable classification is recorded in `interruption_record.json`
as `INTERRUPTED_INFRASTRUCTURE_ATTEMPT`; this is not a PP-UKF candidate
rejection and does not invalidate the target or frozen transport.

The next action is a bounded one-proposal diagnostic separating target
evaluation, HMC bootstrap, and proposal runtime. No further tuning retry is
justified until that diagnostic identifies a localized repair or confirms
that the observed cost is an expected bounded compilation/runtime cost.

## Runtime Diagnostic And Retry Update

The bounded trusted-GPU diagnostic confirmed the runtime cause and the repair:

| Diagnostic | Result | Role |
| --- | --- | --- |
| Eager target evaluation | 19.94 s, finite | explanatory only |
| Eager one-proposal HMC | 41.94 s, finite | explanatory only |
| XLA bootstrap plus one proposal | 3.89 s + 5.35 s, finite | repair validation |

The repair propagates the operational route's `jit_compile=True` into both
reasonable-epsilon bootstrap and proposal execution. Existing eager callers
retain the default `jit_compile=False`; focused regressions passed (`89
passed, 2 skipped`). The diagnostic artifacts are under
`docs/plans/artifacts/bayesfilter-pp-ukf-epsilon-probe-20260721-01/`.

A fresh same-contract retry was launched at
`docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-03/`.
It completed bootstrap and entered `windowed_mass_operational_warmup_start`,
but the runner process disappeared before emitting a segment boundary or
terminal result. No `result.json`, `run_manifest.json`, tuning handoff, or
sampling artifact exists. `interruption_record.json` records this as an
`INTERRUPTED_INFRASTRUCTURE_ATTEMPT`, not a PP-UKF candidate rejection.

The remaining blocker is operational runner liveness/telemetry during the
first compiled window, not the former route-contract blocker and not evidence
against PP-UKF. Do not launch a claim-bearing HMC run or another tuning retry
until that runner termination is localized.
