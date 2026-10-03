# q=20 Factor Backend Source-Receipt Reconciliation

Date: 2026-09-06  
Role: current-dated provenance reconciliation, not a new numerical run  
Source receipt: `docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/`  
P0 receipt: `docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json`

## Decision

The source-synchronized receipt and independent audit were independently
verified by P0 on 2026-09-05. Their recorded SHA-256 values match, all six
chart/beta checkpoint and tuning pairs are present, and the artifacts bind the
same q=20 target signature and `tensorflow_eigh_strict_factor_cached` backend.

This receipt reconciliation supports the narrow factor-route admission for the
q=20 Phase 9B candidate-backend lane using the source-synchronized receipt and
the independently audited September 4--5 evidence. The September 6 result note
and reset memo are current-session provenance. They do not promote the
repository default, the sampler, the transport, or any posterior claim.

The strict backend remains the required comparator and fallback. A strict
comparator must receive its own fresh scope-bound tuning artifacts; no factor
handoff may be transferred to it.

## Evidence boundary

| Decision | Status | Interpretation |
|---|---|---|
| Source receipt integrity | Passed | Manifest and audit hashes are durable and match the supplied values. |
| Six-scope factor handoffs | Passed for narrow backend identity | The P0 receipt verifies presence, parseability, target identity, backend identity, and measured-grid policy. |
| Phase 9B candidate-backend entry | Open for planning | A fresh P1 canary plan may be written; this is not a posterior run. |
| Chart quality | Not admitted | Existing chart-quality concerns remain and require independent threshold calibration before P2. |
| Posterior/HMC readiness | Not assessed | No sequential HMC draws or convergence evidence are supplied by this reconciliation. |
| Repository default | Unchanged | Strict remains the generic default/fallback. |

## Claim boundary

The reconciliation establishes only:

`PROMOTED_Q20_PHASE9_NUMERICAL_BACKEND`

for the proposed q=20 Phase 9B candidate lane. It does not establish IID
Gaussian whitening, a well-trained NeuTra map, posterior correctness,
convergence, mode discovery, sampler superiority, scaling, production
readiness, or default readiness.

## Next action

The P1 sequential-canary subplan has now been executed only as an incomplete
harness attempt. Repair and focused-test the runner, record the three failed
attempts, separate compile from steady-state cost, and refresh the budget
before any further material GPU launch. The strict comparator remains fresh and
independent, the factor route remains a candidate rather than a repository
default, and the existing Phase 9B chart-threshold and downstream P2 gates
remain unchanged.
