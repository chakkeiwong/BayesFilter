# SSL-LSTM NeuTra Phase 0 Reset Result

Date: 2026-07-14

Status: `PHASE_0_ACCEPTED_PHASE_1_AUTHORIZED`

## Decision

The master program is accepted as the active post-A4 SSL-LSTM sequence. The
locked target and completed A0-A3 engineering artifacts are present, the A4
ordinary-HMC archive remains non-admitted, and no A4 forecast calibration was
run. Phase 1 may freeze the no-oracle validation design. This result does not
authorize NeuTra training, HMC, or a posterior/scientific claim.

## Skeptical Audit

| Challenge | Finding |
| --- | --- |
| Wrong baseline | The exact baseline is the A1 four-coordinate SVD-UKF target, not a historical importance, quadrature, or failed HMC approximation. |
| Proxy promotion | Hashes and focused tests establish only artifact/engineering continuity. A4 acceptance and finite draws are not sampler admission. |
| Missing stop | Target contradiction, missing A0-A3 artifacts, or an unresolvable concurrent edit stops handoff. None occurred. |
| Hidden assumption | The target remains the historical unnormalized-prior SVD-UKF approximation; no exact nonlinear likelihood or posterior oracle is assumed. |
| Stale context | The final A4 result, not its earlier repair plans, controls: cumulative ordinary HMC failed R-hat/ESS/MCSE admission. |
| Environment mismatch | No GPU/runtime evidence was generated in Phase 0. Later serious training and HMC remain trusted GPU/XLA work. |
| Artifact relevance | A0-A3 artifacts define target/forecast machinery; A4 artifacts diagnose geometry and are forbidden as calibration or confirmation draws. |

Audit decision: `PASS_FOR_PHASE_0_RECONCILIATION`. No runtime plan was
executed under this audit.

## Frozen Boundary

| Item | Identity or disposition |
| --- | --- |
| Git base | `3d353253dc93a102722e00cbca8803a1b3fce7fa` on `main`; dirty worktree preserved |
| Target semantic SHA-256 | `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Target source SHA-256 | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| Predictive source SHA-256 | `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed` |
| A0 target-lock file SHA-256 | `1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383` |
| A3 fixture SHA-256 | `eb715725fd0501423844ed57e91e981b1eb8ff8b2ff56ac36737d2f46e20e41f` |
| A3 CPU analytic-control SHA-256 | `f8252b9a0f6bba1bc5350b0516ceaddca04006bfe489acc74ac7f13d7846d82b` |
| A3 GPU/XLA control SHA-256 | `5c31b26fbf20a10b754ad3e99bb8dc1481b12c74669c3b60e8e7cae8e080b693` |
| Final A4 public receipt SHA-256 | `29ea0a7461f4c98043977dd02ed8afe8acff552860f40bc1723775ac638d5392` |
| A4 status | `BLOCKED_CUMULATIVE_MIXING_PROMOTION_VETO_MASS_GEOMETRY_REPAIR_INDICATED`; diagnostic and non-admitted |
| Dense-IAF loader source SHA-256 | `85d612b440f239870f801078e36269e6493100f2877f1506313f89f77c3914f0` |
| Transformed-target wrapper SHA-256 | `1da7775cfb4fd191d663f03e434d1c3e416ed35f412bd13162145acacf571c95` |

The semantic target signature is the scientific identity. Source-file hashes
record the current implementation boundary and may change through reviewed
target-preserving engineering work.

## Active And Stale Classification

| Surface | Classification |
| --- | --- |
| A0-A3 target, forecast, and analytic-control artifacts | Active locked engineering inputs |
| A4 chain shards and final states | Preserved failed-candidate evidence; never training, calibration, confirmation, or audit draws |
| Historical importance/quadrature/sequential-reference artifacts | Diagnostic history only; not a posterior reference and not a prerequisite |
| `bayesfilter/inference/neutra_artifacts.py` and focused tests | Clean Phase 2 ownership surface |
| `bayesfilter/inference/batched_value_score.py` and focused tests | Clean Phase 2 integration surface |
| Kalman/QR/Sylvester, runtime-policy, and quadratic-geometry changes | Concurrent lane; preserve and do not edit for this program |
| Existing `bayesfilter/testing` NeuTra modules | LGSSM/testing lessons only; no production dense-IAF trainer exists |

## Evidence Contract Result

| Decision | Primary status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 0 | Required artifacts and boundaries are distinguishable | No target, artifact-presence, A4-disposition, or concurrent-file veto fired | The dirty worktree requires continued path isolation | Execute and close Phase 1 no-oracle design | Posterior correctness, sampler validity, NeuTra quality, predictive equivalence, or readiness |

## Handoff

Phase 1 must freeze disjoint role-coded seeds, independent training/sampling
replications, prior/shell/tail sensitivity probes, analytic controls, and the
claim ceiling. Phase 2 may begin only after Phase 1 returns
`NO_ORACLE_DESIGN_VALID`.
