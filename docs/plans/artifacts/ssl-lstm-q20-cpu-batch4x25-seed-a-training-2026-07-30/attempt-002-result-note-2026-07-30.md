# q=20 CPU Seed-A Resume Attempt 002 Result Note

Date: 2026-07-30
Status: `CPU_DIAGNOSTIC_COMPLETED_SCREEN_PASSED`

## Decision

The authorized seed-A CPU diagnostic campaign completed at 2,000 program
updates. Final support and the final-only 256-row audit were finite, the
predeclared paired validation screen passed, and no result veto fired. Preserve
the candidate as a CPU diagnostic screen pass. Do not promote it to HMC,
posterior, transport, production, or default status.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-strict-cpu-training-plan-2026-07-22.md` |
| Result | `seed-a/result.json` |
| Summary | `summary-attempt-002.json` and `summary.json` |
| Latest checkpoint | `seed-a/checkpoint-2000.json` |
| Environment | Python `3.13.13`; TensorFlow `2.20.0` |
| Device/backend | CPU-only, CUDA hidden, non-XLA diagnostic exception, `float64` |
| Topology | 25 persistent workers x 4 rows; training batch 100; affinity `0-49` |
| Seeds | initialization `(20260719, 12101)`; training `(20260719, 13101)`; validation `(20260719, 14101)` |
| Attempt-002 command | Recorded in `restart-attempt-002-record.md` and `launch-attempt-002.json` |
| Attempt-002 cumulative wall | `35600.635240058 s`, including conservative prior charge `31350 s` |
| Active cumulative cap | `51500 s` |
| Result SHA-256 | `07cc43fd9c9f4a0b79826d0bddb91723cdc0378f9faaa49230d981034a70a90e` |

## Evidence

| Role | Status | Evidence |
| --- | --- | --- |
| Primary completion criterion | Passed | Controller stopped at program step 2000 with `maximum_steps_reached`; final support and audit completed |
| Promotion veto: paired validation loss | Passed | Best-minus-initial paired mean `-31.825414454513`; one-sided 95% upper bound `-22.87873833506061` |
| Promotion veto: final support | Passed | All finite; round-trip max `3.552713678800501e-15`; inverse radius `4.000000000000001` |
| Final explanatory audit | Finite | 256 rows; mean loss `41.3686568531506`; final-only fold `20260721` |
| Resource/thread checks | Passed | 254 attempt-002 checks; 25 configured workers; parent and workers exited cleanly; parent peak RSS `727347200` bytes |
| Result vetoes | None | `vetoes: []` |

The controller found meaningful validation improvements through step 1500. At
step 1750 it restored the best step-1500 trainer state and reduced the learning
rate from `0.0004` to `0.0002`. At step 2000 it reached the maximum-step stop;
best step remained 1500. The result therefore reports terminal program step
2000 and terminal optimizer step 1750. This is correct for the restore-and-run
repair path, not an artifact inconsistency.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Classify seed A as CPU diagnostic screen pass | Passed | Paired validation and final support vetoes passed | One seed; CPU/non-XLA execution | Preserve result and checkpoint artifacts | No convergence or posterior claim |
| End this campaign | Completed | No continuation veto remains relevant | Unused authorized headroom is not a new experiment | Launch nothing further from this authorization | No implicit seed B, retuning, or HMC |
| Withhold promotion | Ineligible execution class | GPU/XLA and independent-seed evidence absent | Cross-backend behavior and seed robustness | Require a separate reviewed campaign before claim-bearing use | No transport/default/production promotion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No nonfinite, support, artifact, thread, memory, or result veto |
| Statistically supported ranking | None; no method/architecture comparator and only one training seed |
| Descriptive-only differences | Checkpoint losses, learning-rate transition, runtime, and audit mean |
| Default-readiness | Ineligible CPU diagnostic exception |
| Next evidence needed | Independent seed replication and claim-bearing batch-native GPU/XLA training before any HMC proposal |

## Scope And Statistical Limits

The paired screen is the predeclared within-stream validation veto and supports
the label `CPU_DIAGNOSTIC_SCREEN_PASSED`. It does not establish superiority over
another method or architecture. The recurring validation batch nominated and
selected best step 1500; it is not untouched model-selection evidence. The
256-row audit used a final-only fold and is finite, but its raw mean is
explanatory because the plan declared no comparative audit threshold or
uncertainty-based ranking criterion.

## Post-Run Red-Team

- Strongest alternative explanation: the observed screen pass may be specific
  to seed A and the CPU/non-XLA execution lane.
- What would overturn the classification: a hash or schema failure, a failed
  final support computation, or evidence that the final-only audit reused a
  tuning fold. None was observed in the checked artifacts.
- Weakest evidence: raw checkpoint and audit loss magnitudes. They remain
  descriptive and do not support ranking or convergence.
- The campaign result answers whether this one CPU stream can complete its
  diagnostic screen. It does not answer whether NeuTra is suitable for HMC or
  whether this architecture is scientifically adequate.

## Nonclaims

This result does not establish convergence, posterior correctness, HMC
readiness, transport promotion, statistical superiority, architecture ranking,
production readiness, scientific validity, seed robustness, or a change to the
repository GPU NeuTra default. It does not authorize seed B or another run.
