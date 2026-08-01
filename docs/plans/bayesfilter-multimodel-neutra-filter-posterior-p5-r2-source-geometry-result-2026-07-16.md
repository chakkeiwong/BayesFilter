# P5 R2 Result: STR-UKF Source-Coordinate Plain HMC

Date: 2026-07-16

Status: `SOURCE_KERNEL_HEALTH_FAILED_REPAIR_TRIGGERED`

## Decision

The exact typed `STR-UKF` posterior identity remains admitted, but the selected
source-coordinate fixed HMC kernel is rejected. The run stopped after its first
1,000-draw warm-up chunk because one transition had
`log_accept_ratio < -1000`. No retained posterior sample was produced.

This is a sampler-kernel health failure. It is not an R-hat cap failure, target
failure, posterior-recomposition failure, or filter rejection. The parent R2
subplan named invalid probes and healthy R-hat-cap exhaustion as affine-repair
triggers, but omitted the case where a health-valid short probe later fails the
same frozen health veto during sequential warm-up. The repair therefore needs
this visible amendment; the failure must not be relabeled as nonconvergence.

## Evidence

- Artifact root:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/plain-hmc/attempt-02`
- Result SHA-256:
  `5c01b1f9b36e1c23b4b09b22c13b98dded8f2200d180f98eb1c2b4706bacec46`
- Typed target signature:
  `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`
- Selected short probe: step `0.04`, eight leapfrog steps, acceptance
  `0.955078125`, zero probe divergences, all states moved, finite target and
  status-valid evaluations.
- Sequential warm-up: 1,000 draws per chain, acceptance `0.95025`, all states
  moved, all samples and target values finite, status telemetry valid, one
  energy-error divergence.
- Warm-up modern R-hat: not evaluated because the health veto fired first.
- Retained draws: zero.
- Wall time: `1592.3095` seconds.
- Recursive artifact ledger SHA-256:
  `2b1ad3878a020f0ad26dbec7b5a008649a2a341f27bc303166ff8ba52002deed`.

The 4,000 finite tuning-only states have a within-chain covariance condition
number of `61.0852`, but the divergent archive will not be used as a covariance
or inference artifact. Its pooled mean may be used only as a mode-locator start
hypothesis. A fresh checked posterior-mode Hessian will be the affine geometry
authority.

## Failure And Repair Classification

| Field | Classification |
| --- | --- |
| Failed object | selected source-coordinate fixed HMC kernel |
| Hard veto | one energy-error divergence in sequential warm-up |
| Still valid | target identity, data, prior, chart, filter value/score, status telemetry, archives |
| Invalid inference | all failed warm-up; zero retained draws exist |
| Repair | checked target-bound posterior-mode/Hessian affine coordinates, then fresh probes, warm-up, and retained sampling |
| Unchanged | target, data, method family, thresholds, caps, hardware class, privacy boundary, total R2 budget |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| reject source kernel, continue bounded repair | no health-valid sequential warm-up | energy-error veto fired | whether local affine geometry removes the source-coordinate pathology | execute R2A mode/Hessian geometry gate, then fresh affine HMC if admitted | HMC convergence, comparator admission, NeuTra quality, filter exactness, calibration, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | failed for the selected source kernel |
| Statistically supported ranking | none |
| Descriptive-only differences | probe acceptance/R-hat/ESS and tuning-only covariance |
| Default readiness | not established |
| Next evidence needed | checked affine geometry and fresh unchanged-gate comparator run |

## Post-Run Red Team

The strongest alternative explanation is a rare fixed-kernel energy excursion
rather than globally poor source geometry. That distinction does not change the
decision: the predeclared zero-divergence gate rejects this fixed kernel. The
affine repair must pass fresh health and convergence gates; it cannot erase or
pool the failed warm-up. A failed or unstable mode/Hessian gate will block the
repair rather than justify threshold relaxation.

