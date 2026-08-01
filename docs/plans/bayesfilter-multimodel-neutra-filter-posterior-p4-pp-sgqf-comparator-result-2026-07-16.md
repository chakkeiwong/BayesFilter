# P4 Result: PP-SGQF Same-Target Plain HMC

Date: 2026-07-16

Status: `COMPARATOR_ADMITTED`

## Decision

The exact admitted fixed-SGQF level-2 predator-prey posterior has a same-target
plain-HMC comparator. A target-specific Laplace geometry diagnostic passed, and
fresh GPU/XLA HMC passed all unchanged warm-up, retained, health, status,
rank/folded R-hat, and ESS gates.

## Evidence

- Geometry root:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-SGQF/laplace-geometry/attempt-01-20260715T165000Z`
- Geometry result SHA-256:
  `b54343fdee59c3f86ffb8f8ac69ba0ea31b7a0c780a4f2eb290374df060cabc3`
- Geometry final score infinity norm: `5.80e-05`.
- Hessian step-size relative gap: `4.00e-09`.
- Raw precision spectrum:
  `(0.9913,1.0246,2.6594,41.4081,43.5566,104.0651)`; no clipping.
- HMC root:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-SGQF/plain-hmc-laplace/attempt-01-20260715T170000Z`
- HMC result SHA-256:
  `015348e162d35cb062be274eb4b420ee881eb364473b5b7ce5acfdca7c0192ec`
- Typed signature:
  `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`.
- Selected kernel: step `0.50`, eight leapfrog steps.
- Warm-up: `2,000` draws per chain; max rank R-hat `0.99946`, max folded R-hat `1.01140`.
- Retained: `4,000` draws per chain; max modern R-hat `1.00193`.
- Minimum bulk ESS: `25,758.44`; minimum tail ESS: `6,408.11`.
- Hard vetoes: none.
- Artifact hash ledger: 25 files verified.
- Focused regression: 9 tests passed.

## Decision And Inference Status

| Field | Status |
| --- | --- |
| Primary comparator criterion | passed |
| Target/status/energy vetoes | clear |
| Statistically supported ranking | none |
| Descriptive-only evidence | probe metrics, acceptance, SGQF/UKF posterior summaries |
| Default readiness | not established |
| Next action | target-specific SGQF dense-IAF protocol and downstream NeuTra agreement |
| Not concluded | SGQF exactness, superiority, calibration, robustness, NeuTra quality, or readiness |

## Post-Run Red Team

The Laplace geometry is local and derived from one synthetic target. Passing
same-target HMC establishes a valid comparator for this cell, not broad SGQF
robustness. Another trajectory could have different curvature or modes. The
weakest evidence remains cross-fixture generality; target identity and
same-target sampler validity are the strongest evidence here.

