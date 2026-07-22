# P6 R2 Subplan: SIR-SGQF Same-Target Comparator

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `COMPLETE_COMPARATOR_ADMITTED`

## Objective And Entry

Build a target-specific affine geometry for typed `SIR-SGQF` signature
`0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`,
then tune and run a same-target plain-HMC comparator with separate adaptive
warm-up and retained archives. No training begins before comparator admission.

Entry requires GPU R1B result SHA-256
`5cca9efae6147dbdcbd5ad12d0371451b58b6d26cc879ad1c267c0f40d100ea2`
and exact identity file SHA-256
`820b94f5158d0db95b9f3ad075d564eef8d0d8a9259b82404093d824d3281c5c`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can affine-preconditioned plain HMC consistently sample the unchanged typed SIR-SGQF approximate posterior? |
| Geometry baseline | origin in the three log coordinates; no UKF point, mass, sample, or identity reuse |
| Mode route | GPU/XLA SGQF value/score, damped Newton, at most 8 iterations |
| Geometry pass | finite/status-valid point, score infinity norm `<=1e-4` or checked step/value stagnation; stable two-step Hessian; valid affine chain rule |
| Comparator pass | separate warm-up and retained modern R-hat gates, retained ESS gates, finite health, no divergence/energy veto, exact target identity |
| Hard vetoes | identity/hash drift, invalid status, unconverged geometry, unstable Hessian, nonfinite sampler, divergence/energy failure, warm-up cap, retained cap, or archive drift |
| Explanatory only | acceptance, mode location, curvature, runtime, posterior truth distance |
| Not concluded | exact filter posterior, epidemiological calibration, NeuTra quality, superiority, forecasting, robustness, readiness |

## Geometry

- FD Hessian of the analytic total score at steps `1e-4` and `5e-5`.
- Trust-radius infinity norm `1.0`.
- Fixed line multipliers `(1,0.5,0.25,0.125,0.0625,0)`.
- Terminal Hessian relative Frobenius gap `<=1e-3`.
- Regularize the symmetrized negative Hessian with eigenvalue floor
  `max(1e-8,1e-6*largest_abs_eigenvalue)`.
- Affine map `theta=center+z@factor.T`, where
  `factor=chol(inverse(regularized_precision))`; include constant log
  determinant and verify round-trip/value/score gaps `<=1e-10`.

Failure stops the comparator at `COMPARATOR_BLOCKED_GEOMETRY`; local geometry
failure is not evidence against NeuTra or the SIR target.

## HMC Runtime

If geometry passes:

- four chains at affine zero plus fixed small offsets;
- eight leapfrog steps;
- step grid `(0.025,0.05,0.10,0.20,0.30,0.40)`;
- `64` burn-in plus `128` draws per tuning probe;
- order health-valid probes by lowest modern R-hat, then maximum minimum
  rank-normalized bulk ESS and grid order; short probes nominate only and
  cannot admit a kernel;
- verify nominees in that frozen order using disjoint `1000` burn-in plus
  `1000` retained draws; fixed-kernel tuning admission requires modern R-hat
  `<=1.01`, finite health, and no energy-error divergence;
- warm-up chunks `1000`, minimum `2000`, recent window `1000`, cap `10000`,
  modern R-hat `<=1.05`;
- retained chunks `2000`, minimum `4000`, cap `10000`, modern R-hat `<=1.01`;
- retained minimum bulk ESS `>=1000` and tail ESS `>=400`;
- modern R-hat is the maximum of rank-normalized split and folded
  rank-normalized split R-hat;
- preserve separate immutable warm-up and retained source-coordinate archives.

Attempt 2 reuses only the hash-bound valid attempt-1 short-probe rows. Tuning
verification seeds are `(20260716,26401+i)`, warm-up root
`(20260716,26501)`, and retained root `(20260716,26601)`. Attempt-1 HMC samples
are never reused or pooled.

## Repair And Handoff

Localized serialization, archive, XLA, or resource failures may be repaired in
fresh roots under the unchanged target, geometry, ladder, thresholds, hardware
class, and budget. A failed kernel candidate consumes the grid rather than
changing the target. On comparator pass, write the result and draft the
target-specific GPU NeuTra training subplan. On comparator failure, close
`SIR-SGQF` as comparator-blocked and continue P7 synthesis.

## Skeptical Audit

Decision: `PASS`.

The plan does not use identity mass despite large origin scores, does not reuse
UKF evidence, and separates local geometry from sampler promotion. Warm-up is
adaptive and retained, modern folded/rank-normalized R-hat controls both stages,
draws cap at 10,000 per chain, and warm-up/retained archives are distinct. A
short probe cannot become comparator evidence, acceptance cannot nominate by
itself, and a geometry or sampler failure remains cell-local.

## Attempt 1 Audit And Repair

Attempt 1 selected step `0.40` solely because it had the largest short-probe
minimum bulk ESS, even though the same probe's modern R-hat was `2.7322`. That
selection was wrong relative to the canonical tuning-admission policy, which
requires modern R-hat `<=1.01`. The attempt was interrupted after three
warm-up chunks; its selection and samples are diagnostic only.

The repair preserves the target, affine geometry, step grid, leapfrog count,
hardware, convergence gates, and comparator budget. It reuses the valid probe
rows, orders them by short-probe modern R-hat before bulk ESS, and binds them to
attempt-1 `tuning_selection.json` SHA-256
`76e204264d38a51079be8866a39b01e038ffc86030666371b748b89bd3b0a5be`,
adds disjoint modern-R-hat tuning verification, and starts comparator sampling
only after a verifier pass. The no-selection branch blocks tuning without
starting comparator warm-up.

Skeptical re-audit: `PASS`. The repaired baseline is still same-target
affine-preconditioned plain HMC; short R-hat and ESS only order candidates;
disjoint modern R-hat is the tuning promotion criterion; the invalid attempt is not
pooled; stop conditions remain tuning exhaustion, health failure, warm-up cap,
retained cap, identity drift, archive drift, or comparator-budget exhaustion.
