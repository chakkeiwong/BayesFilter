# P6 R2 Result: SIR-SGQF Same-Target Comparator

Date: 2026-07-16

Status: `COMPARATOR_ADMITTED`

## Result

The repaired GPU/XLA affine-preconditioned plain-HMC campaign passed for typed
`SIR-SGQF` target signature
`0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`.

- result:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/plain-hmc-affine/attempt-02/result.json`;
- result SHA-256:
  `621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2`;
- recursive-hash ledger SHA-256:
  `31ce7b0cb68d70f847b1b423abd913279ee259699576895e8a08ef00fc98356f`;
- all 29 declared artifact hashes verified.

## Evidence

| Gate | Result |
| --- | --- |
| Fixed-kernel tuning | step `0.20`, eight leapfrog steps, disjoint 1,000 burn-in plus 1,000 draws, max modern R-hat `1.000734 <= 1.01` |
| Tuning health | finite target/state/log acceptance, all chains moved, target status valid, zero declared energy-error divergences |
| Warm-up | 2,000 draws per chain retained separately; latest 1,000-draw max modern R-hat `1.001840 <= 1.05` |
| Retained | 4,000 draws per chain, max modern R-hat `1.000497`, min bulk ESS `15538.66`, min tail ESS `15431.38` |
| Retained health | all chunks finite/status-valid, all chains moved, zero declared energy-error divergences |
| Identity | exact admitted typed target and target-specific Laplace geometry hashes verified before sampling |
| Runtime | `3330.25` seconds on trusted RTX 4080 SUPER GPU/XLA with TensorFlow memory growth |

Acceptance was `0.99275` in tuning verification and approximately `0.994` in
warm-up/retained chunks. It is explanatory only. It neither selected the
kernel nor established convergence.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit same-target comparator | passed modern R-hat, ESS, health, identity, and archive gates | no target, finite, status, energy, cap, identity, or archive veto | filter posterior is SGQF-approximate and scientific calibration is untested | target-specific batched GPU/XLA NeuTra recipe screen | NeuTra quality, SGQF exactness, calibration, forecasting, robustness, superiority, or readiness |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | passed |
| Statistically supported ranking | none; no method ranking was attempted |
| Descriptive-only differences | acceptance, runtime, posterior summaries, and short-probe differences |
| Default readiness | not established |
| Next evidence needed | learned-transport engineering admission followed by fresh NeuTra HMC and simultaneous same-target agreement |

## Repair Accounting And Red Team

Attempt 1 was invalidated because step `0.40` was chosen by short-probe ESS
despite modern R-hat `2.7322`. Its samples were not reused. Attempt 2 reused
only hash-bound short-probe rows, added a disjoint modern-R-hat tuning gate,
and selected step `0.20`. The platform rejected the first process-creation
request by permission-review timeout before any directory existed; the
identical retry launched successfully and does not count as a scientific run.

The strongest alternative explanation is that the excellent diagnostics are
mainly due to the Laplace affine chart and a locally near-Gaussian approximate
posterior. That does not invalidate the comparator, but it means a learned IAF
must be compared against the affine-only chart and cannot be promoted from loss
reduction alone.
