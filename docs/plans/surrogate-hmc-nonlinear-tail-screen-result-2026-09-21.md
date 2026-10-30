# M10 likelihood tail screen: reference repair required

The513-node, radius9 state-grid reference is inadequate for full-prior M10
inference. All four calibration/validation datasets failed13 of29 probes,
for52 failures out of116. These are numerical reference vetoes. The checked
DCTR consumer mechanics remain valid, and the surrogate-HMC research
direction remains open.

The computation follows the actual M10 likelihood law but integrates over a
finite state domain. At calibration-nominal(lambda_Q,lambda_R)=(3*log2,0),
extending radius9 to13.5 changed log likelihood by0.0315115 and the score
by0.255738. At(0,-6*log2), doubling nodes513 to1025 changed log likelihood
by0.7636825 and score by1.0267853. At(-6*log2,0), the corresponding changes
were0.0126532 and0.0944441. Wide process noise needs more state domain;
narrow process or observation noise needs more resolution. The computed
likelihood is therefore wrong relative to a claim of full-prior accuracy at
the declared tolerances. Local checks did not establish that broader claim.

The independently derived likelihood/prior envelope gives unnormalized mass
outside the six-standard-deviation parameter box at most2.73462e-6, and
outside the seven-standard-deviation box at most1.36812e-8. Neither is a
posterior probability without a qualified evidence lower bound. Tail mass
cannot simply be discarded or the normal prior truncated.

Evidence: [report](artifacts/surrogate-hmc-all-models-20260921-m10-tail-screen-01/report.json),
[manifest](artifacts/surrogate-hmc-all-models-20260921-m10-tail-screen-01/run-manifest.json)
and [plan](surrogate-hmc-nonlinear-tail-screen-2026-09-21.md).
Command: `python scripts/screen_surrogate_hmc_nonlinear_tails.py --output
docs/plans/artifacts/surrogate-hmc-all-models-20260921-m10-tail-screen-01`.
This was a CPU-only FP64/XLA diagnostic with GPU intentionally hidden and
fixed TensorFlow threads2/1. It preserved464 per-resolution rows, seeds,
source/data hashes and partial-write capability. Charge26.864560791 CPU
seconds, wall17.399234273 seconds, GPU0. No claim data or retained draws used.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Repair reference |52/116 probes fail | Mesh/domain differences and mass-loss/edge checks | Global posterior-weighted error remains unknown | Evaluate quadrature with intervals following normal densities and both observation roots | Rejection of DCTR or surrogate HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | Present state grid cannot serve as the full-prior reference |
| Statistically supported ranking | None; deterministic numerical comparison |
| Descriptive-only differences | Recorded mesh/domain changes above |
| Default-readiness | No sampling default established |
| Next evidence needed | Independently cross-checked tail quadrature, posterior integration and normalization bounds |

Post-run red team: the domain failure can include the negative root of the
quadratic observation law as process noise grows. A repair that follows only
the positive root would silently change the model likelihood. The next
quadrature must include both roots and compare against the qualified overlap
region. Reference repair, not abandonment of the HMC mechanism, is justified.
