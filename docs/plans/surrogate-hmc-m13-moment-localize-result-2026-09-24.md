# M13 correction restoration localization — 2026-09-24

Controls-12 saved all requested substep flags after 34 focused tests passed.
The worker emitted its result before the supervisor terminated it at 75.36146
CPU seconds against a 75-second worker allowance. This is a budget-terminated
attempt with usable saved diagnostic evidence, not a clean completed run.
No posterior or control admission follows. Total charged:85.921499CPU and
42.985155484GPU seconds. Logs, manifests, source snapshots and full tensors:
`artifacts/surrogate-hmc-closure-20260922-m13-controls-12/`.

All nine rows pass target covariance, initial standardization, diagonal
iterations and pairwise iterations at time0. Row0 fails affine restoration;
rows0 and1 fail output standardization. These reproduce the two aggregate
correction failures. Row0 filtering covariance diagonals are approximately
9.61e-21,1.86e-20,2.50e-19,2.94e-19, around mean
(.614,.823,.253,-.139). Row1 diagonals range1.44e-16 to1.17e-15.
Both filtering ESS values round to1. Extreme weight concentration is therefore
observed; loss of spread when tiny deviations are added to the world-coordinate
mean in FP32 is the leading remaining explanation. Higher-moment overflow was
real but was not the complete cause.

The seven healthy rows retain identical likelihoods and filter moments versus
controls-11. Scores change by at most .01102 in the existing normalized replay
metric after adding diagnostic outputs. This is a warning under the owner's
FP32 policy, not a new scientific stop condition or evidence of posterior bias.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue master repair | All requested flags saved | CPU allowance reached after result emission; two typed correction failures persist | Whether more particles remove severe concentration | Bounded particle-count diagnostic, then fresh scope-specific tuning if viable | M13 force or posterior qualified |

| Inference status | Finding |
|---|---|
| Hard veto screen | Current N256 controls cannot be selected |
| Statistically supported ranking | None |
| Descriptive differences | Validity, ESS, covariance scales and saved-run score differences |
| Default readiness | FP32 remains directed default; repair not promoted |
| Next evidence needed | Particle-resolution check; scoped controls and independent posterior endpoint |

Post-run red team: increasing N may change the realized design and cannot by
itself prove general robustness. The low-variance representation problem can
still recur. Do not silently add covariance inflation, remove correction,
change the prior, or interpret invalid force points as zero posterior density.
