# Paper d100 repair result (2026-08-14)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-paper-d100-repair-plan-2026-08-14.md`

## Decision

| Problem | Evidence | Status | Next justified action |
|---|---|---|---|
| 99% Gaussian screen rejects too often | iid exact calibration rejected `7/32` at 99% (`21.875%`) but only `1/32` at 99.9% (`3.125%`) | 99.9% is a better conservative diagnostic level for this rung, but is not a joint-error guarantee | Use 99.9% uniformly for the repair screen; retain the original 99% results and replicate the Gaussian HMC before promotion |
| Gaussian reverse candidate | 99.9% uniform archive adjudication passed all 11 summaries | Recovered as a viable candidate under the conservative screen | Fresh HMC initialization/training-seed replication is needed; no default promotion |
| Gaussian forward candidate | 99.9% still fails projection-2 mean: estimate `-0.05009`, interval `[-0.09812,-0.00206]`, about `3.43` MCSE from zero | Not closed; likely a residual/shared-direction issue or an HMC/MCSE effect, not fixed by interval widening | Replicate with fresh HMC starts and an independent projection implementation before changing the transport |
| Funnel reverse tails | 99.9% still fails `E[y^2]` (`0.839`) and lower-tail residual second moment (`1.054`) | Genuine candidate tail failure; not an interval-level artifact | Use forward-KL or an explicitly tail-weighted reverse repair; validate with exact tail diagnostics and HMC |
| Funnel forward tails | 99.9% uniform adjudication still passes all structural and quantile-law diagnostics | Closed for this target/seed as a viable positive control | Replicate before any robustness or ranking claim |
| Forward-KL runtime | Code path uses inverse IAF: 100 coordinate solves per stage, three stages, for each row; training and repeated 65,536-row heldout inverse evaluations explain the approximately 60x cost over reverse training | Root cause identified; GPU profiler not executed because trusted approval service returned HTTP 502 before process creation | Run the trusted profile when GPU approval is available; then compare incremental inverse caching, coupling flow, or MAF-oriented training as separate reviewed candidates |

## Interval-level conclusion

The 99.9% level is a conservative per-diagnostic screen and should not be
described as a 99.9% joint test. The exact-iid calibration is diagnostic only,
but it demonstrates that the original all-11-at-99% gate was materially prone
to false rejection at the tested sample size. Applying 99.9% uniformly rescues
the Gaussian reverse archive and leaves the Gaussian forward archive rejected;
it also leaves the reverse funnel tail failure rejected. This is evidence for a
better-calibrated screen, not proof that the rescued Gaussian transport is
exact.

## What is closed

The interval-calibration concern is addressed by a versioned uniform 99.9%
adjudication path and iid calibration artifact. The reverse funnel problem is
closed as a diagnosis: it is tail undercoverage, not an R-hat/ESS failure, and
the forward-KL candidate demonstrates one viable repair route on the exact
target. The 99% historical artifacts remain unchanged.

## What is not closed

The Gaussian forward projection-2 discrepancy remains unresolved. It cannot be
classified as a transport defect without a fresh-start HMC replication and an
independent projection/whitening implementation. The forward runtime diagnosis
is mathematically clear from the inverse IAF algorithm, but a GPU profiler is
still required to attribute the wall time quantitatively. Its launch was
blocked by the trusted approval service (`502 Bad Gateway`), not by a numerical
or device failure; no profile result is claimed.

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Reverse funnel remains vetoed; Gaussian forward remains vetoed at 99.9%; Gaussian reverse and funnel forward pass their respective 99.9% archive screens |
| Statistically supported ranking | None; iid calibration estimates are descriptive and no replicated objective comparison exists |
| Descriptive-only differences | 99% versus 99.9% pass rates, training runtime, clipping, tail estimates, and loss |
| Default-readiness | Not assessed and not promoted |
| Next evidence needed | Gaussian forward fresh-start replication, independent whitening/projection check, trusted GPU profile, and a separately reviewed tail-aware reverse repair if reverse-KL remains a requirement |

## Artifacts

- Gaussian 99.9% adjudications: `gaussian-forward-analytic-adjudication-999-r1/` and `gaussian-reverse-analytic-adjudication-999-r1/`.
- Funnel 99.9% adjudications: `funnel-reverse-analytic-adjudication-999-r1/` and `funnel-forward-analytic-adjudication-999-r1/`.
- iid calibration: `gaussian-interval-calibration-r1/`.
- GPU profile: not created; launch was blocked before process creation by the trusted approval service.
