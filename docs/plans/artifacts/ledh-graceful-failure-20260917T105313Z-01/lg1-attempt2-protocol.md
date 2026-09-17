# LG1 attempt 2: fixed Gaussian draw design

The independent scalar check in `lg1-localization-01/localization.json` agrees
with the shared flow and importance logits to 6.91e-7 scaled error in TF32.
Its exact conditional empirical-mixture reference exposes importance mean
errors up to 0.285 and covariance errors up to 0.243. This explains part of
the observed approximation failure; incoming-cloud shape is another possible
cause. It does not establish global filtering correctness.

Hypothesis: a jointly randomized Halton design for initial standard normals and
all T process innovations reduces finite-design integration error. Use the
existing TensorFlow/TFP `standard_normal_cloud` implementation, with fresh
repository-derived fixed stream identities and an independently randomized,
centered/whitened residual design. Generation is CPU configuration work; the
full canonical filter and analytical score remain GPU/TF32/XLA. This is a new
finite target candidate, not a new default, averaging target or reduced filter.
It changes no data, prior, reset family, UKF lifecycle, correction capability,
support, approximation threshold, or heldout partition. Target identity now
records draw design and hashes the actual arrays and source dependency closure.

Freeze nine calibration arms: N=256,1024,4096 and flow counts 10,3,1. All other
controls retain the recorded warm hypothesis. Rungs are evaluated in order;
larger rungs are needed only if the preceding scope fails. Smaller flow counts
probe the measured proposal-concentration failure; they are not evidence of
superiority. Calibration replicates 0/2, phi -0.9,0,0.9 nominate candidates only.
Any nominee still needs full four-replicate, seven-parameter calibration,
independent derivative checks, two validation datasets, issued tuning authority,
and untouched claim evidence. Selection favors the first smallest passing scope
under the fixed moment 0.05 and centered-log-ratio 0.02 screens; no ranking claim.

Failure mode: low discrepancy can be lost after nonlinear flow/reset, or the
residual design can distort higher moments despite matching the first two.
Early diagnostics: deterministic input identity, finite Gaussian draws, actual
flow/force status, per-time moments/ESS and centered likelihood differences.
No theoretical rate improvement is claimed. If none passes, preserve the
failed candidate and localize reset shape before the last permitted repair.

Bound: 900 seconds GPU wall including compilation, forecast from 120 seconds
for eight preceding N4096 groups, smaller earlier rungs and 1.5x headroom.
Redistribute unused Phase 4 allowance inside the unchanged 64h/16h ceiling;
retain the master repair reserve. This consumes substantive LG1 attempt 2.
Stop on invalid artifacts, repeated runtime failure, missing diagnostics or
budget exhaustion. Candidate accuracy failure is a repair trigger, never an
automatic rejection of the LEDH research direction.
