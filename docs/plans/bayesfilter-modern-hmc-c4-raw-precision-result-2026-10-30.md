# C4 saved-draw audit: arithmetic agrees; reliability remains open

30 October 2026. The corrected audit completed in 5.125527593 seconds. All
seven C3 candidates and all 31 quantities reproduce the stored mean, sample
standard deviation, per-chain lugsail variance, MCSE, precision decisions and
independent-reference decisions. The merged TensorFlow/XLA estimator agrees
with independent NumPy arithmetic. All 643 original source/input hashes match.
The original four passes and three incomplete posterior assessments are unchanged.

The [plan](bayesfilter-modern-hmc-c4-raw-precision-audit-2026-10-30.md) defines
this as a saved-data diagnostic. Committed checkpoints store whitened active
coordinates. Applying the original issued affine map reproduces the retained
model coordinates exactly; independently reconstructing the 31 functionals
differs by at most 1.4210854715202004e-14. The first reader omitted that map
and stopped after 2.284406370 seconds. That failed attempt is preserved. Eight
reader tests now include a nonidentity affine map and its orientation.

## What the calculations explain

For L=25, epsilon=.2, the failing squared-coordinate reference score remains
.1016480121. Its components in reference-SD units are .0525306740 absolute
error, .0483462791 from three estimated MCSEs, and .0007710590 reference
uncertainty. This is a failed operational screen; the small excess above .1
does not establish bias. For L=3/.4, the largest score .1734871327 comprises
.0644647862 error, .1084516059 MCSE margin and .0005707405 reference uncertainty.
L=5/.4 still fails internal precision despite passing reference agreement.

The independent half/base/double batch diagnostic shows sensitivity. At twice
the original batch size, L=9/.4 and L=25/.2 each have one quantity with a
nonpositive per-chain extrapolated variance, which is reported unavailable.
Among available doubled-batch estimates for L=25/.2, MCSE ratios to the original
range from .8570 to 1.3631. This supports investigating estimator stability;
it does not justify selecting a favorable batch size or replacing the original
decision. Marginal standard deviations also vary descriptively relative to the
independent reference. Those differences alone cannot diagnose an unseen tail.

## Decisions and next evidence

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close raw arithmetic/serialization audit | All seven members and 31 quantities agree | Reader error repaired; original outcomes preserved | Finite-run variance estimation and reference coverage | Fresh fixed-count assessment of all members | Reliable stopping or universal posterior correctness |
| Keep all qualified tuning members | Candidate identities and original receipts preserved | Posterior failures remain posterior-only vetoes | Adequacy at other counts and seeds | Preserve all successes, refusals and failures | R-hat as a tuning requirement |
| Keep C4 reliability open | Batch perturbations are explanatory | Two alternative-scale unavailable estimates preserved | Non-Gaussian tails and optional stopping | Prospective paired fixed-count versus replayed stopping diagnostic | New default or calibrated confidence coverage |

| Inference class | Status |
|---|---|
| Hard veto screen | No original numerical veto discovered; reader attempt 01 invalid |
| Statistically supported ranking | None |
| Descriptive-only differences | Batch sensitivity, marginal scales, chain means and reference decomposition |
| Default-readiness | Not established; no estimator or threshold changed |
| Next evidence needed | Fresh, independently seeded fixed-count data and separately qualified tail controls |

The proposed next application diagnostic includes all seven members, with
10,000 retained draws per chain fixed in advance and the same warmup readiness
checks. Replay earlier internal stopping checks on those saved prefixes, without
using reference values to choose counts or candidates. The fixed count is the
existing C3 cap, not a claim of sufficiency. One new independent cohort can
discriminate a specific stopping/count explanation but cannot calibrate general
coverage. Its prospective plan must freeze seeds, cost and complete denominators
before launch. Broader nonlinear-tail calibration remains separate.

## Provenance, accounting and red-team review

Raw results remain under `N5/c4-raw-precision-20261030-01` and `-02`; commands,
input hashes and environment are in their manifests. Compact evidence and
test XML are in `docs/validation/modern-hmc-20261030-c4`. Execution deliberately
hid GPUs and used the unchanged TensorFlow estimator with XLA on CPU. It is
not GPU performance evidence. Both attempts and the focused tests consume the
same 900-second operational reservation. Preparation remains withheld from
the upper campaign balance; separate SIR funds are untouched.

Strongest alternative explanation: ordinary Monte Carlo variation and imperfect
reference-proposal coverage can explain the C3 discrepancy without a sampler
defect. A raw-coordinate or reduction discrepancy would overturn this audit,
but none was found. Its weakest evidence is interpreting seven finite,
outcome-stopped cohorts as information about estimator reliability. A clean
arithmetic audit closes only the tested implementation question.
