# Covariance available from the M13 warmup, 2026-10-02

Using the archived 9,000 warmup steps per chain would have yielded a substantially
different empirical covariance, with stronger negative correlation and a much
smaller observation-scale variance than the frozen matrix. However, keeping the
existing 10% shrinkage toward the broad prior would increase both diagonal
entries relative to the frozen matrix. More warmup alone does not remove the
regularizer's large effect on scale.

This is a deterministic retrospective calculation for the M13/data1704/T20
frozen importance-likelihood posterior used in the surrogate-HMC campaign.
Four chains provide 36,000 correlated warmup states. No new samples, adaptation,
endpoint evaluations or kernel changes were performed. These are the matrices
that could be estimated from the actual saved states. A kernel adapted during
warmup would generate a different trajectory, so its resulting covariance is
not known from this calculation.

## Covariance estimates

Let theta=(log process scale, log observation scale). The displayed entries are
Cov(theta), in log-parameter units, not the latent-state process/observation noise
covariances. Entries 12 and 21 coincide.

| Estimate | Variance 1 | Covariance 12 | Variance 2 | Correlation |
|---|---|---|---|---|
| Original frozen matrix | 0.09739122 | -0.01087504 | 0.06986349 | -0.1318 |
| All warmup, empirical | 0.07749974 | -0.02292811 | 0.03124836 | -0.4659 |
| All warmup, existing regularizer | 0.11779507 | -0.02063529 | 0.07616882 | -0.2179 |
| Last 4,500, empirical | 0.07240546 | -0.01791888 | 0.02712897 | -0.4043 |
| Last 1,000, empirical | 0.07047330 | -0.01990957 | 0.03028555 | -0.4310 |

The full-warmup empirical covariance is

    S = [[ 0.07749974, -0.02292811],
         [-0.02292811,  0.03124836]].

Relative to the frozen matrix, empirical variance 1 is 20.4% smaller, empirical
variance 2 is 55.3% smaller, and negative cross-covariance is about twice as large.
The corresponding empirical SDs are 0.2784 and 0.1768, compared with the frozen
matrix's 0.3121 and 0.2643. The warmup mean is (0.05419, -0.12042), compared with
the frozen center (0.04617, -0.08543). Center changes affect location, not
covariance or curvature.

The actual tuner shrinks toward P=diag(0.48045302, 0.48045302), using

    C = 0.9 S + 0.1 P
      = [[ 0.11779507, -0.02063529],
         [-0.02063529,  0.07616882]].

The saved zero-jitter/eigenvalue settings do not materially alter that formula
in this calculation. Its Cholesky factor is

    L = [[ 0.34321286, 0],
         [-0.06012390, 0.26935839]].

Thus keeping the regularizer would increase variance 1 by 21.0% and variance 2
by 9.0% relative to the frozen matrix. The 10% prior contribution alone is
0.0480453 on each diagonal, which exceeds the empirical observation-scale
variance 0.0312484. It supplies 63.1% of the resulting second diagonal. A 10%
mixture coefficient is not a 10% change in variance when prior and posterior
scales differ substantially.

The last-half and final-1,000 empirical estimates remain qualitatively similar
to the full-warmup estimate. They are sensitivity views, not selected alternatives.
There is still appreciable chain variation: across the full warmup, process-scale
variance ranges from 0.0603 to 0.1074 and correlation from -0.628 to -0.325.
These estimates do not establish stationarity or a known posterior covariance.

## What these matrices imply for the saved geometry

For each fixed matrix C=LL^T, transform unchanged retained draws by
z=L^{-1}(theta-c). Then sample Cov(z)=L^{-1} sample Cov(theta) L^{-T}.
The table uses all 12,750 saved post-warmup draws per chain, none of which
were used to fit the warmup estimates.

| Transform based on | Transformed SDs | Correlation | Covariance condition number |
|---|---|---|---|
| Original frozen matrix | 0.996, 0.609 | -0.256 | 3.083 |
| All warmup, empirical | 1.117, 0.991 | 0.106 | 1.377 |
| All warmup, existing regularizer | 0.906, 0.578 | -0.136 | 2.567 |

The empirical warmup matrix makes these saved draws substantially closer to
identity covariance. With existing shrinkage, correlation is reduced but the
second transformed SD remains only 0.578. The same pattern is visible in the
first 9,000 retained draws: the empirical warmup transform gives SDs 0.945/1.015,
correlation 0.103 and condition number 1.287; existing shrinkage gives SDs
0.766/0.589, correlation -0.097 and condition number 1.754.

These deterministic coordinate comparisons establish a global covariance
mismatch and identify shrinkage as an important contributor. They do not
establish improved future sampling. Re-expressing existing draws does not
improve their temporal mixing, acceptance or inference. In particular, neither
longer mass adaptation nor weaker shrinkage is proved to resolve the low-process-
scale proposal inefficiency observed earlier. No default change is promoted.

## Verification and provenance

Source warmup count and phase were checked; the aggregate warmup tensor exactly
matches 36 archived 250-transition chunks. The history loader rechecked source
hashes, states, seed schedule, fixed settings and archived health. Original
factor/covariance and inverse/forward parity passed, and retained arrays matched
the independently saved archive exactly. The centered-moment covariance formula
agrees with the repository Welford estimator to 7.22e-16. All candidate covariance
matrices are finite and positive definite. The same repository regularizer was
called directly using the saved configuration.

Source anchors: `hmc_tensorflow_tuning.py::_welford_covariance`,
`hmc_tensorflow_tuning.py::_regularize_covariance`,
`surrogate_hmc_m13_cumulative.py::load_history`, and the source/nomination
manifests linked with checksums in `run-manifest.json`. The computation tests
estimator/regularizer call parity; it does not launch or certify a changed HMC
consumer. Prior source artifacts are preserved.

## Decision and inference status

| Decision item | Finding |
|---|---|
| Decision | Warmup supplies a materially different empirical matrix; unchanged prior shrinkage preserves substantial scale inflation. |
| Primary criterion status | Requested saved-state covariance comparison completed; no sampler-promotion criterion introduced. |
| Veto diagnostic status | Archive, continuity, covariance parity, finite/SPD checks pass. |
| Main uncertainty | Autocorrelated, possibly nonstationary warmup; counterfactual adaptive trajectories are unobserved. |
| Next justified action | If tuning repair is requested, evaluate covariance-window and regularization choices jointly, then retune step size/trajectory before retained sampling. |
| What is not concluded | Removing shrinkage is safe as a default, 9,000 warmup steps are sufficient generally, or improved global geometry will cure local proposal failure. |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new numerical-validity or artifact veto. |
| Statistically supported sampler ranking | None; no alternative sampler was run. |
| Descriptive-only differences | Covariance estimates, chain/window sensitivity, and algebraic geometry of existing draws. |
| Default readiness | Unchanged. No covariance, regularizer, warmup or stopping rule changed. |
| Next evidence needed | A scoped tuning-repair comparison including retained cumulative diagnostics, regional proposal behavior and Monte Carlo uncertainty. |

Post-run red team: treating pooled warmup as stationary is the main weakness.
Tail exploration could change all estimates, while local force mismatch could
still dominate sampling even with a better global covariance. A controlled
kernel comparison showing no improvement in regional movement would weaken the
case for covariance repair as the explanation of slow exploration. Retained
samples were not used to select a preferred warmup window or a new default.

## Execution

CPU-only TensorFlow/TFP post-run diagnostic, CUDA deliberately hidden before
import, non-XLA exception. No random draws or HMC transitions. Attempt 01 failed
because its estimator-parity wrapper supplied a dynamic state count to a
static-shape Welford helper. Attempt 02 used the exact [36000,2] signature; no
sampler code was changed. Both attempts and logs are preserved. The successful
analysis completed in 2.542 seconds including TensorFlow import.

Artifacts: `analyze.py`, `report.json`, `run-manifest.json`, `command.log`,
`execution-receipt.json`, and this `result.md` under the versioned output root.
The expired sampling allocation is unchanged. This requested diagnostic is
complete; implementing and testing a different metric would be a new repair
phase, not the master's existing unchanged-kernel continuation.
