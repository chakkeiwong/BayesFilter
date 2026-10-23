# SIR NeuTra live observation — 2026-10-23

Continue the existing frozen-map surrogate-HMC sampler. Tuning and warm-up
are complete; the retained sequence has not met its cumulative R-hat rule.
This observation changes neither the sampler nor its stopping criteria.

The CPU diagnostic read one immutable copy of `fresh-progress.json`, verified
checksums of the loaded tensors and nomination, and excluded all 2,000 warm-up
draws per chain. There are 1,000 retained draws per chain across four chains.
The executable is [inspect_retained.py](inspect_retained.py); its exact command,
Git revision, CPU-only exception, chunk seeds, input hashes and timing are in
[manifest.json](manifest.json). No target evaluation or GPU training took place.
The scientific protocol and pre-run audit are in the [execution plan](../../../sir-neutra-training-execution-2026-10-21.md).

Archived states are the tuner's affine coordinates w. The checked consumer
call chain in `hmc_tensorflow_verified._fresh_sequence` transforms these using
`affine.latent_to_position` before assessing R-hat. The diagnostic uses the
same transformation, z = center + w @ factor.T. These z coordinates are the
input to the frozen learned NeuTra map, not the physical SIR parameters.
The downstream result computes physical-coordinate diagnostics after applying
the learned map in `surrogate_hmc_five_model.inference`.

An executable CPU parity check called the sampler's actual `_affine` consumer
on the archived draws and recomputed its shared rank/folded R-hat diagnostic.
Both the coordinate difference and all R-hat differences were exactly zero;
see [coordinate-check.json](coordinate-check.json).

| Diagnostic | Observed value | Interpretation |
|---|---:|---|
| Warm-up maximum R-hat, latest 1,000 | 1.030659 | Passed the predeclared 1.05 readiness rule |
| Retained cumulative maximum R-hat in z | 1.024754 | Continue toward 1.01 |
| Retained acceptance, chains 1–4 | 81.8%, 79.6%, 80.5%, 78.6% | Descriptive; tuning target remains 70% |
| Mean acceptance probability, chains 1–4 | 81.5%, 80.5%, 79.6%, 79.3% | Descriptive; no convergence claim |
| Recorded divergences / force fallbacks | 0 / 0 | No corresponding health failure |
| Maximum absolute retained energy error | 6.4457 | Finite; warning-only policy |
| z mean | −0.0670, 0.1936, 0.0624 | Descriptive location relative to N(0,I) |
| z standard deviation | 1.2316, 0.9715, 1.1345 | Descriptive scale relative to N(0,I) |
| z covariance eigenvalues | 0.9342, 1.2375, 1.5761 | Observed second moments are close to isotropic |
| z covariance condition number | 1.6872 | Not the Jacobian condition number |
| Largest absolute z correlation | 0.0850 | Small observed linear dependence |

The earlier Jacobian condition range 3.57–4.49 describes how the map stretches
local coordinates; it does not measure residual posterior anisotropy. The
retained covariance above measures the distribution after transforming by
NeuTra, but remains provisional: correlated, incomplete chains can miss tails
or modes. Even exactly identity covariance would not establish Gaussianity or
sufficient whitening for efficient HMC. No comparative improvement is claimed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue the same frozen kernel | R-hat 1.01 not yet reached | Latest finite-state, endpoint, movement and divergence checks pass | Posterior coverage and autocorrelation | Existing master continues cumulative retained sampling, then its declared posterior assessment | Convergence, sufficient whitening, or model/reference agreement |

| Inference status | Finding |
|---|---|
| Hard veto screen | No failed health check in the snapshot; loaded tensor checksums and all 15 bound source hashes match |
| Statistically supported ranking | Not a comparison; no ranking |
| Descriptive-only differences | Acceptance, covariance, correlation and scale relative to nominal N(0,I) |
| Default-readiness | Not established by this observation |
| Next evidence needed | Existing cumulative sampling and terminal original-coordinate R-hat, bulk/tail ESS and MCSE diagnostics |

At 02:16:53 Hong Kong, 9.8198 HMC hours remain; the current worker cap ends
at 12:06:04. The master keeps running toward its cumulative R-hat rule,
20,000 retained draws per chain, or the time cap. It does not add an independent
validation chain or retune from retained draws. The unused 8.5678 training
hours are a separate budget and are not automatically transferred to HMC.

The strongest alternative explanation for apparently good geometry is that
these early chains cover only a locally regular part of the posterior. Broader
retained coverage revealing strong correlations, scale variation or disconnected
modes would overturn that interpretation. The weakest evidence is completeness
of posterior coverage, as the cumulative convergence rule is not yet satisfied.
