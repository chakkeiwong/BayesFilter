# Phase 8 Controlled Calibration Native Review

Date: 2026-07-17

Verdict: `AGREE_SMOKE_ONLY`

The reviewed controlled harness uses the Phase 9 confirmation sample shape
`[4,448,2,10]` per arm without reading any G/H confirmation draw. Synthetic
Gaussian horizon laws are initialized in their invariant distribution, use
draw-axis AR coefficients strictly inside the stationary region, and separate
left/right Philox domains.

Co-primary margins are `0.15` standardized mean and `log(1.15)` log variance,
both strictly smaller than the owner-approved material anchors `0.20` and
`|log(1.25)|`. Required material controls include persistent and horizon-1
mean shifts in both signs and variance ratios `1.25` and `0.80`. Skew and
dependence controls remain explanatory.

The only selectable nomination quantity is MMD tolerance from
`(0.005,0.01,0.02,0.04,0.08,0.16)`. Block length, margins, alphas, bandwidths,
mixture weights, covariance rule, confirmation shape, and families are fixed.
The 20-replication nomination may nominate the smallest tolerance passing all
required family screens. It cannot validate the design. A later 60-replication
run must bind the exact nomination receipt and report one-sided 95% exact
binomial bounds before any design freeze.

The smoke is narrower: one iid-null and one persistent `+0.20` mean replicate.
It always returns `...SMOKE_PASSED_NOMINATION_REQUIRED` and cannot nominate a
tolerance. Serious execution requires all generated/statistical tensors on GPU
and one trace for every compiled program.

Focused checks: `67` calibration/predictive tests passed; Python compilation
and `git diff --check` passed. No material issue remains for the bounded smoke.

## Repair 01 Review

Verdict: `AGREE_SMOKE_REPAIR_01_ONLY`

The original smoke exposed one material implementation defect in the reviewed
runner: the fixed 20-feature margin vector called `tf.concat` without `axis`.
It failed before classification or receipt construction, so it provides no
calibration or power evidence. The repair adds `axis=0` through a narrow
`_feature_margins` helper and directly tests the vector's shape and ordering.

The repair preserves every prospective scientific and statistical choice,
including values, families, seeds, tensor shape, alphas, bandwidths, tolerance
ladder, ridge policy, placement/trace gates, and resource cap. Focused repair
checks passed: `9` controlled-calibration tests; Python compilation; and
`git diff --check`. A separately named repair receipt is required. Nomination
and validation remain unauthorized until that receipt is audited.

## Nomination Review

Verdict: `AGREE_NOMINATION_ONLY`

The passing smoke receipt is now hard-bound by path, hash, historical runner
hash, decision, family scope, tensor shape, pilot lineage, one-trace counts,
GPU/XLA trust metadata, and null selection/validation fields. Validation fails
closed until a future patch embeds the exact nomination receipt and selected
tolerance.

The 20-replication criteria are unchanged. The added sequential rule is
futility-only: it stops before the final replication only when no tolerance
can reach every frozen threshold even under favorable remaining outcomes. It
cannot nominate early, and `_nomination_pass` now explicitly requires all 20
replications. Unit cases cover incomplete-run rejection, recoverable
continuation, mathematically impossible recovery, and the strict pre-final
boundary. The runner emits counts-only progress for supervision.

Focused checks passed: `72` controlled/predictive tests; Python compilation;
and `git diff --check`. The `1800`-second GPU 1 cap is conservative relative to
the full-shape smoke and protects the exact maximum run. No G/H confirmation
draw, tolerance validation, predictive-equivalence decision, or posterior
truth claim is authorized.
