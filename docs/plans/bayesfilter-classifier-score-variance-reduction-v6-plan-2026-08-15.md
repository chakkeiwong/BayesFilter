# V6 Classifier-Score Variance-Reduction Campaign

Date: 2026-08-15  
Status: `PRE_EXECUTION_REVIEW`

## Research Intent Ledger

| Field | Frozen definition |
|---|---|
| Main question | Do common random numbers (CRN), four times as many training paths, or both reduce training-bundle variability of the fixed-path classifier score? |
| Candidate mechanisms | paired `theta-delta`/`theta+delta` simulator noise; train count `8192` instead of `2048` per class and delta |
| Expected failure | CRN-induced dependence or extra data may leave variance unchanged, increase bias, or sharpen a misspecified classifier |
| Primary criterion | paired 95% bootstrap upper bound below 1 for the combined arm's baseline-normalized fixed-path and shared-audit bundle-variance ratios |
| Promotion veto | Gaussian exact MSE worsens with a paired 95% lower ratio bound above 1; non-finite output; invalid balance/pairing; incomplete bundle |
| Continuation veto | broken simulator marginals, broken prefix nesting, data leakage, invalid artifact, GPU/XLA/memory failure, exhausted six-hour GPU budget |
| Repair trigger | infrastructure or serialization failure under unchanged design; candidate failure is a result, not an automatic stop |
| Nonclaims | no exact SIR score, no filter validation, no posterior/HMC/default readiness, no claim about natural score variation over observation paths |

## Four-Arm Design

| Arm | Plus/minus simulator noise | Train paths per class and delta |
|---|---|---:|
| `independent_n2048` | independent | 2048 |
| `crn_n2048` | identical base noise within each plus/minus pair | 2048 |
| `independent_n8192` | independent | 8192 |
| `crn_n8192` | identical base noise within each plus/minus pair | 8192 |

For each of ten outer bundle replicates, all arms use matched model seeds and
the same validation, calibration, test, fixed, and audit paths. The `n2048`
training data are exact prefixes of the corresponding `n8192` banks. The CRN
and independent arms share the minus bank; the CRN plus bank reuses it, while
the independent plus bank uses a separate seed. Pair identity is preserved in
the artifact.

CRN changes dependence between empirical class samples but not either class's
marginal simulator law. Before training, tests must verify both marginals on an
exact Gaussian fixture and verify that pair members never cross a data split.
Every training minibatch is constructed from whole plus/minus pairs in all four
arms. Independent-arm pairs have independent base noises but share a pair ID;
CRN-arm pairs share both pair ID and base noise. Pair-cluster shuffling is
therefore common to all arms and is not an extra CRN treatment.

Only the training split receives the four treatments. Validation,
temperature-calibration, and test splits remain independent plus/minus samples
and are shared across arms within a bundle replicate. This isolates training
simulation variance from validation and calibration noise.

## Frozen Controls And Data

- Models: exact Gaussian harness followed by SIR.
- Cells: all nine `(T,j)` combinations for `T in {20,40,50}` and
  `j in {0,1,2}`.
- Ten independently generated/trained bundle replicates per arm.
- Shared evaluation bank: 128 same-parameter `T=50` paths, sliced into exact
  prefixes; one fixed path is evaluated separately.
- Perturbation grid and anchored basis are unchanged from V5.
- Architecture and L2 controls are frozen from each model's completed V5
  selection artifact. They are held fixed across arms to isolate the two
  variance-reduction mechanisms. This is a controlled ablation, not a claim
  that those controls are optimal at `n8192`.
- Optimizer, epoch cap, early stopping, calibration rule, XLA, dtype, and TF32
  policy are unchanged from V5.
- Bundle replicates use distinct training/validation/calibration/test seeds.
  All four arms inside a replicate use the same non-training splits.

## Variance Estimands

Let `hat{s}_{a,b,c}(Y_m)` be arm `a`, bundle `b`, cell `c`, audit path `m`.
The target is variation across `b` conditional on the same `Y_m`.

Per-cell audit bundle variance:

`V_a,c = mean_m Var_b(hat{s}_{a,b,c}(Y_m))`.

Per-cell fixed-path bundle variance:

`Vfixed_a,c = Var_b(hat{s}_{a,b,c}(Y_fixed))`.

For a joint dimensionless summary, define `w_c` as the between-audit-path
standard deviation of the ten-bundle mean from `independent_n2048`. Then

`J_a = mean_c V_a,c / max(w_c^2, numerical_floor)`

and analogously `Jfixed_a`. The baseline denominators are frozen once and used
for every arm. This separates estimator variability from natural path scale;
the unnormalized cell table remains authoritative if a scale is near zero.

Use a paired cluster bootstrap with 5,000 replicates, resampling bundle IDs and
audit-path IDs while preserving arm pairing and all nine cells. Report 95%
percentile intervals for each arm/baseline variance ratio. Fixed-path ratios
resample bundle IDs only.

## Gaussian Accuracy Guard

For every Gaussian audit path and the fixed path, compute the exact score.
Report per-cell bias, variance, RMSE, and joint paired MSE ratios. Lower bundle
variance does not support a repair if it merely concentrates around a worse
answer.

The combined arm is classified:

- `variance_reduction_supported` only if both joint audit and fixed-path
  variance-ratio 95% upper bounds are below 1;
- `accuracy_harmed` if the Gaussian joint MSE-ratio 95% lower bound is above 1;
- `descriptively_favorable` if point ratios are below 1 but intervals cross 1;
- `no_supported_reduction` otherwise.

Individual CRN and path-count effects receive the same descriptive/statistical
table but do not silently replace the combined-arm primary comparison.

## Evidence Contract

| Item | Role |
|---|---|
| Exact marginal/prefix/pairing tests | hard validity veto |
| Finite fits, positive temperature, optimizer completion | hard execution veto |
| AUC/ECE/saturation/support | explanatory score diagnostics |
| Joint fixed/audit variance ratios | primary variance-reduction criteria |
| Gaussian exact MSE ratio | accuracy veto and explanatory accuracy evidence |
| SIR fixed/audit variance ratios | SIR variance evidence only |
| Runtime | descriptive resource evidence |

The Gaussian phase is followed by SIR even when the candidate fails, provided
the harness remains valid: a candidate failure does not invalidate a
model-specific variance question. Gaussian failure caused by implementation or
mathematical invalidity is a continuation veto.

## Skeptical Audit And Pre-Mortem

| Risk | Disposition |
|---|---|
| Wrong variance target | Crossed bundle/path design isolates bundle variance; path variance is reported separately |
| CRN changes marginal law | exact marginal fixture and paired empirical checks are hard gates |
| Individual-row shuffling discards pairing | explicit pair-ID validation and whole-pair minibatch shuffling for every arm |
| More paths get a different optimizer schedule | optimizer settings are frozen intentionally; this isolates data count but may under-tune `n8192`, recorded as a limitation |
| Shared validation data induce correlation | intentional paired design; uncertainty resamples outer bundles, not individual rows |
| Variance falls through bias | Gaussian exact MSE guard |
| One lucky fixed path | shared 128-path audit bank plus fixed-path result |
| Nine-cell multiplicity | one predeclared joint statistic; per-cell results descriptive unless separately adjusted |
| Ten bundles underpower variance ratios | intervals required; inconclusive is allowed |
| Long-run disconnect | persistent tmux worker with incremental per-bundle artifacts and terminal status files |
| Memory blow-up at 8192 | one coordinate and bundle at a time; short GPU smoke at max count before campaign |
| Stale V5 controls | frozen causal-ablation controls, not transferred optimal defaults |

Audit verdict before implementation: the design answers the requested
variance-reduction question without conflating it with natural path variation
or score correctness. The six-hour campaign budget is bounded as follows:
up to 45 minutes for tests/smokes, 90 minutes for Gaussian, and 225 minutes for
SIR, including at most one localized infrastructure retry per model in a fresh
artifact directory. No sample-count, threshold, or scientific-method change is
authorized after results are inspected.

The full campaign wall-time estimate remains conditional on the maximum-count
smoke because `n8192` performs four times as many optimizer updates per epoch as
`n2048`. Thus the large-count effect is explicitly “more paths under the same
epoch policy,” including its additional compute; it is not a fixed-update
pure-sample-size estimand. If the smoke projects beyond the six-hour budget,
stop and revise the bundle count or compute budget before launch.

## Execution Amendment: Exact Full-Cell Timing

The first maximum-count SIR capacity smoke completed in 206.75 seconds for
`crn_n8192`, `T50_j1`, and three epochs. That result proves device capacity but
is not sufficient for an honest full-campaign projection because the frozen
full fit uses early stopping with a minimum of 15 epochs and larger held-out
splits. Before changing the scientific scope, run one timing-only
`full_cell` diagnostic for the same arm and cell with the exact full profile.
This diagnostic records split generation, training-bank generation, fitting,
and evaluation wall times. It consumes the existing smoke/diagnostic budget,
does not change any estimator or optimization setting, and cannot itself
support a variance-reduction conclusion.

If this exact full-cell timing still makes the nine-cell campaign exceed the
six-hour budget under an arm- and architecture-aware lower bound, the original
campaign stops on its declared compute-budget continuation veto. Any
representative-cell pilot must then be documented as a new exploratory scope;
it must retain all ten bundles and four paired arms and must not inherit the
nine-cell primary claim.

## Post-Profile Harness Repair And Budget Verdict

The exact full-profile `crn_n8192`, `T50_j1` diagnostic completed in 280.17
seconds and stopped normally after 17 epochs. Its stage timings identified a
harness bottleneck rather than an estimator bottleneck:

- shared validation/calibration/test simulation: 160.38 seconds;
- training simulation: 70.68 seconds;
- classifier fit including calibration: 3.98 seconds; and
- audit/fixed evaluation: 0.05 seconds.

The original loop regenerated identical stateless shared splits for every arm
and regenerated the same large minus bank and nested prefixes. The repaired
runner generates each coordinate's shared splits once, generates one large
minus bank plus the CRN and independent plus banks once, slices exact `n=2048`
prefixes, consumes all four arms, and releases the coordinate tensors before
advancing. A focused equality test verifies that every arm's observations,
labels, pair IDs, and hashes exactly equal the original per-arm generator.
This is a harness-efficiency repair: model, seeds, marginal laws, pair
dependence, optimizer, epoch policy, and estimands are unchanged.

The repaired all-four-arm full-profile `T50_j1` diagnostic completed in 302.80
seconds. All four fits were finite and optimizer-complete; CRN identity,
independent nonidentity, nested-prefix hashes, XLA, memory growth, and the
result checksum passed. The TensorFlow allocator peak was 2.10 GB. Shared plus
training simulation took 239.30 seconds total and the four fits took 9.74
seconds total.

Budget projection uses measured costs rather than the earlier invalid
single-cell multiplication. Three coordinate banks cost about 718 seconds per
SIR bundle. Conservatively allowing 90 seconds for all 36 fits and 55 seconds
for startup/artifact overhead gives 863 seconds per bundle, or 144 minutes for
ten bundles. This is below the frozen 225-minute SIR allocation. The prior V5
Gaussian full run completed its more expensive selection-plus-fit workflow in
82.94 seconds, so ten repaired Gaussian bundles remain below the 90-minute
allocation with substantial margin.

Skeptical re-audit verdict: proceed with Gaussian, inspect its hard validity
gates and measured runtime, then proceed with SIR only if the harness remains
valid and the remaining total budget is adequate. The repair does not change
the baseline, primary statistic, audit paths, stopping rules, or nonclaims.

## Execution And Artifacts

1. Implement paired data generation, one-replicate runner, and TensorFlow
   diagnostic aggregator.
2. CPU mathematical/unit tests and maximum-count GPU smoke.
3. Ten-replicate Gaussian campaign and aggregation.
4. Ten-replicate SIR campaign and aggregation.
5. Result/reset memo with decision and nonclaims.

Artifact root:
`docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/`.
