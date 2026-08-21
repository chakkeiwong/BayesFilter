# Weighted forward-KL NeuTra three-mode result (2026-08-12)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md`

## Verdict

The reviewed weighted forward-KL protocol produced one viable corrected-HMC
candidate on the new exact four-dimensional three-component Gaussian mixture.
The successful candidate is a frozen six-stage dense IAF with `(128,128)`
hidden layers, selected at update 8,750 from a 10,000-update GPU/XLA run.
Canonical sequential HMC passed after 2,000 archived warm-up transitions and
1,000 retained transitions per chain.

This is a target-specific viability result. It does not rank weighted forward
KL against reverse KL, prove posterior equality, establish mode discovery, or
promote a default.

## Evidence

| Item | Result |
|---|---:|
| Target | Exact `separated_three_mode_unequal_weight_d4_v1`, weights `(0.5, 0.3, 0.2)` |
| Serious transport | Dense IAF, six stages, `(128,128)`, float64, GPU/XLA |
| Selection rule | Minimum disjoint weighted heldout NLL; selected update `8,750` |
| Selected weighted NLL | `4.40726` |
| Transport heldout maximum component-mass error | `0.00278` |
| Importance ESS fraction | `0.69319` |
| Frozen checkpoint SHA-256 | `b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6` |
| Selected fixed HMC kernel | `L=5`, epsilon `0.3433257029` |
| Sequential warm-up / retained per chain | `2,000 / 1,000` |
| Retained max R-hat | `1.00395` |
| Retained min bulk / tail ESS | `1,059.05 / 1,381.77` |
| Hard numerical/status/movement vetoes | None |
| Archive receipts | Six chunks verified: four warm-up and two retained |

The final component-mass estimates and separate two-sided 99% batch-means
intervals were:

| Component | Truth | Estimate | 99% interval |
|---:|---:|---:|---:|
| 0 | `0.50` | `0.51625` | `[0.47271, 0.55979]` |
| 1 | `0.30` | `0.29500` | `[0.25867, 0.33133]` |
| 2 | `0.20` | `0.18875` | `[0.15548, 0.22202]` |

All intervals contain their respective exact masses. Every retained chain
visited all three hard-assignment components. Aggregate directed hard-assignment
transitions were

```text
[[1684, 258, 122],
 [ 257, 828,  94],
 [ 122,  93, 538]].
```

Thus every component was involved in a transition. These transitions are
explanatory evidence against initialized-component trapping; they are not a
convergence proof. All four marginal mean and all sixteen marginal covariance
intervals contained the exact value. Those moment intervals are reported as
explanatory diagnostics only and were not combined into an uncalibrated joint
test.

## Capacity repair

The first frozen `(64,64)`, three-stage, 1,000-update weighted transport was
not admitted to sequential sampling. Its complete `L=(3,5,10,15,20,25)` grid
had finite values/scores and acceptance between `0.682` and `0.839`, but every
2,000-draw latent verification failed the declared modern R-hat gate:
maximum values ranged from `3.47` to `4.90`. This is a candidate failure under
the declared downstream criterion, not a numerical invalidity and not evidence
against weighted forward KL in general.

The predeclared `(128,128)`, six-stage, 10,000-update capacity repair changed
that outcome. Four of six tuning trajectories passed their fresh 2,000-draw
R-hat verification: `L=3`, `5`, `10`, and `20`; `L=15` and `25` were correctly
rejected. The deterministic tuner selected the viable `L=5` arm. No threshold,
posterior-reference rule, or sampler policy was relaxed.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Three-mode weighted candidate | Passed sequential R-hat/ESS and separate exact component-mass screens | No finite, status, movement, archive, or component-presence veto | One frozen training seed and component-aware proposal/starts | Port the target-specific weighted replay lane to the varying-Hessian smooth targets | No general method, discovery, equality, ranking, or default claim |
| Small capacity candidate | Rejected at tuning before sequential sampling | No numerical veto; all kernels failed verification R-hat | Whether smaller architecture could work with a different target-specific protocol | Preserve as a capacity repair trigger and historical negative evidence | No conclusion about the research direction |
| Reverse-KL arm | Its heldout transport audit is descriptively unstable in this run | Not subjected to matched HMC confirmation | One initialization/seed and objective-specific row law | Run matched target-specific comparator only under a later reviewed fixed budget | No method ranking |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the selected six-stage transport and sequential run |
| Viable candidates | The frozen six-stage weighted transport and selected `L=5` kernel |
| Statistically supported ranking | None |
| Descriptive-only differences | Transport NLL, clipping, acceptance, energy tails, and reverse-KL audit behavior |
| Default readiness | Not assessed |
| Next evidence needed | Target-specific varying-Hessian replay proposal and matched downstream HMC validation |

## Provenance

- Serious training root:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/three-mode/component-aware-width128-depth6-updates10000-r1/`.
- HMC root:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/three-mode/hmc-width128-depth6-r1/`.
- TensorFlow `2.20.0`, TensorFlow Probability `0.25.0`, `tfgpu`, float64,
  XLA enabled, TF32 disabled, one GPU 0 process with verified memory growth.
- Sequential archive manifest SHA-256:
  `78f2af3bca39c6eb8ba48baf3a50178181706d4222d25ca3a3d64ca37ffdacc8`.
- Final result SHA-256:
  `77579b0f450086c4b1ee7e405d469fde81a858f07443f0821859c7aa25ea1a5e`.

## Post-run red team

The strongest alternative explanation is that the component-aware proposal and
component-aware starts make this analytic representation task easier than a
mode-discovery problem. The completed result does not test discovery; the
planned mode-blind proposal is a separate future difficulty rung. A fresh
transport training seed that fails the same frozen target/HMC screens, or a
target-specific varying-Hessian failure after valid proposal preflight, would
limit the scope of this evidence.
