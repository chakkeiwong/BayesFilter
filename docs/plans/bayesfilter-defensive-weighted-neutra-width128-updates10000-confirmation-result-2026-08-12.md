# Weighted NeuTra width-128 confirmation result (2026-08-12)

## Verdict

The six-stage `(128,128)`, 10,000-update weighted forward-KL transport passes the
predeclared component-weight screen on the analytic `(0.8, 0.2)` two-mode target.
Across eight fresh independent training seeds, the estimated minority-mode mass was

\[
\bar\pi_2 = 0.20000077,
\]

with 95% Student-t interval

\[
[0.19782562,\ 0.20217592],
\]

which contains analytic truth `0.2`.

This means the campaign detects no replicated component-weight bias at its current
resolution and the candidate passes this one target's screening rule. A confidence
interval containing truth does not prove equality, equivalence within a declared
margin, HMC validity, posterior correctness, or transfer to another target.

## Fresh-seed policy

Seed 0 selected the width-128 candidate and was excluded from confirmation. The
confirmatory interval uses fresh replication IDs 1--8 only:

| Host GPU | Fresh replications | Execution |
|---:|---|---|
| 0 | 1, 2, 3, 4 | Sequential within GPU |
| 1 | 5, 6, 7, 8 | Sequential within GPU |

The two lanes ran concurrently. Each lane admitted the next seed only after the
previous seed's result, manifest, trainer state, hashes, seed/capacity identity,
XLA receipt, memory-growth receipt, finite checks, and component coverage validated.

Total campaign wall time was `2,944.00 s` (`49.07 minutes`). Mean per-seed runtime
was `732.94 s` on GPU 0 and `647.84 s` on GPU 1.

## Component-weight evidence

| Replication | GPU | Minority mass | Selected update |
|---:|---:|---:|---:|
| 1 | 0 | 0.19717 | 7,000 |
| 2 | 0 | 0.20230 | 8,750 |
| 3 | 0 | 0.19885 | 9,750 |
| 4 | 0 | 0.20510 | 7,250 |
| 5 | 1 | 0.20002 | 9,250 |
| 6 | 1 | 0.19899 | 6,250 |
| 7 | 1 | 0.19760 | 8,000 |
| 8 | 1 | 0.19999 | 8,250 |

Summary:

| Quantity | Value |
|---|---:|
| Mean | 0.20000077 |
| Standard deviation | 0.00260179 |
| Standard error | 0.00091987 |
| 95% Student-t interval | [0.19782562, 0.20217592] |
| Analytic truth | 0.20000000 |

All eight weighted transports were finite and represented both target components.
Importance ESS fractions were about `0.734`--`0.738`, matching the analytic
defensive-proposal calculation and ruling out weight collapse as an explanation.

## Optimization evidence

Selected checkpoints ranged from update `6,250` to `9,750`; no confirmation run
selected terminal update `10,000`. This supports the seed-0 finding that validation
loss reaches a practical noisy plateau under this fixed protocol rather than simply
being truncated while monotonically improving.

This is practical convergence evidence, not a mathematical proof of a stationary
point. Gradient clipping varied from `0.25%` to `23.46%` across seeds, so optimizer
dynamics remain an explanatory uncertainty.

Weighted audit NLL had mean `3.95539` and standard deviation `0.00793`. These values
are close to the separately estimated target self-NLL `3.94921`, but the quantities
were not evaluated as a paired common-random-number difference. Small negative
per-seed plug-in differences can therefore occur from Monte Carlo error and must not
be interpreted as negative KL. NLL agreement is descriptive only in this result.

## Reverse-KL comparator

The matched reverse-KL arm was unstable across fresh seeds: four of eight
base-pushforward audits omitted the minority mode, while the others assigned about
`0.207`--`0.209`. Its mean audit NLL was roughly `12.5`, far above the weighted
arm's `3.955`. These are hard candidate failures for the affected reverse-KL seeds
and descriptively unfavorable density results. No paired method-ranking test was
predeclared, so no statistical superiority claim is made.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Width-128 target screen | Pass: truth inside fresh-seed interval | Pass for all weighted runs | One target geometry | Run remaining r1 analytic variants | No equality proof |
| Practical optimization convergence | Supported by nonterminal selected checkpoints | No finite veto | Variable clipping and fixed LR | Freeze protocol for remaining r1 tests | No stationary-point proof |
| Reverse-KL comparator | Four seeds miss minority mode | Coverage veto in four seeds | Other seeds assign near-correct mass but wrong density | Preserve as comparator | No universal ranking |
| Later rungs | Not yet eligible | Earlier r1 variants untested | Cross-target transfer | Do not start paper/SSL-LSTM yet | No HMC/posterior claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Weighted arm passes all finite, coverage, identity, and artifact gates |
| Statistically supported ranking | None; not tested |
| Statistically supported target screen | Truth lies inside predeclared eight-run interval |
| Descriptive differences | Weighted NLL and coverage are much healthier than matched reverse KL |
| Default readiness | Not assessed and ineligible |
| Next evidence needed | Equal-weight, unequal-covariance, rare-mode, and four-mode analytic targets |

## Post-run red team

The strongest alternative explanation is that the candidate succeeds because the
defensive proposal contains the correct known component locations. This target does
not test discovery of unknown modes. The interval also tests failure to reject a
point value, not formal equivalence inside a predeclared practical margin. A failed
remaining r1 geometry, missing component under soft/unknown assignments, or a fresh
interval excluding truth would overturn broader claims.

## Manifest and artifacts

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` plus recorded dirty state |
| Environment | `tfgpu`, TensorFlow 2.20.0, TFP 0.25.0 |
| Hardware | Two RTX 4080 SUPER GPUs, one process per GPU lane |
| Numerical mode | float64, XLA, TF32 disabled |
| Memory policy | TensorFlow memory growth set and verified before initialization |
| Batch/update budget | 4,096 fresh rows x 10,000 updates per arm per seed |
| Campaign wall time | 2,943.9986 seconds |
| Campaign manifest SHA-256 | `258e749c599d9c1ed19b5ee1adb36615579c2b84d6107fdc395bbf7309560232` |
| Summary result SHA-256 | `75b0c755f7c956dfc350ebfd0c34d91c2b4bd22a85c42359ed7cdcbdb2608c68` |

Artifacts:

- Campaign manifest:
  `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/width128-updates10000-confirmation-campaign-v1/campaign_manifest.json`
- Confirmatory summary:
  `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/width128-updates10000-confirmation-summary-v1/result.json`
- Plan:
  `docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md`

