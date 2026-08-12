# Weighted NeuTra width-128, 10,000-update canary result (2026-08-12)

## Verdict

The owner-requested six-stage `(128,128)` weighted-NeuTra canary completed
successfully. Its audit density is descriptively much closer to analytic truth than
the six-stage `(64,64)`, 2,000-update seed-0 candidate, while its seed-0 minority
mass is slightly less accurate (`0.19719` versus `0.19926`). It provides the first
credible evidence of practical within-run
optimization saturation in this lane: the best held-out checkpoint was update
`8,500`, and validation NLL fluctuated around a plateau after approximately update
`4,000` rather than improving monotonically through update `10,000`.

This is one seed. It nominates the arm for independent replication; it does not
establish r1 promotion, convergence across training seeds, statistical superiority,
HMC validity, or transfer to unknown posterior modes.

## Exact run

| Field | Value |
|---|---|
| Target | Analytic two-mode Gaussian mixture, weights `(0.8, 0.2)` |
| Defensive proposal | Same components, weights `(0.5, 0.5)` |
| Transport | Six IAF stages, two hidden layers `(128,128)` per stage |
| Objective | Self-normalized importance-weighted forward KL |
| Comparator | Matched reverse KL at identical capacity and update budget |
| Updates | 10,000 per arm |
| Batch | 4,096 fresh rows per update |
| Rows per arm | 40,960,000 |
| Device | Host GPU 1, RTX 4080 SUPER |
| Numerical path | TensorFlow float64, XLA, TF32 disabled |
| Memory policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified before initialization |
| Wall time | 652.72 seconds (10.88 minutes) |
| Allocator peak | 286,612,736 bytes |

## Optimization trajectory

Held-out weighted NLL checkpoint summaries:

| Update window | Mean NLL | Minimum | Maximum |
|---:|---:|---:|---:|
| 1--2,000 | 4.12699 | 3.98081 | 4.47032 |
| 2,001--4,000 | 3.98162 | 3.96242 | 4.00229 |
| 4,001--6,000 | 3.96394 | 3.95886 | 3.96912 |
| 6,001--8,000 | 3.96504 | 3.95861 | 3.97379 |
| 8,001--10,000 | 3.96214 | 3.95533 | 3.96815 |

The selected checkpoint was update `8,500`, with held-out NLL `3.95533`. Terminal
NLL at update `10,000` was `3.95987`. Therefore the run is not merely truncated at
an improving terminal checkpoint, unlike most width-64 2,000-update runs.

This establishes practical plateau evidence under the fixed learning rate. It does
not prove a stationary point mathematically: stochastic gradients, 606 clipped
updates (`6.06%`), and checkpoint noise remain.

## Distribution diagnostics

| Diagnostic | Weighted width-128 candidate | Exact target/reference |
|---|---:|---:|
| Minority-mode probability | 0.19719 | 0.20000 |
| Absolute minority-mass error | 0.00281 | 0 |
| Audit weighted NLL | 3.95192 | Target self-NLL 3.94921 |
| Descriptive forward-KL gap | 0.00271 nats | 0 |
| Base-pushforward mean error | 0.02754 | 0 |
| Base-pushforward relative covariance error | 0.00860 | 0 |
| Latent weighted mean norm | 0.03549 | 0 |
| Latent covariance Frobenius error | 0.05484 | 0 |

Both target components were represented. Importance ESS fraction was `0.73574`,
consistent with the analytic defensive-mixture design, so weight degeneracy is not
a failure explanation.

The matched reverse-KL arm selected update `250`, omitted the minority component in
the base-pushforward audit, and had audit NLL `12.3818`. This is a hard candidate
failure for reverse KL on this run, but one seed does not support a general ranking.

## Decision table

| Decision | Primary status | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Width-128 canary | Pass exploratory mass screen; nominated | No weighted-arm hard veto | One seed and fixed learning rate | Independent replicated interval | No r1/HMC/default claim |
| Optimization plateau | Credible practical plateau after 4,000 | 6.06% clipping is explanatory | No gradient/learning-rate-decay proof | Replicate same frozen protocol first | No mathematical convergence proof |
| Reverse-KL comparator | Missing minority mode; reject this candidate | Coverage veto fired | One seed | Preserve as comparator evidence | No universal method ranking |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Weighted arm passed; reverse-KL arm failed mode coverage |
| Statistically supported ranking | None |
| Descriptive differences | Weighted candidate is close to analytic mass and NLL; reverse KL collapsed |
| Default readiness | Not assessed and ineligible |
| Next evidence needed | Eight independent width-128 runs and component-weight interval |

## Post-run red team

The strongest alternative explanation is favorable seed-0 initialization: the
width-64 seed-0 canary also looked better than its eight-run mean. That pattern is
exactly why this result cannot promote the arm. The result would be overturned by
an eight-run interval excluding `0.2`, missing components in any terminal audit, or
failure to reproduce the validation plateau.

## Artifacts

- Run root:
  `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width128-updates10000-canary-v1/`
- Result SHA-256:
  `28eea6ac1853b8532d3a93027ab1b9e201f9e0aec2b19ff209a385a113a6c088`
- Plan:
  `docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md`
