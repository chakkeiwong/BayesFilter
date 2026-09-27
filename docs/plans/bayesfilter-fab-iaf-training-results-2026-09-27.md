# FAB+IAF training: coverage without adequate whitening

The bounded campaign produced valid FAB-trained canonical IAF maps, but **did
not produce the requested well-whitened map with verified coverage of all
modes**. On the known three-mode control, FAB sampled all three mode regions
while reverse KL collapsed to one; FAB's whitening remained poor. On q20,
both completed pairs failed the plan's joint geometry screen. Both sign regions
were sampled, but independent evidence is too weak to establish their posterior
masses or enumerate all modes.

Two of the three planned q20 pairs reached 240 updates in both arms. The third
has reverse KL at 240 and FAB at 96; it is not a matched comparison. All saved
maps have complete finite 1,000-point probes and 4,096-point coverage checks.
The campaign stopped because its remaining compute could not fund another
complete continuation and diagnostic package. The full three-pair ladder and
a target-specific converged training protocol therefore remain unfinished.

## Scientific question and measured quantities

The target is the existing four-parameter q20/T30 UKF approximate posterior,
not an exact state-space posterior. Its beta-1 adapter signature is
`2dd2acd05652b1da245cadb6b61f92d65ed3529175b44a8d33790bdf51b704f2`.
Both trainers use the canonical three-stage IAF, two width-16 ELU hidden layers
per stage, author masks/initialization, coordinate reversal, and conditional
scale `b + 2*tanh(h/2)`. Within each seed, initial map parameters and target
signatures match exactly. The map is FP32, the target and final represented-map
checks FP64, TF32 is disabled, and the numerical kernels use XLA on GPUs with
verified memory growth. Training uses native batches of 32; final coverage
evaluates native batches, with no scalar target fallback.

For independent `z ~ N(0,I)`, the probe computes

\[
e(z)=\nabla_z\{\log p(T(z))+\log|\det DT(z)|\}+z.
\]

The table's score RMS is `sqrt(mean(e_ij^2))`, **per coordinate**, over 1,000
rows; it is not the mean Euclidean norm. Density RMS is the standard deviation
of `log p(T(z))+log|det DT(z)|+||z||^2/2`. Both are zero for an exact Gaussian
pullback. These measurements are taken under the map's own base draws: they
can look small when the map concentrates within only one mode.

Map importance weights are `p(T(z))/q(T(z))`. The independent bank is sampled
from the declared prior/diagonal Gaussian `g`, so its weights are **p/g**.
Values of `log p-log q` on that bank are support-mismatch diagnostics, not
importance weights. For q20 their absolute means include the unknown
normalizing constant and are not normalized KL divergences.

## Results

| Analytic three-mode control, seed 0 | FAB, 240 updates | Reverse KL, 240 updates |
|---|---:|---:|
| Score residual RMS per coordinate | 27.568 | 0.295 |
| Centered log-density RMS | 22.193 | 0.413 |
| Raw target-component responsibility mass | 0.567 / 0.273 / 0.160 | approximately 1 / 0 / 0 |
| Raw nearest-mean occupancy | 0.510 / 0.297 / 0.193 | 1 / 0 / 0 |
| Exact-target forward-KL Monte Carlo estimate | 1.837 | 27.513 |
| Map importance ESS out of 4,096 | 414.8 | 3,005.3 |

Exact component probabilities are 0.5/0.3/0.2. Component responsibility means
have these expectations under the target; nearest-mean cell probabilities are
different objects. Independent exact target draws had nearest-mean occupancy
0.498/0.297/0.205. Both diagnostics establish the observed reverse-KL collapse
and that FAB visits all three known regions. They do not show that the FAB map
matches each mode's shape. The initial untrained IAF score/density RMS was
17.524/17.744: FAB's final geometry is descriptively worse than that simple
baseline, even though its target-to-map density fit is descriptively better.
This is a failed joint fit, not a promoted success.

The high map ESS of the collapsed reverse-KL control is especially instructive:
weights can be stable within one mode while the proposal misses the others.
Neither map ESS nor whitening under proposal draws certifies coverage.

| q20 seed / arm | Updates | Score RMS | Density RMS | Map ESS / 4,096 | Raw positive-sign fraction | Weighted positive-sign fraction |
|---|---:|---:|---:|---:|---:|---:|
| 1 / FAB | 240 | 196.515 | 230.180 | 109.33 | 0.466 | 0.484 |
| 1 / reverse KL | 240 | 7.013 | 6.481 | 56.78 | 0.636 | 0.500 |
| 2 / FAB | 240 | 147.210 | 167.697 | 131.90 | 0.536 | 0.521 |
| 2 / reverse KL | 240 | 6.386 | 6.035 | 11.51 | 0.531 | 0.642 |
| 0 / FAB, incomplete pair | 96 | 213.779 | 193.609 | 102.41 | 0.490 | 0.543 |
| 0 / reverse KL, incomplete pair | 240 | 7.553 | 6.551 | 73.94 | 0.553 | 0.602 |

FAB fails the predeclared descriptive geometry nomination screen in both
completed pairs. Reverse KL also remains far from a Gaussian pullback; lower
residuals alone do not qualify it. The independent prior-bank ESS is only
21.45, 21.52 and 20.05 for seeds 0/1/2, with maximum weights 0.095/0.089/0.108.
Its positive-region estimates vary from 0.366 to 0.486. These are concentrated
importance estimates without a validated uncertainty calculation. They cannot
certify mode masses, and a positive/negative partition does not discover all
modes in a four-dimensional posterior.

The matched q20 initial score/density RMS values were 978.659/462.350 (seed 1)
and 1194.964/416.846 (seed 2). Both trainers reduce these initial diagnostic
values, but that is descriptive learning evidence, not a convergence test.

## Diagnosis and next discriminating work

No recorded q20 replay update activated the correction cap, global gradient
clipping was disabled, and every final scale-saturation fraction was zero at
the existing slope-below-0.1 alert. These observations do not support clipping
or conditional-scale saturation as the explanation for this campaign's poor
whitening.

AIS weight concentration remains severe: the median terminal ESS over the
last ten passes of each completed FAB arm is about **1.38 and 1.34 out of 32**.
Thus a batch often has roughly one effective weighted trajectory. This is a
repair signal for proposal exploration/annealing and training, not proof that
the TensorFlow port, canonical architecture, or FAB objective is wrong.
The tests and artifact checks support engineering correctness in their checked
domains, while the trained candidate fails the desired fit.

The strongest alternative explanation is undertraining or unsuitable
exploration/hyperparameters. This ladder used 240 updates at learning rate
0.001 with inherited Metropolis/AIS settings; the earlier q20 reverse-KL recipe
used path gradients and a much longer, differently calibrated schedule. It was
not reproduced here. FAB uses Optax-form Adam and reverse KL uses Keras Adam,
whose epsilon placement differs. Consequently this is a comparison of two
configured training procedures, not perfect causal isolation of the objective
or a comparison against the best tuned reverse-KL method.

Finishing seed 0 is feasible by exact resume, but its remaining 144 updates
cost about 1,022 worker seconds at its measured rate, plus approximately
450 seconds of cached final diagnostics and recompilation. A practical
allocation is about **1,625 worker seconds (27 minutes)**, against only 339
seconds left here. This would complete the replication table; it would not
by itself repair whitening. The more informative next training study should
calibrate AIS exploration/weight concentration and continuation length, retain
the exact multimodal control, and test whether coverage can survive subsequent
geometry refinement. Any FAB-to-reverse-KL refinement is a proposed new arm,
not an implemented or validated result of this campaign.

Before a mode-coverage conclusion, q20 also needs stronger independent target
coverage evidence than this concentrated prior bank: a reviewed mode-discovery
and mass-estimation procedure, or another adequately effective independent
reference. Increasing the number of proposal-only probe points cannot repair
that missing evidence.

## Decision and inference status

| Decision | Primary criterion | Veto/repair status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain implementation and trained maps as diagnostic results | Completed artifacts and 11 audited scientific/prefix runs pass checks | No nonfinite target/update/probe, hash, pairing or canonical-width failure | Untested inputs and stochastic exploration | Preserve exact checkpoints and source records | Universal correctness or production readiness |
| Reject this bounded FAB recipe for the requested joint fit | Both completed q20 pairs fail the geometry screen; analytic FAB is poorly whitened | Candidate promotion fails; no research-direction veto | Short training, sparse effective AIS trajectories, uncalibrated settings | Calibrate exploration and continued training before another quality claim | FAB/IAF impossibility or statistical superiority of reverse KL |
| Do not certify all q20 modes | Independent reference ESS about 20--22; only sign partition checked | Coverage unresolved | Unknown modes and unknown region-mass error | Obtain stronger independent coverage evidence | Balanced signs imply complete coverage |
| Stop within the compute allocation | 15,861.008 of 16,200 worker seconds charged | Only 338.992 remain; insufficient for a complete continuation | Third pair unfinished | Preserve seed-0 FAB checkpoint at 96 updates | Complete three-seed execution or converged training |

| Inference status | Finding |
|---|---|
| Hard veto screen | All audited numerical/artifact checks pass. Analytic reverse KL demonstrably misses two known regions. q20 coverage certification is unsupported. |
| Statistically supported ranking | None: two completed q20 pairs and one analytic seed, no predeclared uncertainty-based ranking. |
| Descriptive-only differences | Geometry values, importance ESS, component/sign masses, density mismatches and timing. |
| Default readiness | Not established; no HMC/posterior run and no adequately whitened, coverage-qualified map. |
| Next evidence needed | Complete replication if desired, calibrated exploration and training continuation, and effective independent q20 coverage checks. |

Post-run skeptical review: the low reverse-KL residual is invalid as evidence
of global coverage (the analytic counterexample is explicit). FAB's balanced
signs and higher q20 importance ESS are invalid as evidence of Gaussianization.
Unknown normalization is not mislabeled as a KL value. The unfinished seed is
excluded from paired comparison. No remaining algorithmic failure is hidden
behind the scheduling failures, and no scheduling failure is treated as
evidence against FAB. A longer calibrated fit that jointly improves geometry
and independently checked coverage would overturn the candidate-level verdict.
Weak independent q20 coverage is the least conclusive part of the evidence.

## Repairs, budget and reproducibility

The first preflight failed after 15.960 seconds on a NumPy scalar at the JSON
boundary. Repairs also corrected an accidental width-4 configuration and the
wrong importance-weight denominator for prior-bank draws before scientific
execution. Subsequent scheduling estimates were too conservative, causing
valid prefixes to stop early. They were resumed exactly; duplicate fixed
target-bank evaluations were cached after a cached/uncached parity check.
Every attempt has a fresh directory and all earlier evidence is preserved.

Validation: **16 focused FAB tests, 3 campaign regression tests, two complete
GPU preflights and one complete exact-resume GPU smoke passed**. The terminal
stdlib audit verifies checkpoint/frozen hashes, their binding and parameter
equality, all declared file hashes, actual source hashes against the recorded
Git commit, initial-state/target equality within pairs, continuation bindings,
update counts, finite complete probes, memory growth, precision and XLA. All
checks pass across the 11 analytic/q20 result directories.

Charged q20 plus pricing time is **15,861.008 worker seconds (4.406 hours)**;
smokes/control add **161.995 seconds** of their separate 600-second allowance.
Waiting for tool permissions and supervisor idle time are not GPU worker time.
The process accounting conservatively includes worker launch/exit polling.

The reviewed plan is `bayesfilter-fab-iaf-training-campaign-2026-09-26.md`.
Evidence root: `docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/`.
`summary.json` contains machine-readable tables, checks and accounting. The reproducible report
command is:

```sh
python docs/benchmarks/summarize_fab_iaf_campaign_2026_09_26.py
```

Per-arm manifests record exact commands, Git commits, source hashes, seeds,
target/data signature, interpreter, TensorFlow version, devices, memory policy,
precision, XLA, paths and times. `checkpoint.json` contains optimizer/replay
state; `frozen-map.json` is the checked represented map; `progress.json`,
`initial-post-training-1000.json`, `post-training-1000.json` and `coverage.json`
preserve the diagnosis. Both queue files preserve launches, costs and the
unfinished arm. The complete evidence is also preserved in
`campaign-evidence.tar.gz` with a SHA-256 inventory.
