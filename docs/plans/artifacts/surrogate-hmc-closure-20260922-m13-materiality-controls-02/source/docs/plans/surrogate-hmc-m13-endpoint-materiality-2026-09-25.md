# M13 same-data endpoint materiality — 2026-09-25

## Question and evidence contract

The campaign tests canonical analytical DCTR proposals in surrogate HMC with a
separate endpoint posterior. This stage asks whether finite importance noise
or FP32 endpoint rounding could materially change acceptance on the actual
seed-1704 mechanics trajectories. It uses the same observations and full
wrapped observation law, not the older seed-1701 posterior grid.

Baseline: the 4096-particle, 8-replicate fixed-seed FP64 endpoint of compiled
mechanics04, cast to FP32 for acceptance. Comparator: the same full-support
importance algorithm with three independent seed groups of 8 replicates at
4096 and 16384 particles. Prior, observations and law are unchanged. This is a
numerical precision ladder, not a comparison of posterior samplers.

Primary local screen: on each of the 24 actual proposal pairs, a change in
acceptance probability of at most .01 (one percentage point), including an
approximate +/-3 paired-MCSE envelope. This prospective convenience margin
only nominates further posterior testing. Its statistical coverage is not
certified and it cannot qualify the full endpoint or unbounded posterior.
All finite/valid reference outputs, checksummed data match, complete replicate
traces and preserved source identities are mandatory validity conditions.
A failed local screen triggers reference repair; it is not a continuation veto.
Nonfinite estimates, corrupted inputs or a data mismatch stop this stage.

Every transition in mechanics04 was accepted, so the saved next position is
its proposal endpoint. The loader checks this before reconstructing pairs;
a rejected transition would require the unavailable proposed position and
must fail closed. All 25 positions (initial plus 24 states) are used, including
warmup. No state selection uses acceptance, likelihood error or reference ESS.
Nominal and stress situations use the existing 3x3 grids at one and three
prior standard deviations. These off-trajectory checks are explanatory only.
No held-out posterior claim data (seed3701) are consumed.

## Uncertainty calculation and assumptions

For independent positive likelihood estimates L_r, let m=mean(L_r) and
z_r=L_r/m-1. The first-order delta-method MCSE of log(m) is
sqrt(sum(z_r^2)/(R*(R-1))). For a same-seed pair, replace z_r by z_new,r-z_old,r.
This retains common-random-number covariance and removes constant likelihood
offsets from the acceptance question. The approximation can be unreliable
with low importance ESS or dominant weights; record both and do not issue
posterior-reference admission. Each point uses the same replicate seed group,
so correlations across positions are intentional. The pointwise envelopes are
not a simultaneous confidence statement. A particle-count change is independent
unless an actual coupling is established; report it without claiming pairing.

Acceptance is alpha(a)=exp(min(0,a)). It is monotone and 1-Lipschitz, so an
absolute error in log acceptance bounds the absolute change in acceptance
probability. Compute actual probability changes and envelopes from fixed
kinetic/prior contributions. Report FP32 casting effects separately from Monte
Carlo effects. FP32 remains default; FP64 is an independent reference role.

Defaults: particles4096 and replicate8 reproduce the saved endpoint; 16384 is
a fourfold precision probe, not a claimed convergence limit. Three fresh groups
supply24 independent replicate estimates. Seeds2026092501..2503, second9101.
The .01 screen is a declared engineering materiality hypothesis, not a
scientifically calibrated posterior margin. Cheap adversaries are original
fixed-seed replay, three same-budget independent groups, and fourfold particles;
compare each by trajectory, nominal and stress situations. No sampler
superiority or heuristic-dominance promotion is sought.

## Skeptical audit and execution

The prior and data are identical, the scalar estimator is compared before
casting, and rejected proposals cannot be reconstructed silently. Independent
seeds and more particles may all miss important tail contributions; the finite
log-likelihood estimator also has Monte Carlo bias. Passing cannot prove
posterior accuracy. The saved grid with data1701 is explicitly excluded.
Use TF/TFP GPU/XLA and memory growth; no NumPy/runtime fallback. Preserve every
reference evaluation before summarizing. Full likelihood code remains unchanged.

Master stage `m13-endpoint-materiality`, fresh output
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-materiality-01`.
Internal pass cap1600CPU/1600GPU seconds, worker900CPU/1000GPU, review120CPU.
Maximum three attempts, each charged, within unchanged 54000CPU/226800GPU totals.
Stop on validity failure or total/pass/worker limits. Localized implementation
failures may be repaired under remaining budget with fresh output paths.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-endpoint-materiality --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-materiality-01
```

The result must give local materiality, uncertainty, all nominal/stress values,
source/data hashes, seed groups, memory/device provenance and explicit false
endpoint/posterior qualification flags. Within the existing master the next
step is control/reference repair or a same-data posterior precision design,
according to these diagnostics. No new HMC tuning authority is issued here.

Pre-run regression: 29 focused CPU-only tests passed (GPU deliberately hidden),
including common multiplicative likelihood-error cancellation, paired MCSE,
accepted-proposal reconstruction and master registration. The numerical
reference implementation is unchanged.
