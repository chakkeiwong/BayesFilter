# q20 1,000-point latent Gaussian score-residual diagnostic

## Question and evidence contract

The question is whether the frozen q20 NeuTra coordinates are locally close to
the standard-normal target assumed by the Gaussian step-size calculation.  For
latent coordinate `z`, evaluate the exact transformed target supplied to HMC
and compute

`g(z) = grad_z log pi_z(z) + z`.

An exact `N(0, I4)` target has `g(z) = 0` at every point.  The tested object is
the current frozen map `direct-w16-lr0.0005-r0-beta1-u512`, q20/T30 at beta=1,
with identity mass in latent coordinates.  The points are 1,000 fresh,
deterministic-stateless iid draws from `N(0, I4)`.  They are diagnostic points,
not posterior draws.

The primary diagnostic is the empirical distribution of `||g(z)||`, including
the minimum, mean, standard deviation, median, 1%, 5%, 25%, 75%, 95%, 99%, and
maximum quantiles, plus fractions exceeding 0.1, 0.5, 1, 2, 5, and 10.  The
companion diagnostic is

`r(z) = log pi_z(z) + ||z||^2/2`,

whose variation would be zero (up to an unknown additive constant) for an
exact standard normal.  Pointwise target/status finiteness is an engineering
validity screen.  No arbitrary residual threshold is promoted to a Gaussianity
acceptance criterion by this run.

This diagnostic can establish pointwise evidence of mismatch or support for
further investigation.  It cannot establish global posterior Gaussianity,
posterior covariance, HMC acceptance, convergence, map quality, or a
production default.  It also cannot distinguish map capacity, coverage,
nonlinear curvature, tail behavior, and score implementation error by itself.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early check | Status |
|---|---|---|---|---|---|
| 1,000 points | User-requested diagnostic size | Gives a descriptive residual distribution rather than four point checks | Still does not cover all posterior regions | Record all points and their norm distribution | reviewed diagnostic scope |
| `N(0,I4)` point law | Derived from the Gaussian whitening hypothesis under test | Tests the region where the Gaussian step-size formula is intended to apply | Tail or proposal-law points are not posterior samples | Report latent norm quantiles and retain point identities | hypothesis, not a default |
| Frozen map and beta=1 target | Existing canary scope and saved export | Keeps the question tied to the failed Gaussian-root canary | A different map/target has a different tuning scope | Verify source, target, map, and transport hashes | fixed scope |
| Stateless seed | Repository `scoped_seed` for this stage | Reproducible without using training or HMC random streams | One seed is not a replication study | Record seed and all pointwise values | reviewed diagnostic convention |
| One batched TensorFlow/XLA target evaluation | Repository batch-native target contract | Avoids scalar Python evaluation and matches the HMC target path | A target/transport implementation defect would affect all values | Require finite values, scores, and status telemetry | engineering diagnostic |

## Skeptical pre-run audit

The baseline is the exact transformed q20 value/score target, not a proxy loss
or the training bank.  The residual is explanatory evidence only and is not a
promotion gate.  Nonfinite target/score/status values are a hard diagnostic
validity veto; finite but large residuals are evidence against the Gaussian
hypothesis, not evidence against NeuTra as a method.  The map export and source
snapshot are checked before evaluation, and the output is written under a new
versioned directory.  The cap is bounded by the remaining diagnostic ledger;
the run stops on cap, source drift, missing GPU/memory-growth policy, or
invalid target telemetry.  The artifact preserves the exact command,
environment, seed, hashes, device, and all 1,000 rows.

The main pre-mortem is that iid Gaussian points could overrepresent regions the
posterior does not visit, or that a score implementation defect could mimic
non-Gaussianity.  Norm summaries expose the first issue; the existing map
forward/inverse/logdet/pullback parity evidence is recorded as provenance but
does not independently prove the posterior score.  A later posterior/reference
point diagnostic would be required to separate those explanations.

## Budget and execution

Use the existing q20 campaign supervisor with a 900-second diagnostic cap,
leaving the prior campaign and training settings unchanged.  The worker uses
the preserved r2 source snapshot, GPU memory growth before TensorFlow
initialization, float64, and the batch-native transformed target.  To keep the
memory footprint bounded, the 1,000 rows are evaluated as fixed-shape batches
of 20 through the same compiled target signature; this is a diagnostic schedule,
not a scalar sample loop in the target implementation.  No training,
adaptation, HMC transition, tuning artifact, or production default is written.

Expected artifacts:

- `docs/plans/artifacts/q20-1000-point-score-residual-2026-09-22/launch.json`
- a campaign attempt manifest and result under campaign-05;
- `points.json` with all latent points, values, scores, statuses, residuals,
  and `r(z)` values;
- `result.json` with the summary and remaining balances; and
- this plan plus a result note after execution.

## Exact command

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python /home/ubuntu/python/BayesFilter/docs/plans/artifacts/q20-1000-point-score-residual-2026-09-22/run_diagnostic.py
```
