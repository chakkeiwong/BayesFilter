# SSL-LSTM q=20 seed-B predictive equivalence plan (2026-08-08)

## Objective

Run the formal fixed-parameter output-law comparison that was missing from the
previous plug-in diagnostic. Compare the ten-step q=20 predictive law at the
posterior-mean plug-in parameter with the law at the synthetic generating
parameter. Do not compare parameter coordinates as a validity criterion and do
not propagate the 4,000 retained HMC draws as a parameter mixture.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Is the posterior-mean plug-in output law predictively equivalent to the true-control output law under the locked q=20 SVD-UKF forecast operator? |
| Candidate | One posterior-mean physical parameter vector computed from the authenticated 4-chain x 1,000-draw seed-B retained archive. |
| Comparator | One fixed q=20 `PRIOR_CENTER` physical vector, used only as a synthetic output-law control. |
| Predictive object | Independent ten-step output paths, including terminal-state, process, and observation noise, with one complete path as one observation. |
| Primary criteria | Simultaneous 20-feature intervals for predictive mean and log variance; independent-bank cross-chain linear MMD upper interval. |
| Feature margins | Mean `0.15`; log-variance `log(1.15)`. These are reviewed working margins inherited from the July predictive design and remain target-specific hypotheses, not universal defaults. |
| MMD design | q=20 calibration-only path scale; bandwidths `0.5, 1, 2` times the frozen calibration median distance; tolerance selected before material results from q=20 null/controlled-shift calibration. |
| Hard vetoes | Archive/target/transport hash drift; nonfinite paths; invalid terminal covariance; XLA/CPU provenance mismatch; inadmissible covariance or intervals; calibration failure; missing independent-bank provenance; malformed artifact. |
| Promotion decision | `PASS` only if all feature intervals are strictly inside the margins and the MMD upper bound is below the frozen tolerance. A material feature or MMD interval gives `MATERIAL_DIFFERENCE`; otherwise `INCONCLUSIVE_UNDERPOWERED`. |
| Explanatory diagnostics | Parameter summaries, raw means/variances/quantiles, path distances, runtime, and calibration operating counts. |
| Must not conclude | Parameter identification, absolute posterior correctness, mode/tail coverage, model adequacy, sampler superiority, or default readiness. |

## Evidence contract

1. Baseline and comparator are exactly the posterior-mean plug-in and true q=20
   control under target signature
   `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
2. All forecast banks are independently seeded. Shared random numbers may be
   used only for a descriptive sensitivity run, never for the MMD decision.
3. The q=20 calibration bank is generated at the true control before material
   bank seeds are opened. Its center, scale, median path distance, bandwidths,
   and hashes are frozen in `calibration.json`.
4. The material bank uses four labeled iid forecast lanes, 2,048 draws per
   lane, and two forecast replications per draw. The 4,000 HMC draws are used
   only to compute the mean vector.
5. The formal statistic uses the repository TensorFlow/XLA predictive engine
   and `bayesfilter.inference.predictive_equivalence`; a fixed iid forecast
   lane is represented as four independent chains for the dependence-aware
   block calculation. This representation is admissible only because the
   forecast draws are independently generated, not because they are HMC chains.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Posterior mean | User-requested plug-in summary | Coordinatewise mean can be a poor nonlinear representative | Retain median and raw draw summaries as descriptive sensitivity only |
| Four lanes | Statistical API requires at least four disjoint chains for cross-chain MMD | Artificial lane grouping could hide dependence | Use stateless independent noise and record lane seeds; no HMC dependence claim |
| 2,048 material draws/lane | Bounded compute hypothesis; larger than previous 1,024-row diagnostic | Intervals may remain underpowered | Fail as `INCONCLUSIVE`, never pass from a wide interval |
| Block length 16 | Existing predictive-statistics design; divides 2,048 | Wrong dependence correction | Calibration null and iid lane declaration |
| Mean margin 0.15 and log-variance margin log(1.15) | Reviewed July design, transferred as working hypothesis only | q20-specific practical scale may differ | q20 calibration receipt and explicit nonclaim about margin promotion |
| MMD tolerance | Selected only from q20 calibration null/shift families before material bank | Post-hoc tolerance can manufacture a pass | Calibration receipt binds candidate counts and selected value |
| CPU/XLA | Explicit reference/diagnostic exception; GPUs hidden before TensorFlow import | Slower or different backend | Device manifest and XLA trace gate |

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. The comparator is the output law at the fixed synthetic control; no parameter-equality gate is used. |
| Proxy promoted? | No. Means, variances, quantiles, and parameter summaries are explanatory; only simultaneous intervals and MMD decide. |
| Missing stop condition? | No. Calibration hard veto, 900-second cap, nonfinite/device/hash stops, and explicit inconclusive branch are frozen. |
| Unfair comparison? | No. Both arms use the same target, forecast operator, shape, dtype, horizon, and independent-bank contract. |
| Hidden assumptions? | The iid-lane representation and transferred feature margins are exposed as assumptions with diagnostics and nonclaims. |
| Stale context? | The prior plug-in artifact is read only for the retained archive and mean construction; its descriptive result is not reused as a pass/fail authority. |
| Environment mismatch? | CPU-only is explicitly labeled diagnostic; `CUDA_VISIBLE_DEVICES=-1` is set before TensorFlow import and XLA is required. |
| Does the artifact answer the question? | Yes if calibration freezes the MMD design and the material receipt contains interval and MMD decisions. Otherwise stop as invalid/inconclusive. |

Audit verdict: **PASS FOR A BOUNDED Q20 FIXED-PLUG-IN EQUIVALENCE TEST**, with
the margin-transfer limitation recorded above.

## Execution phases

### Phase 1: q=20 calibration

Generate eight independent null/controlled-shift calibration replications at the
true q=20 control. Use 1,024 draws per lane and two forecast replications. The
null compares two independent true-control banks. The controlled alternative
adds a predeclared `+0.20` output-level shift to one bank. Freeze the q=20
standardization center/scale, median path distance, bandwidths, and the first
MMD tolerance whose null remains viable and controlled shift is detected. Stop
if no candidate is viable or any calibration statistic is inadmissible.

### Phase 2: material comparison

Load the authenticated seed-B archive, map all retained rows through the frozen
transport, compute the posterior mean, and generate independent true/mean banks
with 2,048 draws per lane and two replications. Compute the simultaneous
mean/log-variance intervals and MMD interval with the frozen calibration design.

### Phase 3: result and red-team note

Write `calibration.json`, `material.json`, and this plan's result note under a
versioned artifact root. Record command, environment, git state, target and
archive hashes, seeds, devices, XLA status, wall time, and all decision inputs.
State whether the result invalidated the harness or only rejected the current
plug-in candidate.

## Predeclared commands and budgets

```text
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_predictive_equivalence_2026_08_08.py \
  --mode calibration --output-root \
  docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-2026-08-08/r5 \
  --cap-seconds 900

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_predictive_equivalence_2026_08_08.py \
  --mode material --output-root \
  docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-2026-08-08/r5 \
  --cap-seconds 900
```

The material command requires the calibration receipt and refuses to overwrite
it. A 32-row canary is a mechanics-only check and cannot emit a scientific
decision.

The earlier `r1` calibration receipt is preserved as superseded engineering
evidence because it used one calibration replication. `r2` stopped closed after
an implementation error in the null feature contrast. `r3` is the repaired
eight-replication design-freeze attempt; `r4` preserves the candidate-level
diagnostic for the final calibration stop.

## Stop and interpretation rules

- `CALIBRATION_INCONCLUSIVE` means the q=20 MMD design was not frozen; the
  material comparison is closed and no predictive-equivalence decision is
  emitted.
- `INVALID_HARD_VETO` means the comparison did not answer the question.
- `INCONCLUSIVE_UNDERPOWERED` means the harness was valid but the evidence was
  insufficient; it is not equivalence.
- `MATERIAL_DIFFERENCE` rejects this posterior-mean plug-in under this target
  and frozen predictive design. It does not reject the research direction.
- `PASS` supports only bounded predictive-functional equivalence for this
  fixed plug-in, target, data, and forecast design.
`r5` is the terminal calibration-inconclusive receipt after the candidate-level
diagnostic repair. No material comparison is opened from this receipt.
