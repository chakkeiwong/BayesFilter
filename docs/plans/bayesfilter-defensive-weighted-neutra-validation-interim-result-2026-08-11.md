# Defensive weighted NeuTra validation interim result (2026-08-11)

> Superseded by
> `docs/plans/bayesfilter-defensive-weighted-neutra-validation-result-2026-08-11.md`.
> This file preserves the state at the former approval-stream interruption.

## Outcome

The weighted forward-KL implementation and analytic harness are mechanically valid.
The Gaussian canary passed. On the `(0.8, 0.2)` two-mode target, the original
three-stage `(32,32)` transport and the depth-only six-stage `(32,32)` repair both
systematically underallocated minority-mode mass across eight independent runs.
The combined six-stage `(64,64)` candidate is descriptively much healthier, but its
eight-run confirmation is incomplete because one paired GPU launch was rejected by
the platform approval reviewer after its stream disconnected.

This is not a posterior, HMC, paper-suite, SSL-LSTM, or default-readiness result.
Rung 1 remains open and all later rungs remain blocked by the plan.

## Evidence contract status

| Item | Status |
|---|---|
| Scientific question | Can target-weighted forward KL preserve known multimodal mass better than matched reverse KL? |
| Truth | Normalized analytic Gaussian-mixture density and component weights |
| Comparator | Same IAF family, initialization, optimizer, update count, and target under reverse KL |
| Primary r1 criterion | Analytic component weight must lie inside the eight-independent-run Student-t interval |
| Hard vetoes | Nonfinite value, missing component, invalid hash/device/XLA/memory receipt, wrong target or capacity identity |
| Explanatory only | Loss, latent moments, runtime, clipping count, and single-run covariance errors |
| Nonclaim | Passing one target or one canary does not validate HMC or SSL-LSTM |

## Implementation verification

Focused CPU/reference and CPU-XLA checks:

```text
21 passed in 13.30 s
18 passed in 8.53 s after capacity/replication parameterization
```

The tests cover exact mixture value/score, inverse/log-determinant round trip,
weighted reduction, finite-difference gradient agreement, deterministic XLA update,
matched reverse-KL construction, checkpoint restore, analytic moments, and interval
mathematics.

## Gaussian canary

Artifact:
`docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r0-gaussian/canary-v1/`

| Diagnostic | Weighted forward KL | Reverse KL |
|---|---:|---:|
| Audit weighted NLL | -0.80746 | -0.74103 |
| Latent covariance Frobenius error | 0.06945 | 0.48916 |
| Base-pushforward relative covariance error | 0.02179 | 0.24122 |
| Base-pushforward mean error | 0.00411 | 0.00710 |

All exploratory Gaussian gates passed. Importance ESS fraction was `0.7911`, the
maximum normalized weight was `3.42e-5`, and the normalizer-ratio estimate was
`0.00056`. The run used TensorFlow `2.20.0`, TFP `0.25.0`, float64, XLA, one RTX
4080 SUPER, verified memory growth, and `500 x 4096` updates/rows per arm. Wall time
was `29.77 s`; allocator peak was about `89.6 MB`.

## Three-stage two-mode candidate

The first 500-update canary recovered minority mass `0.1298`; its importance ESS
fraction was healthy (`0.7357`), so the failure was not weight collapse. At 2,000
updates, a seed-0 canary improved to `0.1600`, motivating the predeclared eight-run
test.

Artifact:
`docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/unequal-080-020-replication-summary-v1/`

| Quantity | Weighted forward KL | Reverse KL |
|---|---:|---:|
| Mean minority mass | 0.17249 | 0.11554 |
| 95% Student-t interval | [0.16278, 0.18219] | [0.11165, 0.11942] |
| Analytic truth | 0.20000 | 0.20000 |

All eight weighted runs were finite and observed both modes. The truth is outside
the weighted interval, so this candidate is rejected relative to the r1 target.
The comparison shows descriptive improvement over reverse KL; no paired ranking
test was predeclared, so no statistical superiority claim is made.

## Depth-only repair

The single six-stage `(32,32)` canary recovered minority mass `0.19061` and audit
NLL `4.00746`, which nominated it for replication.

Artifact:
`docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width32-replication-summary-v1/`

| Quantity | Value |
|---|---:|
| Mean minority mass | 0.18998 |
| Standard deviation across runs | 0.00474 |
| 95% Student-t interval | [0.18602, 0.19394] |
| Analytic truth | 0.20000 |

Again all eight runs were finite and observed both modes, but the interval excludes
truth. Depth alone reduces, but does not eliminate, systematic underallocation.

## Combined capacity candidate

The single six-stage `(64,64)` canary recovered minority mass `0.19926`, audit NLL
`3.97393`, latent covariance error `0.04850`, pushforward mean error `0.01532`, and
relative pushforward covariance error `0.01306`. This materially improves the
depth-only canary and is the active target-specific candidate.

Completed independent runs are replication `0` (canary) and replications `1`, `2`,
and `3`. Replication `3` finished after the paired wrapper's approval stream had
already disconnected; its hashes, capacity identity, XLA/GPU receipt, and memory
growth receipt all verify. Replication `4` did not launch. Its output directory was
created but is empty; it is ineligible and should not be reused.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Gaussian mechanics | Passed exploratory gates | No veto | One seed, calibration thresholds | Allowed r1 canary | No multimodal/HMC claim |
| Three-stage `(32,32)` | Failed eight-run interval | No hard veto | Target-specific only | Reject candidate, repair capacity | Weighted objective not rejected |
| Six-stage `(32,32)` | Failed eight-run interval | No hard veto | Residual systematic bias | Reject depth-only candidate | No componentwise conclusion yet |
| Six-stage `(64,64)` | Confirmation incomplete | No veto in completed runs | Four of eight runs complete | Finish replications 4--7 sequentially | No promotion/default/HMC claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for all terminal artifacts; empty replication-4 directory is ineligible |
| Statistically supported ranking | None; method differences are descriptive only |
| Descriptive-only differences | Weighted arms recover more minority mass and lower audit NLL than reverse KL |
| Default readiness | Not assessed and not eligible |
| Next evidence needed | Four remaining combined-capacity replications, eight-run interval, then remaining r1 target variants |

## Run manifest summary

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` plus recorded dirty state |
| Environment | `tfgpu`, Python 3.13, TensorFlow 2.20.0, TFP 0.25.0 |
| Hardware | RTX 4080 SUPER; one visible GPU per process |
| GPU policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`, repository helper verified growth before initialization |
| Numerical mode | float64, TF32 disabled, XLA enabled and compile receipt observed |
| Training | Batch-native TensorFlow target, batch 4096, no scalar/map fallback |
| Seeds | Disjoint initialization, train, selection, audit, and base-audit domains per replication |
| Artifact integrity | Per-run SHA-256 manifests; aggregation verifies result hashes and capacity/seed identity |
| Plan | `docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md` |

Exact commands, device identity, allocator bytes, wall time, target parameters,
optimizer settings, git status, seeds, and hashes are preserved in each run's
`run_manifest.json`, `result.json`, and `artifact_hashes.json`.

## Negative-result classification

- Implementation failure: not supported; focused tests and Gaussian mechanics pass.
- Tuning/budget failure: supported for the 500-update arm; 2,000 updates repaired
  much of the error.
- Capacity failure: supported for the three-stage and depth-only candidates on this
  target.
- Diagnostic failure: not supported; analytic truth, healthy importance weights,
  independent streams, and hash checks agree.
- Evidence against weighted forward KL: not supported; increasing capacity moves
  mass and NLL toward truth, but final combined-capacity replication is unfinished.

## Post-run red team

The strongest alternative explanation is that the combined IAF works only because
the modes and defensive proposal are known and well separated. Even a passing
eight-run interval would not establish discovery of unknown SSL-LSTM modes. A
failed combined interval would overturn the generic-IAF repair and trigger a
componentwise or augmented-state transport. The weakest current evidence is the
unfinished combined-capacity replication and the use of one two-mode geometry.
