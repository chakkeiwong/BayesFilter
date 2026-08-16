# Defensive weighted NeuTra validation result (2026-08-11)

> Superseded by the fresh-seed width-128 confirmation in
> `docs/plans/bayesfilter-defensive-weighted-neutra-width128-updates10000-confirmation-result-2026-08-12.md`.
> This file remains the terminal record for the earlier width-64 capacity ladder;
> its conclusion that generic IAF scaling was exhausted is no longer current.

## Verdict

The defensive weighted forward-KL implementation is mechanically correct on the
tested analytic authorities and clearly improves the tested multimodal transport
over the original low-capacity candidate. However, every generic IAF capacity in
the reviewed ladder fails the predeclared replicated component-weight criterion.
The strongest six-stage `(64,64)` candidate recovers mean minority mass `0.19478`,
with 95% Student-t interval `[0.19045, 0.19910]`, which excludes analytic truth
`0.20`.

Therefore:

- weighted forward KL remains scientifically viable;
- the six-stage `(64,64)` generic IAF is rejected for r1 promotion;
- generic IAF scaling is exhausted under this plan;
- r1 and all later rungs remain blocked pending a componentwise or augmented-state
  transport repair;
- no HMC, posterior, paper-suite, DSGE, SSL-LSTM, or default-readiness claim is
  supported.

## Research target

Claimed target: an invertible NeuTra density whose base pushforward reproduces the
normalized analytic two-component Gaussian mixture, including component weights
`(0.8, 0.2)`.

Quantity computed: eight independent training and audit replications per selected
capacity, with component probabilities estimated by analytic target
responsibilities on 65,536 base-pushforward draws per replication. The campaign
computed a two-sided 95% Student-t interval across training replications.

Verdict: the computed quantity is the planned component-mass diagnostic. Its
interval excludes the target value for all three generic IAF candidates. This is
wrong relative to exact component-weight recovery, although the strongest
candidate's absolute mean error is only `0.00522` and is descriptively close.

## Execution recovery

The earlier paired launch failed because the automatic approval review stream
disconnected. The recovery used one process, one GPU, and one attached terminal
session at a time. Replications 4--7 then completed without any stream disconnect:

| Replication | Wall time (s) | Visible host GPU | Artifact status |
|---:|---:|---:|---|
| 4 | 98.79 | 1 | Valid, v2 fresh root |
| 5 | 99.27 | 1 | Valid |
| 6 | 99.16 | 1 | Valid |
| 7 | 102.16 | 1 | Valid |

Every run recorded XLA enabled, float64, TF32 disabled, batch-native TensorFlow
target evaluation, no row-wise/scalar fallback, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
and verified repository memory-growth policy. Result, manifest, and trainer-state
SHA-256 values match every declared artifact hash.

## Analytic results

### Gaussian mechanics

The Gaussian canary passed all exploratory gates.

| Diagnostic | Weighted forward KL | Reverse KL |
|---|---:|---:|
| Audit weighted NLL | -0.80746 | -0.74103 |
| Latent covariance error | 0.06945 | 0.48916 |
| Pushforward relative covariance error | 0.02179 | 0.24122 |
| Pushforward mean error | 0.00411 | 0.00710 |

Importance ESS fraction was `0.7911`; maximum normalized weight was `3.42e-5`.
This validates mechanics and permits multimodal testing, but does not establish a
multimodal or posterior claim.

### Multimodal capacity ladder

| Weighted IAF | Mean minority mass | SD across runs | 95% interval | Truth | Primary status |
|---|---:|---:|---:|---:|---|
| 3 stages, `(32,32)` | 0.17249 | 0.01161 | [0.16278, 0.18219] | 0.20000 | Fail |
| 6 stages, `(32,32)` | 0.18998 | 0.00474 | [0.18602, 0.19394] | 0.20000 | Fail |
| 6 stages, `(64,64)` | 0.19478 | 0.00517 | [0.19045, 0.19910] | 0.20000 | Fail |

All weighted runs represented both modes and were finite. Capacity reduces the
systematic minority-mass deficit, but the strongest interval still misses truth by
`0.00090` at its upper endpoint.

For the strongest candidate, per-run minority masses were:

```text
0.19926, 0.19288, 0.19953, 0.18948,
0.18875, 0.19966, 0.19945, 0.18922
```

Mean audit weighted NLL was approximately `3.99289`; the separately measured target
self-NLL was `3.94921 +/- 0.00162`, leaving a descriptive forward-KL gap near
`0.04368`. This supports residual density mismatch independently of the component
weight interval.

The high-capacity matched reverse-KL arm selected its initial checkpoint and had
audit NLL near `12.9`; its responsibility-based component fraction near `0.208` is
not evidence of a correct density. Component classification without density
agreement is explanatory only.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Gaussian mechanics | Pass exploratory gates | Pass | One analytic family | Permit multimodal rung | No posterior claim |
| 3-stage `(32,32)` | Fail interval | Pass | Target-specific | Reject candidate | Objective not rejected |
| 6-stage `(32,32)` | Fail interval | Pass | Residual bias | Reject candidate | Componentwise need not yet proved alone |
| 6-stage `(64,64)` | Fail interval | Pass | Small but systematic deficit | Stop generic scaling; componentwise repair | No HMC/default claim |
| Research direction | Not rejected | Harness valid | Unknown-mode transfer unresolved | Build analytic componentwise/augmented test | No SSL-LSTM readiness |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Pass: all terminal runs finite, both modes present, hashes and identities valid |
| Statistically supported ranking | None; no paired method-ranking test was predeclared |
| Descriptive differences | Larger weighted IAFs have component mass and NLL closer to analytic truth |
| Default readiness | Not assessed; all candidates ineligible |
| Next evidence needed | Componentwise/augmented repair plus fresh eight-run interval on the same target |

## Negative-result classification

- Implementation failure: not supported. Mechanics, gradients, XLA, checkpointing,
  and analytic utilities pass focused tests.
- Proposal-weight failure: not supported. ESS fractions remained about `0.734` to
  `0.738`, with small maximum normalized weights.
- Budget failure: the 500-update failure was partly budget-driven, but selected
  checkpoints at 1,900--2,000 updates and the exhausted capacity ladder do not
  justify more generic scaling.
- Capacity/model-family failure: supported for tested generic IAFs on this target.
- Diagnostic failure: not supported. Analytic truth, independent seeds, untouched
  audit streams, and hash-checked aggregation agree.
- Evidence against weighted forward KL: unsupported. The objective remains viable;
  its generic transport family is the failed candidate.

## Post-run red team

The strict interval tests equality of the mean trained transport's component mass
to analytic truth. The absolute miss is small, so this result does not show that the
candidate would be practically harmful in every downstream task. Nevertheless,
relaxing the criterion after seeing the result would invalidate the plan, and the
remaining NLL gap independently indicates residual mismatch.

The strongest alternative explanation is incomplete target-specific optimizer or
architecture tuning rather than topology. Against that explanation, width, depth,
and budget were all increased and improved the result but did not remove the
replicated bias. A componentwise or augmented-state repair is now the smallest
discriminating experiment. A future fresh eight-run interval containing `0.20`
under unknown or soft mode assignment would overturn the current candidate
rejection.

## Artifacts

- Plan: `docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md`
- Canonical summary:
  `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width64-replication-summary-v1/result.json`
- Canonical result SHA-256:
  `f6e01bd17fe24551d090a1c16da7aafad07e0640d7a3b202d5fccbcf47039a3a`
- Reset memo:
  `docs/plans/bayesfilter-defensive-weighted-neutra-validation-reset-memo-2026-08-11.md`

Terminal focused verification: `24 passed` in `10.24 s`; warnings were existing
TensorFlow Probability and Python 3.15/gast deprecations only.
