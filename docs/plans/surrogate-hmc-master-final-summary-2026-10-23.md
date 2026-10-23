# Surrogate-HMC master program: five-model completion

The requested five-model sampling program is complete. LGSSM, KSC stochastic
volatility, predator–prey, range bearing and SIR all completed retained sampling
and passed cumulative modern R-hat <=1.01. SIR's completed NeuTra run closes the
remaining model gap in the 21 October summary. No planned tuning, training or
sampling phase remains active or pending.

The goal was to test surrogate HMC: LEDH supplies the proposal force, and the
declared endpoint supplies the MH correction. The results establish completion
under the agreed sampling rule for those numerical targets. Agreement with an
exact model posterior is a separate, unevaluated claim and is not an additional
required phase of this program. ESS and MCSE are precision diagnostics only.

## Completed runs

There are four chains per model. Counts below are per chain; all retained draws
are inference draws and all warm-up draws are excluded. The first four named
models in the original request use T50; range bearing uses the saved T20 fixture.
N4032 is the requested particle count. Individual manifests retain their actual
precision, target, prior, filter tuning and source provenance.

| Model | Warm-up | Retained | Maximum parameter R-hat | Acceptance range | Result |
|---|---:|---:|---:|---:|---|
| LGSSM | 2,000 | 1,000 | 1.007082 | 73.9–74.5% | Passed |
| KSC SV | 2,000 | 1,000 | 1.005998 | 68.5–72.6% | Passed |
| Predator–prey | 3,000 | 6,000 | 1.007034 | 71.3–72.0% | Passed |
| Range bearing | 2,000 | 1,000 | 1.003620 | 68.0–70.2% | Passed |
| SIR NeuTra | 2,000 | 2,000 | 1.009603 | 75.9–79.3% | Passed |

Each entry was checked against its saved HMC JSON, including `VERIFIED`,
`posterior_sampling_complete=true`, counts and parameter-chart R-hat. The
[machine-readable completion report](artifacts/surrogate-hmc-five-model-completion-20261023-01/summary.json)
records exact evidence paths and hashes. Original run artifacts remain unchanged.
The [SIR terminal report](artifacts/sir-neutra-training-20261022-neutra-only01/final-result-20261023.md)
explains its corrected assessment and preserved precision diagnostics.

## Simulation truth and posterior checks

All 17 simulation truths fall within their empirical equal-tailed 95% posterior
credible intervals. Every parameter's modern R-hat is strictly below 1.01,
including recomputation in physical model coordinates. Acceptance is within
65–75% for every chain of LGSSM, KSC SV, predator–prey and range bearing. SIR's
75.9–79.3% is above that requested band; sampling completion does not imply
that this separate tuning diagnostic passed.

The [full parameter table](artifacts/surrogate-hmc-five-model-completion-20261023-01/posterior-audit-03/report.md)
and its JSON report give every truth, interval endpoint, R-hat and chain
acceptance. The CPU audit verified input hashes, reconstructed the nomination
affine transformation and SIR's frozen NeuTra map, and traced the simulation
truths to their data generators. Reconstructed means and standard deviations
agree with saved reports within 1.8e-15; acceptance and parameter-chart R-hat
agree exactly. Warm-up draws are excluded. These are checks on the five saved
datasets, not a repeated-dataset coverage study or proof of exact-posterior
accuracy.

## Decision and remaining housekeeping

| Decision item | Finding |
|---|---|
| Decision | All five requested models completed the sampling program. |
| Primary criterion | All recorded parameter-chart cumulative modern R-hat maxima are <=1.01 and sampling completed. |
| Veto diagnostic status | All five saved sampler results report `VERIFIED`; SIR's previous ESS-based completion veto was removed under the owner directive. |
| Main uncertainty | Finite-run convergence diagnostics and approximate endpoint accuracy remain distinct. |
| Next justified action | No further compute phase is required. Complete the owner's requested commit, main/remote merge, push and branch synchronization under the closeout plan. |
| What is not concluded | Exact-posterior accuracy, statistical superiority, or default readiness. |

| Inference status | Finding |
|---|---|
| Hard veto screen | Saved sampler validity/completion checks passed. |
| Statistically supported ranking | None; these different model runs are not a method comparison. |
| Descriptive-only differences | Acceptance, counts, ESS and timing do not rank methods or models. |
| Default readiness | Not assessed by this completion report. |
| Next evidence needed | None for the agreed sampling completion rule; new accuracy or efficiency claims would require their own evidence. |

The master assessment repair passed 35 focused CPU tests. No extra GPU work was
used for reassessment, aggregation or the interval audit. Repository integration
is tracked in the [closeout plan](surrogate-hmc-posterior-closeout-2026-10-23.md).
The earlier Git integration record describes the repository state before these
subsequent changes.
