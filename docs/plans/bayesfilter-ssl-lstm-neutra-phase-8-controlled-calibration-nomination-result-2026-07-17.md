# SSL-LSTM NeuTra Phase 8 Controlled Calibration Nomination Result

Date: 2026-07-17

Decision: `PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED`

## Evidence Contract Outcome

| Field | Result |
| --- | --- |
| Question | Can the frozen `[4,448,2,10]` design nominate an MMD tolerance while satisfying every required null/equivalence coverage and material-power screen? |
| Exact baseline | Frozen Phase 8 mean/log-variance Bonferroni intervals plus cross-chain MMD, using the target-pilot bandwidth ladder and fixed margins/alphas/block length |
| Primary criterion | No tolerance passed; all six candidates became mathematically unable to reach the frozen 20-replication thresholds after replication 5 |
| Hard vetoes | None: source/pilot/smoke lineage, covariance, MMD, finite values, GPU/XLA placement, one-trace gates, serialization, and resource gates passed |
| Explanatory diagnostics | Coverage, observed interval widths, condition numbers, ridge choices, MMD intervals, and per-family early counts |
| Nonclaims | No G/H confirmation result, predictive-equivalence result, posterior truth/correctness, sampler ranking, model adequacy, or rejection of predictive validation as a research direction |

The immutable receipt is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-nomination.json`,
SHA-256
`ec112880f6e9f33432ad5c12f2ccc81efd71b40a75470fca45293a7aba225b49`.
It stopped after five of the planned 20 replications in
`96.7839861450484` seconds under the prospective futility rule. No tolerance
was selected and validation remained closed.

## What Failed

The failure is feature-interval power, not MMD tolerance selection:

- every required null/equivalence family had zero PASS outcomes at every MMD
  tolerance in the first five replications;
- required material families produced only zero to two material detections;
- identical-law and controlled-truth simultaneous coverage was mostly intact,
  including `5/5` for all but one true-equivalent family (`4/5`);
- maximum simultaneous 20-feature interval widths were approximately
  `0.27-0.39` for required families;
- every covariance was admissible with ridge zero and modest reported condition
  numbers; and
- all six compiled surfaces traced once.

The current strong material rule asks an interval to lie wholly beyond the
equivalence margin. For mean controls, the material anchor `0.20` is only
`0.05` beyond the current `0.15` margin. The observed half-widths are much
larger than that separation. MMD tolerance changes cannot repair a feature
gate that is inconclusive for every tolerance.

The two-arm covariance scaling was re-derived from the implementation. Passing
`[2*left_influence,-2*right_influence]` as eight chains is correct because the
long-run helper divides by eight chain means; this yields the sum of the two
independent four-chain mean covariances. The underpower result is not a hidden
factor-of-two covariance bug.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the frozen 448-draw design | Failed by prospective futility | No hard veto | Which combination of precision, scientifically anchored margins, and valid equivalence logic restores power | Run a fresh controlled feasibility ladder before acquiring more HMC draws | Predictive-validation direction failure |
| Keep validation and Phase 9 closed | No tolerance exists | Pass | No selected design to validate | Hard-bind this receipt in the repair ladder | Any G/H predictive statement |
| Preserve target pilot and machinery | All execution/validity gates passed | Pass | Pilot dependence may differ from AR controls | Reuse bandwidth/scale lineage; do not reopen G/H confirmation | Posterior correctness or complete mode coverage |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed |
| Statistically supported ranking | None; no viable tolerance remained |
| Descriptive-only differences | Per-family counts, interval widths, MMD values, and condition numbers are descriptive at five replications |
| Default-readiness | Not applicable and not established |
| Next evidence needed | Fresh-seed controlled comparison of sample-size and interval-design repairs, followed by independent 60-replication validation if a candidate is nominated |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `ffaaaf903354e095da126dbfa47878c34717c5b8` with dirty worktree preserved |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py --mode nomination --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-nomination.json --wall-cap-seconds 1800` |
| Environment | `tfgpu`; TensorFlow `2.20.0`; TFP `0.25.0`; Python `3.13.13` |
| Device | physical GPU 1 exposed as TensorFlow `/GPU:0`; XLA JIT; TF32 enabled; `float64` tensors |
| Seed | `(14001,14002)` |
| Wall time | `96.7839861450484` seconds |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-plan-2026-07-17.md` |
| Result | this file |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

## Post-Run Red Team

Strongest alternative explanation: the AR(0.6) synthetic controls may be more
dependent than the eventual target confirmation forecasts. That cannot rescue
the frozen design prospectively because the required controlled family was
part of its declared robustness contract. Conversely, simply dropping the AR
control after seeing failure would be post-outcome weakening.

What would overturn the repair need: a reproduced arithmetic or lineage defect
in the interval construction. The covariance scaling and receipt gates were
checked and no such defect was found.

Weakest evidence: only five stochastic replications executed. The futility
claim itself does not rely on extrapolating their observed rates; it follows
from frozen count thresholds and the maximum favorable outcomes remaining.
