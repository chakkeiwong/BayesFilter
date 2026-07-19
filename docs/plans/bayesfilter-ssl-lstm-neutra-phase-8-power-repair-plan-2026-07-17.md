# SSL-LSTM NeuTra Phase 8 Predictive Power Repair Plan

Date: 2026-07-17

Status: `NOMINATION_UNDERPOWERED_BOUNDED_LADDER_CLOSED`

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can a prospectively defined design repair achieve controlled equivalence and material-difference power without weakening lineage, using G/H confirmation outcomes, or acquiring HMC draws before feasibility is shown? |
| Exact baseline | Failed 448-draw symmetric-Bonferroni nomination receipt `ec112880...`; target-pilot bandwidths and all random/lineage contracts remain fixed |
| Candidate mechanisms | More precision at `1984` confirmation draws per chain; midpoint margins; and separate intersection-union TOST bounds for the global all-features equivalence claim while retaining Bonferroni bounds for any-feature material claims |
| Primary promotion criterion | On 20 fresh design-development replications, one prespecified arm satisfies the same `18/20` coverage, `16/20` required decision, and at-most-`1/20` false-decision screens for every required family and nominates the smallest passing MMD tolerance |
| Promotion vetoes | Any required family underpowered; invalid interval algebra/error-control derivation; covariance/MMD invalidity; or selecting an arm/tolerance outside the prospective order |
| Continuation vetoes | Source/receipt drift, wrong random hierarchy, G/H confirmation leakage, nonfinite output, GPU/XLA/trace failure, corrupted artifact, cap exhaustion, or no viable candidate at the maximum rung |
| Explanatory only | Per-arm/per-family rates, widths, runtime, skew/dependence behavior, and descriptive comparisons among viable arms without fresh statistical ranking evidence |
| Nonclaims | No posterior truth, G/H predictive equivalence, sampler superiority, model adequacy, default readiness, or HMC acquisition need until controlled feasibility passes |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-power-repair-result-2026-07-17.md` plus JSON receipts under the Phase 8 artifact directory |

## Candidate Ladder

All arms use identical generated tensors, target-pilot bandwidths, block length
16, alphas `0.03/0.02`, the same covariance/MMD code, and material anchors
`0.20` mean and `+/-log(1.25)` variance.

| Arm | Draws per chain | Mean/log-variance margin | Equivalence bound | Material bound | Role |
| --- | ---: | --- | --- | --- | --- |
| A | 448 | `0.15/log(1.15)` | symmetric Bonferroni | symmetric Bonferroni beyond margin | immutable failed baseline receipt |
| B | 1984 | `0.15/log(1.15)` | symmetric Bonferroni | symmetric Bonferroni beyond margin | precision-only repair |
| C | 1984 | `0.10/0.5*log(1.25)` | symmetric Bonferroni | symmetric Bonferroni beyond margin | precision plus midpoint margins |
| D | 1984 | `0.10/0.5*log(1.25)` | componentwise one-sided `alpha=0.03` TOST used through an intersection-union all-features claim | Bonferroni simultaneous interval beyond margin | enhanced repair |

The midpoint margins maximize the smaller distance between the zero-equivalent
center and the approved material anchors. They tighten the equivalence region
relative to Arm A rather than relaxing it.

For Arm D, no multiplicity correction is needed for the all-features
equivalence claim: its null is the union that at least one feature lies outside
its margin, while rejection requires every component TOST to reject; under any
point in that union the probability of rejecting all components is bounded by
the level of the true component null. The any-feature material claim is a
union, so it retains the Bonferroni simultaneous interval. No categorical
mathematical claim will be made until this derivation is represented in tests
and focused review.

Prospective selection order is `B`, then `C`, then `D`: select the first arm
that passes every required screen, and within it the smallest passing MMD
tolerance. This prefers the least changed valid design. If none passes, stop
without HMC acquisition.

## Skeptical Pre-Execution Audit

- Wrong baseline: prevented by hard-binding the failed nomination receipt and
  preserving it as Arm A rather than rerunning a convenient baseline.
- Proxy promotion: a smoke or early descriptive rate cannot nominate an arm;
  promotion requires all 20 fresh design replications.
- Unfair comparison: B/C/D share generated tensors and differ only in declared
  sample/decision mechanisms; selection order is prospective, not metric rank.
- Hidden assumption: 1984 is chosen because it equals the previously planned
  but not yet acquired 2048-draw Phase 7 checkpoint minus the permanently
  excluded 64-draw pilot, not because a G/H confirmation result was opened.
- Stale context/leakage: no retained archive or confirmation forecast path is
  accepted by the controlled runner; pilot inputs are hash-bound calibration
  constants only.
- Misleading pass: synthetic feasibility cannot establish actual G/H
  equivalence, posterior correctness, or that additional HMC draws will pass
  sampler admission.
- Tuning failure versus idea failure: failure of B/C but success of D isolates
  interval logic; failure of all arms rejects this bounded design ladder, not
  predictive validation in general.

Audit disposition: `PASS_IMPLEMENTATION_AND_CPU_TESTS_ONLY`. GPU smoke remains
closed until the runner, tests, exact output/cap, and focused native review are
recorded below.

## Required Artifacts And Checks

- separate controlled repair runner; no mutation of the immutable failed
  receipt;
- strict binding of target pilot, passing smoke, and failed nomination;
- TensorFlow/TFP `float64`, XLA-default implementation;
- analytic interval-algebra fixtures for Arms B/C/D;
- TOST intersection-union boundary tests and Bonferroni material-bound tests;
- shared-tensor arm comparison and seed replay tests;
- incomplete-run nomination rejection and prospective futility tests;
- proof validation and HMC acquisition remain closed;
- focused native review before one full-shape GPU smoke; and
- a separately frozen 20-replication command only after smoke audit.

## Resource And Stop Boundary

The first GPU action will be one replication of one null and one persistent
mean material family at the exact `[4,1984,2,10]` shape. Freeze its command and
cap only after timing tests. It cannot nominate an arm or tolerance.

No additional Phase 7 HMC transition, target-pilot replay, G/H confirmation
forecast, or validation run is authorized by this plan. Those actions require
a passing controlled repair nomination and a subsequent exact amendment.

## GPU Smoke Freeze

Focused native review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-power-repair-native-review-2026-07-17.md`,
verdict `AGREE_GPU_SMOKE_ONLY`.

Focused checks passed: `82` power-repair/controlled/predictive tests; Python
compilation; and `git diff --check`. The smoke uses fresh seed
`(15501,15502)`, one iid null and one persistent `+0.20` mean family, and the
exact `[4,1984,2,10]` shape. It exercises all seven compiled surfaces and all
three candidate decision paths but always returns
`PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED` with null selection.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-power-repair-smoke-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-power-repair-smoke-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_power_repair_2026_07_17.py --mode smoke --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/power-repair-smoke.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without automatic retry on
binding drift, nonfinite output, inadmissible covariance/MMD, GPU/XLA/trace
failure, selection leakage, serialization failure, or cap exhaustion. A pass
opens receipt audit and prospective nomination freeze only.

The smoke passed in `14.874700225074776` seconds with decision
`PHASE8_POWER_REPAIR_SMOKE_PASSED_NOMINATION_REQUIRED`. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/power-repair-smoke.json`,
SHA-256
`d0a23bcd3ebd2340c955824941ce6b726d6eee0b2c538d96a38c518188f212f3`.
All seven compiled surfaces traced once; both covariance rows selected ridge
zero with condition numbers `26.8666` and `23.4928`; selection remained null;
and the exact source/result bindings replayed. Descriptively, the iid null
passed B/C/D, B was inconclusive for persistent `+0.20` mean, and C/D detected
that material row. These are one-replication smoke observations, not a ranking
or candidate nomination.

## Power Repair Nomination Freeze

The material runner now hard-binds the exact power-repair smoke receipt and
the historical source hash that produced it. It uses fresh root seed
`(16001,16002)`, all 13 frozen families, and at most 20 replications. B/C/D
reuse the same generated tensors in each family/replication. Selection order
remains B then C then D, and the smallest passing tolerance within the first
passing arm. A candidate cannot nominate before 20 complete replications.

The only sequential stop is futility: strictly before replication 20, stop if
no arm/tolerance can meet every frozen count threshold even after all remaining
outcomes are assigned favorably. A failure triggers a bounded design stop; it
does not authorize another margin/sample-size search or reject predictive
validation as a research direction.

Skeptical audit disposition: `PASS_POWER_REPAIR_NOMINATION_ONLY`. The failed
448-draw design remains the exact baseline; all repair mechanisms and selection
order were declared before fresh outcomes; the smoke is not promoted; no G/H
confirmation input or retained archive is accepted; and the receipt will
distinguish execution validity from candidate power. A pass authorizes only
fresh-validation design plus a separate decision on whether to acquire the
prospective additional HMC draws.

Focused nomination review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-power-repair-native-review-2026-07-17.md`,
verdict `AGREE_NOMINATION_ONLY`. Focused checks: `83` tests; Python compilation;
and `git diff --check` passed.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-power-repair-nomination-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-power-repair-nomination-cuda timeout 2460s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_power_repair_2026_07_17.py --mode nomination --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/power-repair-nomination.json --wall-cap-seconds 2400
```

Resource contract: one invocation on physical GPU 1, at most `2400` runner
seconds (`0.6667` GPU-hour) plus `60` seconds cancellation margin. Stop without
automatic retry on binding drift, invalid interval/covariance/MMD output,
nonfinite value, hard-veto computation, non-GPU placement, retracing,
serialization failure, cap exhaustion, or prospective futility. No validation,
HMC acquisition, G/H confirmation, or predictive-equivalence action is bundled.
