# SSL-LSTM NeuTra Proper-Score Direct Calibration Plan

Date: 2026-07-17

Status: `EXECUTED_CURRENT_CANDIDATE_REJECTED_REPAIR_REQUIRED`

## Research Intent Ledger

| Item | Prospective contract |
| --- | --- |
| Main question | Does the repaired growing-HAC, joint-confidence-region procedure have acceptable finite-sample coverage and decision behavior on the declared controlled predictive laws? |
| Exact baseline | The implemented equal-weight average proper-score loss over ten horizons, evaluated with a 95% joint 20-feature Wald region and growing Bartlett HAC. The historical fixed-16 and coordinate-margin methods are closed historical diagnostics, not admissible comparators. |
| Candidate mechanism | Add a horizonwise maximum-loss veto, evaluated over the same joint region, so one-horizon material discrepancies cannot be diluted by averaging. |
| Expected failure mode | The joint region may be too wide for equivalence, growing HAC may be unstable in 20 dimensions, or local effects may remain underpowered at 4,096 draws. |
| Promotion criterion | At one prospective draw rung, simultaneous one-sided exact binomial bounds establish coverage at least `0.90`, required-decision probability at least `0.80`, false-decision probability at most `0.05`, and invalid-procedure probability at most `0.05` for every required controlled family. |
| Promotion veto | Failure of any of those familywise-controlled operating targets blocks the design at that rung. It does not reject predictive validation, NeuTra, HMC, or the SSL-LSTM. |
| Continuation veto | Malformed inputs, non-finite generated laws, GPU/XLA/device failure, implementation/authentication failure, corrupted receipt, resource-cap exhaustion, or evidence that the declared truth/loss algebra is wrong. A failed 4,096-draw candidate alone is not a continuation veto; it triggers the 8,192-draw repair rung. |
| Repair trigger | Any valid but non-passing 4,096-draw operating screen triggers the prospectively frozen 8,192-draw rung with an independent seed domain. |
| Explanatory only | Point losses, per-family decision counts, condition numbers, KKT residuals, and runtimes. MMD is deliberately omitted from this primary calibration; any later MMD result remains explanatory unless separately calibrated. |
| Forbidden conclusions | No G/H equivalence or difference, posterior correctness, HMC readiness, sampler ranking, model adequacy, statistical superiority, default readiness, or scientific validation of NeuTra. |

## Scientific Loss Contract

For feature vector

\[
  \delta=(\delta_{\mu,1},\ldots,\delta_{\mu,10},
           \delta_{\log v,1},\ldots,\delta_{\log v,10}),
\]

define the horizon loss and equal-weight average loss

\[
 r_h(\delta)=\frac12\delta_{\mu,h}^2+
              \frac14\delta_{\log v,h}^2,
 \qquad
 L_{\rm avg}(\delta)=\frac1{10}\sum_{h=1}^{10}r_h(\delta),
 \qquad
 L_{\max}(\delta)=\max_h r_h(\delta).
\]

Both criteria are evaluated on the same 95% 20-dimensional confidence
ellipsoid. Equivalence requires the exact upper bound for `L_avg` and every
exact horizonwise upper bound to lie below their thresholds. Material
difference requires the exact lower bound for `L_avg`, or at least one exact
horizonwise lower bound, to exceed its threshold. All other valid outcomes are
inconclusive.

The declared negligible anchors are standardized mean shift `0.05` and
variance ratio `1.05`. The declared material anchors are standardized mean
shift `0.20` and variance ratios `1.25` and `0.80`. On the proper-score scale,

\[
 K_N=\max\left\{\tfrac12(0.05)^2,
                  \tfrac14\log(1.05)^2\right\}=0.00125,
\]

\[
 K_M=\min\left\{\tfrac12(0.20)^2,
                  \tfrac14\log(1.25)^2,
                  \tfrac14\log(0.80)^2\right\}
     =\tfrac14\log(1.25)^2\mathrel{\approx}0.0124481.
\]

Freeze the maximin additive-separation threshold

\[
 K_{\rm avg}=K_{\max}=\frac{K_N+K_M}{2}
 \mathrel{\approx}0.0068491.
\]

This midpoint is taken in scientific loss, not independently in mean and
variance coordinates. It maximizes the smaller additive clearance from the
nearest declared negligible and material anchors. The same numerical threshold
is used for the two contracts because their persistent and one-horizon anchor
losses have the same horizonwise scale; their roles remain distinct.

One average loss alone is inadmissible: a persistent negligible mean shift has
`L_avg=0.00125`, while a one-horizon material `1.25` variance ratio has
`L_avg=log(1.25)^2/40`, approximately `0.0012448`. No single average threshold
can classify both as declared. The horizonwise veto is therefore part of the
primary contract, not an optional diagnostic.

## Controlled Families

Four independent chains per arm, two forecast replications, ten horizons, and
independent arm banks are used. Required families are:

- equivalence: IID null; a stationary draw-cluster law with latent AR(1)
  coefficient `phi=0.6` (forecast-replication noise reduces the observed path
  autocorrelation); persistent mean shift
  `0.05`; persistent variance ratio `1.05`;
- material: persistent mean shifts `+0.20` and `-0.20`; one-horizon mean shifts
  `+0.20` and `-0.20`; persistent variance ratios `1.25` and `0.80`; and a
  one-horizon variance ratio `1.25`;
- explanatory only: a skewed law and a stronger cross-horizon-dependence law.

The truth vector is known analytically for every required family. Skew and
dependence families do not have a required decision and cannot affect
promotion.

## Statistical And Sequential Contract

- Confidence-region alpha: `0.05`, with `chi2_20(0.95)` radius.
- HAC: per-chain centered Bartlett, multiplier `1.0`, bandwidth
  `floor(N^(1/3))`, condition-number limit `1e8`, and ridge ladder exactly
  `(0.0,)`. A positive ridge is forbidden for inference.
- Draw ladder: `4,096`, then `8,192` draws per chain. Each rung uses an
  independent Philox seed domain. Stop after 4,096 only if every controlled
  target passes; otherwise continue to 8,192 unless a continuation veto fires.
- Replications: `256` per family per material rung. A smoke uses separate seeds
  and is mechanics-only; it is not a statistical look.
- There are 11 required families, four operating claims per family, and at most
  two material looks: 88 one-sided claims. Each reported operating bound is an
  exact Clopper--Pearson bound at tail probability `0.05/88`. This Bonferroni
  construction gives simultaneous coverage at least 95% without assuming
  independence across metrics, families, or rungs.
- Coverage counts whether the known 20-vector lies inside the implemented joint
  ellipsoid. Invalid procedure replications count as uncovered and are also
  reported separately.
- Required decision means `PASS` for an equivalence family and
  `MATERIAL_DIFFERENCE` for a material family. False decision means the opposite
  decisive result. Invalid and inconclusive outcomes do not count as false
  decisions, but cannot be hidden because required-decision and invalid-rate
  bounds are co-required.
- With 256 replications, up to two false/invalid observations are capable of an
  upper bound below `0.05`, and sufficiently high observed coverage/power can
  cross their lower targets. The replication count is therefore capable of
  answering the frozen gate without requiring a literally perfect false-event
  record; it does not guarantee a pass.

## Evidence Contract

| Evidence role | Contract |
| --- | --- |
| Primary promotion criteria | Simultaneous exact-binomial lower bounds for coverage and required decisions, and upper bounds for false decisions and invalid procedures, at one rung |
| Promotion vetoes | Any required family misses any one of the four targets |
| Continuation vetoes | Invalid run-level artifact, invalid controlled-law algebra, device/XLA/non-finite failure, cap exhaustion, or prohibited input access |
| Repair triggers | A valid 4,096 failure triggers 8,192; focused code/test failure triggers visible repair and recheck before GPU material execution |
| Explanatory diagnostics | MMD, continuous loss/bound summaries, condition numbers, KKT residuals, and timing |
| Result artifact | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/proper-score-direct-calibration-{smoke,material}.json` plus a result/reset note beside this plan |
| Nonclaims | Passing validates this controlled statistical design only; it does not validate target posterior samples or open confirmation |

## Implementation And Checks

Add a horizon-specific authenticated proper-score loss and a dual-decision API
without changing the existing average-loss behavior. Add a new standalone
runner; do not mutate closed historical runners. The runner accepts only
`smoke` and `material`, generates controlled laws internally, refuses receipt
overwrite, records strict JSON, source hashes, Git/environment/device/XLA/TF32
provenance, seeds, wall time, and trust basis, and exposes no HMC, retained
archive, model-file, or confirmation input.

Focused CPU-hidden checks cover threshold-anchor separation, the impossibility
of average-only classification, horizon-loss matrices, dual decision branches,
same-region dimension/radius, growing-HAC and zero-ridge enforcement, exact
confidence-region extrema, family roles/truth, exact-binomial simultaneous
allocation, sequential stopping, seed separation, prohibited input text,
strict receipts, and smoke/material claim boundaries.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_predictive_equivalence_principled_repair.py \
  tests/test_ssl_lstm_neutra_proper_score_direct_calibration.py
```

Then run one trusted GPU/XLA smoke under 600 seconds. Only a passing smoke and
focused review authorize the material command. Material execution has a
9,000-second runner cap plus 60 seconds cancellation margin, at most `2.5`
GPU-hours. It may stop after the first rung; it must stop after the second.

No HMC, NeuTra training, private retained archive, G/H forecast suffix,
confirmation result, network call, package install, model-file change, or
default-policy change is authorized.

## Pre-Mortem

- Misleading pass from optimistic synthetic laws: the result is restricted to
  declared controlled laws and cannot promote G/H.
- Failure caused by tuning rather than the idea: `kappa_HAC=1.0` is frozen; a
  miss can motivate a separately planned multiplier study but cannot be tuned
  on these outcomes and rerun as if prospective.
- Local discrepancies diluted by averaging: the horizonwise primary veto
  repairs this by construction.
- Separate intervals presented as joint proof: all loss bounds use one joint
  ellipsoid, and all Monte Carlo operating claims use a simultaneous exact
  binomial allocation.
- Invalid HAC outcomes hidden as inconclusive: invalid probability is a
  co-primary operating target and invalid runs count against coverage and
  required decisions.
- Multiple-look selection: both rungs are prospectively declared and included
  in the 88-claim familywise allocation.

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. The direct implemented procedure is evaluated; fixed-16 and Gaussian-rescaling preflights are not promoted. |
| Proxy promoted? | No. Direct path generation, HAC fitting, ellipsoid construction, and exact loss decisions are repeated end to end. MMD remains explanatory. |
| Missing stop condition? | No. Two rungs, independent seeds, simultaneous look allocation, device/artifact vetoes, and a 2.5 GPU-hour cap are frozen. |
| Unfair comparison? | No method ranking occurs. Every required family uses the same procedure and prospective replication count. |
| Hidden assumptions? | Gaussian controlled laws, AR stationarity/mixing, independent chains/arms, known standardized scales, Wald approximation, thresholds, and local proper-score interpretation are explicit. |
| Stale context? | The plan uses the repaired July 17 APIs and explicitly rejects the impossible one-budget handoff. It reads no retained or confirmation artifact. |
| Environment mismatch? | TensorFlow/TFP `float64`, GPU/XLA default, trusted provenance, and CPU-hidden references match repository policy. |
| Can the artifacts answer the question? | Yes for finite-sample behavior on the declared laws; no target or posterior claim is attempted. |

Audit decision: `PASS_FOR_CONTROLLED_IMPLEMENTATION_AND_SMOKE`. Material GPU
execution remains conditional on focused tests, one focused native review, and
the mechanics-only smoke. A candidate failure is a repair signal; only a true
continuation veto stops the declared ladder.

## Smoke Close And Material Freeze

The first smoke attempt failed before receipt creation because per-replication
`tf.vectorized_map` generation lowered to an XLA-GPU-unsupported variant
`TensorListReserve`. Dense batched Philox generation repaired that issue. The
second attempt exhausted its cap compiling Python-unrolled trust-region loops,
again without a receipt. Replacing those loops with fixed-count XLA
`tf.while_loop`s preserved all equations and iteration counts while repairing
compile feasibility.

The final trusted GPU/XLA smoke passed in `14.8377` seconds. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/proper-score-direct-calibration-smoke.json`,
SHA-256
`7554ac456684e02eb802f60320fb7fda927df5d0159bdbfee29402873159398b`.
Both compiled wrapper surfaces traced once, both families had zero invalid
rows, and the receipt explicitly records that it is not statistical evidence.

Material skeptical-audit disposition: `PASS_MATERIAL_CONTROLLED_LAWS_ONLY`.
The smoke is hard-bound by exact hash; all thresholds, families, seeds,
replications, looks, binomial allocation, and stop rules predate material
outcomes; no historical covariance-rescaling proxy is used; MMD is omitted;
and no HMC, retained archive, or confirmation input exists. A valid 4,096 miss
must continue to 8,192. Any passing rung stops the ladder.

Frozen material command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-proper-score-direct-material-pyc \
CUDA_CACHE_PATH=/tmp/bayesfilter-proper-score-direct-material-cuda \
timeout 9060s /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_proper_score_direct_calibration_2026_07_17.py \
--mode material \
--output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/proper-score-direct-calibration-material.json \
--wall-cap-seconds 9000
```

This authorizes at most 2.5 trusted GPU-hours plus a 60-second cancellation
margin for controlled calibration only. Shared GPU occupancy is recorded and
runtime is descriptive. Stop without retry on receipt drift, non-finite
generation, GPU/XLA failure, corrupted output, or cap exhaustion.

## Material Close

Both rungs completed in `247.5009` seconds. The immutable material receipt has
SHA-256
`fc4781d98a69fbf1002c0f2b76955e023abde4471187c48a6df01f47e712ebf7`.
The candidate failed the required-decision gate, particularly persistent
negligible mean equivalence and local material variance detection. The run had
zero invalid rows and zero false decisions. Post-run review also found that
256 replications made the per-family simultaneous coverage certification
underpowered. Full interpretation and the next boundary are in
`docs/plans/bayesfilter-ssl-lstm-neutra-proper-score-direct-calibration-result-2026-07-17.md`
and the matching reset memo.
