# PP-UKF Statistical Compatibility And Guard Repair Result

Date: 2026-07-21

Decision: `VIABLE_PAIR_SET_WITH_THREE_LOCALLY_SUITABLE_PRIMARIES`

## Outcome

The user's diagnosis was correct. The preceding classifier was too strict: it
required a working interval to be wholly inside `[0.65,0.75]`, although the
tuning decision should reject only when the evidence is statistically outside
the band. It also computed uncertainty over twelve within-replication chain
means instead of the three independently seeded replication means.

The corrected policy uses a two-sided 90% Student-t working interval over three
replication means (`df=2`). It rejects below only when the interval upper bound
is below `0.65`, rejects above only when the interval lower bound is above
`0.75`, and otherwise retains the arm as `provisional_viable`. Compatibility is
a tuning nomination, not proof that the true mean is in-band.

Only `L=3` is statistically rejected. Primaries `L=(5,9,13,18,25)` are
compatible. Fresh exact-epsilon one-hop guards show complete locally suitable
neighborhoods for `L=(13,18,25)`. `L=5` and `L=9` remain compatible primaries
but both of their guards are statistically outside the band.

## Corrected Primary Results

| L | Epsilon | Mean acceptance | 90% replication interval | Corrected disposition | Local guard result |
| ---: | ---: | ---: | --- | --- | --- |
| 3 | 1.162718463 | 0.61412 | `[0.60267,0.62558]` | `needs_lower_epsilon` | not eligible |
| 5 | 0.845201109 | 0.77108 | `[0.74987,0.79229]` | statistically compatible | both guards fail high |
| 9 | 0.983032751 | 0.63306 | `[0.59706,0.66907]` | statistically compatible | both guards fail low |
| 13 | 0.790480953 | 0.75614 | `[0.71054,0.80175]` | statistically compatible | both guards compatible |
| 18 | 0.887851484 | 0.66279 | `[0.62677,0.69880]` | statistically compatible | both guards compatible |
| 25 | 0.836099880 | 0.66630 | `[0.64487,0.68773]` | statistically compatible | boundary guard compatible |

The point estimates for `L=5` and `L=13` are above `0.75`, but their
replication-level intervals overlap the band and therefore do not support
statistical rejection. Conversely, `L=3` remains below at both 90% and 95%
working levels; its rejection is not an artifact of choosing 90%.

## Fresh Guard Results

| Parent L | Guard L | Exact inherited epsilon | Mean acceptance | 90% replication interval | Disposition |
| ---: | ---: | ---: | ---: | --- | --- |
| 5 | 4 | 0.845201109 | 0.82599 | `[0.77692,0.87507]` | `needs_higher_epsilon` |
| 5 | 6 | 0.845201109 | 0.76484 | `[0.75459,0.77509]` | `needs_higher_epsilon` |
| 9 | 8 | 0.983032751 | 0.60000 | `[0.56399,0.63602]` | `needs_lower_epsilon` |
| 9 | 10 | 0.983032751 | 0.59438 | `[0.56263,0.62613]` | `needs_lower_epsilon` |
| 13 | 12 | 0.790480953 | 0.77714 | `[0.71699,0.83729]` | statistically compatible |
| 13 | 14 | 0.790480953 | 0.75011 | `[0.71674,0.78348]` | statistically compatible |
| 18 | 17 | 0.887851484 | 0.66651 | `[0.61837,0.71466]` | statistically compatible |
| 18 | 19 | 0.887851484 | 0.66261 | `[0.63353,0.69169]` | statistically compatible |
| 25 | 24 | 0.836099880 | 0.64698 | `[0.61011,0.68385]` | statistically compatible |

`L=25` is the upper boundary of the reviewed domain, so `L=24` is its only
admissible one-hop guard. No recursive guard expansion or guard retuning was
performed.

## Barrier And Resource Results

| Gate | Result |
| --- | --- |
| Corrected primary barrier | Passed, `6/6`; five compatible and one statistically below |
| Statistical unit | Passed: three fresh seeded replication means, each aggregating four chain means |
| Parent reconstruction | Passed, `5/5`; exact epsilon bits and calibrated-state hashes matched |
| Exact-epsilon guard barrier | Passed, `9/9`; no retuning and no failures |
| Compatible guard count | `5/9` |
| Locally suitable primary neighborhoods | `L=13`, `L=18`, and boundary `L=25` |
| Hard target/health veto | None in any primary or guard |
| Native divergence | Not exposed by the TFP HMC kernel; zero divergence is not claimed |
| GPU/XLA/memory policy | GPU and XLA active; float64; TF32 setting recorded; memory growth verified |
| Guard repair wall time | `3,818.798120 s` |
| Cumulative charged time | `13,750.560450 s` (`3.8196 h`) |
| Campaign ceiling / headroom | `14,400 s` / `649.439550 s` |
| Retained sampling | Not launched |

The prospective continuation projection was `3,584.832002 s`; actual time was
about `234 s` higher but remained within the unchanged total campaign ceiling.
No budget increase occurred.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject only `L=3` | Its 90% interval is wholly below `0.65` | Statistical lower-side rejection; no target-health failure | Three replications remain a small sample | Do not carry its epsilon forward | No rejection of PP-UKF |
| Retain `L=5,9,13,18,25` as compatible primaries | Every interval overlaps `[0.65,0.75]` | No hard veto | Overlap is compatibility, not in-band proof | Use guard results to assess local suitability | No ranking among primaries |
| Reject local suitability for `L=5,9` | Both exact-epsilon neighbors are statistically outside | Guard-screen veto only | Independently retuned neighbors answer a different question | Do not use these primary neighborhoods for the next handoff | Their primaries are not statistically rejected |
| Preserve `L=13,18,25` as locally suitable | Every admissible exact-epsilon neighbor is compatible | No primary or guard veto | Wide `df=2` intervals and no convergence evidence | These are eligible for a separately planned frozen-kernel validation | No retained-sampling authorization or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto evidence | `L=3` statistically below; guards `L=4,6` above and `L=8,10` below; no implementation/target-health veto |
| Statistically compatible candidates | Primaries `L=5,9,13,18,25`; guards `L=12,14,17,19,24` |
| Statistically supported ranking | None; `L=13,18,25` are not ranked |
| Descriptive-only differences | Point means, epsilon values, interval widths, and runtime |
| Default readiness | Not established |
| Next evidence needed | Fresh frozen-kernel validation of the three locally suitable primaries, with convergence and posterior/reference gates declared before execution |

## Evidence Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Passed corrected classifier tests, exact parent reconstruction, complete barriers, immutable source lineage, and terminal artifacts |
| Numerical/sampler validity | Three primary neighborhoods survive bounded tuning compatibility; convergence and native divergence remain unavailable |
| Scientific interpretation | Evidence supports viable tuning candidates only; it neither ranks them nor establishes posterior correctness or PP-UKF scientific validity |

## Run Manifest

- Git commit at execution: `0fff464ab456b72a010007e552c1e2d761624afe`
  with the existing dirty worktree preserved.
- Python: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python 3.11.14.
- TensorFlow: 2.19.1; NVIDIA GeForce RTX 4080 SUPER; XLA enabled;
  float64; TensorFlow memory growth verified.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Artifact root:
  `docs/plans/artifacts/bayesfilter-pp-ukf-statistical-compatibility-guard-repair-20260721/attempt-01/`.
- Source primary artifact is hash-bound in `private_result.json`; all source
  primary final screens were reclassified without rerunning them.

Command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-pp-ukf-stat-pycache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_pp_ukf_statistical_compatibility_guard_repair_20260721.py \
  --output-root \
  docs/plans/artifacts/bayesfilter-pp-ukf-statistical-compatibility-guard-repair-20260721/attempt-01 \
  --frozen-transport \
  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json
```

## Post-Run Red Team

The strongest alternative explanation is that three replications yield wide
and assumption-sensitive intervals, allowing noisy arms to remain compatible.
That is intentional under the user-specified rejection rule, but it means
compatibility is weaker than proof. The fresh guard pattern adds discrimination:
`L=5` and `L=9` fail decisively on both sides, while all admissible guards for
`L=13,18,25` overlap the target band.

The weakest evidence is the `df=2` Student-t working model and the absence of
native divergence and convergence diagnostics. Fresh frozen-kernel validation
could overturn any of the three nominations. Treating interval overlap as a
ranking, using these tuning draws as posterior samples, or declaring all five
compatible primaries equally suitable would not be supported.
