# PP-UKF State-Continuing Epsilon Repair Result

Date: 2026-07-21

Decision: `SUPERSEDED_STATISTICAL_CLASSIFICATION`

Correction: the numerical primary screens in this file remain valid, but the
strict-containment classifier and its no-admissible-pair conclusion are
superseded. The corrected rule uses three independently seeded replication
means as statistical units and rejects only when the 90% working interval is
wholly outside `[0.65,0.75]`. Under that rule only `L=3` is rejected; the other
five primaries are statistically compatible. Fresh required guard results are
recorded in
`docs/plans/bayesfilter-pp-ukf-statistical-compatibility-and-guard-repair-result-2026-07-21.md`.

## Outcome

The tuning implementation and admission bug are repaired. The campaign used
the declared target acceptance probability `0.70`, continued each chain state
from adaptation through calibration, applied at most three bounded epsilon
repairs, and classified only fresh replicated final evidence. All six primary
arms completed without a target-health, identity, resource, or artifact veto.

No primary passed the strict admission rule. Therefore no `L-1` or `L+1`
same-epsilon guard was eligible, no candidate was selected, and no retained
sampling was launched. This is a valid negative tuning result for this bounded
repair protocol. It is not an implementation failure and does not reject
PP-UKF as a method.

The repair removed the suspicious universal-high-acceptance pattern. Final
means now occurred on both sides of the practical region, which is consistent
with epsilon materially affecting the kernel. The remaining problem is that
short calibration evidence did not reliably predict the three-replication
final screen, and the three-repair cap did not center every arm closely enough.

## Complete Results

| L | Final epsilon | Final calibration mean | Calibration reached `[0.68,0.72]` | Fresh final mean | Working interval | Disposition |
| ---: | ---: | ---: | :---: | ---: | --- | --- |
| 3 | 1.162718463 | 0.59550 | no | 0.61412 | `[0.59057,0.63767]` | `needs_lower_epsilon` |
| 5 | 0.845201109 | 0.71512 | yes | 0.77108 | `[0.75760,0.78456]` | `needs_higher_epsilon` |
| 9 | 0.983032751 | 0.69066 | yes | 0.63306 | `[0.60894,0.65719]` | `needs_lower_epsilon` |
| 13 | 0.790480953 | 0.71205 | yes | 0.75614 | `[0.72681,0.78548]` | `needs_higher_epsilon` |
| 18 | 0.887851484 | 0.69337 | yes | 0.66279 | `[0.64722,0.67835]` | `unresolved_budget` |
| 25 | 0.836099880 | 0.72579 | no | 0.66630 | `[0.63973,0.69288]` | `unresolved_budget` |

The working intervals are bounded tuning heuristics over twelve chain-run
means, not confidence intervals supporting a stochastic ranking. The point
means for `L=18` and `L=25` were inside `[0.65,0.75]`, but their intervals
crossed the lower boundary, so they were correctly not admitted. The point
means for `L=5` and `L=13` were above `0.75`; interval overlap cannot make them
viable under the repaired classifier.

## Calibration Paths

| L | State-continuing epsilon and calibration-mean path |
| ---: | --- |
| 3 | `0.884509903@0.79315 -> 1.061411883@0.72529 -> 1.273694260@0.49847 -> 1.162718463@0.59550` |
| 5 | `0.845201109@0.71512` |
| 9 | `0.819193959@0.75104 -> 0.983032751@0.69066` |
| 13 | `0.790480953@0.71205` |
| 18 | `0.810493809@0.73800 -> 0.972592571@0.50333 -> 0.887851484@0.69337` |
| 25 | `0.729242929@0.77485 -> 0.875091514@0.53833 -> 0.798845604@0.72051 -> 0.836099880@0.72579` |

Calibration was directional evidence only. It did not contribute to admission.
Every adjacent calibration-state signature matched, and each of the three
fresh final replications began from that arm's frozen calibrated-state
signature. Thus the previous restart-from-original-state error did not recur.

## Barrier And Resource Results

| Gate | Result |
| --- | --- |
| Strict classifier | Passed: out-of-band point means are directional failures; boundary-crossing intervals are unresolved |
| Six-primary barrier | Passed, `6/6` complete with no primary failure |
| State continuation | Passed for every adaptation/calibration link and every final-screen start |
| Viable-primary expansion | Zero viable primaries, therefore zero legitimate guards |
| Guard barrier | Complete, `0/0`; no guard epsilon or retuning event exists |
| Target/value/score/status health | No hard rejection in any arm |
| Native divergence | Not exposed by the TFP HMC kernel; zero divergence is not claimed |
| GPU/XLA/memory policy | GPU present, XLA enabled, float64, TF32 setting recorded, memory growth verified |
| Repair-attempt wall time | `2,937.756936 s` |
| Cumulative charged time | `9,931.762330 s` (`2.759 h`) |
| Campaign ceiling / unused budget | `14,400 s` / `4,468.237670 s` |
| Retained sampling | Not launched |

The prospective primary projection was `4,913.569199 s` including the 50%
margin; actual repair time was lower. No campaign-budget increase occurred.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the protocol repair | Correct target, state continuation, epsilon direction, strict classification, and complete fresh-evidence barrier all verified | No engineering or artifact veto | Native divergence remains unavailable | Keep the repaired classifier and driver as the tuning basis | No posterior or convergence claim |
| Admit no PP-UKF pair | No final interval was fully contained in `[0.65,0.75]` | `L=3,5,9,13` failed directionally; `L=18,25` remained unresolved | Final screens differed materially from 32-step calibration means | Any further attempt must use fresh calibration/final partitions and target the calibration-to-final variability | PP-UKF is not rejected |
| Run no guards or sampling | No primary satisfied the prerequisite | Barrier logic passed | Suitability near `L=18,25` remains unresolved | Preserve remaining budget; do not spend it by weakening evidence or reusing final data | No retained-sampling readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No implementation, target-health, finiteness, state-lineage, identity, GPU, resource, or artifact veto fired |
| Viable candidates | None under strict replicated admission |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-L epsilon, calibration means, final means, intervals, and runtime |
| Default readiness | Not established |
| Next evidence needed | A fresh bounded protocol aimed at calibration-to-final stability, with untouched final evidence; do not reuse this final screen for admission |

## Evidence Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Passed focused tests, six-primary barrier, strict classifier, state-signature continuation, target/transport/metric lineage, and terminal artifact checks |
| Numerical/sampler validity | Acceptance targeting remained variable; no primary admitted; native divergence and convergence were not available as evidence |
| Scientific interpretation | The earlier universal-high result was a tuning-protocol artifact; this repair gives no evidence for a viable pair and no evidence against the PP-UKF research direction |

## Run Manifest

- Git commit at execution: `0fff464ab456b72a010007e552c1e2d761624afe`
  with the existing dirty worktree preserved.
- Python: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python 3.11.14.
- TensorFlow: 2.19.1; one logical GPU; XLA enabled; float64; TensorFlow
  memory growth verified before logical-device initialization.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Root seed: `(20260721, 9400)`; per-run seeds are preserved in the private
  artifact.
- Artifact root:
  `docs/plans/artifacts/bayesfilter-pp-ukf-state-continuing-epsilon-repair-20260721/attempt-01/`.

Command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-pp-ukf-repair-pycache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_pp_ukf_state_continuing_epsilon_repair_20260721.py \
  --output-root \
  docs/plans/artifacts/bayesfilter-pp-ukf-state-continuing-epsilon-repair-20260721/attempt-01 \
  --frozen-transport \
  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json
```

## Post-Run Red Team

The strongest alternative explanation for the missing viable pair is not that
epsilon has no effect. The observed calibration paths respond strongly and in
the expected direction. The stronger explanation is that 32 calibration
results have too much state and Monte Carlo variability to locate an epsilon
whose separate three-by-96 screen is reliably centered. `L=5`, `L=9`, and
`L=13` reached the narrow calibration region and then finished outside the
practical region on fresh evidence. `L=18` and `L=25` were descriptively close
but too uncertain for strict admission.

The weakest part of the evidence is the short calibration window and the
working interval's heuristic status. A fresh protocol that produces stable,
replicated calibration-to-final agreement would overturn the present
inconclusive result. Reclassifying boundary-crossing intervals, tuning on these
final screens, or claiming zero divergences would not.
