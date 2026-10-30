# Physical-proposal repair: execution result

20 October 2026. Completed. Both trained guides fail the heuristic comparison:
their simulated-label moment losses exceed the recursive Gaussian reference
in both observation regimes, with positive pointwise paired confidence
intervals. They also have higher pooled loss than the conditional-only
proposal. The known collapse is repaired, but improved moment estimation by
the trained augmentation is not established. Governing plan:
`../../ledh-flow-aware-proposal-repair-20261020.md`.

## Question and implementation

Can a physical current-observation proposal prevent the diagnosed LEDH
importance-weight collapse, and do simulated-moment-trained cSMC/KDM future
messages improve independent moment risk beyond that simple repair?

The optional implementation uses the existing shared LEDH executor and
analytical recursion. It retains the UKF covariance lifecycle, dual-cap GenUT,
Contract E--Chol, OT(1,5), terminal balance32 and K=N64. A fixed beta=.5 selects
one of the baseline LEDH and physical Gaussian components, using independent
uniform labels and the full mixture density. Beta=1 is a flow-cancellation
comparator. No canonical or production default was changed.

The reported moment loss uses simulated terminal first/raw-second-moment
labels. Score labels and posterior-moment oracles are not used for training
or selection. Correct density and finite-program derivatives do not establish
equality to the exact model score.

## Engineering evidence

- 43 distinct focused CPU tests pass:37 in `cpu-tests03.log` and14 in
  `cpu-tests04.log`, with8 overlapping tests. CPU-only devices were explicitly
  hidden. Coverage includes nonlinear conditional-moment derivatives, full
  mixture density, beta=0 parity, beta=1 cancellation, dynamic cSMC/KDM
  consumers, shared-executor wiring, invalid uniforms and reporting vetoes.
- `integration-01/result.json`: all four physical-proposal arms pass both
  fixtures, including the known failed plain-LEDH fixture. Plain LEDH remains
  invalid on that exposed fixture. All12 healthy-fixture analytical/central-FD
  comparisons pass; maximum relative discrepancy8.458709342e-8. These are
  engineering checks, not independent efficacy observations.
- `math-code-audit.md` records the mathematical target, checked derivations,
  executable call chain and MathDevMCP limitations. Matrix checks returned
  inconclusive; lexical code comparison is not a semantic proof. The explicit
  positive-definiteness assumption requested by the document audit was added.
- The98-page PDF builds without warnings. New proofs, derivatives, pseudocode,
  result table and interpretation were inspected after rendering; all30 table
  cells match the structured results. The protected supervision text and154
  main-document mathematical blocks remain intact. See
  `document-validation.json` for the PDF checksum and `math-code-audit.md` for
  the audit limitations. The manuscript is
  `docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf`, section15.11.

## Independent evaluation

The complete18-run training ladder finished before selection was frozen at
2079.719240449 seconds. cSMC selects pooled coefficients, optimizer seed
263025002, step.15 and the24-update extension after its8-update fit
(32 cumulative updates; validation loss.9163468810739055). KDM
selects time-specific coefficients, the same seed and step, and8 updates
(validation loss.9314235873970942). Selection uses validation moment loss only;
`train-01/frozen-selection.json` was written before holdout access.

The independent comparison uses32 fresh datasets with two particle
replications each. It includes plain LEDH, the conditional-only comparator,
the unguided half-mixture, untrained and trained guide arms, and three cheap
Gaussian moment references. Low/high-observation strata use a threshold frozen
from calibration. Paired uncertainty clusters both replications by dataset;
bootstrap intervals are pointwise, not simultaneous.

All544 records are complete:448 filter evaluations and96 deterministic
heuristic evaluations. Every arm has zero invalid records. In particular,
plain LEDH is valid on this fresh sample; the exposed regression fixture does
not establish a failure-rate reduction on new data. Each observation regime
contains16 datasets. The following are descriptive means of calibrated
simulated-label loss, not direct oracle moment errors. Lower is favorable;
the common irreducible label noise cancels in population risk differences.

| Arm | All32 | Low16 | High16 |
|---|---:|---:|---:|
| Plain LEDH | .823548 | .920918 | .726179 |
| Conditional only, beta=1 | .818726 | .951452 | .685999 |
| Half-mixture, no future guide | .993677 | 1.198801 | .788553 |
| cSMC, untrained adapter | .954779 | 1.145529 | .764029 |
| cSMC, trained adapter | .971912 | 1.189299 | .754524 |
| KDM, untrained adapter | .947899 | 1.154907 | .740890 |
| KDM, trained adapter | .951518 | 1.164091 | .738945 |
| Prediction only | 1.325386 | 1.648705 | 1.002068 |
| Observation only | .965693 | 1.208127 | .723259 |
| Recursive Gaussian | .626742 | .760924 | .492559 |

Paired differences below are trained-arm loss minus comparator loss, with
2000-replicate dataset-cluster percentile95% intervals. These are predeclared
pointwise contrasts, not a simultaneous ranking of all methods.

| Comparator | cSMC difference [95% interval] | KDM difference [95% interval] |
|---|---:|---:|
| Plain LEDH | .148363 [-.025425,.345555] | .127970 [-.054100,.340968] |
| Conditional only | .153186 [.066461,.251363] | .132792 [.041363,.247269] |
| Half-mixture, no future guide | -.021765 [-.057102,.013251] | -.042158 [-.092305,.001876] |
| Own untrained adapter | .017133 [-.027241,.063483] | .003620 [-.004336,.013151] |
| Prediction only | -.353474 [-.554468,-.188686] | -.373868 [-.567048,-.218398] |
| Observation only | .006219 [-.104204,.121457] | -.014174 [-.122585,.092270] |
| Recursive Gaussian | .345170 [.200724,.509934] | .324777 [.181807,.491879] |

The recursive-Gaussian losses are lower in both strata: trained cSMC excess
loss is.428375 [.170771,.756844] in low observations and.261966
[.149226,.392242] in high observations; KDM excess loss is.403167
[.138101,.748221] and.246387 [.146892,.367340], respectively. Both
machine-readable heuristic-dominance verdicts are false. Additional small
observation-only underperformance flags are descriptive vetoes; their intervals
cross zero and do not establish a ranking. Full42 contrasts are retained in
`train-01/result.json`.

## Decisions and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain optional physical-proposal implementation | Density, total-tangent, wiring and exposed-collapse checks pass | No engineering veto on checked cases | Rare-event reliability outside the fixtures | Preserve as an optional diagnostic mechanism | Canonical admission, default safety or lower new-data failure probability |
| Do not promote trained cSMC or KDM | No supported gain over own untrained or no-future half-mixture controls | Both fail recursive-Gaussian comparison in both strata | Small search, finite capacity and scope; pointwise intervals | Diagnose component allocation and guide conditioning separately | The simulated-label objective is wrong, or the research direction is rejected |
| Close this bounded pilot | Full protocol and holdout complete | No continuation veto fired | No downstream score comparison | Preserve holdout; use fresh partitions for any subsequent tuning | Better model-score accuracy or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No invalid fresh-data records; the exposed plain-LEDH fixture still fails, while its four physical repairs pass |
| Statistically supported ranking | Pointwise pooled comparisons support higher trained-guide loss than conditional-only and recursive-Gaussian controls; both strata support the recursive-Gaussian deficit |
| Descriptive-only differences | Mean ordering of cSMC versus KDM; tiny plain versus conditional difference; observed changes versus untrained adapters; apparent improvements versus the no-future half-mixture |
| Default-readiness | No. Both learned arms fail promotion criteria; FP64 diagnostic evidence does not admit the default TF32 route |
| Next evidence needed | A mechanism diagnostic before another training campaign, followed by fresh calibration/validation/holdout and downstream score evidence for any score claim |

Engineering correctness, numerical validity and efficacy give different
answers. The finite-program derivative is correct on the checked fixtures;
all fresh runs are numerically valid; the learned methods fail the specified
moment comparison. Nothing in the squared-error identity guarantees that a
small optimizer search finds a useful adapter. The independent trained-versus-
untrained contrasts do not support a training benefit, even though validation
selected lower loss. This is evidence against these trained candidates, not
against simulated-label moment training in principle.

The no-future half-mixture already has descriptively larger loss than the
conditional-only proposal. That motivates checking mixture allocation and
finite-particle/reset effects before increasing guide capacity or training
budget. It does not identify which mechanism caused the gap: the planned
primary contrasts do not isolate those effects. The next smallest useful
artifact is a fixed-cloud component decomposition, recording conditional
first/raw-second-moment estimates, component weights and pre/post-reset
moments. It must remain explanatory and cannot tune beta on this holdout.
The conditional-only comparator remains viable as a diagnostic control;
its mean near plain LEDH does not establish superiority.

Post-run red team: the strongest alternative explanation is insufficient
adapter search or a poorly chosen fixed mixture allocation, rather than a
false learning identity. Thirty-two datasets, one short SIR scope and
pointwise bootstrap intervals are the weakest evidence for generalization.
A fresh, adequately tuned comparison that supports lower moment risk over
the no-future and heuristic controls would overturn the present candidate
decision. No such result is currently available. The completed pilot found
no remaining infrastructure failure to repair, so the reserved retry was
not needed.

## Provenance and resources

Branch `ledh-refactor-with-policy-fix`, base commit
`4d6915cc66e701c4c7b657e094306d298fdd3e76`, with recorded working-source hashes
in each run manifest. GPU execution uses `tftwogpu`, FP64, TF32 off, XLA on,
verified memory growth and escalated access. This is a scoped diagnostic
exception, not evidence for the default FP32/TF32 route.

Actual GPU commands, from the LEDH worktree:

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.score_study.physical_proposal_diagnostic_campaign --mode integration --budget-seconds 900 --output docs/plans/artifacts/ledh-flow-aware-proposal-repair-20261020/integration-01
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.score_study.physical_proposal_diagnostic_campaign --mode train --budget-seconds 5100 --output docs/plans/artifacts/ledh-flow-aware-proposal-repair-20261020/train-01
```

Integration charged115.110395934 GPU-process seconds; training and evaluation
charged2501.845760758 seconds for2474 endpoint calls. Stage charge is
2616.956156692 seconds over two launches;4183.440736201 stage seconds and
24245.397855901 parent-campaign seconds remain. The unused localized-repair
allowance is still bounded by800 seconds and one launch; unused training time
does not authorize an unplanned scientific sweep. All eight recorded source
hashes still match after execution. Peak TensorFlow allocator use was
122725632 bytes; memory growth was verified on the RTX4080SUPER.
