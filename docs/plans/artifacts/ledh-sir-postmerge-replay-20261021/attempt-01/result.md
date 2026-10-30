# Completed frozen SIR replay after the main merge

21 October 2026. The merge preserves the guided moment results and the previous
negative scientific conclusion in this diagnostic scope. It does not preserve
plain LEDH within the predeclared numerical tolerance, and this replay shows no
speedup. Both lanes completed all 2,496 holdout rows and 432 nonzero analytical
directional checks. Every row is finite and valid; no required validity monitor
was removed. The pre-merge lane reproduces every archived quantity exactly.

The complete machine-readable result is
[comparison-02/comparison.json](comparison-02/comparison.json); the
[generated tables](comparison-02/comparison.md) retain numerical, conditional
heuristic and runtime comparisons. `status=complete` with
`equivalence_passed=false` is the terminal numerical verdict. The analyzer's
exit code 2 records that verdict, not an incomplete campaign.

## Scope and provenance

This is an engineering replay of previously exposed data, with no refitting or
selection. The baseline is the actual 834-module pre-merge working snapshot
based on `4d6915cc66e701c4c7b657e094306d298fdd3e76`, including the uncommitted
guide implementation. The merged 980-module snapshot is based on
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`. Both snapshots include their own
411 auxiliary experiment modules. Neither lane is represented by a clean Git
commit alone. Isolated imports, source stability, frozen input hashes and the
shared custom-op binary hash pass. The original archived study did not retain
the binary hash; exact old-result reproduction is the additional dependency
tie-out, not proof of historical binary identity.

The environment is `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`,
TensorFlow `2.20.0-dev0+selfbuilt`, NVIDIA GeForce RTX 4080 SUPER,
GPU UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`. Both numerical lanes used
escalated GPU access, verified memory growth before logical-device
initialization, FP64, TF32 off and XLA on. SIR has 18 states, three parameters,
T=5 and N=64; OT remains `(1,5)`, terminal balance 32, exact chunk K=64 and
24 flow substeps. All ridge, damping, correction and cap controls remain frozen.
This scope is a diagnostic exception, not the default TF32 production target.

Data version is `canonical_sir_d18_x0_then_transition_observe_v1`. Dataset seeds
are 263043001 through 263043064; each particle arm retains its four original
replicates, particle-seed base 283030000 and reset seed 263040002. Directional
checks use datasets 263043001, 263043004, 263043021 and 263043052, every particle
replicate and all three coordinate directions. Per-row particle seeds and input
hashes are archived. Each lane's `manifest.json` records the exact command,
environment, controls, source/input hashes, plan and result paths. The final
paired files are `before-retry02/` and `after/`; copied rows retain their
original timing origins.

## Numerical result

The eight non-plain particle arms pass every numerical field comparison. Their
largest absolute moment-loss change is 1.101e-12, value change 5.060e-12 and
directional derivative change 1.253e-10. All three cheap heuristics reproduce
exactly. Plain LEDH fails the fixed absolute-plus-relative tolerances:

| Plain LEDH quantity | Maximum absolute change | Failed rows / compared rows |
| --- | ---: | ---: |
| Moment loss | 4.282215e-6 | 37 / 256 |
| Standardized moment estimate | 4.549255e-5 | 90 / 256 |
| Finite-program log value | 1.252784e-4 | 23 / 256 |
| Nonzero analytical directional derivative | 2.394952e-2 | 26 / 48 |

The largest value change is at dataset 263043014, replicate 1:
`-417.0955706818255` becomes `-417.0954454034517`. The largest derivative change
is at dataset 263043021, replicate 2, direction 0:
`1128.444299101828` becomes `1128.420349579072`. These are finite-program
analytical derivatives, not oracle scores. The historical zero-direction
fields remain zero and supply no score-accuracy evidence. All direction-value
invariance checks pass. The observed changes establish numerical non-equivalence
at the declared tolerance; they do not identify which version is more accurate.

The mean plain-arm moment loss changes from 0.765211387 to 0.765211372.
Trained neural KDM remains 0.770636507, its initialization 0.774066276,
adapted cSMC 0.863626197 and adapted RBF 0.872853862. Recursive Gaussian remains
0.533512432. All nine particle arms still have larger observed losses than
recursive Gaussian in every prespecified marginal and joint regime. Every
heuristic-veto assignment is identical before and after the merge.
This retains the previous promotion veto; it does not establish a fresh
population ranking or reject the entire guide research direction. There are
only two nearer-boundary datasets, one in each observation stratum. Those
latent-defined strata are descriptive stress checks; the conditional-label
moment-MSE decomposition does not automatically apply after that stratification.

## Runtime and interpretation

After excluding a pair whenever either call is cold, plain LEDH averages
0.295326 to 0.664222 seconds per call, a 2.249-fold increase. Conditional
Gaussian averages 0.291059 to 0.686908, a 2.360-fold increase. The paired
dataset bootstrap intervals for those time differences are respectively
[0.295501, 0.442610] and [0.315544, 0.474342] seconds. The other seven particle
arms have observed ratios 1.0465 to 1.1236; all seven difference intervals
include zero. There is no measured speedup in this run.

These intervals use 5,000 paired dataset resamples with seed 263050001 and
describe variation in this sequential replay. They cannot remove time-varying
machine-load confounding. The old lane needed a continuation and has 12 cold
calls; the merged lane has six. Cold calls and process origins are preserved
separately. A causal performance regression requires an interleaved comparison
under controlled load. The present timings do not prove that the merge itself
caused the slowdown or that its optimizations are ineffective in other scopes.

Subsequent investigation on21 October identifies the retained flow QR work in
optimized consumer HLO and reproduces the selective slowdown with alternating
timings. See [the runtime follow-up](../../ledh-sir-postmerge-runtime-20261021/attempt-01/result.md).
That report provides the checked mechanism; it does not turn the sequential
ratios here into universal causal estimates or change this replay's results.

## Explanation checked and remaining uncertainty

Source inspection follows `build_backward_guided_ledh_kernel` at
`bayesfilter/highdim/ledh_backward_guide_tf.py:354` to the shared
`_value_and_analytical_score_impl` call at line 365. The paired executable runs
exercise every named arm through this endpoint. They test version parity, not
general conformance of other execution lanes.

The merge regroups weighted polynomial arithmetic in
`bayesfilter/highdim/higher_moment_contract_e.py:167` and delegates the Cholesky
differential at line 116. On the first plain pair, observation-normalizer
differences begin at 4.4e-16 and reach about 1.72e-6 at the last step; cap scales
also separate later while all validity checks remain true. This supports
amplification of arithmetic-order differences as an explanation. It does not
isolate a causal operation or rule out an implementation error. A controlled
source ablation or independent numerical reference would discriminate them.

The static scalar shortcuts in `bayesfilter/highdim/ledh_small_matrix_tf.py:13`
apply only to orders one and two. They therefore do not directly replace this
SIR scope's 18-state and 9-observation matrix solves. This is a dimensional
limitation of that particular optimization, not a runtime attribution.

The strongest alternative explanation for timing is changing machine load;
for numerical drift it is a substantive arithmetic/derivative defect rather
than harmless regrouping. Either would require a focused follow-up. The weakest
scientific evidence remains reuse of an exposed, short-horizon FP64 dataset and
the tiny boundary strata. Independent untouched moment-risk evidence would be
needed to change the existing scientific conclusion.

## Execution, repair and budget

| Launch | Outcome | Charged seconds |
| --- | --- | ---: |
| `before/` | Missing auxiliary dependency; failed before particle evaluation | 35 |
| `before-retry01/` | Process cap; verified rows preserved | 1281.361979 |
| `before-retry02/` | Completed remaining rows; exact archive tie-out | 869.329002 |
| `after/` | Complete without continuation | 2543.222979 |
| GPU contention probe | Read-only diagnostic | 1 |
| Total | Four launches; fifth unused | 4729.913960 |

The recorded charge is 78.832 minutes of the authorized 19,800 seconds,
leaving 15,070.086 seconds unused. The additional four-hour allowance was not
needed. All replay processes have exited; the campaign is complete.

The first report, preserved in `comparison-01/`, failed because its compartment
label was `boundary` while frozen rows use `nearer_boundary`. Correcting the
report label and rejecting absent required regimes changes no numerical row or
group assignment. The focused regression exposed a misplaced test assertion;
after moving it back to its original test, all ten CPU-only checks pass in
0.19 seconds (`analyzer-tests-final-retry01.log`). Both failed reporting/test
logs remain preserved. The final CPU-only command, with GPU devices deliberately
hidden, was:

```sh
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/analyze_sir_postmerge_replay.py --before docs/plans/artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/before-retry02 --after docs/plans/artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/after --output docs/plans/artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/comparison-02
```

Terminal review confirms complete paired grids, exact baseline reproduction,
unchanged inputs/controls, isolated stable sources, no invalid rows, preserved
validity checks, unchanged heuristic vetoes and finite directional comparisons.
The parity failure remains visible and is not reclassified as a validity failure.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Replay complete; reject full numerical equivalence | Plain arm fails; other eight arms pass | No invalid rows; heuristic promotion veto unchanged | Numerical cause unresolved; timing confounded | Isolate plain arithmetic sensitivity and interleave timing before an equivalence or speed claim | No guide promotion, true-score accuracy, canonical admission or broader-scope conclusion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Numerical-equivalence veto for plain LEDH; no new validity veto; existing heuristic veto retained |
| Statistically supported ranking | No new scientific ranking; runtime intervals do not identify a causal merge effect |
| Descriptive-only differences | Moment losses on exposed data; runtime means and version differences in this fixed scope |
| Default-readiness | Not assessed; frozen diagnostic controls are not newly tuned defaults |
| Next evidence needed | Focused numerical attribution, controlled performance replication, and fresh untouched data for any later guide-improvement claim |
