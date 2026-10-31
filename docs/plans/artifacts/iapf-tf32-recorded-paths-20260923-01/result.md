# Phase20 result: complete traces validate the mechanism; six precision vetoes remain

The fresh d2 comparison completed 36,864 likelihood evaluations with a
contemporaneous, numerically neutral trace for every evaluation. Eighteen of
24 comparisons meet the unchanged precision criterion. All24 simultaneous
approximate99% intervals lie inside the2% margin; six comparisons remain
vetoed because their largest centered squared contribution exceeds20%.
Those six are the stress guide on fresh datasets1 and2, and BPF on fresh
dataset1, each compared with FP64 and ordinary FP32. Their largest shares are
21.40%–27.91%. This is insufficient precision evidence, not evidence of bias.

The experiment uses the local six-parameter linear Gaussian model, d=o=2,
N64/T5, three fixed datasets and four frozen guides. There are1024 new
particle streams per dataset/guide/mode. Their seed ranges are disjoint from
phase18. No guide, runtime source, default or acceptance criterion changed.
Phase18's broader d1/d5/long-horizon vetoes and phase19's inability to recreate
21 old d2 TF32 executions remain recorded in their original results.

## What the recorded paths establish

The review checks160 source hashes,87 output hashes,211 fixture hashes and
all36,864 trace records. Every trace is bitwise neutral for the value,
fixed-label score and cloud. All original18 independent R controls pass in
both FP64 and ordinary FP32. TF32 still passes7/18 strict path/score controls;
the other11 have not been waived. GPU placement, XLA, memory growth and the
RTX4080 identity are recorded and checked. Eleven independent Kalman/joint
Gaussian checks pass in the frozen fixture.

Among the72 outcome-selected pairs used for explanation,62 first diverge at
an ancestor choice and10 have no changed discrete decision. Maximum cloud
error is0.001211 before the first different choice and1.614455 afterward.
These selected counts are not population frequencies. The original execution
is available for every selected case; no later compiled replay is substituted.
This phase does not independently regenerate the uniforms to reconstruct
the ancestor indices; that check was performed in phase19's replay diagnostic.

For example, BPF on fresh dataset1, replicate744, has normalized likelihood
2.277203 in FP64 and4.018148 in TF32. The first CDF discrepancy is at most
0.00029607, but selects one different ancestor after the first observation.
Cloud error grows from0.001089 to1.193965. Conditional on those two recorded
CDFs, the exact23-bit uniform-lattice disagreement probability for one draw is
0.00474119, or0.3034 expected changed ancestors among64. This supports the
rounding-to-resampling amplification mechanism; it does not prove a systematic
change in the estimator's mean. The minimum effective particle counts are
about9.19 in this example and about1.30 in an influential stress-guide case.

The stress guide has higher observed squared likelihood error than each of
the three cheap guides in all27 conditional heuristic comparisons. This
vetoes a guide-quality promotion. These descriptive MSE differences establish
no statistical ranking; the deliberately random stress guide is not a trained
iAPF proposal.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept complete-trace observability repair | All36,864 records complete and neutral | Integrity and independent-reference controls pass | Other compiler executions may differ | Preserve actual traces in further comparisons | Bitwise reproducibility across compilations |
| Conditional precision remains partial |18/24 comparisons pass; all intervals inside2% | Six tail-share vetoes;11 old strict TF32 failures retained | Rare paired errors at1024 streams | One fixed larger fresh confirmation with the same criterion | TF32 default readiness or absence of bias |
| Retain stress-guide quality rejection |27/27 observed heuristic losses | Quality promotion veto | Descriptive MSE, fixed observations | Retain every guide and dataset in confirmation | General iAPF failure or paper replication |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical/reference/artifact controls pass; declared precision and strict-path vetoes remain |
| Statistically supported ranking | None;18 conditional comparisons meet the predeclared approximate equivalence criterion |
| Descriptive-only differences | MSE, maxima, ESS and selected-path frequencies |
| Default-readiness | Not established; numerical runtime and defaults unchanged |
| Next evidence needed | One predeclared fresh4096-stream comparison; no repeated enlargement until a pass |

## Execution and budget

The escalated command was `diagnose_iapf_tf32_recorded_paths.py --execute`
through the tftwogpu interpreter. FP64, FP32 and TF32 ran sequentially, followed
by the independent R summary. CPU-only terminal inspection used
`CUDA_VISIBLE_DEVICES=-1 python3 docs/benchmarks/review_iapf_tf32_recorded_paths.py`.
The four attempt manifests preserve exact commands, Git89065bc6, source
closure, input hashes, seeds, environment, HLO and allocator evidence.
`completed.json` identifies authoritative results; `review-01/` contains
verification, decisions, selected streams and localization. No run failed.

Charges are1181.892213 GPU process seconds and19.520040 CPU seconds,
including preflight and terminal review. Remaining campaign balance is
64886.266828 CPU seconds and168699.170118 GPU seconds, with no reservations.
The reviewed phase stayed within its1800/120-second GPU/CPU limits.

Post-run skeptical review: finite-sample uncertainty, guide inefficiency and
compiler-sensitive floating arithmetic remain plausible explanations. The
complete traces remove the lost-executable gap for this experiment. They do
not justify pooling selected cases, treating small log-likelihood error as
unbiasedness, or promoting fixed-label derivatives to marginal model scores.
The evidence supports one fixed larger confirmation, retaining all12 cells
and all old failures. Further automatic sample growth after that confirmation
is not part of the next plan.
