# iAPF and MDPF/KDM-IWSG replication master program

Updated: 28 September 2026, Hong Kong.
**Both scientific replications are INCOMPLETE. The first bounded repair tranche
has finished; R4c finite-fitter/controller repair is now active.**

This is the authoritative entry point for the two replications. The detailed
[whole repair plan](iapf-mdpf-whole-repair-plan-20260928.md) supplies the scientific
contracts and remaining phases. The executable dispatcher is
[run_replication_master.py](../../scripts/run_replication_master.py).
Its status reports process state, scientific conclusions and resource balance
separately. A completed subprocess never makes a replication complete.

The current [result](artifacts/iapf-mdpf-whole-repair-20260928-01/result.md),
[self-review](artifacts/iapf-mdpf-whole-repair-20260928-01/master-review.md) and
[checkpoint](iapf-kdm-replication-checkpoint.md) supersede the active-status
paragraphs in the [preserved master history](iapf-kdm-replication-master-program-history-before-20260928-repair-close.md).
Historical evidence remains available; old next-step blocks are not current
instructions.

## The two methods and completion scope

**iAPF:** Guarniero, Johansen and Lee, The Iterated Auxiliary Particle Filter,
JASA 112(520), 1636–1647, DOI 10.1080/01621459.2016.1222291.
Replicate the paper's twisted particle filter, finite backward Gaussian fitting,
positive floor, ESS resampling and adaptive controller, then the declared
linear-Gaussian experiment: d=5, 10, 20, 40, 80; T=100; and the source's
1,000 algorithm repetitions on fixed generated datasets. Use Kalman likelihoods
and the stated BPF/fully adapted comparators. Account for fitting/adaptation
cost. The publisher PDF is [stored locally](../../.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf).

The original executable fitting recipe has not been recovered. The final
paper and thesis support a finite-BFGS reconstruction family; the two tested
recipes are explicit hypotheses. Neither may be described as the recovered
author implementation.

**MDPF/KDM-IWSG:** Younis and Sudderth, Differentiable and Stable Long-Range
Tracking of Multiple Posterior Modes, NeurIPS 2023. Replicate kernel-density
mixture filtering and importance-weighted sample-gradient training: the actual
Appendix B parameter-learning experiment, and the declared bearings-tracking
protocol, objectives/refinement, comparators and independent repetitions.
Author code is [local](../../.localresources/code/younis-mdpf-neurips-2023),
commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.

The existing full author-source NLL trial and evaluation remain valid within
their recorded limits. One trained checkpoint and saved-output analysis do
not complete the learning/comparator/repetition scope. The source's nonlinear
toy dataset is not Appendix B's controlled linear-Gaussian/bimodal-observation
model.

These replications precede extensions. GPU port, score/HMC readiness and
scientific replication are distinct conclusions; none follows just from
another's success.

The owner has authorized continued gap closure. The active next phase is
[R4c finite fitter/controller parity](artifacts/iapf-mdpf-whole-repair-20260928-01/finite-port-spec.md),
with explicit source, numerical and budget limits.

## Current phases and evidence

| Phase | State | Checked evidence / exact remaining work |
| --- | --- | --- |
| R0: reconcile sources and recipes | COMPLETE for explicit reconstruction scope | Final iAPF paper/thesis reconciled; original executable settings still unrecovered. See execution-spec.md below |
| R1: executable master | COMPLETE for implemented tranche | Exact commands, preflight, dependencies, manifests, snapshots, timeouts, locks, recovery and budgets; 12 regression tests pass |
| R2: independent iAPF finite-fit diagnostic | COMPLETE; replication open | 20 controls pass; A_raw 8/8 and B_scaled 5/8 adaptive runs complete; three rejected amplitude endpoints; no ranking |
| R3a: saved MDPF source and conditional screen | COMPLETE; replication open | 5,000 trajectories, source metric parity and three point heuristics in 12 strata pass; normalized-density gate remains open |
| R3b: MDPF learning and density comparators | NOT IMPLEMENTED / specification required | Freeze actual Appendix B coefficients/initialization or label their reconstruction; implement its learning experiment and matched normalized UKF/BPF density comparisons |
| R4a: paper-target iAPF frozen-twist values | COMPLETE within tested FP64 scope | All ten paired cases pass on CPU, GPU graph and GPU/XLA; one trace per signature |
| R4b: legacy graph localization | COMPLETE diagnostic | Actual legacy endpoint has wrong paper-target model/time/resampling conventions; independent graph optimizer interaction reproduced in fresh processes |
| Remaining R4: complete TF/TFP consumers | NOT IMPLEMENTED | Finite fitter, adaptive controller, actual iAPF consumer and relevant score checks; batch-native MDPF learning/training consumers |
| R5: untouched scoped replication | NOT IMPLEMENTED | Complete source experiment cells, conditional baselines, predeclared uncertainty/margins and adequate full-suite resources |
| R6: terminal whole-program review | NOT RUN | Requires R3b, complete R4 and R5 evidence; this tranche's review does not substitute |

[Execution specification](artifacts/iapf-mdpf-whole-repair-20260928-01/execution-spec.md);
[MDPF source/screen specification](artifacts/iapf-mdpf-whole-repair-20260928-01/mdpf-source-and-screen-spec.md);
[value-port specification](artifacts/iapf-mdpf-whole-repair-20260928-01/value-port-spec.md);
[legacy graph specification](artifacts/iapf-mdpf-whole-repair-20260928-01/legacy-graph-spec.md).

## Interpretation and required repairs

The old iAPF consumer is wrong relative to this paper-replication target:
it uses a different six-parameter model, an extra initial transition and
compulsory resampling. Existing component parity does not repair that mismatch.
The new explicit-matrix TensorFlow value kernel computes the paper-target
recursion and matches the R reference on the tested fixtures.

The old graph also returns empty tensors on a tiny deterministic example.
Fresh-process diagnostics recover correct outputs by disabling the meta
optimizer or individual arithmetic/dependency/constant-folding optimizations.
This localizes a graph-optimization interaction in the tested self-built
TensorFlow version; the unique faulty transformation remains unidentified.
It is not evidence that XLA is generally broken. The new kernel passes with
default optimizations and XLA. No global optimizer settings were changed.

The finite-fitting objective still admits density/amplitude escape.
Stationarity is an explanatory diagnostic for these finite-BFGS recipes,
not a universal filter-validity veto. A_raw is viable on eight pilot runs;
B_scaled has three observed endpoint failures. Neither is statistically
established as preferable. The original finite recipe and regularizer remain
uncertain.

The **next implementation task** is to carry the frozen finite-BFGS objective,
coordinates, initializer, endpoint guards and floor into a shared TensorFlow
fitter, followed by the adaptive controller and actual paper-target consumer.
Before numerical execution, freeze a bounded subphase contract and add real
commands to the dispatcher. Preserve full failing endpoints and step
diagnostics. Check objective/gradient and fitted-twist behavior, then full
filter/controller behavior with matched inputs or justified distributional
checks where optimizer branches diverge. Do not substitute the old bounded,
profiled fit or reintroduce stationarity as an unexamined universal requirement.

In parallel as an independent workstream, specify the actual Appendix B model
and implement MDPF density comparators under the source information budget.
This statement does not authorize spawning additional agents. There is no
need to wait for iAPF to settle before doing valid MDPF work.

An implementation/specification gap is recorded honestly rather than supplied
with a pretend launch command. Remaining phases cannot be executed by the
current dispatcher until their actual implementations and contracts exist.

## Commands, resources and permission handling

Run from /home/chakwong/BayesFilter:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py status
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py validate
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py run-ready
```

run-next executes only the next eligible implemented phase. recover reconciles
interrupted manifests without inventing a successful exit. run-ready stops
after a failed child so that its failure is investigated before retry. It
cannot launch unimplemented phases or rerun completed phases silently.

The sixteen recorded attempts consumed 642.551210915 new CPU seconds and
19.785990647 new GPU seconds, including failed launches and compilation.
Including inherited charges, remaining resources are **948.351743483 CPU
seconds and 36.561867355 GPU hours**. status is the live authority. The first
tranche's launch cap was revised from 12 to 16 for documented local repairs,
without increasing total compute; all sixteen slots are now accounted for.
A subsequent bounded phase may be added under existing authorization and the
remaining budget after its scientific/engineering contract is recorded.

The whole repair plan's 36.56-hour GPU allocation is a ceiling, not a runtime
forecast or new allowance. Full repeated MDPF replication exceeds that balance
at the observed 8.589 hours per source trial. Do not reduce repeat counts and
retain a full-replication claim.

Use the already approved direct interpreter/script invocations. GPU probes
and jobs require escalated access and verified memory growth before device
initialization; CPU diagnostics hide GPUs explicitly. Stable direct commands
avoid shell substitutions that prevent allow-list matching. No permission-rule
changes, broad shell approvals, package changes or renewed approval for local
repairs were needed. External messaging, expanded paid compute, broad
environment changes or a material change in scientific target remain separate
boundaries.

## Stop conditions and review verdict

Invalid target/comparator/data, unusable evidence, missing required numerical
diagnostics or exhausted budget stop the affected experiment. A candidate
failure triggers its planned repair; it does not reject the method's research
direction. Conditional heuristic underperformance vetoes promotion. Small
pilot metrics and one-checkpoint evaluations remain descriptive.

Self-review accepts this implemented tranche and its limited conclusions.
Twelve executable dispatcher checks pass. Remaining source assumptions,
missing density comparisons, missing training-seed inference and missing
fitter/controller/score consumers are explicit gaps. The full master is
**not finished**, and no scientific/default/HMC promotion is issued.
