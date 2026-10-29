# Replication gap closure — 6 October 2026

Authority: [master](controlled-smc-kdm-replication-master-program-20260928.md).
The target remains a TensorFlow implementation with no worse numerical behavior
than the released author code. Intrinsic author failures remain acceptable
baseline outcomes. Numerical improvements and original iAPF remain deferred.

The owner subsequently added 48 GPU hours and 48 CPU hours. The live balance is
48.18 GPU hours and 90.06 CPU hours. The earlier resource limit is superseded;
the missing implementation, cost and qualification specifications remain.
[Accepted criteria](controlled-smc-kdm-acceptance-criteria-20261006.md)
now provide explicit practical margins and uncertainty rules. They are not
retroactive changes to failed diagnostic screens.

## Evidence now established

The [DGELSD audit](controlled-smc-dgelsd-arithmetic-result-20261006.md) localizes
the selected Lorenz mismatch to the source reference's divide-and-conquer
bidiagonal least-squares path. Outer scaling, diagonal flooring and splitting do
not explain those cases. The actual reference is author MATLAB-language code
executed in Octave6.4.0/system LAPACK3.10.0, not the author's original MATLAB
installation. Branch matching and a generic TensorFlow SVD are insufficient.
This finding explains a compatibility gap; it supplies neither a tested TF repair
nor a downstream quality ranking.

The [KDM wiring check](controlled-smc-kdm-checkpoint-wiring-result-20261006.md)
establishes the actual loaded checkpoint for all22 active bearings evaluations:
NLL-trained, including tasks labelled MSE. Their local checkpoint files are
absent. Preserve the source configuration for replication and report this naming
ambiguity separately from the paper's intended checkpoint.

The [circular KDM comparison](controlled-smc-kdm-circular-consumer-result-20261006.md)
now checks the actual author resampling, particle-weight and posterior-NLL
consumers. All 157 float64 checks pass; four float32 gradient comparisons fail
the unchanged bound. Broadcasting concentration before the source log-I0
polynomial reduces one discrepancy but does not close the screen. CPU
localization also differs, so XLA alone is not the explanation. The consumer
uses constructed model outputs and fixed author samples; neural training and
circular sampling remain unimplemented.

## Closure sequence and readiness

| Work | Next discriminating action | Acceptance / stop condition | Readiness |
|---|---|---|---|
|Lorenz numerical compatibility|Specify a TF implementation of the source's conditional Cholesky/LU/DGELSD operations, starting with the saved45x45 bidiagonal D&C systems and known-answer controls; alternatively assess the existing algebraically equivalent fit under the independently defined no-worse contract below|Same mathematical objective and source settings; finite/domain invariants; actual policy/forward outputs. A source-call compatibility claim requires executable consumer evidence. Small residual or exact branch choice alone is insufficient|Root cause localized; a complete compatible TF implementation is not yet known/tested|
|No-worse criterion|Freeze downstream metrics, acceptable numerical discrepancy, uncertainty analysis, source-calibration and untouched claim partitions before evaluating candidates for admission|No extra invalid cases where the reference is valid. Any equivalence margin needs numerical/metric justification independent of candidate discrepancies. Inconclusive inference remains inconclusive|Concrete margins are now owner-approved; freeze source-only calibration, precision and execution details before a claim run|
|Sensitivity calibration|Use fresh source seeds and predeclared regimes/directions if assessing the user's1e-8 input-sensitivity hypothesis; record resampling/stopping changes and both signs|This characterizes source numerical sensitivity only. Neither its largest observation nor a single median automatically becomes an accuracy/non-inferiority margin|Mechanism/execution harness exists; budget/sample-size specification still needed|
|Autonomous filter validation|Run source and eligible TF candidate at identical settings on untouched seeds, including the preserved author failure; retain every outcome|Match mathematical quantities and report domain/validity before likelihood/variance. A source failure is accounted for, not repaired. Replaying a supplied failed policy cannot prove autonomous failed-trajectory equivalence|After a candidate and admissible criteria exist|
|KDM circular float32 arithmetic|Inspect the remaining source Normal variance/division and gradient-reduction ordering; run only a bounded experiment that distinguishes a stated hypothesis|Keep the existing bound and fixtures for localization; no clipping, precision promotion or posterior/resampling-density substitution. A raw gradient screen does not establish downstream training quality|Float64 consumers checked; float32 compatibility unresolved after two GPU attempts and CPU localization|
|KDM baseline specification|Use the verified NLL checkpoint paths and dependency graph; verify dataset generation, optimizer update, checkpoint conversion and actual source training consumer|No Cappe-for-AppendixB substitution; no MSE-checkpoint substitution based on task labels; no primitive-gradient-only training claim|Path ambiguity resolved; learning/data checks remain|
|KDM training|Cost component pretraining, MDPF and A-MDPF independently, with declared repetitions and held-out density/state metrics; then run author and TF training|Same information, objective, stages and evaluation. All stages and compute must be allocated before a full launch|Checkpoints missing; new 48-hour GPU grant permits costed continuation, but full author/TF training coverage remains uncosted|
|Other selected source experiments|Recover AppendixB coefficient/initialization/schedule and missing multimodal driver; cost Table1 design; keep source and paper-result scope explicit|Unknown specifications remain unknown. Missing original data permit only an explicitly identified generative-design replication|Unresolved; do not silently reconstruct as exact source|

There are two distinct ways a numerical translation can be supported: reproducing
the reference operations closely enough on fixed inputs, or demonstrating that
an equivalent mathematical implementation satisfies a justified downstream
no-worse criterion. We have not completed either route for autonomous Lorenz
fitting. The audit does not authorize an endless bitwise-matching requirement,
and it does not authorize accepting the existing candidate just because the
reference is ill-conditioned.

The current strict polynomial screen is a localization diagnostic. The six
previous perturbation cases are calibration/debug data, not fresh claim data.
A sensitivity interval answers how the *source changes under perturbed random
inputs*. It does not measure estimator error against truth. A likelihood that
is numerically higher is also not automatically more accurate. These distinctions
must survive any executable criterion. No unexamined percentage tolerance is
introduced in this update.

## Review, resource boundary and execution record

Skeptical review found two material risks: treating source self-sensitivity as
a quality threshold, and selecting a KDM model from its task label. The first
is now governed by the accepted prospective criteria; the second has been resolved by an
executable source consumer audit. Original MATLAB/Octave equivalence is still
unverified. Heuristic superiority is outside this replication target; author
behavior is the comparator.

Earlier on 6 October, the authorized continuation executed three bounded DGELSD reference attempts
(two preserved exact-chain-check failures, one successful amended diagnostic),
then one successful KDM metadata audit. Charges are1.046148 and0.566610 CPU
seconds respectively, zero GPU. The circular continuation then consumed
37.76449583284557 GPU seconds and 5.536095131188631 CPU seconds. The subsequent grant raises the remaining active
allocation to 324216.641335726 CPU seconds and 173464.9919998964 GPU seconds.
Numbered attempts remain 160; the supplementary ledger charges each diagnostic
exactly once. The explicit owner grant is recorded once. No numerical job, package mutation
or numerical default change was made in this criteria update.

All unresolved items above remain incomplete in the machine-readable supervisor.
The current program can report and verify completed evidence; it is not an
unattended end-to-end executable for unspecified training and qualification.
The next meaningful numerical action needs a concrete candidate/criterion and
a bounded experiment specification. Do not spend the remaining GPU time rerunning
already rejected branch-only or generic-SVD comparisons.

Terminal review checks: source/summary hashes, supervisor accounting, preservation
of false comparison verdicts, later-attempt precedence, syntax, and the focused
replication tests. Their execution receipt is saved beside this continuation's
artifacts. These engineering checks do not promote scientific status.

Earlier engineering review completed: 19/19 focused tests passed; all preserved
output checksums in both successful audit manifests match; all three changed
Python modules compile. Status, progress and checkpoint agree on incomplete
replication and the exact remaining allocation. Evidence: artifacts/controlled-smc-kdm-replication-20260928/closure-review-20261006-01/.
The initial review path-join error and its repair are recorded there. No
independent external review is claimed.

Latest numerical results and terminal verification are preserved in the circular
consumer result linked above and kdm-circular-review-20261006-01/. No independent
external review is claimed.

## Accepted continuation

The owner accepted the numerical criteria and directed execution on 6 October.
The [bounded neural consumer phase](controlled-smc-kdm-neural-consumer-20261006.md)
is active. It preserves the old float32 failures and tests the missing actual
neural/optimizer consumer before full training allocation. The new compute
grant is already in the ledger and must not be credited twice.

The neural replay phase is now complete: [result](controlled-smc-kdm-neural-consumer-result-20261006.md). All actual loss, trainable-gradient, clipping and Adam checks pass in both dtypes; two float32 angle-coordinate checks remain failed. Native sampling, full training, cost calibration and untouched qualification remain. See the active checkpoint for the charged balance.

The native sampling phase passes 28/28 source replay, native moment, validity, placement and trace checks in both dtypes. See [result](controlled-smc-kdm-native-sampling-result-20261006.md). Full training/stages, cost calibration and untouched qualification remain open.

## 7 October batch-64 continuation

The [batch-64 replay](controlled-smc-kdm-neural-batch64-result-20261007.md)
passes 896/896 checks in both dtypes. It measures a first update, including XLA
compilation; steady-state and full-training cost remain unmeasured. The previous
float32 angle and circular-gradient failures remain open on their original
fixtures. The source stage/controller and native consumer integration are the
next bounded phase; no full paper-scale training run is started yet.

Batch-64 fixed replay passes 896/896; full training and cost estimation remain open. Audit source training stages/checkpoint selection/scheduler/early-stop, implement native sampling in the neural consumer and verify checkpoint/controller against source. Then cost a native epoch before source-only calibration and untouched paired qualification. Preserve existing float32 failures; no E0.

## 7 October: controller and native neural bridge
The [bridge result](controlled-smc-kdm-training-bridge-result-20261007.md) passes 423 source controller, 1648 GPU native-consumer and nine guard/restore checks. Native resampling, source scheduler/checkpoint semantics and update resume are checked. Remaining KDM implementation work is native true-state initialization, weight pretraining, shared-variable batch specializations and complete source stages/data epochs, followed by actual epoch cost and accepted untouched qualification. Earlier float32 failures remain open. This closes no full replication claim.
