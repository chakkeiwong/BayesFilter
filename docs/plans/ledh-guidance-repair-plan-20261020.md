# Repair the guided LEDH implementation and complete its learning design

Updated21 October 2026. Status: Phases A/B COMPLETE; Phase C IMPLEMENTATION AND BOUNDED DIAGNOSTIC EVALUATION COMPLETE. The user authorized
execution and selected a SIR-specific neural KDM proposal, explicitly an
adaptation of the replicated method. The confirmed software repairs passed;
the neural learner is implemented and its completed independent comparison does
not establish a moment gain. The optional512 rung is incomplete; independent
model-score accuracy remains unevaluated. No candidate is promoted.
See the [neural protocol](ledh-sir-neural-kdm-adaptation-20261020.md) and the
[Phase-A result](artifacts/ledh-guidance-repair-20261020/attempt-01/result.md).

Worktree: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Branch: `ledh-refactor-with-policy-fix`; inspected base `4d6915cc66`. Preserve
unrelated dirty files. No branch change, commit, push, or environment change.
Evidence: [consumer audit](artifacts/ledh-guidance-implementation-audit-20261020/result.md)
and [training diagnosis](artifacts/ledh-moment-training-failure-diagnosis-20261020/result.md).

## Question and evidence contract

First, do the executed consumer and result assembler report and validate the
quantities they claim without changing the accepted numerical calculation?
Second, which missing learning capabilities should implement the documented
moment-training proposal, and does the completed learner help the actual LEDH
estimator? The second question cannot be answered by repairing diagnostics.

Engineering comparators are the exact pre-repair healthy fixtures and the
existing complete 544-row result. The malformed copies from the audit are
negative controls. Engineering completion requires correct consumer flags,
strict result integrity, retained conditioning diagnostics, reproducible source
capture, and unchanged healthy values/analytical scores within existing test
tolerances. Passing tests is not evidence of better moments or scores.

For later scientific evaluation, the baseline ladder must include plain LEDH,
untrained guidance, the corresponding restricted adapter, the enhanced learner,
and relevant simple estimators. Use freshly generated, disjoint calibration,
validation and final evaluation partitions. The exposed 544-row holdout is
read-only engineering evidence, never new tuning or confirmation data.

## Phase A: repair confirmed defects and omissions

| Item | Implementation and acceptance check | Status |
| --- | --- | --- |
| Guide validity | Propagate the physical branch's actual guide and proposal validity into the trace. Test valid, invalid and inactive-guide consumers; retain overall rejection of invalid guides and unchanged healthy values/derivatives. | Complete |
| Paired-result integrity | Validate expected dataset/arm/replicate keys, uniqueness, required fields, finite losses, validity flags, and paired particle seeds before aggregation. Reject missing, extra, duplicated or unpaired records as incomplete/invalid, never as valid comparisons. Accept the original complete grid with unchanged summaries. | Complete |
| Conditioning margin | Carry the computed minimum relative SPD margin through the shared guide/proposal call chain into trace and run records. Distinguish inactive/not-evaluated entries from measured margins. Add consumer-level propagation tests; do not add clipping, ridges or new thresholds. | Complete |
| Source provenance | Snapshot every package Python source, including untracked dependencies, plus hashes and tracked dirty diff/status. Verify at finalization; do not retroactively certify historical omitted source. | Complete |

Use focused shared-module changes, not a second numerical implementation. Turn
the audit counterexamples into normal regression tests. Test the executed
`PhysicalCampaign` assembler, not only the older analyzer. Preserve original
audit scripts and data as evidence. No OT, covariance, guide density, mixture
weight, derivative, training objective or default-policy change in this phase.

Run CPU-only focused tests first with `CUDA_VISIBLE_DEVICES=-1`, two intra-
and inter-op threads, and the existing `tftwogpu` interpreter. The existing
physical-proposal, moment-training and SIR-guide tests are the initial suite;
add targeted result-integrity and provenance tests and run their explicit
paths. Run a bounded GPU/XLA consumer smoke only after CPU regressions pass,
using escalated GPU access and verified memory growth. That smoke checks the
changed trace structure and graph compatibility, not training efficacy.

Phase-A artifacts: a fresh attempt directory under
`docs/plans/artifacts/ledh-guidance-repair-20261020/`, containing commands,
logs, before/after checks, source snapshot, manifest and result note. Preserve
each attempt. Budget: at most 30 minutes CPU wall time and 600 GPU-process
seconds, charged to the existing remaining campaign budget; at most three
localized repair/test iterations before recording an unresolved cause. No
numerical workaround to make a failing invariant pass.

## Phase B: reconcile the mathematics, claims and implementation

Checked distinction: the present physical proposal uses ordinary filtering
importance correction; full cSMC twisting is not required for that target.
The replicated neural KDM hardcodes a three-state bearings filtering model,
whereas SIR has eighteen states and the current RBF object is a future message.
The user selected a new SIR neural proposal adaptation. Its conditional target,
evaluable density, shared consumer wiring and analytical theta derivatives must
be specified before the dependent campaign. Do not describe it as a completed
source-faithful SIR replication or silently insert a filtering density as a
future-likelihood message.

Review the relevant propositions in
`docs/papers/ledh_younis_kdm_score/ledh_supervised_guidance.tex` and its parent
document against the actual consumer call chain. Write a compact mapping of
each mathematical quantity to its implementation, derivative and test.

1. Distinguish a future-informed proposal with ordinary filtering correction
   from full controlled-SMC twisting. The current proposal is not wrong merely
   because it does not twist the intermediate targets. If the intended design
   requires twisting, specify intermediate targets, normalizers, incremental
   weights and analytical derivatives before implementing it. Do not add
   twisting merely to justify an existing label.
2. Locate the actual replicated KDM trainer and cSMC fitter. Inspect their
   local paper and author-code anchors before claiming source faithfulness.
   Specify how their outputs enter the shared LEDH proposal and moment loss;
   keep the present RBF approximation explicitly named as an approximation.
3. Replace the assumption that 3/12 adapter scalars constitute full learning
   with an explicit trainable parameterization and capacity protocol. Verify
   gradients reach the intended parameters at the real training endpoint.
4. Keep the current two-component physical-mixture probability distinct from
   the earlier three-density covariance-mixture allocation. Specify which
   proposal family the repaired design uses, its sampling rule and complete
   density. Do not silently substitute one beta parameter for the other.
5. Preserve simulated-label independence, conditioning, fixed feature scaling
   and the conditional squared-error identity. Separate proposal-training
   moments from filtering moments and from likelihood-score accuracy.

Completion requires derivations, consumer wiring/parity tests and accurate
LaTeX descriptions, not merely helper functions. The chosen SIR neural proposal
adaptation is implemented; source-identical bearings KDM and full cSMC twisting
remain distinct from this design and are not Phase-A bug fixes. Material
departures from the agreed mathematical design require owner direction; routine
implementation within that design does not require new approval ceremony.

## Phase C: implement and evaluate the agreed learner

After Phase B, fill in the exact target-specific protocol before serious runs:
trainable family/capacity search, optimizer search, scaling checks, training
budget ladder, independent seed policy, frozen selection rule, fresh partition
seeds, explicit commands, and versioned output directories. Do not reuse the
small pilot's settings as universal defaults or tune against the exact score.

Use simulated moment labels for training. Evaluate filtering-moment accuracy
and downstream analytical likelihood-score accuracy separately; references
are evaluation-only. Predeclare the paired uncertainty analysis, required
improvement and numerical tolerances before opening final evaluation data.
Moment-loss improvement alone cannot promote a score-estimation claim.

Construct the cheap adversary set from SIR structure: prior predictive moments,
an observation-informed plug-in estimator, a recursive Gaussian filter, and
the conditional-only proposal. Explain each and compare conditionally in the
predeclared observation regimes and boundary/interior cases. These are sanity
vetoes, not training or selection targets. Beta=1 success is not LEDH success.

Retain shared UKF, GenUT and Contract E--Chol semantics, analytical recursive
scores, canonical chunking and TensorFlow/TFP implementation. OT(1,5) remains
the existing diagnostic comparator setting, not evidence of scope-specific
tuning. Any claim-bearing run must satisfy the current per-scope tuning policy
with independent partitions; do not retune merely to remove roundoff errors.

Phase-B derivations and executable consumer checks are recorded in
`artifacts/ledh-sir-neural-kdm-20261020/attempt-01/math-code-review.md`.
The user-selected neural adaptation and reviewed Phase-C protocol are in
`ledh-sir-neural-kdm-adaptation-20261020.md`. Three actual GPU/XLA optimizer
updates pass after repairing XLA specialization and a WSL command-buffer failure.
The sixteen-configuration screen and both128-update continuations are complete.
Warm continuation updates initially took about6.5seconds, but the first optional
512 process exhausted its cap after362 valid updates. Attempt16 completed final
evaluation using only completed32/128 checkpoints. All1472 rows are valid, but
the neural candidate fails both the primary criterion and heuristic veto.
Attempt19 completed the fresh restricted comparisons:1024 valid paired rows,
unchanged package source, adapted cSMC loss.863626 and adapted RBF loss.872854.
Their adapted-minus-unadapted intervals include zero; every restricted candidate
fails the recursive-Gaussian heuristic comparison in every required regime.
It charged3685seconds, leaving4891.197895057seconds. See the terminal
`artifacts/ledh-sir-neural-kdm-20261020/attempt-01/result.md` and combined
`completion-analysis.json`. The timeout repair note records the full
charge and fourteen passing recovery tests. No failed state enters selection.
The active checkpoint records process supervision, recovery charges and exact
next action; neither validation loss nor successful training establishes efficacy.
MathDevMCP's conditional-probability audit is inconclusive; direct derivation
and numerical consumer checks are the available positive evidence. No expansion
of the parent's remaining 23766.197895057 GPU-process seconds is authorized
here (Phase A charged 60 seconds, including setup). Deduct subsequent costs;
stop as under-budgeted rather than promote an
inadequately trained candidate. Failed candidates trigger the specified repair
ladder, not automatic rejection of the entire research direction.

## Defaults, diagnostic roles and stopping rules

| Choice or finding | Provenance/status | Risk and earliest check | Role |
| --- | --- | --- | --- |
| Existing healthy fixtures and tolerances | Engineering baseline | Shared bug; retain independent density/finite-difference checks | Engineering pass criterion |
| Missing/duplicate/unpaired rows | Reproduced counterexamples | Invalid paired inference; reject before aggregation | Evidence-validity veto |
| Invalid numerical computation | Existing validity contract | Non-finite or inadmissible computation; consumer rejection test | Numerical veto; unresolved defect stops affected runs |
| SPD margin | Already computed diagnostic | Silent loss of conditioning information; trace propagation test | Explanatory only unless separately calibrated |
| 3/12 parameters, beta=.5, small minibatches | Restricted pilot choices, not reviewed general defaults | Weak family or signal; capacity, allocation and gradient checks | Repair hypotheses, not promotion evidence |
| Better validation moment loss | Nomination diagnostic | Overfitting or wrong downstream target; untouched consumer evaluation | Selection only under frozen protocol |
| Conditional heuristic underperformance | Constructed sanity comparisons | Complex method misses salient regimes | Promotion veto, not continuation veto |
| Exposed holdout, wrong target/derivative, corrupted provenance | Evidence/implementation failure | Cannot answer stated question | Stop affected scientific interpretation; repair without overwriting evidence |

Initial skeptical review, before implementation: Phase A was ready for bounded execution because its baseline,
counterexamples, scope and pass conditions are explicit and require no
scientific ranking. The earlier 23 passing tests missed consumer reporting and
assembly defects; new tests must therefore exercise those exact endpoints.
The original 544 rows are sound on completeness/pairing, so neither new defect
is established as the cause of poor training. Phase B required design
reconciliation, not a declaration that full twisting will help. Phase C was
contingent on a target-specific protocol, subsequently supplied by
`ledh-sir-neural-kdm-adaptation-20261020.md`. Its completed neural comparison
does not establish a gain; passing implementation tests alone cannot establish
scientific progress.

At each completed phase update the checkpoint and master with exact work done,
remaining budget, evidence and next action. Terminal scientific results need
decision and inference-status tables separating hard failures, viable candidates,
descriptive differences, uncertainty-supported rankings and default readiness.
No phase is complete simply because a document or test says it is planned.
