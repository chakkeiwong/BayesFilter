# KDM released-source replication closure

14 October 2026. Authorized by the user's request to refresh the master and
continue. The authoritative master remains controlled-smc-kdm-replication-master-program-20260928.md.

## Question and fixed scope

Does the current TensorFlow implementation reproduce the released bearings
MDPF and A-MDPF training/filtering operations on the declared tested inputs,
including source-intrinsic failures? Compare against author commit
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18 and the locally stored NeurIPS 2023 paper.
The selected consumer is the existing 25-particle bearings training and
evaluation program. Accepting this bounded implementation does not reproduce
unreconstructed Appendix B drivers or every original experiment.

Controlled-SMC bounded fidelity stays accepted. Original iAPF, Task3,
enhancements, new numerical protections and population-quality certification
are outside this closure. A paper-scope limitation is not automatically a
released-code implementation failure.

## Evidence contract and research intent

| Item | Role and decision |
|---|---|
| Complete selected-operation coverage | Primary acceptance criterion: each operation has inspected paper/source anchors, a local consumer-to-kernel chain and executable evidence. |
| Current-source binding | Acceptance criterion: numerical runtime hashes match the evidence relied on; stale evidence must be qualified or refreshed with a focused check. |
| Same-input replay | Acceptance criterion: use each existing test's declared, quantity-specific tolerance. Do not impose a universal 1e-5/1e-6 rule on all gradients or outputs, or revise tolerances after looking at results. |
| Stochastic operations | Verify common-random-input arithmetic separately from the native sampling law. Equal numeric seed labels in Torch and TF do not imply identical random draws. Branch sensitivity requires direct localization, not blanket forgiveness. |
| Numerical and failure validity | Acceptance criterion on tested author-valid cases: no additional invalid port results. Author-intrinsic failures may remain. |
| Native endpoint and training wiring | Acceptance criterion: actual MDPF and A-MDPF consumers reach the tested kernels; same-checkpoint metrics agree under the saved tests and controller/optimizer/stage transfer have executable checks. |
| Source/hash/schema/coverage failure | Continuation veto for the affected inference and repair trigger. Preserve failed evidence; repair only a demonstrated local mismatch. |
| Independently trained final NLL/RMSE | Explanatory diagnostic. Preserve all unfavorable cases. A residual finite difference alone neither establishes a translation defect nor proves equivalence. |
| GPU/XLA, trace and memory evidence | Engineering validity for the recorded route, not statistical performance evidence. |
| Missing Appendix B driver/settings | Limits original-paper reproduction; report separately from the released bearings implementation. |

Mechanism under investigation: remaining apparent gaps may be stale coverage
records, untested source operations, genuine translation mismatches, or valid
random-training differences. Cheap comparator: the actual pinned author
operation on the same inputs, plus unchanged replay controls. Other filtering
algorithms do not test source fidelity and are not acceptance criteria here.

Not concluded: population noninferiority, algorithm superiority, exhaustive
numerical robustness, exact independent-training trajectories, production/HMC
readiness, original iAPF completion or all-paper reproduction.

## Execution

1. Inspect source and local call chains for data/masks, architecture/constants,
   initial particles, pretraining cloud/loss, KDM density, IWSG/resampling,
   proposal/weight/bandwidth heads, full loss/TBPTT, optimizer, controller,
   checkpoint/stage transfer, native filtering and metric aggregation.
2. Assemble a machine-readable coverage matrix linking anchors to saved
   executable evidence, checking current numerical source hashes and all
   relevant result/validity fields. Add bounded AST wiring checks where they
   can verify consumer reachability without importing a GPU framework.
   Static wiring checks support reachability; they do not replace numerical
   parity. Do not call a prose inspection executable evidence.
3. Inspect saved unit01/unit02 intervention histories and existing first-
   divergence replays. Identify a specific unchecked operation before proposing
   another numerical test. No automatic training ladder.
4. Inspect the pinned released source inventory for Appendix B and the selected
   bearings configuration; record what is absent or not reconstructed.
5. Audit the matrix for overclaims and issue one of: accepted bounded
   released-source replication; accepted with explicit scope limitations; or
   blocked by a named unmatched operation/implementation defect. Whole-paper
   reproduction retains its own verdict. Refresh master and checkpoint.

Executable audit:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/audit_kdm_source_coverage.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/source-closure-20261014/attempt03

This standard-library-only evidence/wiring audit deliberately sets
CUDA_VISIBLE_DEVICES=-1 and imports no numerical framework. New numerical work,
if justified by step3, needs a bounded substep recorded here before launch,
retaining the current scientific contract and campaign balance.

## Assumption/default audit and skeptical review

| Choice | Provenance/status | Risk | Earliest diagnostic |
|---|---|---|---|
| Released 25-particle bearings configuration | Existing source pin and approved implementation scope | Confusing it with all paper experiments | Inspect configuration and list scope limits |
| Existing parity tolerances | Previously declared tests; reviewed per quantity | Blanket tolerance hides error or creates a false failure | Inspect assertion code and saved outcomes |
| Reusing completed experiments | Existing immutable evidence | Code drift or a passing summary omits a failed branch | Check hashes, executed variants, terminal fields and failure lists |
| Same-start interventions | Saved source loader and exact TF old-start controls | Mistaking a large reduction for full attribution | Retain residuals; compare preexisting first-divergence evidence |
| No new training | Closure audit convenience, not a scientific claim | Untested operation overlooked | Explicit complete matrix with fail-closed missing-evidence handling |
| Source-intrinsic sensitivity retained | Owner replication target | Forgiving arbitrary port errors as sensitivity | Require a source-only reproducer or common-branch replay for that explanation |

Pre-execution skeptical audit: passes with two corrections to the earlier
discussion. The suggested numerical tolerance is not universal; preserve each
checked test's contract. Missing original-experiment source does not block a
narrower released-source verdict. No proxy metric promotes the implementation.
Coverage must be demonstrated for actual consumers, including A-MDPF, and old
or incomplete evidence cannot silently be treated as a current pass.

Pre-mortem: a file can exist without being called, an old pass can predate a
runtime edit, and generic resampling tests can miss learned A-MDPF weighting.
AST call-chain checks plus pinned numerical evidence distinguish those cases.
A gap that survives this audit must name the operation and smallest resolving
test, not trigger another generic sweep.

## Bounds and artifacts

At most three local audit executions, 600 CPU seconds total, no GPU allocation
for this phase; within remaining 205463.582169 CPU / 71779.382403 GPU seconds.
Write a fresh attempt directory; preserve logs, manifest, source/evidence
hashes, coverage JSON/Markdown, decision and terminal review. Stop on the cap,
corrupt evidence or an unresolved scope mismatch; classify rather than invent
a passing result. Do not overwrite earlier experiments. No environment changes,
external messaging, new allocation or approval ceremony.

Result: controlled-smc-kdm-source-closure-result-20261014.md.
The final result must distinguish bounded implementation acceptance, numerical
validity, descriptive training results and unavailable original-paper coverage.

## Audit-harness repair, 14 October 2026

Attempts01/02 are invalid audit-development outputs: they did not enforce the
evidence contract and assigned an unconditional passing verdict. Their
acceptance is withdrawn; numerical experiments are unchanged. One corrected
third audit is authorized as a localized harness repair within the same 600 CPU
second cap, no GPU use and unchanged scientific criteria. Before it runs, check
negative cases for missing/failed assertions, hash mismatches and broken wiring.
Review source-hash differences by affected numerical function, rather than
blanket acceptance or blanket rejection of an older surrounding harness.

## Corrected pre-execution review

The corrected auditor passed 14 focused positive/negative checks. Its decision
now depends on nonempty per-operation assertions, pinned source and saved-result
hashes, and consumer wiring; a missing file, changed implementation, missing
compiled binding or failed assertion prevents acceptance. The wiring checker
resolves the local TensorFlow keyword dictionaries rather than mistaking them
for uncompiled calls. It imports no TensorFlow and makes no new GPU claim.

Skeptical audit: the comparator is the pinned released author implementation;
matched-input numerical results and source-controller checks are the primary
evidence. Native cloud moments remain diagnostics. The endpoint metric replays
cover nine saved clouds plus three intervention clouds, with NLL and RMSE checked
on each; they are not 24 independent training replications. The known old
sampler change is checked by exact AST equality after removing only the optional
pretraining concentration argument. Three changed surrounding harnesses have
reviewed, hash-bound scope/observation changes; numerical runtime hashes remain
unchanged. Changed outer stage-isolation wrappers are excluded from the direct
operation evidence: the unchanged direct reference/diagnostic, runtime, input
and result hashes are checked instead.

No material unexamined default or new scientific threshold is introduced by
this audit. It can accept only the selected tested source implementation.
It cannot settle independently trained quality equivalence, all-paper coverage
or unseen numerical failures. Execute attempt03 under the unchanged stop/budget
conditions; preserve any failed check and report the named gap.
