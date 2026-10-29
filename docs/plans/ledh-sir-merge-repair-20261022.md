# Guided SIR merge and runtime repair

22 October 2026. Execute on `ledh-refactor-with-policy-fix` in
`.localresources/worktrees/ledh-iapf-execution-20260922`, HEAD `90a3acebd`.
The owner requests tracing the slowdown and incorrect merge issues, reviewing
the repair, and executing it. Preserve unrelated unfinished research work.

## Question and evidence contract

Which changes in the actual guided SIR consumer caused additional GPU work or
changed numerical outputs, and can equivalent shared calculations repair them?
The consumers are plain LEDH, conditional proposal, physical mixture, and the
frozen neural/restricted guide replay. Trace each from the endpoint to flow,
density, UKF, reset, correction and validity routines. Audit the restored local
changes as well as both merge parents. Merely finding a faster implementation
elsewhere does not establish that this consumer calls it.

Comparators are the preserved pre-merge source, merged QR source, and completed
21 October checked-elimination repair. The primary correctness criterion is
same-input value, raw moments, loss, analytical directional derivative and
validity agreement at the existing absolute-plus-relative `1e-8` tolerance.
The existing pre/merged plain discrepancy must be localized before claiming
pre-merge equivalence. Mathematical equivalence alone cannot waive that screen.
An independently checked result can identify which version needs repair; it
cannot justify silently widening the screen or changing the target.

Engineering promotion requires passing correctness and invalid-input checks,
plus paired warm runtime intervals below one against the current repair for
the affected plain/conditional endpoints. Physical-mixture runtime must have
an upper paired interval below 1.10. Use eight alternating blocks on the four
previously selected ordinary/boundary/high-loss datasets, with zero and nonzero
directions separately. Report direct pre-merge timing without claiming numerical
equivalence if that separate criterion fails. Compiler work counts, stage
timings, memory and proxy losses explain results; they do not certify science.

Invalid numerical output, lost checks, incorrect tangents, changed fixed inputs,
source mutation during a run, or incompatible guide/reset semantics veto a
candidate. They trigger a bounded repair, not abandonment of the research
direction. Exhausted compute, corrupted input evidence, or an unresolved change
of mathematical target stops dependent work. No score-quality, training,
cross-model, TF32 default, posterior, or HMC promotion is sought.

## Execution and repair order

1. Freeze current sources. Inventory changed modules on the consumer chain and
   compare both parents, restored work and compiled evidence. Trace reset
   controls, physical-mixture tuple wiring, covariance lifecycle, tangents and
   status propagation. Record confirmed defects separately from hypotheses.
2. Add a diagnostic-only step trace and bounded source substitutions where
   necessary to locate the first plain-arm numerical divergence. Isolate flow,
   density, UKF and reset arithmetic; compare independent formulas on identical
   intermediate inputs. Diagnostic ablations must never become runtime fallbacks.
3. Profile the real consumer and isolated hot calculations. Derive shared,
   algebraically equivalent repairs before implementing them. Candidates include
   eliminating redundant factorizations/solves, compact covariance arithmetic,
   and an observation-space flow determinant. Do not replace the guided canonical
   recurrence with the distinct native batch algorithm. Preserve general model,
   full covariance, guide, mixture, analytical derivative and invalidity support.
4. Add focused regression, wiring and mathematical parity tests for confirmed
   bugs/repairs. Check invalid/nonfinite inputs and FP32/FP64 mechanics. Run a
   small GPU-XLA consumer check before the existing full frozen replay and paired
   timing. Reuse the replay harness, inputs and tolerances; do not retrain.
5. Review the final code, compiled consumer and complete outputs. Classify every
   investigated change as causal, noncausal, repaired or unresolved. Update the
   master checkpoint with the evidence, budget and next action. No commit/push.

## Assumptions and skeptical review

| Choice | Provenance and role | Failure mode and earliest check |
| --- | --- | --- |
| FP64, N64, T5, 24 flow steps, OT(1,5), balance32 | Existing user-frozen diagnostic scope | Not a tuned production default; verify manifest/input hashes |
| Preserve all validity checks | Current numerical safety contract | Checks may expose unused expensive work; optimize the calculation, not remove the check |
| Existing covariance and reset mathematics | Canonical guided consumer | Native batch is not a drop-in replacement; verify call chain and independent derivatives |
| Same-matrix elimination | Previous repair baseline | Cost may remain high; stage profile and direct before/after consumer timing |
| Observation-space identity, if used | Exact determinant lemma, candidate | Finite-arithmetic/domain differences; derive assumptions, compare full-space reference and invalid cases |
| Four exposed regimes/eight blocks | Previous bounded timing protocol | Device load and dependence; alternate order, paired intervals, no general speed claim |
| Existing heldout replay | Engineering regression only | Already exposed data cannot establish fresh guide improvement |

Cheap adversaries are unchanged pre-merge, merged QR, current elimination and
independent dense solves/finite differences. Evaluate each applicable endpoint
and regime separately. This is an implementation comparison, not learned-method
selection. No safety ridge, clipping, damping or threshold is introduced.

Pre-mortem: a faster reduced method could masquerade as a repair; protect the
canonical recurrence and test consumer wiring. A numerical difference could be
mislabelled roundoff; identify the first divergent operation and its amplification.
A microbenchmark could hide remaining consumer overhead; final timings must use
the real endpoint. Masking a failure could falsely pass; invalid cases and status
propagation remain mandatory. A successful merge could still leave optimized
code unreachable; report that separately from a merge conflict.

Skeptical plan review: PASS for bounded investigation and equivalent repair.
The earlier plan left pre-merge numerical drift and remaining runtime unisolated;
this plan explicitly includes both. It does not assume that the two native-batch
and guided recurrences are equivalent. Runtime cannot override failed parity.
An observation-space optimization requires its own checked derivation and focused
tests before installation. No unreviewed numerical protection/default is selected.

## Budget and artifacts

Parent campaign has 12835.769523 GPU-process seconds remaining. Allocate at most
10000 seconds to this repair, including probes, compilation, tests, failed
attempts and resident-worker time; at most four candidate implementations.
Use fresh attempt directories; preserve earlier evidence. CPU source inspection
and small reference tests are not GPU budget. This request authorizes execution
within the remaining local campaign budget without per-attempt approval.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted GPU,
memory growth configured and verified before initialization, XLA on, TF32 off,
existing `XLA_FLAGS=--xla_gpu_enable_command_buffer=` exception. CPU diagnostic
tests set `CUDA_VISIBLE_DEVICES=-1`. No pfor, package change or remote compute.

Artifacts: `docs/plans/artifacts/ledh-sir-merge-repair-20261022/`. Preserve source
hashes, actual commands, Git identity, environment/device/memory policy, input
seeds and hashes, timings, per-attempt results and budget. Final result includes
decision and inference-status tables and a red-team note. Exact commands are
saved before their launch; adaptive diagnostic commands follow the above scope.

## Timing-harness repair after attempt04

All 216 paired consumer outputs agree exactly, but block-level timing is
inconclusive for conditional and physical endpoints: intervals span substantial
improvement and regression. Concurrent GPU utilization remains about81% after
the workers exit, and paired blocks vary greatly in duration. Treat attempt04
as preserved descriptive evidence; do not select only favorable blocks.

Attempt05 pairs each individual input/arm/direction immediately across the two
persistent workers, alternating order. It retains the same four inputs, eight
blocks, controls, parity checks and block-based uncertainty calculation. This
shortens the time between paired observations without excluding inconvenient
measurements. GPU contention can still prevent a ranking. A protocol regression
must show identical coverage before launch. Existing checked elimination is the
explicit baseline factorization for HLO inspection; expecting QR here would
compare against the wrong baseline. Reviewer/executor skeptical audit: PASS for
this localized measurement repair under the unchanged campaign budget.

## Candidate 2: restore dropped reset acceptance checks

The pre-merge canonical consumer required row stochasticity within 1e-6,
column total-variation residual within 1e-4, and a finite column-residual
tangent. The merge removed these checks when adopting structured failure
status. The shared reset's `valid` flag checks factorization/scaling validity;
it does not check these marginal residuals. Thus the replacement is weaker.
Restore the original checks at their original consumer, retaining separate
value/force failure status. This changes acceptance of invalid computations,
not the reset map, balancing count, or accepted numerical outputs. Do not
change the shared reset helper's documented factorization-validity contract.

First inject finite row/column defects at that consumer and demonstrate that
the current endpoint accepts them. Then restore the checks and require rejection
plus a no-fire healthy case. Run the focused regression and frozen replay on
the final candidate. Skeptical review: PASS; this repairs a concrete lost
invariant, keeps the original thresholds, and cannot silently relax parity.
The safety-only activation repair is also an actual merge regression, but that
option is off in this campaign and cannot explain the observed slowdown.

## Residual-cost diagnostic after attempt08

The final source passes repair-versus-merged replay, but direct pre-merge timing
still shows slower plain nonzero and both conditional directions. Before closing
the investigation, isolate the newly live determinant checks in a disposable,
explicitly ineligible diagnostic source copy. Remove only the dependence of flow
validity on determinant factors and determinant accumulators. Keep full-state
finiteness checks, all numerical formulas, correction and reset controls. This
lets XLA discard determinant work where its numerical outputs are unused. It is
not an admissible implementation: singularity detection is deliberately weakened
for causal measurement, and the working-tree runtime retains every check.

Attempt09 compares final candidate2 against that diagnostic copy using the same
four inputs, eight adjacent paired blocks, three endpoints and two directions.
Both workers must reproduce the merged archive, preserve healthy validity flags,
and trace once. Inspect HLO to confirm elimination of the determinant helper in
plain/conditional and its retention in physical mixtures. Runtime intervals are
explanatory only; no ablation can be promoted or used for scientific inference.
Parity, coverage or source-mutation failure invalidates the timing interpretation.
The physical endpoint is a control for machine-wide timing variation, although
removing checks can also change its generated code. No unsafe-input correctness
claim is possible. Cap the attempt at1800 aggregate GPU-process seconds within
the remaining3354.683606 seconds; preserve all previous evidence.

Default/assumption audit: exposed inputs and eight-block pairing are inherited
diagnostic controls, not fresh validation. A compiler dead-code prediction is a
hypothesis checked by optimized HLO. Shared-device timing is a known confounder;
retain block intervals and never infer exclusive-device cost. A failed safety
ablation is not a veto on the accepted guarded repair. Skeptical audit: PASS for
this bounded causal diagnostic; it answers residual overhead without silently
removing guards from the runtime or changing the numerical target.

## Execution outcome and terminal review

The bounded plan is executed. Two confirmed merge regressions are repaired:
reset marginal rejection and safety-only correction activation. The exact
observation-space flow determinant preserves analytical derivatives and reduces
SIR factorization order18 to9. All123 distinct regression tests pass, including
nine generic conditioning/failure cases retained from candidate03. Candidate02
passes all2928 full-replay rows and all six paired timing criteria against the
21 October repair. No data, settings, tolerance, guide or scientific target changes.

Direct pre-merge timing is complete and does not establish full speed recovery:
plain nonzero and both conditional directions remain slower. Plain pre-merge
numerical equivalence still fails. The correction reference checks identify
ill-conditioned amplification, but not one unique initial floating operation.
Attempt09 confirms material overhead from retained determinant validity work.
It preserves exact healthy outputs, but its deliberately weaker singular-flow
checks are a hard runtime veto; the diagnostic source copy is deleted after its
patch, hashes, HLO and outputs are preserved. Existing runtime guards stay intact.

Terminal skeptical executor review: PASS for the equivalent guarded repair and
faithful reporting; FAIL for any claim of complete pre-merge recovery. Baselines,
full raw moments, analytical scores, call-chain wiring, invalid inputs, flag
semantics and source identity are checked. Proxy losses and timing cannot waive
failed pre-merge parity or the prior heuristic-dominance veto. The strongest
remaining timing confounder is shared-device contention; individual safe-factor
and masking costs are unquantified. No independent reviewer was used this stage.

The result is partial engineering recovery, not rejection of controlled SMC or
KDM. A subsequent numerical repair requires an independent reference and explicit
conditioning/protection criteria; additional equivalent optimization requires
fresh parity and timing. Neither can silently relax checks or tolerances. The
research direction remains open, and no training/score-quality promotion follows.
Full evidence and unresolved findings: artifacts/ledh-sir-merge-repair-20261022/result.md.
Before the candidate03 continuation,7683.699763 of10000 GPU-process seconds
were charged. The terminal continuation charge and disposition follow below.
No task process remains active. No commit or push.

## Candidate03 continuation within the existing parent budget

The completed ablation identifies a concrete next equivalent optimization.
Continue under the reviewed derivation and bounded protocol in
[cholesky-determinant-candidate.md](artifacts/ledh-sir-merge-repair-20261022/cholesky-determinant-candidate.md).
It reuses the innovation factor and tests a checked positive-definite determinant
ratio. Candidate02 remains protected until this candidate passes all required
checks. The local allocation is extended from10000 to12800 within the unchanged
12835.769523 owner-authorized parent remainder at stage start. No additional
owner compute, different hardware, tolerance change or scientific target is used.

Terminal continuation review: candidate03 passes all216 paired consumer checks,
including exact flags, but only three of six declared runtime criteria pass.
The plain zero/nonzero and conditional zero intervals remain inconclusive;
favorable point estimates cannot substitute for the predeclared acceptance rule.
Do not retain candidate03. Preserve its source and results; restore candidate02.
No full replay or additional pre-merge campaign is required for an unretained
candidate. The nine generic conditioning tests pass against restored candidate02;
all31 determinant tests pass. Final hashes match980 package and414 auxiliary
files to the protected candidate02 snapshot. This is an engineering retention
decision, not a rejection of the determinant identity or the research direction.

The one-further-candidate stage is complete. Total charge8375.548099 of12800
GPU-process seconds;4424.451901 remain unused locally and4460.221424 in the
unchanged parent allocation. No task process, commit or push. Numerical recovery
and complete pre-merge speed recovery remain unresolved as stated above.
