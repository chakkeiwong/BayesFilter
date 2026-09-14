# HMC procedure repair: execution audit and reset note

Completed: 2026-09-15. The filename follows the September 14 plan and gap review.
Baseline: `9329cadf3296211ccfa9e2539235225e041ad36a`, with task changes in the working tree.

The nine engineering findings in the [gap review](bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md)
are repaired under the [audited repair plan](bayesfilter-hmc-whole-procedure-repair-plan-2026-09-14.md).
Supported public configuration branches now use one candidate-set controller.
The controller independently measures each exact `(L, epsilon)`, verifies every
measurement survivor, runs declared repair/refinement/evidence stages, and
retains every verified member. It supplies no implicit winner. R-hat is
reporting-only throughout tuning; posterior assessment remains separate.

This closes the reproduced library and integration defects. It does not
qualify a MacroFinance target, demonstrate posterior convergence, establish a
sampler ranking, or qualify target-scale archive performance. No MacroFinance
files or real-data campaign were changed or executed.

## Plan review and resulting decisions

The pre-execution skeptical audit used the requested lifecycle as its comparator,
rather than treating existing first-admission tests as the specification. It
required agreement among numerical observations, controller decisions, receipts,
and replay; fixed all evidence rungs before execution; and treated public-config
translation as an API migration requiring real public-call tests. Wrong
baselines, stale consumer context, unsupported numerical defaults, misleading
short-chain proxies, missing stop conditions, unfair evidence allocations, and
source-only restart claims were explicitly considered in the plan.

The terminal review followed preparation, dispatch, scheduling, evidence,
persistence, consumer selection, and posterior delegation. It found and repaired
additional integration defects during validation: preparation elapsed time was
affecting candidate seeds; GPU-scoped host serialization selected an unavailable
GPU kernel; the NeuTra consumer imported the historical tuner; exports omitted
other verified members; checkpoint reload needed observation-to-tensor agreement;
and the position-field adapter needed the caller's acceptance target. Retired
options now produce migration errors instead of silently restoring an older
procedure. Default config values are normalized before detecting customization.

The final inventory review also found that the required NeuTra route ledger was
ignored as generated JSON. A narrow `.gitignore` exception makes the ledger and
the new retained-member posterior entry visible for version control. Other
pre-existing ignore rules were preserved.

## Finding dispositions

All test names below are in `tests/test_hmc_whole_procedure_repair.py` unless a
different file is named. They are engineering regressions, not posterior studies.

| Finding | Executed repair and decisive checks | Disposition and limits |
| --- | --- | --- |
| G1: configuration selects another procedure | `hmc_candidate_set_public.py` translates ordinary and frozen-transport preparation into `HMCTuningCandidateSetController`; `hmc_candidate_set_position_field.py` shares that scheduler. Public ordinary, fixed-transport, position-field, and actual windowed-preparation tests passed. | Closed for supported public routes. Historical private implementations remain historical. Unsupported transports and legacy customization fail explicitly. |
| G2: evidence roles disappear at the numerical boundary | `hmc_candidate_set_execution.py`, controller receipts, and retained validation preserve evidence validity, promotion vetoes, and repair eligibility separately. `test_real_oversized_step_is_repaired_and_failed_measurement_is_durable`, `test_movement_veto_rejects_parent_but_measures_smaller_step_child`, and shared-invalidity checks passed. | Closed. An immobile parent remains rejected while an eligible smaller-epsilon child is measured. Shared corruption disables the whole scope. |
| G3: inconclusive searches appear complete without further work | Frozen evidence rungs use fresh streams of the same kernel; unfinished rungs resume; exhausted rungs become `inconclusive_at_cap`. `test_inconclusive_resume_extends_same_candidate_and_reaches_explicit_cap` passed for both inconclusive decision classes. | Closed. Terminally inconclusive candidates remain identifiable but are not replayable. Repeated-look intervals have no nominal sequential-coverage claim. |
| G4: numerical tuning cannot durably resume failures | `hmc_candidate_set_checkpoint.py` preserves execution, all observations, immutable evidence, and bounded chunks even with zero verified members. Checks cover interruption after a chunk, fresh-process resume exporting two members, rejected observation/tensor disagreement, and position-field recovery within a stage. | Closed for tested mechanics. Restart begins after geometry is frozen; preparation itself is not a resumable warmup checkpoint. |
| G5: proposals and all-survivor refinement remain consumer work | The common controller supports explicit per-L proposals, optional pilots, bounded all-survivor epsilon/L refinement, declared expansion, and trajectory proposals. Automatic public translation enables one pilot and refinement round. Broad-pilot and all-survivor tests passed. The operational final-metric epsilon bound is preserved with its geometry identity. | Closed for the declared search. Proposal factors and convenience grids remain hypotheses; no acceptance qualification transfers across L. |
| G6: reversed repair factors, cycles, duplicates, and truncated integers | Factors must exceed one; touched integer controls reject fractional/bool inputs; exact pairs are deduplicated. Reversals can propose an unvisited geometric interior point; domain crossings can measure an unvisited boundary. Corresponding negative and proposal-history tests passed. | Closed. Interior and boundary proposals require their own measurements; acceptance monotonicity is not assumed. |
| G7: call quotas fail to protect required work or describe compute | Reservations cover mandatory stages. Extensions/retries draw explicit top-ups from unreserved budget. The ledger records calls, transitions, conservative `(L+1)` work, preparation/execution time, and chunk progress. Gradient-cap, interrupted-rung, deferred-repair, and elapsed-accounting regressions passed. | Closed at the stated accounting precision. Gradient work is an estimate. Native calls and compilation cannot be preempted; time is checked between chunks. |
| G8: R-hat still rejects or ranks tuning candidates | Active tuning rejects the retired opt-in R-hat requirement with a migration message; historical fixed-transport classification no longer gates or tie-breaks on R-hat. Reporting errors/nonfinite values cannot erase evidence. Agent guidance and affected chapters now distinguish posterior assessment. | Closed for tuning. Existing cumulative posterior R-hat/ESS requirements remain active through `run_sequential`. |
| G9: reference, registry, examples, and consumers overstate integration | Rewrote the tuning reference and book chapter, aligned neighboring chapters and generated registry tables, migrated the active NeuTra consumer to explicit member IDs and all-member export, and added posterior-controller delegation. Public, consumer, documentation, migration, and route-policy tests passed. | Closed for the implemented interface. XLA defaults follow owner policy; tiny fixtures do not qualify arbitrary targets or every consumer campaign. |

The common result retains all verified IDs and no nominee. Consumers can export
each verified member and choose one later without retuning. A member's immutable
epsilon/L, verified endpoint, fresh posterior seeds, and XLA policy are checked
before delegation to the existing sequential posterior controller. Coordinate
conversion applies the tuning binding's geometry before the consumer's model
transformation. Posterior rejection does not remove a member from the tuning set.

## Validation record

Evidence is in [the artifact directory](artifacts/hmc-whole-procedure-repair-2026-09-14/).
The [run manifest](artifacts/hmc-whole-procedure-repair-2026-09-14/run_manifest.json)
records commands, environment, source inventory, seeds, timing boundaries,
device policy, and preserved attempts. The
[test summary](artifacts/hmc-whole-procedure-repair-2026-09-14/validation_summary.json)
aggregates the latest recorded result for each current test ID.

| Attempt | Recorded result | Interpretation and repair |
| --- | --- | --- |
| CPU r1 | 149 passed, 28 failed | Public result/config migration, stale consumer import, documentation expectations, and position-field interruption needed repairs. |
| CPU r2 | 273 passed, 8 failed | Repaired timing-dependent identity; updated historical/private versus public tests, XLA expectations, fixed-transport epsilon/reporting assumptions, and guide checks. |
| CPU r3 | 310 passed, 2 failed | Corrected one outdated initial-epsilon assertion and added the new posterior entry to the route ledger. |
| CPU r4 | 140 passed | Focused execution, checkpoint/replay, public policy, consumer, and route regressions passed after those repairs. |
| CPU r5 | 1 passed | Actual windowed preparation reached the automatic broad pilot. |
| CPU r6 | 63 passed | Final configuration, position-field acceptance-target, public dispatch, and documentation regressions passed. |
| GPU r1 | Failed before sampling | Host `SerializeTensor` was incorrectly placed in GPU scope. Tensor serialization/parsing now explicitly use CPU, including when the caller enters a GPU device scope. |
| GPU r2 | Passed; six candidates, four verified | GPU/XLA tuning, checkpoint resume, explicit member export/reload, and a retained block succeeded. This was prior to the final config corrections. |
| GPU r3 | Passed; six candidates, four verified | Repeated those mechanics with the final runtime sources; completion was `complete`. Source closure matches the final files. |

The aggregate is **314 current unique tests with passing latest results**, zero
remaining failures or skips. This is a progressive regression set, not one run
of the entire repository suite. The earlier unparameterized position-field
test was replaced by target `0.7` and `0.8` cases and is excluded from that count.
Historical scheduling tests now explicitly exercise private historical helpers;
public regressions assert the new common result and lifecycle.

Recorded pytest wall time is 1,089.08 seconds. All CPU numerical checks
deliberately hid GPUs with `CUDA_VISIBLE_DEVICES=-1`, set
`TF_FORCE_GPU_ALLOW_GROWTH=true` and `BAYESFILTER_TEST_DEVICE_SCOPE=cpu`, and used
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. They are mechanics/reference
exceptions. Development snippets were not individually timed; these results
must not be presented as a complete performance accounting of development.

GPU r3 used trusted execution on physical GPU 0, an RTX 4080 SUPER, with
verified memory growth before initialization, TF32 enabled, and XLA compilation.
The readiness helper initially classified desktop CUDA contexts as busy; bounded
inspection found GPU 0 at 0% utilization with over 31 GiB free while GPU 1 was
running research work. GPU 1 and other processes were preserved. The actual
tuning samples report `/GPU:0` placement. The final smoke recorded 58.96 seconds
after framework initialization and a TensorFlow allocator peak of 578,816 bytes;
that is neither whole-process memory nor steady-state performance evidence.
Python was 3.13.13, TensorFlow 2.20.0, and TFP 0.25.0. GPU r1 timing was not
captured. GPU r2/r3 timing excludes imports and initial memory-policy setup.

The complete book built successfully to
`/tmp/bayesfilter-hmc-procedure-guide-20260915-r3/main.pdf` (552 pages).
The tuning chapter retains its nine equation environments, its unnumbered
display, and existing labels; the substantive changed procedures are replaced in the prose and
examples. Rendered pages 409, 413, and 419 were inspected during r2. The final
r3 page 413 was inspected again after clarifying what the mean Metropolis
probability averages. The inspected pages have no clipping or overlap.
Three pre-existing unresolved citations (`Gorinova2020`, `Pakman2014`,
`Afshar2015`) remain elsewhere in the book. This build and local visual review
do not substitute for human editorial acceptance of the prose.

## Engineering, numerical, and scientific decisions

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete this library repair | Public common lifecycle, repair, restart, replay, and consumer integration regressions pass. | No unresolved regression failure in the checked set; corrupt/shared-invalid evidence remains rejected. | Behavior and cost at consumer scale are unmeasured. | Review the concrete diff; use the common interface in a separately scoped target campaign. | No universal target/default-readiness certificate. |
| Preserve all four GPU fixture members | Each has its own passing fixed-kernel verification. | Failed pairs were not promoted. | Short Gaussian fixture evidence. | Select a member explicitly if conducting a separate posterior mechanics check. | No statistically supported ranking or posterior convergence. |
| Keep tuning and posterior assessment separate | Actual sequential-controller test accepts a checked frozen member and preserves tuning membership. | Posterior rejection remains possible and does not invalidate the tuning implementation. | No full posterior study was run. | Run target-specific cumulative posterior validation under its own plan. | No inferential accuracy, efficiency, or scientific conclusion. |

| Inference status | Assessment |
| --- | --- |
| Hard veto screen | Implemented validity, scope corruption, verification, and replay checks were exercised. A failed candidate remains rejected even when a repair child is allowed. |
| Statistically supported ranking | None sought or supported. |
| Descriptive-only differences | Acceptance, runtime, memory, and the number of survivors describe these fixtures. |
| Default-readiness | Shared public scheduling is implemented and regression-tested. GPU/XLA defaults follow owner policy; target-specific numerical and scientific readiness remains unestablished. |
| Next evidence needed | Exact consumer target/data/prior identity, value/score fidelity, required intermediate telemetry, representative memory/restart checks, and a separate posterior validation campaign. |

## Remaining limits and reset context

The strongest alternative explanation for the successful tests is that small
Gaussian and synthetic fixtures miss target-scale geometry, integrator telemetry,
resource, or posterior problems. A real public-call counterexample to exact-pair
qualification, complete cohort retention, checked restart, or frozen replay
would reopen the corresponding finding. The weakest evidence is consumer-scale
cost and qualification; neither was measured here.

- Complete target/data/prior lineage remains the consumer's responsibility.
  Automatic translation labels adapter-signature-only coverage as incomplete;
  hashes and start probes cannot prove unlisted dependencies unchanged.
- Supported frozen transport codecs are diagonal-affine and dense-IAF.
  Arbitrary callbacks are unsupported. Position-field tuning shares scheduling
  but retains conditional mechanics authority, without exact-score member replay.
- Checkpoint restart covers frozen preparation and subsequent tuning. It does
  not resume an interrupted mass-adaptation preparation from an arbitrary window.
- Retained archives embed evidence and validate predecessor history.
  Target-scale archive size, reload latency, and long continuation remain unqualified.
- Proposal factors, L grids, evidence-rung multipliers, and chunk/call ceilings
  are documented bounded search choices, not optimal settings for every target.
- This is a completed local engineering repair and audit. No external review,
  commit, push, training run, or real-data posterior campaign was performed in
  this task. The twelve pre-existing dirty tracked files were preserved.

Use this note and the repair plan as current status. Older unification plans,
audits, failed attempts, and numerical results remain historical evidence; their
deferred-implementation prose is not the current interface specification.
