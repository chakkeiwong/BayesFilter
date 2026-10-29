# E5 isolated DZ5 initializer and target integration

Target comparison is complete through 03808; see the [result](filter_gradient_dz5_target_result_20260925.md).
The notes below preserve execution and failure provenance. Adapter admission, fresh
independent score oracle, initializer and staged-lifecycle qualification remain open.

Current execution checkpoint, September 25: repair merge `9d8202b77` is pushed.
The new read-only snapshot `dz5-candidate-source-merged-9d8202b77-r1` contains
721 sources and the same three frozen input artifacts. Its manifest SHA-256 is
`fade74c7e959ad0b33dbc6116a62f8ea44a33fee9a3042e37bf438b3e828c259`.
No numerical qualification is inferred from snapshot construction. The original
three import attempts consumed 31.909861 seconds / three workers; all new work
uses the remaining allocation below.

The first step-2 checks compare candidate graph/XLA on the actual CDF callback at
B=1/4/46/68, with initial, changed and exact replay inputs. B=4 uses the original
CDF truth/prior bank and original invalid-row mutations (NaN at [1,18], -100 at
[2,22]); require exact rejected values/zero scores, unchanged valid rows, status
equality and existing 1e-8/1e-7 value/score tolerances. Require one stable trace,
unchanged HLO, no host callbacks and post-execution source/module checks. Save
failed numerical records as well as passing ones. These are partial engineering
checks until the archived-source comparison passes; they cannot issue admission.
FP64 with TF32 disabled matches the original CDF qualification. The sine input
perturbations are deterministic coverage probes, not a sampling distribution.

GPU children preserve the runner-selected non-desktop UUID, expose the host device
files, and verify growth before creating
logical devices. CPU children retain a private minimal device mount. Both bind
the entire project snapshots read-only. The parent records exact child commands
and hashes; the child records device settings and allocator current/peak bytes.
These checks measure numerical behavior; shared graph/XLA process timings and
RSS are explanatory only and cannot establish before/after cost or leak freedom.
Run the stable runner's `test --group dz5_merged_import_cpu --device CPU
--test-timeout-seconds 300`, then each `dz5_merged_target_<B>_<cpu|gpu>` group with
the matching device and a 900-second parent / 840-second child bound. One worker
at a time. Stop on unexpected numerical or provenance failure and localize under
the unchanged retry/cumulative limits before continuing the batch ladder.

Skeptical review of this continuation found two defects in the prepared draft:
it omitted invalid-row isolation and could lose nonfinite failure records during
JSON writing. Both are repaired before launch. A graph/XLA match alone cannot
establish an unchanged target: the archived CDF target also predates a harmless-
in-CDF but unqualified-by-this-unit enclosing-asset argument extension. Compare
the exact archived target separately, record source differences, and do not
relabel it as the identical callback. Existing analytical-score tests and the
archived CDF qualification supply comparison rationale, not fresh admission.

03787 fails before target evaluation because snapshot r1 omitted the untracked
`_symmetric_sylvester_ops.so` import dependency. Preserve r1 and the failure.
Snapshot `dz5-candidate-source-merged-9d8202b77-r2` copies that exact prior frozen
binary (SHA-256 `cc2ad31f8eab27bb90449b7e77eae0c6aac34ab7ac4dfe4662510a0eb96a7022`)
without a rebuild or installation; it contains 722 source/dependency files and
the unchanged three inputs. Its manifest SHA-256 is
`b416266f4c22e64fb4e746271e7ae5920e297ac662cb4a331717741c727330f5`.
The three intentionally retired GenUT modules are not restored. Presence of the
native binary is an import prerequisite; target GraphDef/HLO must still exclude
host custom-op callbacks. This localized source-closure repair consumes the same
E5 allocation; 03787 adds one worker and 10.087955 seconds.

03788 passes the repaired snapshot import/fixture/source check in 13.996866
seconds. The cumulative E5 allocation now uses five workers / 55.994682 seconds.
The independent comparator snapshot `dz5-archived-cdf-source-31f0067f9-r1`
reconstructs the exact original CDF source archive plus its pinned continuation
supplement: 158 source/dependency records and 124 input records, with zero source
overrides. Its manifest SHA-256 is
`c6bbb4daaa91636d30376522f11d8fb499b7e58cd32ae51f6d1b269e6a8b1b42`.
The comparator imports only target dependencies and checks every actual project
module against those archived hashes; it does not require a newer initializer.
Run its registered `dz5_archived_target_<B>_<cpu|gpu>` groups as independent
reference checks. Compare saved inputs, complete target values/scores and exact
integer/Boolean statuses to candidate records at the original tolerances. A
baseline XLA numerical failure must be reported and localized; it cannot qualify
a before/after speed ratio. Both sources remain in separate child processes.

03789 fails during outer XLA compilation, before a completed target evaluation:
the draft attempted to return the raw API's string metadata, causing unsupported
`_Retval DT_STRING`. Use the existing `credit_target_numeric_outputs` boundary
for XLA, exactly as original CDF qualification does. The graph reference selects
the same nine numeric outputs from its explicit non-JIT call. Strings remain
outside numerical execution; status codes, validity and chart/support diagnostics
remain compared. No target, filter, tolerance or snapshot changes. Preserve the
failed report (its pre-call `target_evaluated=true` flag is a harness error; the
exception establishes that no evaluation completed) and correct that flag to
be set after a synchronized completed call. This consumes another worker and
24.519226 seconds; the same unit has 18 workers remaining before the retry.

03790 passes the merged B=1 numerical/trace/HLO/replay/source checks in 47.619811
seconds. Archived B=1 attempt 03791 stops during import (10.337432 seconds): its
old lazy inference facade tries to import unrelated `hmc_stage_resume` while
resolving `ValueScoreCapability`. The original pinned CDF qualification explicitly
primes that class from `posterior_adapter` before importing the target (its
`qualify_dz5_cdf_runtime.py` lines 79--82). Match that existing import sequence in
the archived diagnostic child. No snapshot, numerical source or class is
substituted; the class's defining module and every imported byte remain audited.
This is an original-harness compatibility repair, not an admission refresh.

03792 passes archived B=1 in 45.414304 seconds. Direct comparison with merged
03790 finds identical inputs, values, scores and discrete statuses for all three
initial/changed/replay records. Remaining target checks are registered as
`matrix --stage tests --test-batch dz5_remaining_target_cpu` and
`--test-batch dz5_target_gpu` (900-second workers). Each ordered batch/old-new pair
uses a fresh worker; stop on the first failure. Input generation stays on CPU
so GPU comparisons consume identical clouds, and result-device assertions verify
that numeric outputs execute on the declared backend. This does not alter the
already checked CPU fixture. The E5 unit has used nine workers / 183.885455
seconds; the six remaining CPU and eight GPU workers fit the 24-worker ceiling.
Subsequent initializer/staged integration requires a separately recorded bounded
unit within the same cumulative campaign budget, because these target checks
consume the original unit's available worker slots.

03793--03798 pass all six remaining CPU workers. The four explicit old/new
comparisons in `dz5-target-cpu-comparison-03798.json` pass every saved value,
analytical score, discrete status and numeric chart/support diagnostic. The
analyzer also verifies source/command/HLO checksums and loaded-module import
policy, and rejects five adverse mutations. The first GPU matrix launch is
rejected by the runner before creating a worker because GPU group registration
was missing. Add those eight groups to the existing `TEST_DEVICES` table; do not
weaken its fail-closed device check. This preflight used no numerical worker.

03799 fails before target evaluation: the GPU parent sees its selected UUID and
verifies growth, but the read-only-root child fails CUDA initialization. A bounded
driver-only trace identifies `openat(/proc/self/task/<thread>/comm, O_WRONLY)`
returning EROFS while CUDA creates a helper thread; `cuInit` returns 304. The
paired probe with bubblewrap `--proc /proc` returns 0. Add that ordinary proc
mount to GPU children; both project snapshots remain read-only. No numerical
code, binary, device choice or target setting changes. Preserve both traces in
`dz5-gpu-container-probe-r1/r2` and charge each full 30-second probe reservation
through supplemental GPU records (60 seconds total, within the 14,400-second
unit cap). These driver-only probes created no numerical tensors or contexts.
03799 consumes one of the 24 numerical worker slots and 9.736150 seconds.
There are eight slots left for the repaired GPU target matrix; subsequent policy
and initializer units retain their separate campaign allocations.

This is accepted TF/TFP execution-repair integration under the master E5 phase.
The selected consumer remains the completed 23-parameter CDF preparation route,
with initialization batches1/46/68 and training batch64. Current MacroFinance
memory has advanced to MD work; CDF retained-r5 is complete. This unit is a fresh
BayesFilter engineering qualification, never a restart of CDF or an intervention
in the active MD controller. Preserve every live source and campaign artifact.

Read-only call-chain review confirms
`scripts/prepare_dz5_cdf_proposal.py::main` calls
`bayesfilter_estimation_initialization.initialize_dense_local`.
Its `CreditTrainingTarget.batch_value_score_and_validity` is already atomic.
The shared initializer still owns Python attempt/partition recurrences, NumPy
cloud materialization and a locator configured with jit_compile=False. The
candidate thin adapter maps the same configuration to the seeded BayesFilter
controller, then formats completed records and writes NPZ without NumPy.
There is no target, model, score, prior, training or tuning change in the adapter.

The comparison must bind the exact initializer numerical closure. The earlier
fitter reference defect proves that copying only the wrapper is insufficient.
Use the fully pinned3582b4ac initializer reference already checked in the dense
unit, the frozen original external initializer, and one identical actual CDF
callback. Separately qualify that callback on the repair branch against the
archived CDF target source and the explicit graph reference. Never substitute
the completed CDF sampler's qualification for evidence on changed BayesFilter
bytes. Preserve original artifacts even when their numerical code differs.

Before a numerical launch, create versioned complete source trees for candidate
BayesFilter and the relevant MacroFinance source/data closure. Record Git
commits, dirty-file hashes, original qualification/source archive hashes and all
copied inputs. Mount the complete trees read-only at their canonical paths in
an isolated child. Do not patch canonical path checks or rely on changed live
BayesFilter modules. Missing lazy dependencies fail the smoke and trigger an
explicit source-closure repair. Audit actual loaded modules against the snapshot
and scan active imports for forbidden MacroFinance-local filtering/HMC modules.

The original CDF admission binds112 target-source files, including BayesFilter
dependencies. It is stale for this repair branch by design. First run the exact
target qualification without issuing a new adapter admission. Only passing fresh
target/device/score/rejection evidence may issue the engineering admission used
to construct the real CreditTargetAdapter. Preserve its non-retention scope;
never update old qualification hashes to make them appear current.

Execute the following bounded ladder through the stable campaign runner:

1. Standard-library source snapshot/usage audit and isolated CPU import/shape
   smoke. Confirm fixture identity,23 parameters,96 observations, the original
   coordinate/filter contract and read-only qualified TFP special.py hash.
2. Exact target value/analytical score, row validity, changed operands and replay
   at batches1/46/68 (plus4 for transitions). Require the existing CDF1e-8
   absolute/1e-7 relative comparison, exact status/rank/support decisions, one
   trace per declared shape and HLO without host callbacks. Graph/CPU are explicit
   reference arms. Use the original invalid-row isolation check.
3. Thin-adapter original-record/NPZ/row-accounting comparison, first on the
   already-qualified D1/D3 fixtures and then on the actual CDF target and frozen
   recipe. Preserve every initializer decision and the original starts/seeds.
   Pending isotropic/condition-record findings remain vetoes; do not normalize
   them away. A bounded rejected initializer can qualify error behavior but
   cannot replace a healthy accepted-initializer comparison.
4. Put actual initializer calls under the exact qualified independent DZ5
   parent supervisor. Check quiet healthy completion, timeout termination,
   descendant cleanup, two sequential worker lifetimes and parent RSS recovery.
   No callback heartbeat is required inside a compiled numerical stage.
5. Exercise a tiny excluded transition with the existing BayesFilter archive
   runner and actual adapter, retaining target-status and replay diagnostics.
   This is mechanics only. The HMC tuning interface and capability registry
   remain authoritative; the chain runner must not issue a tuning artifact.

Reserve at most24 workers /14400 charged seconds across both devices within
the existing32CPU/52GPU caps. Use300-second focused workers and900-second actual
target/initializer workers; stop at first unexpected numerical/provenance
failure. Failed work consumes the same allowance. At most three localized
harness retries per unchanged case; no repeated unchanged numerical failures.
Run one numerical worker at a time, with verified GPU growth and automatic
non-desktop selection. Record source hashes, exact commands, data/seeds, device,
elapsed time, JUnit and full records in the numbered master output directories.
Matched before/after cost comparisons additionally require clean sharing and a
matching physical UUID. Stop for scope change or exhausted cumulative budget.

Review: the largest risk is silently importing the live canonical checkout or
self-stamping an admission from stale CDF evidence. Complete source mounting,
actual-loaded-module auditing and fresh target evidence address those errors.
The source/initializer compare uses identical callbacks to isolate execution;
separate target qualification covers filter changes. Early rejected paths and
tiny transitions cannot establish healthy initialization, full HMC readiness,
posterior correctness, identification or native in-process eviction. No
MacroFinance deployment or final merge follows from a partial ladder pass.

Import harness failures03704 and trusted repeat03705 both fail at the first
TensorFlow eager context creation: std::random_device cannot open its device
through the read-only root bind. This is independent of target/filter numerics;
neither worker evaluates the target or constructs an admitted adapter. Add
bubblewrap's private minimal `/dev` mount for CPU random/null devices and direct
ordinary plotting caches into the existing private `/tmp`. Both project trees
remain read-only and every source hash is still checked before/after. GPU hiding
and the import-only scope are unchanged. This localized harness repair consumes
the existing24-worker/14400-second unit; preserve both failed child reports.

03706 passes the isolated import/fixture check in13.942 seconds. All572 snapshot
source hashes match before and after, every loaded project module matches its
recorded source, both canonical project mounts are read-only, and the exact
23-parameter/96-observation fixture, filter contract and pinned TFP special hash
match. The preparation snapshot predates the five-file SVD/ownership change;
it must remain immutable. Create a fresh versioned snapshot binding the current
runtime for step2, then repeat import verification before numerical evidence.
No target evaluation, adapter admission, training or retained chain was run.
The three import attempts consume31.909861 seconds of the existing unit.

September25 architecture boundary: the frozen CDF recipe and any old learned or
affine NeuTra map are historical under the owner's new canonical IAF directive.
The generic target/initializer numerical checks in steps1--4 do not qualify a
neural architecture and may still identify execution defects. Step5's excluded
legacy consumer, if retained, is historical mechanics only and cannot close
current NeuTra integration. Before current NeuTra handoff, integrate and test
`NeuTraTransportConfig.hoffman_author_iaf` through the single shared numerical
authority, with target-specific configuration provenance and no reuse of old
maps as canonical evidence. No new training recipe or architecture departure is
authorized by this note. The new572-source snapshot
`dz5-candidate-source-ea94aac96-r1` binds the repaired numerical checkpoint but
predates canonical transport integration; keep it immutable and labeled.
