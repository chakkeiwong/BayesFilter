# Frozen HMC continuation: live-budget check repair

27 October 2026. The original qualification stopped cooperatively at 18:50
Asia/Shanghai with 608 completed work trials and three verified candidates.
The automatic resume failed at 18:51 before loading its checkpoint. The
campaign is incomplete. Its settled balance is 260,052.56232782436 inclusive
seconds; this repair uses that balance, without another credit or a second
charge for the original run. Separate SIR funding is unchanged.

## Root cause and research intent

The frozen worker passed `replace_wall_time_limit=True` to `execution_budget`
with a deadline but without the explicitly required live `check` callable.
The runtime correctly rejected that invocation. The resume produced no new
numerical evidence. The coordinator then tried to read a checkpoint that had
never been copied, masking the original error with `FileNotFoundError`. Prior
tests exercised numerical recovery directly and mocked the outer lifecycle;
they did not exercise the actual worker's resume branch.

The engineering question is whether the same frozen checkpoint can continue
under the newly authorized finite deadline, with its target, numerical source,
candidate set, seeds, observations, completed trials and GPU UUID unchanged.
The comparator is the already tested frozen `copy_parent`, `resume_checkpoint`
and export/reload implementation. We add a host-only recovery entrypoint that
composes those functions, supplies a real live deadline check, and records its
own source separately. The original snapshot and evidence remain untouched.
The shared-checkout worker also receives the missing check for future runs.

Pass requires the real budget context in the worker path, resumed evidence
prefixes unchanged, no replay of completed trials, correct deadline exhaustion,
exact verified/exported/reloaded membership, and visible primary errors even
when no checkpoint exists. Invalid source/data, numerical identity mismatch,
corrupt evidence and missing required runtime provenance veto continuation.
Budget exhaustion is resumable; candidate movement/acceptance failures do not
stop other candidates. R-hat remains reporting-only. Timing is explanatory.
This repair establishes neither posterior convergence nor release readiness.

## Skeptical audit and assumptions

Changing the frozen harness in place would invalidate its source-bound
checkpoint. Instead, the external host entrypoint imports the original
snapshot's numerical modules and existing recovery functions. It records
its extra host source hash and verifies all original source hashes before and
after work. It cannot rebind or modify the original numerical specification.
The entrypoint deliberately does not copy or replace a sampler/controller.

The live check uses the actual supervisor-owned monotonic deadline, retaining
the original 900-second export/reload and 120-second supervisor closeout
reserves. The hard supervisor still bounds in-flight native calls. The GPU
UUID, below-50% launch screen, 4,096-MiB headroom, 60-second queue checks and
1,800-second reassessment are inherited resource choices, not scientific gates.
Verified memory growth, GPU/XLA, float64 and TF32-off are unchanged. Seeds and
all numerical thresholds come from the preserved checkpoint and cohort.

The plan passes skeptical review: it repairs an API-call mismatch rather than
changing a statistical policy or weakening the runtime guard. Regression tests
must traverse the actual worker with the real budget context, not mock away
the failed call. A real numerical interrupted/resumed test complements these
host tests. A fresh GPU continuation must show checkpoint advancement before
we report recovery as working. No NUTS route or other method is introduced.

## Implementation and execution

1. Correct the shared harness's deadline-check call and coordinator error
   precedence. Add explicit carry-forward accounting for an already settled
   parent and a separately recorded external recovery entrypoint.
2. Add the host-only frozen recovery entrypoint. Reuse original target build,
   checkpoint copy/load, controller resume, native audit, export and reload.
   Preserve prior evidence checksums and record all numerical source imports.
3. Run CPU engineering tests with GPUs hidden: actual worker resume branch,
   expired/enclosing budget, error before checkpoint, unchanged old evidence,
   zero extra credit and no double charge, plus existing numerical recovery,
   source-isolation and GPU-selection tests. Cap each test invocation at
   300 seconds, an operational debugging ceiling; charge actual elapsed time.
4. Launch one repaired continuation from the original valid checkpoint under
   `N5/budget-resume-repair-20261027-01`, using the remaining settled balance
   after diagnostic costs. Never resume the failed empty child directory.
   Use trusted GPU execution and the original snapshot custom operations.
5. Confirm fresh checkpoint progress and refresh the master. At terminal
   closeout compare all verified/exported/reloaded members, settle inclusive
   elapsed time once, and record the next justified master phase. Unexpected
   errors are repair triggers requiring diagnosis, not blind retry.

Exact launch command, environment, source hashes, original binding, wall time
and artifact paths are recorded in the new allocation, launch receipt and
worker/supervisor manifests. The failed October 27 extension remains preserved.
Unused time belongs to the existing campaign, not a fresh 72-hour grant.

## Pre-mortem

A successful import could hide loading current shared sources; inspect actual
module paths and hashes. A harmless mocked lifecycle could miss the real budget
API mismatch; invoke the actual context manager and worker branch in tests.
A restarted checkpoint could duplicate evidence or grant a new statistical
budget; retain the original recovery audit and only its documented lost-call
allowance. An error before checkpoint creation could again hide its cause;
read the worker error first. A successor ledger could charge old time twice;
test carry-forward independently and preserve the prior terminal settlement.

## First regression pass

117 tests passed, including all new actual-worker budget-context checks and
the interrupted numerical recovery check. One nine-candidate integration
stopped with `shared_invalidity`: its checkpoint records that
`bayesfilter/inference/hmc_candidate_set_tuning.py` changed during the test.
Another task is editing the shared checkout. This is a source-isolation
failure of that test run, not an observed numerical or recovery defect.
The failed attempt costs 50.875963799 seconds and remains recorded.

Before launch, repeat the combined regressions in a separate temporary test
tree containing the unchanged campaign numerical snapshot plus only the
repaired host scripts and test files. This isolates the test against concurrent
edits without changing the active campaign snapshot or relaxing its source
checks. A read-only import check already confirms the new recovery entrypoint
loads the original frozen modules and all 1,063 source hashes still match.

## Verified repair and launch decision

In the isolated run, 115 tests passed, including the numerical nine-candidate
qualification/export and interrupted-recovery checks. Three mocked worker tests
failed only because the temporary tree has no Git checkout. Their Git-provenance
call is now explicitly mocked; all 41 tests in the affected worker/coordinator
blocks pass on rerun. Thus all 118 selected tests have passing evidence; the
earlier failed attempts remain recorded and charged. The isolated numerical
source did not change during execution.

The new host entrypoint loads the original frozen target, controller, copy,
resume, audit, export and reload functions. It supplies both an explicit
deadline and the required live check. Tests verify the old cumulative clock
can be replaced while an enclosing expired budget still stops execution.
The coordinator now reads a primary worker error before looking for a
checkpoint, carries an already settled balance without charging its parent
again, and uses a separately hashed host entrypoint.

The two isolation-related failures were test setup failures, not evidence
against the candidate method. The repair has no new statistical defaults.
The launch remains diagnostic and completion is not promised by the budget.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Resume original valid checkpoint | Actual-worker budget API and numerical recovery tests pass | All original source hashes match; no unresolved tested recovery veto | Real target resume/replay duration and candidate outcomes | Trusted GPU continuation, then verify fresh trials | Posterior validity, ranking, default or release status |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Source and evidence checks retained; errors before checkpoint remain visible. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Counts and elapsed times. |
| Default-readiness | Not established. |
| Next evidence needed | Fresh checkpoint advancement, complete qualification/export/reload, then master C2/C3. |

Post-repair red-team: mocks alone cannot establish real-target recovery. The
real numerical recovery test covers checkpoint mechanics; the resumed GPU run
must still demonstrate advancing evidence before recovery is reported healthy.

## Recovery confirmed — 27 October 2026, 23:09 Asia/Shanghai

The repaired detached coordinator launched at 23:05:29, PID 81020. Its
supervisor was PID 81025 and its numerical worker PID 81032. The worker used
the original GPU1 UUID, verified pre-initialization memory growth, TF32-off
and the frozen GPU/XLA source. The checkpoint loader retained the original
binding, candidates, observation prefix and accounting prefix. No lost-call
gradient allowance was needed. Fresh numerical chunks confirmed progress
beyond the prior checkpoint. This was the live status at 23:09; the terminal
outcome is recorded below. See `recovery-progress-confirmation.json` under
`N5/budget-resume-repair-20261027-01`.

All diagnostic test attempts consumed 178.446859005 seconds. The launch
carried 259,874.115468819 seconds after tests, before ongoing inclusive runtime.
No new credit was issued, and neither the original run nor the failed resume
was charged again. The master was updated; terminal closeout settled this
continuation and audited the exported/reloaded set.

## Terminal qualification and follow-up — 28 October 2026

The resumed numerical qualification completed at 05:41 Asia/Shanghai. Its
terminal checkpoint records **1,312 completed trials** across the frozen
nine-candidate cohort, with no missing returned chunks and no saved partial
chunks. Seven candidates passed the frozen acceptance/health verification;
two failed the candidate-local movement gate (L=3, epsilon=0.8 and L=5,
epsilon=0.8). All seven passing candidates were exported, fresh-process
reloaded, endpoint-checked and continued with finite samples. The two failures
remain failed and were not retried or relabeled. R-hat was reporting-only.
These outcomes do not establish posterior convergence, consumer admission,
ranking, default promotion or release readiness. The authoritative records are
`N5/budget-resume-repair-20261027-01/result.json`,
`result.md`,
`qualification-resume-01/evidence/result.json`,
`native-audit.json` and the terminal tuning checkpoint.

The CPU-only follow-up command
`CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_modern_hmc_snapshot_resume.py tests/test_modern_hmc_budget_extension.py tests/test_modern_hmc_frozen_diagnostic.py`
passed **50 tests in 84.50 seconds**. The checkout at test time was based on
`d0705f7956a6e224b46cb478667ddd82b0953d32`; its unrelated concurrent edits
were left untouched. The prior terminal settlement records 236,112.308665
seconds. Charging this follow-up suite leaves approximately 236,027.809
seconds (65.56 hours) in the combined allowance; the separate SIR balance is
unchanged. This is an inclusive remaining budget, not a completion-time
estimate.

The next phase is C2 exact-source public admission. Inspection confirmed the
MacroFinance Phase 9 loader requires both a hard-coded BayesFilter commit and
a clean BayesFilter worktree. It expects
`43a7f113ddbadf13c6421b7b6d7b2d26f5a4f1e9`; this checkout is at
`d0705f7956a6e224b46cb478667ddd82b0953d32` and has extensive unrelated dirty
changes. The loader therefore rejects before importing TensorFlow or executing
the numerical target. This is a source-authority mismatch, not evidence of a
new target, score or sampler failure. Do not bypass the gate or mutate the
MacroFinance source pin without review. C2 must first establish an exact clean
commit containing the intended tested BayesFilter bytes, then obtain the
consumer's source-pin update and rerun its public entry point. Its Phase 9
launch contract is CPU-only; this integration check makes no GPU or posterior
claim.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close resumed qualification | Complete frozen cohort; native audit and seven-member export/reload sets agree | Two candidate-local movement vetoes remain recorded; no missing calls | Narrow synthetic MacroFinance geometry | Preserve all outcomes and advance to C2 | Posterior correctness, ranking, broad consumer admission |
| Advance C2 planning | Source mismatch was reproduced from the consumer loader and authority check | Public entry remains closed until exact clean-source pin is reviewed | Which precise BayesFilter patch set the consumer owner will admit | Build a source-diff/commit proposal, then rerun without weakening the gate | Consumer installation, GPU readiness, release |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Qualification accounting, source hashes, verified-set export/reload and endpoint continuation pass; two candidates failed the declared movement check. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Acceptance and candidate outcomes identify viable/rejected members but do not rank viable candidates. |
| Default-readiness | Not established. |
| Next evidence needed | Exact-source public admission, then separately designed posterior/reference/precision evidence. |

Post-run red-team: endpoint preservation and finite continuation do not verify
the consumer's whole public call chain. The strongest alternative explanation
for the C2 block is solely a stale source lock; until the source diff and public
route are inspected under one admitted clean commit, that remains a diagnosis,
not a repaired integration. A changed intended dependency or public-path
score mismatch would require a new C2 repair before posterior work.
