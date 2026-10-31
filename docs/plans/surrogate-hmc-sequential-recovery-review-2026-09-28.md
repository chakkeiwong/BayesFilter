Execution update,2026-09-28: recovery01 failed during selection loading before
any M13 tuning or sampling. The localized repair below passed92 tests and an
actual GPU M13 construction/preflight. Recovery03 is authorized within the same
60000CPU/60000GPU allocation. Its live status will be in
artifacts/surrogate-hmc-m13-sequential-recovery-03/.
Earlier pending-approval and75-test statements below are historical.

# Surrogate-HMC master repair review — 2026-09-28

The executable repair passes the checked engineering and Gaussian mechanics
criteria. The scientific goal remains testing whether an imperfect analytical
DCTR force can support reliable model-parameter inference through surrogate HMC.
This continuation tests M13 with the previously frozen endpoint and data1704;
it does not substitute exact-gradient HMC. LG1 stays accepted/closed. The
all-model campaign and M13 posterior qualification remain incomplete.

The active entry point is [run_surrogate_hmc_recovery.py](../../scripts/run_surrogate_hmc_recovery.py).
Its [prospective plan](surrogate-hmc-sequential-recovery-master.md) is part of
the existing [closure master](surrogate-hmc-closure-master-program-2026-09-22.md).

## Repairs and checked behavior

| Finding | Repair | Executable evidence |
| --- | --- | --- |
| Fixed500 warmup could stop before readiness | Minimum2000/chain, latest1000 window, modern R-hat<=1.05, every1000, max10000 | Failed-window continuation and warmup-cap tests; real GPU path |
| First failed1000-draw verification ended the candidate | Continue the same frozen kernel and chain states cumulatively; R-hat<=1.01, max10000 | Multi-look controller test verifies windows, continuity and distinct seeds |
| Retained handoff omitted fresh chunk seeds | Preserve every nomination/fresh seed and reject reuse | Public GPU runner continuation and reused-seed negative checks |
| Warmup failure could lack a useful report | Archive every250-transition health chunk; report readiness/counts/terminal cause even before verification | Health, maximum and pre-first-chunk budget tests |
| Parent loss could lose watchdog and usage | Independent guard, process-group cleanup, CPU/wall caps, durable receipts | Actual supervisor-loss and worker/descendant accounting tests |
| Stale parent summary could omit a late crash receipt | Reconcile from worker receipts; unresolved receipt stops continuation | Late-receipt and missing-receipt regressions |
| A new output root could reset accounting or attempt count | Carry measured prior recovery charges; one fresh scientific attempt | Cross-root budget/attempt regressions |
| Next phase needed manual interpretation/launch | Refresh next-phase-plan after each phase/chunk; qualified tuning automatically enters retained assessment | Real master check plus mocked execution of qualified and unqualified branches |
| Tiny diagnostic replay differences could stop execution | Recompute diagnostics; warn on value differences while enforcing the same health/readiness/stopping decisions | Roundoff-warning and altered-decision regression |

Retained assessment preserves the existing calibrated cumulative looks1000/2000,
R-hat<=1.01, bulk/tail ESS>=400, and simultaneous MCSE-aware reference comparison.
The retained maximum stays2000; expanding that comparison requires corresponding
uncertainty calibration. Acceptance remains explanatory, never readiness evidence.
All warmup and fresh tuning-verification draws are excluded from retained estimates.

## Call-chain review and execution

The master calls the existing M13 public consumer, which calls tune_hmc_kernel
with its typed force/endpoint binding. The verified TensorFlow implementation
runs the sequential controller through its existing typed chunk graph. Reload
recomputes health and stopping decisions from saved chunks before the verified
runner can provide retained draws. Source/binding identity, tensor hashes,
state handoffs and seed history remain checked. The real GPU qualification
exercised both an exact Gaussian force and a deliberately rotational imperfect
force through public tuning, reload, retained sampling and negative archive/
binding checks. M13's expensive repaired call chain has not yet been executed.

Final command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/scripts/run_surrogate_hmc_recovery.py check --output /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-sequential-check-02

Result:75 tests passed; both GPU/XLA Gaussian arms passed. All199 snapshotted
sources remained unchanged. CPU tests intentionally hid GPUs. GPU workers
verified memory growth and used TensorFlow/TFP with XLA. The report preserves
environment, seeds, Git identity, source snapshots, phase history, command logs,
resource receipts and next-phase plan.

| Gaussian mechanics arm | Retained maximum modern R-hat | Kernel parity maximum | Checks |
| --- | ---: | ---: | --- |
| Exact force | 1.000352584 | 0 | Passed |
| Imperfect rotational force | 1.000927592 | 0 | Passed |

These diagnostic values do not rank samplers or establish M13 convergence.
Final check cost153.110704 CPU-worker seconds and84.006540 GPU-seconds.
All measured development and both check runs total347.819418 CPU-worker
seconds and167.537321 GPU-seconds, carried into the proposed recovery allocation.
Supervisor CPU is separately reported; historical interrupted CPU remains unknown.

Evidence: [final review summary](artifacts/surrogate-hmc-sequential-check-02/review-summary.json),
[master report](artifacts/surrogate-hmc-sequential-check-02/report.json),
[run manifest](artifacts/surrogate-hmc-sequential-check-02/run-manifest.json),
[GPU qualification](artifacts/surrogate-hmc-sequential-check-02/gpu-smoke/qualification/report.json).
Check01 is preserved as evidence before the final accounting/materiality repairs.

## Execution decision and limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Engineering repair passed |75 regressions and both real public GPU checks pass | No tested mechanics veto; missing historical CPU still prevents claiming an old balance | Expensive M13 mixing and downstream agreement | Approve the separately requested allocation, then start this master's M13 continuation | M13 posterior qualification or all-model completion |
| M13 posterior decision pending | Retained gates have not run | No new M13 candidate has been evaluated | Can readiness and verification fit the hard compute caps? | One fresh attempt, then conditional retained assessment | Rejection of surrogate HMC |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Mechanics, archive identity and process-loss checks pass; no M13 scientific verdict |
| Statistically supported ranking | None |
| Descriptive-only differences | Gaussian R-hat, means, runtime and acceptance |
| Default readiness | Broad posterior/default readiness not established |
| Next evidence needed | Proper M13 warmup and cumulative verification, then retained reference agreement |

The additional60000CPU/60000GPU allowance is still awaiting the user's response.
This is an expanded-compute boundary under AGENTS.md, not a new review gate.
The executable master has run its repair-check phase. No new M13 research run
was launched. No Claude review was requested.

After approval, use the following already allow-listed command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/scripts/run_surrogate_hmc_recovery.py start --output /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-m13-sequential-recovery-01 --cpu-budget 60000 --gpu-budget 60000

The start action detaches the supervisor from the terminal. It executes checks,
GPU qualification, M13 public tuning and conditional retained assessment without
another manual launch between phases. Draw and resource ceilings remain binding.
A killed supervisor triggers safe worker shutdown and usage recovery; automatic
restart of a partially completed HMC phase is not implemented. Do not claim that
archiving chunks alone provides a resumable sampler.

The rules file has63 entries after two narrow additions. The exact check, start,
execute and status command forms above were validated against the installed file.
See [actual-command validation](artifacts/surrogate-hmc-sequential-development-01/actual-command-allowlist-validation.json).
The existing broader interpreter permission is unchanged. Shell execution rules
do not waive scientific budget approval or unrelated tool permissions.

Post-run red-team: the strongest remaining alternative explanation for M13
nonqualification is insufficient initialization/mixing time or a resource cap.
The weakest scientific evidence remains the absence of a repaired M13 retained
run; passing Gaussian mechanics cannot close that gap. The next phase is part of
the same master and preserves the target, force, endpoint, precision policy,
untouched confirmation data and calibrated posterior criteria.


## Recovery01 launch failure and repaired consumer — 2026-09-28

The earlier Gaussian checks did not exercise M13's real saved selection. Its
loader correctly rejected a changed whole-file verifier hash. This was a
pre-sampling launch failure, not evidence against surrogate HMC.

Reviewed dependency chain: materiality_controls_stage -> four_chain_cost.chunk_health
-> hmc_tensorflow_verified._health. Force, endpoint, native HMC probe, selection
issuer, and33 other saved dependency hashes remain identical. The inspected old
verifier snapshot matches the selection's recorded hash. _health's only reachable
change adds no_force_fallback; the unchanged caller already conjuncts this exact
predicate. Six executable cases (healthy, fallback, nonfinite, bad endpoint,
divergence, energy failure) preserve the caller decision.

The new compatibility loader permits only this reviewed old/current hash pair,
checks the function AST and other dependencies, validates all preserved evidence
checksums and recomputes the original decision. It neither rewrites the selection
nor promotes a historical tuning artifact. Any further source change fails closed.
The actual selected_components and repaired-initialization preflight passed on GPU,
with selected multiplier0.001, verified memory growth and no inference launched.

Master changes: actual M13 preflight precedes tuning; a durable phase boundary
distinguishes preparation from sampling. Recovery01's complete receipt, exact
traceback and snapshotted call order prove no scientific attempt began. Its
review is hashed to those files, and every measured charge carries forward.
An absent/unknown boundary still consumes the attempt conservatively; an actual
sampling start cannot be hidden by choosing another directory. The next-phase
plan is still refreshed automatically.

Evidence:
- artifacts/surrogate-hmc-sequential-development-01/launch-repair-tests-01/command.log:
  92 tests passed;48.273494CPU seconds, deliberate CPU-only diagnostics.
- artifacts/surrogate-hmc-sequential-development-01/launch-m13-preflight-01/report.json:
  PASS;4.670508CPU/4.828041GPU seconds; no sampling.
- artifacts/surrogate-hmc-m13-sequential-recovery-01/m13-public-tuning/pre-sampling-review.json:
  reviewed failure classification and original evidence hashes.

| Decision | Primary criterion | Veto status | Uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retry within current allocation | Engineering regression and actual preflight pass | No changed selected force/endpoint; unknown/source drift still blocks | M13 mixing remains untested by repaired sequence | Start recovery02, same public tuner and subsequent conditional retention | M13 posterior qualification or all-model completion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No M13 candidate executed; preparation failure repaired |
| Statistically supported ranking | None |
| Descriptive-only differences | Test/preflight resource use only |
| Default readiness | Not established by this repair |
| Next evidence needed | Fresh M13 warmup/verification and retained/reference assessment |

Post-repair red-team: a changed called health predicate or force dependency would
invalidate compatibility; mutation regressions reject both. The weakest previous
check was Gaussian-only wiring, now supplemented by actual M13 setup. Construction
still cannot establish runtime health or convergence, which the next phases test.

Remaining allocation before recovery02: {'cpu_worker_seconds': 59444.052611, 'gpu_seconds': 59737.51661103399}.


### Recovery02 report-schema correction

Recovery02 passed92 CPU tests and both real GPU qualification arms. The new
supervisor condition expected a top-level status field, but the actual report
stores the verdict in its arms. The supervisor therefore stopped before M13
preflight or tuning. Its terminal candidate-not-qualified label was wrong:
no scientific candidate had run. The original report is preserved.

The repaired reader checks both exact/imperfect arms, all recorded checks, XLA,
memory growth, parity and completed archive/continuation checks. A saved actual
report regression now exercises that format, with negative cases for missing
arms, incomplete checks, failure and invalid parity. Execution failures retain
EXECUTION_REPAIR_REQUIRED even when the subprocess itself exited successfully.
20 focused orchestration tests passed in0.08s (0.256696CPU including guard).
No scientific target or criterion changed; this is a harness repair.

Remaining before recovery03: {'cpu_worker_seconds': 59334.016332, 'gpu_seconds': 59684.95771071303}. Same approved allocation.
