# Full80 Walnutpie comparison — continuation state

## Completed, 31 October 2026

The requested comparison is complete. `comparison-01` stopped normally at
12:46 Shanghai with 105 adaptive transitions and 1400 retained draws per
chain. All posterior screens passed: max R-hat 1.005552, min bulk/tail ESS
1435.83/953.43, max controller MCSE/SD .02775. All 80 mean differences from
the saved HMC reference passed the predeclared simultaneous MCSE screen;
compatibility is established only at available precision. Identity latent
mass remained fixed, and both warmup and retained GPU/XLA graphs executed.
Supervisor exit code is 0; no worker remains active. Do not relaunch.

Charged additional GPU-worker time is 72285.101 seconds (20.079 hours), leaving
179714.899 seconds (49.921 hours) unused. Qualification's 1702.288 seconds
remain separate. Saved HMC worker time was 5.823 hours; the 3.447 observed
worker-time ratio is descriptive because warmup/recovery costs differ.

Terminal review verified ten source snapshots and 2409 saved sample/checkpoint
hashes, then reproduced both streams' posterior diagnostics and the comparison
from saved draws. Read `terminal-review-02/checks.json` and its manifest. The
first review attempt failed broad model-source binding: MacroFinance files
were edited after target compilation. Preserve `terminal-review-01/failure.json`.
The successful diagnostic imports only the checksum-verified pinned diagnostic
runtime and never imports/evaluates the changed model. New sampling using the
old binding remains blocked until the bound model closure is recovered;
do not change expected hashes to erase that limitation.

The final explanation, decision/inference tables and post-run review are in
`bayesfilter-full80-walnutpie-result-2026-10-30.md`. The older active/pending
statements below are historical and superseded. No Git commit/push was requested.

## Active run, 30 October 2026

`comparison-01` launched at 08:41:46 UTC / 16:41:46 Shanghai on GPU1
(`GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`). Persistent supervisor PID
1054363 owns worker PID 1054364. Do not launch a duplicate. The fresh
four-chain FP64/XLA run uses identity mass, author .80 target and .1 initial
step plus search. Target scope matches the saved HMC binding and GPU memory
growth was verified before logical initialization. The worker is compiling
the target; no new posterior results are available at this update.

New allowance: 252000 GPU-worker seconds; first worker hard cap 251900.
Control/status and complete numerical evidence are under
`docs/benchmarks/artifacts/full80-walnutpie-20261030/comparison-01/`.
`supervisor.json` is authoritative for elapsed charges and termination;
`controller.json` records committed chunks once initialization completes.
Do not infer process absence from sandbox PID visibility. Read trusted process
status or the supervisor outputs. The prior qualification remains charged
separately. A fresh check of budget, controller handoff and resume rejection
passed: three focused tests in 1.99 seconds, ten deselected.

The remaining sections retain the prelaunch history and are superseded by
this active-run record where they describe pending authorization or no worker.

**Authorization update, 30 October 2026:** the owner granted 70 additional
GPU-worker hours. The pending-permission paragraph below is historical.
The runner now enforces 252000 total additional seconds across comparison
attempts; comparison-01 is prepared with a 251900-second worker cap. Execute
the corrected .80 protocol without requesting the same permission again.

User request: test Walnutpie against MacroFinance's trained full80 NeuTra HMC
with fixed identity latent mass. BGL80 was resolved to DZ5 financial full80;
the separate BGL135 model is not used. MacroFinance remains read-only.

The initial GPU qualification has ended. No worker from this task is intended
to remain active. `qualification-01/supervisor.json` records exit -9 from its
1700-second hard limit, 1702.288 seconds charged. Reconciled manifest/result
state that frozen sampling did not complete. Preserve all pilot evidence.
There are four adaptive transitions per chain and no retained posterior draws.
Target/map/score parity, row isolation, identity mass and endpoint caches passed.
The saved HMC baseline reconstruction passed all original screens.

The corrected comparison executable is
`docs/benchmarks/full80_walnutpie_sampling.py`, with separate support, sampling
helpers and reporting modules. Full source-protocol rationale and proposed
command are in `bayesfilter-full80-walnutpie-comparison-2026-10-30.md`.
Final focused run: 12 tests passed in 7.04 seconds; deliberately CPU-only.

The pilot used .70 inherited from HMC. Do not resume it into posterior work.
Fresh corrected work uses the author's .80 base-macro quality target, .1
initial step plus search, 50--1000 adaptation, five-step publication, .10
cross-chain relative step tolerance, per-chain handoff, and fixed identity mass.
Keep all frozen-kernel outputs including rejections. No extra HMC burn-in or
acceptance-band qualification is imposed. Posterior compatibility still needs
all 80 physical R-hat/ESS/MCSE screens and uncertainty-aware mean comparison.

Permission pending: an asynchronous question asks for up to 50 additional
GPU-worker hours, a two-hour corrected-protocol pilot, or pause. Do not treat
elapsed time as approval. The existing 1800-second qualification allocation
has only 97.712 seconds left and is not enough for another target compile.
Do not spend MacroFinance's historical remaining budget. The early timing
scenario is 30--50 hours including heuristic reserve; it is based on only
two .70 warmup transitions and is not a reliable prediction of .80 sampling.

If a 50-hour grant arrives, recheck GPU availability with the trusted probe,
launch the documented fresh comparison with trusted permissions and verified
growth, monitor the first 50 adaptation transitions and phase handoff, and
continue independently until completed or a real budget/validity boundary.
If only two hours is granted, use that aggregate cap and label resulting
cost/mixing evidence as a pilot; preserve any corrected-protocol checkpoint
for an eventual identical-source continuation. A future resume must match
policy, source hashes, target binding and complete checkpoint state.

No git action is requested for these new files. Preserve unrelated dirty work.
The previous commit/merge/push task was already completed and must not be
repeated. Final result note is `bayesfilter-full80-walnutpie-result-2026-10-30.md`.
