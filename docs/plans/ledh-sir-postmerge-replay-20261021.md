# Frozen SIR replay after the main merge

Status: COMPLETE. Both lanes contain2496 holdout rows and432 nonzero analytical
directional checks. Exact old-result tie-out and all validity checks pass.
Eight non-plain arms pass numerical parity; plain LEDH fails the frozen
tolerance. The negative scientific conclusion and heuristic vetoes remain
unchanged. Four launches charged4729.913960 of19800 seconds; no process is live.
See the [terminal result](artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/result.md)
and `comparison-02/` under that artifact root. The execution updates below are
preserved history, not instructions to resume completed work.

Active question: did merge `90a3ace` change the completed SIR moment results,
the finite-program analytical derivatives, validity, or evaluation cost? The
user authorized this paired replay on 21 October 2026. No fit is selected or
trained anew. Work is on `ledh-refactor-with-policy-fix` in the LEDH worktree.

## Evidence contract and intent

The baseline is the complete 834-module source snapshot of
`ledh-sir-neural-kdm-20261020/attempt-16-finalized/final-holdout/source`, including
the pre-merge uncommitted implementation, not the clean pre-merge Git commit.
Its hashes match the frozen neural checkpoint and the pre-merge WIP backup;
all snapshot files have been checked. The comparator is a new snapshot of the
merged working tree. Both execute the same diagnostic runner in isolated
processes. Imports must resolve to the requested snapshot.
The first launch exposed an omitted dependency in that historical snapshot:
`experiments.dpf_implementation` is imported transitively by the kernel. The
launch completed three heuristic rows but failed before particle evaluation.
The repaired isolated sources also include
the old Git version of tracked experiment modules and verified pre-merge
untracked modules, with a separate auxiliary hash index. The merged snapshot
includes its own experiment modules. Both use the same hashed custom-op binary;
its file modification date predates the original study, but that study did not
archive a binary hash. The old numerical tie-out therefore also checks this
dependency. CPU-only import and initial-cloud shape regression passes with GPU
devices deliberately hidden. This is a harness repair with controls unchanged.

Replay all 64 saved datasets and four original particle seeds, with the frozen
32-update neural KDM fit, its initial network, plain LEDH, conditional Gaussian,
the unchanged physical mixture, and adapted/unadapted quadratic cSMC-derived
and positive RBF proposals. Inspection of the restricted diagnostic files
confirmed they retain the full exported message coefficients. Load those saved
coefficients directly in both versions; no fitting operation is needed.
Require the old replay to reproduce the archived moment estimates/losses before
interpreting the new replay. Recompute all three existing cheap heuristics:
prediction alone, final-observation update, and recursive Gaussian updating.
Report them within the original high/low-observation and boundary/interior
groups; a loss to a heuristic remains a promotion veto, not a replay veto.

The primary engineering criterion is unchanged validity and numerical agreement
of values, standardized first/raw-second moments, moment losses, and analytical
parameter derivatives (absolute plus relative tolerance `1e-8`). Raw moments
use `1e-5 + 1e-8 * abs(reference)` because their units are much larger. Record
actual discrepancies, not just pass/fail. Historical score fields were evaluated
with the zero parameter direction and are not model scores. Compute all three
unit-direction analytical derivatives on four datasets selected before running:
the first occurrence of each observation/compartment regime, four particle
replicates, all nine arms. These are finite-program derivative parity checks;
there is no oracle score-accuracy claim.

Wrong input/fit hashes, source drift, invalid import isolation, failure to
reproduce old losses, missing paired rows, nonfinite values, new invalid rows,
or direction-dependent values invalidate an equivalence conclusion. Preserve
such failures and investigate a localized harness error; do not alter scientific
controls to force agreement. Valid finite differences between versions answer
the question negatively and require explanation, not automatic suppression.
Runtime is explanatory only: synchronized per-evaluation timings, first calls
separate from warm calls, same device, paired dataset summaries and bootstrap
intervals. Sequential processes and concurrent GPU use limit causal speed claims.
The reporting calculation averages paired warm calls within each dataset before
resampling datasets (5,000 percentile bootstrap draws, fixed seed263050001).
Drop a timing pair if either call is cold; report cold calls separately and
retain each process segment's origin. This interval describes the replay's
timing variation and cannot identify a causal speedup from sequential runs.

This is an engineering regression on previously exposed data. It cannot support
fresh generalization, guide selection, true-score accuracy, improved training,
canonical admission, new defaults, or broader TF32/long-horizon performance.

## Default audit and pre-mortem

All scientific settings come from the archived manifest: SIR d18, T5, N64,
theta0, FP64/TF32off, XLA, 24 flow substeps, OT `(1,5)`, balance32, K=N64,
Contract E and the same correction/ridge/damping/cap controls. Their purpose is
strict replay, not endorsement as universal settings. Their failure mode is an
invalid inherited candidate, exposed by archived-loss tie-out and retained
validity checks. Calibration arrays, observation/label arrays, reset seed,
particle seeds, network initialization and trained weights remain fixed.
The earlier candidate's failed moment-improvement criterion remains in force.

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, elevated GPU access,
`CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, memory growth
configured and verified before initialization, and the previously required
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` exception. The other GPU is busy.
An empty CUDA device list or wrong TensorFlow/custom-op ABI is a harness failure,
not scientific evidence. No pfor or autodiff score implementation is introduced.

The main misleading-pass risks are comparing zero directional derivatives,
using clean Git instead of actual old source, retuning on holdout, and timing
compilation as evaluation. This design explicitly excludes those mistakes.
Skeptical audit: PASS for a frozen engineering replay; comparator, scope,
uncertainty, heuristics, validity vetoes and nonclaims are explicit. Reconstructing
restricted fits is unnecessary; the archived messages and the old end-to-end
result tie-out freeze the complete comparison.

## Execution and budget

Artifact root: `docs/plans/artifacts/ledh-sir-postmerge-replay-20261021/attempt-01`.
Record source/runner/input/fit hashes, original and current commits, command,
environment, device policy, seeds, per-call timing, wall time and all results.
The exact commands are emitted to the run manifest before each launch.
Use `scripts/replay_sir_postmerge_diagnostic.py` with `--lane before` and
`--lane after`; both consume the archived frozen message files.

Total GPU process cap: 19800 seconds (the approved90 minutes plus four additional
hours), including probes and localized
retries; at most five launches. The user approved this extension after the
initial replay proved slower than projected. This is the newly authorized merge
diagnostic, separate from the completed training continuation. Start with the
complete before replay; verify old-result tie-out, then run the merged replay.
Inspect progress and stop a stuck process within this total cap. The first
failed launch charged35 seconds and the first repaired launch charged about
1281 seconds before its cap. The approved continuation may resume only
verified rows in a fresh output directory; it must not alter the grid or
controls. If the projected remainder still exceeds the cap, preserve complete
evidence and report the untested scope rather than silently reducing the paired
grid or changing controls.

Execution update: the repaired pre-merge process has reproduced the archived
moment results through35/64 datasets and324 directional checks. Its observed
cost exceeds the initial projection. The user approved the90-minute extension.
A restart-only harness change
can copy verified completed rows into a fresh output, checking source, input,
binary and lane identities and never repeating or changing particle draws.
Preserve the exact initial runner as `replay-runner-v1.py`; record each segment
and cold compilation separately. This is not a change to the numerical method.

After comparison, write the numerical/validity/runtime report, decision and
inference-status tables, strongest alternative explanation and next action;
refresh the concise checkpoint. Do not commit or push as part of this replay.

## Measured runtime update during the merged replay

The merged lane reached19/64 datasets and216/432 directional checks after
1227.430 seconds. Recent ordinary datasets take roughly45--50 seconds,
compared with roughly20--30 seconds in the pre-merge lane. Paired warm timings
through the first19 datasets are descriptively1.4--1.6 times longer for most
guided arms and3.2--3.5 times longer for plain/conditional arms. Concurrent CPU
work was observed; a causal performance attribution is not justified.

The original2600-second merged process cap may require the already permitted
fifth launch. Allowing for its compilation and remaining rows, the full replay
is now projected to exceed the approved5400-second campaign cap. A proposed
7200-second total cap was proposed to preserve the same64 datasets, four particle
replicates, nine particle arms, three heuristics,432 nonzero directions,
frozen fits/controls, and at most five launches. This is a budget amendment
only. The owner instead authorized four additional hours, giving19800 total
GPU-process seconds. The current authorized process continues and preserves every
completed row. No scientific tolerance or criterion changes.

Keep the running process's2600-second cap unchanged. If needed, the fifth
launch resumes its verified rows in a fresh directory with a3600-second cap,
well inside the enlarged total. Stop once the full paired replay and reporting
are complete; the extension does not authorize new training or parameter tuning.

Skeptical amendment audit: the runtime change is measured, the complete
pre-merge replay reproduces the archive exactly, and all inspected merged rows
remain finite/valid. Numerical differences are preserved as results. Continuing
the same grid answers the original question; reducing the grid would not.

## Reporting repair after complete execution

Both lanes completed all2496 holdout and432 directional rows. The merged
process finished in2543.222978871 seconds without a fifth launch. The first
CPU-only reporting attempt preserved its failure in `comparison-01/`: the
reporter expected compartment label `boundary`, whereas the frozen data use
`nearer_boundary`. This is a reporting implementation error, not an empty
experimental group or a numerical failure. The repair uses the actual saved
label, explicitly rejects a missing required regime, and adds a focused
coverage regression. Rerun only the report in fresh `comparison-02/` after
that regression passes. Skeptical repair audit: PASS; the original rows,
group assignments, numerical controls, pairing and tolerances remain fixed.
No additional GPU run is needed.

Terminal review: ten focused CPU-only reporting checks pass after the label
repair. The full analyzer completes with exit2 because plain-arm parity fails;
it reports zero invalid rows, complete pairing and unchanged input controls.
All heuristic veto assignments agree between versions. No further launch,
parameter change or numerical repair belongs to this completed replay.
