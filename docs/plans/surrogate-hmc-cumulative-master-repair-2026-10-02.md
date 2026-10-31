# Repair the master's cumulative posterior accounting

Owner instruction: repair the master, reclassify the saved results, and show
their correct interpretation. No additional sampling budget is inferred.

## Research intent and evidence contract

The question remains whether the frozen M13 surrogate-HMC sampler provides
usable inference for its specified importance-likelihood posterior. Every
healthy, fixed-kernel post-warmup draw is eligible, including the public tuner's
draws historically called verification. Warmup remains excluded. Passing the
sampling rule must not trigger another sample set or a reset of the count.

For this owner-requested sampling procedure, numerical health plus maximum
rank-normalized split/folded R-hat <=1.01 completes sampling. The cumulative
maximum is 20,000 per chain, checked in 250-transition health chunks and at
1,000-transition cumulative diagnostic intervals, plus saved-state and terminal
assessments. ESS, MCSE, acceptance and runtime describe precision/behavior;
ESS below the old 400 screen produces a warning, not another sampling phase.
Checksum, state/seed continuity, target/kernel identity and health failures
veto continuation. A deadline or sample cap ends an inconclusive sequence.

Sampling qualification is explicitly separate from reference-equivalence or
all-model campaign qualification. The old six-look equivalence calibration is
not valid at arbitrary newly pooled counts and will not be relabeled passed.
No exact-likelihood, method-superiority, or default-readiness claim is made.
The existing frozen endpoint and analytical proposal field remain unchanged.
No new method comparison is performed, so no heuristic ranking is inferred.

## Skeptical audit and assumptions

The confirmed defect is that the current master consumes the public tuner's
passing artifact but discards its samples from subsequent diagnostics. Its
retained controller also imposes an additional per-phase sample cap. Repair
both: a single total count, shared history, and no new simulation after a passing
saved cumulative diagnostic. When later saved samples already exist, assess
all of them even if a shorter prefix once passed. Do not cherry-pick the prefix.

The public tune_hmc_kernel remains the artifact authority. The shared native
runner still generates any future transitions. Archived binding/checksum and
state/seed checks establish pooling eligibility; an executable reconstruction
checks the claim. Runtime and reclassification must call the same loader,
diagnostics and classifier, avoiding a reporting-only fix.

The 1.01 threshold, 20,000 maximum and removal of an independent phase come from
the owner's instructions. Existing 250/1,000 chunk/check cadences are retained
as engineering choices; immediate and terminal assessments prevent those
cadences from hiding already available evidence. Rank diagnostics retain the
repository's bounded graph-mode non-XLA exception. New HMC remains GPU/XLA;
tests and archive analysis intentionally hide GPUs before importing TensorFlow.

## Implementation, validation and execution

Repair the active deadline master and expose a reclassify CLI mode that reads
a stopped run and writes a new result directory. Preserve original reports,
manifests and source snapshots. Reuse the public tuner and frozen-runner APIs;
do not alter numerical kernels, retune, change data or repeat warmup.

Tests must prove: a passing prefix causes zero new sampling; existing suffixes
are included before deciding; no warmup inclusion or double counting; the
20,000 cap applies to the total; seed/state identity violations fail; low ESS
is reported without a separate phase; bad execution cannot qualify; and the
actual master worker calls the repaired consumer. Compare the reclassification
against the independently reconstructed 9,000 and 12,750 diagnostics.

Bound local validation to ten minutes per CPU test attempt and five minutes
per archive analysis attempt. These are routine repair/inspection operations,
not an extension of the expired GPU campaign. Preserve logs and failure/repair
notes under docs/plans/artifacts/surrogate-hmc-cumulative-master-repair-20261002-01.
Write the fresh interpreted result under
docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-01.

The audit passes: the repair changes accounting and stopping orchestration,
not the sampled target or frozen numerical kernel. Existing exact archive
continuity and reproduced prefix R-hat provide the smallest known-good check.
Refresh the active checkpoint and next-phase plan after reclassification.

## Additional call-chain findings in final review

The legacy recovery master still reached the old suffix-only controller through
retained_stage, and a legacy qualified flag could bypass cumulative reassessment
in the deadline master. Repair these orchestration paths too. Preserve the old
consumer only as an explicitly historical comparison fixture. Sampling success
is reported separately from reference-equivalence success in every repaired
master. Add executable wiring and legacy-status regressions before finalizing.
These are the same accounting repair, with no change to target or sampling
budget. Source migration permits only the four reviewed orchestration modules;
original saved snapshots remain mandatory and numerical dependencies may not
drift. Subsequent result versions preserve earlier reclassification evidence.

The older standalone retained wrapper also reaches the repaired controller.
Its obsolete terminal-status test was repaired so an inconclusive scientific
result is not mislabeled a worker crash, and its source manifest now includes
the cumulative controller. A mocked CPU wrapper regression checks both pass
and inconclusive result propagation without touching GPU devices.

## Completed execution, 2026-10-02

Repair, source review and archive reclassification are complete. The final 65
focused regressions passed. The repaired master reproduces the independent
prefix/full diagnostics exactly and emits identical pooled sample bytes.
Final result: artifacts/surrogate-hmc-m13-reclassified-20261002-02/result.md.
Execution and review: surrogate-hmc-cumulative-master-repair-result-2026-10-02.md.
No new HMC was generated. The old plans and original source/run artifacts remain
historical evidence; this repair plan governs cumulative sample accounting.
The active checkpoint and final next-phase plan record the expired sampling
allocation, 12,750 of 20,000 draws used per chain, and no independent validation
phase. A future authorized continuation remains part of the same master.
