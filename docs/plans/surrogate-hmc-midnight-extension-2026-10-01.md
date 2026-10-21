# M13 continuation to midnight, October 1, 2026

Owner instruction: “extend the deadline to midnight tonight.” Midnight at the
end of October 1 in Asia/Hong_Kong is 2026-10-01T16:00:00Z (October 2, 00:00 HKT).
The original start remains 2026-09-30T19:37:42Z. The cumulative CPU-worker and
GPU allowance increases from 54,000 to 73,338 seconds; prior work remains charged.
The maximum remains 20,000 total post-warmup draws per chain, including the
saved verification prefix and retained draws. No warmup or tuning is repeated.

## Research intent and evidence contract

The question remains whether the same frozen M13/data1704/T20 surrogate-HMC
sampler meets the existing convergence and calibrated posterior criteria.
The target, frozen proposal field, endpoint correction, four chains, seed
sequence, GPU/XLA implementation, FP32 force and FP64 endpoint are unchanged.
The exact comparator paths, constructed reference/heuristic set, conditional
situations and interpretation limits remain those of
`surrogate-hmc-m13-resume-2026-10-01.md`. This is a resource extension, not a new
candidate, tuning attempt, scientific comparison, or independent replication.

Verification uses cumulative modern R-hat <=1.01. Retained assessment keeps
the original six planned looks, R-hat <=1.01, bulk/tail ESS >=400 and calibrated
posterior equivalence. Acceptance and runtime remain explanatory. Health,
target, checksum, state/seed continuity, source drift, and accounting failures
veto continuation; qualification, sample caps and exhausted resources stop it.
No exact-likelihood, all-model, efficiency, superiority, or default-readiness
claim follows merely from this extension.

## Skeptical audit and execution

The active guard freezes its wall deadline, CPU allowance and OS CPU limit at
launch. Editing a JSON deadline alone would not extend execution. Changing
files loaded by the active worker would also invalidate its source check.
Therefore leave that worker and its source closure intact. A separate bounded
supervisor waits for its normal exit and automatically continues a healthy
budget-stopped verification or retained sequence in a fresh output directory.
It does nothing further if the current run qualifies or reaches another stop.
The public tuner remains the verification authority; the public frozen runner
remains the retained sampling authority. A retained continuation includes every
old retained chunk in subsequent cumulative checks. Saved chunks are validated,
not regenerated. Preserve the old receipt and manifest without rewriting them.

Audit risks: falsely extending only metadata; a second simultaneous GPU worker;
resetting sample counts, seeds, tuning, or accumulated resource usage; dropping
retained draws after a phase change; extending the calibrated look schedule;
promoting a nonzero worker exit; or declaring qualification from R-hat alone.
Tests must cover these boundaries, including public-runner resumed versus
uninterrupted Gaussian chains and deadline/parent-stop decisions. Tests are
CPU-only mechanics diagnostics, not M13 inference evidence. No GPU smoke is
needed because no numerical implementation or GPU configuration changes.

Defaults: midnight is interpreted in the user's stated Hong Kong timezone;
20,000 is inherited from the explicit per-chain instruction; the 120-second
shutdown allowance and guarded chunk forecasts are existing conservative
operational choices, not convergence thresholds. Parent sources and identity
are checked before a launch. Any failed test is a repair trigger; changing the
scientific target or exceeding midnight/sample limits is forbidden.

Execution root: `docs/plans/artifacts/surrogate-hmc-m13-midnight-20261001-01`.
Development root: `docs/plans/artifacts/surrogate-hmc-m13-midnight-development-01`.
Parent root: `docs/plans/artifacts/surrogate-hmc-m13-resume-20261001-01`.
Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/run_surrogate_hmc_deadline_extension.py execute --output
docs/plans/artifacts/surrogate-hmc-m13-midnight-20261001-01`.
The extension's run manifest, progress, next-phase plan, checkpoint, worker
receipt and terminal result preserve the actual deadline and cumulative budget.
This continuation is part of the existing master; it needs no further approval.


## Local repair found by the handoff regression

The new end-to-end worker wiring test found a real reporting defect in the
active launcher: file_hash(tuning.artifact_manifest_path) passes a string to
a helper requiring a Path. The public tuner saves its complete artifact
before that call. The extension uses Path explicitly. It also recognizes
only this specific old traceback with a completed accounting receipt and a
healthy saved tuning sequence, then sends that artifact through the public
loader or public resume tuner for full validation before any new transition.
The failed old receipt is preserved as failed; no execution success or tuning
admission is fabricated. Other worker failures remain continuation vetoes.
This localized reporting repair is within the already authorized campaign.

## Final launch audit

Passed: unchanged active numerical/source closure; exact retained-chain continuation parity; public tuner/runner wiring; cumulative counts and seed continuity; frozen six-look assessment; parent wait/no-overlap; deadline and cumulative resource accounting; no promotion on failed execution. CPU tests 01: 16 passed. CPU tests 02: reporting defect found and preserved. CPU tests 03 after repair: 18 passed, 1 numerical test deselected because its implementation was unchanged after its passing run. Full logs and receipts are in the development root; all attempts remain charged. This establishes extension mechanics, not M13 convergence.
