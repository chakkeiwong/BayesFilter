# M13 saved-state continuation — 2026-10-01

Owner deadline update, October 1: continue through midnight ending October 1 in Hong Kong (2026-10-01T16:00:00Z), retaining the 20,000 total post-warmup draws per chain cap. The active extension and updated checkpoint are under docs/plans/artifacts/surrogate-hmc-m13-midnight-20261001-01. See surrogate-hmc-midnight-extension-2026-10-01.md. The original worker's historical allocation below is preserved; the companion automatically continues its saved chains under the updated deadline.

User authorization: resume the stopped run, another 15 hours, maximum 20,000
samples. Counts follow the existing per-chain convention. The maximum covers
all post-warmup verification plus retained transitions, including the existing
2,250 per chain; the 9,000 discarded warmup transitions are excluded.
This is continuation of scientific attempt 2, not another tuning attempt.

## Question and evidence contract

Can the SAME frozen surrogate-HMC sampler on M13/data1704/T20 finish its
cumulative convergence check and then the existing posterior comparison?
Proposal: canonical analytical DCTR FP32/TF32 GPU/XLA, N4096, fixed selection,
step mixture (.125,1), saved affine metric/epsilon/L. Endpoint: the SAME frozen
FP64 full-support4096x8 trajectory-importance potential plus prior.
The endpoint is an approximate likelihood target, not the exact likelihood.

Verification promotion: modern rank/folded Rhat <=1.01 on all cumulative
post-warmup verification draws, checked every1000, maximum20000 per chain.
No warmup or candidate selection is repeated. Every250-transition chunk must
pass existing finite, endpoint, movement, divergence, energy<=1000, and force
fallback checks. Corruption, source/target/kernel drift or a health failure is
a continuation veto. Rhat above threshold is an extension trigger.
After verification passes, keep its draws excluded from posterior estimates
under the existing ordinary-tuner protocol. Continue retained sampling with
the existing six-look1000/2000/4000/6000/8000/10000 assessment, Rhat<=1.01,
bulk/tail ESS>=400 and calibrated posterior equivalence. Retained draws are
additionally capped by20000 minus the verification count. Preserve the full
six-look calibration even if only a prefix is affordable. Never invent an
extra posterior-equivalence look at the budget/sample boundary.

Comparator: frozen same-data posterior reference and health-checked RWM baseline
already used by the campaign. The saved RWM baseline is not posterior-qualified;
its results are descriptive and cannot support an efficiency ranking. Existing controls/reference remain frozen.
Acceptance and runtime explain behavior only; no speed ranking, default
promotion, exact-likelihood or all-model claim follows from this continuation.
Healthy budget/sample exhaustion is inconclusive, not rejection of surrogate HMC.

## Budget and execution

Authorization received2026-09-30T19:37:42Z; hard deadline2026-10-01T10:37:42Z
(18:37:42 Hong Kong October1). Total new allowance54000 wall seconds and at
most54000 CPU-worker/GPU seconds each, including preparation/testing.
One continuing worker, with localized harness repairs within this allowance.
No fixed ten-hour reserve, phase time split, or fresh scientific attempt.
The existing watchdog enforces actual remaining resources. Before another
chunk, estimate whether it fits; preserve results before the hard deadline.
Stop on posterior qualification, health/validity veto, maximum samples or
resource exhaustion. No automatic extension beyond the user allocation.

Versioned development root:
docs/plans/artifacts/surrogate-hmc-m13-resume-20261001-development-01
Execution root:
docs/plans/artifacts/surrogate-hmc-m13-resume-20261001-01
Parent:
docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-03/m13-public-tuning-02

Expected command (trusted GPU access):
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_resume.py execute --output docs/plans/artifacts/surrogate-hmc-m13-resume-20261001-01
The supervisor invokes its guarded worker, saves exact commands, environment,
Git/source hashes, seeds, memory-growth policy, wall/CPU accounting, next-phase
plans and terminal decision. GPU smoke and live work use TF32, GPU and XLA.
Focused unit tests are explicitly CPU-only with CUDA hidden.

## Skeptical audit and assumptions

The old stop was a time allocation stop, not failure at2000 samples. Restarting
warmup, resetting Rhat history, reusing seeds, or admitting the failed artifact
as passed would answer the wrong question. Resume must first replay archived
health, states, seeds and stopping decisions, then add only new transitions.
The numerical kernel and target binding must remain identical. A narrowly
recorded source migration may change orchestration/count limits only; compare
saved hashes and preserved source, and never silently relabel the old result.
The public tune_hmc_kernel remains the only tuning authority for this route.

Defaults: same4chains, saved metric/step/L and thresholds are a frozen baseline;
none is retuned.20k is the owner's upper bound, not a convergence guarantee.
250 health chunks and1000 diagnostic cadence are existing reviewed choices.
The saved initialization and finite endpoint approximation are unchanged
limitations. Healthy longer chains may still mix poorly; cumulative Rhat shows
this early. A pass can still disagree with the reference; posterior comparison
remains required. Monte Carlo differences remain descriptive unless the existing
calibrated assessment supports a decision.

The cheapest discriminating checks are deterministic stop/resume equivalence
including exact seeds/states, no-call tests forbidding repeat adaptation,
corruption rejection, extension beyond10000, cumulative sample cap, unchanged
kernel source, and bounded GPU archive/binding replay before a long chunk.
Audit verdict: feasible after those checks; do not launch until they pass.
No Claude or subagents are required. All previous evidence remains intact.

## Reporting

Decision table at termination: criterion status, veto status, uncertainty,
next justified action and unsupported claims. Inference status distinguishes
health, convergence, posterior agreement, descriptive metrics and ranking.
Checkpoint and next-phase plan refresh on every saved chunk and phase change.

## Repair audit progress

The first40 CPU tests passed, including deterministic resume/uninterrupted
equivalence, archive corruption rejection and public-tuner dispatch without
repeat nomination.15 numerical/helper definitions are AST-identical to the
preserved implementation, and the extracted result writer is identical.
The first GPU preflight correctly rejected the new orchestration version at
the saved-control source check; an explicit reviewed version was added while
retaining all health/native-kernel comparisons. The second exposed an overly
strict new comparator prerequisite: the pre-existing RWM is healthy but not
posterior-qualified. The resume now preserves the original health/reference
prerequisite and reports that qualification limitation explicitly. Neither
preflight generated transitions. Logs and receipts are under the development root.


## Launch review — passed

All targeted tests now pass. The original40-test batch passed; the additional
selection-compatibility tests passed; repaired boundary fixtures and documentation
checks passed17 tests. The separate real TensorFlow/XLA Gaussian test passed:
a budget stop after250 verification draws resumes through the public tuner,
reuses the frozen nomination and original archive prefix, passes verification
and reloads through the ordinary verified-artifact loader. These are mechanics
checks, not M13 convergence evidence. The GPU preflight revalidated all45 M13
chunks, unchanged binding,9000 warmup and2250 verification counts, saved stopping
decisions and memory growth without generating transitions.

The final skeptical audit passes. No target/force/integrator/Rhat change occurred;
only resumable orchestration, explicit sample limits and source-version recognition
changed. The reference and full six-look calibration remain frozen. Failed
preflights and the synthetic fixture repair are preserved with their receipts.
No independent review was requested. Remaining uncertainty is mixing and posterior
agreement under the same approximate endpoint, not saved-state recoverability.
The launch review is in development-01/launch-review.json. Proceed within the
original October1 10:37:42Z deadline, with no reset of the allowance.
