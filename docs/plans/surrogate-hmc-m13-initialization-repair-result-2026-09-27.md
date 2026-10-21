# M13 initialization repair result — 2026-09-27

The initialization repair failed public verification because the four chains
did not mix sufficiently. All fresh warmup and verification chains moved,
with finite states and valid endpoints, no divergences, and no force fallback.
The maximum of bulk and folded rank-normalized R-hat was 1.137819 and 1.046165,
above the predeclared 1.01 threshold. This candidate is rejected for retained
handoff. The result does not reject surrogate HMC as a research direction.

The claimed target remains the posterior defined by the frozen full-support
trajectory-importance MH endpoint. The canonical analytical DCTR field is
used only for proposal dynamics. This run computed public tuning and fresh
verification using that route; it did not compute a qualified retained
posterior. Posterior/reference agreement for this candidate remains unchecked.

The original zero-movement and fallback failure was absent in fresh
verification. Nomination selected epsilon=0.07105464358253688 and L=2.
Two fallback events occurred during adaptation, zero during nomination
verification, and zero in fresh warmup or verification. These observations
do not establish global force reliability. The independent same-endpoint RWM
comparator remains healthy; no superiority or speed claim is supported.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject current tuning handoff | Fresh R-hat fails 1.01 | Other recorded fresh checks pass | Longer-run mixing and posterior agreement | Assess a bounded mixing remedy after execution repair | Surrogate HMC is generally invalid |
| Hold further research launches | Reliable budget accounting unavailable | Missing parent completion/accounting; attempt exceeded planned wall limit | Why the parent and waiting supervisor disappeared; actual CPU charge | Repair parent-loss handling and reconcile charges | Five-hour cap was enforced |

The worker report was written at about08:39 HKT after a02:26 launch.
Its internal tuning timer reports21927.65seconds; initial parent-manifest to
final worker-report file times span about22407.85seconds (6h13m28s).
These are elapsed-time observations, not recovered CPU resource accounting.
The parent report and accounting.json were never written; the worker
run-manifest still says RUNNING. At14:31 HKT no campaign processes remained
and the tool sessions were unavailable. All188 parent-snapshot source hashes
still matched. The retained output directory does not exist.

Code inspection finds the wall/CPU watchdog in run_job lives in the parent,
while the worker starts in a new session. That arrangement leaves an
unhandled parent-loss case. The observed evidence establishes failed
watchdog/accounting protection; it does not establish what killed the parent.
Do not invent a successful parent completion record, claim exact resource
charges, or silently treat the missing receipt as zero expenditure.

The single authorized repair attempt has been used. Its planned wall cap was
exceeded. The previous checkpoint's remaining-budget figures exclude this
attempt and must not be used as current available compute. No second research
attempt or retained sampling is authorized by the failed verification.

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Fresh R-hat fails; execution accounting is incomplete |
| Statistically supported ranking | None |
| Descriptive-only differences | Fresh zero fallback and all-chain movement versus the earlier trap |
| Default-readiness | Not established by this exploratory candidate |
| Next evidence needed | Reliable execution/accounting, bounded mixing repair, then independent retained posterior assessment |

The strongest alternative explanation is insufficient trajectory exploration
or chain length rather than an incorrect invariant target. Initialization
alone has not solved mixing. A fresh, qualified verification followed by
passing retained posterior assessment could overturn this rejection.
The weakest evidence is the lost execution accounting; no estimate is being
substituted for measured CPU expenditure.

Execution provenance: original command, commit, environment, seeds, GPU
designation and source snapshots are in the initial run-manifest.json under
docs/plans/artifacts/surrogate-hmc-m13-initialization-repair-20260927-01.
The worker report records the verified memory-growth policy, target, force,
FP32/TF32 settings and public tuning manifest. This note supplements those
files without rewriting original evidence. The source checks and selected
diagnostics are saved in status-reconciliation-20260927.json in that root.

Existing-master next action: resolve the launcher/accounting failure, then
review the remaining compute and a specific mixing remedy. The master is
incomplete. LG1 remains closed; confirmation data3701 remains untouched.

## Warm-up protocol clarification

The saved public-tuning manifest records 500 discarded warm-up transitions per
chain, followed by 1,000 fresh verification draws per chain across four chains.
The reported R-hat uses those 4,000 verification draws only; splitting produces
eight half-chains of 500 draws. Earlier mass/step-size adaptation trajectories
are separate tuning work, not additional warm-up for these freshly started
verification chains. No retained posterior sampling followed the failed check.

The call chain in `bayesfilter/inference/hmc_tensorflow_verified.py` runs both
fixed-length phases before `_fresh_checks`. It does not calculate warm-up
R-hat, test warm-up readiness before verification, or extend warm-up. Its
`warmup_*` checks establish numerical health and movement only. Therefore the
run did use the public tuning interface, but did not use the sequential
warm-up readiness protocol. Calling it fully compliant with that protocol
would be wrong. No checked evidence establishes that 500 warm-up transitions
were sufficient for M13.

For comparison, the shared NeuTra controller in
`bayesfilter/inference/neutra_hmc.py` defaults to a minimum of 2,000 warm-up
transitions per chain, checks modern R-hat on the latest 1,000 against 1.05,
extends warm-up in chunks up to 10,000, and requires warm-up to pass before
entering retained sampling. That controller is not called by this typed
surrogate-force verification route. Its scope and binding requirements must
be respected when repairing this gap; an arbitrary proposal field must not be
relabeled as an exact target gradient to fit another runner.

The failed verification supports rejection of this handoff, but does not
separate insufficient warm-up from slow stationary exploration. Warm-up
readiness and bounded continuation belong to the existing master's mixing
repair, after execution/accounting and budget reconciliation. This audit
changes no sampler code, thresholds, saved run evidence, or launch authority.

## Sampling continuation audit

The shared sequential controller accumulates post-warm-up draws and checks
diagnostics cumulatively, rather than testing only the newest chunk. Its
defaults are 1,000-draw increments, minimum 1,000, maximum 10,000 per chain,
and modern R-hat <=1.01; a supplied full-convergence diagnostic can add the
required posterior criteria. Health and campaign resource checks can stop
continuation before the draw maximum.

The M13 retained consumer already has a separate bounded staged implementation:
250-draw health chunks, cumulative assessments at 1,000 and 2,000 per chain,
and a 2,000-draw maximum. It requires R-hat <=1.01, bulk/tail ESS >=400 and
the predeclared MCSE-aware reference agreement. Mixing failure remains
INCONCLUSIVE and permits the next scheduled look. Its interval calibration
explicitly covers those two looks; adopting more looks or a 10,000-draw cap
requires a matching uncertainty procedure and budget, not just a loop edit.
See surrogate_hmc_m13_retained.py::consume_chunks and
surrogate_hmc_m13_posterior_assessment.py::assess.

This retained stage was never entered. The earlier typed public tuning verifier
ran a fixed 1,000-draw fresh verification and issued no handoff when it failed.
Thus the missing continuation is in the public verification stage as well as
warm-up readiness. The 1,000 count came from the verifier's documented minimum,
but a minimum alone cannot establish sufficient chain length. Existing-master
repair must treat that count as an initial diagnostic checkpoint, continue
the same frozen kernel and chain states with unused seeds under predeclared
health/force and compute limits, and evaluate cumulative draws. A failed
early R-hat check remains a failed checkpoint; it does not reject the method.
No code was changed or additional scientific run launched in this audit.
