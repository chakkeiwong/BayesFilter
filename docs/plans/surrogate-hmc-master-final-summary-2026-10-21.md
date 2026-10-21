# Surrogate-HMC master program: terminal campaign summary

## Goal and completion status

The goal is to test surrogate HMC on LGSSM, KSC stochastic volatility,
predator–prey, SIR and range bearing. The LEDH analytical score supplies the
proposal force; the declared endpoint target supplies the Metropolis–Hastings
correction. Acceptance and convergence diagnose sampling of that declared
target; they do not establish its agreement with an exact model posterior.

The allocated five-model execution has ended. LGSSM, KSC SV, predator–prey and
range bearing passed their recorded cumulative sampling rule. SIR exhausted
its worker budget during warm-up and produced no retained draws. Therefore
execution is complete, but the five-model sampling objective is incomplete.
No new sampling, training or reference campaign is launched by this closeout.

The subsequent native batching implementation and bounded capacity campaign are
complete. The SIR NeuTra source/training audit is documented; the target-to-trainer
adapter, complete optimizer-update checks, target-specific map training and
downstream NeuTra HMC are still pending. No trained SIR NeuTra map is claimed.

## Master program and executed settings

The executable entry is `scripts/run_surrogate_hmc_five_model_master.py`, with
the shared implementation in `bayesfilter/testing/surrogate_hmc_five_model.py`.
The bounded continuation scripts and saved next-phase plans retain their actual
commands, selections, budgets and terminal per-model states. Their old automatic
continuation flags must not be read as authorization for a fresh campaign.

The requested particle count is N4032, with exact K2016 transport tiles, ordinary
FP32 and TF32 disabled, GPU/XLA, configured GPU memory growth, and model-specific
literature priors. Horizons are T50 for the first four models and the saved T20
range-bearing fixture. Selected OT controls include epsilon 1 and five iterations;
range bearing uses 16 flow steps. Individual run manifests remain authoritative
for historical precision exceptions, source snapshots and selected controls.
The new batching comparison uses uniform FP32; it does not retroactively change
the arithmetic of earlier SIR attempts.

Four chains use adaptive warm-up and cumulative retained sampling. The campaign
uses minimum 2,000 warm-up transitions per chain, recent-window modern R-hat
<=1.05 with a 10,000 maximum, then cumulative modern R-hat <=1.01 or a retained
maximum of 20,000 per chain. There is no additional discarded validation phase.
Finite energy errors and deterministic force-fallback counts are reported;
finite energy errors alone do not stop the run. The MH correction is retained.

## Terminal model results

| Model | Final interpretation | Evidence |
|---|---|---|
| LGSSM | Saved sampling rule passed | Preserved in the [terminal extension result](artifacts/surrogate-hmc-48h-extension-20261018-01/run/result.md) and its earlier literature-prior run |
| KSC SV | Saved sampling rule passed | Same terminal extension record; earlier literature-prior run retained |
| Predator–prey | Recorded status VERIFIED: sampling rule passed | [Terminal HMC artifact](artifacts/surrogate-hmc-48h-extension-20261018-01/run/predator_prey-retry-01/predator_prey/inference/hmc.json) |
| Range bearing | Fresh requested 70%-target rerun passed | [Rerun result](artifacts/surrogate-hmc-range-bearing-accept70-20261019-01/result.md) |
| SIR | BUDGET_EXHAUSTED during warm-up; no posterior samples | [Terminal HMC artifact](artifacts/surrogate-hmc-48h-extension-20261018-01/run/sir_d18-energy-retry-01/sir_d18/inference/hmc.json) |

The fresh range-bearing run retained 1,000 draws per chain after 2,000 warm-up
transitions. Chain acceptance was 70.2%, 68.0%, 68.4% and 68.7%; maximum modern
R-hat was 1.003620. Bulk ESS was 1,111–2,221 and tail ESS 1,688–1,910. There
were zero non-finite arithmetic divergences and one retained force fallback.
Large finite energy errors were preserved as warnings. Worker time was
25,883.80 seconds, within the separate eight-hour allocation.

SIR completed 5,500 warm-up transitions per chain with warmup_ready=false.
The last scheduled readiness check, at 5,000, had maximum modern R-hat
3.417349361. The 5,500 chunk passed its numerical health checks; readiness was
not due at that chunk. The 48-hour extension ledger charged 161,236.65 worker
seconds and recorded 11,563.35 aggregate seconds unspent. Its worker stopped
under its own bounded allocation; do not relabel the aggregate ledger as zero.

## Completed computational repair and revised interpretation

Native model/proposal/reset operations preserve the leading batch dimension
and propagate all parameter directions in one recurrence. Shared contractions
avoid pair-by-state and pair-by-parameter intermediates; native batched sharding
handles larger logical batches. The isolated source-tree suite passed 119 tests.
Old/new full-size FP64 scores agreed within 4.35e-11 relative vector norm on the
four checked rows. A logical B128 using native S4 shards completed in 333.77
seconds including compilation, with 0.761 GiB allocator peak and exact repeated
B4 output parity. These are bounded engineering results, not a full optimizer
memory measurement or a statistically supported timing ranking.

The owner corrected the numerical acceptance policy: a fixed 1% discrepancy
from the old implementation or its FP64 recurrence is not a scientific veto.
Compare old and new likelihoods/scores with the Zhao–Cui adapted model reference
and account for estimation uncertainty. At the SIR nominal point, the largest
batching change is 0.022 times the reference's reported between-fit SE. The
change is accepted as immaterial for continuing integration. References for the
other three benchmark points and variance across independent LEDH designs are
not available in that comparison. No broader equivalence claim is established.

The [materiality policy](sir-batching-materiality-policy-2026-10-21.md) explains
paired-design uncertainty and shared-reference cancellation. The
[batching result](sir-native-batching-results-2026-10-21.md) and
[reclassified summary](artifacts/sir-native-batching-20261021-01/summary-materiality.json)
preserve all measurements. The original summary remains historical evidence.

## Decisions and remaining work

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close allocated execution | Four models passed; SIR warm-up incomplete | SIR budget ended; no retained draws | SIR mixing and independent posterior agreement | Preserve terminal outcomes; continue only under an explicit bounded follow-on plan | All five models succeeded |
| Retain native batching repair | Algebra, graph, isolation and bounded capacity checks passed | Full-native B128 OOM; S4 alternative completed | Complete training-update cost | Parent Phase 1 adapter and update checks | A SIR NeuTra model has been trained |
| Retire percentage-only blocker | Nominal change accepted as immaterial | No automatic 1% veto | Broader paired reference uncertainty unmeasured | Carry uncertainty-aware comparisons into planned accuracy assessment | Reference bias or arithmetic sensitivity is absent |

| Inference status | Finding |
|---|---|
| Hard veto screen | SIR has no retained posterior; unsharded B128 exceeds the tested memory cap |
| Statistically supported ranking | None across methods or runtime configurations |
| Descriptive-only differences | Acceptance, ESS, timing, precision gaps and nominal reference-SE ratios |
| Default readiness | FP32 remains requested; complete NeuTra and exact-posterior validity unestablished |
| Next evidence needed | Existing parent adapter/update work, bounded target-specific training and downstream sampling; paired references for broader accuracy claims |

The next implementation work is already Phase 1 of the
[SIR NeuTra integration plan](surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md).
It is not a new precision-localization detour. This user-requested closeout
performs documentation, version control and synchronization rather than silently
launching a new GPU allocation. Historical result status labels and source
snapshots are preserved, including failed attempts.

## Closeout and integration audit

The user authorized committing all outstanding branch work, merging to main,
merging origin/main, pushing main and bringing main back into this branch.
The branch is `ledh-graceful-failure-execution-20260917`; the reviewed batching
implementation commit is `65a379b2c`. Existing pending code, documentation and
campaign artifacts are included in the closeout rather than discarded.

Before execution: preserve all source and results; inspect pending file types
and sizes; run focused regressions and the repository commit hook; merge in the
existing clean main worktree; resolve source conflicts by preserving both
capabilities; rerun affected checks; use a normal, non-force push. Confirm final
main, origin/main and this branch point to the same commit and both worktrees
are clean. Scientific metrics are not rerun or improved by a Git merge.
This audit passes for the requested repository integration.

The closeout check parsed all 301 new or modified Python source files without
errors. The focused CPU-only suite passed 71 tests covering neural-force HMC,
marginal weights, sequential assessment, the five-model master and literature
priors. The command, log and result are preserved in
[the closeout validation record](artifacts/surrogate-hmc-closeout-20261021-01/validation.json).
The branch and initial main-merge oracle hooks each passed all three tests.
The initial main merge passed 103 distinct focused tests, including two new
backward-guidance failure regressions; the remote integration passed 82 focused
tests. The [integration record](surrogate-hmc-git-integration-2026-10-21.md)
records the conflict resolutions, environment repair, commands and evidence.
