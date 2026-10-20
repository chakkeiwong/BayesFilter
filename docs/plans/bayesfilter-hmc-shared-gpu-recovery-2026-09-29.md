# Shared-GPU admission and bounded HMC recovery

The owner correctly rejects exclusive GPU availability as a requirement.
One GPU is sufficient, but other compute processes may share it. The current
supervisor already observes contention and extends a progressing fit. Native
tuning and posterior chunks preserve completed work. The campaign nevertheless
requires an empty device at three admission layers, and a fit that exhausts
its nominal allowance has no automatic checkpoint retry. Those are engineering
gaps, not numerical evidence against a model.

Question: can the existing exact numerical procedure start on a shared GPU,
continue while making progress, and resume an incomplete contention-interrupted
fit without resetting its seed, checkpoint, attempts or cumulative allowance?
The baseline is the current frozen SSM campaign. Scientific targets, observations,
candidate search, verification, posterior criteria and all 32 main slots remain
unchanged. Completed C1 recovery work is reused and never rerun.

The repair adds an explicit shared-device admission policy. Trusted visibility
and valid device telemetry are required; another PID is not a launch veto.
At initial selection, prefer idle capacity, otherwise the lowest observed GPU
utilization, using free memory as a tie breaker. This is a scheduling heuristic,
not an estimate of throughput or a promise that a workload fits. The selected
UUID remains fixed through preflight, pricing and main. Memory growth remains
mandatory. Unknown telemetry keeps bounded waiting; OOM, numerical failures
and corrupt evidence require diagnosis rather than blind retries.

Keep observed-contention extension while a child is progressing. Add at most
one automatic retry after a cooperative budget stop or supervised timeout with
trusted observed contention, no final assessment, and unused fit allowance.
The retry resumes committed tuning/posterior checkpoints; unfinished preparation
uses the existing archive-and-restart rule. It does not repeat completed fits
or change random streams. A retry may spend the unused portion of the originally
declared base-plus-contention cap, including cold-start overhead; no new base or
extension is created. Preserve each attempt receipt and charge all elapsed time.
Attempt-count and cumulative-budget checks apply across coordinator restarts.

The stage and fit admission layers must honor the same explicit policy. Shared
mode is opt-in for this campaign; historical policies remain readable. The
campaign's original data and numerical settings are frozen into a new source
version because execution source identity is strict. The old queue can be
stopped at its waiting stage, settling its reservation; preserve its completed
mechanics evidence as prior-source evidence and repeat fresh preflight.

Budget choices are scheduling hypotheses, not calibrated runtime quantiles.
The bridge's existing 170-second nominal fit gets at most 170 seconds of
contention/recovery allowance (one extra nominal attempt, total 350 per cell
including its existing 10-second outer margin). Each complete-fit pilot keeps
1,175 nominal seconds and moves the inherited 600-second admission allowance
to contention/recovery (total 1,785 per cell including margin). These add
5,480 seconds to the original preflight/pricing envelopes, financed by the
existing six-hour repair allocation. Main fits retain measured 1.5x complete-fit
cost limits, with the existing split between nominal and contention allowance.
The 36-GPU-hour total, 22-hour main ceiling, current ledger and original
42/46/48-hour deadlines are unchanged. Main pricing must also fit the *remaining*
wall time and cumulative SSM cap. An incomplete or retried outer pilot is not
an uncensored price. An inner checkpoint retry is included in its complete
enclosing fit cost and must be recorded.
If one pricing fit still exhausts its resource budget, leave that lane unpriced
and continue the independently completed, affordable lanes. Only explicit
outer timeouts or child budget/timeout receipts authorize this continuation;
unexplained worker failures stop for diagnosis. This closes a call-chain gap:
the price calculator already rejects incomplete prices, but the queue formerly
stopped before invoking it whenever any pilot returned a nonzero exit.

CPU implementation, focused regressions and notes have a conservative
1,800-worker-second allocation from the existing balance. Artifacts go under
`docs/plans/artifacts/hmc-shared-gpu-recovery-2026-09-29/`. GPU execution uses
the tfgpu environment, trusted permissions, TF/TFP XLA and verified memory
growth. The original active ledger remains
`docs/plans/artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`.

| Diagnostic | Role |
| --- | --- |
| Shared-device admission, successful checkpoint recovery, exact cumulative charges and unchanged seeds | Engineering pass criteria |
| Source/data mismatch, corrupt checkpoint, numerical error, unexplained worker crash | Continuation veto for the affected work |
| Trusted contention with resource stop and no final assessment | Bounded recovery trigger |
| Posterior readiness/precision or candidate rejection | Promotion veto only; never retry to select a favorable outcome |
| Hard fit/cell/stage/campaign budget or original deadline | Stop; preserve incomplete dispositions |
| Utilization, process counts, runtime, ESS and acceptance | Explanatory unless separately assigned a scientific role |

Skeptical audit: merely disabling admission would leave nominal timeouts and
restart loss unresolved. Blind retry would reset budgets and could select on
posterior outcomes. This plan instead tests busy admission, cleared contention,
unknown telemetry, no-progress compilation, recovery only for incomplete
resource stops, same-source checkpoint reuse, bounded attempts and outer limits.
Use deterministic fake-clock scheduling tests, real subprocess recovery and
existing Gaussian/beta-binomial and SSM pipeline regression coverage. All CPU
tests hide GPUs. Actual GPU preflight and full-fit prices still precede main
admission; short smokes cannot establish posterior correctness. The plan passes
with these bounds. A failure under persistent contention may exhaust the
budget, but cannot silently become a numerical rejection or reset the clock.

## Implementation audit before shared-device launch

The focused policy/recovery set passed 97 tests in 127.53 seconds. Real K0 and
K7 subprocesses were deliberately stopped after a committed numerical chunk,
then automatically resumed through the normal public pipeline. Previously
written numerical chunks/evidence kept identical checksums, final candidate
inventories were preserved, retained sampling excluded warmup, and completed
fits were not rerun. Contention was injected in those CPU tests; they do not
establish actual GPU throughput. An initial test stopped on a nonempty map of
empty chunk lists; that test-design failure is preserved, and the trigger now
requires an actual committed chunk.

A further compatibility set passed 110 tests in 149.43 seconds, including
actual Gaussian/beta-binomial isolated pipelines, supervisor cleanup, reuse,
remaining fit limits, original deadlines and planner integrity. These test sets
overlap; their counts are not additive independent evidence. The official tuning
chapter builds, and the rendered shared-device/recovery paragraph was inspected.
The final wiring checks separately cover bounded pilot failures and price caps.

The old service was stopped during its preflight-pipeline capacity wait. It had
completed another eight mechanics cells in 61.788 seconds; it had launched no
public preflight child. Its waiting receipt is settled. All previous charges
are carried forward through their original roots. A fresh source is required
because the optional execution policy changed. Only three package modules
change from the preceding frozen source: timeout policy, isolated fit recovery,
and campaign planning/provenance. No target, filter, score, integrator or sampler
module changes. The external queue honors shared admission and continues
independent priced lanes after explicitly diagnosed resource exhaustion.

The implementation audit confirms that sharing reaches every admission layer,
the same GPU UUID is retained, recovery is limited to one additional attempt,
and all attempts consume the original hard fit and enclosing caps. A long
compilation can receive a bounded retry even before its first completed chunk
if trusted contention was observed; that does not establish numerical progress.
If the whole cap is spent, recovery cannot restart it. Exact launched commands,
source/data hashes, receipts and live status are preserved in the artifact root.
