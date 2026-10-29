# Surrogate-HMC finite-energy warning and saved-state continuation

Owner instruction, 19 October 2026 Hong Kong: energy error alone must not stop
the run; preserve MH rejection, repair the master, and resume range bearing.

Question: can the existing endpoint-corrected surrogate-HMC sampler continue
from the saved range-bearing state without treating a large finite energy
error as a continuation veto? The comparator is the exact archived kernel,
target, data, random seed stream, and saved transitions. Numerical proposal
and MH acceptance code must remain unchanged. This is a controller repair,
not a claim of posterior accuracy or improved sampling efficiency.

Evidence contract: tests must show a finite energy-threshold exceedance emits
a warning and permits continuation, including replay of the energy-only
stop; nonfinite accepted states, invalid endpoints, corrupted archives, and
unrelated health failures still block. Independent MH rejection must remain
intact. R-hat stopping, warmup exclusion, sample caps, priors, N=4032, FP32,
16 range-bearing flow steps and all other frozen controls remain unchanged.
Large energy errors are explanatory diagnostics, not continuation vetoes.

Execution: reuse the existing 48-hour ledger, maximum six attempts and hard
deadline 2026-10-20 08:00:41 UTC (16:00 Hong Kong). Preserve the active
predator-prey worker. Insert saved-state range bearing before SIR, with no
overlapping GPU workers. Use a fresh output directory, never overwrite the
previous failed attempt. Audit actual queue and source-binding compatibility
before launch; repair only known controller/schema drift, never mask kernel
or target drift. GPU commands require escalation and memory growth.

Default audit: energy threshold 1000 is retained as an observability threshold
only, following the owner decision. No retuning, clipping, new target, or new
scientific setting is introduced. Acceptance alone does not prove mixing;
cumulative modern R-hat and existing sample budgets remain the sampling rule.
The practical risk is inefficient repeated rejection; report movement,
acceptance and energy diagnostics without selecting on isolated outliers.

Pre-execution skeptical audit: PASS, 19 October 2026 Hong Kong. The public
master calls the public tuner, which dispatches to saved-state continuation.
Both fresh and resumed sampling now use warning-only finite-energy diagnostics.
The underlying neural-force transition still computes the complete Hamiltonian
difference and applies log(U) < min(0, -delta_H). Positive large finite errors
are rejected by MH; large negative errors are not automatically rejected.
Arithmetic/nonfinite failures and the other health checks remain separate.
The original numerical-kernel hashes match the saved range-bearing run.

Validation: 78 controller/resume/queue tests and five targeted MH tests pass.
The actual range-bearing adapter, binding, scope, nomination, six saved chunks,
checksums, state/seed sequence and aggregate tensors pass replay on the original
GPU, with memory growth enabled. It preserves 1500 warmup transitions per chain
and zero retained draws; the same archive now reaches the next sampling chunk
instead of the energy veto. No new HMC transitions ran during preflight.
Frozen configuration equality is checked after JSON normalization. The next SIR
archive also passes source-migration validation.

Preflight exposed and repaired an archive-reader mismatch: an absent phase is
written as {}, whereas the reader recognized only null. Both forms now work;
an actually missing populated phase still fails. A CPU replay encountered
last-bit GPU/CPU differences in the initial affine solve; no equality tolerance
was weakened. The original-GPU replay passes exact state equality.

Queue audit: preserve the already-running predator–prey process, then resume
range bearing before SIR with an eight-hour phase cap and the original shared
172800-second/six-attempt budget. Deadline remains 2026-10-20 08:00:41 UTC.
No new nomination or tuning flags are passed on resume. Its output is
run/range_bearing-energy-resume-01 inside the existing extension directory.
The preserved predator–prey process loaded the pre-repair controller before
these edits; its runtime policy must not be relabeled as the new controller.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Resume existing range-bearing kernel | Exact archive replay and regression tests pass | No remaining archive or source mismatch; original non-energy health checks pass | Mixing and cost after continuation | Activate repaired queue within existing deadline | Posterior accuracy or convergence |

Artifacts: docs/plans/artifacts/surrogate-hmc-energy-warning-repair-20261019-01.
Archive source, patch, CPU-only focused test logs, actual launch command and
handoff receipt, then record resumed progress and residual uncertainty here.

## Restart audit after the predecessor finished

The original supervisor finished at 2026-10-18 17:52:31 UTC while this repair
was being tested. Predator-prey completed with VERIFIED status. SIR stopped
on the empty-phase aggregate-reader mismatch described above, before sampling.
An initial handoff launch correctly rejected the now-stale assumption that
the predecessor was still running; it launched no worker.

The repaired master now supports an idle-queue restart. It requires the old
supervisor to be absent, no running budget attempt, the expected terminal model
statuses, unchanged original source pins, and at least two attempts and usable
time remaining. Four of six attempts have been used. The next attempts are
range bearing from its saved 1500 warmup transitions per chain (eight-hour cap),
then SIR in a fresh output directory with the remaining original budget.
Predator-prey's completed result is preserved. No deadline extension is added.

Skeptical restart audit: PASS. The extra stopped-queue test verifies both the
successful idle case and rejection of an active ledger. All 23 queue tests
pass, alongside 56 controller/resume tests and five MH tests (84 total).
No proposal, acceptance, target, prior, particle design or random stream changes
are introduced. Actual replay remains the evidence for saved-state compatibility.

Exact launch from this checkout, with escalated GPU permissions:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_queue_repair.py launch --resume-range-bearing
```

The restart receipt is run/queue-energy-repair-01/handoff.json; new outputs are
run/range_bearing-energy-resume-01 and run/sir_d18-energy-retry-01 under the
existing extension directory. Confirm receipt takeover, the current worker PID,
and replayed progress before reporting that sampling has resumed.
