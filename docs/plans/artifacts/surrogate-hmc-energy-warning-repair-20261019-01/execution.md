# Executed finite-energy warning repair

Recorded 2026-10-18T18:36:35.502350+00:00. Range bearing is RUNNING from the saved state: 1500
warmup transitions per chain, 0 retained draws at this check. The
worker log confirms saved-state replay, and the original GPU was initialized.
This is a live restart, not a completed sampling result.

Large finite energy errors are warnings in fresh and resumed chunks and final
admission checks. Proposal dynamics and the MH acceptance test are unchanged.
The finite energy threshold remains 1000 only for reporting. Genuine arithmetic
and nonfinite failures, invalid endpoints, corruption and lack of chain movement
remain separate checks. The historical energy veto is available only when
validating an old archive and cannot control new sampling.

The saved archive passes exact original-GPU replay: six chunks, 1500 warmup
draws per chain, unchanged states, seeds, frozen configuration and numerical
kernel hashes. The old energy-only stop is preserved in its original directory;
its copied history now emits an energy warning and permits continuation.
An empty-phase reader mismatch was also repaired; both null and empty-object
representations work, while missing populated aggregates still fail.

The focused evidence comprises 84 passing tests. Logs and before/after source
copies accompany repair-manifest.json. CPU archive replay exposed last-bit
device differences; no equality tolerance was relaxed, and GPU replay passed.

The first handoff command found that the predecessor had already finished and
correctly refused to attach. The idle-restart branch was then tested and launched
successfully. Predator-prey had completed VERIFIED. SIR's immediate empty-phase
reader failure is repaired, and its retry follows range bearing automatically.

The queue preserves the original 172800 worker-second, six-attempt budget and
deadline 2026-10-20T08:00:41.921494+00:00. The range-bearing restart has an eight-hour
cap; SIR receives the remaining budget in a fresh directory. Five attempts have
now been started, including this active attempt. No new budget was granted.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Resume saved range-bearing kernel | Regression tests and real archive replay pass | No non-energy replay veto | Future mixing and cost | Continue existing master automatically | Convergence or posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | Archived non-energy health checks pass; new chunks remain subject to those checks |
| Statistically supported ranking | None; this is a controller repair |
| Descriptive-only differences | Historical energy veto is now a warning; no new performance comparison |
| Default-readiness | Not assessed; owner-directed stopping policy implemented |
| Next evidence needed | Actual cumulative retained-chain diagnostics under the frozen kernel |

Post-run red-team limitation: replay and tests establish continuation mechanics,
not eventual mixing. Many rejections could still leave insufficient effective
samples; cumulative diagnostics and limits remain active. The finite endpoint
approximation is unchanged, and exact-posterior agreement is not established.

Actual launch command:
```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_queue_repair.py launch --resume-range-bearing
```
