# State-space XLA repair and continuation

Superseded at September 29, 02:10 Shanghai: this service passed eight fresh
mechanics cells, then waited for an empty GPU before public preflight. The owner
requested contention recovery rather than that exclusivity requirement. The
waiting service was stopped and its charge settled. The active continuation is
`../hmc-shared-gpu-recovery-2026-09-29/result.md`; no public preflight fit from
this attempt was interrupted or rerun. The launch-time status below is history.

At September 29, 00:25 Shanghai, the repaired continuation is launched and
waiting for GPU capacity. Trusted telemetry identifies foreign compute
processes on all three eligible GPUs. Service:
`bayesfilter-hmc-ssm-xla-repair-20260929-r1.service`. Live state is in
`runtime-r2/status.json` and `runtime-r2/queue-ssm-progress.json`.

All ten C1 recovery fits completed with the original numerical source, data
and seeds. The post-hoc beta-binomial inventory is 256/256 complete. Original
coverage failures remain; this does not replace the original confirmation.
The nine fresh retries took 2,468.172 seconds. Their checksums, fit identities,
GPU/growth manifests and completion receipts were rechecked before reuse.

The earlier eight GPU mechanics cells passed, but all four public bridge
cells stopped before HMC compilation because the target advertised
`full_chain_xla_diagnostic_ready=False`. No pricing or main work ran. This
was an adapter qualification failure, not observed numerical instability.

The repaired adapter advertises full-chain diagnostic capability only for
its XLA configuration. Public rejection guards remain active. Sixty-nine
tests passed in 73.80 seconds: all eight full-chain compilations, deterministic
leapfrog parity, capability scope, public prepared measurements, completed-fit
reuse, corrupted-evidence rejection, cumulative accounting and sequencing.
CPU tests intentionally hid GPUs. The first diagnostic had an invalid
graph/XLA RNG comparator; its failed receipt is preserved. Identical explicit
momenta give the valid numerical comparison without changing tolerance.

The new source identity is
`a5b3da867f6b7969b0783e4cec7904526990910904fdc01910cfb24f699d92c2`.
Only the campaign capability metadata and snapshot dependency list changed
inside the package. The external queue now skips checked complete recovery
work and includes stage receipts stored outside its runtime directory in
the cumulative SSM charge. Copied model data, references, seeds and suites
are byte-identical to the preceding attempt.

The queue will repeat all GPU preflight cells, then run complete-fit pricing
and affordable main cases. A failed preflight stops that sequence for diagnosis.
No C1 sampling is scheduled. `launch-command.json`, `configuration.json`,
`engineering-verification.json` and `prepared-r3/manifest.json` preserve exact
commands, source, environment, inputs and tests. The plan is
`docs/plans/bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Resume repaired GPU preflight | 69 engineering tests passed; original data preserved | All GPUs currently busy | Actual GPU full-chain compatibility | Bounded queue then fresh preflight | Main readiness or convergence |
| Reuse C1 recovery | All ten clean completions independently checked | Original coverage failures remain | Calibration remains unresolved | Do not rerun recovered fits | Original confirmation repaired |
| Preserve campaign limits | 57.878 GPU hours before launch; prior SSM charge 1,100.810 seconds | Original deadlines and ceilings remain binding | Available uninterrupted device time | Price complete fits and retain all 32 dispositions | Every planned fit can finish |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Prior interface failure preserved; fresh GPU preflight pending |
| Statistically supported ranking | None |
| Descriptive-only differences | Compilation/runtime and telemetry |
| Default readiness | Not established |
| Next evidence | GPU public lifecycle, uncensored prices, per-member posterior checks |

The GPU ledger remains `hmc-ssm-funded-2026-09-28/grant-ledger.json`.
The 13,919.996-second cumulative queue allowance subtracts its prior 521.072
seconds; waiting consumes this allowance and is charged once. The repair's
conservative CPU charge is 1,800 seconds from the previous 43,038.430-second
balance, leaving 41,238.430. Nested test time is not charged again. The 36-hour
SSM GPU ceiling and 22-hour main cap are unchanged. New work stops September
29 at 23:03:52 Shanghai, computation September 30 at 03:03:52, and reporting
is due at 05:03:52. These are the original deadlines, not a new 48-hour clock.

The weakest evidence is still the missing GPU public-pipeline preflight.
A successful CPU XLA test cannot remove that gap. Later sampler failures may
reflect geometry or the K7 filtering approximation; they must be classified
separately from this interface repair and from machine contention.
