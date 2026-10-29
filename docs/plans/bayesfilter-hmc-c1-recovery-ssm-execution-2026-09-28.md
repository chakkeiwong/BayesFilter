# C1 recovery and state-space execution checkpoint

The first sequence stopped with its charges settled. C1 beta-binomial fit 78
completed normally in 237.381 seconds with an independently saved assessment.
Its source/data/seed identity matches the original fit exactly. The child
manifest verifies GPU/XLA placement, TF32 and memory growth configured before
device initialization. Fit 79 then timed out at 1,500.301 seconds despite its
full 750-second extension: 48 of 50 telemetry samples recorded foreign GPU
processes and durable numerical progress remained recent. Eight later fits
were deferred after the admission wait. Recovery cost 2,337.993 seconds.
The following SSM admission spent 480.004 seconds waiting and launched no
numerical worker. All charges are in the original ledger exactly once.

The owner subsequently added 50 GPU hours. The active funded continuation is
recorded in the [plan amendment](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
and `artifacts/hmc-ssm-funded-2026-09-28/`. It preserves this completed fit and
queues before the remaining work. The earlier running-status paragraphs below
describe the first launch, not current execution. No original coverage finding
has been repaired by the resource retry.

The [reviewed plan](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
and [exact launch command](artifacts/hmc-c1-recovery-ssm-2026-09-28/launch-command.json)
record the experiment and resource bounds. Live outputs are under
`artifacts/hmc-c1-recovery-ssm-2026-09-28/sequence-r1/`; the enclosing log is
`sequence-r1.log` in its parent directory. The refreshed state-space package
is `prepared-ssm-r1/`, with all 32 planned slots preserved.

The repaired optional timeout policy accumulates trusted observed contention
intervals and grants bounded allowance while recent durable progress exists.
Admission waits prevent an immediate launch against a busy selected GPU.
The parent retains the original C1 numerical package for the ten children;
the state-space snapshot changes only the four scheduling modules. No epsilon,
L, verification, warmup, retained-sampling or posterior criterion was relaxed.

The implementation passed 103 controller/executor/planner tests and a further
29 tests, including both actual isolated Gaussian/beta-binomial public pipelines
and recovery launch/accounting checks. Planner tests overlap. A CPU-only
summary replay on the original source also preserved all 256 beta-binomial
slots and all ten original timeouts. The engineering verification receipt
records timing and source hashes. The conservative CPU repair charge is 1,800
seconds from the existing CPU balance, encompassing these tests and reporting.

Trusted telemetry found C1's original GPU occupied by a foreign process. The
sequence uses idle GPU 0 of the same reported model and memory capacity; its
UUID is recorded in every launch manifest. A hardware scheduling change cannot
establish that the revised timeout policy improves runtime. Regression tests
establish its prescribed behavior; the recovery asks whether the original fits
can finish under available resources.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Run ten original C1 seeds once | Ten complete fits with unchanged identities; pending | Source mismatch, invalid GPU provenance or non-budget worker failure stops recovery | Runtime and posterior outcome of each seed | Finish within 4,800 seconds plus 30 seconds for shutdown/accounting | Original confirmation repaired or recalibrated |
| Start SSM after recovery settlement | Checked GPU mechanics and complete-fit prices; pending | Invalid filter values/scores, source or lifecycle evidence | Pricing and affordable case count | Automatic preflight, pricing and funded main slots | All 32 fits affordable or converged |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Original C1 timeouts and coverage failures remain recorded; new terminal checks pending |
| Statistically supported ranking | None |
| Descriptive-only differences | Retry runtime and later per-member diagnostics |
| Default readiness | Not established by these engineering tests or reruns |
| Next evidence | Recovery receipts, GPU preflight, complete-fit prices, per-slot SSM outcomes |

The initial settled grant has 34,266.930 GPU seconds; the separate 1,200-second
NeuTra reserve leaves 33,066.930 seconds for this entire sequence. The recovery
reserves at most 4,830 seconds and is charged once after it stops. SSM then
prices against the actual balance while retaining its four-hour reserve.
The 48-hour campaign name is a wall-time limit, not another GPU grant.

The strongest alternative explanation for successful retries is simply removal
of competing work, rather than better timeout extensions. Failed posterior
checks can remain after successful execution. The original and post-hoc results
must therefore be reported separately, with missing fits and unfunded SSM slots
preserved in their denominators.
