# Bounded process containment of native compilation residency

GPU03373 also passes both sequential children with complete original numerical
records, four signatures/200calls per child and verified process cleanup. Child
startup RSS is845.520/845.414MiB and fourth-release RSS is4528.031/4528.371MiB.
Their startup mappings are2602/2589, with4213/4199 at fourth release. Both start
with zero live TensorFlow allocator bytes and finish with5376current/292608peak
bytes. Supervised wall times are139.238/138.536seconds. Parent RSS rises by
3.781MiB after reading results; mappings remain2509. No256MiB parent trigger
or increasing child-startup residency appears. Earlier wording about return to
baseline means the fresh children's startup, not exactly zero parent growth.

The unit has used3/4parent workers and564.785667/2400charged seconds, including
preserved03289. CPU and GPU support the tested fresh-process containment
mechanism. Ordinary Python collection inside each child still leaves native
host memory resident; this is not native in-process eviction. Applying the
mechanism to the actual complete DZ5 consumer and all cost-trigger dispositions
remain open. Exact evidence is `run-03373/process-containment.json`, its child
reports, supervisor receipts and provenance logs under the campaign artifact root.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| GPU lifecycle supported | Both full comparisons, fresh startup and cleanup pass | No numerical, cleanup or parent-growth veto | Two sequential workers and sampled memory only | Integrate bounded worker lifetime with actual E5 consumer | No in-process eviction, unbounded-lifetime guarantee or exact peak bound |

Earlier CPU evidence and preserved harness failure follow.

CPU run03290 passes two sequential fresh workers supervised by the exact current
DZ5 `supervise` function. Each child performs four public block signatures
(D3,D5,D3,D5), 50 alternating-input calls per signature, full pinned3582b4ac
comparisons, one trace per owner, and actual Python owner/graph/callback
collection. Both children exit normally and disappear from `/proc` before the
next starts; each produces six parent heartbeats. No external runtime was edited
or external numerical campaign run.

| Observed stage | First child | Second child |
| --- | ---: | ---: |
| Initial RSS (MiB) | 566.547 | 566.523 |
| RSS after fourth Python release (MiB) | 4447.461 | 4447.527 |
| Initial mappings | 2473 | 2470 |
| Mappings after fourth Python release | 11620 | 11617 |
| Supervised wall time (seconds) | 91.178 | 91.230 |

Parent RSS rises from517.012 to520.590MiB after reading child evidence, a3.578MiB
increase; parent mappings stay2454. The predeclared256MiB parent-growth trigger
does not fire. The second child returns to the first child's initial residency
within0.024MiB and three mappings. These observations support fresh-process
containment of the tested compilation lifecycle. They do not demonstrate native
eviction in a live process, arbitrary-lifetime safety or an exact peak bound.

Run03289 is preserved as a harness failure. Its first child completed the four
candidate signatures, then tried to load the frozen BayesFilter reference from
the MacroFinance Git database because the actual supervisor sets its own cwd.
The diagnostic loader now binds Git lookup to its own checkout. The reference
revision, supervisor, numerical source and all comparisons are unchanged. The
failed child was reaped normally. That failed parent cost96.185419seconds;
the successful parent cost186.637869seconds. The unit has used2/4 parent workers
and282.823288/2400charged seconds. Sequential child time is included in parent
elapsed time and is not charged twice.

The GPU containment preflight on2026-09-23 at05:35UTC declined after its bounded
checks: GPUs0/1 were desktop-protected and GPUs2/3 were busy with foreign work.
No GPU child launched and no GPU qualification follows from the CPU result.

Evidence root: `artifacts/filter-gradient-repair-20260917/`. Read
`run-03290/process-containment.json` for exact supervisor/source hashes, commands,
parent memory, child stage observations and JUnit/result hashes;
`contained-{0,1}` contains complete numerical records and logs. Run03289 remains
alongside it. Reproduce with the stable campaign runner and
`test --group process_containment_cpu --device CPU --test-timeout-seconds 900`;
the corresponding GPU group uses the same workload and trusted selector.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| CPU lifecycle supported | Both original comparisons and process cleanup pass | No mismatch, timeout, missing record or parent-growth trigger | Only two sequential workers; observed stages omit exact peaks | Qualify GPU and apply only to a declared bounded consumer lifecycle | No in-process native release or general memory-safety certificate |
| E4 remains open | Python collection and CPU process containment established | Actual consumer/cost gates still incomplete | GPU and actual DZ5 integration remain | Finish GPU and E5 integration; explicitly dispose cold/RSS/device triggers | No whole-campaign completion or main merge |

Post-run review: baseline reset could reflect process teardown rather than a
compiler cache fix; that is exactly the containment mechanism tested. Ordinary
Python collection inside each child still leaves native host residency. A later
child starting progressively larger, failed PID cleanup, or actual consumer
requirements for unbounded signatures would invalidate extending this result.
