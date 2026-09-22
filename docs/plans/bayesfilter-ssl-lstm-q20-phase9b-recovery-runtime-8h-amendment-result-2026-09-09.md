# Phase 9B eight-hour amendment execution result

Updated: 2026-09-12 (Asia/Shanghai).
Status: `M4_P1_TERMINAL_FAILURE_SOURCE_CLOSURE_AND_STRICT_MOVEMENT_VETO`.
Plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Master: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.

## Current execution: corrected bounded P1 is terminally stopped

The September 12 retry is not complete. Factor completed four warmup chunks but
failed final source-closure validation because bound source files changed during
execution. Strict completed four chunks but failed the declared
`chain_without_movement` veto in chunk 3 for chain 3. The strict sampled states,
target values, and log-acceptance values were finite; the movement veto remains
binding and is not relaxed. The factor failure is an infrastructure/provenance
failure, while the strict failure is a sampler-health veto. Neither supports a
posterior, convergence, ranking, or research-direction claim.

The preserved terminal attempt is
`numerical-repairs/eigh-refinement-r2/launches/p1-d1884c1974/`; factor consumed
8,719.6063 worker seconds and strict 13,165.7078. The next action is to repair
source freezing and closure verification before launching a fresh versioned P1
attempt. P2 remains blocked.

At **September 12, 18:47 Shanghai**, the corrected bounded P1 retry is running.
The earlier r2 attempt, `p1-3ce9ecdb29`, failed at 14:06 Shanghai at
the setup identity guard after 10.84 measured GPU-worker seconds total
(factor 5.3681, strict 5.4675); it produced no scientific P1 result.
The repair records the committed pre-accounting setup
identity as reusable and binds all new stream/checkpoint work to the current
source closure. The corrected service
`bayesfilter-q20-phase9b-p1-r2.service` started at 18:35:54 Shanghai and
launched both arms in parallel on
non-display GPUs 1 and 0 with verified TensorFlow memory growth. Both workers
passed setup and are executing their first P1 chunks.

The live attempt is
`numerical-repairs/eigh-refinement-r2/launches/p1-d1884c1974/`. It reserves
46,800 seconds, against 15,270.51202170887 unreserved seconds remaining after
24,329.48797829113 measured seconds. The run is incomplete; wait for its
terminal P1 receipts before updating promotion status. No posterior or method
ranking follows from the launch.

The unspent balance before live settlement is 62,070.51202170887 seconds
(17.24 aggregate GPU-hours); reserved time is not settled spending.

Focused recovery/ledger tests pass **80/80 in 38.52 seconds**, including six
new setup-identity and successful-sibling reuse regressions. This is a
deliberate CPU-only check with `CUDA_VISIBLE_DEVICES=-1`; it consumes no GPU
budget. JUnit evidence is campaign
`setup-identity-validation-20260912-r1.xml`.
The original failed launch remains preserved as evidence of the provenance
bug and is not used as a scientific result.

### Phase 0 closeout and continuation audit

Both fresh training/tuning arms, both actual SIGKILL/resume canaries and both
full runtime/health measurements are complete. The runtime workers each
complete two 500-transition-per-chain chunks, four chains per chunk, with
finite required values, valid target status and no health veto. Verified
worker durations are 4,370.652188197 seconds for factor and
6,639.417831131999 for strict. The accounting repair settles strict from its
checksum-verified receipt instead of retaining the conservative full-cap
charge. Cumulative released reservations can exceed the allocation because
the same budget can be reserved, released and reused; they are not new funds.

| Arm | Measured first/steady chunk seconds | Forecast including one-chunk reserve and grace | Assigned P1 cap seconds |
| --- | --- | --- | --- |
| Factor | 2,197.6376 / 2,161.7456 | 16,519.6636 | 18,000 |
| Strict | 3,297.7092 / 3,329.4030 | 25,283.0138 | 28,800 |

These measured/derived forecasts are descriptive, not statistical bounds.
Both fit the assigned caps and original aggregate budget. Evidence is r2
`runtime-complete.json` and `phase0-readiness.json`, whose status is
`PASS_CURRENT_SOURCE_PHASE0_FOR_BOUNDED_P1`.

The September 12 continuation audit verifies that the target, comparator,
eight-sweep numerical method, committed tuning, seeds, criteria and stop
conditions are unchanged by the accounting/setup-provenance repair. Reusable
setup differs only in coordinator/ledger provenance; numerical-source changes
remain rejected, and new streams bind current execution sources. The current
GPU/XLA memory-growth receipts and non-display placement remove the historical
environment mismatch. Recovery equality and healthy runtime chunks answer
mechanics readiness only; they cannot replace convergence or posterior checks.
No scientific failure is erased, no threshold is relaxed and no new budget is
created. The declared numerical, artifact, resource and budget stops still bind.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close M4-P0 for bounded P1 | Recovery equality, full two-arm health, affordability and setup restoration pass | No veto in the four completed runtime chunks; failed historical attempts preserved | Longer trajectory behavior and forecast error | Continue the already active P1 attempt | Convergence or posterior admission |
| Keep P1 running within its allocations | Both workers restore setup and start their first chunks | No terminal P1 health result yet | P1 schedule completion and terminal screens | Inspect committed chunks, then both terminal receipts and measured costs | P1 pass, P2 authorization or superiority |

| Inference status | Current finding |
| --- | --- |
| Hard veto screen | Both arms pass P0 runtime health; P1 is incomplete |
| Statistically supported ranking | None; neither viable arm is ranked |
| Descriptive-only differences | Runtime, forecast and energy differences; two runtime chunks per arm do not establish superiority |
| Default-readiness | Not established |
| Next evidence needed | Terminal P1 screens, untouched sequential validation and downstream comparisons with uncertainty |

Red-team note: the repaired fixed banks and four healthy runtime chunks may
still miss ill-conditioned covariances on longer trajectories. A new health
or source-identity failure would trigger localization rather than relaxed
guards. Forecast uncertainty and unobserved P1 outcomes remain the weakest
evidence; a passing launch does not settle either.

### Previous execution snapshot

At **September 12, 02:45 Shanghai**, both fresh training/tuning arms complete.
Actual interruption/resume passes exact sample/trace equality for both arms,
reusing the first committed chunk and recomputing the interrupted second.
Receipts are r2 `reference-complete.json`, `interrupt-complete.json`,
`resume-complete.json` and `canary-result.json`. The interruption workers'
nonzero exits are the intended SIGKILL events, not failed recovery outcomes.

In `numerical-repairs/eigh-refinement-r2/launches/runtime-c355a7aa08/`, factor
completes both 500-transition-per-chain chunks; both health receipts pass.
Strict completes its first chunk with passing health and is running its
second. All three completed chunks have finite required values, valid target
status and no health veto. Factor worker elapsed time is 4,370.652188197
seconds; strict has no completed worker receipt yet. Finite extreme energy
values remain explanatory under the unchanged plan, not a convergence test.

The service remains active and P1 has not started. Both-arm timing and budget
forecasts await strict completion. The ledger records 13,308.582441157134
settled seconds, 57,600 reserved, 15,491.41755884286 unreserved; the completed
factor runtime cost awaits settlement with this wave. No budget is added.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue active runtime wave | Both recovery canaries exact; factor 2/2 and strict 1/2 runtime chunks healthy | No veto in completed chunks | Strict second chunk and new affordability forecast | Finish strict, settle costs, evaluate bounded P1 admission | Phase 0/P1 completion or posterior validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Completed runtime chunks pass; strict terminal result pending |
| Statistically supported ranking | None |
| Descriptive-only differences | Chunk/worker timings and finite energy extremes |
| Default-readiness | Not established |
| Next evidence needed | Strict terminal health, two-arm affordability, bounded P1 and downstream uncertainty-supported evidence |

### Earlier launch snapshot

The reboot restores matching NVIDIA module/NVML 580.178.04. The actual current
eight-sweep code passes the saved 32-row banks and original four-row endpoint
on GPU/XLA for both backends. Odd-dimensional positive/negative controls pass;
all four static signatures trace once and preserve HLO. Memory growth is
verified; TensorFlow allocator peak is 269,139,456 bytes. No numerical
threshold, derivative equation, target or health check is relaxed.

| GPU fixed bank | Invalid rows | Maximum likelihood residual | Maximum scaled analytic-score residual |
| --- | --- | --- | --- |
| Factor, 32 rows | 0 | 6.821210263296962e-13 | 1.2342530403212524e-14 |
| Strict, 32 rows | 0 | 2.2737367544323206e-12 | 4.434080564216233e-14 |
| Original endpoint, four rows per backend | 0 | 7.105427357601002e-15 | <=1.2205032944508617e-15 |

Artifact: `launches/eight-sweep-gpu-validation-5c70410169/`; supervised cost
53.115703790001135 GPU-worker seconds (worker-reported 52.004676009 seconds).
The 137 CPU tests are still current. Eight-sweep migration activates
`numerical-repairs/eigh-refinement-r2/`, preserving all prior evidence and
original start/ledger bytes; 124 bundles/3,109 tensors verify. Its receipt and
source snapshots are in
`runtime-health-diagnostic-20260911-r1/eight-sweep-migration-7a685780c8/`.

At September 12, **00:44:39 Shanghai**, service
`bayesfilter-q20-phase9b-eigh8-20260912-r1` starts fresh setup/tuning in
`r2/launches/reference-67d95ab1f2/`. Both workers are active in parallel:
factor on non-display GPU 1, strict on non-display GPU 0. Each has verified
memory growth and XLA. Setup/tuning is not yet complete. The unchanged
coordinator will run actual interruption/resume canaries, both full runtime
health checks, affordability estimates and bounded P1 when admissible.

At 00:48 Shanghai, both trained charts are committed and all six 32-row
preflights (beta 0, 0.5 and 1 per arm) pass. Both public tuners are now active
in separate processes. Completed calls inspected so far have finite
samples/targets/scores/log acceptance and valid target status. Their short
screening results do not substitute for the full runtime health check.

At launch: **9,523.699322834134 consumed**, **7,200 reserved**,
**69,676.30067716587 unreserved seconds**. The unspent balance before live
worker settlement is 76,876.30067716587 seconds (21.3545 GPU-hours); reserving
time is not spending it. Per-arm ceiling remains 28,800 seconds. The prior
runtime forecasts do not price eight sweeps and are not reused.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue repaired numerical scope | Both fixed GPU banks and endpoint agree with native CPU reference | All tested rows valid; old failures preserved | Wider trajectory stability and new runtime cost | Fresh parallel tuning/recovery/runtime, then affordable P1 | Convergence, posterior validity or universal eigen accuracy |
| Resume execution | Matching driver versions, growth and eligible non-display GPUs | No current host/source/budget veto at launch | Live setup/tuning outcome | Monitor persisted stage receipts and stop on declared failures | Guaranteed completion within remaining budget |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Fixed-bank numerical repair passes; no fresh full-runtime result yet |
| Statistically supported ranking | None |
| Descriptive-only differences | Diagnostic timings and residual magnitudes cannot rank stochastic candidates |
| Default-readiness | Not established; no posterior or P2 promotion |
| Next evidence needed | Current-source recovery/health and two-arm runtime; downstream sequential checks and uncertainty for any ranking |

Post-run red-team: successful fixed inputs may miss other ill-conditioned
covariances. A fresh preflight or runtime numerical failure would require
localization; it would not justify reseeding or relaxing a guard. The greatest
remaining uncertainty is trajectory stability and cost after eight sweeps.

## Historical pre-reboot state: CPU repair passes; GPU validation blocked

Trusted GPU checks at September 11, 15:36 Shanghai still fail: loaded driver
580.173.02 versus installed NVML 580.178.04. Unattended upgrades installed
the new driver at 06:02, after the failed preflight. A host reboot is requested
by the OS; no disruptive host action is performed without operator approval.
No GPU worker is running and no budget is reserved.

The remaining preflight defect is localized on saved factor row 15: four
Jacobi sweeps stop with eigenpair residual 2.08864e-11, above the unchanged
8.91555e-12 bound. The covariance is positive definite (native CPU minimum
1.7476690e-11), and instrumented/uninstrumented one-row value and score agree
exactly. Eight and twelve sweeps both pass; no guard is relaxed. This is a
convergence defect in the refinement implementation, not candidate invalidity.

The eight-sweep prototype passes both saved 32-row banks against same-input
native CPU full likelihood/analytic-score evaluation (64/64 valid):

| Arm | Maximum likelihood residual | Maximum scaled score residual | Fixed-bank equivalence |
| --- | --- | --- | --- |
| Factor | 2.3874235921539366e-12 | 1.0014280349879253e-13 | Pass, CPU/XLA only |
| Strict | 1.4779288903810084e-12 | 7.365481351286461e-14 | Pass, CPU/XLA only |

The core default changes only from four sweeps to eight, preserving all
residual, orthogonality, SPD, analytic-derivative and health checks. Added
fixed-row regressions fail on both old paths and pass with the repair. All
**137 focused CPU tests pass** (121.77 seconds, zero failures/errors/skips),
covering the eigensolver, filter, principal root and recovery coordinator;
`git diff --check` passes. The
old numerical migration is stale: GPU full-bank validation and a fresh
source-bound namespace are required before new parallel tuning/recovery/
runtime/P1. Do not reuse four-sweep receipts or forecasts. The CPU bank check
uses 411.4079 wall seconds and zero GPU seconds. Earlier partial CPU timeouts
are preserved separately and do not establish failed numerical outcomes.

Artifacts under `runtime-health-diagnostic-20260911-r1/`:
`preflight-row-localization-r1/`, `preflight-eight-sweep-bank-r1/`, and
`cpu-attempt-reconciliation-20260911T0414Z.json`.

`eight-sweep-validation-a8df05dcf2/validation.json` records source snapshots,
hashes, exact regression command/environment and unchanged budget. Its
read-only resume check rejects the stale migration before launch and verifies
byte preservation of the original start, ledger and migration. Current totals
remain 9,470.583619044133 consumed, zero reserved and 76,929.41638095587
aggregate GPU-worker seconds remaining. No GPU seconds are spent on this
CPU-only repair.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep GPU work stopped | Trusted inventory unavailable | Driver/library mismatch | Host restart scheduling | Operator restores matching driver versions | GPU hardware failure or scientific rejection |
| Accept the bounded CPU numerical repair | Both fixed banks agree with same-input CPU reference | No invalid rows; original failures preserved | GPU arithmetic and wider inputs untested; new runtime cost unknown | Regression, host repair, GPU validation, fresh numerical migration | GPU clearance, convergence or universal eigen accuracy |

| Inference status | Current finding |
| --- | --- |
| Hard veto screen | Original strict runtime and four-sweep preflight remain rejected; current fixed-bank CPU repair passes; host GPU inventory fails |
| Statistically supported ranking | None; no new stochastic candidate comparison |
| Descriptive-only differences | CPU timings and numerical residuals do not rank methods |
| Default-readiness | Not established; GPU validation, fresh tuning/recovery/runtime and downstream checks remain |
| Next evidence needed | Matching drivers, unchanged-input GPU checks, current-source recovery/health and two-arm timing; downstream uncertainty for any ranking |

### Post-run skeptical review

Engineering: fixed-row red/green regressions and the existing recovery suite
pass; stale-source restart is prevented. Numerical: the computed principal-root
path now agrees with native CPU evaluation on the declared fixed banks within
the predeclared tolerances. It remains a floating-point approximation with
explicit residual/orthogonality rejection, not an exact or universally
convergent eigensolver. Scientific: no new training, sampling, convergence or
method-ranking evidence is produced.

The strongest alternative explanation is CPU/XLA-specific success that fails
on GPU or on other ill-conditioned covariances. Original GPU physical banks
were not saved, so CPU reconstruction alone cannot establish bitwise GPU
replay. A fixed-bank GPU failure, derivative mismatch or fresh runtime health
veto would overturn wider repair acceptance and trigger another localized
repair. The weakest evidence is unmeasured GPU validity and cost of eight
sweeps. Do not transfer the old timing forecast or infer that 21.3693 remaining
GPU-hours guarantees completion. After host repair, validate the saved failed
inputs on GPU before fresh parallel tuning/recovery and a two-arm forecast.

Latest attempt: after migration passes **135 combined CPU regressions**, both
workers start in parallel on non-display GPUs 0 and 1 in
`numerical-repairs/eigh-refinement-r1/launches/reference-b6ab845731/`.
The service runs September 11, 04:52:31–04:52:55 Shanghai. Both reject fresh
beta-zero chart preflight before any optimizer update or HMC tuning. GPU
memory growth and XLA initialize correctly; this is not a launch-permission
failure. No worker is now active. The subsequent CPU localization and repair
are recorded here; a same-command retry or reseeding remains unjustified.

That failed attempt consumes 42.425130410002776 GPU-worker seconds. Current
ledger: **9,470.583619044133 consumed**, zero reserved,
**76,929.41638095587 remaining seconds**. The original campaign start and
allocation are unchanged. The migration receipt verifies 101 committed bundles
and 2,697 tensors. The endpoint evidence below does not clear this wider bank.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the preceding four-sweep attempt | Fixed 32-row chart screen fails on both arms | Preoptimizer gate remains enforced | GPU reproduction of the eight-sweep repair is pending | Preserve failure; validate repaired code on fixed banks | Successful repaired-scope training, tuning or recovery |

The saved covariance is positive definite. Independent high-precision and CPU
checks reject covariance-subtraction cancellation as the supported cause of
the runtime stop. The GPU/XLA eigensolver's internal binary32 Jacobi cutoff
causes a false-negative minimum eigenvalue in this binary64 filter. Both
strict and factor-cached paths share this bug. The bridge's subsequent NaN
guard and proposal-energy rejection behave correctly and remain unchanged.

The integrated binary64 eigen-refinement repair passes 122 CPU regressions and
`launches/candidate-diagnostic-a7e029690b/strict/` on GPU/XLA. Both backends
return four valid rows, trace once, and agree with the independent CPU target
within 7.11e-15 in likelihood and 6.10e-15 in scaled analytic score. The prior
full-filter prototypes also pass central differences at two step sizes.
No actual negative eigenvalue is clipped. This validates a local numerical
repair, not a new sampler or posterior.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Migrate both numerical scopes | CPU/GPU value/score and sign controls pass | Original unhealthy strict chunk stays rejected | Wider trajectory stability and repaired runtime unmeasured | Fresh parallel chart/tuning, recovery and runtime | Convergence or posterior correctness |
| Preserve campaign allocation | 9,428.15848863413 seconds consumed; 76,971.84151136587 remain; zero reserved | No budget veto | Current two-arm runtime forecast unavailable | Retain 28,800-second arm ceiling and measured P1 gate | Guaranteed completion within budget |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Original strict runtime rejected; shared eigensolver defect reproduced in both paths; local repaired checks pass |
| Statistically supported ranking | None |
| Descriptive-only differences | Prototype/runtime timings and residuals do not rank stochastic candidates |
| Default-readiness | Not established; repaired-scope tuning/recovery/runtime/P1 are pending |
| Next evidence needed | Fresh current-source health/recovery and downstream sequential sampling; uncertainty-supported comparisons for any ranking |

The explicitly numerical source migration and fresh namespace are activated.
The first fresh attempt fails preflight as recorded above. Use
`numerical-repairs/eigh-refinement-r1/` within the original campaign, with its
same ledger and lock; neither old sibling's tuning or canary receipt clears the
new numerical closure. The governing investigation/continuation plan is
`bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md`.

Post-run red team: the strongest remaining alternative is that four refinement
sweeps solve this endpoint but not other poorly conditioned matrices. Residual,
orthogonality, sign and full runtime health checks must continue to veto; a
future failure triggers localization, not relaxed thresholds or retries until
convergence happens. The weakest evidence is trajectory coverage, not the
independent sign check at this endpoint.

## Historical snapshot: invalid covariance localized

This snapshot predates the positive-definiteness and eigensolver diagnosis;
its covariance-cause and strict-only continuation statements are superseded.

The prefix and cached-gradient diagnostics have completed. No campaign GPU
worker is active at this update; no new tuning or P1 has launched. The exact
159-transition prefix matches all saved samples, acceptance ratios/decisions
and sampled targets bitwise. The first invalid evaluation is the second
leapfrog endpoint of transition 159, chain 4: finite model parameters, but a
placement-covariance minimum eigenvalue of -1.4256699903299468e-11. Numerical
status rejects that row; the bridge deliberately emits NaN, contaminating the
remaining leapfrog and producing the stored negative-infinite acceptance.
Checkpoint corruption/reporting does not explain this stop. Covariance
conditioning/cancellation versus a target implementation defect remains open.

The strict-only tuning draft was withdrawn before launch. The serializer-only
wrapper/source migration are restored, with 56 CPU regressions passing.
Independent validation checks 33 committed bundles and 376 tensors. Evidence:
`runtime-health-diagnostic-20260911-r1/postrun-validation.json`,
`launches/prefix-diagnostic-83873ea777/strict/prefix-comparison.json`, and
`launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json`.
The cached-gradient preceding-transition control is exact; one unrelated finite
L=3 acceptance ratio differs by approximately 4e-16, so that replay is not
claimed wholly bitwise exact.

Settled accounting: **9,269.455504760117 consumed**, **zero reserved**, and
**77,130.54449523988 remaining aggregate GPU-worker seconds** (21.4252 hours).
Caps remain 86,400 aggregate seconds and 28,800 per arm. Next: inspect the
covariance recursion at the saved finite endpoint, retaining all invalid-row
and energy guards. A numerical repair needs equivalence/derivative tests and
an honest source migration; fresh strict tuning needs new namespaces and
recovery evidence. Factor's successful unchanged-scope evidence is preserved.

## Historical prefix launch snapshot

At **September 10, 17:08:21 UTC**, user service
`bayesfilter-q20-phase9b-prefix-diagnostic-20260911-r1.service` starts the
159-transition diagnostic prefix in `launches/prefix-diagnostic-83873ea777/`.
Worker PID 869730 uses the original non-display GPU 1; memory growth is verified
before initialization, XLA compilation is observed, and headroom monitoring is
active. The worker has a 1,800-second cap and preserves the same target, chart,
public tuning handoff and seeds. Added traces expose proposed states, gradients,
potential and momenta. It carries TFP's accepted kernel results from the original
initial bank rather than reconstructing a single transition's initial gradient.
Exact agreement with the saved prefix is a measured check, not assumed.

Completed tiny CPU and same-GPU one-step diagnostics **do not reproduce** the
failure. Their controls differ from the saved state by 3.064e-9 and 2.509e-9,
respectively; the problematic proposal becomes finite. No seed bug or numerical
fix is established. These diagnostics cannot remove the original runtime veto.
The same-GPU diagnostic consumes 42.782334198011085 worker seconds; its
TensorFlow allocator peak is 34,018,304 bytes. All numerical dependency hashes
remain unchanged. Its exact diagnostic sources are saved with its results.

Current settled accounting: **8,577.284164943063 consumed**, **1,800 reserved**,
**76,022.71583505694 unreserved seconds**. The unspent balance before settling
this live worker is 77,822.71583505694 seconds (21.6174 GPU-hours). Do not count
the reservation as new spending or add it to that balance. P1 remains paused;
the service performs debugging only and does not auto-retune or relax a veto.

## Completed recovery and strict runtime rejection

The resumed service stopped on **September 10, 2026, 17:34:11 Shanghai**.
Both reference/recovery arms completed, and actual interruption/resume canaries
passed exact tensor and trace equality. The first committed chunk was reused
and the interrupted second chunk was recomputed. The serializer failure is
resolved; these short recovery fixtures do not establish numerical stability.

In `runtime-c48f1f0535`, factor completed both 500-transition calls and all
health checks. Strict committed the first call, then failed the unchanged
nonfinite-acceptance and energy checks. A checksum-verified CPU inspection
finds **one negative-infinite acceptance ratio, zero NaNs and zero positive
infinities**, at transition 159/chain 4 (one-based). That proposal was rejected;
all sampled states and sampled-state targets/status remained finite/valid.
Recomputing the original health check reproduces both vetoes. Proposal states
and proposal scores were not saved, so the internal numerical cause remains
under investigation. This is a different failure from undefined short-chain
tail ESS in the earlier tuning serializer.

At the runtime-stop snapshot, the ledger records **8,534.501830745052 consumed**, **zero reserved**,
and **77,865.49816925495 remaining aggregate GPU-worker seconds** (21.6293
GPU-hours). The 86,400-second allocation and 28,800-second per-arm ceiling are
unchanged. No P1 worker is running; Phase 0 and P1 are not complete. Do not
restart the identical unhealthy checkpoint or infer a two-arm forecast from
the successful factor arm alone.

Investigation plan: `bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md` (local artifact name).
Stored inspection and bounded proposal-replay diagnostics are preserved under
the campaign's `runtime-health-diagnostic-20260911-r1/`. The next justified
action is localization of the original rejected proposal, followed by a
regression-tested implementation repair or fresh public scope-specific tuning
as the evidence warrants. Existing vetoes and failed evidence are preserved.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Recovery mechanics pass | Both same-GPU canaries replay exactly | None in the canaries | Eight-transition coverage only | Preserve receipts | Universal interruption recovery |
| Reject current strict runtime candidate | Two-arm full-health criterion fails | One nonfinite proposal acceptance/energy | First internal source of invalidity not yet exposed | Localize, then repair or retune using fresh partitions | Target or research-direction rejection |
| Keep factor timing evidence | Both calls pass health | None in those calls | One timing replication, distinct chart/backend | Reuse successful sibling; await valid strict timing | Superiority or a joint runtime forecast |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Current strict runtime fails; both factor calls pass. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Timing and acceptance differences; finite energy extremes. |
| Default-readiness | Not established; no posterior admission. |
| Next evidence needed | Supported repair, fresh full-health strict runtime and bounded P1; multi-seed downstream evidence for any ranking. |

Post-run red team: retained states can stay finite because TFP rejects invalid
proposals; this cannot erase the predeclared energy veto. Conversely, this one
candidate failure cannot establish that the target or method is wrong. The
weakest evidence is the unsaved proposed-state telemetry. A controlled replay
that does not reproduce the original transition cannot establish its cause.

## Historical September 10 repair and resumed execution

Execution resumes at **September 10, 2026, 08:23:01 UTC / 16:23:01
Asia/Shanghai** as user service
`bayesfilter-q20-phase9b-serializer-20260910-r1.service`, coordinator PID
804934. The new reference attempt is `reference-4f43b85252`. It reuses the
completed factor reference and both saved charts, and reruns only strict's
uncommitted tuning. Strict runs on non-display GPU 1; memory growth is verified
before device initialization and its XLA compilation is observed. Both original
non-display comparator GPUs qualify at preflight; subsequent two-arm waves
remain process-parallel when device availability permits. The display GPU is
not used by this launch.

At 08:29:18 UTC, strict worker PID 804943 has completed ten tuning calls.
The service remains active and both its execution-source binding and preserved
checkpoint-source identities verify. This snapshot is saved as
`serializer-repair-validation-20260910-r1/resumed-execution-snapshot.json`.
The full GPU recovery canary and P1 have not yet completed.

The serializer now stores NaN and positive/negative infinity as tagged values,
restores Python floats and tuples, and uses the public tuner's existing safe
artifact hash. That hash also preserves the old finite factor checkpoint.
An initial, unlaunched repair draft changed tuple hashing; testing against the
real factor checkpoint exposed that incompatibility before any GPU restart.
The corrected patch passes **161 CPU tests** in 43.99 seconds, including
actual disk reload, veto preservation, invalid-marker rejection, legacy hashes,
SIGKILL publication boundaries, source-migration accounting and controller replay.

The saved strict artifact itself round-trips through a separate diagnostic
checkpoint with all six NaN tags, selected candidate 0 and the candidate 1
nonfinite-efficiency veto preserved. That diagnostic does not publish a live
tuning checkpoint or substitute for the interrupted public tuner. Original
committed evidence, campaign-start, plan and budget remain unchanged through
the repair preflight. New worker reservations and elapsed time continue in the
same ledger; no new allocation is minted.

The final `source-migration.json` binds exactly one changed source file, the
recovery wrapper. Numerical setup, tuning, HMC/controller, health, seeds and
runtime-estimation functions are unchanged. Jobs distinguish the new execution
source from preserved setup/stream identities. The unlaunched migration draft
is archived as `source-migration-superseded-draft-1.json`. The frozen amendment
plan is not edited.

### Why the NaN occurred

The stopped strict artifact locates the diagnostic in selection replication 1
of candidate 1 (`L=8`), parameter `observation_bias.0`. Input draws are recorded
finite and R-hat is finite (about 2.2589), while upper-tail ESS is NaN. Its
minimum propagates to tail ESS and minimum tail ESS; those three fields appear
twice because final-kernel evidence repeats candidate-selection evidence.

`hmc_convergence.py` builds binary q05/q95 indicators and passes them to
`_real_fft_cross_chain_ess`. A constant indicator has zero within- and
between-chain variance. Its autocorrelation formula divides zero by zero, so
ESS is undefined even though the original draws are finite. A CPU regression
with two high observations among 64 otherwise equal values reproduces the
finite-input/finite-R-hat/NaN-upper-tail pattern and retains
`nonfinite_convergence_diagnostic`. This is a mechanism reproduction, not a
reconstruction of the unsaved original draws. The eight-results/four-chain
selection window is a mechanics screen, not posterior convergence evidence.

The public tuner already handles undefined diagnostic values in JSON and
rejects that candidate's selection efficiency. The recovery wrapper alone
used strict JSON on raw floats, causing a serialization exception **after**
tuning returned. This is not evidence of NaN HMC states or target values in
that artifact. The repair does not suppress diagnostics, replace NaN with zero,
relax a gate, or select the rejected candidate.

### Evidence and next steps

Repair plan: `bayesfilter-ssl-lstm-q20-phase9b-nonfinite-checkpoint-repair-2026-09-10.md`.
Evidence under the existing campaign's
`serializer-repair-validation-20260910-r1/` includes the exact source diff,
CPU test log/XML, saved-artifact diagnostic and protected hashes, validated
migration receipt, and trusted GPU inventory/placement receipts. The actual
service command is the unchanged P1 entrypoint with `--campaign-root` and
`--resume`; its separate log is `supervisor-serializer-20260910-r1.log`.

At the 08:25 UTC snapshot, settled spend remains 2,187.64961813399 seconds,
with 3,600 seconds reserved for the active strict worker and
80,612.35038186601 seconds unreserved. Live worker time is not yet settled;
84,212.35038186601 seconds was the unspent balance **before** restarting.
The eight-hour arm ceiling and 24-GPU-hour total are unchanged.
Next are completion of strict reference, actual interruption/resume canaries,
two 500-transition runtime measurements per arm, Phase 0 closeout and bounded
P1 if the current forecasts and remaining budget permit. P2 is not auto-launched.
No reliable end time is available until current timing completes.

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Resume localized infrastructure repair | Saved public and legacy checkpoint round-trips pass; 161 CPU checks pass | Candidate 1's nonfinite-efficiency veto preserved; no numerical gate relaxed | Fresh GPU reference, recovery and full timing are still in progress | Reuse completed factor evidence, finish strict, then the planned canary/runtime/P1 sequence | No completed current canary/P1, posterior validity or research-direction rejection |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Undefined selection efficiency vetoes candidate 1 in the stopped artifact; the serialization defect invalidated checkpoint publication, not all candidates. |
| Statistically supported ranking | None established by this repair or the tiny selection window. |
| Descriptive-only differences | Short-chain ESS, R-hat, acceptance and runtime observations. |
| Default-readiness | Not established; CPU recovery and successful tuning are not posterior admission. |
| Next evidence needed | Current GPU recovery equality and full health/timing, then bounded sequential validation; uncertainty-supported replication is required for method ranking. |

Post-run red-team note: poor mixing or repeated short-window values remain
alternative explanations for the undefined ESS. Serialization success cannot
resolve them. A restored-hash mismatch, altered veto or failed GPU canary would
overturn the engineering pass and trigger a localized investigation. The weakest
scientific evidence remains the tiny selection window; it is not promoted.

## Historical failure discovered during ETA check

The service stopped at **2026-09-09 16:36:39 UTC** (September 10, 00:36:39
Asia/Shanghai), not at the worker timeout or campaign budget ceiling. Factor
completed its reference worker in 896.9274621399818 parent-measured seconds.
Strict completed 25 tuner calls and wrote its public tuning result, but failed
inside `typed_tuning_payload` while hashing the in-memory result for checkpoint
publication: `ValueError: Out of range float values are not JSON compliant: nan`.
Its lifetime of 1290.722155994008 seconds is charged despite the failure.

The public strict result reports `passed: true`, selected candidate 0, and a
nonfinite selection-efficiency veto for candidate 1. Its JSON contains no
nonfinite numeric literals; the failure is in the separate typed-checkpoint
encoder/hash path, which does not yet preserve nonfinite diagnostics. This is
an infrastructure repair trigger, not evidence that the selected target or
whole research direction failed. Any repair must preserve the rejected
candidate's veto and must not change NaN into a favorable score. The exact
in-memory NaN field and lossless round-trip repair remain to be checked.

Settled budget: **2187.64961813399 seconds**, zero reserved, **84212.35038186601
seconds (23.3923195505 GPU-hours) remaining**. The complete two-arm reference,
interruption/resume canary, runtime forecast and P1 have not completed. No
reliable finish ETA is available until repair and fresh runtime measurement.
Preserve the completed factor worker, both committed charts and strict tuner
artifacts. Repair typed serialization, add nonfinite round-trip regressions,
record a source migration without resetting spent budget, and resume only the
unfinished work. No new GPU work was launched by the ETA inspection.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Repair checkpoint serialization before resume | Two-arm reference incomplete | Typed checkpoint publication failed; selected public tuner candidate passed its limited screen | Exact nonfinite field and lossless restoration not yet checked; fresh full runtime absent | Focused serialization/recovery repair within remaining allocation | No completed canary/P1, posterior correctness, candidate ranking or research-direction rejection |

The live snapshots below are historical observations preceding this failure.

## Engineering repair and pre-run audit

The unused prepared ledger can now be adopted without minting another budget;
its original bytes are archived and interrupted initialization is idempotent.
The actual P1 CLI delegates to the parallel checkpoint coordinator. Chunk
archives use global checkpoint indices, actual stage seeds and previous states;
cumulative archives are count-versioned. P1 uses distinct seeds, measured
per-arm reservations, the full-status shared controller and unchanged TFP HMC.
Failed warmup prevents retained sampling and is recorded as candidate failure.
A partial resource stop is not misreported as completed P1 and remains resumable.

No numerical-kernel/status-reuse optimization is introduced. The audit rejected
making that optimization an entry prerequisite after the cap increased. The
finiteness-only tuning shortcut is not equivalent to complete sampled-state
health and is not substituted for it.

Focused validation: **128 passed**, 191 dependency deprecation warnings, 35.83s.
CPU-only command uses `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true`
and `/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q` on
`test_ssl_lstm_q20_phase9b_recovery_runtime.py`,
`test_ssl_lstm_q20_phase9b_p1_canary.py`, `test_ssl_lstm_q20_phase9b_readiness.py`,
`test_durable_tensor_checkpoint.py`, `test_campaign_budget_ledger.py`,
`test_parallel_tuning.py`, `test_display_gpu_policy.py`, and
`test_neutra_hmc_route_policy.py`, all under `tests/`. Compile checks pass.
Retained-stage fixture diagnostics are deliberately forced to exercise control
flow; those tests establish recovery mechanics, not convergence.

Trusted NVIDIA inventory at 23:47 +08:00 passes: GPU 1 idle, GPU 0 at 2% with
337 MiB used, GPU 2 at 36% with 768 MiB used. These are snapshots, not guaranteed
availability. Launch selection rechecks display identity, load and memory;
unrelated processes are not terminated. GPU framework growth is verified in
each worker before logical-device initialization.

## Campaign and live artifacts

Root:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
Budget: 86,400 aggregate GPU-worker seconds, each worker capped at 28,800.
At the 16:04:50 UTC snapshot, 7,200 seconds are reserved for the reference wave,
zero seconds are settled and 79,200 seconds remain unreserved. Live elapsed
worker time has not yet settled; zero settled time does not mean zero usage.
Older campaigns and their budgets remain separate and unchanged.

The actual P1 entrypoint started at **16:00:10 UTC**, detached user service
`bayesfilter-q20-phase9b-24gpuh-20260909.service`, coordinator PID 704521.
Service invocation uses `Type=exec`, the repository working directory,
`KillMode=control-group`, `TimeoutStopSec=30`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`CUDA_VISIBLE_DEVICES=-1` for the framework-free coordinator, and
`PYTHONUNBUFFERED=1`. Workers override visibility with their selected UUID.
The command is the frozen plan's P1 entrypoint with the same campaign root and
`--resume`. Console log: `supervisor-20260909T155926Z.log` under the root.

First attempt: `launches/reference-561614e446/`. Factor worker PID 704530 runs
on GPU 0 (`GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`), not the display device.
Its chart checkpoint is committed and nine fresh public-tuner chain calls have
completed. The launch inventory found GPU 1 at 64% utilization from another
process, above the owner's 40% threshold, so strict queues. The display GPU
remains untouched. This is process-parallel-capable execution but **one active
arm at this snapshot**, not two simultaneous workers. Placement is rechecked
before queued work starts; no unrelated process is killed.

Worker memory growth is verified before logical-device initialization. Source
and frozen-plan binding still match. A committed bundle checksum verifies;
the chart bundle stores its checkpoint as a typed Python payload (no separate
TensorFlow tensor files in that bundle). GPU sequential health, interruption
comparison and full timing are not yet complete. Snapshot:
`execution-snapshot-20260909T160450Z.json`. Older ledger SHA-256 values still
match the prior result: historical ten-hour `f7ad2484ca723cfb210a38a60ae3ea4f841cff46a321f6294285562e7c5207ce`,
older readiness `1961d07cf421b06abfb6ec4b62830b44e4052849505d72cd3cf0cea087160415`.

`campaign-start.json` freezes source hashes, plan, command, Python, Git state,
budget and claim boundary. `campaign_budget_ledger.json` records all reservations
and parent-measured settlements. Per-attempt jobs, supervision and worker
manifests preserve seeds, source/target/training identities, GPU, memory growth,
XLA/TF32, wall time and output paths. `phase0-readiness.json` is issued only after
fresh recovery, complete runtime health and affordable measured allocations.
`p1-result.json` requires actual bounded P1 terminal evidence; partial results
have separate versioned names. Canary and validated timing files remain frozen.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue approved fresh campaign | Engineering checks pass; GPU reference setup/tuning running | Factor startup growth passes; strict queued by GPU 1 load; no new sequential health result yet | Fresh charts may tune poorly or mix slowly | Finish canary/runtime; automatically run bounded P1 if ready | No recovery/health or posterior claim from startup alone |

## Inference status

| Evidence class | Current conclusion |
|---|---|
| Hard veto screen | CPU mechanics pass; fresh GPU candidate vetoes not yet evaluated |
| Statistically supported ranking | None |
| Descriptive-only differences | Historical timings motivate feasibility only, not factor/strict superiority |
| Default-readiness | Not established |
| Next evidence needed | Fresh GPU replay/health/forecast; bounded P1 results; later multi-seed uncertainty and downstream posterior/reference validation |

Post-run red-team pending. Main pre-run alternative explanation: an inherited
small chart or short bounded schedule can fail mixing even with a correct
controller. That rejects the present candidate/screen, not the entire transport
research direction. Source identity, corrupted evidence or unavailable required
health are genuine continuation vetoes.
