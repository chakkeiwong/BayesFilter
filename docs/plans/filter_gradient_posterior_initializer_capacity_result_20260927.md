# Complete posterior initializer capacity result

The repaired graph/XLA initializer passes all eight CPU/GPU reuse cases,
with full original records, independent Gaussian checks, exact twenty-call
replay and one stable owner/trace. Repeated owner replacement passes Python
collection but leaves substantial native host memory. The frozen prior CPU
D3 implementation crashes; that cell remains failed and supplies no capacity
ratio. This unit closes the bounded capacity investigation, not the whole
repair program or actual-consumer lifecycle qualification.

The [plan](filter_gradient_posterior_initializer_capacity_20260927.md) uses
D1/D3 factor_max=1 identifiable Gaussian fixtures and frozen Git 031692a0b
complete reference records. Numerical runtime is unchanged from 352776f63.
The earlier [36-worker cost cohort](filter_gradient_posterior_initializer_cost_result_20260927.md)
remains sealed. These single-process trajectories are descriptive, without
a statistical ranking or a general memory bound. Graph control retains the
original compiled dependencies; it is not an entirely non-XLA program.

| Device / D | Arm / run | Cold RSS MiB | After twenty warm calls MiB | Final GPU current / maximum per-call peak bytes |
|---|---|---:|---:|---|
| CPU / 1 | graph / 04532 | 1839.3 | 1841.8 | CPU reference |
| CPU / 1 | prior / 04533 | 1481.0 | 9836.1 | CPU reference |
| CPU / 1 | xla / 04529 | 2204.7 | 2207.3 | CPU reference |
| CPU / 3 | graph / 04535 | 3210.9 | 3214.7 | CPU reference |
| CPU / 3 | prior / 04536 | unavailable | failed | unavailable |
| CPU / 3 | xla / 04534 | 3797.2 | 3800.3 | CPU reference |
| GPU / 1 | graph / 04541 | 2255.4 | 2257.9 | 115,456 / 1,235,968 |
| GPU / 1 | prior / 04542 | 1822.1 | 9540.0 | 28,672 / 566,272 |
| GPU / 1 | xla / 04540 | 2492.3 | 2494.8 | 14,592 / 185,088 |
| GPU / 3 | graph / 04545 | 3531.3 | 3534.4 | 125,440 / 1,302,016 |
| GPU / 3 | prior / 04543 | 2943.8 | 16644.4 | 28,928 / 580,096 |
| GPU / 3 | xla / 04544 | 3933.3 | 3936.3 | 14,848 / 383,232 |

All accepted arms preserve both original and changed inputs. Every candidate
retains two numerical dependencies and one trace; XLA HLO is unchanged by
the changed operands and no host numerical callbacks appear. The observer
controls retain the same payload and snapshot counts without numerical calls:

- CPU, 04531: reuse reporting adds 2.47 MiB; owner reporting adds 0.24 MiB.
- GPU, 04548: reuse reporting adds 2.60 MiB; owner reporting adds 0.35 MiB.

The similar small candidate/observer growth supports reporting allocation as
an explanation for this bounded reuse trajectory. It is not an exact byte
subtraction or proof of leak freedom. The prior D1 CPU and D1/D3 GPU cases
pass the numerical gates despite substantial host-memory growth.

Run 04536 fails with SIGSEGV after 243.995367 seconds, without completed
capacity JSON or JUnit. GDB diagnostic 04537 reproduces native allocation
failure after 278.808786 seconds during call eleven. Ten completed calls
pass replay and independent Gaussian checks. Maps rise from 13,914 after
call one to 63,459 after call ten against vm.max_map_count=65,530.
The log reports LLVM "Cannot allocate memory" at lines 666--667, followed
by defunct JIT resources and SIGSEGV in ORC IRCompileLayer / XLA JitCompiler.
The mapping limit is strongly supported, but the exact failed syscall and
peak count were not captured. The late external monitor is empty and is
not evidence. No system limit, global cache, environment or package changed;
no further prior-D3 CPU retry is planned.

| Device / D | Owner replacement run | RSS after first / fourth owner MiB | RSS after all Python owners released MiB | Post-release heap MiB | Post-release GPU current bytes |
|---|---|---|---:|---:|---:|
| CPU / 1 | 04530 | 2199.3 / 4082.9 | 3455.9 | 2792.6 | N/A |
| CPU / 3 | 04539 | 3793.0 / 6780.9 | 6780.9 | 5881.7 | N/A |
| GPU / 1 | 04546 | 2489.3 / 4328.7 | 3718.6 | 2764.9 | 9,984 |
| GPU / 3 | 04547 | 3923.3 / 6559.6 | 6559.6 | 5493.3 | 10,240 |

All weak references to each target, owner, compiled function and dependency
scope clear. Most retained host memory is in the process heap. This repeats
the distinction established by the [KDM native investigation](filter_gradient_kdm_native_ownership_result_20260926.md):
Python collection does not prove native executable eviction. Retain one
owner for fixed callbacks/settings, pass varying inputs as tensor operands,
and use explicit bounded worker lifetimes when configurations change.
Initializer-specific and actual DZ5 containment remains to be executed.

| GPU dimension / arm | Current allocation on warm owner return, bytes | Current after formatting, bytes | Returned tensor payload, bytes |
|---|---|---:|---:|
| 1 / graph | 173,312--173,568 | 115,456 | 8,704 |
| 1 / xla | 88,320--88,320 | 14,592 | 8,704 |
| 3 / graph | 212,480--215,040 | 125,440 | 39,216 |
| 3 / xla | 124,928--124,928 | 14,848 | 39,216 |

The graph-control device-peak trigger is reproduced and localized before
public formatting. Each per-call peak is already reached on owner return,
and completed current allocation stays constant across twenty calls. The
returned payload is identical in size across graph/XLA. These observations
separate transient numerical allocation from returned tensors and retained
current bytes. They do not identify an exact allocator operation or waive
capacity checks on larger/actual targets. Default XLA uses lower device
peaks in these fixtures; this is descriptive, not statistical superiority.

The independent v3 analyzer preserves 15 valid numerical cells, failed
04536, two valid observers and explanatory diagnostic 04537. It checks
complete original/Gaussian/replay records, all memory/collection stages,
environment/fixture identity and the full 690-file original-source closure.
Only the five reviewed reporting/diagnostic paths may differ across workers;
every allowed difference is listed in the JSON. Failed cells receive no
invented memory values or capacity ratios. Frozen prior groups are
explanatory comparators, while candidate and observer gates remain mandatory.

| Decision | Criterion / veto | Uncertainty and next action | Not concluded |
|---|---|---|---|
| Accept bounded candidate reuse evidence | Eight CPU/GPU cases pass numerics, replay and stable ownership | Check real consumer dimensions and settings | General leak freedom or D23 capacity |
| Preserve failed prior CPU D3 | Allocation failure reproduced with native stack | Keep both failures; no unchanged retry or ratio | Exact failed syscall or a numerical mismatch |
| Require explicit native lifecycle | Python owners clear, host retention persists | Qualify actual initializer/DZ5 worker exit and parent recovery | Native cache eviction |
| Record graph peak attribution | Peak precedes formatting; current bytes stay flat | Larger consumers retain their own capacity gates | Exact allocation owner or universal harmlessness |
| Keep main unmerged | DZ5, reporting/precision and terminal F01--F20 remain open | Execute the next linked consumer unit | Whole-program completion |

Local skeptical result review: the observer is a separate process and can
only explain the scale of small reporting growth. HLO/reference checks occur
after memory measurement. Weak references answer Python ownership only.
The GDB wrapper perturbs execution and cannot replace timing measurements.
The weakest attribution is the exact native allocation owner; direct native
allocation evidence could revise that explanation without changing the
observed retention. No independent reviewer is claimed.

Next: [actual DZ5 adapter and process lifetime](filter_gradient_dz5_initializer_adapter_20260928.md).
Analysis: `posterior-initializer-capacity-cpu-gpu-04548.json`. Archive and verification receipt:
`artifacts/filter-gradient-repair-20260917/posterior-initializer-capacity-04550-evidence.tar.gz`
and `posterior-initializer-capacity-04550-verification.json`.

Final qualification: analyzer/runner 04549 passes 35 checks; policy 04550
passes 152. Ruff passes the new capacity files and runner. The campaign test
file retains two unrelated existing C408 findings; focused E9/F/I passes.
Whitespace checks pass. Reopened archive verifies all 145 members, 10,565,782
bytes, SHA-256 `765ac35e862eecd5dc1878212088fe6f3c69886344f74c18b7bf1035f3bdade7`.
CPU-only and combined analyses were independently reproduced exactly.

The unit charged 1,361.772949 CPU and 2,517.999363 GPU seconds across 23
workers, 3,879.772313 of 7,200 combined seconds. Global charged totals are
95,591.738717 CPU and 90,517.751450 GPU seconds; 29.446739 CPU and 26.856180
GPU hours remain under the unchanged 56/52-hour caps. The added 24 CPU hours
are already included. The next adapter unit has not started.
