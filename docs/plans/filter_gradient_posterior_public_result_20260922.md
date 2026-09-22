# Public posterior execution and repeated costs

The public posterior curvature endpoint now invokes one XLA controller and
formats its completed records. Host input validation remains explicit. There
is no eager retry for an unsupported callback. Numerical authority remains
original `3582b4ac`; `dba39e048` is the immediate public cost comparator.

Runs 02782--02787 and 02788--02793 pass 135 CPU and 135 GPU checks, respectively:
complete D1/D3/D5 records and callback logs, changing inputs, compiled public
wiring, invalid geometry with no downstream use, unsupported callbacks, dense
consumers and existing regressions. Shared GPU correctness timings are excluded
from costs. Run 02794 passes 123 policy, runner and cost-provenance checks.

All 18 CPU cost workers 02795--02812 pass, with three fresh processes per
arm/extent. Every original and changed-input result matches original3582b4ac
at unchanged tolerances. The full public call includes validation and payload
formatting; cold totals also include factory construction and tracing. The
diagnostic graph arm and XLA arm retain their declared solver differences.

| Dimension | Arm | Median cold total (s) | Median warm call (ms) | Maximum observed RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| 3 | Prior public | 0.213 | 145.368 | 587.621 |
| 3 | Graph reference | 1.828 | 9.156 | 672.520 |
| 3 | XLA public | 3.020 | 2.018 | 961.094 |
| 5 | Prior public | 0.247 | 172.020 | 587.438 |
| 5 | Graph reference | 1.806 | 10.090 | 672.094 |
| 5 | XLA public | 3.048 | 2.527 | 965.098 |

The cold >2x and extra-host-RSS >256 MiB investigation triggers fire for both
dimensions. XLA traces add about 34 MiB, followed by about 355--360 MiB during
first execution. Observed growth over twenty warm calls is at most 0.157 MiB
for XLA. This localizes most observed growth to first execution, but does not
separate compiler startup, executable storage and allocator retention or prove
leak freedom. Graph size is 2,733 nodes / about 417 kB and HLO about 1.98 MB at
both dimensions. Cold cost and host memory remain open investigations; lower
warm measurements do not waive them.

The September 22 12:21 UTC GPU matrix attempt failed its bounded device
preflight before any numerical worker launched. GPU2/GPU3 were shared by PID
2318611, usually at 99--100% load, with about 39,836 MiB free each. Available
memory means the owner's desktop fallback condition was not met. No GPU cost
comparison or reservation/live-memory inference is made.

Evidence: numbered manifests, JUnit and `posterior-public-memory.json` files in
`artifacts/filter-gradient-repair-20260917/`; analysis is
`posterior-public-costs-cpu-02812.json`. The stable runner records exact commands,
seeds, source hashes and environment. Costs use 18/36 workers in their declared
7,200-second tranche. Cumulative charges through02812 are
57,212.454669882936 CPU / 55,919.89945038133 GPU seconds under 32/52-hour caps.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Keep qualified posterior wiring | Full original/public CPU/GPU checks pass | No numerical waiver | Other endpoints are independent | Continue sequential enclosure | Not whole-repository completion |
| Preserve CPU costs | All 18 records pass | Cold/RSS triggers open | Few repeats; first-use allocation not attributed | Isolate compiler startup and longer reuse | Not a general performance ranking or memory bound |
| Defer GPU cost comparison | No valid paired run | Device contention | Clean timing window unavailable | Retry only under clean preflight | No GPU cost conclusion |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Numerical/provenance CPU checks pass; GPU cost launch vetoed |
| Statistically supported ranking | None asserted |
| Descriptive differences | Warm calls lower; cold time and host RSS higher |
| Default-readiness | Not established by these endpoint measurements |
| Next evidence | Memory attribution, unshared GPU repeats, remaining endpoints and integration |

Post-run review: the strongest alternative explanation for much of the memory
increase is process-wide first-use XLA compiler initialization. A minimal XLA
prewarm versus a fresh-process endpoint, followed by long same-signature reuse,
can distinguish that from per-endpoint executable cost. Twenty calls alone
cannot answer the retention question. The preceding CPU signature-churn crash
remains preserved and is not explained away by these small stable graphs.

## Completed GPU repeats and harness repair

All18 GPU workers02835--02852 pass on physical GPU2
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`. Each pair has the same source,
inputs and environment, two clean preflight observations and no observed
foreign compute process during measurement. Memory growth is verified before
initialization. Sampling can still miss brief sharing between observations.

| Dimension | Arm | Median cold total (s) | Median warm call (ms) | Maximum observed RSS (MiB) | Allocator peak (bytes) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | Prior public | 2.324 | 251.278 | 1126.379 | 8414464 |
| 3 | Graph reference | 5.859 | 53.215 | 1223.402 | 8444160 |
| 3 | XLA public | 7.181 | 7.272 | 1219.961 | 49920 |
| 5 | Prior public | 2.372 | 306.758 | 1122.969 | 8421120 |
| 5 | Graph reference | 5.908 | 57.804 | 1215.172 | 8458496 |
| 5 | XLA public | 7.760 | 11.756 | 1220.965 | 58368 |

Only the cold >2x trigger fires on GPU. Extra observed host RSS is93.58/98.00MiB;
XLA growth across twenty warm calls is at most0.075MiB. Allocator observations
are separate from device reservation. The result is descriptive cost evidence
for these small fixtures, not a general CPU/GPU or algorithm ranking.

Run02833 first exposed pytest conftest's default CPU scope replacing the UUID
environment after TensorFlow discovery. No cost was measured in that failed
worker. The runner now sets and records `BAYESFILTER_TEST_DEVICE_SCOPE`, checks
it before TensorFlow import and after pytest, and the cost validator requires
visible scope for GPU. An actual-conftest regression and contradictory-scope
negative cases pass in02834 with all127 policy checks. Earlier numerical GPU
fixtures observed initialized logical GPUs; the bug was post-discovery scope
metadata and cannot be used to relabel their measured GPU tensors as CPU.
Renew subsequent qualification under the corrected harness.

The CPU and GPU source hashes differ because this harness repair occurred
between cohorts. Each device's cost comparison is internally matched and has
its own analyzer output; no combined-source or cross-device ranking is claimed.
GPU analysis: `posterior-public-costs-gpu-02852.json` in the same artifact root.
The localized retry raises the count allowance from36 to38 workers without
changing the7,200-second cost ceiling;37 workers are used, including02833.

The earlier decision to defer GPU costs is superseded by these completed
repeats. Cold attribution remains open on both devices and CPU host-RSS
attribution remains open. The next recorded diagnostic uses minimal-XLA startup
and3,000 alternating-input calls. This does not close the independent sequential,
block, consumer, ownership, final audit or merge gates.


## Startup, reuse and observer attribution

02874--02881 pass all eight D3/D5 fresh/minimal-XLA-control CPU/GPU probes,
with3000 alternating complete public calls each and full original comparisons
after measurement.02882/02883 pass the observer-only controls. Analysis and
its executable source are saved as posterior-residency-analysis-02883.json and
analyze-posterior-residency-02883.py in the shared artifact root. All eight
residency workers have identical source hashes; the controls are a separate
recorded harness cohort. GPU2 and verified memory growth remain unchanged.

| Device | Minimal XLA RSS growth | Endpoint cold RSS after tracing, prewarmed | Reuse RSS growth | Observer-only RSS growth |
| --- | ---: | ---: | ---: | ---: |
| CPU | about64MiB | 293.45/297.43MiB at D3/D5 | 1.68--1.75MiB | 1.35MiB over seven intervals |
| GPU | about71MiB | 260.68/262.45MiB at D3/D5 | 1.61--1.67MiB | 1.43MiB over seven intervals |

The generic first-XLA explanation is only partial: substantial endpoint
compilation residency remains after the minimal control. Executable mapping
counts and GPU allocator current/peak stay unchanged over reuse; all eight
Python graph weakrefs release after cache clearing, while resident memory
persists. The retained mapping observer stores456 per-file/category records
per snapshot and itself grows RSS with no numerical calls. Its growth is of
the same scale as much of the reuse increase, so the latter cannot be reported
as unqualified numerical-call leakage. This is not an exact allocation census.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Preserve public posterior | All original records pass through repeated reuse | No numerical veto | Endpoint/configuration coverage remains bounded | Continue other endpoints | Not whole-repository completion |
| Retain memory investigation | Startup attribution and observer controls pass | Earlier cold/RSS triggers remain | Native compiler/executable retention is not individually identified | Coordinated ownership and signature-churn attribution in E4 | No native eviction or general leak-freedom claim |

Post-run review: observer retention is a demonstrated alternative explanation
for small RSS growth, but cannot explain the substantial cold compilation
residency. No timing ranking follows from one process per condition. The known
many-signature LLVM mapping failure remains preserved and is not waived by
fixed-signature reuse. The ten-worker diagnostic tranche is complete.
