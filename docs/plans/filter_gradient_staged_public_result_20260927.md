# Public staged locator integration result

The public `locate_joint_center_staged` now uses the shared native checkpoint
and continuation programs. All 24 distinct correctness cases pass on both CPU
and GPU, and all twelve CPU/GPU cost arms pass. This is repair-branch evidence;
main-branch and whole-campaign acceptance remain open.

The default retains two XLA stages separated by one external checkpoint
validator. Starts and scales are explicit operands; a single owner is reused
for the same callback, static configuration and device. Bound methods compare
their receiver and function identities, so repeated attribute access does not
rebuild the program. Different receivers or configurations cannot share the
owner. The host-clock diagnostic remains explicitly non-JIT. Real compiler or
device errors propagate without an eager retry or a fabricated completed
optimizer record; this deliberate error-boundary change is in the
[plan](filter_gradient_staged_public_integration_20260927.md).

CPU 04266--04270 checks complete original `3582b4ac` records and exact callback
order on original, changed and replayed D1/D3 inputs, including nonlinear,
constant, invalid, capped, rejected and validator-exception cases. All seven
existing staged consumer assertions execute through the actual public function.
04272 checks actual XLA compilation failures at both stages and nested
validation. 04271's numerical records matched, but its final trace assertion
inspected an unused internal owner; that failed harness attempt is preserved.
04279 adds bound-method/configuration isolation and renews public validation,
Python owner collection, frozen derivatives, graph labels and wall diagnostics.
Checkpoint and continuation each trace once and retain unchanged HLO for
changed input values. 04280 passes all 129 policy checks; coverage remains
270 guarded sources and 1,424 exact allowances. The existing host-clock
allowance moved to its explicitly named diagnostic function, and public-loop
coverage expanded without adding an allowance.

The six fresh CPU cost workers 04273--04278 preserve all original records
within each execution mode. The analyzer validates source, configuration,
inputs, original dependency hashes, CPU visibility and complete results.
Original graph/XLA selection differences remain in the output (two D1 and six
D3 field differences); they are not treated as equivalent. Graph/XLA timing
therefore has no equal-result ranking.

| Dimension | Arm | Build + cold s | Median warm ms | RSS after warm calls MiB | RSS growth over 3 warm calls MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Original public XLA | 6.6513 | 6212.647 | 2341.859 | 1139.926 |
| 1 | Repaired public graph | 5.7435 | 7.290 | 829.398 | 0.082 |
| 1 | Repaired public XLA | 6.5288 | 3.867 | 1183.586 | 0.008 |
| 3 | Original public XLA | 5.9321 | 5792.430 | 2360.047 | 1138.980 |
| 3 | Repaired public graph | 5.4397 | 12.563 | 830.516 | 0.023 |
| 3 | Repaired public XLA | 6.0578 | 3.805 | 1197.387 | 0.027 |

Reuse removes the original's per-call stage construction and compilation.
These timings include public result materialization and the external validator;
HLO inspection follows the timed memory observations. Compared with the original,
no declared cold/warm/RSS trigger fires at either extent. XLA retains about
354--367 MiB more observed host RSS than graph mode. This is compilation/runtime
residency evidence, not a live-tensor measure or proof of a leak. Three warm
calls and Python object collection cannot establish native executable eviction,
a long-term plateau, or a global memory cap. One process per arm yields
descriptive differences, not a statistical performance ranking.

The frozen cost cohort precedes the bound-method cache extension; its function
callback behavior and numerical kernels are unchanged. 04279 independently
checks the new method-identity branch. Both source states are archived. The
static production-call audit found no further internal consumers beyond the
public export. The inspected MacroFinance consumer explicitly requests a
non-JIT host deadline; it remains a diagnostic route and its actual target
is not qualified by these API fixtures.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Retain repair on isolated branch | CPU/GPU complete records, calls, ownership and errors pass | Actual targets and terminal review remain | Continue single-locator public integration | Main or whole-campaign readiness |
| Retain descriptive CPU cost result | Same-mode original records and provenance pass | Short observations, cross-mode selection differences, bound-method arm tested separately | Preserve raw results and qualify GPU/repeated terminal costs | Native eviction or general speed guarantee |
| Keep wider findings open | Policy subset passes | Actual target, other public controllers and F01--F20 terminal review remain | Continue master program | Whole-repo compliance, HMC, posterior or canonical LEDH correctness |

GPU correctness 04281--04286 passes the same 24 cases. The fresh six-arm GPU
cost cohort 04287--04292 uses physical GPU 2, UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, after the selector admitted it.
The analyzer verifies identical source/environment/input/reference identities,
verified memory growth, device sharing observations, complete original records,
single traces and unchanged HLO. No cross-mode record discrepancy occurs in
this GPU fixture. CPU cross-mode discrepancies above remain unresolved.

| Dimension | Arm | Build + cold s | Median warm ms | RSS after warm calls MiB | RSS growth over 3 warm calls MiB | Observed allocator peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Original public XLA | 8.7564 | 8274.493 | 2593.648 | 1100.852 | 81152 |
| 1 | Repaired public graph | 9.2181 | 16.980 | 1267.258 | 0.059 | 119040 |
| 1 | Repaired public XLA | 8.7437 | 6.025 | 1398.332 | 0.027 | 65024 |
| 3 | Original public XLA | 8.9305 | 8462.971 | 2599.668 | 1103.969 | 82432 |
| 3 | Repaired public graph | 9.1999 | 52.354 | 1277.668 | 0.047 | 133632 |
| 3 | Repaired public XLA | 8.9444 | 7.272 | 1401.238 | 0.008 | 66560 |

No declared GPU cold/warm/RSS/allocator trigger fires. GPU XLA retains about
124--131 MiB more host RSS than graph. The strongest alternative explanation
for the large warm difference is precisely the changed ownership: the original
reconstructs its stages each call. These observations measure that repair,
not an identical-graph compiler ablation. Native eviction, long-term memory
plateaus and performance across actual targets remain unchecked. The weakest
evidence is one process per arm and three warm calls; broader cost acceptance
requires the master program's repeated terminal comparisons.

The unit used 16 CPU workers / 735.537893 seconds and 12 GPU workers /
803.906523 seconds, within its existing 7,200-second reservation. Receipts
`staged-public-cpu-verification-04278.json` and
`staged-public-cache-policy-verification-04280.json`, together with
`staged-public-gpu-verification-04292.json`, preserve every raw attempt,
the independent analyzer and source snapshots. All 74 GPU archive members
were reopened and verified against SHA-256 hashes. The global charge through
04292 is 87735.052659 CPU / 79511.376549 GPU seconds, leaving 31.629152 CPU /
29.913507 GPU hours under the 56/52-hour caps.
