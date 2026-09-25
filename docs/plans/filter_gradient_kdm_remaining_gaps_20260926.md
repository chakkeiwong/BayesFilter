# KDM remaining numerical and memory gates

The native auxiliary and shared reset have passed their bounded FP64 CPU/GPU
execution checks, and matching-mode FP32 reset parity passes. The master program
remains open. This follow-up resolves two findings preserved by the completed
KDM execution unit; it does not reopen a canonical LEDH rebuild or change a
numerical acceptance threshold.

The first question is why original and repaired FP32 XLA reset outputs differ
from the original eager computation beyond `atol=rtol=1e-6`. Run 03922 shows
bitwise original/native XLA equality, so the loop conversion is not the cause.
Use the committed seed-131 operand fixture and first compute an independent
FP64 reference from the *rounded FP32 inputs*. Compare original eager, original
graph/XLA and repaired graph/XLA outputs and tangents to that reference. Record
condition numbers, scaled residuals and roundoff bounds for the three reset
factorizations. Do not infer severe ill-conditioning from one small parity
miss. If attribution needs intermediate tensors, require the instrumented
program to reproduce the original failure before using its records as a witness;
exposing intermediates can change compiler fusion.

The unchanged cross-mode test is a promotion veto and repair trigger, not a
continuation veto. A valid numerical repair must preserve accepted inputs,
complete outputs and derivatives and repeat CPU/GPU finite-difference and
reset tests. Truly unresolved numerical inputs must report an error; no
threshold relaxation, silent ridge change or replacement score is allowed.

The second question is whether repeated public-owner construction retains
unbounded host resources. Single-owner runs 03947/03948/03951/03952 have one trace
and stable memory over 20 exact replays; they do not test growing owner counts.
The graph survived the first collection in those isolated workers, but the
combined trace-only check 03953 released its graph. Direct reference inspection
03950 found graph tensors/operations and no gradient-registry retaining edge.
These observations do not prove a permanent leak. Use fresh processes with
fixed counts 1/3/6, completed output release, repeated garbage collections,
registered-function counts, Python graph weak references, current RSS and
allocator statistics. Compare public reconstruction with retained-owner calls.
If growth is reproducible, identify a retaining dependency before changing
ownership. Never replace mutable callback semantics with an unreviewed global
cache. Native executable/allocator eviction is a separate property from Python
graph collection.

These are execution and resource diagnostics only. They cannot establish LEDH
admission, HMC readiness, posterior quality or target-scale performance. Retain
the existing FP64 matched cost cohort; a source/lifecycle change requires a new
versioned comparison before its costs can replace that cohort. The tiny KDM
fixture cannot close the master's target-capacity/compiler-memory findings.

Budget: at most 12 CPU workers / 2400 process-seconds and 6 GPU workers / 1200
process-seconds within the existing 56/52-hour global caps. Use the stable runner
with explicit device and unique campaign run directories. Preserve failures;
freeze runtime/scripts/tests during each numerical worker. Stop integration on
unexplained numerical changes, missing evidence, source drift or exhausted
budgets; proceed with diagnosis when only the candidate fails.

Skeptical review: unchanged XLA outputs do not resolve the cross-mode question,
and one stable retained owner does not bound repeated construction. Rounded-input
FP64 is essential for a fair precision comparison. The original eager result
is a frozen implementation comparator, not an exact mathematical oracle. Those
distinctions prevent a false numerical repair or a false memory-leak claim.

03957/03958 execute the first diagnostic. CPU XLA is closer to the FP64 reference
than eager (0.321 versus 1.025 unchanged tolerance units), while GPU XLA is
0.062 units versus approximately 181 for GPU eager/graph. Original and repaired
XLA are identical. Factor condition numbers are approximately 60--153 on CPU;
these do not support a severe-ill-conditioning classification. GPU results
require a fresh explicit FP32-no-TF32 reference arm to distinguish TensorFloat32
arithmetic from algorithm or reset-factor effects. Keep both failures and the
same numerical tolerance; do not change the production TF32 setting. Then check
the rounded-input FP64 primal with an independent NumPy diagnostic implementation
and directional finite differences before considering any comparator correction.

The first follow-up is complete through 03972; see
[the precision/memory result](filter_gradient_kdm_precision_memory_result_20260926.md).
Use the final two CPU workers in this unit for an explicit Linux allocator
diagnostic: repeat the same six constructions in graph/XLA mode, call glibc
`malloc_trim(0)` only after output release and garbage collection, and compare
before/after RSS with the retained function/graph/device counts. This test-only
operation can distinguish free host-allocator pages from still-live native
compiler allocations. It must not be installed in runtime as a speculative fix.
If trimming does not remove the slope, the next bounded unit must attribute
native compilation ownership before any cache/lifetime implementation change.

03973 confirms that allocator trimming returns roughly 140--160 MiB but leaves
the construction slope: post-trim RSS grows from 1069 to 2428 MiB over six XLA
builds. The next unit is therefore a native ownership repair, with at most
8 CPU workers / 1800 seconds and 4 GPU workers / 900 seconds inside the same
global caps. Record glibc allocated/free/mapped bytes (`mallinfo2`) and process
mapping categories across 1/3/6 constructions. Test enclosing compilation with
the two mixture functions compiled only by their owner, while the standalone
mixture API retains its own XLA default. Compare optimized HLO, complete outputs,
one-trace replay and memory slope before accepting any change; this tests whether
redundant nested compiler ownership contributes to the retained host allocation.

If nested ownership does not explain the slope, preserve the rejection and test
one bounded process lifetime per changing callback/configuration, with explicit
retained owners within that process. Do not introduce hidden callback caches,
global context resets, allocator-trimming side effects or eager fallback into
the public numerical API. There are currently no non-test repository consumers
of this KDM auxiliary; callers needing repeated evaluation already have the
explicit stable-signature factory. This bounds an authorized engineering repair
without pretending the general constructor lifetime is fixed.

The full-call-chain audit must still include `ledh_numerical_safety_tf` (including
its static rank-expansion loop and squeezed validity shape); guarding the outer
KDM module is not whole-repository compliance. The current work preserves that
helper's shape semantics. No failing native allocation or numerical gate is
silently converted into a passing terminal disposition.
