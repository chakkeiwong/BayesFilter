# Host-memory observation limits

Recovery through 01654: no numerical worker is active. The 441 focused
qualification cases remain passing. Default XLA before/after and standalone/
enclosed records pass at D=3 and D=5. The eight-arm comparison is preserved in
`structured-memory-comparison-01650.json`; D=5 graph/XLA fails 23 numeric fields
at the unchanged 1e-10 tolerance and is **not accepted**. 01651 crosses prepared
clouds with both fitter modes; 01652 proves the same eight numeric fitter gaps
on identical data in frozen f06fd505 and current compact source. Same-mode
before/after records pass. No tolerance, optimizer or method is changed.

The D=5 resource trigger is explained by two 33-branch compact-shape dispatches
(CPQR initializer and Jacobian QR), versus two five-branch dispatches at D=3.
Fixed-input public host overhead is 446.820 MiB, GPU peak 97,024/195,840 bytes,
and cold 13.883/30.541 seconds. Over six changing active sizes, 01653/01654
retain all records while old/new first-sequence times are 70.215/29.857 seconds
and host peaks roughly 2.79/1.97 GiB. New sizes cost about 11 seconds each in
original compact fitting and about 0.11 seconds in the repaired fixed-capacity
path after the first call. Second-sequence host growth stays below 0.1 MiB.
`structured-eligibility-cost-disposition-01654.json` and its exact analysis
script preserve inputs, source hashes, HLO branch counts and full comparisons.
This is a bounded, explained compilation tradeoff, subject to final repeats;
it does not establish a statistical ranking or arbitrary-capacity memory bound.

Budget through 01654: GPU 30,927.589910224 / 187,200 and CPU
39,575.574552332 / 115,200 seconds (remaining GPU 156,272.410089776 and CPU
75,624.425447668). The extension is counted once. Commit/push the execution
checkpoint with the graph/XLA blocker explicit; continue independent proposal/
outer-controller repair without calling the checkpoint terminal-qualified.
Main remains unmerged. No F01--F20 terminal disposition is closed.

Earlier checkpoint details follow.

September 21 structured preparation: resource lifetime has a separate graph
retention mechanism. TensorFlow 2.19.1's
`python/ops/custom_gradient.py:487--524` registers a process-level closure that
retains the custom operation's graph tensors. Defining the affine preparation
pullback directly in the consuming fit graph kept that graph and its covariance
guard alive after cache eviction/collection (01626/01627). Tracing the small,
resource-free affine operation in its own stable function before caller tracing
repairs the tested ownership boundary (01628; expanded CPU/GPU checks in
01636/01638). Exact preparation derivatives, geometry-only frozen derivatives,
post-eviction execution and subsequent consumer/guard release all pass. This
addresses those Python graph/resource references, not TensorFlow's distinct
native executable cache. Cost measurements for the structured integration are
pending at this checkpoint.

September 21 guard-cost results (01608--01615): matching healthy inputs and
full optimizer settings yield exactly equal original before/after fields in
graph-reference and XLA. XLA host peak increases are 7,630,848 and 6,250,496
bytes at dimensions three and five; GPU peak increases are 3,328 and 2,560
bytes. Graph host increases are 12,185,600 and 819,200 bytes. No declared guard
cost trigger fires. Twenty warm calls have only a few KiB host growth and small
returning device fluctuations, including D=5 candidate peak 102,912 returning
to 101,888 bytes. One trace holds in all arms. This is finite observation,
not a general leak proof or terminal repeated comparison. See
`factor-guard-cost-comparison-01615.json` and its preserved analysis script.

September 21 continuation: executable mapping count is a separate resource
from host bytes. Combined CPU qualification fails in LLVM's mapped executable
section allocator (01597). The external observer in 01598 records 65,021 maps
just before failure against vm.max_map_count=65,530, despite about 194.8 GiB
available RAM, unlimited address/data/RSS limits and no cgroup memory cap/OOM.
Its one-second samples do not capture the instantaneous limit. CPU test-module
cache clearing in 01599 frees no mappings at 33,801 and 64,785 and the run fails
again. TensorFlow 2.19.1's installed
`include/tensorflow/compiler/jit/device_compilation_cache.h:62` states that its
device cache owns compiled executables and has no eviction policy. Bounded
Python factory caches and successful FuncGraph/resource garbage collection
therefore do not imply bounded native compiler retention.

The preserved analysis is
`artifacts/filter-gradient-repair-20260917/factor-executable-allocation-disposition-01599.json`
under the main checkout's artifact root, with header/evidence checksums and
parent-owned samples that survive native failure. Qualification uses all 38
cases in three bounded fresh CPU reference processes and the unchanged combined
GPU/XLA process, plus lifetime and consumer gates. This does not repair or
admit indefinite CPU compilation across distinct signatures, change OS limits,
or turn the failed cleanup into a passing result. Fixed-signature reuse and
changed-data HLO checks remain necessary for numerical runtime.

September 20 continuation: one TensorFlow trace and stable repeated-input warm
memory do not establish bounded XLA compilation. The factor/COD diagnostic
`factor-compilation-memory-comparison-01514.json` records only two runtime HLO
inputs where six were expected: training tensors became compiler constants.
Four changed same-shape clouds added 335,155,200 host bytes and each took about
6.4 seconds despite one trace. Replacing data-dependent scalar slices with
gathers retains all six inputs; the same sequence adds 61,440 bytes and each
new cloud takes about 10 ms. Graph-reference modes stay near 0.2 MiB growth.
Final compiler/memory checks must include changed numerical inputs and explicit
runtime-input HLO inspection for this path. This bounded diagnostic does not
close complete-fitter numerical gates or establish a general memory limit.

The campaign reports the maximum sampled `/proc/self/status` `VmHWM`, plus
per-phase `VmRSS` and TensorFlow allocator current/peak bytes. These are separate
metrics. The sampled host high-water field is approximate on this host; it is
not an exact allocation trace or a guaranteed monotone process peak.

Review of 444 measurement artifacts through run 00823 found four decreases
from the prepared to traced snapshot, all in the new seeded-initializer
candidate. The decreases were 9,945,088, 9,879,552, 9,805,824 and 9,728,000 bytes
in runs 00816, 00818, 00821 and 00823, respectively. Both snapshots in each
pair reported `VmHWM == VmRSS`.

A routine CPU-only standard-library diagnostic on September 18 allocated and
released a 64 MiB bytearray in a fresh `/usr/bin/python3` process. It imported
only gc, json, pathlib and resource; it did not import TensorFlow or access a GPU.
The diagnostic was executed in the isolated campaign checkout as `python3 -c`
with this body:

```python
import gc
import json
import pathlib
import resource

out = []

def sample(stage):
    fields = {
        line.split()[0][:-1]: int(line.split()[1]) * 1024
        for line in pathlib.Path("/proc/self/status").read_text().splitlines()
        if line.startswith(("VmHWM:", "VmRSS:"))
    }
    fields.update(stage=stage,
        ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    out.append(fields)

sample("start")
block = bytearray(64 * 1024 * 1024)
sample("64MiB_live")
del block
gc.collect()
sample("64MiB_released")
print(json.dumps(out, indent=2))
```

| Stage | VmHWM bytes | VmRSS bytes | ru_maxrss bytes |
|---|---:|---:|---:|
| Start | 10,772,480 | 10,772,480 | 8,388,608 |
| 64 MiB live | 77,959,168 | 77,959,168 | 75,497,472 |
| 64 MiB released | 75,497,472 | 10,846,208 | 75,497,472 |

The 2,461,696-byte decrease in this independent process demonstrates that
TensorFlow/XLA is not necessary for the reporting anomaly. This diagnostic does
not identify the kernel accounting mechanism or prove an exact physical peak.
Use maximum sampled values consistently in comparisons and avoid interpreting
small differences as allocator behavior. Existing artifacts remain unmodified.
TensorFlow allocator growth, driver reservation, and host RSS are distinct;
none of these observations establishes or rules out a leak in an unmeasured
workload. The campaign's allocation-stability and 256 MiB regression gates are
unchanged.

| Decision | Criterion | Veto | Uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain approximate host telemetry | Same metric in both arms | No gate relaxed | Exact process peak unavailable | Check phase maxima and live allocation separately | No exact host-peak or XLA leak conclusion |
