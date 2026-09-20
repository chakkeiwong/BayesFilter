# Host-memory observation limits

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
