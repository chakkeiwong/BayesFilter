# Fixed fitted-APF and input-owner resource acceptance

Runs05203--05227 pass the first resource-acceptance subunit:18 fresh GPU cost
workers, six CPU/GPU1024-call reuse checks and161 terminal readback/policy checks.
All three paired comparisons have exactly equal complete shared numerical
outputs and diagnostics. Fixed fitted-APF is in scope; adaptive iAPF is deferred.
Runtime algorithms, input streams, tolerances and policy allowances did not change.

| Complete callable | GPU warm before -> after (ms) | After/before warm ratio, approximate95% interval | Cold before -> after (s) | Extra warm host RSS (MiB) |
| --- | --- | --- | --- | --- |
| Fixed fitted Gaussian APF |19.470 ->8.485|0.4358 [0.4293,0.4424]|6.431 ->9.616|108.77--109.63|
| Fixed fitted nonlinear APF |19.595 ->8.584|0.4381 [0.4322,0.4440]|6.353 ->10.068|110.25--111.04|
| Gaussian twist input endpoint |9.713 ->7.107|0.7318 [0.7222,0.7415]|7.239 ->9.691|21.25--24.09|

Times are means of three fresh-process arm medians; ratios use paired log
means and a Student-t interval with two degrees of freedom. Arm order is
counterbalanced; each worker takes30 synchronized warm samples on GPU3 UUID
GPU-b8045e28-4433-ec7a-77a5-db0636748322. These small-sample intervals describe
the declared fixtures/hardware, not a universal speed distribution. Preflight
and in-run observations found no sharing; every worker verified memory growth.
Timing and primary memory samples precede HLO export/comparison compilation.

The higher cold cost and host RSS are accepted measured execution tradeoffs
for these owners. They appear at enclosing-XLA compilation/first execution;
subsequent reuse retains one trace and one cache entry. The cold overhead is
amortized after approximately290,338 and942 same-shape calls respectively at
the measured warm rates. This arithmetic does not promise those break-even
points for other workloads. The exact TensorFlow/compiler arena responsible
for native residency has not been identified; Python collection does not
release that residency and is not described as doing so.

Each current complete callable executes1024 times with two alternating
seed/theta cases. Independent original references are checked after the primary
samples at unchanged bounds. Both backends retain exact same-input replay and
one compiled owner/trace/cache entry. Late512-call RSS growth is0/0 bytes for
Gaussian inputs,8192/0 bytes for fitted Gaussian and16384/0 bytes for fitted
nonlinear (CPU/GPU). GPU allocator current bytes stay1792 for Gaussian inputs
and2816 for each fitted case. Peaks are47360/34048/34304 bytes respectively.
These live/peak allocator measurements differ from sampled GPU process
reservation, which includes context/compiler/library allocations (440MiB in
representative cost workers). The reaped workers leave no own process or GPU
context in the saved observations.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
|Accept scoped owner resource tradeoff|Exact shared outputs; lower warm cost in all pairs; stable bounded reuse|Cold/RSS triggers explained as retained cold-execution costs, not continuing observed growth|Three timing pairs and one shape per callable|Reuse this evidence if numerical dependencies remain unchanged|Universal speed/capacity or native-memory eviction|
|Keep broader resource phase open|This subunit covers three declared callables|Streaming's failed stricter CPU study remains failed|Remaining-SVD lifetime, angle/subspace guard and SQMC preparation costs|Run registered remaining units under the same budget|Whole-program completion or main merge|

The plan is filter_gradient_resource_acceptance_20260930.md. Every raw manifest
records exact commands, source hashes, environment, input fixture/seed identity,
placement, growth, wall time and process-exit observations. Artifacts are under
/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917;
run05227/resource-owners-readback.json independently recomputes comparisons and
checks source/device/numerical/reuse evidence. No worker failed in this subunit.
Its charge is55.507851 CPU and344.677191 GPU process-seconds. The verified
archive resource-owners-05203-05227-evidence.tar.gz and adjacent verification
JSON preserve every raw record and the source checkpoint in the worktree's
artifact directory.

Primary-agent review: a small tensor allocator does not explain total host or
CUDA residency, and an interval over repeated calls is not an interval over
independent processes. Separate measurements and process-paired analysis avoid
those errors. Residual risks are finite shape coverage and sampled sharing.
No general leak-freedom, optimization convergence, canonical LEDH, posterior or
HMC conclusion follows. No independent agent was used.
