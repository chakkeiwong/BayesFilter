# Qualification of the approved numerical contracts

The owner approved both bounded proposals on 2026-09-21. The factor comparison
allows `atol=rtol=1e-8` only for the named iterative-fit geometry and error fields;
all other numerical fields remain at `1e-10`, with exact discrete values,
schemas and shapes. The dense condition is infinite under the existing
deficient-rank check. No optimizer, seed, rank threshold or consumer decision
rule changes. The original `3582b4ac` dependency closure remains the authority;
the original trust solver uses graph mode because its XLA errors are known.

## Evidence contract and review

Require comparator mutation tests, independent analytic geometry, same-state
objective/gradient checks, complete D3/D5 original lifecycles and consumers,
and dense rank-boundary checks on CPU and GPU. Keep original raw condition
ratios and the previous strict factor discrepancies in numbered artifacts.
Before enabling uniform XLA, measure original/graph/XLA dense kernels in fresh
processes at both dimensions, with identical initial and changed inputs. Inspect
cold trace/compile time, warm time, host RSS/VmHWM and GPU allocator current/peak;
retain the existing >20% warm-time and >256 MiB host-memory investigation
triggers. Dependency results do not establish complete controller costs.

Pre-run skeptical review: a broad recursive tolerance could hide errors in
scores, decisions or thresholds. The comparator therefore recognizes the exact
factor schema and named fields, and mutation tests challenge those boundaries.
Condition infinity describes numerical rank under the unchanged policy; it
must not be interpreted as a proof of exact algebraic singularity. Complete
consumer comparisons check rejection semantics independently. Frozen clouds
isolate this correction; separate RNG tests remain required. Tiny analytic
fixtures cannot establish accuracy on every model. This is primary-agent review.

Use the approved driver prefix with `test --group GROUP --device CPU` or
`--device GPU --test-gpu-index 3`, and the registered 300-second ceiling per
focused worker. Cost groups also use fresh workers and the existing campaign
budget: 32 CPU and 52 GPU process-hours. No concurrent campaign workers; freeze
runtime/tests/driver during execution. Source hashes, exact commands, environment,
device/memory policy and elapsed time are recorded by the driver under
`docs/plans/artifacts/filter-gradient-repair-20260917/run-NNNNN/`.

## Results

| Run | Scope | Result |
| --- | --- | --- |
| 02106 | CPU comparator mutation tests | Passed |
| 02107 | CPU original/current prepared factor fits and analytic geometry | Both cases passed |
| 02108 | CPU dense rank boundaries, posterior consumer and uniform public records | All 28 cases passed |
| 02109 | CPU D3 complete original lifecycle, changed inputs and resource lifetime | Passed |
| 02110 | CPU D5 complete original lifecycle, changed inputs and resource lifetime | Passed |
| 02111--02112 | CPU original/current-data same-state objective and gradient trajectories | Both passed at unchanged 1e-10 |
| 02113--02122 | CPU complete original lifecycles, refinement, terminal and fixed-center fitting | All 22 cases passed; fixed-center comparator remains strict |
| 02123 | Policy/controller checks | All 72 passed |
| 02124--02125 | GPU3 factor analytic accuracy and dense-condition/consumer qualification | 2 + 28 passed |
| 02126 | GPU3 complete quadratic numerical suite | All 50 passed, including the formerly failing singular designs |
| 02127 | GPU3 complete D3/D5 original lifecycle/changed-input/resource checks | Both passed |
| 02128--02133 | GPU3 complete original lifecycle, refinement and terminal records | All 21 cases passed |
| 02134 | GPU3 independent factor geometry | All 12 passed |
| 02135 | GPU3 fixed-fit, factor, initializer, selection, stability, I/O and posterior consumers | All 223 passed |

The first GPU3 attempt stopped before launching a worker: six idle samples each
reported 409 MiB allocated and zero utilization. An unrelated MacroFinance GPU
process (PID 3689573) held contexts on GPUs 2 and 3. No process was interrupted,
and no idle threshold was changed. CPU qualification continued. Recheck before
the next GPU launch; no GPU evidence is inferred from the CPU results.

The unrelated process later released both GPUs to 18 MiB. The matrix resumed
with fresh idle preflights; GPU3 memory growth is verified in each worker log.
Source and test files remain frozen throughout the matrix. Focused runtime/test
Ruff checks and whitespace checks pass. A broader driver lint inspection found
five unchanged warnings (two import-order, two C408 and a duplicate identical
GPU-device mapping); `git show HEAD` reproduces all five. They are not new
failures from this implementation and no repository-wide lint pass is claimed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Qualify the approved correctness changes | All 495 CPU/GPU cases pass | No numerical or consumer veto; historical failures preserved | Dense costs and enclosing/public integrations | Finish fresh-process costs, then qualify uniform controller | No terminal acceptance or merge readiness |

No timing ranking is supported by these correctness runs. Public uniform and
sequential integration, broader controllers, external deadlines, actual DZ5
transitions and final F01--F20 dispositions remain open.

Correctness checkpoint through 02135: all **495** cases pass (157 CPU, 338 GPU).
The approved contracts now pass their declared initial CPU/GPU qualification,
including independent analytic accuracy and complete original records. The
fixed-center wrapper uses a different result schema; its original comparison
remains at the strict `1e-10` gate and passes 02122 without widening the allowance.
Fresh original/graph/XLA dense dependency costs follow before public uniform
integration. Historical discrepancies remain preserved; this checkpoint does
not close any F01--F20 terminal finding or establish whole-program readiness.

The first CPU dense cost matrix (02136--02141) passes its per-worker checks.
Observed D5 warm medians are 0.592 ms original and 0.753 ms XLA; this crosses
the existing 20% investigation trigger. Run two additional fresh processes per
CPU arm/dimension under the same frozen source and compare all initial/changed
fields. Keep all first-run observations. The repeated-process median and range
are descriptive diagnostics; they cannot establish a statistical speed ranking.
GPU costs and the complete cross-arm record comparisons are still pending.

The first cross-arm comparison, through 02147, matches every initial and changed
field at `1e-10`. GPU3 D5 also crosses the warm-time trigger (2.440 ms original,
3.379 ms XLA), so perform the same two fresh repeats on GPU3. No host-memory or
device-memory trigger fires: CPU XLA adds about 189 MiB; GPU XLA observed host
RSS is about 38 MiB lower and allocator peaks are 24,064/35,328 bytes versus
roughly 8.4 MB originally. Preserve all cold costs and raw samples. The analysis
is `dense-numerics-cost-comparison-02147.json` under the campaign artifact root.

All 36 fresh-process cost workers (02136--02171) pass, and every initial/changed
field matches the original at `1e-10` with one trace. Three process medians:

| Device/dimension | Original warm ms | Graph warm ms | XLA warm ms | Investigation |
| --- | ---: | ---: | ---: | --- |
| CPU D3 | 0.564 | 1.380 | 0.608 | No trigger |
| CPU D5 | 0.592 | 1.940 | 0.753 | XLA +27.1% versus original |
| GPU3 D3 | 2.110 | 6.262 | 1.481 | No trigger |
| GPU3 D5 | 2.440 | 8.640 | 3.363 | XLA +37.8% versus original |

The D5 warm-time finding persists across repeats. Do not declare the cost gate
closed or silently accept the regression. Preserve the full analysis in
`dense-numerics-repeats-02171.json`. Next localize the solver/SVD/eigenvalue costs
and measure their enclosing-controller impact. A native diagnostic controller
can be developed under the authorized repair plan while public uniform XLA
integration remains blocked. Investigate spectral boundary behavior first,
given the earlier compiler eigensystem defects in adjacent dependencies.
The original uses its specialized COD op; the candidate has native TensorFlow
CPQR/COD. This is a plausible cost explanation, not measured attribution yet.

No memory trigger fires in this matrix. CPU XLA observed peak RSS is about
189 MiB above the original; GPU XLA host peak is about 38 MiB lower. GPU live
warm allocation is constant at each extent and warm host changes over twenty
calls are small. This is limited scope allocation evidence, not leak freedom.
Cold compilation remains an explicit cost. No statistical ranking is claimed.
