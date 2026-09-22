# GPU selection and remaining repair work

Current checkpoint: [remaining-gap diagnosis and repair roadmap](filter_gradient_remaining_gap_diagnosis_20260922.md), through02636. It supersedes earlier current-state claims below; source guard coverage and main merge remain incomplete.

Owner instruction on September 22 supersedes the GPU3-only campaign restriction:
use an available GPU that does not serve the remote desktop; use a desktop GPU
only when every other GPU is above 50% utilization and lacks enough free memory.
This changes device allocation, not the numerical target, tolerances, budget,
single-worker rule or verified TensorFlow memory growth.

The current machine has four RTX 4090 devices. GPU1 drives the display, while
GPU0 runs `gnome-remote-desktop-daemon` (encoding); GPUs2/3 have only incidental
4 MiB Xorg contexts and are idle. Protect both actual desktop roles. Incidental
Xorg contexts alone do not classify all GPUs as desktop devices. Re-discover
display state and remote-desktop compute processes at each preflight.

Select a non-desktop device with utilization <=50% and >=8 GiB free, preferring
lower utilization, then more free memory, then lower index. The 8 GiB launch
headroom is a conservative scheduling floor for bounded fixtures, not a device
allocation limit or universal fit guarantee. The former <=100 MiB used-memory
restriction is retired. Require two consecutive admissible samples for the same
UUID, within six samples two seconds apart. Permit desktop fallback only if
every non-desktop GPU is both >50% busy and below that free-memory floor; the
chosen desktop GPU must itself have enough headroom and <=50% utilization.
Record the full selection, free memory, utilization, desktop reasons and UUID.
Pass the UUID to `CUDA_VISIBLE_DEVICES` so CUDA enumeration cannot silently
redirect the chosen NVIDIA index to a different physical device.

Automatic selection defaults on; explicit indices 0--3 remain validated by the
same rule. Pin one physical GPU across each matrix and matched cost group.
Recheck before every worker; never reuse an arm/repeat from another GPU. On
contention a matrix can resume on another eligible device with fresh matched
arms, not mixed hardware. Correctness checks may share a <=50% GPU, but shared
or desktop preflights do not establish clean terminal performance comparisons.
The terminal comparison excludes these records. Matrix resume requires recorded
matching UUIDs; earlier index-only evidence stays archived rather than silently
acquiring a physical identity. This preflight is a scheduling observation, not
a guarantee that no other workload starts during execution.
Record every compute process, including an idle process: clean timing preflight
requires no other compute process and <=5% utilization in both final samples.
Repeats must rerun previously shared arms rather than reuse their timings.
The stable approved campaign command prefix and total 32 CPU / 52 GPU-hour caps
remain unchanged. No process, display setting, package or environment is changed.

Review: selecting a display GPU from `display_active` alone misses the remote
encoder; selecting every Xorg device would exclude all four. UUID/process
metadata resolves both. Same-device filtering is also needed for pytest-based
cost matrices, which previously filtered only device class. Test both ordinary
selection and the exact conjunctive fallback condition before GPU qualification.
Use a CPU policy check, then the newly repaired preparation on an eligible GPU,
then the pending proposal/pilot/fit and shared-solver qualification. Preserve
failure artifacts and stop affected numerical work on disagreement.

The remaining gaps, independent of scheduling, are:

| Gap | Required work before completion |
| --- | --- |
| Complete numerical execution boundaries | Connect preparation, pilot, fit and exact replay into the full geometry initializer and iterative control; handle dynamic retained rows, batch extents and permutation generation without padded calls or host numerical decisions. Finish sequential outer lifecycle/reporting and ordered block-coordinate control. |
| GPU qualification | Preparation, proposal/callback, fit and pilot GPU checks now pass. Shared COD derivative/factor/lifecycle/consumer renewal also passes. Complete matching uniform public and new-controller GPU costs; update post-run analyzers to enforce the selected UUID instead of hardcoding GPU3. Source-frozen terminal evidence remains required. |
| Posterior public integration | Rejected D3 comparison resolved by the September 22 owner decision: check rejection, reporting and no downstream use; preserve discarded matrix differences as explanatory evidence. 150 CPU/GPU posterior checks pass in 02622--02625. The uninstalled allowance and unfinished GPU reference are superseded. Public endpoint integration and full endpoint qualification remain open. |
| Cost investigations | First-execution allocation and nested callback-cache retention are now localized in02629--02634; native cache versus allocator attribution remains open. Explain and disposition cold compilation, roughly 300--525 MiB extra host RSS in some enclosing controllers, standalone preparation/report overhead, and outstanding dense/mass/selector timing triggers. Bounded 3,000-call stability is not general leak freedom or compiler-cache eviction. |
| Actual consumers | Validate actual DZ5 targets/transitions and repair independent parent deadlines before buffering callback progress. Current credit target uses rectangular SR-UKF with target-only qualification; geometry callbacks still materialize into NumPy. No external source pin change counts as compatibility evidence. |
| Audit and terminal evidence | Expand the partial exact-source guard (213 sources, 1,306 exact exemptions at 02595), verify active consumer call chains and classify every F01--F20 finding. Many repairs exist, but none has a final closed disposition. Freeze source, run all required suites and matched two-extent/three-process comparisons. |
| Integration | Integrate refreshed remote main 01d67ec410, resolve any conflicts, recheck affected code and perform terminal review. Merge only after full qualification. |

Canonical LEDH algorithm rebuilding remains outside this execution-repair
campaign by the owner's earlier decision. Unsupported canonical claims remain
blocked; compiled diagnostic scores cannot replace analytical recursive scores.

The scheduler and policy suite passes 98 checks in both 02596 and 02597.
02598 automatically selected GPU2 by UUID with memory growth verified. It
passed 37 preparation checks and failed 14 direction/product checks before
their XLA arm: Grappler rewrites the three-term uint64 high-word sum to AddN,
which has no ordinary GPU uint64 kernel. Preserve that failed artifact.
The high word of a 53-bit significand squared is below 2**42; add its three
terms as int64 and cast the result back to uint64. This changes neither product
bits nor rounding and requires no optimizer disablement or numerical tolerance
change. The existing 2,092 exact integer-reference products and full preparation
suite cover the repair; renew CPU and GPU checks before proceeding.

02599 CPU and 02600 GPU pass all 51 preparation checks after that repair.
02601 passes 48 GPU proposal/control checks, 02603 passes 21 GPU callback
checks, 02604--02607 pass 40 GPU fit checks, and 02608--02612 pass 57 GPU pilot
checks: 217 GPU checks in total, all on the recorded GPU2 UUID. These include
original complete records, boundaries, changed inputs and resource ownership.
They do not establish whole public initializer compilation. Final scheduler
checks in 02602 pass 102 cases; no numerical policy exemption was added.

Focused Ruff passes for the new scheduler/test and repaired numerical module;
critical-error Ruff and whitespace pass for all touched code. The broader
legacy driver/test files retain pre-existing style warnings and a repeated
identical device-map key; these are not represented as a full lint pass.

02613--02619 renew all 311 shared-solver GPU checks: 34 solver/active-row,
28 conditioning, 14 boundary, eight derivative, two factor, two complete original
lifecycle and 223 downstream consumer checks. The unchanged current-source
102-case policy result is reused by the matrix. This continuation passes 528
GPU checks in total, without skipped or failed final cases. Audit 02620 inventories
2,988 working Python files / 2,987 parsed / one unchanged vendor-reference error.
The guard passes at 213 sources / 1,306 exact exceptions and remains partial.

Charges through 02620 are 52,280.32015160784 CPU / 49,321.5887504554 GPU seconds;
about 17.48 CPU / 38.30 GPU process-hours remain. No worker is active. Structured
checkpoint: `artifacts/filter-gradient-repair-20260917/gpu-selection-qualification-02620.json`.
Exact commands, source hashes, environment, memory growth and per-run times
remain in the numbered run manifests/logs. Remote main is now 2c2419c0.

| Decision | Primary criterion | Vetoes / uncertainty | Next action | Unsupported conclusion |
| --- | --- | --- | --- | --- |
| Retain automatic GPU scheduling | 102 policy/controller tests pass; all workers selected recorded GPU2 UUID | Two-sample preflight cannot guarantee later exclusive device use | Keep rechecking each worker and pin matched comparisons | No unrestricted desktop use or hard allocation cap |
| Retain GPU-qualified internal components | 528 GPU checks and 51 renewed CPU preparation checks pass; failed 02598 preserved and repaired exactly | Whole public enclosure and matched costs remain open; the rejected posterior comparison is resolved by subsequent 02622--02625 validation | Enclose complete initializer/control; renew GPU costs | No full-repair, public readiness, HMC or canonical LEDH claim |
| Keep main merge blocked | Syntax/partial source guard passes | F01--F20 terminal dispositions, consumer transitions and remote integration remain incomplete | Complete evidence then integrate/retest/review | Component qualification is not terminal completion |

Post-run review: the uint64 rewrite was a device-kernel compatibility defect,
not evidence against the mathematical rounding procedure. Exact integer and
original-record checks discriminate the repaired arithmetic. The strongest
remaining limitation is that internal prepared-input components do not prove
the complete public call chain. No new timings were admitted from this run,
no tolerance changed, and numerical source/tests/driver remained frozen within
each worker/matrix. Desktop roles and matched physical identities are recorded;
other users can still start workloads after preflight, so final performance
comparisons require their own clean observations and repeated measurements.

September 22 continuation through 02626 resolves the rejected posterior
comparison under the [owner-directed rejection criterion](filter_gradient_rejected_dense_precision_decision_20260922.md).
All 150 CPU/GPU posterior checks, one additional focused CPU check and 102 policy
checks pass. GPU2 remains selected with verified memory growth. Runtime,
conditioning limits, accepted-result tolerances and numerical allow lists are
unchanged. Charges are now 52,437.52665588881 CPU / 49,589.058448168376 GPU seconds;
17.43 CPU / 38.23 GPU process-hours remain. The earlier 02620 charges are a
historical checkpoint. No worker is active; main remains unmerged.
