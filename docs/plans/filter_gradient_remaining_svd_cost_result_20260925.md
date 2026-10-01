# Remaining SVD consumer cost and memory result

Date: 2026-09-25
Status: `COST_COHORT_COMPLETE_CAPACITY_ATTRIBUTION_OPEN`

The complete affected-consumer cohort is qualified under the frozen before
closure `9696666e4` and the candidate source set at commit `ea94aac96`. It uses
the actual dense, block, sequential, quadratic and COD composition, 7x3 and
11x5 designs, 20 synchronized warm calls, changed operands, replay, HLO and
caller/resource ownership checks. CPU and GPU each have three fresh-process
repeats for prior graph, prior XLA, candidate graph and candidate XLA. The GPU
cohort uses one verified non-desktop physical device,
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, with memory growth and clean sampled
sharing. The independent analyzer recomputed every raw numerical screen and
validated source, environment, JUnit, device, replay and ownership evidence.
Reports are [CPU](artifacts/filter-gradient-repair-20260917/remaining-svd-cost-cpu-03736.json),
[GPU](artifacts/filter-gradient-repair-20260917/remaining-svd-cost-gpu-03760.json),
and the immutable [receipt](artifacts/filter-gradient-repair-20260917/remaining-svd-cost-receipt-03760.json).

The candidate and prior graph arms pass the independent exact-quadratic checks.
Graph-to-graph warm ratios are 0.982 (D3) and 1.015 (D5) on CPU, and 0.954
(D3) and 0.867 (D5) on GPU. Their cold ratios are 0.984/0.989 on CPU and
0.985/1.012 on GPU. These are matched descriptive observations, not a claim of
performance superiority. All measured caller programs trace once, have no host
callbacks, preserve changed-input HLO, and release the caller function, graph,
callback, owner and resource. The reusable shape-only primitive has zero
captures; its custom-gradient registry graph is explicitly retained for the
worker lifetime and is not presented as native-memory eviction evidence.

The prior XLA arm fails the independent numerical screen on both dimensions and
both devices. Its output is preserved as failure evidence and no prior-XLA
speed ratio is admitted. The candidate XLA arm passes the numerical screen.
Compared with candidate graph, its cold ratio is 2.997 (D3) and 3.154 (D5) on
CPU, with maximum observed RSS increases of 344.1 and 354.7 MiB. GPU cold
ratios are 1.865 and 2.072, with maximum observed RSS increases of 20.3 and
20.6 MiB. Candidate-XLA warm ratios are 0.352/1.184 on CPU and 0.350/1.509 on
GPU for D3/D5. The D5 GPU warm increase and the cold and host-residency
thresholds trigger lifecycle and capacity attribution. TensorFlow allocator
peaks and process GPU reservations are reported separately; the GPU allocator
ratios are approximately 0.006/0.013 and reservation ratios are 0.907 in the
candidate-to-graph comparisons, so they do not measure or prove total CUDA
context or executable memory.

The CPU and GPU evidence agree on the mechanism boundary: the magnitude-normalized
SVD is numerically valid and the reusable XLA composition changes compilation
and host residency. The measurements do not identify which XLA compilation
arena or TensorFlow/native cache accounts for the host increase. Fresh-process
containment and many-signature capacity remain separate work. The result is an
engineering cost disposition, not a reason to change the numerical algorithm,
comparison tolerance, XLA default, or canonical NeuTra policy.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Admit numerical candidate and graph parity | All accurate arms pass independent checks; graph-to-graph records and ownership match | Prior XLA remains failed and excluded | Broader public/capacity behavior is untested | Run public/actual-DZ5 consumer qualification and terminal source audit | Posterior accuracy, HMC readiness or superiority |
| Retain candidate XLA cost | Candidate output is accurate and provenance/device/replay checks pass | Cold/host and D5 GPU warm triggers require attribution | Native executable and TensorFlow cache ownership is unresolved | Use fresh-process lifecycle and bounded capacity diagnostics | Native-memory eviction or unlimited concurrent capacity |
| Preserve prior-XLA failure | Independent analyzer reproduces invalid precision/rank status | Speed ranking is vetoed | Failure mechanism belongs to the old route | Keep as historical baseline evidence only | Any statement that XLA is generally inaccurate |

Runs 03710 and 03711 are preserved harness failures. Run 03728 wrote a passing
worker result but its parent did not record termination; it is excluded from the
cohort and charged its full 300-second reservation under the existing campaign
rules. The final accepted cohort contains 48 workers and does not alter the
campaign target, method, hardware class, gates or global budget.

Terminal review on recovery: both saved analyses reproduce exactly from their
raw records, and the receipt's three artifact checksums match. Final checks
03761/03762 pass 25 analyzer and 129 policy tests; focused Ruff and whitespace
checks pass. The 54-worker unit consumed 957.793425 seconds, including failures,
the interrupted reservation and final checks, within its 7200-second allowance.
The committed `remaining-svd-cost-evidence-03762.tar.gz` preserves the available
raw records, manifests, logs and JUnit for runs 03709--03762. Its checksum and
the verification outcome are in `remaining-svd-cost-verification-03762.json`
in the same artifact directory. The original shared output tree remains intact.

The strongest alternative explanation for the timing changes is compilation
and process-lifetime overhead specific to this small composition. The weakest
evidence is full consumer capacity: this experiment does not measure the actual
DZ5 program or native eviction. A failed independent reanalysis, changed source
closure, or shared-device observation would invalidate the corresponding cost
comparison. None is present in this recovered cohort. Keep capacity triggers
open rather than extrapolating fixture success to actual targets.

Next-step review: remote integration must preserve current canonical IAF and
the repaired numerical authority. The old source freeze cannot qualify merged
code; focused post-merge tests and a fresh DZ5 snapshot are mandatory before
actual target execution. Branch integration is authorized and does not promote
the repair to main or close the master program. No new numerical method,
tolerance, training recipe, or compute allowance is introduced.
