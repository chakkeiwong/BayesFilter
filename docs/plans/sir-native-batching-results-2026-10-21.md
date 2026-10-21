# Native batching repair: SIR likelihood and analytical score

The requested batching repair is implemented and tested. The graph no longer
duplicates the model body for every parameter row, and shared Gaussian/OT
contractions avoid pair-by-state and pair-by-parameter intermediates. A logical
batch of 128 now completes through native four-row shards within 0.761 GiB of
GPU allocator memory. **The FP32 numerical criterion remains unmet.** This
commit provides an engineering repair and its evidence, not a numerically
admitted SIR training callback.

Plan: [execution and evidence contract](sir-native-batching-execution-2026-10-21.md).
Review: [mathematical and implementation audit](sir-native-batching-implementation-review-2026-10-21.md).
Machine-readable results: [summary.json](artifacts/sir-native-batching-20261021-01/summary.json).
This is the batching prerequisite within the existing
[SIR NeuTra integration plan](surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md).

## What changed and what was tested

Model values retain shape [B,...] and analytical derivatives [P,B,...]. SIR
integration shares each primal stage across all three parameter derivatives.
One shared proposal/filter/reset recurrence handles the model families; native
model adapters contain their physical laws. Fixed-signature TensorFlow/XLA
functions enclose the complete numerical evaluation. Python performs setup and
reporting, not per-row target dispatch. No pfor, scalar mapped fallback, or
autodiff score was introduced. The scalar public endpoint uses the same
recurrence with B=1; training eligibility still requires B>1.

Compact map covariances preserve shared structure. Coordinate accumulation
computes direct pairwise squared distances without [B,K,K,D] residuals. Packed
weighted moments propagate the complete finite-Sinkhorn and barycentre
derivatives without [P,B,K,K] score tensors. The cache remains O(B N²); this is
not a linear-memory OT implementation. Nonshared range-bearing covariance
mixtures still have larger pairwise intermediates, and its local polar callback
still repeats some primal work across parameter directions. SIR capacity does
not establish range-bearing capacity.

The optional sharded callback executes one compiled loop over native batches
larger than one. It pads an incomplete final shard by repeating the last row and
truncates outputs afterwards. Tests cover nondivisible sizes and invalid-row
isolation. The repair also makes post-failure ancestor histories independent
of other rows and fixes scalar covariance-tolerance output handling. Neither
repair changes healthy likelihoods or scores.

**119 tests pass in an isolated tree** built from baseline commit `f6bb90a00`
plus the intended source changes and prerequisites (403.32 seconds, two
dependency deprecation warnings). The suite covers seven model-callback
variants, five families of complete-filter comparisons, optional correction,
all parameter directions, fixed designs, permutations, duplicates, invalid
rows, scalar entry, shard parity, graph scaling and benchmark preparation.
An earlier working-tree suite also passed 81 checks. These are complementary
checks, not 200 independent scientific replications.

The isolated run first exposed an absent locally compiled custom op and a
legacy test's unrelated campaign dependency. The custom op is recorded as an
environment prerequisite. That older integration test is excluded from this
scoped commit; independent tests cover the repaired algebra. The benchmark's
own HMC-master import was removed, and all six prepared input hashes match the
saved baseline. Failures, retries and exact overlay files are preserved in the
[isolated-tree manifest](artifacts/sir-native-batching-20261021-01/isolated-tree.json).

## Frozen workload and numerical result

SIR has 18 state dimensions, nine observations per time, and three parameters.
All capacity results use T=50, N=4032, K=2016, uniform FP32, TF32 disabled,
16 flow steps, five reset iterations, epsilon 1 and zero optional correction
steps. The four standardized parameter rows are 0 and 0.05 e_j; actual
parameters are theta=0.35 q. The data, random design (seed 261008301), mixture
weights and controls are frozen to the saved comparison. Larger batches repeat
the four rows: this measures batch capacity, not posterior coverage.

The reference below runs the same finite filter in FP64 after casting the
identical FP32-prepared inputs. It is an arithmetic reference, **not an exact
SIR likelihood or score oracle**. Old and native FP64 scores agree to at most
4.35e-11 in relative vector L2; likelihood relative differences are at most
2.65e-16. This supports algebraic equivalence on these inputs.

| q row | Old FP32 score | Native FP32 score | Common FP64 reference score | Old error | Native error |
|---|---|---|---|---:|---:|
| 0 | (-184.1851, 98.4641, 59.6112) | (-184.2087, 98.4651, 59.6116) | (-184.1461, 98.4489, 59.6244) | 0.0202% | 0.0304% |
| 0.05 e1 | (-308.0964, 175.7104, 55.3816) | (-308.0920, 175.6988, 55.3921) | (-308.2686, 175.6646, 55.5629) | 0.0708% | 0.0691% |
| 0.05 e2 | (-91.4810, -116.0206, 58.2442) | (-94.6902, -114.4330, 57.7729) | (-83.4125, -120.0598, 59.4593) | **5.7689%** | **8.0571%** |
| 0.05 e3 | (-165.5156, 83.8191, 39.7200) | (-165.6985, 83.9680, 39.7518) | (-164.0741, 82.9671, 39.6841) | 0.8904% | **1.0150%** |

Errors use ||candidate-reference||₂/max(1,||reference||₂) per row. The maximum
native-versus-old FP32 difference is **2.2740%**, above the predeclared 1%
criterion. Native FP32 likelihood errors against FP64 are at most 1.46e-7
relative (0.0000146%); accurate likelihood values do not establish accurate
derivatives. CPU versus GPU FP32 scores also differ materially, so CPU output
cannot be adopted as truth. No tolerance was relaxed and FP32 remains the
requested arithmetic; FP64 was used only to diagnose the discrepancy.

GPU B8 matches native B4 exactly. B16/B32/B64 differ from B4 by at most
0.01175% in score-vector norm, within the batch-parity criterion. Sharded
B128/S4 matches B4 exactly. These checks answer row-batching consistency; they
do not erase the shared discrepancy from the FP64 recurrence.

## Capacity and timing

GPU: NVIDIA RTX 4080 SUPER, TensorFlow 2.20.0-dev0+selfbuilt, XLA on,
`tftwogpu` environment. The reviewed logical-device limit is 12 GiB; this is an
explicit memory-growth-policy exception, configured before initialization.
Allocator peaks measure tensor allocation, not whole-process NVIDIA reservation.
CPU-only arms hide GPUs and use 16 intra-op threads. GPU arms use four.

| GPU execution | First call, including compilation | Warm latency | Peak allocator | Outcome |
|---|---:|---:|---:|---|
| B4 | 38.20 s | 15.27 s median, 3 calls | 0.521 GiB | Complete |
| B8 | 56.81 s | 39.08 s median, 3 calls | 1.030 GiB | Complete |
| B16 | 88.17 s | 57.88 s median, 3 calls | 2.028 GiB | Complete |
| B32 | 139.40 s | 117.03 s median, 3 calls | 4.038 GiB | Complete |
| B64 | 252.08 s | 226.02 s, 1 call | 8.014 GiB | Two valid evaluations; repeat set hits 600 s cap |
| B128 | No completed evaluation | — | — | OOM: requested 16.078 GiB under 12 GiB cap |
| Logical B128, native S4 | 333.77 s | Not measured | 0.761 GiB | One valid evaluation, exact B4 parity |

For context, the saved old B4 kernel took 49.68 seconds warm and used 2.66 GiB.
The observed change is about 3.25 times faster and 80% less allocator memory.
Old B32 failed a 21.04 GiB allocation; native B32 completes within 4.04 GiB.
These are descriptive engineering comparisons. Shared-device activity,
concurrent CPU work and few repeats prevent a statistically supported runtime
ranking or a claim that the selected shard size is optimal.

CPU B4 completes first/warm evaluations in 214.58/206.12 seconds using
0.529 GiB, then hits the repeat cap. CPU B8 completes its first evaluation in
498.51 seconds using 1.033 GiB; its warm repeat is interrupted at the arm cap.
CPU B16 and B32 have no completed evaluation within 600 seconds. All arms
have ended. A timeout is not an out-of-memory
finding, and a valid completed call remains evidence even when later repeats
are interrupted.

## Decisions and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Preserve batching implementation and tests | Native graph, row isolation and FP64 parity pass | No algebraic mismatch found in checked scope | Long nonlinear FP32 sensitivity | Retain as unpromoted engineering candidate | All scopes are numerically admitted |
| Use batched sharding as a capacity option | B128/S4 completes within cap; exact B4 parity | No capacity veto on this evaluation | Complete optimizer-step memory unmeasured | Test actual target-to-trainer adapter after numerical repair | A SIR NeuTra map has been trained |
| Withhold FP32 numerical promotion | Maximum score mismatch 2.274% versus old, 8.057% versus FP64 | **1% criterion fails** | Amplification mechanism not yet localized completely | Target-preserving FP32 stability investigation within the same integration plan | LEDH, SIR or NeuTra is invalid |

| Inference status | Finding |
|---|---|
| Hard veto screen | B128 full-native OOM; uniform FP32 score criterion fails |
| Statistically supported ranking | None; no uncertainty-based runtime comparison |
| Descriptive-only differences | Latency, allocator peaks, CPU/GPU order sensitivity |
| Default readiness | Not established; candidate implementation does not authorize scientific use |
| Next evidence needed | FP32 reference agreement, actual training call chain, whole-update gradient parity and memory |

The engineering, numerical and scientific conclusions remain separate.
Graph duplication and pairwise-memory defects are repaired for the checked
callback; uniform FP32 score stability is unresolved; posterior correctness and
trained-map quality were not tested. No HMC or optimizer update ran. The next
FP32 stability investigation belongs to this same integration direction. The
bounded batching execution ends after its declared diagnostics rather than
silently launching a new training or posterior campaign.

Post-run red team: changing reduction order may amplify an already sensitive
SIR recurrence even when the real-arithmetic formula is unchanged. Full-size
FP64 agreement does not certify every parameter region, nor prove a score is
the derivative of an exact model likelihood. Repeated rows are a capacity
fixture, not a representative training distribution. A materially different
reference result, a new call-chain defect or a failure on fresh parameter rows
would require revising the corresponding conclusion.

## Reproducibility

Baseline commit `f6bb90a00` records the preceding benchmark and design. Each
run manifest records its exact command, environment, device, input/source
hashes, seeds, timing clocks, guards and results. The current Git revision plus
dirty source snapshots identifies the actual executed code; Git HEAD alone
would be insufficient. The
[verified source archive](artifacts/sir-native-batching-20261021-01/source-archive.json)
preserves all executed source variants. Original logs and failed attempts
remain alongside the successful results.

`assemble_summary.py` in the result directory rebuilds the compact summary
using standard-library post-run arithmetic. The complete CPU/GPU budget ledger
is in `summary.json`: supervised GPU arms consume 2,753.16 seconds (45.89
minutes); CPU arms consume 2,401.44 seconds, and all logged pytest attempts
another 1,068.66 seconds, totaling 57.83 minutes. This accounts for failed
attempts and excludes editing, reporting and small untimed setup/import commands.
Both totals remain below the two-hour plan ceilings.
