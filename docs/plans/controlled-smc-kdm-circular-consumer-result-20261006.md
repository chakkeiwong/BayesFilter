# Circular KDM consumer continuation — 6 October 2026

The TensorFlow implementation now executes the Normal/Normal/von-Mises
resampling-density, IWSG, posterior-NLL and connected weight-to-loss consumers.
All 157 float64 checks pass against the actual author methods. Float32 passes 153
of 157 checks; four gradient comparisons remain outside the predeclared bound.
The complete 315-check GPU screen therefore fails. Both replications remain
incomplete. This result establishes matched-input consumer parity in float64
only; it does not establish neural-filter training or autonomous sampling.

## What was checked and repaired

[Plan](controlled-smc-kdm-circular-consumer-20261006.md).
The comparator executes the unmodified author resample_particles,
weigh_particles and KDE_NLL.compute_loss method bodies, real KDE implementations,
and the actual bearings apply_norm method. Constructor settings are supplied
explicitly; the positive neural-weight output and downstream cloud are
constructed fixtures. All author files used by these endpoints match commit
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18. The reference is installed CPU PyTorch,
not a reproduction of the authors' original machine/environment.

Paper equations14–15 explain the stopped samples and importance-weight
derivatives. Equation18 and the bearings setup specify the posterior NLL and
Gaussian/circular product kernels. The released source has two distinct finite
density programs: resampling normalizes weights after adding1e-8 and uses
circular concentration1/(bandwidth+1e-8); posterior NLL uses supplied weights,
concentration1/bandwidth and a final1e-8 density floor. The new TF consumers
preserve those differences and the installed PyTorch log-I0 polynomial.
Replacing the posterior density with the resampling density changes NLL by
up to 16.79019 in the constructed tiny/unnormalized-weight control. That is a
counterexample to reusing one endpoint, not an observed error in a trained model.

Attempt 01 passed 311/315 checks. The largest raw angular-bandwidth gradient
discrepancy was 0.3070068 at reference 598.8538 (~0.0513%). Source inspection
found concentration was broadcast to particles before log-I0, while the initial
translation differentiated a shared normalizer after particle reduction.
Preserving the source broadcast and subtraction order reduced this selected
discrepancy to 0.0369263 (~0.00617%) in attempt 02, but did not clear all four
failing comparisons. The connected consumer bandwidth discrepancy increased
to 0.000119925. This repair is insufficient and is not a quality improvement claim.

The float32 primary bound remains 128*machine-epsilon*(1+abs(reference));
float64 remains1e-9*(1+abs(reference)). Neither tolerance was changed after
observing results. Tests cover ordinary, angular-seam, tiny-weight and far-tail
cases in both dtypes, with batch2 and25 particles. Posterior densities, NLL
values and their gradients pass in all eight cases. The remaining GPU failures
are raw IWSG weight gradients in ordinary/seam cases, raw IWSG bandwidth
gradient at the seam, and the connected consumer bandwidth gradient at the seam.
All finite/domain, negative-bandwidth, negative-weight, zero-mass, periodicity,
stopped-sample and device checks pass. This does not numerically integrate the
full product density or test a TF circular random sampler.

The unchanged candidate was then replayed on CPU without XLA. Five float32
comparisons fail the same bound, while all float64 comparisons pass.
Thus removing GPU/XLA does not remove float32 disagreement. The experiment
changes both device and compiler, so it cannot separately assign their effects.
The CPU run intentionally hides GPUs; its CUDA-no-device startup message is
expected and is not a broken-driver finding.

## Engineering and scientific decisions

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept bounded float64 consumer parity | All 157 checks pass | No nonfinite/domain, device, trace or gradient failure | Four constructed cases per dtype, matched author samples | Retain as reference for the missing full neural consumer | Full replication, sampling or training correctness |
| Do not accept complete float32 parity | Four GPU checks fail | Fixed comparison bound exceeded | Finite reduction/derivative ordering; downstream impact unmeasured | Audit remaining Normal variance/reduction order if pursuing arithmetic parity; preserve failures | KDM or its training cannot work |
| Reject XLA-only explanation | CPU also has float32 mismatches | No runtime or artifact failure | CPU/device and compiler not isolated independently | Treat numerical execution order as a candidate mechanism | A unique compiler defect or exact instruction-level cause |
| Preserve distinct source loss endpoints | Counterexample and actual-method comparisons | Wrong-density substitution detected | Training normally supplies normalized weights | Use explicit consumer roles in the neural port | 16.79 NLL error in an actual trained filter |
| Full campaign remains incomplete | Lorenz fitting/P1 and full KDM training absent | Missing required evidence | Unspecified criteria/training cost and missing checkpoints | Complete contracts/ports and cost prerequisites before launch | No-worse filtering quality or paper reproduction |

| Inference status | Finding |
|---|---|
| Hard veto screen | Four float32 GPU comparisons fail; no observed NaN/crash/domain failure on valid fixtures |
| Statistically supported ranking | None; this is deterministic matched-input parity |
| Descriptive-only differences | Individual errors and timing; no estimator-quality or speed ranking |
| Default readiness | Not established; runtime candidate is not wired into a trained filter |
| Next evidence needed | Full neural consumer, optimizer/checkpoint conversion, author/TF training under an independently justified no-worse contract |

Each of the six GPU endpoint graphs traces once. The implementation is batched,
uses no per-sample Python/TF loop, pfor or NumPy numerical route, and contains a
bounded 8-step TensorFlow Horner loop. XLA compilation is recorded in the logs.
Memory growth was verified before initialization. Allocator peak for attempt 02
was 202752 bytes; this is live TF allocator evidence, not total process residency.
Actual CUDA device 0 was RTX 4080 SUPER. The initial nvidia-smi ordering assumption
was wrong and is corrected in the plan; both comparisons used the same GPU.

Strongest alternative explanation: framework floating-point scheduling accounts
for the remaining differences although the algebra agrees. The passing float64
and failing CPU float32 checks support this interpretation but do not prove every
finite operation is faithfully ordered. Inspecting the Normal source still shows
variance-then-division versus the translated standardized residual; this is a
remaining arithmetic difference, not a mathematical target difference. No
gradient clipping, damping or precision promotion was adopted to conceal it.
A larger/domain-specific counterexample could overturn the bounded parity
conclusion. The weakest evidence is that no real neural training trajectory was
tested. Candidate screen failure is not research-direction rejection.

## Resources and preserved execution

GPU attempts 01 and 02 consumed 19.885164164938033 and 17.879331667907536 seconds,
including their CPU author children. CPU localization consumed 5.536095131188631
seconds. Remaining active budget: 664.9919998964178 GPU seconds (~11.08 minutes)
and 151416.64133572605 CPU seconds. No new compute grant or job is running.

Evidence directory:
artifacts/controlled-smc-kdm-replication-20260928/.
Subdirectories kdm-circular-20261006-01 and -02 contain immutable attempt inputs,
source snapshots, outputs, manifests, logs and execution receipts.
kdm-circular-cpu-localization-20261006-01 preserves the explicit non-XLA reference.
kdm-circular-review-20261006-01 contains the aggregate decision, checks and tests.

The GPU substage has used both predeclared attempts. The additional CPU
localization was separately specified before execution. Do not treat another
identical GPU rerun as progress. The master retains Lorenz source-compatible
fitting or independently justified downstream no-worse qualification, plus
missing KDM neural/training/checkpoint work. Complete training cannot be promised
within the remaining GPU balance.

## Terminal review

The focused reporting, fixed-policy forward and fitting/forward suite passes
20/20 tests; all six changed Python modules compile. Preserved GPU output and
captured-source hashes verify. The supervisor, progress report and refreshed
checkpoint agree on both incomplete replications, no running job, the exact
remaining budget, and the failed float32 screen. In particular, completing an
audit cannot erase numerical_screen_passed=false or the per-dtype verdicts.
Routine CPU-only engineering verification took 0.6661304330918938 seconds and
is recorded separately from the research campaign.

The master, active execution contract and gap-closure sequence now link this
result. The review preserves the original numerical bounds, author failure
policy and replication-only scope. It found no basis to promote a full
replication or to infer training quality from primitive/constructed-consumer
agreement. Full neural execution and downstream impact remain the material
limits. No independent external review was performed.
