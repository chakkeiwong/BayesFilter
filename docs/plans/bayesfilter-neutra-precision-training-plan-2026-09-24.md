# NeuTra precision repair and q20 training evaluation

User authorization: implement configurable FP32/TF32 training, commit and merge
with the remote and push, then evaluate the repaired training. The target is
the existing q20/T30 four-parameter UKF posterior approximation. This is not a
method-comparison campaign or a request for latent mass adaptation.

## Engineering question and execution order

1. Separate transport/optimizer precision from target-evaluation precision in
   the shared numerical implementation. Preserve existing FP64 checkpoints and
   facades. Expose FP32 explicitly; use dtype-appropriate inverse tolerances.
   Make promotion of trained FP32 weights to a frozen FP64 evaluation map an
   explicit, recorded conversion, followed by fresh diagnostics.
2. Test independent equations, mixed-precision target gradients, standard/path
   estimators, inverse/tail behavior, checkpoint continuation and frozen replay.
   Run a bounded trusted GPU/XLA FP64 / FP32 / FP32-with-TF32 comparison using
   identical parameters and inputs. TF32-enabled metadata is not proof that
   Tensor Cores were used; inspect compiler evidence where available.
3. Commit only the NeuTra work and its necessary dependencies in an isolated
   checkout, merge the remote, resolve conflicts, rerun affected checks and
   push. Preserve concurrent unrelated dirty HMC and governance work.
4. After synchronization, run target-specific pricing/calibration, sustained
   training, independent post-training verification and affordable downstream
   qualification. Record a fully specified training amendment after actual
   prices are known and before starting the training ladder.

## Evidence contract and intent ledger

| Item | Contract |
|---|---|
| Main question | Can corrected source-based IAF/conditional NAF, adequate optimization and configurable precision produce useful q20 transport geometry? |
| Precision comparator | Identical arrays/inputs evaluated with FP64, FP32 and FP32 with TF32; FP64 is a reference, not presumed necessary for training. |
| Training baseline | Preserved depth-four 2,048-update q20 map, freshly evaluated on common banks; matched continuation controls where affordable. |
| Engineering pass | Finite valid batched gradients, independent derivative agreement at precision-appropriate tolerances, inverse reconstruction, exact same-dtype resume, explicit precision provenance. |
| Training improvement criterion | Predeclared paired heldout loss differences with uncertainty and fresh 1,000-point score geometry, replicated across initialization/noise seeds. Neither alone establishes posterior correctness. |
| Downstream promotion | Exact frozen-map public fixed-transport HMC qualification and existing model-coordinate posterior checks, with identity latent mass. |
| Promotion veto | Invalid target rows, nonfinite map/gradients, failed inverse/derivative checks, missing diagnostic, or failed downstream checks. |
| Continuation veto | Broken target/implementation or corrupted artifacts, deadline, exhausted total budget, or inability to fund the declared complete replication. A poor candidate is not a continuation veto. |
| Repair triggers | Precision mismatch, unstable optimizer, persistently improving map at a rung, scale/conditioning failure, or inadequate capacity; use bounded declared repair branches. |
| Explanatory diagnostics | Gradient/update norms and clipping, validation trajectory, residual quantiles, log-density correction, tail slices, compile/steady timings. No ranking from single-run maxima or timings. |
| Nonconclusions | No universal TF32 adequacy, no exact-paper-experiment reproduction, no convergence from acceptance or loss, no posterior promotion from a short fit. |

## Defaults, resources and skeptical audit

FP64 target arithmetic is inherited from the existing checked q20 evaluator;
it is a compatibility baseline, not evidence that every computation requires
FP64. FP32 flow/Adam with TF32 is the requested candidate. Original source
code uses FP32. Stable log-domain NAF equations remain unchanged. Existing
FP64 inverse tolerances of 1e-11 cannot be reused for FP32: choose the FP32
engineering tolerance relative to machine epsilon and test independent
reconstruction, tails and implicit derivatives. Do not silently relax a
scientific convergence threshold.

The last q20 ledger is
`artifacts/q20-short-fit-canary-2026-09-23/accounting-summary.json`, with
143330.11637 campaign worker-seconds (39.81 hours) remaining. The old canary
allocation is closed and is not reused. This execution draws from the existing
campaign balance, conservatively charging worker wall time including failed
attempts and concurrent workers. The deadline remains 2026-09-25 18:00
Asia/Shanghai. Initially reserve at most 1,800 CPU and 1,200 GPU seconds for
precision engineering; these are convenience caps within the existing balance,
not new compute grants. Reconcile before allocating the serious training study.
Integration amendment: the earlier uncommitted q20 implementation is a necessary
dependency of post-training diagnostics and must be committed with the shared
authority. Its compatibility checks expanded the CPU engineering reservation to
3,000 seconds (a convenience cap within the same remaining allowance); GPU
engineering remains capped at 1,200 seconds. Charge repeated failed attempts.

Artifacts use fresh directories under
`docs/plans/artifacts/neutra-precision-training-2026-09-24/`. Every serious
worker records command, source hashes/commit, environment, memory-growth
verification before GPU initialization, actual dtype/device/XLA/TF32 policy,
seeds, target/data identity, checkpoints, timing and remaining allocation.
GPU occupancy is inspected with trusted permissions; independent training
workers may use free GPUs, and their times are summed.

Skeptical audit before implementation: the material risks are confusing TF32
flags with actual FP32 arithmetic; evaluating a different posterior after a
cast; changing old artifact meanings; false inverse failures from FP64
tolerances; comparing different initial weights; stopping training at a smoke
rung; and including unrelated active work in a commit. The design addresses
these through explicit precision boundaries, parameter-matched checks,
compatibility tests, priced sustained training with independent validation,
and isolated Git integration. No scientific quality conclusion is permitted
from the precision smoke. Audit passes for engineering; the priced training
amendment must pass its own scientific audit before execution.

## Precision check specification

`check_neutra_precision_2026_09_24.py` runs each mode in a separate process on
GPU1, initially observed idle. It uses three stages, two width-16 layers, four
dimensions and 32 rows, matching the order of the q20 training workload. Both
IAF and DSF use standard and path estimators. Eight repeated updates per cell
test compilation/optimizer mechanics only. A coupled precision test and
independent inverse derivatives also run in the CPU-hidden unit suite.

The GPU analytic target is an anisotropic Gaussian plus a quartic term, with
its value and score evaluated in FP64. Identical represented weights and
inputs are used for each CPU reference. Report scaled gradient L2 error and
scaled loss error. Limits are 1e-9 for FP64, 1e-4 for FP32 and 0.01 for TF32:
engineering screening hypotheses allowing operation accumulation beyond the
respective machine precision, not scientific-equivalence tolerances. A failure
requires diagnosis, not automatic widening. Test trained-map inversion on
10-times base-normal inputs; scaled roundtrip limits are 1e-7 for FP64 and
0.002 for FP32/TF32. Exported FP64 maps must also reconstruct within 1e-7.
Preserve compiler HLO, timings and actual variable/target dtypes. Actual q20
calibration must subsequently compare precision drift with gradient variability
and check the exact frozen evaluation map.

## q20 pricing and calibration entry (after Git synchronization)

Use the preserved depth-four checkpoint from
`q20-training-repair-2026-09-23/campaign-01/attempts/00005-continue-depth/worker/data/cohort-00000.json`,
candidate `direct-w16-lr0.0005-r0`, with 2,048 lifetime updates. Check its
target/bridge identities and reconstruction before using it. The historical
scalar correction is explanatory evidence, not the baseline for the new full
conditional NAF.

The first target worker is a pricing/calibration run, capped at 2,400 worker
seconds from the same campaign. Preserve the original baseline and construct
configured IAF and DSF candidates. The control uses the legacy four-stage
width-16 tanh map. New candidate hypotheses are three-stage ELU IAF with free
scale bias at width16, and three-stage conditional DSF with sixteen mixture
components and widths16/32. The mixture count follows Huang appendix E’s
IAF-DSF setting; the conditioner widths are target-specific hypotheses. Width16 comes from the existing workload; width32
is an explicit capacity alternative. Three stages and the IAF initializer come
from the checked source profile; DSF width and component count need local
evidence and are not claimed paper defaults.

Use one fixed diagonal affine initialization derived from 8,192 draws of the
saved baseline for all new maps. This is a warm-start approximation to an
already available map, not a posterior estimate or coverage claim. The draw
count is a cheap power-of-two convenience choice for moment stability. Generate
these internal initialization draws in a batched TensorFlow graph; no external
training dataset or scalar target loop is introduced. Record the derived center
and scales, and reject nonpositive/nonfinite scales.

Price compile and repeated full updates separately at batch32 and batch128,
the two previously executable batch sizes. Inspect standard and path gradients
on common unchanged parameters; compare their sampled gradient variation and
the FP32/FP64 discrepancy. Eight minibatches are a low-cost sanity screen, not
an optimizer/variance ranking. A precision discrepancy greater than one tenth
of the observed minibatch-gradient RMS variability is an investigation trigger;
the fraction is an explicit precision-screen hypothesis, not an HMC tolerance.

The priced amendment must fund complete independent training replications,
preserved checkpoints, disjoint validation and final banks, all 1,000-point
checks, and a downstream qualification reserve. It must not stop because a
candidate becomes trial-eligible. Each continuing-improvement signal at a rung
must reach its funded continuation or receive an explicit budget-limited
disposition. Sparse validation (at checkpoint rungs) replaces per-update
validation. Failed candidates trigger the declared optimizer/capacity repair,
not rejection of the NeuTra research direction.

## Integration audit, 2026-09-24

The checkpoint-migration fixture allocated exactly the protected 7,200-second
repair reserve. After migration/pricing, its correct result is a budget pause;
the test now checks that no unfunded training starts. No budget guard changed.
The refresh controller had drifted from the current estimation objective: it
omitted the method label in training pricing and still requested classical
preparation. It now prices plain NeuTra at beta one with its frozen chart and
identity latent mass. A real supervised-worker test covers exact checkpoint
restore, pricing scope and reuse without duplicate charges.

An existing downstream defect remains: the generic posterior information check
applies continuous tail ESS to `positive_theta_2`, a binary indicator. Its upper
tail indicator is constant, so the known Gaussian success fixture is rejected
despite finite R-hat, bulk ESS and event MCSE. The intended success test is
retained as a strict expected failure. This is a diagnostic repair trigger and
blocks downstream posterior promotion; it does not invalidate the independent
target, transport-gradient, loss or score-residual training evaluation. Repair
this diagnostic with explicit discrete-quantity semantics before using it for
an HMC promotion decision. Do not remove the quantity or loosen ESS thresholds.
