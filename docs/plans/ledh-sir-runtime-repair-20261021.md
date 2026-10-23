# Repair the merged SIR runtime without weakening validity

21 October 2026. Authorized by the user's request to plan, review, and execute
a thorough repair. Worktree: `ledh-iapf-execution-20260922`; branch:
`ledh-refactor-with-policy-fix`; starting HEAD: `90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`.
Existing dirty work belongs to the ongoing guidance study and must be preserved.

Status: completed and accepted within the tested SIR runtime scope. The
[terminal result](artifacts/ledh-sir-runtime-repair-20261021/result.md) records
all numerical, timing and provenance checks. No full pre-merge speed recovery
or scientific guide promotion is claimed.

## Question and evidence contract

Can the shared analytical flow retain its determinant, total directional
derivative, and failure checks while recovering the runtime lost after merging
main? The immediate comparator is the frozen **post-merge** implementation in
`artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/merged-source` (verified
against the preceding runtime manifest). Pre-merge timing is
context only: it omitted live validity work and is not a correctness authority.

The exact target for each flow step is M = I + epsilon A,
log|det M|, and epsilon tr(M^{-1} dA), on the nonsingular finite branch.
Keep all particle, auxiliary, covariance, determinant, and force validity
checks. Preserve the physical proposal mixture, analytical recursion, UKF,
GenUT, Contract E, OT (1,5), particle count 64, horizon 5, 24 substeps,
FP64/TF32-off, exact chunk 64, data, fitted guides, and seeds. This is numerical
implementation validation on exposed data, not tuning or a new scientific claim.

Primary acceptance requires:

1. Independent determinant/solve and finite-difference tests; healthy and
   invalid-input checks in FP64 and FP32; GPU/XLA compilation and bounded tracing.
2. Exact flags and unchanged healthy state/affine updates; FP64 determinant
   and tangent agreement within atol=2e-10, rtol=2e-9 on well-conditioned
   primitives. FP32 uses atol=2e-5, rtol=2e-4. Deliberately ill-conditioned
   cases are reported separately, never used to relax these thresholds.
3. Consumer replay against the merged source: same validity and finite outputs;
   loss/moments/value/analytical score use the existing replay tolerances,
   atol=rtol=1e-8 (verified in `compare_numbers`). No tolerance widening after results.
4. Alternating warmed paired timings across multiple existing datasets,
   zero and nonzero directions and plain/conditional/physical endpoints.
   A paired 95% interval for the geometric runtime ratio below 1 on plain and
   conditional supports a scoped speed claim. Physical-mixture non-harm requires
   an upper bound <=1.10; otherwise investigate before accepting the repair.

Invalid outputs, lost singular/nonfinite detection, changed target/settings,
source mismatch, or parity failure veto acceptance and trigger localized repair.
They stop further campaign stages until explained. Compilation, trace counts,
HLO factorization structure, allocator memory, and serialization times explain
the mechanism; they are not score-accuracy or posterior evidence. Failure to
improve speed rejects this optimization, not the LEDH research direction.

## Implementation and mathematical review

First test one shared pivoted LU factorization of the **same M**. If PM=LU,
log|det M|=sum log|U_ii| because |det P|=det L=1. Solve
LY=P dA, UX=Y, then the derivative is epsilon tr X by Jacobi's formula.
This removes construction of Q and reuses the factor for the derivative.
It changes neither the matrix nor the mathematical derivative and works for
nonsymmetric matrices and arbitrary leading batch dimensions. A zero/nonfinite
pivot or nonfinite input remains rejected; identity placeholders are only used
under false validity. Nonfinite tangents invalidate force without invalidating
a healthy value. No ridges, clipping, thresholds, or autodiff score are added.

An observation-space determinant identity is a possible later optimization:
with S=R+lambda HPH', det M=det(R+(lambda-epsilon/2)HPH')/det S.
It needs stronger covariance/roundoff checks than same-matrix LU. Do not install
it silently if LU fails: record a revised derivation and validation first.

Stages:

1. Freeze current source, preserve original QR as a diagnostic comparator only,
   and probe native TensorFlow LU/solve GPU-XLA support on a tiny fixture.
2. Implement a shared checked determinant/tangent helper, connect the actual
   canonical flow consumer, and test singular/nonfinite factors and derivatives,
   batch axes, signs, conditioning, and analytical directional parity. Check
   existing guide, marginal/physical-mixture, canonical-score and safety tests.
3. Audit the native batch path. Do not replace the guided consumer with a
   superficially similar batch kernel: it must support its guide, physical
   mixture, moment trace, correction, and analytical derivative contract.
   Fix the shared root cause first. Document any remaining batching integration
   work separately if it is a different implementation project.
4. Run a small actual-consumer parity check, then the full existing frozen
   holdout/direction replay if the check passes. Run alternating timing blocks
   on at least four already exposed datasets, with one-time compilation excluded.
   Inspect optimized HLO from the consumers, not only from the helper.
5. Analyze paired timings, numerical and invalid-input results; review the
   final diff and result; update checkpoint and master. Stop at a documented
   accepted repair or a concrete veto, never at an unexplained failed attempt.

## Defaults, assumptions, and sanity checks

| Choice | Provenance and role | Risk / earliest diagnostic |
| --- | --- | --- |
| Same-matrix pivoted LU | Exact algebraic replacement, candidate | GPU support or pivot instability; compile probe and QR/solve comparison |
| Frozen SIR FP64 scope | Existing merged replay, comparator | Cannot establish TF32 or other-model performance; test FP32 mechanics separately |
| Existing OT and guide controls | User-frozen diagnostic settings | Not retuned or eligible for default promotion |
| Eight repeats over >=4 datasets | Bounded timing protocol | Load autocorrelation; alternate version order and report dataset/block pairs |
| 10% physical timing non-harm | Engineering acceptance margin | No permission for numerical degradation; separate strict parity checks |
| Four dataset regimes | Existing holdout strata | Not fresh scientific evidence; state exact identifiers before launch |

The cheap engineering adversaries are (a) unchanged merged QR, which must be
outperformed without losing checks, (b) direct determinant plus independent
solve, which checks the numerical target, and (c) unchanged pre-merge HLO,
which explains dead-code elimination but cannot authorize removal of checks.
Evaluate by endpoint and direction, with ordinary and boundary/high-loss
datasets where present. No learned-method superiority is being assessed.

Pre-mortem: a microbenchmark could pass while the consumer still uses QR;
consumer wiring and HLO checks prevent this. A faster helper could alter physical
mixture weights; full replay and score comparisons prevent this. Batched timing
could compare different algorithms; batching is not substituted without a
separate equivalence check. A QR-based test alone could repeat the same error;
independent solves and finite differences supplement it. Near-singular matrices
can distinguish rejection policies; preserve and explain those cases explicitly.

## Budget, environment, commands, and artifacts

Parent remaining authorization is 14657.101235 GPU-process seconds. Allocate
at most 7200 to this repair, including probes, tests, idle resident workers,
failed attempts, replay and timing. At most three implementation candidates;
use a fresh attempt directory for every serious launch. Stop launches at the
cap. No training, new data selection, remote compute or environment mutation.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted GPU
access, verified memory growth before initialization, XLA on, explicit stable
input signatures, TF32 off for SIR; existing WSL command-buffer exception.
CPU unit/reference runs explicitly set `CUDA_VISIBLE_DEVICES=-1`. No pfor.

Root: `docs/plans/artifacts/ledh-sir-runtime-repair-20261021/`.
Record source snapshots/checksums, exact commands, branch/HEAD, environment,
seeds, memory growth and allocator peak, elapsed process charges, every attempt,
raw rows, HLO, and terminal JSON/Markdown decision and inference-status tables.
Commands will use `scripts/replay_sir_postmerge_diagnostic.py` and a bounded
extension of the existing runtime diagnostic; append exact invocations to the
execution record before launch. Tests are explicitly part of this repair task.

## Skeptical plan review before execution

PASS with the revisions above: the comparator is merged QR, not pre-merge
outputs; the target and all validity work remain fixed; runtime is evaluated
at real consumers; no proxy moment or guide metric becomes a scientific
promotion criterion; all exposed data are labeled; scope, budget and stop
conditions are explicit. Avoid the observation-space shortcut until its extra
assumptions are independently checked. Do not wire an incompatible native
batch endpoint solely because its name suggests speed. Independent review is
advisory; material numerical findings must be resolved before the relevant run.

### Compatibility finding and reviewed implementation refinement

Native `tf.linalg.lu` has no registered XLA_GPU_JIT kernel in this environment
(probe.json, 1.902 seconds). Implement partial-pivot Gaussian elimination in a
single TensorFlow while-loop, applying each row exchange and elimination to
both M and dM. The final upper-triangular solve yields M^{-1}dM without storing
Q or explicitly forming an inverse. This is the same LU algebra, not a new
scientific candidate. Check finite factors/nonzero pivots, retain separate
value/force masks, and test arbitrary leading batch dimensions. No eager or
non-XLA fallback. Independent Claude review returned an authentication failure;
the recorded skeptical self-review remains the available plan review.

### Timing allocation fixed before consumer execution

Use the first existing analytical-score dataset in each of the four original
joint regimes: 263043001, 263043004, 263043021, 263043052; replicate zero,
zero direction and coordinate-zero nonzero direction. These IDs come from the
merged replay manifest, independently of candidate outputs. Eight alternating
blocks rotate endpoint and dataset order. Average paired log runtime ratios
over the four datasets within each block, then form an approximate Student-t
95% interval over the eight block means (7 df). Exponentiate its endpoints.
Record per-regime intervals too; any regime with a measured regression remains
visible and vetoes a blanket speed claim. Timing uncertainty is conditional on
these inputs, hardware and runs, with independent block errors an approximation.
Require both directions to meet the stated endpoint criteria. The warm phase
checks all four datasets before collecting timings. Cap this attempt at2400
GPU-process seconds. This is implementation non-harm validation; the unchanged
scientific heuristic comparisons remain in the full replay and are not tuned.

The batch-path audit found a different API and computation: its input is
batched theta with shared observations, initial cloud, innovations and three
covariance-mixture probabilities; it calls `build_maps`, not the guided
canonical recursion. It has no backward-guide/physical-mixture/moment-callback
arguments. Substituting it would change the method. This repair therefore
targets the actual shared guided flow; batch integration is not required to
remove the diagnosed regression and is not smuggled into the comparison.
