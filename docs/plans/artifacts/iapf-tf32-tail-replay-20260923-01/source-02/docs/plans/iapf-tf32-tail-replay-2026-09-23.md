# iAPF phase 19: locate the influential TF32 discrepancies

Phase18 completed its frozen comparison: 51/88 mean-precision contrasts pass,
37 are tail-unresolved, including 22 with intervals crossing the 2% margin.
All22 exact-guide contrasts pass. Continue the authorized localization remedy;
do not reclassify the unresolved comparisons or tune their acceptance margin.

## Question and evidence contract

Do the largest paired likelihood discrepancies start with a changed discrete
sampling decision, or with large arithmetic differences while decisions still
agree? Baselines are the actual saved phase18 FP64 and FP32 executions of
`make_fitted_twist_kernel`; candidate is the same TF32 consumer. Use the local
six-parameter linear Gaussian model, unchanged guides/data/draws, not the
paper-study model or a new fitting algorithm.

Select the three largest centered squared paired differences in every failed
phase18 contrast from its checked `review-01/tail-influence.json`. Deduplicate
cell/arm/replicate keys and replay all selected keys in all three modes. This
is deliberately outcome-selected debugging evidence, not new replications and
not an estimate of bias, failure frequency, or a mean-equivalence test.

Primary engineering criterion: every replayed value, score and cloud is finite;
its log likelihood is bitwise identical to the original worker CSV value;
tracing is bitwise neutral; every recorded ancestor and mixture choice agrees
with an independent host reconstruction from the actual CDF/probability and
binary-lattice uniform. A replay mismatch vetoes a causal interpretation of
that case and triggers source/environment/compiler localization, not deletion.
Broken inputs, hashes, GPU/XLA/growth provenance, nonfinite outputs, or budget
exhaustion are continuation vetoes. Local harness faults may be repaired within
the same campaign and phase allowance, preserving each attempt.

Explanatory diagnostics: first differing decision, cloud error before/after it,
CDF differences at that decision, exact lattice probability of an ancestor
disagreement conditional on the two CDFs, and minimum effective particle count
from successive CDF differences. Terminal resampling cannot affect the already
computed likelihood; mark such differences separately. These diagnostics cannot
clear the phase18 precision veto. No new fitting, default, marginal-score,
paper-replication, KDM, canonical LEDH, or HMC claim.

The already constructed heuristic set is BPF, observation-adapted Gaussian,
and exact future Gaussian guides. Phase18's conditional comparison across
dimension, dataset and horizon remains authoritative (90/99 observed stress
guide losses, descriptive only). Replaying selected seeds cannot establish
guide quality or rank viable stochastic methods.

## Mathematical check and assumptions

For L=2^23 uniform lattice points, the right-sided inverse CDF assigns label j
to integer indices in [ceil(L F[j-1]), ceil(L F[j])). Clip boundaries to [0,L]
and force the last upper boundary to L to reproduce the consumer's last-index
clamp. The agreement probability of two CDFs is the sum of the lengths of the
intersections of equally labelled integer intervals, divided by L. Its
complement is an exact conditional disagreement probability, not an empirical
population frequency. Gaussian/floor branch reconstruction uses U<p directly.

| Choice | Provenance/status | Failure risk | Early check |
|---|---|---|---|
| Worst three per unresolved contrast | Declared diagnostic selection | Selection bias | No population inference; preserve all original records |
| Same GPU, XLA, dtype and TF32 switch | Phase18 environment | Fresh compilation may alter arithmetic | Exact saved-value replay before interpretation |
| Existing consumer with optional tracing | Actual shared call chain | Instrumentation changes fusion | Bitwise neutral value, score and cloud |
| Exact FP32 inputs, 23-bit draws | Repaired phase18 loader | Decimal serialization or seed drift | Source/hash checks and exact saved-value replay |
| Host CDF/ESS calculations | Independent stdlib diagnostic | Endpoint convention error | Verify every ancestor with bisect-right and clamp |
| No numerical stabilization changes | Frozen comparison object | Existing tail problem persists | Preserve tail veto; this is localization, not repair promotion |

Skeptical audit: PASS. The comparison uses the real consumer and exact original
seeds, retains adverse cases, and explicitly forbids treating selected replays
as statistical evidence. A larger random sweep before this localization would
not distinguish a changed discrete path from arithmetic corruption. Source
checks and exact replay detect stale context and changed compilation. This is
self-review, not an independent-review claim.

## Execution and budget

Work in the isolated `ledh-refactor-with-policy-fix` checkout. Inputs and
budget opening balance come from phase18, which is settled: CPU64909.13763951734
seconds and GPU170009.73650610587 seconds remain in the owner-authorized campaign.
Phase caps: 120 CPU seconds, 2400 GPU process seconds, six attempts; reserve
at most600 seconds per GPU worker. No wall-expiry or new human approval.

Output root: `docs/plans/artifacts/iapf-tf32-tail-replay-20260923-01/`.
Freeze source closure, plan, selection, input hashes and original result paths.
Run `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/diagnose_iapf_tf32_tail_replay.py --execute` with trusted GPU
permissions. Use RTX4080 UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, growth
verified before initialization, XLA on, BLAS/OMP1 and TF intra/inter2; reject
inherited XLA overrides. CPU preparation/reporting uses standard-library code
with GPU intentionally hidden. Save exact commands, Git/source/input hashes,
environment, seeds, hardware/compiler provenance, wall time, logs and all
selected traces. Complete a terminal result with decision/inference tables,
update the master and concise checkpoint, and charge every attempt.
