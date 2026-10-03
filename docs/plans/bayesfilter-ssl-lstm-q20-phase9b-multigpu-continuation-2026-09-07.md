# Phase 9B multi-GPU continuation

Date: 2026-09-07
Status: `EXECUTED_STOPPED_BUDGET_INFEASIBLE`
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`
Parent: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`

## Current disposition

Executed on non-display GPU 1. Factor first/steady calls took 1,360.86/1,390.70
seconds. The 8,314.36-second six-chunk factor extrapolation before overhead
exceeds the 2,600-second arm cap. The strict arm was interrupted; no complete
two-arm forecast or passing M4-P0 closeout exists. The settled campaign balance
is 183.80 seconds, with no active reservations. Do not repeat the original
allocation or launch P1. The plan below preserves the pre-run evidence contract;
the terminal result and next Phase 0 repairs are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-result-2026-09-07.md`.

## Active amendment and skeptical audit

The owner's three-GPU placement instruction and request to continue supersede
the GPU-0-only placement rule, not the target, numerical method, evidence
requirements, or campaign cap. Historical run receipts remain unchanged.

The previous requirement to fund a *known feasible* complete P1 schedule before
measuring its unknown runtime was circular. A bounded readiness diagnostic is
authorized within the existing campaign remainder. Its completion does not
authorize P1: a complete measured schedule must still fit the then-remaining
budget and the 2,600-second arm cap. No larger compute budget is inferred.

The skeptical audit retains the exact strict comparator, chart 0, beta 1,
four chains, fresh backend-specific training/tuning, and 500-transition chunks.
Two repeated calls to the same compiled controller distinguish first-call cost
from steady-state cost; neither establishes posterior validity. The shared
controller caches its shape-specific programs. Required source checks, archive
checks, per-chain movement, finite values, XLA, memory growth, and forecast
completeness remain vetoes. P2 remains blocked. A missing runtime measurement is
a reason for bounded measurement, not a scientific rejection.

## GPU placement and assumptions

Use `bayesfilter_non_display_first_load40_headroom5g_v1`. Before importing any
accelerator framework, obtain a trusted live inventory, prefer eligible
non-display GPUs, and consider display GPUs only if no non-display GPU qualifies.
Pin `CUDA_VISIBLE_DEVICES` to the selected UUID and require
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import; verify growth before initialization.

| Choice | Provenance and justification | Failure mode and early check | Status |
|---|---|---|---|
| Utilization at most 40% | Owner instruction; exactly 40% qualifies | A stale sample misses a busy device; query in each launcher | Placement rule, not a speed claim |
| At least 5 GiB headroom | Owner instruction, implemented as at least 5,120 MiB free at the live selection check | Growth is not a hard cap and other jobs can allocate later; record allocator telemetry and recheck inventory at arm boundaries | Admission rule, not a continuous reservation |
| Reject when either load or memory fails | Conservative interpretation of the owner's ambiguous “and”; both availability conditions must hold, including display fallback | Avoids selecting a memory-starved idle GPU or a busy empty GPU | Explicit operational assumption |
| Display classification | Live `display_active`, `display_attached`, and graphics-process provenance | Xorg has a 4 MiB context on all three GPUs; graphics presence alone would wrongly classify all as display GPUs | Active/attached display flags determine preference; processes remain recorded |
| Tie break | Lower utilization, then more free memory, then inventory index | A single sample cannot rank GPU performance | Deterministic scheduling convenience only |

Do not kill display or other-user processes. Missing utilization, memory,
identity, or display classification fails closed for that device. No eligible
device means no launch. The selected GPU UUID, PCI bus, name, and total memory
must match the readiness measurement before using its forecast for P1; physical
identity is recorded, not inferred from logical `/GPU:0`.

## Evidence, budget, and stop conditions

The question is unchanged: can the repaired two-arm P1 execute its complete
schedule with valid provenance within budget? The primary M4-P0 pass criterion
is a passing two-arm runtime receipt and a complete affordable P1 forecast.
Identity, finite-value, all-chain movement, archive, graph, memory, and budget
failures veto readiness. Compile/steady timing and allocator values explain
failure; they are not sampler-ranking criteria. No convergence, posterior,
whitening, superiority, or default-readiness claim is authorized.

The campaign cap stays **5,200 seconds**, with **1,832.61 seconds** of historical
estimated spending and **3,367.39 seconds** nominally available before this
continuation. One diagnostic may use at most the remaining balance, with a
60-second termination/settlement reserve (a convenience safety allowance, not a
scientific tolerance). Its initial workload deadline is therefore **3,307.39
seconds**. An external timeout bounds a noninterruptible compiled call; signal
failure or forced termination is recorded and settled against the same ledger.
No automatic second diagnostic or P1 launch occurs without checking remaining
funds. A failed or partial diagnostic is never promoted to a complete forecast.
The closed factor campaign's budget is not transferred.

Outputs are fresh children of
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/`.
Keep raw NVIDIA inventory, selection rationale, source/plan hashes, Python and
conda environment, seeds, commands, elapsed time, allocator data, sample archives,
and shared-ledger settlement. CPU unit tests intentionally hide GPUs and cannot
establish runtime readiness.

## Execution order

1. Implement and test the placement helper and both launcher integrations;
   refresh the source-only P1 audit. Do not change unrelated GPU defaults.
2. Run trusted selection and the bounded readiness diagnostic in the `tfgpu`
   environment, with a fresh output directory and an external wall-time bound.
3. If measurements pass, compare the complete schedule with the remaining ledger
   and per-arm caps. Only an affordable schedule can issue the M4-P0 closeout.
4. Execute P1 only after that closeout, with fresh tuning and current placement.
   Otherwise record the actual blocker and the smallest justified next action.
   Additional compute or a scientific contract change requires owner direction.

Pre-mortem: most of the old 1,423-second chunk may be runtime rather than
compilation. The diagnostic could exhaust its allocation before measuring both
arms; that is budget-limited readiness evidence, not rejection of either sampler.
Conversely, a fast measurement could miss slow posterior modes: P2 and its
downstream reference/uncertainty requirements remain independent.
