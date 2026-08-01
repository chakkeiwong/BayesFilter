# Predator-Prey And Generalized-SV FD Root-Cause Repair Subplan

Date: 2026-07-11

Status: `ITERATION_5_GPU_CONFIRMATION_READY`

Time bound: eight hours maximum from goal creation.

## Objective

Identify the root causes of the remaining predator-prey and generalized-SV
finite-difference failures, repair confirmed bugs, and repeat the smallest
discriminating diagnostic until both rows have a defensible explanation and
the relevant bugs are fixed.

The owner-directed pass rule remains FD-only:

```text
r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)
pass iff max_j(r_j) <= 0.05 * sqrt(p)
```

This plan may repair how FD evidence is measured or a confirmed derivative
implementation bug. It may not relax the tolerance, change the target scalar,
post-select a favorable result, or promote an FD pass into HMC or posterior
validity.

## Research Intent Ledger

| Field | Intent |
| --- | --- |
| Main question | Are the two failures caused by manual derivative errors, value/score route mismatch, fixed-step float32/TF32 resolution, or a nonsmooth branch? |
| Candidates under test | Predator-prey Gate B `T=1,N=2`; generalized-SV Gate C `T=4,N=10000`, with smaller exact-route fixtures used only to localize derivative math. |
| Expected failure modes | Manual JVP differs from autodiff; score and value objectives differ; central endpoints collapse or are quantized at `h=1e-4`; FD does not form a stable step plateau; XLA/TF32 changes the result relative to FP32-no-TF32. |
| Promotion criterion | For each row, a repaired production-policy FD computation passes the owner rule and is supported by an independent derivative comparator or a stable, predeclared step plateau. |
| Promotion veto | Wrong scalar/inputs/seed/branch, nonfinite output, manual JVP/autodiff mismatch after repair, no stable FD region, endpoint evidence missing, or original source artifact mutation. |
| Continuation veto | The diagnostic harness cannot reproduce route identity, artifacts are corrupted, or the eight-hour bound is reached. A candidate failure alone is not a continuation veto because later iterations are explicitly repair phases. |
| Repair trigger | A localized manual derivative mismatch, fixed-step endpoint collapse/quantization, or policy code that cannot record coordinate-specific steps and endpoint values. |
| Explanatory only | A single FD step, objective ULP counts, raw endpoint differences, runtime, and tiny-shape behavior without independent derivative support. |
| Must not be concluded | No HMC readiness, posterior correctness, full score admission, default readiness, runtime superiority, or calibrated confidence interval follows from this work. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Exact baseline | The current fixed-randomness value-only scalar and current compact manual JVP at identical theta, prepared tensors/noise, transport settings, seed, precision, and shape. |
| Primary correctness comparator | Before transport can feed a later likelihood, `tf.GradientTape` of the identical value-only graph is valid. After transport, use `tf.GradientTape` through the unchanged raw forward transport with `transport_ad_mode=full`, plus central FD; the production value route's stabilized backward stops are deliberately not the total derivative. Autodiff remains reference-only and cannot become the production score route. |
| Production comparator | Central FD in GPU/XLA float32/TF32 with coordinate-specific predeclared steps, recording `f(theta+h)`, `f(theta-h)`, their difference, effective rounded step, and FD value. |
| Primary pass criterion | Manual JVP agrees with reference autodiff on smooth bounded fixtures; production FD has a stable step region and passes `max_j(r_j) <= 0.05*sqrt(p)`. |
| Veto diagnostics | Objective mismatch, parameter-order mismatch, random-input mismatch, branch/floor instability across the selected step neighborhood, or source-hash mutation. |
| Explanatory diagnostics | FP64 FD ladder, float32-no-TF32 arm, objective ULP ratio, per-component intermediate JVP comparisons. |
| What will not be concluded | A tiny reference pass alone does not establish production FD correctness; a production FD pass alone does not prove the manual JVP. |
| Preserved artifacts | Structured JSON and Markdown under `docs/plans/artifacts/ledh-predator-generalized-fd-root-cause-repair/`; original Phase 9 shards remain unchanged. |

## Skeptical Plan Audit

The old baseline `h=1e-4` is not a justified universal finite-difference step.
It is especially weak for physical parameters `K=114` and `a=25`, and the
stored predator-prey FD values lie on a coarse float32 output-quantization
grid. Generalized-SV likewise repeats the exact FD value `-0.0190734863` in two
directions. Therefore the old FD result is a repair trigger, not sufficient
evidence that either manual derivative is wrong.

A larger step that merely produces a passing number would also be insufficient.
The plan requires either reference autodiff agreement plus a stable step region,
or localization and repair of a derivative mismatch. Runtime, memory, and one
favorable step remain explanatory. The target, fixed randomness, transport
branch, parameter coordinates, and owner tolerance remain fixed.

The first run is CPU/FP64 and bounded because it can distinguish derivative
math from production rounding cheaply. GPU work starts only after the
diagnostic harness and Iteration 1 CPU evidence are recorded. All GPU/CUDA
commands require trusted/escalated execution and new artifact paths.

Audit decision: `PASS`. The commands below answer the stated questions and
have explicit repair triggers and stop conditions.

## Iterations 1-4 Execution Update

The first four iterations found and repaired two independent bugs:

1. Predator-prey's stored zero FD for `a` was a float32 objective-resolution
   failure caused by applying the absolute step `1e-4` to every coordinate.
   FP64 manual JVP, full autodiff, and a stable relative-step ladder agree.
2. Generalized-SV's manual transport JVP replayed `max_iterations` annealing
   updates, while the unchanged raw forward transport executes at most
   `max_iterations - 1` because its loop condition is `i < max_iter - 1`.
   The repair makes the manual finite-Sinkhorn sensitivity use the same bound.
   At caps `1`, `2`, and `10`, the repaired manual score agrees with
   full-transport autodiff to about `1e-15` on FP64 fixtures.
3. The shared GPU FD harness now uses
   `cbrt(float32_epsilon) * max(1, abs(theta_j))`, divides by the actual rounded
   endpoint separation, and records both parameter and objective endpoints.
4. CPU-hidden float32/XLA debug replays pass the unchanged owner FD-only rule:
   predator-prey max relative error `0.0005032690 <= 0.1224744871` and
   generalized-SV max relative error `0.0036914167 <= 0.0866025404`.

The CPU/XLA replays are engineering diagnostics only. Production confirmation
still requires the exact trusted GPU/XLA/TF32 commands in
`docs/plans/ledh-predator-generalized-fd-root-cause-repair-gpu-commands-2026-07-11.json`.

## Iteration 5 Exact GPU Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do the repaired score and coordinate FD policy pass on the original failing GPU/XLA/TF32 shapes? |
| Comparator | Same prepared-input fingerprint, score shard, and value-only FD shard for seed `81120`. |
| Predator-prey shape | `T=1,N=2`, Sinkhorn cap `10`. |
| Generalized-SV shape | `T=4,N=10000`, Sinkhorn cap `10`. |
| Primary criterion | Every endpoint is finite and noncollapsed, stored endpoint arithmetic validates, and the owner maximum-coordinate FD rule passes. |
| Vetoes | Wrong device/XLA/TF32 provenance, hash/input mismatch, nonfinite or collapsed endpoint, failed reconstructed FD arithmetic, or failed owner threshold. |
| Explanatory only | Runtime, memory below the existing prefix screen, raw endpoint magnitude, and differences from CPU fixtures. |
| Must not be concluded | No HMC readiness, posterior correctness, full-time admission, statistical superiority, or default-readiness claim. |
| Artifact | Four new v4 shards under `docs/plans/artifacts/ledh-predator-generalized-fd-root-cause-repair/`; historical Phase 9 shards remain immutable. |

Skeptical audit: `PASS`. These are the smallest production-target commands
that reproduce the two original failures after both localized repairs. They do
not change the target, seed, transport policy, tolerance, precision default, or
historical evidence, and their endpoint records answer the stated FD question.

## Iteration 1: Reference And Resolution Localization

1. Add a diagnostic harness that, for both rows:
   - prepares inputs/noise once;
   - evaluates the manual JVP and value objective at identical fixed inputs;
   - computes reference autodiff of the value-only graph on CPU/FP64 tiny
     fixtures;
   - computes a central-FD ladder with endpoint values and effective rounded
     steps;
   - records branch-sensitive/floor diagnostics where available.
2. Use the predeclared relative central-step ladder
   `c * max(1, abs(theta_j))`, with
   `c in {1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2}`.
3. Classify each coordinate:
   - `manual_jvp_bug` if FP64 manual JVP disagrees with autodiff beyond
     `rtol=1e-6, atol=1e-8` on a smooth tiny fixture;
   - `fixed_step_resolution_bug` if manual JVP agrees with autodiff and the FD
     ladder converges to it away from the collapsed/quantized small steps;
   - `nonsmooth_or_unresolved` if no stable region exists.

Iteration 1 fixtures:

- predator-prey: `T=1,N=2`, seed `81120`, no GPU claim;
- generalized-SV: `T=2,N=8`, seed `81120`, no GPU claim;
- both use float64, TF32 disabled, XLA disabled only as an explicit reference
  exception.

## Iteration 2: Localized Repair

- If a manual JVP bug is found, add the missing/corrected term, add a focused
  regression against reference autodiff, and rerun Iteration 1.
- If fixed-step resolution is confirmed, replace the scalar `fd_step` in the
  shared FD harness with a coordinate-specific policy. The initial candidate is
  `cbrt(float32_epsilon) * max(1, abs(theta_j))`; its acceptance requires that
  it fall inside the predeclared stable region for every coordinate. Record
  endpoints and effective steps in all future FD artifacts.
- If no stable region exists, localize by disabling transport at a tiny fixture,
  then add flow, weights, and transport back sequentially. This localization is
  explanatory until compared with autodiff at each layer.

## Iteration 3: Production Confirmation

After code repair and focused CPU tests:

1. Run trusted GPU/XLA float32/TF32 diagnostics with new output paths for the
   original failing shapes.
2. Predator-prey: `T=1,N=2`, seed `81120`.
3. Generalized-SV: first `T=4,N=10000`, seed `81120`; use a smaller GPU rung
   only if the full stored shape cannot run after a bounded implementation
   repair.
4. Require endpoint evidence, effective per-coordinate steps, finite outputs,
   correct GPU/XLA/TF32 provenance, and the owner FD-only pass rule.
5. If TF32 remains unstable but FP32-no-TF32 passes, classify TF32 numerical
   resolution as the remaining production bug; do not silently change the
   repository default without a separate evidence decision.

## Pre-Mortem

- The harness could compare different random samples. It must prepare and hash
  tensors/noise once per arm.
- Autodiff could traverse a different transport mode. It must call the exact
  value-only function used by FD with identical arguments.
- A large step could cross a nonsmooth max/min/floor branch. The ladder must
  expose non-monotone or discontinuous behavior; no isolated passing step may
  be selected.
- Tiny FP64 success could be overgeneralized. Production confirmation remains
  mandatory.
- A GPU rerun could overwrite Phase 9 evidence. All paths are new and original
  source hashes are checked before and after.

## Stop Conditions

- Stop successfully when both rows have an identified root cause, confirmed
  bugs are fixed, focused regression tests pass, and production confirmation
  either passes or identifies a remaining default-policy numerical limitation
  without mislabeling it as a derivative bug.
- Stop at eight hours with an explicit unresolved-hypothesis ledger if the
  success condition is not reached.
- Stop immediately for source-artifact mutation, target/seed mismatch, corrupt
  evidence, or nonfinite reference arithmetic.
