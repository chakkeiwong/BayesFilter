# Continuation after the supplied residual-map experiment

Status: reviewed next-phase design; execute after the M28 terminal result and
budget reconciliation. M28's numerical results belong in its result note, not
in this prospective design. Its archive, count and public-pipeline checks are
not reasons to rerun successful fits. Learning a map remains upstream of tuning.

## Research intent and remaining requirements

The immediate question is which parts of a complete ordinary tuning and
posterior fit consume the measured GPU time, and whether the measurement can
be collected without changing the numerical experiment. M27 prices put even
one 384-fit Gaussian confirmation inventory at about 69.49 GPU hours. That is
larger than the remaining campaign allowance. Neither smaller pilots nor
successful M28 posteriors establish general stopping coverage.

A concrete profiling gap must be repaired before collecting evidence:
`execution.worker` currently starts `cProfile` in the coordinator. With
`isolate_fits=True`, the numerical work happens in `fit_process.fit_worker`,
so this profile mainly measures waiting. It cannot identify numerical-worker
costs. The existing plain-worker profile remains valid for its own process.

| Diagnostic | Role and decision |
| --- | --- |
| Exact numerical replay between unprofiled and profiled arms | Primary engineering criterion: candidates, observations, receipts' numerical contents, archived tensors and posterior decisions must agree, excluding declared clocks/paths |
| Actual child-process profile containing tuning and retained sampling calls | Primary instrumentation criterion; coordinator wait time cannot substitute |
| Source/design mismatch, lost evidence, abnormal exit or wrong GPU policy | Continuation veto for the affected comparison until repaired |
| Posterior health/readiness/precision | Posterior promotion criteria only; a failed member does not invalidate profiling or remove a verified candidate |
| Per-stage elapsed time, call counts, graph counts, RSS and GPU allocator peaks | Explanatory diagnostics; nominate a bottleneck, never establish superiority from one pair |
| Exhausted phase or total campaign allowance | Continuation veto; preserve every failed or unfinished cell |

## M29: measure the numerical worker

Allocate at most **1800 CPU and 4800 GPU worker-seconds** from the terminal M28
balance, with at most two numerical workers. This is a convenience development
ceiling, not new compute authority. Four GPU attempts of at most 1000 seconds
each leave 800 seconds for readiness and localized infrastructure repair;
child limits are 980 seconds. These inherited margins remain hypotheses.
Charge setup failures, abnormal exits and retries exactly once at the outer
worker boundary. Use fresh directories under `m29-r1/`.

1. Move optional isolated-fit profiling into the numerical child, with a
   profile per child attempt and a `finally` path that preserves useful
   diagnostics on failure. The coordinator must record where the child
   profiles are, rather than label its wait profile as numerical execution.
   Keep profiling off by default. Preserve source/design identity, random
   streams, receipts, restart behavior and TensorFlow graph/XLA policy.
2. Test the instrumentation using an actual small public ordinary Gaussian fit
   and a beta-binomial fit. Check that the profiles contain numerical pipeline
   calls, that failure paths preserve the profile without masking the original
   exception, and that resume cannot silently skip a requested missing profile.
   A repeated completed numerical fit must not be rerun merely to fabricate
   missing historical profiling evidence; report its profile as unavailable.
3. Freeze two paired GPU designs, Gaussian and beta-binomial. Each pair uses
   the same numerical inputs, source, starts, seed and tuning/posterior controls
   with profiling off/on. Use new convenience root seeds **2026092391** and
   **2026092392**, respectively. Transcribe the complete numerical design from
   M27's `m27-gpu-gaussian-lugsail-2026092283.json` and
   `m27-gpu-beta-binomial-lugsail-2026092284.json`. Preserve each model's own
   counts and settings, including lugsail and its fixed-count comparator;
   explicitly enumerate differences for seed, design name, process timeout
   and profiling. This is a replay/profiling experiment, not a new adequate
   posterior allocation. Compare all resolved count fields with an explicit
   design table before launch, as in M28. Do not pool sibling, replay or
   profiling arms as independent fits.
4. Validate the suite and freeze a committed-base-plus-owned-changes snapshot.
   Run the public validation CLI with isolated children, serial GPU jobs,
   trusted access, memory growth and XLA. Record profiling overhead separately
   from unprofiled elapsed time. Attribution from `cProfile` includes host
   overhead and synchronization; it is not a device-kernel timing trace.
5. Audit numerical parity and evidence integrity. Report actual stage costs,
   profile completeness, graph counts and allocator peaks. Identify one
   measured bottleneck if the profile supports one; otherwise write the
   smallest additional diagnostic that would distinguish compilation,
   execution, posterior diagnostics and serialization. No broad refactor or
   non-XLA fallback is authorized by a slow profile alone.
6. Reprice the unchanged confirmation inventory using unprofiled runs and the
   existing uncertainty/power requirement. Any later optimization must be
   tested under its own behavior-preservation plan before new timing is used
   for an affordability decision. Finish the phase with cost reconciliation,
   a terminal result and an executable next phase.

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; CPU smokes hide all GPUs
before import. All serious commands, source hashes, device/memory policy,
seeds, timing and result paths go into attempt manifests. A fresh directory is
required per launch. The terminal M28 ledger is the only opening budget.

## Skeptical audit and default provenance

The review rejects a coordinator-only profile as a wrong baseline for numerical
cost attribution. It also rejects comparing different seeds or design-derived
streams as exact replay. Inspect how the profiling option participates in
identity before launch; if the option changes numerical streams, repair the
instrumentation boundary or use one identical scientific design with profiling
declared separately in the execution manifest. Do not weaken the parity check.

The M27 controls are inherited target-specific development baselines. Their
successful posterior checks establish neither coverage nor timing upper
bounds. The two new seeds are convenience identifiers, not optimized choices.
Profiling overhead may increase runtime enough to hit a cap; preserve that
failure as instrumentation affordability evidence. The smallest early check
is the CPU pair, followed by validation of the concrete designs. One pair per
model can locate engineering cost, but cannot support a speed ranking or new
scientific/default claims. These restrictions make the proposed development
phase informative even when no optimization is justified.

## Subsequent program

| Remaining gap | Required continuation | Closure boundary |
| --- | --- | --- |
| Residual-map delivery across settings | Classify M28 member failures, if any, using health, model-coordinate precision and budget evidence; repair confirmed implementation errors before changing a map or allocation | Success for amplitude .5 and two seeds is a tested fixture, not universal partial whitening or learned-map quality |
| General warmup/MCSE/stopping calibration | Reduce a measured cost without changing the experiment, or derive and independently calibrate an affordable statistical design before confirmation | Current adequate full-fit inventory remains under-budgeted; do not relabel a smaller pilot |
| Global exploration | Fresh same-target multimodal fits, predeclared mode occupancy and crossings, preserved start banks and independent reference uncertainty; keep numerical and global criteria separate | Existing zero-variance mode-quantity tests are engineering checks, not proof that unknown modes are discoverable |
| Subtle full-fit defects | Source-grounded design and fresh null/power calibration before confirmation | Extreme injected defects and frozen-kernel tests do not establish sensitivity to subtle adaptive-fit errors |
| Exact MacroFinance integration | Recheck availability of matching target/source, data, prior, coordinates and independent uncertainty-bearing reference inputs | Synthetic models cannot replace missing consumer evidence |
| Maintenance and throughput | Profile the child; change one established bottleneck; require numerical replay, resource and integration checks | Graph caching and extraction are proposals until measured and audited |

Candidate rejection and research-direction rejection remain separate. A
posterior cap or failed global exploration screen normally motivates the next
discriminating repair; it does not change tuning membership or authorize
threshold relaxation. No launch token, per-retry approval or review chain is
required within the unchanged scientific scope and remaining owner budget.
