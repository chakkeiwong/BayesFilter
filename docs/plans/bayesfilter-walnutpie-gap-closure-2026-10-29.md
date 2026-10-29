# Walnutpie initialization, run control and R2P gap closure

## Question and authorized scope

Can the standalone TF/GPU/XLA Walnutpie port provide the pinned C++
initialization choices and stopping rules, plus the paper's randomized two-point
(R2P) transition, without a host sampler callback or changing public inference
policy? The owner explicitly requested planning, review and execution. This
standalone scope does not restore a repository NUTS comparator/default route.
Prior deterministic binaries and evidence remain immutable.

Sources: Walnutpie 5710b0334ba69cf0caa35fc02a2338b61d756953 (config.hpp,
util.hpp, adapt.hpp, sampler.hpp); JMLR 27 paper 25-1452, sections 3.1–3.4,
4.1/4.3, Appendices A–C; author Python source at
895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a, WALNUTSpy/adaptiveIntegrators.py
and WALNUTS.py. Paper and author source are preserved under .localresources.
No newer moving branch defines the target after this pin.

## Research intent and evidence contract

| Role | Contract |
|---|---|
| Main question | Close API/controller gaps and implement correct R2P numerical weights within full TF/XLA computations |
| Mechanism | XLA initialization options, compiled stopping statistics, shared tree traversal with R2P edge probability corrections |
| Exact baselines | Existing deterministic core plus compiled pinned C++ initialization/controller formulas; author R2P macro integrator and independent paper-equation tree reference |
| Primary pass criterion | Source-anchored numerical/discrete parity, negative controls for missing probability weights, continuation and padding correctness, full GPU XLA execution |
| Promotion veto | Incorrect cache/state, weight, RNG consumption or stop logic; nonfinite retained state; callback fallback; failed numerical comparisons |
| Continuation veto | Wrong mathematical target unresolved; missing author/paper sources; invalid launch/memory policy; exhausted total compute budget |
| Repair trigger | Compile/shape/serialization failure or localized invariant failure: fix and rerun within budget |
| Explanatory diagnostics | Runtime, work, compiler size/memory, short funnel behavior; never posterior or superiority evidence |
| Nonconclusions | Universal equivalence, same RNG streams, asynchronous timing equivalence, funnel convergence, ESS/sec superiority, public-default readiness |
| Evidence | Versioned run manifests, source hashes, checks, HLO, logs and result note under the paths below |

## Implementation sequence

1. Add initialization choices: supplied or zero/normal positions; supplied/unit/
   score-based masses and optional geometric mass averaging; supplied steps or
   upstream fixed-momentum initial search. Keep the existing bounded search as
   default, add an explicitly selected uncapped source mode; nonfinite inputs
   remain invalid. Do not pretend Gaussian starts are Stan constrained-model
   initialization.
2. Add synchronized batched run control with the C++ warmup relative step/mass
   agreement rule, minimum/maximum counts, and the exact legacy log-density
   R-hat stopping formula. Compile numerical stopping and online moments with
   transitions. Return actual counts/masks and distinct convergence/cap/invalid
   reasons. Support checkpoint continuation. Wall-clock deadlines, interrupts
   and callbacks stay outside XLA at bounded chunk boundaries, explicitly a
   different scheduling implementation from C++ asynchronous workers.
3. Add optional R2P: critical exponent search, probability 2/3 at c and 1/3 at
   c+1 (2^c and 2^(c+1) microsteps), reverse probability, and cumulative log
   correction at both endpoints. Biased progressive tree weights are
   joint_log_density + cumulative_log_reverse_over_forward_probability.
   Preserve geometric endpoints even when their weights become zero. Include
   optional per-macro uniform +/-20% jitter, sharing the chosen interval in its
   forward/reverse checks. Default deterministic behavior remains unchanged.
4. Qualify against source and independent references on CPU XLA, then trusted
   GPU XLA; verify full-fit composition, runtime data, no callbacks, continuation,
   caps, RNG/weight edge cases and nonfinite handling. Retest deterministic core.
5. Record results, limitations, examples and terminal skeptical review.

## Default and assumption audit

| Choice | Provenance / status | Failure risk and early check |
|---|---|---|
| FP64, TF/TFP, stable signatures, XLA; GPU execution | Repository and existing port baseline | CPU-only compilation misreported as GPU; inspect device/HLO and verified growth |
| Existing fieldwise deterministic default | Protected baseline | Changed arithmetic/RNG; rerun existing tests and native oracle |
| Initial search cap 64 | Existing engineering safety default | Rejects otherwise source-valid long search; explicit uncapped mode and tests, no silent clipping |
| Zero/unit or Gaussian start scale supplied by caller | C++ InitConfigBuilder | Invalid support or model mismatch; explicit validity, no automatic retry distribution |
| Warmup min/max 50/1000, step tol .1, mass tol 1, publish stride 5 | Pinned C++ defaults, compatibility only | Agreement is not convergence; C++ snapshot parity and no posterior promotion |
| Sampling min/max 50/1000, logp R-hat <=1.01 | Pinned C++ defaults, compatibility only | Log density can hide separated modes; label legacy diagnostic, test pathological case |
| Chunk length 5 | Convenience matches warmup publication stride | Deadline overshoot; record/check only between synchronized chunks, never promise hard real-time cancellation |
| R2P probability 2/3, jitter .2, exponent cap 10 | Author Python defaults, source-compatibility hypotheses | Missing +1 support or different cap behavior; test c=max and reverse-zero cases |
| Paper endpoint criterion | Remark 13 and author code | All intermediate energies must remain finite; explicitly check |
| R2P warmup | Optional Walnutpie Adam/metric warmup is a local combination | Not Appendix C reproduction; label it; support supplied fixed step/mass with zero warmup |
| rtol/atol 2e-11 for local FP64 parity | Existing qualified oracle tolerance | Accumulated chaotic paths amplify roundoff; separate local versus full-path evidence |
| Three test seeds (17,29,43) | Convenience engineering fixtures | Not posterior uncertainty evidence; do not rank samplers |

R2P support is the transition in the paper, not a claim to reproduce its complete
experimental tuning protocol. The author code's capped critical search is a
well-defined deterministic selector (falls back to maxC); the random finer
branch can use maxC+1. Record cap hits and distinguish this from establishing
that the error threshold was attained. Nonfinite trajectories cannot supply
positive weights. Diagonal-metric extension follows the same kinetic energy and
velocity convention as the existing port; identity-metric author parity is the
first test. Walnutpie warmup for R2P is explicitly a local optional combination;
minimum microsteps stay one and h/metric freeze for retained draws. Appendix C
quantile tuning and the paper's long posterior experiments are separate from
this implementation request; do not advertise their reproduction.

## Skeptical review before execution

Review found and repaired three material risks:
- A mere randomized microstep count omits equations (13)–(14). Store cumulative
  corrections separately from physical Hamiltonians and test unequal forward/
  reverse probabilities and zero support through tree selection.
- The author Python WALNUTS.py backward two-leaf branch appears to omit adding
  the second leaf's log correction before its weight (around lines 453–462).
  Copying that code would contradict equations (13)–(14). Use unmodified author
  integrator parity plus an independent equation-level whole-tree reference;
  add a direct negative-control test for this omission. No claim of identical
  author whole-chain RNG paths.
- C++ controller agreement uses asynchronous snapshots and logp-only R-hat,
  not modern parameter diagnostics. Test the same snapshot formulas; use
  synchronized device iterations and explicitly report the schedule difference.

A clock/callback cannot be placed in a pure XLA numerical graph without losing
that contract. The host controller therefore only schedules compiled chunks,
checks clock/cancellation and emits records. It never evaluates the model,
performs sampler steps or computes numeric stop statistics in Python.

Review passes for implementation after these revisions. This is an author
skeptical review, not independent-agent review. No approval chain is required.

## Execution budget and commands

New campaign ceiling: 3600 cumulative worker seconds, at most 18 launches,
including repairs. These are convenience cost limits matching the prior scoped
campaign, not convergence thresholds. Individual jobs limited to remaining
budget; start with sub-minute mechanics tests. Do not extend the budget silently.
Versioned root: docs/benchmarks/artifacts/walnutpie-gaps-20261029/attempt-NN.
Result: docs/plans/bayesfilter-walnutpie-gap-closure-result-2026-10-29.md.
Use /home/ubuntu/anaconda3/envs/tfgpu/bin/python; CPU jobs explicitly hide GPU;
GPU jobs require trusted permissions, TF_FORCE_GPU_ALLOW_GROWTH=true and verified
per-device growth before initialization. No new package/environment installs.

Planned driver: python -m docs.benchmarks.walnutpie_gpu_tf.gap_master
--phase tests|validate|deterministic --device cpu|gpu --output <fresh attempt directory>.
Each manifest records commands, source pins/hashes, git base, environment,
seeds, TF32/XLA, device/memory provenance, wall time and failures. No performance
ladder or long posterior run is authorized by this engineering budget.

## Completion audit amendments

The implementation audit additionally found invalid supplied zero/negative steps
could leave uncapped search running, an already-exhausted controller checkpoint
could make no progress indefinitely, and R2P inherited an initial minimum greater
than one for a nondefault target-macro setting. These are execution defects, not
evidence against R2P. Gate searches on finite positive inputs, terminate exhausted
checkpoints, guard host scheduling progress, and force R2P minimum one at both
initialization and adaptation. Explicitly handle all-zero-weight subtree selection
without undefined negative-infinity subtraction. Regression tests precede final
CPU/GPU parity. These repairs retain the original target and budget.

Final validation must rerun the existing 528-record deterministic C++ oracle after
the shared-core edits. Use the tracked prebuilt executable read-only and record
its hash; do not rebuild or replace the existing package. Runtime code changes
after a passing validation require requalification; documentation-only edits do
not change the qualified numerical code.

The final boundary pass also checks reciprocal-mass finiteness, preserves invalid
RNG-key status before random-start advancement, and supports explicitly zero
warmup control limits without a transition. These are engineering edge fixtures,
not additional posterior experiments.

## Completion / reset note

Completed on 2026-10-29. Initialization, synchronized run control, optional R2P,
source audit and terminal self-review are complete for this plan. Final checks:
48 unit tests, 504 R2P macro/tree comparisons per backend, and 528 deterministic
C++ records per backend. All runtime hashes match final qualification snapshots.
The result note records exact source differences, commands, budget and evidence.
The prebuilt package is unchanged. No posterior or speed ranking is inferred,
and R2P Appendix C quantile tuning remains outside this transition-port scope.
No further campaign phase is pending.
