# Stan NUTS as a TensorFlow Probability transition kernel

Status: reviewed before implementation; implementation and bounded engineering
validation completed. See the
[result](bayesfilter-stan-tfp-interface-result-2026-10-27.md).

## Question and scope

The owner asks for an equivalent for TFP after the direct TensorFlow/XLA
Stan port. We interpret this as a TFP `TransitionKernel` using the audited
Stan transition and its existing warm-up, compatible with `sample_chain`.
The direct port already uses TensorFlow; changing the interface is not a
performance optimization. Stock TFP NUTS with Stan adaptation remains the
separate faster comparator on the smaller recorded targets.

Keep this implementation in the authorized standalone `docs/benchmarks`
scope. Do not change public sampler defaults, retired routes, Stan tree
selection, numerical adaptation, or the existing benchmark results.

## Implementation sequence

1. Inspect installed TFP 0.25 `TransitionKernel`, `sample_chain`, and random
   seed contracts alongside the direct port.
2. Add a TFP kernel that delegates to the existing `stan_nuts.Kernel`, with
   structured results, fixed metric, step size, explicit seed handling,
   numerical validity flags, and stable-signature XLA numerical functions.
   Expose the existing compiled Stan warm-up through a checked bridge;
   discard its draws from retained sampling.
3. Test direct-transition agreement, unit/diagonal/dense metrics, explicit
   and continuation seeds, TFP sampling/thinning, adaptation handoff, invalid
   inputs, and compilation of the enclosing sampling loop. Document limits.
4. Record test outcomes and distinguish interface correctness from posterior
   and performance evidence. Review the final diff against this scope.

## Evidence contract and research intent

Main question: can TFP drive the source-audited transition without changing
its results or losing continuation/health information? Mechanism: a thin
interface sharing the numerical authority. Baseline: the committed direct
Stan kernel at e38348bee, with identical target, state, metric, epsilon and
per-chain stream states. Primary pass criterion: matching transition states,
telemetry and RNG consumption, plus successful enclosing XLA compilation
and TFP continuation. No posterior or speed promotion is sought.

Promotion vetoes: differing discrete decisions, random-stream overlap, lost
validity, unsupported seed semantics, adaptation in retained sampling, or a
non-XLA numerical fallback. These trigger local repairs. Continuation vetoes:
an irreparable core discrepancy, damaged prior evidence, exhausted budget,
or a required change to sampler semantics. Runtime and trace counts are
explanatory diagnostics only. Existing native-oracle evidence supports the
shared core, not untested wrapper paths.

## Assumptions and skeptical pre-run audit

- TFP interface, not a replacement tree algorithm: inferred from the preceding
  discussion. A wrapper cannot explain away or fix measured execution cost.
- FP64, `[chains, dimension]` independent rows, unit/diag/dense inverse metrics:
  inherited from the audited port. General nested TFP states and mixed dtype
  support are outside this implementation; reject instead of silently coercing.
- Frozen-kernel sampling: Stan warm-up remains a separate shared authority.
  TFP burn-in alone is not Stan adaptation. Verify handoff and continuation.
- Seed policy is material: TFP supplies a new seed per transition, whereas
  the direct port carries `(stream key, counter)`. Explicitly define both
  paths and verify independent chain keys; do not offset counters under a
  common key or silently ignore supplied TFP seeds.
- TFP convention `is_calibrated` describes the fixed kernel's correction,
  not convergence. XLA validity flags require host checking.
- CPU-only verification is an explicit small reference/engineering exception,
  using `CUDA_VISIBLE_DEVICES=-1` before framework import. It cannot select
  the execution backend or establish GPU performance.
- Test sizes/seeds are deterministic convenience fixtures, not posterior
  sample sizes or scientific defaults. FP64 comparisons use the established
  native-port test tolerance where needed; exact integers/booleans/RNG states.

Audit verdict: proceed with the interface. The wrong-baseline risk is avoided
by comparing against the existing port under common inputs; speed claims use
only the prior qualified result. Proxy compile/smoke metrics cannot promote
posterior validity or performance. No new benchmark campaign is authorized
by this plan.

## Bounded engineering validation

Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, TF 2.20 / TFP 0.25.
CPU hidden-GPU launch variables: `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `BAYESFILTER_PRELOAD_CUSTOM_OP=0`.
Use explicit single-thread TF settings as in the previous reference tests.
Commands: focused pytest for `tests/test_stan_nuts_tfp.py`, plus unchanged
standalone-route regression tests. Store exact commands, source hashes, git
commit, versions, wall time and logs in fresh attempt directories beneath
`docs/benchmarks/artifacts/stan-tfp-interface-20261027/`.
Budget: at most 4 test attempts and 600 cumulative subprocess seconds. These
are convenience-chosen engineering limits, not scientific cutoffs. Local
wrapper repairs consume this budget. Stop rather than start a speed or
posterior campaign. Result note goes beside this plan.
