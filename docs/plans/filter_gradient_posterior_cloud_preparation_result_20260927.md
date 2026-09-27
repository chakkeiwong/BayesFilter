# Posterior cloud preparation result

The compiled preparation program reproduces every normal/ball draw and
permutation key exactly at D1/D3, including changed seeds/radii and exact replay.
It uses `geometry_tf_philox_cpu_xla_v1` through the existing normal/ball kernels.
All draw loops and cloud assembly execute in TensorFlow/XLA. Outputs stay on CPU
when the caller's device context is GPU, one trace and HLO are retained across
changed inputs, and returned derivatives are frozen. No target callback is
accepted by this preparation interface.

04380 caught an implementation error: the candidate used minimum_uniform=0.0
for curvature as well as movement. The original posterior `_sample_ball` uses
0.25; restoring that curvature-only argument repairs the exact comparison.
The failed source and test are preserved. CPU 04381 passes both dimensions;
GPU-visible 04382 passes the same exact comparisons and verifies CPU placement.
04382 is CPU generation invoked from a GPU workload, not a GPU RNG substitute.
The byte-verified 031692a0b reference closure and input/output hashes are saved.

Policy 04383 passes 141 checks. Three exact fixed-schema allowances cover only
configuration-time attempt/partition/role hashing in `posterior_seed_keys`.
They permit no numerical draws, target evaluations, sample iteration or numerical
decisions. The compiled generator itself is fully guarded. The unit uses three
CPU workers / 29.307885 seconds and one GPU-visible worker / 10.438007 seconds
inside the four-CPU/two-GPU, 1,200-second reservation. The receipt
`posterior-cloud-preparation-verification-04383.json` reopens and verifies its
raw attempts, source snapshots and checksums.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Retain preparation dependency | Exact original draws/keys, CPU placement and stable compilation pass | Full initializer composition untested | Integrate the enclosing public program | Whole-initializer qualification |
| Preserve the failed attempt | Curvature radial setting error localized directly in source | Pre-fix outputs are unusable | Use only corrected generator for integration | Any waiver of exact stream equivalence |
| Keep metadata allowances narrow | Only static key hashing is allowed on host | Wider runtime audit still required | Retain exact AST enforcement | Permission for numerical Python loops |

Skeptical review: a statistically similar stream would have passed a weak
distribution test while changing the supplied-seed algorithm. Exact equality
caught the wrong radial distribution before integration. The remaining weak
boundary is the complete exported initializer, including the default two-factor
fit, result reporting and original derivative behavior. No scientific or HMC
claim follows from this preparation test.
