# Reset Memo: Predator-Prey And Generalized-SV FD Root-Cause Repair

Date: 2026-07-11

Status: `GOAL_COMPLETE_BOTH_ROOT_CAUSES_FIXED_AND_VALIDATED`

## Read First

The authoritative result is:

- `docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-result-2026-07-11.md`
- result SHA-256:
  `42630b9ab97cdcb39d4ecd8c0fdc172647a63b86c5c3a478fd8efd23352f1fed`

The reviewed execution design and exact trusted GPU commands are:

- `docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-subplan-2026-07-11.md`
- `docs/plans/ledh-predator-generalized-fd-root-cause-repair-gpu-commands-2026-07-11.json`

This memo is separate from
`docs/plans/bayesfilter-ledh-score-wiring-repair-reset-memo-2026-07-10.md`
because that shared memo has overlapping edits from another active lane. Do
not overwrite or revert those unrelated changes.

## Current State

The two historical FD failures are resolved at their original failing shapes.

| Row | Root cause | Repair | Trusted GPU result |
| --- | --- | --- | --- |
| predator-prey | Universal absolute `h=1e-4` under-resolved a float32 objective, especially for physical-scale coordinates `K=114` and `a=25`. The historical `a` FD was exactly zero because the two objective endpoints rounded to the same float32 value. | Use `h_j=cbrt(float32_epsilon)*max(1,abs(theta_j))`, form representable float32 endpoints, divide by their actual separation, and preserve both parameter and objective endpoints. | At `T=1,N=2`, maximum coordinate relative error is `0.000503381988322 <= 0.122474487139`; pass. |
| generalized-SV | The manual finite-Sinkhorn JVP replayed `max_iterations` updates although the raw forward loop condition permits at most `max_iterations-1`. It differentiated a different map. | `_manual_dense_finite_steps()` now returns `max_iterations-1`; this feeds dense, streaming, and blockwise manual finite modes. | At `T=4,N=10000`, maximum coordinate relative error is `0.00487298671266 <= 0.0866025403784`; pass. |

The pass rule remains FD-only:

```text
r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)
pass iff max_j(r_j) <= 0.05 * sqrt(number_of_parameters)
```

Predator-prey has six parameters, so its threshold is about `12.247%`.
Generalized-SV has three parameters, so its threshold is about `8.660%`.
Actual-SV's separate two-parameter `7.071%` threshold is not changed or
reinterpreted here.

The `0.0049215666...` appearing in the step formula is
`cbrt(float32_epsilon)`, a central-difference numerical step coefficient. It
is not the rejected arbitrary `0.005` error tolerance and is not a general
gradient or HMC tolerance.

## Confirmed Causal Chain

### Predator-Prey

1. The compact score was already nonzero for `a`:
   `-0.1401509791612625`.
2. The historical float32 FD at absolute `h=1e-4` was stored as exactly `0.0`.
   With nonzero parameter separation, this can occur only because the float32
   objective numerator was zero; the two endpoint objective values were equal.
3. On a CPU/FP64 `T=1,N=2` fixture, manual JVP, ordinary autodiff, and the FD
   ladder agree. For `a`, they are near `-0.11672239524`, so the mathematical
   derivative is not zero.
4. At the repaired GPU step, `a` uses actual half-step
   `0.12303924560546875`; its endpoint objectives differ by
   `-0.0345001220703125`, giving FD `-0.1401996612548828`.
5. All six production endpoint pairs are finite, noncollapsed, and validated
   by reconstructing their float32 arithmetic.

Classification: `fixed_step_resolution_bug`, repaired.

### Generalized-SV

1. The unchanged raw forward transport uses `i < max_iter - 1` in both dense
   and streaming Sinkhorn loops.
2. The former manual helper returned `max_iter`, so the manual JVP replayed one
   extra update.
3. Before repair, tight FP64 manual-versus-full-transport-autodiff comparisons
   failed at caps 2 and 10 in all three directions.
4. Changing only the manual update count to `max_iter-1` reduces discrepancies
   to roughly `1e-15` or less at caps 1, 2, and 10.
5. The original production shape then passes the same-scalar FD check with all
   three endpoint numerators nonzero.

Classification: `manual_transport_jvp_off_by_one`, repaired.

Ordinary stabilized `GradientTape` after transport is not a total-derivative
reference because its deliberate backward stops omit transport dependence.
Use full autodiff through the unchanged raw forward transport for bounded
reference checks. The generalized-SV pseudo-observation derivative was also
audited and is intentionally zero because `H*x + (y-H*x) = y`.

## Code State

The bounded repair touches these implementation/evidence surfaces:

- `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py`
  fixes the manual update count.
- `bayesfilter/ledh_fd_policy.py` defines the owner FD-only rule and the
  coordinate step policy.
- `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py` is schema v4,
  records endpoint evidence, and reconstructs stored FD arithmetic during
  validation.
- `docs/benchmarks/diagnose_ledh_predator_generalized_fd_root_cause.py`
  compares manual JVP, ordinary autodiff, full-transport autodiff, and a
  predeclared FD ladder on CPU/FP64 fixtures.
- `tests/test_ledh_compact_transport_jvp.py` binds public caps 1, 2, and 10 to
  zero, one, and nine manual updates.
- `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py` covers exact
  command authorization, source/provenance checks, endpoints, arithmetic, and
  tamper rejection.
- `tests/highdim/test_ledh_predator_generalized_fd_root_cause_diagnostic.py`
  binds the reference diagnostic to identical prepared inputs and value graph.
- `tests/highdim/test_ledh_phase9_fd_policy_reclassifier.py` preserves the
  corrected FD-only policy behavior.

Several model benchmark and contract files have overlapping uncommitted work
from the surrounding score-wiring lane. Preserve it. Do not use destructive
git commands or revert files merely to isolate this result.

## Final Trusted Artifacts

All four are terminal schema-v4 artifacts. Each records commit, dirty-worktree
disclosure, Python/TensorFlow environment, GPU/XLA/TF32 provenance, source and
governance hashes, exact command, seed, shape, and serialized prepared-input
fingerprint.

| Artifact | SHA-256 |
| --- | --- |
| `iteration5-predator-prey-t1-n2-gpu-xla-score.json` | `789218ad78a9cedd5e9393d60f72b024ff944f4c98f72c85feee883445ea70d8` |
| `iteration5-predator-prey-t1-n2-gpu-xla-fd.json` | `e6064fec5b9f5a444248d20b9342991d35907859ab58e783ea3c177d714f5bca` |
| `iteration5-generalized-sv-t4-n10000-gpu-xla-score.json` | `44c67898d63f47db23f1115c6bd48cff4c1645057bd1814ab728860841a1bf8f` |
| `iteration5-generalized-sv-t4-n10000-gpu-xla-fd.json` | `0605c2be019b5558d1f83eb71aea1fe93765b9fbccf7592b173a1fb185ba6163` |

Directory:

`docs/plans/artifacts/ledh-predator-generalized-fd-root-cause-repair/`

The paired prepared-input fingerprints are:

- predator-prey:
  `a4de81d972a3e4b445252b8fb73bb00b5df599980548c13ce3fee935bf421727`;
- generalized-SV:
  `d8e322dbaac08835fc0499c5161d77ba02777deeac2d4f26022e77ca3202c643`.

The final FP64 generalized-SV total-derivative reference is
`iteration3-generalized-sv-t2-n8-fp64-post-transport-repair.json`, SHA-256
`e69a2a79a212fea4c48710537abfc1fd7f62f1dbc53a4efa8f00ff852cabf610`.

The immutable governance hashes embedded in every final GPU shard are:

- subplan:
  `09a37a5de32289927f5daba72d5f59f037f586d86a19551709eeedf92b93f3e0`;
- exact command manifest:
  `b88cb82c114449b1c54a960ac02d608340381ef3b87c448bb7b07795a3ad8a9e`.

Historical Phase 9 shards remain unchanged. Do not overwrite them; they are
the before-repair evidence.

## Verification State

Final checks completed on 2026-07-11:

| Check | Result |
| --- | --- |
| Independent validation of all four final GPU shards using current schema-v4 validators | pass |
| Paired score SHA, score vector, and prepared-input fingerprint checks | pass |
| Endpoint finiteness/noncollapse and exact float32 arithmetic reconstruction | pass |
| Full compact GPU/XLA harness suite | `93 passed, 2 warnings in 96.96s` |
| Shared helper and all five nonlinear model/cross-model contract suites | `144 passed, 2 warnings in 305.33s` |
| Scoped Python compilation | pass |
| Scoped `git diff --check` | pass |

The final pytest commands intentionally set `CUDA_VISIBLE_DEVICES=-1` before
TensorFlow import. They are engineering regressions only. The production-target
evidence is the trusted visible-GPU artifacts, which used `/GPU:0`, float32,
TF32 enabled, and `jit_compile=True`.

## Result Boundaries

Correct conclusions:

- Predator-prey's historical zero `a` FD was a float32 step-resolution
  artifact, not a zero derivative.
- Generalized-SV's manual finite-Sinkhorn derivative was wrong relative to its
  claimed forward map because of one extra replayed update.
- Both bugs are fixed on the checked routes and original failure shapes.
- Both final FD comparisons pass the owner rule.

Unsupported conclusions:

- HMC readiness or acceptable HMC trajectories;
- posterior correctness or reference-posterior agreement;
- generalized-SV full `T=1008` admission;
- complete multi-seed score admission for either row;
- statistical or runtime superiority;
- calibrated 95% coverage;
- applicability of this FD step policy as an actual-SV tolerance, a general
  gradient tolerance, or an HMC tuning rule.

## Exact Next Step

No work remains for this bounded eight-hour root-cause goal. A future lane may
use the repaired helper and endpoint-rich FD harness, but it must write a new
evidence contract before any of the following:

1. full-time or multi-seed score admission;
2. HMC execution or HMC-readiness claims;
3. posterior/reference validation;
4. changes to production defaults or public APIs.

Do not rerun the four frozen GPU commands merely to reconfirm this result.
Rerun only if a relevant code or governance hash changes, and then write new
artifact paths rather than overwriting Iteration 5.
