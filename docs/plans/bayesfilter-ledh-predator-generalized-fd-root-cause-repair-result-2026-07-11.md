# Predator-Prey And Generalized-SV FD Root-Cause Repair Result

Date: 2026-07-11

Status: `ROOT_CAUSES_CONFIRMED_REPAIRED_AND_GPU_FD_CHECKS_PASS`

## Outcome

Both finite-difference failures now have confirmed, different root causes, and
both bugs are fixed.

1. Predator-prey was a finite-difference measurement-resolution bug. The old
   absolute central step `h=1e-4` was applied to every float32 parameter,
   including `K=114` and `a=25`. For `a`, the two float32 objective evaluations
   were identical, so the objective numerator and stored FD were exactly zero.
   The derivative was not zero: on the `T=1` CPU/FP64 reference graph, no
   transported state can feed a later likelihood, and the manual JVP, ordinary
   autodiff, and a stable relative-step FD ladder agree.
2. Generalized-SV had a manual transport-JVP implementation bug. The unchanged
   raw Sinkhorn forward loop performs at most `max_iterations - 1` annealing
   updates, but the manual JVP replayed `max_iterations` updates. The manual
   derivative therefore differentiated a different transport map. Replaying
   `max_iterations - 1` updates makes the manual JVP agree with autodiff through
   the unchanged raw forward transport at Sinkhorn caps 1, 2, and 10 to about
   machine precision on FP64 fixtures.

The original failing production shapes now pass the owner-directed FD-only
rule on trusted GPU/XLA/float32/TF32:

| Row | Shape | p | Worst coordinate relative error | `0.05*sqrt(p)` | Worst coordinate | Result |
| --- | --- | ---: | ---: | ---: | --- | --- |
| predator-prey | `T=1,N=2` | 6 | `0.000503381988322` (0.0503%) | `0.122474487139` (12.247%) | `u` | pass |
| generalized-SV | `T=4,N=10000` | 3 | `0.00487298671266` (0.4873%) | `0.0866025403784` (8.660%) | `mu` | pass |

All nine final coordinate pairs had finite, representably distinct parameter
endpoints, distinct objective endpoints, and nonzero objective numerators.

## Scope Of The Two Numerical Constants

The FD error tolerance and the FD evaluation step are different quantities:

| Quantity | Value | Role |
| --- | --- | --- |
| FD-only error threshold | `0.05 * sqrt(p)` | Owner-directed pass/fail rule over individual coordinate relative errors. The `5%` choice mirrors the conventional 95% threshold but is not a computed confidence interval. |
| Float32 central-step coefficient | `cbrt(2^-23) = 0.00492156660115185...` | Numerically derived balance for a smooth central difference, where truncation is `O(h^2)` and roundoff is `O(epsilon/h)`. It is multiplied by `max(1, abs(theta_j))`. |

The step coefficient is not a return to the rejected arbitrary `0.005`
tolerance. For example, the repaired predator-prey half-step is approximately
`0.123039` for `a=25` and `0.561059` for `K=114`, not `0.005` and not `1e-4`.
The FD is divided by the actual representable separation
`theta_plus_j - theta_minus_j`, not by an idealized `2*h`.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Scientific/engineering question | Determine whether the two failures came from a manual derivative bug, a value/score route mismatch, float32 fixed-step resolution, or nonsmooth behavior, then repair confirmed bugs. |
| Exact baseline | Identical fixed-randomness value-only scalar and compact manual JVP at the same theta, prepared inputs, seed, transport policy, precision, and shape. |
| Primary reference comparator | Before transported state can affect a later likelihood, ordinary `GradientTape` of the identical value graph is usable. After transport affects later likelihoods, the valid total-derivative reference is `GradientTape` through the unchanged raw forward transport with `transport_ad_mode=full`, plus central FD. |
| Production criterion | Every endpoint is finite and noncollapsed; stored endpoint arithmetic reconstructs exactly; and `max_j(abs(score_j-FD_j)/max(abs(score_j),abs(FD_j),1e-12)) <= 0.05*sqrt(p)`. |
| Hard vetoes | Wrong scalar, parameter order, seed, prepared inputs, transport branch, device/XLA/TF32 provenance, source hash, nonfinite/collapsed endpoint, failed endpoint arithmetic, manual-JVP/reference mismatch after repair, or failed owner threshold. None fired in the final evidence. |
| Explanatory only | Runtime, GPU memory, objective ULP counts, individual step-ladder values, and ordinary stabilized autodiff after transport. |
| Artifact | Four schema-v4 trusted GPU shards and the FP64 localization artifacts under `docs/plans/artifacts/ledh-predator-generalized-fd-root-cause-repair/`. |
| What is not concluded | No HMC readiness, posterior correctness, full-time generalized-SV admission, score admission, statistical superiority, runtime superiority, confidence-interval coverage, or broad default readiness. |

The skeptical pre-run audit passed because the old universal `h=1e-4` was not
a defensible float32 comparator for coordinates spanning `0` through `114`,
and because a larger passing step alone was not accepted as evidence. The plan
required independent derivative agreement, a predeclared step ladder, exact
endpoint records, and production-target confirmation.

## Claimed Target Versus Computed Quantity

| Row/route | Claimed target | Quantity formerly computed | Relationship after repair | Evidence anchor |
| --- | --- | --- | --- | --- |
| predator-prey old FD | Coordinate derivative of the realized fixed-randomness finite-`N` value scalar. | Central difference of the same scalar with absolute `h=1e-4` in float32. For `a`, the two objective values rounded to the same float32 number, producing numerator `0` and FD `0`. | The old FD was wrong as a measurement of the nonzero derivative. The repaired coordinate-scaled central FD agrees with the compact score. | Historical FD shard SHA `738c59f9...`; FP64 reference SHA `d4a99903...`; final GPU FD SHA `e6064fec...`. |
| predator-prey compact score | Total coordinate derivative of the same realized finite-`N` scalar at fixed prepared inputs. | Manual JVP. | Correct on the checked fixture: FP64 manual JVP and autodiff agree, and the final float32 GPU FD agrees under the FD-only rule. | `iteration1-predator-prey-t1-n2-fp64.json` and final GPU pair. |
| generalized-SV old manual JVP | Total derivative through the exact finite-iteration raw forward transport. | Derivative through one extra annealing update: `max_iterations` manual updates versus at most `max_iterations-1` forward updates. | Wrong relative to the claimed forward map. | Raw forward conditions at `annealed_transport_tf.py:1512` and `annealed_transport_tf.py:5598`; pre-repair cap-2/cap-10 artifacts. |
| generalized-SV repaired manual JVP | Total derivative through the exact finite-iteration raw forward transport. | Manual replay with `max_iterations-1` updates. | Equal to full-transport autodiff to approximately `1e-15` on checked FP64 cap-1/cap-2/cap-10 fixtures and consistent with production FD. | `_manual_dense_finite_steps` in `experimental_batched_ledh_pfpf_ot_tf.py:81`; post-repair FP64 SHA `e69a2a79...`; final GPU pair. |
| generalized-SV ordinary stabilized autodiff | Not the claimed total derivative once transported state affects later likelihoods. | Gradient through the stabilized forward route, whose backward stops deliberately omit some transport dependence. | Different by design; it is not a valid comparator for the total derivative after transport. | Post-repair FP64 diagnostic records both ordinary and full-transport autodiff. |

The generalized-SV pseudo-observation derivative was also audited. Its discarded
term is intentional: `H*x + (y - H*x) = y`, so its total derivative with
respect to the state cancels. It was not the source of the failure.

## Root-Cause Evidence

### Predator-Prey

The historical GPU shard used `h=1e-4` and stored:

| Parameter | Score | Historical FD | Historical relative error |
| --- | ---: | ---: | ---: |
| `K` | `-1.3080407381` | `-0.9918212891` | `0.241750464` |
| `a` | `-0.1401509792` | `0.0` | `1.0` |

The same FP64 fixture gives manual/autodiff/FD values near
`-0.11672239524` for `a`, proving that zero was not the mathematical
derivative. In the repaired trusted GPU run, `a` has:

| Field | Value |
| --- | ---: |
| score | `-0.140150979161` |
| nominal half-step | `0.123039165029` |
| actual half-step | `0.123039245605` |
| minus objective | `-64.0429382324` |
| plus objective | `-64.0774383545` |
| objective numerator | `-0.0345001220703` |
| reconstructed FD | `-0.140199661255` |
| coordinate relative error | `0.000347234032` |

Thus the former integer-looking `0` came from equal float32 objective
endpoints at an under-resolved step, not from a zero derivative and not from
rounding the printed FD to too few decimal places.

### Generalized-SV

The raw dense and streaming forward loops both stop on
`i < max_iter - 1`. The former helper returned `max_iter`, so every manual
dense/streaming finite route replayed one update that the forward map did not
execute. The helper now returns `max_iter - 1`, including zero updates for a
public cap of one.

The discriminating FP64 fixtures show:

| Sinkhorn cap | Manual update count after repair | Maximum manual versus full-transport autodiff discrepancy | Status |
| ---: | ---: | ---: | --- |
| 1 | 0 | approximately `1.4e-17` absolute | pass |
| 2 | 1 | approximately `9.7e-17` absolute | pass |
| 10 | 9 | approximately `6.3e-17` absolute | pass |

Before repair, cap 2 and cap 10 failed the tight reference comparison in all
three generalized-SV directions. For cap 10, for example, the old manual `mu`
score was `-0.0720954568755` while full-transport autodiff was
`-0.0721257718654`. After repair they are exactly equal at the stored FP64
precision.

The final production-shape generalized-SV values are:

| Parameter | Score | FD | Relative error |
| --- | ---: | ---: | ---: |
| `gamma_unconstrained` | `-0.015261708759` | `-0.015306465328` | `0.00292403034` |
| `log_tau` | `-0.034229077399` | `-0.034196406603` | `0.00095447492` |
| `mu` | `-0.029305875301` | `-0.029163068160` | `0.00487298671` |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Replace the universal fixed FD step in the shared GPU FD harness with `cbrt(float32_epsilon)*max(1,abs(theta_j))` and divide by the actual endpoint separation. | Passed on the original predator-prey failure shape; all six endpoint pairs and reconstructed values validate. | No endpoint, provenance, same-input, arithmetic, or owner-threshold veto fired. | This numerical step policy is checked for these FD diagnostics, not calibrated as a universal optimizer/HMC perturbation policy. | Keep the endpoint-rich schema and validator for future float32 FD evidence. | No broader gradient tolerance or confidence interval is defined. |
| Change all manual finite-Sinkhorn replay modes from `max_iterations` to `max_iterations-1` updates. | FP64 manual JVP matches full-transport autodiff at caps 1, 2, and 10; the original generalized-SV production shape passes FD. | No same-scalar, transport-route, arithmetic, provenance, or threshold veto fired. | Full `T=1008`, multiple seeds, and downstream HMC behavior were not run. | Treat the off-by-one implementation defect as fixed; use separate reviewed gates for broader admission. | No full-row generalized-SV or HMC readiness claim. |
| Close this bounded root-cause goal. | Both failures have causal explanations, code repairs, focused tests, shared-helper regressions, and trusted production-target confirmation. | All four final shards pass current schema-v4 validators. | Other untested score defects may still exist outside these two causes and shapes. | Preserve the result and reset memo; do not overwrite historical Phase 9 shards. | No scientific ranking or posterior claim. |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No hard veto fired in the final predator-prey or generalized-SV evidence. The former predator-prey zero FD is rejected as under-resolved numerical evidence; the former generalized-SV manual score is rejected because it differentiated the wrong finite-update map. |
| Statistically supported ranking | None. This was deterministic fixed-input derivative validation, not a stochastic method comparison or ranking. |
| Descriptive-only differences | Runtime, memory, individual endpoint magnitudes, and error margins are descriptive. |
| Default-readiness | No new broad default-readiness conclusion. The owner-directed GPU/XLA/TF32 execution policy remains separate from this FD-only check. |
| Next evidence needed | Full-time/multi-seed score admission, HMC readiness, posterior/reference agreement, or confidence-coverage claims each require their own reviewed evidence contract and artifacts. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | The manual update-count defect is fixed in the shared helper. The v4 FD harness stores actual endpoints and validators reconstruct every float32 arithmetic operation. Focused and cross-model tests pass. |
| Numerical validity | FP64 reference comparisons and stable step ladders isolate the two causes. Trusted GPU/XLA/TF32 checks pass at the original failure shapes. |
| Scientific interpretation | FD-only derivative evidence for one fixed seed at the stated shapes. It does not establish posterior correctness, HMC validity, or comparative scientific performance. |

## Artifact Bindings

| Artifact | SHA-256 |
| --- | --- |
| historical predator-prey score | `82eb75a8710a6c4219419b5f9c14f670e371554a3c7943a2a3fb5e03f1c28f5c` |
| historical predator-prey FD | `738c59f9967ec86dfc09be7bfb315e4cc9fdfc04a22cec95292527405f1b3127` |
| historical generalized-SV score | `3fb140284b74a02efb8fe57562f0f33ee75a1012bd1dd6cdc554be71c59e71d6` |
| historical generalized-SV FD | `edc896ef4a41772e29487257b3c6e01c8543b780aacf930fa607dd43479b8b08` |
| predator-prey FP64 reference | `d4a9990340ad27dc9d99cc533a219a696893655b103c5491773f60978657e8c2` |
| generalized-SV post-repair FP64 reference | `e69a2a79a212fea4c48710537abfc1fd7f62f1dbc53a4efa8f00ff852cabf610` |
| final predator-prey GPU score | `789218ad78a9cedd5e9393d60f72b024ff944f4c98f72c85feee883445ea70d8` |
| final predator-prey GPU FD | `e6064fec5b9f5a444248d20b9342991d35907859ab58e783ea3c177d714f5bca` |
| final generalized-SV GPU score | `44c67898d63f47db23f1115c6bd48cff4c1645057bd1814ab728860841a1bf8f` |
| final generalized-SV GPU FD | `0605c2be019b5558d1f83eb71aea1fe93765b9fbccf7592b173a1fb185ba6163` |
| frozen root-cause subplan | `09a37a5de32289927f5daba72d5f59f037f586d86a19551709eeedf92b93f3e0` |
| frozen exact GPU command manifest | `b88cb82c114449b1c54a960ac02d608340381ef3b87c448bb7b07795a3ad8a9e` |

The four final shards embed the last two governance hashes, current code-source
hashes, their source-value artifact hashes, and their paired prepared-input
fingerprints. Historical Phase 9 shards were not modified.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus scoped uncommitted repairs and unrelated dirty work disclosed in each JSON artifact |
| Branch | `main` |
| Exact GPU commands | Four commands in `docs/plans/ledh-predator-generalized-fd-root-cause-repair-gpu-commands-2026-07-11.json`; each exact command is also embedded in its output shard |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu`; Python `3.11.14`; TensorFlow `2.19.1` |
| GPU status | Trusted visible `/GPU:0`, trust basis `owner_designated_managed_session_visible_gpu_trusted` |
| Precision | float32 tensors, TF32 enabled |
| Compilation | `tf.function(..., jit_compile=True)` |
| Random seed | `81120` for both rows |
| Data/input version | SHA-256 serialized prepared-input fingerprints: predator-prey `a4de81d972a3e4b445252b8fb73bb00b5df599980548c13ce3fee935bf421727`; generalized-SV `d8e322dbaac08835fc0499c5161d77ba02777deeac2d4f26022e77ca3202c643` |
| Predator-prey wall time | score `21.2117s`; FD `8.25488s` |
| Generalized-SV wall time | score `74.4890s`; FD `45.2530s` |
| Score peak allocation | predator-prey `0.0376 MiB`; generalized-SV `35.2310 MiB`; explanatory only |
| Plan file | `docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-subplan-2026-07-11.md` |
| Result file | `docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-result-2026-07-11.md` |

CPU/FP64 localization and all final test runs intentionally used
`CUDA_VISIBLE_DEVICES=-1`; they are reference or engineering checks, not GPU
production evidence. The trusted GPU result is preserved only in the four
bound shards above.

## Verification

| Check | Result |
| --- | --- |
| Current schema-v4 validation of all four final shards, including paired hashes, prepared-input identity, endpoint reconstruction, source/governance hashes, GPU provenance, and recomputed FD decision | pass |
| Full compact GPU/XLA harness tests | `93 passed, 2 warnings in 96.96s` |
| Shared finite-Sinkhorn helper plus fixed-SIR, predator-prey, actual-SV, generalized-SV, KSC-SV, cross-model, root-cause, and FD-policy regressions | `144 passed, 2 warnings in 305.33s` |
| Focused transport primitive checks during repair | `13 passed` |
| Focused model/diagnostic and generalized-SV contract checks during repair | `21 passed` |
| Focused FD-policy checks during repair | `10 passed` |
| Focused endpoint-validator checks during repair | `9 passed` |
| Exact root-cause command authorization checks during repair | `4 passed` |
| Python compilation and scoped `git diff --check` | pass |

The two warnings in the final pytest runs are TensorFlow Probability
`distutils.version` deprecation warnings and do not affect these checks.

## Post-Run Red Team

- Strongest alternative explanation for predator-prey: the larger coordinate
  step could cross a branch and happen to agree. The predeclared FP64 ladder,
  manual-JVP/autodiff agreement, smooth values across multiple relative steps,
  and endpoint-rich production replay argue against that explanation at the
  checked fixture.
- Strongest alternative explanation for generalized-SV: the mismatch could
  have come from a different omitted term. The exact cap-1/cap-2/cap-10
  before/after experiment changes only the replayed update count and collapses
  the manual/full-autodiff discrepancy to approximately `1e-15`, which
  identifies the off-by-one cause.
- Result that would overturn the implementation conclusion: the repaired
  manual JVP fails full-transport autodiff for the same finite-update map under
  a smooth bounded fixture, or source inspection shows another forward update
  not represented in the manual replay.
- Result that would overturn the production FD conclusion: a source-bound
  rerun at the same inputs fails endpoint reconstruction, lacks a stable local
  step region, or exceeds the owner threshold without a changed target.
- Weakest evidence: the trusted runs use one fixed seed; predator-prey is the
  tiny original `T=1,N=2` failure and generalized-SV is only the original
  `T=4` prefix, not `T=1008`.

## Explicit Nonclaims

- This does not establish HMC readiness or useful HMC trajectory behavior.
- This does not establish posterior correctness or posterior/reference
  agreement.
- This does not admit generalized-SV at full `T=1008` or any row at a complete
  multi-seed score gate.
- This does not rank methods or show statistical superiority.
- This does not calibrate a 95% confidence interval. The `5%` constant only
  mirrors the conventional threshold selected by the owner.
- This does not make the FD step formula a general tolerance for actual-SV,
  HMC, optimization, or gradients outside this float32 FD diagnostic.
