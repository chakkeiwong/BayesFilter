# SIR Latent Pre-Clipping Law And Score Repair Result

Date: 2026-07-16

Status: `PARTIAL_SUCCESS_LATENT_LAW_AND_CONTRACT_E_SCORE_REPAIRED_REMAINING_ADMISSION_GAPS`

Plan: `docs/plans/bayesfilter-sir-latent-preclip-law-score-repair-plan-2026-07-16.md`

## Verdict

The original Austria SIR failure had two different causes:

1. the clipped simulator and Gaussian transition density were different
   probability laws; and
2. previous filtering marginal and transport/reset derivatives were not part of
   the claimed full score.

The first cause is repaired by filtering the continuous pre-clipping state
`z_t`, with exact source time order `x_0=z_0` and
`x_t=clip_susceptible(z_t)` for `t>=1`.  Paired fixed-noise tests prove that the
new latent representation reproduces the source simulator's physical states and
observations at `J=1,2,9`, including forced clipping.

The Contract E candidate now computes a total derivative of the same finite
latent-law value program.  It carries particle and normalized-weight tangents
through the previous filtering marginal, SIR RK4 dynamics, LEDH flow, weight
normalization, streaming transport, and Contract E--Chol moment/reset map.  A
stopped previous-marginal score fails the reduced oracle test, while the total
score matches autodiff and same-scalar finite differences.

This is not a complete SIR admission.  The route remains
`candidate_not_canonical_not_admitted`: the repository production identity
registry is still empty for this SIR route, fixed-TTSIRT derivative ownership is
still blocked, and no exact `d=18` filtering oracle exists.

## Claimed And Computed Quantities

| Item | Claimed target | Quantity actually computed | Verdict |
| --- | --- | --- | --- |
| Simulator law | clipped Austria SIR observed-data law | continuous latent `z_t` law followed by deterministic physical projection | equal by paired-noise construction and tests |
| Reduced reference score | derivative of fixed-grid latent filtering scalar | explicit normalized filtering-score recursion including previous-marginal carry | correct for the fixed grid; matches AD/FD |
| Contract E score | derivative of the same finite Contract E--Chol candidate value | forward three-direction total tangent including streaming transport and reset moment/weight dependence | correct for executed finite scalar; matches AD/FD |
| `d=18` scientific likelihood | exact clipped-SIR likelihood | finite `N=32` Contract E candidate | different; approximation accuracy not established |
| Zhao--Cui comparator score | total derivative of fixed-TTSIRT source route | not implemented | blocked |

## Defects Found And Repaired

1. **Initial time-order error in the proposed mathematics.**
   The source simulator does not clip the initial draw.  The repaired target uses
   `x_0=z_0`; clipping begins after the first transition.
2. **Under-resolved dense reference.**
   Low-order Legendre and Hermite grids passed own-scalar derivative checks but
   failed value/score refinement by `O(10^-1)` to `O(1)`.  A susceptible-axis
   split grid and an order/range ladder established convergence only at the
   higher `29/33` orders.
3. **Missing previous-marginal score.**
   A stopped-carry negative control differs from the autodiff filtering score at
   `T=2`; the repaired recursion includes this derivative.
4. **Contract E candidate initially used Python horizon/RK4 loops.**
   Both are now `tf.while_loop` bodies.  Remaining Python direction loops are
   fixed parameter dimension three and are not horizon-scaling loops.
5. **XLA string return.**
   GPU attempt 1 failed because the compiled function returned string metadata.
   Metadata now lives outside the numeric compiled graph.
6. **CPU/GPU preparation mismatch.**
   Device-local TensorFlow RNG produced different prepared bytes.  A CPU-only
   preparation artifact now owns all tensors and hashes; CPU and GPU consume the
   identical serialized values.
7. **Clipping-boundary chart was diagnostic instead of fail-closed.**
   Exact susceptible zeros make the pathwise derivative undefined.  The final
   source marks exactly those points invalid and poisons carried/reported
   numerical state for `t>=1`.  The unclipped `t=0` state remains admissible,
   including at zero.  The implementation does not impose an arbitrary
   near-zero exclusion band.
8. **Candidate factory JIT default lagged the repaired CLI.**
   The benchmark CLI defaulted to XLA, but the Python factory still defaulted
   `jit_compile=False`.  The factory now defaults true, while explicit non-JIT
   remains limited to labeled CPU reference/debug runs.

## Reduced Reference Evidence

The `J=1,T=2` split-panel reference at order 33 gave stable refinement:

- order 29 to 33 at radius 6: value difference about `1.35e-6`, worst score
  difference about `4.86e-6`;
- radius 6 to 7 at order 33: value difference about `2.53e-5`, worst score
  difference about `1.77e-4`;
- outer-node posterior mass below `3e-10` at radius 6 and below `6e-13` at
  radius 7.

The manual filtering score matches autodiff to `2e-9` and coordinate finite
differences to `2e-7` on the test fixtures.  These are numerical reference
checks, not exact error bounds.

The deterministic Contract E particle ladder was descriptive:

| Particles | Value gap to reduced reference | Score gap |
| ---: | ---: | --- |
| 8 | `+0.0086726` | `[+6.64e-5,-4.81e-5,+2.27e-5]` |
| 16 | `+0.0028932` | `[+6.19e-5,+6.67e-5,-6.70e-4]` |
| 32 | about `-6.0e-4` | finite, small descriptive differences |

The ladder supports viability and decreasing value error on this fixture.  It
does not statistically rank particle counts or prove general convergence.

## `d=18` Results

All rows use parameter order
`[log_kappa_scale,log_nu_scale,log_obs_noise_scale]` and `N=32`.

| Horizon | Execution | Value | Score | Maximum coordinate FD relative error | Reset status |
| ---: | --- | ---: | --- | ---: | --- |
| 2 | CPU float64, bound preparation | `-69.1394005463` | `[4.20702942,-1.56618202,3.45811811]` | `3.69e-10` | both pass |
| 2 | trusted GPU/XLA, same bound preparation, pre-boundary-fix source | same within `5.68e-14` | max CPU/GPU delta `9.06e-14` | `5.38e-10` | both pass |
| 5 | CPU float64 | `-164.943371253` | `[28.2118819,-12.4064619,-8.56419382]` | `5.86e-9` | all five pass |
| 20 | CPU float64 | `-678.573793358` | `[-26.8867479,-45.0915853,5.40444173]` | `1.18e-7` | all twenty pass |

The `T=20` minimum row-transport mass was `0.72865`; every tested path was away
from the exact clipping boundary.  These results establish same-scalar score
wiring for the finite candidate.  They do not establish `d=18` oracle accuracy.

The trusted GPU artifact compiled a cluster with XLA on an NVIDIA GeForce RTX
4080 SUPER and used `/GPU:0`.  However, the source was subsequently tightened to
fail closed at exact clipping-boundary points.  The attempt budget was exhausted,
so there is no post-boundary-fix exact-source GPU certificate.  Because the
tested prepared path was away from the boundary, the numerical program on that
path is unchanged, but exact-source GPU certification remains pending.

## Artifact Ledger

| Artifact | SHA-256 |
| --- | --- |
| bound `T=2,N=32` preparation | `5a5db3a1e9b2e3702f823452d3d1246a9b291ad03dc3abfd727a39cf207392b8` |
| bound CPU `T=2,N=32` result | `ef5cbb24932382beabdb9aa952ab0f89ddd47e72075231932f68ba09622b8a40` |
| bound GPU/XLA attempt 3 result | `373eb4f59606a6bb31d0d26c76031c2460730a96f47ce07f2e77d33cc46724e4` |
| intermediate post-boundary-fix CPU `T=2,N=32` result | `040a26d38ac3153a1bc2d4053c4596351617127e26d1158eb14eb3d5db6f1a62` |
| final-source CPU `T=2,N=32` result | `34ee87cfb90ef5d1033dcb8eeba6cd8d8b22073fb04847e5e3aa59ec6db29a96` |
| `T=5,N=32` preparation | `6ad5928950058be00b523ca0aabffe4b0134997665ecf43570f443f17720c1ff` |
| `T=5,N=32` CPU result | `cc947e413b2a2884f34e148d3dbd9ba85a8594161fba8338da4d8a0f56c0582f` |
| `T=20,N=32` preparation | `2f3c3222ca9ecaea0ab49adc890d752637f24c0452ea11fa9bddcdcc33c5dcc8` |
| `T=20,N=32` CPU result | `95d466ccb3acc142cce404148ecce727c9c6b6798240421b2b39db4f24eae498` |

The final-source CPU `T=2` result is stored under
`docs/benchmarks/artifacts/sir_latent_preclip_repair_20260716/cpu_bound_t2_n32_final_source_attempt01/`.
It passed with value `-69.13940054633028`, score
`[4.207029415814342,-1.5661820189298876,3.4581181147594955]`, and maximum
coordinate FD relative error `3.6936035764568315e-10`.  It explicitly used
`--no-jit-compile` as a deliberate CPU reference/debug exception; the candidate
factory and serious GPU route default to XLA JIT.

## Verification Close Record

Current-source deliberate CPU-only focused checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-sir-repair \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
  tests/highdim/test_sir_latent_preclip_tf.py \
  tests/highdim/test_sir_latent_preclip_reference_tf.py \
  tests/highdim/test_ledh_contract_e_latent_sir_tf.py
```

The final run passed `17` tests in `132.01` seconds with two TensorFlow
Probability deprecation warnings.  It includes negative controls for an exact
post-transition clipping boundary, tiny nonzero states on both sides of the
boundary, the unclipped `t=0` identity chart, and the XLA-on factory default.

The shared Contract E cloud/reset suite produced `15 passed, 1 failed`.  The
only failure is the pre-existing persisted Phase 3 aggregate output hash:

- stored `output_tensor_sha256`:
  `8b9f7895d33b98321d4bad947001d3f2a464da237410c69765d79541c969669a`;
- current recomputation:
  `cf6d98f759d60439a02debda4b8e7207c44d7c7d420d09217befcfaf9345c5c4`.

The other seven fields in that diagnostic dictionary compare equal before the
hash assertion.  This is outside the new SIR files and was neither regenerated
nor hidden.  It remains an evidence-serialization audit item, not a passing
shared-suite claim.

## Phase Close Classification

Phases 1--3 and the terminal audit are complete.  Phase 4 is partial because a
`J=2` spatial oracle is still absent.  Phase 5 is partial because the GPU/XLA
certificate predates the final boundary fail-close source change.  Phase 6 is
blocked by the preserved fixed-TTSIRT derivative gaps.  Therefore the repair
scope is closed as a partial success; the broader admission program is not
complete.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| adopt latent pre-clipping SIR target as the repaired simulator-law candidate | pass | no law mismatch | representation is an extension, not author filtering implementation | bind reviewed target identity in future SIR contracts | source-faithful Zhao--Cui filtering |
| accept reduced total-score mathematics | pass | stopped-carry negative control behaves correctly | quadrature has no formal exact error bound | retain as reduced reference | exact nonlinear likelihood |
| accept finite Contract E--Chol candidate score wiring | pass for executed CPU finite programs | boundary now fail-closed | no `d=18` oracle and no canonical identity | build production route identity and post-fix GPU certificate | canonical/HMC/leaderboard readiness |
| close fixed-TTSIRT comparator | blocked | preserved blocker labels | derivative design remains unwritten/source-grounded | successor source-anchored derivative program | Zhao--Cui gradient comparison |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | latent-law, reduced reference, CPU finite-score, reset, and FD gates pass; exact-source post-fix GPU and identity gates remain open |
| Statistically supported ranking | none |
| Descriptive-only differences | particle ladder and candidate/reference gaps |
| Default readiness | false |
| Canonical Contract E readiness | false; repository-owned production route identity not issued |
| HMC readiness | false |
| Leaderboard readiness | false |
| Next evidence | production identity/factory, post-fix GPU/XLA rerun, `J=2` spatial reference, `d=18` accuracy evidence, fixed-TTSIRT total derivatives |

## Run Manifest Summary

- Git commit at serious runs: `d269f5bbd8531b878d4f25897a357fbc8f172488`.
- Worktree: dirty research worktree; unrelated changes preserved.
- Environment: TensorFlow `2.19.1`, Python `3.11.14`.
- CPU reference runs deliberately used `CUDA_VISIBLE_DEVICES=-1`.
- Trusted GPU/XLA run used NVIDIA GeForce RTX 4080 SUPER and recorded
  `owner_designated_managed_session_visible_gpu_trusted`.
- GPU attempts: three, exhausting the plan budget.
- No package install, environment mutation, HMC runtime, public release, or
  default-policy change was performed.

## Post-Run Red Team

The strongest alternative explanation for the good same-scalar results is that
they prove only internal differentiation of a finite candidate, not filtering
accuracy.  The reduced `J=1` reference partly addresses that issue, but there is
no tractable `d=18` oracle.  A wrong finite approximation can have an exact
gradient of its own scalar.

The weakest evidence is therefore cross-method scientific accuracy at `d=18`,
followed by the absent exact-source post-boundary-fix GPU certificate.  A future
result that shows stable same-target disagreement against a credible `J=2` or
spatial reference would overturn any broad accuracy interpretation while leaving
the law and derivative wiring repairs intact.

## Remaining Gaps

1. Register and review a repository-owned Contract E--Chol SIR route identity;
   no caller-stamped identity is acceptable.
2. Rerun the patched source on trusted GPU/XLA with the bound preparation.
3. Add a `J=2` spatial-coupling reference before generalizing the `J=1` accuracy
   result.
4. Establish `d=18` approximation accuracy or uncertainty; same-scalar FD is not
   sufficient.
5. Implement source-grounded previous-marginal and fixed-TTSIRT
   proposal/transport derivatives.  Existing blockers remain:
   `BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED` and
   `BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED`.
6. Only after these gates may HMC or leaderboard admission be reconsidered.
