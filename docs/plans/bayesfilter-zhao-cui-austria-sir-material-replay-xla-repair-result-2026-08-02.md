# Zhao-Cui Austria SIR Material-Replay XLA Repair Result

Date: 2026-08-02

Status: `PASS_T1_T2_MATERIAL_REPLAY_AND_FD_TANGENT_MECHANICS`

## Outcome

The exact-zero core-replay gate is retired for this lane. T1 and T2 now pass
the five-significant-digit material replay rule

```text
abs(replayed - parent) <= 5e-12 + 5e-6 * abs(parent)
```

on gauge-invariant full-density, retained-prefix-marginal, and scalar screens.
Both claim programs use TensorFlow/XLA numerical kernels with TensorFlow-owned
optimizer, TT-axis, microbatch, parameter-direction, and finite-difference
control flow. No NumPy numerical path, `PyFunc`, eager fallback, or
Python-controlled numerical loop is present in the claim computation.

The issued scores are finite-difference tangent mechanics, not exact autodiff:

- T1 cumulative score: `[-5.680764842992403, 2.062892896503854, -4.892212997553713]`.
- T2 cumulative score: `[-9.488823626497833, 2.7924278478508313, -7.555446764939688]`.

T2 additionally uses the documented scalar-consistency radial projection. The
raw core finite difference did not reproduce the direct scalar derivative; the
artifact preserves that mismatch and the applied correction.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit T1 material replay and FD-tangent mechanics | Passed all functional/scalar, manual, step-halving, strict-load, and memory gates | None fired | Finite-difference truncation and local linearization | Consume only through strict v4 loader for bounded mechanics work | Exact derivative, arbitrary-theta score, HMC readiness |
| Admit T2 material replay and scalar-consistent FD-tangent mechanics | Passed all functional/scalar, corrected-manual, step-halving, strict-chain, and memory gates | Raw core tangent mismatch fired and was repaired by an explicit radial projection | T2 shape tangent remains a finite-step estimate; radial projection constrains only the scalar derivative | Consume only through strict v3 T1-to-T2 loader for bounded mechanics work | Raw core FD correctness, exact autodiff, posterior correctness, later horizons |
| Stop after T2 | No T3+, HMC, or later-horizon run was authorized | Campaign boundary | Scientific validity beyond the frozen T1/T2 programs is not evaluated | Require a fresh plan and evidence contract for any expansion | Source-faithful Zhao-Cui parameter algorithm or production/HMC status |

## Evidence

### T1

Passed artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-material-replay-xla-20260802/t1-material-fd-tangent-issuer-02/result.json`.

- Status: `PASS_T1_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT`.
- Issuer identity: `cc8460bffd737bcf682434c8ff49c9c52ceb8af45ec81fba92a5afcb4d1556d0`.
- Child identity: `5a006e8f55423cb08e6b3b1b08443c6ac8fb3af1c637ff48c20ed7941cae0603`.
- Maximum functional normalized residual: `3.6608e-9`.
- Scalar log residual: `4.2633e-14`.
- Independent `h=1e-4` versus issued `h=5e-5` score residual: at most `5.2807e-6`.
- GPU allocator peak: `249,138,944` bytes under the `6,442,450,944` byte cap.
- Wall time: `40.40` seconds.

Root-cause evidence:
`t1-packed-trajectory-diagnostic-01` showed divergence before the first update
because basis tables were evaluated eagerly while the admitted authority
evaluated basis algebra inside XLA. XLA basis precomputation repaired the full
96-step trajectory; `t1-packed-trajectory-diagnostic-02` passed every
checkpoint.

### T2

Passed artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-material-replay-xla-20260802/t2-material-fd-tangent-issuer-02/result.json`.

- Status: `PASS_T1_T2_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT`.
- Issuer identity: `9b6dfaecdd311741facca0b31fb1e69c0accf82a79fd66ad76cc0481ca377313`.
- Child identity: `17e33778c558e62972eb5bfe342e297520ab1475b3722602aeab7827c60cf263`.
- Maximum functional normalized residual: `1.7637e-6`.
- Scalar absolute residual: `4.2633e-14`.
- Independent `h=1e-4` versus issued `h=5e-5` cumulative-score residual: at most `7.5443e-5`.
- Raw core-tangent increment score: `[-3.8050161699756435, 0.729398729308185, -2.6632442936816347]`.
- Direct scalar increment derivative: `[-3.808060062269192, 0.7295350059877137, -2.6632337544896245]`.
- Radial correction: `[-0.0015219461619934882, 6.813834044575678e-5, 5.26959605780412e-6]`.
- GPU allocator peak: `269,102,592` bytes under the `6,442,450,944` byte cap.
- Wall time: `310.82` seconds.

The raw mismatch is real relative to a raw-core-tangent correctness claim. The
quantity issued instead is explicitly defined as the raw FD shape tangent plus
a first-core radial component that enforces the direct finite-program scalar
derivative. See
`docs/plans/bayesfilter-zhao-cui-austria-sir-t2-scalar-consistency-repair-note-2026-08-02.md`.

## Attempts And Repairs

| Attempt | Classification | Result |
|---|---|---|
| T1 primal diagnostics 1-2 | XLA infrastructure failures | Static loop bounds and cyclic index schedule repaired; no numerical claim produced |
| T1 primal diagnostics 3-5 | Invalid core gate, then valid functional failures | Gauge-dependent core gate retired; functional failure preserved and localized |
| T1 trajectory diagnostics 1-2 | Numerical localization | Eager/XLA basis split identified and repaired |
| T1 issuer 1 | CUDA/XLA higher-order autodiff failure | Empty directory preserved; no tangent artifact issued |
| T1 issuer 2 | FD-tangent repair | Passed and strictly reloaded |
| T2 primal diagnostic 1 | Material replay preflight | Passed |
| T2 issuer 1 | Manual core-tangent parity failure | Empty directory preserved; tolerance was not relaxed |
| T2 FD diagnostics 1-2 | Derivative localization | Raw core mismatch confirmed; replay-base hypothesis rejected |
| T2 issuer 2 | Scalar-consistency projection repair | Passed and strict T1-to-T2 chain reloaded |

## Verification

CPU-hidden reference/testing run:

```text
20 passed, 2 warnings in 95.54s
```

This covered packed contraction and optimizer parity, active clipping, nested
XLA gradients, material tolerance boundaries, T1/T2 graph inspection with no
host callbacks, carried T1 marginal target parity, scalar-projection invariants,
and strict loader/tamper rejection.

GPU evidence used the RTX 4080 SUPER, TensorFlow 2.19.1, FP64, XLA JIT, and a
6,144 MiB logical-device hard limit. Full run manifests are in the issued
artifacts.

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for finite values, material replay, strict inputs/loaders, manual corrected score, step-halving FD, XLA graph, and memory |
| Statistically supported ranking | Not applicable; no stochastic method ranking was attempted |
| Descriptive-only differences | Raw core residuals, runtime, and diagnostic trajectory differences |
| Default readiness | Not established |
| Next evidence needed | Fresh target-specific plan for arbitrary-theta validation, later horizons, or HMC; preferably an eligible exact/manual derivative route rather than finite differences |

## Post-Run Red Team

Strongest alternative explanation: the local score agreement may be specific
to the origin and the selected finite-difference steps. The T2 radial projection
guarantees scalar consistency by construction and therefore is not independent
evidence that raw core coordinates are differentiated accurately.

Evidence that would overturn this closeout: failure of strict reload, a
same-program third-step-size check outside the declared tolerance, a direct
manual/autodiff derivative showing a materially different origin score, or a
functional replay failure on the frozen complete clouds.

Weakest evidence: derivative validity beyond the local origin. These artifacts
support bounded finite-program mechanics only.

