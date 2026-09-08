# Remaining-Gaps Closure and Hypothesis Campaign Result

Date: 2026-08-17  
Plan: `docs/plans/bayesfilter_direct_factor_srukf_remaining_gaps_closure_and_hypothesis_plan_2026_08_17.md`  
Review: `docs/plans/bayesfilter_direct_factor_srukf_remaining_gaps_closure_and_hypothesis_plan_review_2026_08_17.md`  
Artifact root: `docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/`

## Terminal disposition

`EXECUTED_GPU_GAPS_CLOSED_WITH_SCIENTIFIC_BLOCKERS_PRESERVED`

The managed-session GPU rerun closed every environment-dependent item. It
restored the frozen T10 `SVX-ZC` lane to executable status after its current
GPU/XLA gate and prior terminal sequential-HMC evidence were reconciled. This
is a scoped NeuTra/HMC metadata correction, not direct-factor SR-UKF evidence.

## Results

| Hypothesis | Result | Consequence |
|---|---|---|
| H1 KSC Gaussian-sum repair | T20 dense/value/score/mass/permutation gates passed for caps 7, 16, 32, 64, 128, 256; GPU 3 XLA canary matched CPU with value gap `7.11e-15` and score gap `2.22e-15` | Separate KSC surrogate route is admitted for its bounded scope; historical `KSC-UKF`, exact SV, HMC, and direct-factor claims remain excluded |
| H2 exact-SV SGQF level | GPU 3 tested levels 10, 12, 16, 20, 24 against level 32; every level failed, with the best dense-prefix value gap `0.00338696` per observation against the `0.001` gate | `SVX-SGQF` remains blocked by a scientific negative result, not an environment failure |
| H3 PP-ZC batch-native target | No source-anchored batch-native posterior/chart contract registered | `PP-ZC` remains blocked |
| H4 STR-ZC extension | No target program exists; a structural UKF initializer cannot provide target identity | `STR-ZC` remains blocked |
| H5 SIR-ZC observed-data score | Available teacher/latent/proposal scores are not the observed-data target score | `SIR-ZC` remains blocked |
| H6 SVX-ZC capability | GPU 3 gate passed: CPU/GPU value and score gaps `7.11e-15`, same-program FD gap `4.62e-9`, zero permutation gaps, valid statuses, memory growth verified; target signature `decc...cab`, current adapter signature `a915...12b` | Restore the frozen T10 registry row with `xla_hmc_ready=true` and `full_chain_xla_diagnostic_ready=true`; keep `runtime_autodiff_for_hmc=false` and all exact-filter/broad-validity nonclaims |
| H7 global singular analytical score | Falsified; rectangular singular probe returns `value_only_rank_discovery` | Keep value-only support route and reject scores across rank/pivot/sign/branch events |

## Verification

The focused campaign suite passed:

```text
33 passed, 3 warnings
```

The post-harness regression passed:

```text
4 passed, 3 warnings
```

After the GPU evidence and registry correction, the focused adapter, registry,
remaining-gap, and shared-procedure suite passed:

```text
67 passed, 3 warnings
```

The terminal combined verification then exposed and repaired an order-dependent
TensorFlow trace-cache issue in the batched SVD LGSSM authority. The kernel
requires static parameter, state, and observation dimensions for XLA, so its
`tf.function` no longer requests relaxed retracing across incompatible model
dimensions. A regression test now compiles consecutive 18-parameter and
2-parameter calls. The complete focused SR-UKF, SVD-authority, adapter,
registry, and policy suite passed:

```text
121 passed, 3 warnings
```

The literature survey was compiled twice with `pdflatex` and exited zero on
both passes. JSON artifacts validate with `python -m json.tool`; the artifact
root includes SHA-256 checksums in `artifact_hashes.json`.

Warnings are pre-existing HDF5/TensorFlow Probability compatibility and
deprecation warnings. The earlier no-GPU diagnosis was a workspace-sandbox
visibility artifact. Host-level checks found four RTX 4090 GPUs. Following the
requested preference order `3, 2, 1, 0`, GPU 3 was selected; every TensorFlow
process set `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth before
logical-device initialization, and used XLA JIT. No campaign process remains
on GPU 3.

The SVX-ZC GPU sub-artifact records the capability flags as false because it
was generated before the metadata repair. That historical observation is
preserved. Its passing gates, the exact target-signature match to
`sequential-hmc-attempt01/SVX-ZC/result.json`, and the current adapter-signature
match to the training artifacts are the evidence for the later correction.

## Remaining actionable gaps

1. For PP-ZC and SIR-ZC, obtain/cite exact Zhao-Cui paper and author-source
   operations, then implement only a source-faithful target contract if one
   exists. For STR-ZC, a new reviewed extension derivation is required first.
2. Preserve the analytical-score veto at singular support, rank changes,
   repeated singular values, QR pivot/sign changes, and angular branch cuts.
3. Treat the tested exact-SV SGQF ladder as falsified unless a new mathematical
   hypothesis changes the approximation family or the declared accuracy gate;
   increasing the tested level alone did not repair the dense-prefix error.
