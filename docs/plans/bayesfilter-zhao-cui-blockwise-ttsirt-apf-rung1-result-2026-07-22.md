# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Result

Date: 2026-07-22

Status: `PASS_ENGINEERING_RUNG1`

Route classification: `extension_or_invention`

## Verdict

The fitted blockwise TTSIRT proposal passed the predeclared engineering screen
at the canonical synthetic scope `d=24,T=3,N=256`. The selected degree-10,
rank-4, scale-2.5, defensive-mass-`1e-6` proposal remained finite, achieved a
minimum ESS fraction of `0.562903`, and matched the analytical score of the
same frozen finite scalar to a maximum finite-difference error of `0.003844`.
All value and score arms executed with TensorFlow XLA on the shared GPU.

This establishes that fitted scalar adjacent TTSIRT conditionals can be
composed into a working 24-block fixed-proposal APF under one shared ancestor
genealogy for this independent Gaussian diagnostic. It does not establish a
source-faithful Zhao-Cui filter, coupled nonlinear scalability, an exact
randomized likelihood estimator, HMC validity, Austria SIR or NAWM support,
production KR closure, or default readiness.

The terminal source audit corrected one pre-run metadata label: compiler `v1`
called the whole reordered finite-grid compiler a `fixed_hmc_adaptation`.
That was wrong under the Zhao-Cui lane policy. Compiler `v2` classifies the
whole route as `extension_or_invention`; only freezing the cited sampling and
correction operations is a `fixed_hmc_adaptation`. This metadata-only repair
does not alter the states, proposal densities, weights, score recursion, or GPU
kernels evaluated in `gpu_attempt01`.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain the blockwise fitted-TTSIRT APF as an engineering candidate | Passed at `d=24,T=3,N=256` | All predeclared fit, measure, map, finite-value, score, ESS, XLA, device, and memory-growth gates passed | Independent scalar Gaussian blocks may hide coupled nonlinear TT-rank and weight-collapse failures | Design a separately tuned coupled nonlinear block/rank rung with a fresh evidence contract | No source-faithful Zhao-Cui, exact pseudo-marginal, HMC, Austria SIR, NAWM, production, default-readiness, or superiority claim |

## Canonical Evidence

| Diagnostic | CPU precheck | Trusted GPU claim | Role |
| --- | ---: | ---: | --- |
| Status | `PASS_CPU_REFERENCE_PRECHECK` | `PASS_ENGINEERING_RUNG1` | Execution classification |
| Selected degree / rank / scale / defensive mass | `10 / 4 / 2.5 / 1e-6` | `10 / 4 / 2.5 / 1e-6` | Frozen tuning result |
| Maximum full-proposal heldout relative sqrt-density RMS | `0.00844072` | `0.00844072` | Explanatory fit diagnostic |
| Maximum reference-target quadrature mass error | `4.35e-8` | `4.35e-8` | Measure/target gate |
| Paired-core conditional formula tie-out | `0` | `0` | Implementation identity gate |
| Maximum inverse/forward roundtrip error | `1.67e-16` | `1.67e-16` | Diagnostic grid-map gate |
| Candidate minimum ESS fraction | `0.562903` | `0.562903` | Primary downstream gate, threshold `>=0.5` |
| Candidate same-scalar score/FD maximum error | `0.003846` | `0.003844` | Primary score gate, threshold `<=0.03` |
| Warmed repeatability absolute error | `0` | `0` | Engineering gate, threshold `<=1e-5` |
| Candidate output device | CPU | `/device:GPU:0` | Placement gate |

The tiny CPU/GPU numerical differences are consistent with the declared
float32 online execution and do not change any gate. The CPU precheck is
nonclaiming; only the trusted GPU execution issued the terminal engineering
status.

Matched single-branch arm diagnostics from the GPU artifact are descriptive:

| Arm | Minimum ESS fraction | Score/FD max error | Log likelihood |
| --- | ---: | ---: | ---: |
| Exact predictive auxiliary | `1.000000` | `0.000666` | `-89.332947` |
| Exact uniform auxiliary | `0.571157` | `0.002997` | `-89.212166` |
| Fitted TTSIRT uniform auxiliary | `0.562903` | `0.003844` | `-89.218216` |

The matched arms isolate auxiliary-law and fitted-proposal effects, but one
frozen branch supplies no uncertainty basis for ranking the viable arms.

## GPU And Memory

The pre-launch shared-device snapshot reported an RTX 4080 SUPER with
`16376 MiB` total, `2536 MiB` used, `13510 MiB` free, and `39%` utilization.
An eight-second sample remained at `34-39%` SM and `3-4%` memory utilization,
so the bounded online workload was launched rather than competing with the
earlier `74%` load.

The runner set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and
used the repository memory-policy helper to configure and verify growth on
every visible physical GPU before logical-device initialization. No logical
device memory limit or whole-device preallocation was used. Offline fitting
and proposal compilation were explicitly placed on CPU; only the three
float32 XLA APF arms were placed on GPU.

| Manifest field | Value |
| --- | --- |
| Memory policy | `bayesfilter.tensorflow.gpu_memory_policy.v1`, `memory_growth` |
| Growth verified | `true` on `/physical_device:GPU:0` |
| Full-device preallocation disabled | `true` |
| TF32 / XLA | `true / true` |
| TensorFlow allocator current / peak | `459264 / 482560` bytes |
| Offline fit / proposal compile / total wall time | `4.976 / 22.605 / 31.889` seconds |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

After process exit, the device snapshot reported `2221 MiB` used and
`13825 MiB` free. Memory growth limits eager reservation; it is not a hard
allocator cap.

## Attempt Ledger

| Attempt | Outcome | Classification | Repair or decision |
| --- | --- | --- | --- |
| `cpu_reference_attempt01` | Obsolete `[0]` indexing reduced vector ancestors to a scalar | Harness failure; not candidate evidence | Removed scalar indexing and passed the exact-arm shape smoke |
| `cpu_reference_attempt02` | Bitwise repeatability rejected float32 values equal at printed precision | Harness failure; not candidate evidence | Installed and recorded absolute tolerance `<=1e-5` |
| `cpu_reference_attempt03` | `PASS_CPU_REFERENCE_PRECHECK` at canonical scope | Nonclaiming CPU reference | Authorized the shared-GPU claim run |
| `gpu_attempt01` | `PASS_ENGINEERING_RUNG1` at canonical scope | Terminal rung-1 engineering evidence | Proceed only to a fresh coupled nonlinear plan |

Artifacts:

- `docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/cpu_reference_attempt01/failure_result.json`
- `docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/cpu_reference_attempt02/failure_result.json`
- `docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/cpu_reference_attempt03/result.json`
- `docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/gpu_attempt01/result.json`

The first two failures invalidated their harness executions, not the target,
math, proposal candidate, or research direction. The repaired attempts stayed
within the user-authorized continuation budget and used fresh output paths.

## Engineering Ledger

Implemented and exercised in the terminal GPU artifact:

- actual squared-TT fits to normalized initial and adjacent Gaussian targets;
- algebraic full-support coordinates and positive defensive mass;
- paired-core prefix marginalization for conditional proposal density;
- batched fixed-grid inverse-CDF generation with scalar-route parity tests;
- 24 independent proposal-uniform streams under one shared ancestor genealogy;
- complete pointwise APF importance correction after float32 state rounding;
- matched predictive-auxiliary and uniform-auxiliary exact reference arms;
- fixed randomness and an analytical recursive score of the same finite scalar;
- TensorFlow float32, TF32, XLA, GPU placement, and verified memory growth; and
- fail-closed structured artifacts for execution role and every promotion gate.

The finite-grid inverse remains a diagnostic numerical approximation. Its
roundtrip and density checks do not prove that it samples the fitted proposal
law exactly or that the resulting likelihood scalar is an unbiased estimator.

## Source-Anchor Audit

| Operation | Paper anchor | Pinned author-source anchor | Classification |
| --- | --- | --- | --- |
| Squared-TT density plus defensive mass | Eq. (13), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/SIRT.m:74` | `source_faithful` operation; the tuned mass value is an `extension_or_invention` |
| Paired-core marginal and generic prefix conditional | Proposition 2 and KR construction, `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:592` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43` | `source_faithful` operation only |
| Frozen uniforms, settings, sampling, and correction | Algorithm 3, `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:890` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21` | `fixed_hmc_adaptation` |
| `(previous,current)` prefix order | Zhao-Cui filtering order differs at Eq. (20), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807` | Author conditional API inspected above | `extension_or_invention` |
| Fixed-grid trapezoid/bisection inverse | Paper uses algebraic CDF construction plus root finding, `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:651` | Author uses `CDFconstructor`/`invert_cdf` in `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:78` | `extension_or_invention` |
| Block product, fixed-branch APF scalar, and analytical score | Not present as this route in Zhao-Cui | Not present as this route in pinned author code | `extension_or_invention` |

The paper and pinned author snapshot were inspected directly for this audit.
No source-faithfulness claim is made for the assembled compiler or APF route.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the declared synthetic engineering rung |
| Viable candidates | The selected fitted-TTSIRT uniform-auxiliary arm remains viable; both exact reference arms also passed their engineering checks |
| Statistically supported ranking | None |
| Descriptive-only differences | One-branch ESS, value, score, fit, timing, and cross-arm differences |
| Default readiness | Not assessed |
| Next evidence needed | Scope-specific tuning and a coupled nonlinear multi-seed block/rank ladder with uncertainty-aware downstream diagnostics |

## Post-Run Red Team

The strongest alternative explanation is that the independent product Gaussian
target is unusually favorable: it can hide cross-coordinate rank growth and
nonlinear conditional mismatch. The short `T=3` horizon can likewise hide
sequential weight collapse. A coupled nonlinear rung that fails ESS, measure,
or same-scalar score gates would reject transfer of this block factorization,
even though it would not by itself reject every fixed-TTSIRT/APF architecture.

The weakest evidence is the single frozen branch and diagnostic finite-grid KR
inverse. A defensible scientific or default claim would require a source-scoped
model, target-specific tuning, untouched multi-seed evaluation, uncertainty
analysis, and an eligible production conditional-transport implementation.

## Verification

Canonical GPU command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 python \
  docs/benchmarks/run_zhao_cui_blockwise_ttsirt_apf_rung1.py \
  --output-root docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/gpu_attempt01 \
  --dimension 24 --time-steps 3 --particle-count 256 --seed 220723
```

Focused CPU-only regression command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py \
  tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py \
  tests/highdim/test_transport.py \
  tests/highdim/test_p83_minimal_source_route_transport_slice.py
```

Result: `21 passed` in `9.68 s` with two third-party TensorFlow Probability
deprecation warnings.

The first post-audit regression run reported six metadata failures because the
classification repair had also changed `source_contract_level`, an existing
API-capability discriminator used by the P83 protocol. That was a localized
metadata compatibility error, not numerical or candidate evidence. The repair
restored `source_contract_level="fixed_ttsirt"`, documented that it is not a
source-faithfulness claim, and kept the independent route and operation
classifications. The complete focused suite then passed.
