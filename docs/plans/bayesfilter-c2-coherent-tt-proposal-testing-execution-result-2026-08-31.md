# C2 Coherent TT Proposal Testing: Execution Result

Date: 2026-08-31

Plan: `docs/plans/bayesfilter-c2-coherent-tt-proposal-testing-plan-20260831.md`

Status: Stage 0 and Stage 1 executed; Stage 2 executed with its predeclared
continuation veto; later conditional stages were not promoted.

## Evidence Contract

Question: does complete exact-factor importance weighting repair the finite
carried-density likelihood scalar, and does the retained TT proposal remain
usable at the C2 (n=4,T=20) scope?

Comparator: the unchanged Hermite retained-TT proposal, bootstrap conditional,
transformed-observation Student, Gaussian-hint, stationary Gaussian, and fixed
half TT/Student mixture arms, with the screened PF result used only as a
reference-compatibility diagnostic.

Primary screens: finite values and scores, complete-mixture denominator,
finite-target identity, direct-target/integration validity, and a paired
minimum-ESS contrast. A candidate loss to a constructed cheap adversary is a
promotion veto. ESS and likelihood differences are descriptive unless the
declared paired uncertainty evidence supports the specific contrast.

Nonclaims: no exact pseudo-marginal likelihood, exact posterior, HMC
readiness, default readiness, universal superiority, or source-faithful full
Zhao--Cui reproduction.

## Executed Stages

### Stage 0: invariants

Command (CPU-only by design):

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-c2-coherent-stage0-all \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  pytest -q -p no:cacheprovider \
  tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py \
  tests/highdim/test_c2_gaussian_hermite_proposal_tf.py \
  tests/highdim/test_c2_sv_frozen_fixture_diagnostic.py \
  tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py \
  tests/highdim/test_c2_transformed_observation_student_proposal_tf.py \
  tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py \
  tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py \
  tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py \
  tests/highdim/test_c2_coherent_plan_math.py
```

Result: **48 passed**, 0 failed, 0 skipped, 15.53 seconds. The JUnit artifact
is `docs/benchmarks/artifacts/c2_coherent_tt_proposal_testing_20260831/attempt01/stage0_all_junit.xml`.
The tests cover target/capture wiring, APF terms, finite-difference score
parity, Gaussian/Student/mixture proposal mechanics, the RBF Gram formula,
the Hermite antiderivative, the duplicate-constant singularity, the Student
moment boundary, and the C2 nonzero-observation envelope.

### Stage 1: exact-factor proposal ladder

Command:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py \
  --mode serious \
  --output-root docs/benchmarks/artifacts/c2_coherent_tt_proposal_testing_20260831/attempt01/stage1_serious
```

The run used TensorFlow `2.20.0-dev0+selfbuilt`, TFP `0.25.0`, float64,
TF32 enabled, XLA enabled, GPU memory growth, two visible GPUs, and twelve
paired seeds at 8,192 particles over 20 steps. The run manifest is
`docs/benchmarks/artifacts/c2_coherent_tt_proposal_testing_20260831/attempt01/stage1_serious/run_manifest.json`.

| Family | Mean total | Minimum observed normalized ESS | Engineering |
| --- | ---: | ---: | --- |
| retained TT | -68.4726003 | 0.000198658 | PASS |
| bootstrap conditional | -66.6980230 | 0.119788 | PASS |
| transformed Student | -66.7189193 | 0.122834 | PASS |
| fixed half TT/Student DMIS | -66.6822291 | 0.0897829 | PASS |
| Gaussian hint | -66.7334317 | 0.0852800 | PASS |
| stationary Gaussian | -66.8102974 | 0.00247346 | PASS |

The paired DMIS/retained-TT minimum-ESS ratio was 5.3839341 with 95% bootstrap
interval [5.1832197, 5.5999536], positive in all 12 branches, exact two-sided
sign-test p = 0.00048828125. This supports the predeclared paired ESS
contrast, not a global family ranking. The fixed-half DMIS total differed from
the stored PF reference by 0.0157258 nats; that comparison is descriptive and
is not an exactness certificate.

The heuristic-dominance veto fired because simpler Gaussian/Student/bootstrap
arms were better at predeclared salient times. The full branch result is
`docs/benchmarks/artifacts/c2_coherent_tt_proposal_testing_20260831/attempt01/stage1_serious/result.md`.

### Stage 2: independent finite-target integration

The first launch was rejected before computation because the legacy driver
requires a direct child of `c2_n4_root_cause_20260828`. That localized harness
repair did not change the target, settings, hardware, or budget. The repaired
run used:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python docs/benchmarks/diagnose_c2_n4_frozen_target_stage2_20260828.py \
  --output-root docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt05_coherent_20260831 \
  --mode formal --jit-compile
```

At (t=3), the independent decomposition found a material fit term
(+1.524930\pm0.001131) and a material inherited-state term
(-0.012090\pm0.002958). At (t=2), the QMC half-width was 0.001281,
slightly above the 0.00125 limit; at (t=4), it was 0.400877. The declared
hard veto `christoffel_qmc_uncertainty` therefore fired, and the result is
`STAGE_2_UNRESOLVED`, not a causal classification. The result is preserved in
`docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt05_coherent_20260831/stage2_result.md`.

## Decision Table

| Decision | Primary criterion | Veto/status | Next justified action |
| --- | --- | --- | --- |
| exact-factor DMIS route | engineering and score screens pass | no finite-value or denominator veto | retain as correctness comparator |
| retained-TT proposal | paired ESS contrast fails practical-utility comparison | severe ESS collapse | use only as a proposal component/control variate until repaired |
| fixed half mixture | finite and paired ESS contrast passes | heuristic veto; alpha/nu not calibrated | tune alpha/nu only after target-integration precision is repaired |
| recursive map | not implemented in current call chain | Stage 2 target integration veto | repair QMC/integration evidence, then implement lagged moment contractions |
| basis ladder | not executed | conditional on Stage 2/map boundary | frozen one-step fitter study at first divergent time |
| Student TT measure | not executed | no promotion trigger | keep Student as proposal; product-Student measure is a separate future route |

Inference status: hard engineering screens passed; the paired ESS contrast is
statistically supported; the broader family ranking is not supported; all
normalizer and proposal-quality differences beyond that contrast are
descriptive; default readiness is false; and the next evidence needed is a
precision-repaired independent finite-target integration followed by the
frozen one-step fitter test.

## Artifact Digests

```text
stage0_all_junit.xml  d958a48595f4bd071017fe87f13e2920f9894fc0667e7e412afec4412f59b3a6
stage1_serious/result.json  4d369243ba9c05197099f565930442fae2562e70f41084cbb735d12d6bdb9dee
stage1_serious/result.md  b1ec00f2d4ff1ec6d044aa6f11ec6414bf7959341497fe7ba16957e6ae4f314f
stage2_result.json  1fa83052fef5f84208661849d904c8895d9f73a4e279deded32c03eb4c2bb708
stage2_result.md  21216cd949a87c1c05df38da5efdd429af695ff4094fcfb4fba64782ca4cbe96
```

The manuscript and MathDevMCP audit identities are recorded in the plan's
documentation-audit section. No later conditional stage is promoted by this
execution result.

Final manuscript artifacts: the 27-page PDF is
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.pdf`
(SHA-256 `81e36d1acdbc32600f0d621f002e7a08113f8bf76241d667bcf8fff4c759c799`),
and the audited TeX source digest is
`11d8622befa67e4d00d51b0f425442e09969a6ada143f2b977dc67a1d21ada34`.
The final proof-presentation repair handles the $r=1$ Hermite case explicitly;
it does not alter the mathematical target or the execution contract.
