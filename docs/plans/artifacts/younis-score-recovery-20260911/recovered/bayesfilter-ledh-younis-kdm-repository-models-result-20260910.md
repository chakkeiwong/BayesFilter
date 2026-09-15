# Existing-Model KDM Correctness Batch

Date: 2026-09-10
Status: FIRST_FIVE_ADAPTER_BATCH_PASS; ALL_MODEL_TESTING_INCOMPLETE.
Plan: `docs/plans/bayesfilter-ledh-younis-kdm-repository-model-testing-plan-20260910.md`.

## Outcome

The first cross-model audit reproduced and repaired missing analytical
derivatives in native generalized SV and KSC, and a graph-construction defect
in the Austria SIR adapter. All five existing factories now pass all-coordinate
and mixed-direction callback checks and actual two-step canonical/raw-IWSG
endpoint checks. The terminal run passed 39 tests in 150.158 seconds.

This does not finish testing the repository's models. Actual SV has no adapter
in this module, several other families remain outside this endpoint, the
tested initial clouds are fixed, and no new model-score accuracy comparison
has been run. Further DSGE implementation is deferred by the owner's redirect.

## Findings And Repairs

1. Native generalized SV omitted the process-covariance derivatives with
   respect to both log innovation scales, and the direct beta dependence of
   the flow observation and its Jacobian. The callback used `H dx` where
   `H dx + dH x` is required. Added those three callbacks/terms without
   changing the value program or heteroskedastic likelihood.
2. KSC's flow observation offset is `2 log(beta)`. Its tangent omitted
   `2 d log(beta)`, despite the likelihood tangent already including it.
   Added the missing flow term. Corrected the false docstring claim that
   the approximate KSC mixture must have exactly the actual-SV score.
3. Austria's adapter constructed the eager-validated `SpatialSIRSSM` while
   tracing, triggering `.numpy()` on a symbolic tensor. It now obtains the
   same constant adjacency from the existing neighbor-set helper. An
   executable equality test checks the original model's adjacency.
4. The first endpoint harness incorrectly required a nonzero pairwise
   configuration in scalar KSC. Its exact absence is now asserted; all four
   multidimensional cases require a nonzero off-diagonal target and correction.

The touched adapter module no longer imports NumPy; scalar mathematical
constants use `math`. NumPy remains only in this independent diagnostic test.
The final module docstring corrects its stale six-model claim and describes
parameter capture and the SIR target boundary explicitly.

## Executable Results

Both finite-difference step sizes, `2e-5` and `5e-6`, pass. Each row includes
every parameter coordinate plus a mixed direction. The error is
`abs(analytical - FD) / max(1, abs(FD))`, maximized over both step sizes and
all directions. All models use `T=2`, nonconstant initial covariance marks,
Contract-E, the actual UKF covariance lifecycle, diagonal correction, both
caps, and pairwise correction wherever dimension exceeds one.

| Adapter | D / N | Directions | Canonical max relative error | Raw-IWSG replay max relative error |
|---|---|---|---|---|
| Diagonal LGSSM | 3 / 12 | 6 | 1.400e-10 | 9.590e-10 |
| Native generalized SV | 2 / 8 | 6 | 9.027e-11 | 4.607e-10 |
| KSC SV | 1 / 4 | 3 | 1.514e-10 | 1.237e-10 |
| Predator-prey | 2 / 8 | 7 | 2.496e-8 | 3.982e-9 |
| Austria continuous SIR recurrence | 18 / 72 | 4 | 8.812e-10 | 6.495e-9 |

The full endpoint test builds each adapter from symbolic current theta, binds
the direction before tracing, and uses one explicit stable signature per
canonical/anchor/replay function. It asserts exactly one trace, direction-
invariant values, valid correction/resampling, exact anchor replay, all-pairs
density work, and heterogeneous carried covariance marks. Its bandwidth is
`B=0.2 Q(theta)` with the complete `dB`; finite-difference perturbations
recompute it. Anchor samples, proposal densities, and component indices remain
frozen in replay. This tests the actual public endpoint call chain, not a
separate implementation of the filter.

Six existing shared tests also pass: zero-bandwidth full-trajectory identity;
joint covariance/map/bandwidth derivatives; sequential anchor/replay parity;
canonical no-hook parity; next-step state/weight/covariance consumption; and
next-step log-weight-tangent consumption. No tolerance was relaxed.

## Remaining Discrepancies

**Initialization is not yet a complete model-score implementation.**
`ledh_canonical_score_tf.py` initializes state and covariance tangents to zero.
The native SV initial law in `native_generalized_sv.py::initial_log_density`
has scales `sigma / sqrt(1-rho^2)`, which depend on theta. The current tests
deliberately hold their supplied initial cloud/covariance fixed. They therefore
cannot justify a total-score claim for a parameter-dependent initialization.
Carry the initial-state and covariance tangents through the shared executor
and both KDM endpoints before model-score comparisons that use that law.

**Austria target identity remains wrong if described as the clipped simulator.**
`models.py::zhao_cui_sir_austria_model` selects
`clip_susceptible_after_noise`. The current canonical adapter uses continuous
additive Gaussian states and no clipping operation. It also is not identical
to `sir_latent_preclip_tf.py::LatentPreclipSIRSSM`, which maps earlier latent
states through the clipping map before later transitions. The derivative
test proves the derivative of the implemented continuous recurrence, not
equality to either of those two targets. Resolve this existing-model mismatch
with the already-defined latent-preclip convention and time indexing; do not
silently rename the recurrence or substitute a Gaussian noise floor.

**Coverage is five existing factories, not all models.** Actual SV, prior-mean
generalized SV, the common-suite variants, coupled nonlinear blocks, structural
fixtures, hard-bound targets, and SSL-LSTM are recorded in the active plan.
Some need adapters, some need support/API work, and all need explicit target
and reference checks. A historical UKF/SGQF/HMC result is not a KDM result.

**No new scientific utility evidence.** The completed matrix-LGSSM campaign
remains negative in its tested scopes. These repairs are not a reason to
reinterpret those results: that campaign used its own complete matrix-model
callbacks. No derivative or pass count establishes that KDM helps elsewhere.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept localized repairs | All 39 terminal checks pass | No terminal correctness/numerical veto in these fixtures | Untested regimes and parameter-dependent initialization | Extend existing-model correctness and identity coverage | All-model faithfulness |
| Keep KDM unpromoted | No new score-error comparison | Previous matrix-LGSSM promotion veto unchanged | Cross-model error and adequate statistical power | Fresh per-model calibration/pilot/holdout after correctness | KDM benefit or direction rejection |

| Inference status | Finding |
|---|---|
| Hard veto screen | Six derivative failures and one graph-wiring defect reproduced, repaired and retested; SIR target identity remains unresolved. |
| Statistically supported ranking | None from this deterministic batch. |
| Descriptive-only differences | Endpoint runtimes and finite-difference residual sizes; not method rankings. |
| Default-readiness | Not established; CPU/fp64 reference exception only. |
| Next evidence needed | Initial-law and target-identity fixes; remaining-model endpoints; GPU/XLA parity; adequately powered model-score comparisons. |

Post-run red team: the strongest alternative to a claim of complete
correctness is that both value and tangent compute the wrong model, especially
at initialization and SIR boundaries. That remains a real limitation, not a
qualification removed by small FD residuals. Different parameter points,
longer horizons, boundary regimes, and GPU/TF32 may expose additional defects.
The weakest part of coverage is that the small fixtures intentionally isolate
recurrence derivatives rather than reproduce each model's full data regime.

## Run Manifest And Attempts

Git base: `12f8a208`, branch `kdm-total-score-continuation-20260909`, isolated
worktree `.claude/worktrees/kdm-score-campaign-20260909`. Tests ran against the
recorded local source changes. Interpreter:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; TensorFlow
`2.20.0-dev0+selfbuilt`. CPU only, GPU deliberately hidden by
`CUDA_VISIBLE_DEVICES=-1`; `jit_compile=False` diagnostic exception, two
intra-op and two inter-op threads. Data: synthetic fixed diagnostic tensors,
not an external dataset. Seeds: `2026091017` callbacks, `2026091019` endpoints.
Endpoint sizes and directions are in the table and JUnit properties.

Artifact root:
`docs/benchmarks/artifacts/ledh_younis_kdm_repository_models_20260910/`.

| Attempt | JUnit receipt | Tests / failures | Pytest wall seconds | Classification |
|---|---|---|---|---|
| 01 | `attempt01/callbacks.xml` | 26 / 6 | 8.687 | Reproduced missing derivatives before repair. |
| 02 | `attempt02/repaired_and_endpoints.xml` | 28 / 1 | 25.859 | Derivatives fixed; scalar-pairwise harness assertion wrong. |
| 03 | `attempt03/five_model_audit.xml` | 32 / 1 | 110.171 | Four endpoints pass; Austria graph construction fails. |
| 04 | `attempt04/five_model_and_shared_regression.xml` | 39 / 0 | 150.158 | All terminal checks pass. |

Total measured pytest wall time: 294.875 seconds, plus interpreter startup
overhead. All four launches ended below their 300-second hard timeout, within
the 900-second batch budget. The four-attempt batch is complete; do not treat
unused seconds as authority for an unplanned statistical sweep.

All four commands used this prefix from the isolated worktree:

```sh
env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 timeout 300s python -m pytest
```

Attempt 01 arguments:

```sh
tests/highdim/test_ledh_younis_kdm_repository_models.py -q --junitxml=docs/benchmarks/artifacts/ledh_younis_kdm_repository_models_20260910/attempt01/callbacks.xml
```

Attempt 02 arguments:

```sh
tests/highdim/test_ledh_younis_kdm_repository_models.py -q --tb=short --junitxml=docs/benchmarks/artifacts/ledh_younis_kdm_repository_models_20260910/attempt02/repaired_and_endpoints.xml
```

Attempt 03 arguments:

```sh
tests/highdim/test_ledh_younis_kdm_repository_models.py -q --tb=short -o junit_family=legacy --junitxml=docs/benchmarks/artifacts/ledh_younis_kdm_repository_models_20260910/attempt03/five_model_audit.xml
```

Attempt 04 arguments:

```sh
tests/highdim/test_ledh_younis_kdm_repository_models.py tests/highdim/test_ledh_younis_kdm_integrated_tf.py::test_zero_bandwidth_is_exact_full_trajectory_call_chain_identity tests/highdim/test_ledh_younis_kdm_integrated_tf.py::test_full_feedback_total_tangent_includes_model_covar_map_and_bandwidth tests/highdim/test_ledh_younis_kdm_resampling_tf.py::test_sequential_anchor_replay_and_total_tangent_match tests/highdim/test_ledh_younis_kdm_resampling_tf.py::test_post_reset_extension_preserves_canonical_no_hook_behavior tests/highdim/test_ledh_younis_kdm_resampling_tf.py::test_post_reset_state_weight_and_covariance_values_reach_next_pfpf_step tests/highdim/test_ledh_younis_kdm_resampling_tf.py::test_post_reset_log_weight_tangent_is_consumed_by_next_pfpf_logits -q --tb=short -o junit_family=legacy --junitxml=docs/benchmarks/artifacts/ledh_younis_kdm_repository_models_20260910/attempt04/five_model_and_shared_regression.xml
```

Terminal-run SHA-256 values:

| File | SHA-256 |
|---|---|
| `ledh_canonical_models_tf.py` at execution | `ff643a3c8e77983b676000b3005fb634bfe2043a190c62082ce76b835d3facc1` |
| `ledh_canonical_score_tf.py` | `a87928049fe5f47bfa5e899f9b5e5de38bb817eac417d235f31dd94a89d8c26a` |
| `ledh_younis_kdm_resampling_tf.py` | `cc85b71a27b5aed686cf90b452013054f73eff0314248aaee59146311903e68b` |
| `ledh_younis_kdm_integrated_tf.py` | `3f07f1b769e83adfe54b722d5aa228cb3f40aa602c8fb986db7dfb47d13e21a0` |
| `_symmetric_sylvester_ops.so` | `661f11b9db1f6e9ab9ce4aae8ae86591779cdbdfe6fda777a7c87956ec2686fa` |
| `test_ledh_younis_kdm_repository_models.py` | `789bf4e2ce42c209db3f5e84f32b10c8683480cecb66f421daa1f9c871aa0a4c` |
| Attempt 04 XML | `069d05b392c64fb82f5173e0a9487eb153be8793a0c7c4fe146fde0c6efda297` |

After execution only the adapter's module docstring changed. Its AST excluding
that docstring, before and after, hashes to
`767964992cfa939b3b33f161cb68ffbe7cdfdc8a838ddc4c4c2b1d08ee3e79d0`.
Other dependencies remain at the base commit; the recorded custom op is an
ignored local build. Two TensorFlow Probability distutils deprecation warnings
were emitted; there were no skipped tests.

## Document Consistency

The LaTeX keeps the existing KDM derivations and negative LGSSM evidence, defers
further DSGE work, and derives the newly repaired SV flow terms explicitly.
Its Section 8 discussion distinguishes a correct derivative of the supplied
finite program from agreement with the model score. The 20-page PDF builds
with two `pdflatex -interaction=nonstopmode -halt-on-error` passes; changed
equations on page 19 and the remaining-model discussion were inspected in the
rendered PDF. Human readability
review remains pending. No previous equation, citation, or empirical finding
was removed.
