# Phase 0 Result: Contract, Baseline, And Source Fingerprint

Date: 2026-07-11
Status: `PHASE_CLOSED_PHASE1_REVIEW_AGREED`

## Outcome

Phase 0 established a stable, reproducible read-only baseline. No Kalman QR
worker was active, the exact relevant source fingerprint remained unchanged,
and all eleven discovered CPU-row/GPU-preflight artifacts are classified
`historical_debug_only_nonresumable_under_repaired_schema`.

No algorithmic, test, benchmark, or historical runner source was edited. No
TensorFlow import, CUDA/GPU probe, XLA compile, or benchmark execution occurred.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Sufficient current evidence exists to begin surgical harness repair. |
| Baseline | Git HEAD `52ee244498988e046a6356f926003b581103083b`, dirty relevant paths, seven exact SHA-256 hashes, environment metadata, reset memo, and historical artifacts. |
| Primary criterion | Passed locally: strict inventory exists, historical files are non-promoting/non-resumable, opening/closing fingerprints match, and Phase 0 review converged. |
| Vetoes | No active worker, source drift, missing artifact, ambiguous historical disposition, or environment conflict remains. Phase 1 review is the final handoff gate. |
| Explanatory only | Historical compile failures, timing, graph size, and previous GPU visibility. |
| Not concluded | Implementation correctness, compile viability, timing rank, current GPU readiness, HMC/posterior/default/production/scientific validity. |

## Environment

- Interpreter: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`
- Python: CPython 3.13.13
- TensorFlow distribution metadata: `tensorflow==2.20.0`
- TensorFlow Probability module provider: `tfp-nightly==0.25.0`
- NumPy distribution metadata: `numpy==2.1.3`
- Conda: `CONDA_DEFAULT_ENV=tfgpu`, `CONDA_PREFIX=/home/ubuntu/anaconda3/envs/tfgpu`
- `CUDA_VISIBLE_DEVICES` and `XLA_FLAGS`: unset in the inventory process
- GPU identity: `not_probed_phase0`

Package versions were resolved with `importlib.metadata.packages_distributions()`.
This corrected the failed assumption that the import name
`tensorflow_probability` was also the distribution name; TensorFlow itself was
not imported.

## Source Fingerprint

| Path | Status | SHA-256 |
| --- | --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | untracked | `0002e3270785a0f699894a148673b5abadded66806836df96b728c5c31d420ac` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | modified | `9434c3e0ccbc53b51545eddaee673095595cb7003b60e446c192ca7a255d84a5` |
| `bayesfilter/linear/kalman_qr_tf.py` | modified | `cc99674daf80a3f26b230b38fa799a84162e3b4ca3ee459bcea3c58318013216` |
| `bayesfilter/linear/qr_factor_tf.py` | modified | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |
| `tests/test_linear_qr_batched_analytical_score_tf.py` | untracked | `b8525aaa7c9c5551e1ee80dfccc526c071f3145579e8ab0f977a88185f667e79` |
| `tests/test_linear_kalman_qr_derivatives_tf.py` | modified | `d7ae40b725124220a7a10d85ef6e94bec07e89dce31b659b51797b9c1f6dd7d3` |
| `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py` | untracked | `9ad4eaf9f837984be7dd52b14684e2b59097aa5183d92aece99924f2b515eadd` |

Opening and closing Git HEAD, relevant status, and all seven hashes matched
exactly. The full machine-readable record is
`docs/benchmarks/kalman_qr_batched_xla_repair_phase0_inventory_2026-07-11.json`.

## Confirmed Defects

1. `_batched_model_tensors` statically unrolls the batch axis.
2. Scalar analytical and reverse-mode correctness comparators statically unroll `B`.
3. Three analytical QR/Cholesky/covariance helpers statically unroll `P`.
4. The diagnostic batched-autodiff reduction is constructed outside
   `GradientTape`, explaining its `None` gradient and NaN fallback as a harness
   defect rather than evidence against the math.
5. Timings include `.numpy()`, full host transfer, and Python conversion.
6. All methods execute in one process, coupling their failure evidence.
7. Failed-row artifacts can be labeled `complete`; the historical runner then
   resumes them based only on that label.
8. JSON output permits non-standard NaN/Infinity tokens.
9. The historical runner can return success despite GPU-preflight failure when
   CPU rows do not fail.

The batched analytical time recursion itself uses `tf.while_loop`. Phase 0 does
not claim that previously tried vectorized derivative formulas are correct;
Phase 3 must revalidate them against the current loop implementation.

## Historical Disposition

- Nine CPU row artifacts were found: six report complete/parity-passed and
  three report `complete` despite five failed rows each.
- Two GPU preflight artifacts were found: both report `complete` despite their
  only row failing in the coupled autodiff-row-loop XLA path.
- The overnight supervisor reports `complete_with_failures`, three CPU
  failures, and `blocked_gpu_xla_autodiff_preflight`.
- Every discovered row/preflight artifact is historical debug evidence only.
  None is eligible for resume, promotion, method ranking, or current-source
  compile/runtime claims.

## Hypotheses To Test

| Hypothesis | Discriminating phase |
| --- | --- |
| Tensor-algebra fixture construction reduces B-axis graph duplication without changing values. | Phase 2 |
| Parameter-axis tensor formulas reproduce current derivative helpers and reduce P-axis graph growth. | Phase 3 |
| Moving reduction inside the tape repairs batched autodiff, subject to row-independence and scalar parity. | Phase 4 |
| Method isolation allows analytical GPU evidence even if reverse-mode GPU XLA remains invalid. | Phase 7 |

## Checks And Review Trail

- Strict metadata inventory used only Python standard-library package metadata.
- Opening and closing Python-process-only benchmark filters found no worker.
- Inventory discovery found exactly nine CPU and two GPU JSON artifacts.
- All referenced paths exist and their SHA-256 hashes were recomputed.
- Phase 0 Codex substitute review rounds 1 and 2 returned `REVISE`; the visible
  subplan was patched. Round 3 returned exact `VERDICT: AGREE`.
- The substitute review is weaker than Claude Opus review. Claude was
  policy-blocked before liveness probing, so this result makes no Claude
  availability or agreement claim.

## Decision Record

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 0 after Phase 1 review agrees | Passed locally | No Phase 0 evidence veto remains | Phase 1 plan consistency and boundary safety | Review the refreshed Phase 1 subplan, then implement harness-only repairs | No math, compile, timing, GPU, or scientific promotion |

## Handoff

Phase 1 was authorized to start after
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase1-harness-integrity-subplan-2026-07-11.md`
received exact `VERDICT: AGREE` in substitute-review round 3. The 2026-07-09 runner and all dated outputs
remain immutable historical evidence.
