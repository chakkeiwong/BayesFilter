# Kalman/UKF compiled runtime repair result

The active TensorFlow Kalman/UKF numerical kernels and the identified analytical
derivative callers now use tensor algebra and native TensorFlow recurrences.
The final audit found no numerical Python-loop violations in its guarded routes.
Square and fixed-branch rectangular SRUKF targets, both repaired QR derivative
primitives, and the actual native HMC transition archive passed CPU/XLA and
GPU/XLA engineering qualification. This is an implementation result; it does
not qualify the integrated DZ5 target or establish a speedup.

Baseline: Git `21828174`. The repairs are local changes on that baseline.
Plan: [compiled runtime repair](kalman_ukf_compiled_runtime_repair_20260916.md).
Consumer instructions: [MacroFinance handoff](kalman_ukf_macrofinance_handoff_20260916.md).
All paths below are relative to
`docs/plans/artifacts/kalman-ukf-runtime-20260916/`.

## Repair and audit boundaries

| Route | Repair and execution boundary |
| --- | --- |
| Covariance, QR, SVD, masked and correlated Kalman | Native date recurrences; batched parameter algebra, gains, QR/Cholesky derivatives, and analytical QR Hessians. |
| Batched linear Kalman | Shared fixed-shape compiled recurrence; independent chains remain tensor batch axes. The generic LGSSM adapter calls the existing batched analytical QR score. |
| Direct and rectangular factor SRUKF | Batched QR derivative directions, rectangular normal-space `dQ` term, optional joint transition callback, numeric validity outputs under outer XLA. |
| SVD/cubature/UKF/CUT sigma-point routes | Native date loops, tensor history buffers and batched gain derivatives; explicit branch vetoes survive ignored XLA assertions. |
| Factor downdates | Sequential matrix rotations use a TensorFlow loop; chains and derivative directions stay batched. |
| Fixed SGQF | One compiled tensor value/analytical-score recurrence with first-failure status and histories. Python step records are assembled after the kernel; `fixed_sgqf_tensor_result` is the enclosing-target API. |
| Owned model/filter callers | SSL-LSTM derivative algebra and parameter embedding, SIR/predator-prey RK4 recurrences, UKF scout, SV mixture UKF wrappers and the retained linear moment recurrence. |
| HMC mechanics reference | The QR test target supplies its analytical score through a custom-gradient bridge. Autodiff remains an independent diagnostic. The existing native archive runner requires no implementation change for the qualified fixture. |

`scripts/audit_kalman_ukf_runtime.py` discovered 734 Python files and followed
200 modules through static imports and the public linear/nonlinear lazy-export
tables. `audit-v3/current.json` contains the inventory, source hashes, native
loop/compilation declarations and materialization boundaries. There are zero
guard violations and zero stale exceptions. The 113 exact-AST exceptions cover
fixed state schemas, trace-time metadata, shape/input validation, and host
reporting. A regression test inserts numerical work into an otherwise exempt
schema function and requires the audit to reject it.

The audit is a bounded static guard, not a proof about arbitrary callbacks or
every loop in the repository. The inventory retains independent NumPy/reference
filters, historical LEDH, adjacent TT/transport preparation, host orchestration,
and SSL posterior forecast simulation loops. Those are outside the repaired
numerical-filter claim. The experimental scalar target fallback is not the
batch-native runtime route. No rebuilt canonical per-particle LEDH UKF endpoint
was found; the historical implementation cannot establish canonical admission.
The NumPy MacroFinance adapter is explicitly identified as diagnostic/reference.

`audit-v3/baseline.json` is the pre-repair inventory. Its exception mismatches
include changed structural loops, so its violation count is not a count of
numerical defects. The earlier `audit-v2/current.json` was accidentally named
`current` while generated with `--baseline 21828174`; its embedded baseline
field identifies it correctly. Use `audit-v3` for the final comparison.

The factor changes retain the analytical QR/Cholesky rules in
`docs/chapters/ch12_factor_derivatives.tex` and the direct block-QR UKF in
`ch17_square_root_sigma_point.tex`. No autodiff or pfor replaces runtime scores;
no jitter, precision, point rule, QR sign, rank, support or pivot choice was
changed. The SIR loop rewrite preserves four steps of `.005` and the author's
fourth-stage half-step. The mathematical anchor is P30's `eq:p27-sir6`
(`docs/plans/bayesfilter-highdim-nonlinear-filtering-paper-first-scholarship-p30-zhao-cui-alg5c2-expanded-note-2026-06-03.tex:10727`).
The inspected author source is
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/sir_step.mlx`,
`matlab/document.xml` code cell, displayed lines 5–18 in the preserved
`phase1-zhao-cui-mlx-normalized-extracts-2026-07-12.md:175` artifact. This changes
the execution of the existing model operation, not the Zhao-Cui filtering route.

## Durable checks

| Artifact | Result | Wall seconds |
| --- | ---: | ---: |
| `recovery-v1/callers.junit.xml` | 35 passed | 28.58 |
| `recovery-v1/core.junit.xml` | 85 passed | 103.86 |
| `recovery-v1/reference_hmc3.junit.xml` | 13 passed | 18.07 |
| `recovery-v2/remaining_paths.junit.xml` | 53 passed | 133.04 |
| `recovery-v2/wrappers_callbacks.junit.xml` | 19 passed | 62.62 |
| `cpu-v5/result.json` | Engineering qualification passed | 36.26 |
| `gpu-v4/result.json` | Engineering qualification passed | 64.51 |

These are 205 successful test executions with some overlapping selections, not
205 distinct tests or a whole-repository suite. They cover independent finite
differences, analytical Hessians, chain isolation, nonlinear/off-origin inputs,
rank/support boundaries, invalid rows, and graph-size checks across parameter
counts and observation horizons. The last suite checks SV mixture UKF score
wrappers and the SIR/predator-prey callbacks. Commands, environment, test IDs,
source correspondence and checksums are collected in `final-v1/manifest.json`.

Final qualification used the existing `tf-gpu` Python environment, TensorFlow
2.19.1, float64 kernels, two intra-op threads, one inter-op thread and one
OpenBLAS thread. CPU is an explicit engineering reference with GPUs hidden.
GPU used physical GPU2, an RTX 4090, in the trusted execution context with
verified memory growth before initialization. Recorded allocator peak was
2,097,152 bytes; peak host RSS was about 1.68 GiB. No environment was installed
or changed. TF32 configuration is recorded, but these fixtures use float64.

| Compiled function | CPU cold / warm seconds | GPU cold / warm seconds |
| --- | --- | --- |
| Stack QR derivative, B4/P18/N9/K39 | 0.787 / 0.00143–0.00149 | 1.123 / 0.00087–0.00131 |
| Rectangular QR derivative, same dimensions | 0.434 / 0.00111–0.00115 | 0.717 / 0.00090–0.00115 |
| Square target, B2/P3/N2/T4 | 3.575 / 0.00077–0.00096 | 4.606 / 0.00143–0.00170 |
| Rectangular target, matched full-rank fixture | 3.736 / 0.00094–0.00105 | 5.057 / 0.00190–0.00202 |
| Native HMC archive, 3 transitions/L2 | 4.787 / 0.00247–0.00305 | 8.320 / 0.01166–0.01191 |

Cold time includes tracing, compilation and a synchronized invocation. Warm
numbers are two synchronized repetitions, descriptive only. They are not an
old/new performance comparison and cannot select a device for DZ5. Each exact
outer function traced once and has saved optimized HLO. Reachable graph bodies
contain no `PyFunc`, `PyFuncStateless` or `EagerPyFunc`. The native archive block
includes coordinate transport, mass whitening, target/analytical score, momentum
generation, leapfrog, Metropolis selection, transition recurrence and buffers.

Target values, scores and terminal factors/moments agree with the independent
execution mode within 1.1e-15 in these fixtures; finite-difference score gates
also pass. HMC replay errors are at most 2.3e-16 and the Hamiltonian identity
residual is at most 2.3e-16. Replay uses the actual archived momenta because TF
and XLA RNG streams differ. The outer functions are stable; warnings from
fresh convenience filter calls in the separate replay are not repeated traces
of those qualified outer functions.

## Recovery, failed attempts and limits

The machine crash preserved source and qualification JSON/HLO, but removed
temporary test logs. Earlier unarchived test summaries are not used as durable
evidence. A VS Code reload later lost the focused-test process handle without a
complete JUnit receipt; the 53-test selection was rerun into `recovery-v2`.
Two HMC reference attempts failed at eager/XLA autodiff boundaries. The repaired
sampler now consumes the analytical QR score; its independent curvature oracle
uses eager covariance/Joseph calculations with a non-pfor tape. Both failed
receipts are retained beside the successful third attempt.

The first final audit write failed because its directory did not exist. The
precreated `cpu-v4`/`gpu-v2` directories are empty; `cpu-v4` was refused by the
qualification script before numerical work. `gpu-v3` stopped before qualification
because sandbox CUDA initialization saw no device. The unchanged `gpu-v4`
trusted-context retry succeeded. These are harness/execution-boundary failures,
not evidence against the filter mathematics. Earlier `cpu-v1/v2/v3` and `gpu-v1`
artifacts remain preserved and are superseded for final-source claims.

The campaign retained its 90 CPU-minute / 20 GPU-minute bounds and short-command
timeouts. Surviving artifacts give exact wall times for their runs; pre-crash
temporary logs do not permit a complete aggregate CPU-time reconstruction.
No long sampling, tuning or training campaign was launched. No independent
agent review was requested; this result includes local code review and tests.

New modules and scripts pass focused Ruff checks; `git diff --check` passes.
Changed-file lint contains 65 inherited diagnostics; line references embedded
in three messages need normalization when comparing with the baseline. The
repository-wide lint run also finds existing debt and a malformed unrelated
`.gitignore` glob. This task does not claim repository-wide lint success.

Material limits remain:

- Python numerical loops could previously unroll at trace time; their removal
  does not prove an interpreter ran on every target evaluation.
- The native CPU/GPU qualification uses a full-rank rectangular fixture.
  Singular charts have focused tests, but this is not complete GPU coverage of
  every filter family or arbitrary model callback.
- Eager and XLA SVD computations can differ at roughly 3e-7 in value and 8e-7
  in score on existing fixtures. Like-mode comparisons retain their tolerances.
- Fixed SGQF's inherited eigenfactor branch still uses Cholesky-style
  solve/derivative semantics. This compilation repair preserves it and does
  not establish correctness or score eligibility on that branch.
- No new GPU profile measures residual host-controlled special-function loops.
  The saved DZ5 profile attributed 97.5% of host predicate transfers to
  MacroFinance gamma/CIR work; this repair does not show that cost is removed.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete the reusable runtime repair and handoff | Guarded numerical loops removed; actual outer target/archive XLA executed | Numerical, nonfinite, branch, callback and tracing checks pass for declared fixtures | Consumer callback/rank contract and remaining CIR synchronization cost | MacroFinance integrates and qualifies its exact target, then measures its device configurations | DZ5 speedup, scientific admission, posterior correctness, convergence or LEDH admission |

The strongest alternative explanation for a misleading success is that the tiny
fixtures omit the expensive CIR callback and the consumer's actual singular
directions. A failed integrated value/score/support comparison or failed outer
archive compilation would overturn consumer readiness. The retained evidence
supports the reusable implementation and its tested contracts only.
