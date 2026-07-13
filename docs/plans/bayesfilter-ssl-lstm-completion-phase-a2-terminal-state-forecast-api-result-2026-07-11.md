# Phase A2 Result: Terminal State And Forecast API

Date: 2026-07-13 (Asia/Shanghai)

Status: `PASSED_FOR_A3_PLANNING_ONLY`

## Outcome

Phase A2 produced a typed TensorFlow `float64` terminal-state and ten-step
forecast API for the A0/A1-locked scalar SSL-LSTM target. It extracts the
complete three-coordinate terminal filtered Gaussian from the accepted
historical SVD-UKF route, admits covariance roundoff through the reviewed
symmetric principal eigen-square-root policy, and simulates complete state and
observation paths from externally materialized stateless-Philox innovation
banks.

The implementation now fails closed on nonfinite free draws, every materialized
innovation tensor, every terminal diagnostic, and every eager or compiled
forecast output. Materialized tensor hashes are the replay authority. Philox
seeds are generation metadata and are not represented as cross-backend bitwise
floating-normal regeneration evidence.

The exact CPU-hidden focused suite passed `87/87`. Fresh CPU/XLA generation and
full fresh-process replay verification passed all 15 hard checks. Fresh trusted
GPU/XLA generation and full trusted replay verification passed all 17 hard
checks on two RTX 4080 SUPER GPUs. The bounded implementation review returned
`VERDICT: AGREE`; it is a Codex substitute review, explicitly weaker than
Claude.

This result authorizes only A3 subplan drafting and bounded review after the
pre-result checkpoint and A2 result review pass. It does not authorize A3
implementation, forecast-moment inference, HMC, NeuTra, calibration,
predictive equivalence, model adequacy, product/default changes, or scientific
claims.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the bounded A2 terminal/forecast engineering surface for A3 planning | Focused suite plus independently verified CPU/XLA and trusted GPU/XLA artifacts passed conjunctively | No active A2 engineering veto; all terminal status, finite-value, covariance, replay, recursion, parity, XLA, placement, crosslink, and write-boundary checks passed | The historical filter is approximate, the design uses ten frozen parameter points and two forecast draws with two replications, and production/oracle lineage remains untested independently | Generate the executor ledger and pre-result checkpoint, obtain a hash-bound result review, then draft and review the A3 oracle/statistics subplan | Posterior correctness, exact nonlinear filtering, predictive equivalence, calibration, model adequacy, HMC/NeuTra readiness, superiority, performance, public/default/product/release readiness, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the bounded A2 terminal extraction and ten-step forecast engineering contract |
| Statistically supported ranking | Not applicable; no stochastic method ranking or sampler comparison ran |
| Descriptive-only differences | CPU/GPU residuals, covariance residuals, compiler diagnostics, devices, and wall times are descriptive engineering observations only |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | Independent analytic LGSSM oracle and predictive-statistics machinery in A3, then later reviewed calibration, sampler, equivalence, and audit phases |

## Separate Evidence Ledgers

| Ledger | Status | Evidence and boundary |
| --- | --- | --- |
| Engineering correctness | `passed_a2_bounded_terminal_forecast_surface` | `87/87` focused tests, including 18 terminal-trace parser regressions; CPU and trusted GPU generation plus fresh replay verification; exact bank/path replay; finite admission; covariance, recursion, parity, XLA, placement, and fail-closed trace gates |
| Numerical/sampler validity | `not_assessed` | No HMC chain, convergence, posterior-reference, or sampler-validity run occurred |
| Computational predictive equivalence | `not_assessed` | No moments, MMD, simultaneous equivalence interval, or method comparison ran |
| Synthetic generative calibration | `not_assessed` | No replicated data, coverage, SBC, PIT, or calibration experiment ran |
| Empirical model adequacy | `not_assessed` | No application-data or held-out adequacy experiment ran |

## Research Intent Ledger

| Field | A2 disposition |
| --- | --- |
| Main question | Can the accepted A1 target expose a complete, replayable, fail-closed terminal-state and ten-step forecast surface on the repository TensorFlow/XLA route? |
| Candidate/mechanism | Approximate historical SVD-UKF terminal Gaussian, symmetric principal covariance square root, complete-state draw, and fixed materialized innovation-bank recursion |
| Expected failure mode | A1 drift, invalid terminal covariance, nonfinite input/output, wrong process-noise placement or observation timing, hidden RNG, replay failure, CPU/GPU mismatch, XLA/placement failure, or invalid artifact |
| Promotion criterion | Full focused suite plus structured CPU-hidden and trusted GPU/XLA canaries pass conjunctively and survive fresh-process verification |
| Promotion veto | Any protected hash, finite/status, covariance, parity, replay, recursion, XLA/HLO, device, crosslink, cache, or write-boundary failure |
| Continuation veto | Invalid target/data/math, required A1 change, corrupt evidence, unsupported environment, or required action outside the reviewed write set/authority |
| Repair trigger | The implementation review exposed missing finite admission and ambiguous provenance; the repair added fail-closed admission and clarified materialized replay authority |
| Explanatory diagnostics | Residual magnitudes, covariance eigenvalues/residuals, trace counts, compiler messages, device inventory, and wall time |
| Must not be concluded | Posterior equality/correctness, predictive equivalence, calibration, sampler validity, model adequacy, superiority, performance, or default/product readiness |

## Skeptical Audit At Closeout

| Risk | Result |
| --- | --- |
| Wrong baseline | Avoided: the comparator remained the accepted A1 historical SVD-UKF target and artifacts |
| Proxy promotion | Avoided: focused tests and CPU/GPU parity promote only bounded A2 engineering status |
| Missing stop condition | Finite/status, covariance, target parity, recursion, replay, XLA/HLO, GPU placement, artifact, review, and write-boundary vetoes were enforced |
| Unfair comparison | CPU and GPU consumed the identical persisted hexadecimal bank, points, target, dtype, config, and horizon |
| Hidden assumption | Approximate filter, terminal Gaussian, complete-state draw, process/observation noise placement, horizon, dtype, cluster unit, and replay authority are explicit |
| Stale context | The repaired source/test/generator hashes were refrozen before the full rerun; A1 protected hashes and `HEAD` were rechecked |
| Environment mismatch | CPU deliberately hid GPU and is labeled reference-only; serious evidence used trusted GPU/XLA with device/JIT/TF32 provenance |
| Artifact insufficiency | Both artifacts contain tensors, terminal diagnostics, HLO text/hashes, trace counts, device placement, source/input hashes, crosslinks, commands, and run manifests and passed fresh replay |

Audit status: `PASSED_FOR_A2_ENGINEERING_CLOSEOUT_AND_A3_PLANNING_ONLY`.

## Implementation Inventory

| Role | Path | SHA-256 |
| --- | --- | --- |
| Production terminal/forecast module | `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py` | `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed` |
| Lazy nonlinear exports | `bayesfilter/nonlinear/__init__.py` | `674679585bc57f1b8ba68f44db28205f202bb8fba1e3a6f47110d604fc801ed4` |
| Focused A2 tests | `tests/test_ssl_lstm_predictive_tf.py` | `1812b338ff90633d2fa627642af8ba65425bdaf1c11211f8944d7207ecbded2c` |
| Artifact generator | `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py` | `8738e6dc0ea2162e71f99a4a53397c772a01b9e1113973221d8c10e3162e97b1` |
| Independent verifier | `docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py` | `d0195063a1686a5332b6788bd1171ffc998370bd3578ceeb64edea240a2511ee` |
| Protected A1 target | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py` | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| Accepted A2 subplan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md` | `6b6b9799782be3304ecbd2dee465c52285688b5e2d1b3087d911ccad1279bbb0` |
| Accepted Round 6 subplan review | `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round6-2026-07-12.md` | `846574f1d6140efd5ff8e10f772f0d886be916585f30ccdac6960bd1eacfeaa1` |
| Implementation and terminal-trace review | `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-implementation-codex-substitute-review-2026-07-12.md` | `1210e2fcced29448cbcdba7a4ce1dcee93326e3f317e27ec65d45c30364f23fb` |

The public predictive module supplies typed terminal, config, innovation-bank,
path, and provenance dataclasses; scalar and statically sized batch extraction;
scalar and batch forecast surfaces; reusable XLA-compiled terminal and forecast
programs; and an eager debug/reference forecast surface. A2 adds only the five
reviewed predictive dataclasses to the lazy package exports.

## Local Checks And Focused Tests

| Check | Result |
| --- | --- |
| In-memory static compilation | Passed for production, tests, generator, and verifier |
| Whitespace audit | Passed |
| Backend/import scan | No NumPy, PyTorch, JAX, or benchmark import in production |
| RNG scan | No stateful forecast RNG; only reviewed stateless Philox generation |
| A2-named repository cache scan | Empty before/after focused and runtime commands |
| Protected A1 target hash | Matched exactly |
| Exact CPU-hidden focused suite | `87 passed, 15112 warnings in 6795.67s` |
| Closed focused trace SHA-256 | `9bc681ce9071cc73e94ee0be85e809871cd3486ce778bb6a591bf3cb3471cdaf` |
| Bounded implementation review | `VERDICT: AGREE`; Codex substitute, weaker than Claude |

The 18 new cases relative to the prior 69-test run exercise the fail-closed
terminal trace parser, including `readlink(...)` nonmutation classification,
malformed/truncated records, PID cardinality, syscall families, resolved
destinations, and component containment. The warnings are TensorFlow/TFP/gast
deprecation and XLA compilation warnings; no test failed.

## Boundary And Innovation Bank

| Field | Value |
| --- | --- |
| `HEAD` | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Pre-run boundary status | `A2_SCOPED_BOUNDARY_FROZEN` |
| Pre-run boundary SHA-256 | `674d5124c92fe093f807195039c9d4a243c5ec403563631d5569ea77d2259cfc` |
| Boundary trace SHA-256 | `78b3d81cdf1f378d54fbe4baa9f8726aa4bb3a032482bfa74d8dfbe5e53fa6be` |
| Innovation bank SHA-256 | `7940480cd92299e8078557cf5b8c61da1408ac8dcd218006f8e258c368c625b5` |
| Innovation evidence signature | `a8b322f4966ab05937d7776a72aa963bf15f4948bff3c8681eeadc9e9bd3d633` |
| Root seed / role / arm | `[20260712,1202]` / `paired_diagnostic_shared` / `0` |
| Terminal/process/observation tensor hashes | `2208c9103505f56dcbc1ec25b5b7547dddf0452644142254154951dcd09ab053` / `b58c2d1dd5c73574035d8748f60648dde20abce255509f08452ded528d3d2f5d` / `01e6c68bda42350e82b88496e07a02c6abda67cd2903df28ac92f8f3040b5798` |

The materialized tensor hashes are authoritative for replay. The seed, role,
arm, and family seeds document generation and are validated as integer
metadata; they do not assert cross-backend bitwise regeneration of transformed
floating normal values.

## Terminal And Forecast Diagnostics

| Diagnostic | CPU | Trusted GPU | Status |
| --- | ---: | ---: | --- |
| Terminal rows admitted | `10/10` | `10/10` | Passed |
| Minimum raw terminal-covariance eigenvalue | `3.150886552283595e-08` | `3.150886552284136e-08` | Positive; no material-indefiniteness veto |
| Maximum covariance symmetry residual | `0.0` | `0.0` | Passed |
| Maximum covariance projection residual | `2.2204466035905014e-16` | `1.1102232192710774e-16` | Passed reviewed tolerance |
| Maximum factor reconstruction residual | `2.220446458187691e-16` | `1.6653345614084944e-16` | Passed reviewed tolerance |
| Filter likelihood parity residual | `0.0` | `0.0` | Passed |
| Total target parity residual | `0.0` | `0.0` | Passed |
| Forecast exact replay residual | `0.0` | `0.0` | Passed |
| Maximum eager/XLA residual | `2.220446049250313e-16` | `4.440892098500626e-16` | Passed |
| Maximum CPU/GPU residual | N/A | `4.440892098500626e-16` against `1.9027307940783945e-12` | Passed |
| Terminal/forecast concrete trace count | `1` / `1` | `1` / `1` | Passed |
| Canonical numerical output placement | `CPU:0` | `GPU:0` | Passed role-specific policy |

All continuous residuals are descriptive engineering diagnostics. They do not
support a statistical ranking or scientific conclusion.

## CPU/XLA Run Manifest

| Field | Value |
| --- | --- |
| Artifact/status | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json`; `CPU_REFERENCE_CONTRACT_PASSED` |
| Artifact SHA-256 | `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2` |
| Evidence signature | `912ae925d4e7edde6980a84b75b524aab85f9194733f057513c01c6d430a6d52` |
| Fresh verification | `A2_RUNTIME_ARTIFACT_VERIFIED`; log SHA-256 `623765e1f4cdfeb2078289e9f94d208221d74a81fababb606b5084bec6fbca05` |
| Git commit / dirty state | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`; dirty worktree recorded in artifact |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TFP distribution metadata reported `not_installed` |
| Device / GPU policy | One physical/logical CPU; `CUDA_VISIBLE_DEVICES=-1`; GPU intentionally hidden |
| XLA / dtype / TF32 | `jit_compile=True`; `float64`; TF32 state recorded `true` though no GPU was visible |
| Seed | `[20260712,1202]`; materialized bank consumed |
| Wall time | `3130.586829600972` seconds |
| Trust basis | `cpu_hidden_reference_exception_not_gpu_evidence` |
| Data version | A1 observation raw SHA-256 `aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98` |

## Trusted GPU/XLA Run Manifest

| Field | Value |
| --- | --- |
| Artifact/status | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json`; `GPU_XLA_CANARY_PASSED` |
| Artifact SHA-256 | `0294b06527620336e970bf6a57fd2e0f1a8466502bf47f9595a533d10ca23521` |
| Evidence signature | `9bb522772e7cc42aced7f5a7ebbf79fc9579537b35ad219f0d370415c25bebcc` |
| Fresh verification | `A2_RUNTIME_ARTIFACT_VERIFIED`; log SHA-256 `5f18805883bd38d8fd42a4c25c034c969bfcba93b35934c1028075f4f0331b9e` |
| Git commit / dirty state | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`; dirty worktree recorded in artifact |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TFP distribution metadata reported `not_installed` |
| Device | Two physical/logical NVIDIA GeForce RTX 4080 SUPER GPUs plus CPU; all canonical outputs on `GPU:0` |
| XLA / dtype / TF32 | `jit_compile=True`; CUDA XLA; `float64`; TF32 execution enabled and recorded |
| Seed | Same persisted `[20260712,1202]` bank as CPU |
| Wall time | `1345.5081332969712` seconds |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| CPU crosslink | CPU SHA-256 `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2`; signature `912ae925d4e7edde6980a84b75b524aab85f9194733f057513c01c6d430a6d52` |

## Exact Commands

Focused suite:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/focused-tests-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -p no:cacheprovider -q tests/test_ssl_lstm_predictive_tf.py tests/test_ssl_lstm_posterior_tf.py
```

CPU generation and fresh verification:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --mode cpu-reference --bank docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.json --bank-log docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.log --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.log

CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-verification-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference-verify.log
```

Trusted GPU generation and fresh verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --mode gpu-xla-canary --bank docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.json --cpu-reference docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.log

PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-verification-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary-verify.log
```

## Repair Record

| Event | Classification | Repair and evidence |
| --- | --- | --- |
| Initial exact harness import failure | Harness/bootstrap defect | Added repository-root bootstrap only in the permitted generator/verifier; refroze boundary and reran affected artifacts |
| Initial GPU verifier bank mismatch | Verifier defect, not a GPU numerical failure | Reconstructed exact persisted hexadecimal bank rather than regenerating backend-dependent normal-transform bytes; refroze and regenerated both runtime artifacts |
| Implementation review Round 1 `REVISE` | Production admission/provenance defect | Added finite admission for free draws, three banks, and seven outputs; clarified replay authority and A1 adapter provenance; added negative tests |
| Focused rerun | Engineering regression validation | `87/87` passed, including 18 trace-parser regressions, and the closed trace stayed within reviewed write roots |
| Implementation review Round 2 | Bounded advisory review | No material findings; `VERDICT: AGREE`, weaker than Claude |
| Final runtime rerun | Conjunctive A2 runtime validation | Fresh CPU and trusted GPU generations plus fresh-process full replay verifiers passed |
| First terminal closure audit | Verifier parser defect, not an A2 runtime failure | An unanchored `link(` pattern misclassified read-only `readlink(...)`; the parser was replaced with complete syscall parsing and focused fail-closed tests |
| Hardened terminal contract review | Bounded advisory review | Accepted only the narrow one-explicit-PID `strace -f -qq -yy -s 65535 -e trace=%file` contract; every successful mutation must be an annotated write open under an allowed root; `VERDICT: AGREE`, weaker than Claude |

None of these failures invalidated the A1 target, data, filter mathematics, or
predictive-validation research direction. They were localized implementation,
harness, and verifier defects and were repaired inside the reviewed A2 write
set. Prior artifacts were treated as stale after every relevant byte change.

The accepted A2 subplan remains byte-for-byte unchanged. Under the user's
explicit "fix that and continue" direction, recorded in the approval-boundary
ledger as the narrow human-authorized trace-contract repair, every newly
generated A2 ledger, checkpoint, closure, or closure-verification trace uses
`/usr/bin/strace -f -qq -yy -s 65535 -e trace=%file`. This authority repairs
only the stale closure-regeneration chain; historical run-manifest command
strings remain historical and are not reused as authority for the hardened
terminal audit.

## Candidate Versus Research Direction

The pre-repair implementation candidate was ineligible for promotion because
it did not fail closed on nonfinite innovations/outputs and its provenance was
ambiguous. That was an engineering defect, not evidence against the terminal
Gaussian approximation, forecast recursion, machine-learning-style predictive
validation idea, NeuTra idea, or SSL-LSTM research direction.

The repaired candidate passes A2's bounded engineering screen. That does not
promote the broader research direction scientifically. Failure or invalidity
of the required independent analytic LGSSM oracle is a continuation veto in
A3; predictive equivalence and calibration remain later gates.

## Post-Run Red Team

The strongest alternative explanation is shared implementation lineage: CPU
and GPU agreement, exact replay, and target parity can preserve a common model,
filter, terminal-covariance, or recursion error. The historical SVD-UKF itself
is approximate and is not an exact nonlinear-filter oracle.

What would overturn this A2 engineering result: mutation of a bound source,
target, bank, artifact, plan, or review without restarting the affected
checkpoint; a newly reproduced nonfinite/status/covariance/recursion/placement
failure; or failure of the A3 analytic scalar-LGSSM oracle.

The weakest evidence is the small frozen design: ten parameter rows and only
two forecast draws with two replications. Those counts exercise interfaces and
replay behavior; they are not a statistical sample size and cannot establish
moments, tail behavior, calibration, equivalence, or model adequacy.

## Exact A3 Handoff

A3 subplan drafting becomes eligible only after:

1. `executor-write-ledger.json` binds the closed boundary, focused, CPU, and
   GPU traces plus this draft result;
2. fresh CPU and GPU verifiers pass immediately before the pre-result final
   checkpoint;
3. `final-checkpoint.json` passes and binds the implementation review and this
   result;
4. this exact A2 result receives a hash-bound bounded
   `CODEX_SUBSTITUTE_REVIEW` `VERDICT: AGREE`.

Those four conditions authorize drafting the A3 subplan from the exact A2
interfaces and actual artifact signatures. They do not authorize A3
implementation.

A3 must be forecast-oracle and predictive-statistics engineering only. Its
minimum intended scope is:

- an independent analytic scalar LGSSM 1-to-10-step forecast oracle;
- mean, variance/log-variance, central moments, quantiles, and cross-horizon
  covariance surfaces;
- separately labeled fixed-bandwidth MMD U- and biased V-statistics;
- chain-aware batch/block/bootstrap infrastructure;
- simultaneous equivalence interval machinery; and
- explicit fail-closed statuses and an oracle-failure continuation veto.

A3 must not run sampler comparisons or claim predictive equivalence. A3
implementation remains forbidden until its exact subplan receives
`VERDICT: AGREE`, the A2 post-result closure verifies, and the terminal
read-only trace audit passes and is recorded.

## Nonclaims

- Not posterior correctness, parameter-posterior equality, HMC validity,
  convergence/readiness, or NeuTra evidence.
- Not exact nonlinear filtering, predictive equivalence, calibration, model
  adequacy, model selection, or statistical ranking.
- Not performance superiority, public API, default, product, release, or
  scientific readiness.
- Not full-parameter or expanded-mask support.
- Not a Zhao-Cui source-faithfulness result.
