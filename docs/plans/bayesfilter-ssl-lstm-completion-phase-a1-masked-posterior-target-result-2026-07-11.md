# Phase A1 Result: Reusable Masked Posterior Target

Date: 2026-07-12

Status: `PASSED_FOR_A2_PLANNING_ONLY`

## Outcome

Phase A1 produced a production-owned, typed, four-coordinate TensorFlow masked
posterior target that preserves the A0-locked historical SVD-UKF estimand and
unnormalized prior convention. The target passed source, mask, prior,
historical-route, derivative, scalar/batch, eager/CPU-XLA, and finite-input
checks at all ten frozen points. All three nonfinite reject cases passed
separately. The trusted GPU/XLA target canary passed at the ten finite points.

The final integration blocker was a missing local TensorFlow binding in
`bayesfilter/inference/hmc.py::static_unroll_chain_value_and_score`. Adding one
function-local `import tensorflow as tf` repaired the exact failure without
changing A1 source, tests, harness, frozen boundaries, or structured evidence.
The previously failing test then passed, the complete reviewed suite passed
`75/75`, and the CPU and GPU artifacts both passed unconditional full
fresh-process recomputation with unchanged evidence signatures.

This result authorizes only A2 subplan drafting and review. It does not
authorize HMC, NeuTra, forecasting runtime, calibration, model selection, or a
scientific/product/default claim.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept A1 engineering target extraction for A2 planning | Passed all reviewed source, ten-point CPU, three-case reject, 75-test, and ten-point trusted GPU/XLA gates | No active A1 veto; the integration defect was repaired and the complete checkpoint reran | Ten frozen points and shared implementation lineage do not establish target-wide, posterior, sampler, or predictive validity | Obtain a fresh hash-bound A1 result review, then draft and review the A2 terminal-state/forecast API subplan | Posterior correctness, HMC/NeuTra readiness, predictive equivalence, calibration, model adequacy, performance, public/default/product/release readiness, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the bounded A1 engineering target and ten-point GPU/XLA canary after the integration repair |
| Statistically supported ranking | Not applicable; no stochastic method comparison or ranking ran |
| Descriptive-only differences | Residual magnitudes, compile messages, devices, and wall times are descriptive within the engineering canary |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | A2 terminal-state/forecast API and later independent oracle, sampler, predictive-equivalence, and calibration phases |

## Separate Evidence Ledgers

| Ledger | Status | Evidence and boundary |
| --- | --- | --- |
| Engineering correctness | `passed_a1_bounded_target_extraction` | `75/75` tests; exact historical replay, finite differences, CPU-XLA, reject behavior, and ten-point trusted GPU/XLA parity passed |
| Numerical/sampler validity | `not_assessed` | No HMC chain, convergence, posterior-reference, or sampler-validity run occurred |
| Computational predictive equivalence | `not_assessed` | A1 did not implement or run forecasting or moment comparison |
| Synthetic generative calibration | `not_assessed` | No replicated data or coverage/PIT experiment ran |
| Empirical model adequacy | `not_assessed` | No application-data or held-out adequacy experiment ran |

## Research Intent Ledger

| Field | A1 disposition |
| --- | --- |
| Main question | Can a production-owned four-coordinate TensorFlow target preserve the exact A0 historical estimand and expose a graph-native target-only GPU/XLA value/score surface? |
| Candidate/mechanism | Typed parameter mask and posterior target preserving the historical SVD-UKF route and unnormalized Gaussian prior |
| Expected failure mode | Target/signature drift, derivative mismatch, invalid finite reject, static-shape failure, XLA failure, or CPU/GPU parity failure |
| Promotion criterion | All reviewed source/tests plus exact ten-point CPU and trusted GPU/XLA gates and complete final checkpoint |
| Promotion veto | Any source, mask, historical, derivative, reject, CPU/GPU, artifact, or mandatory-test failure |
| Continuation veto | Target migration, protected/A1-owned committed drift, forbidden backend bridge, corrupt evidence, or required out-of-scope target repair |
| Repair trigger | The observed missing TensorFlow binding triggered the smallest one-line integration repair and complete checkpoint rerun |
| Explanatory diagnostics | Residual magnitudes, compilation messages, device inventory, and wall time |
| Must not be concluded | Posterior correctness, sampler validity, predictive equivalence, calibration, model adequacy, superiority, or default readiness |

## Skeptical Audit At Closeout

| Risk | Result |
| --- | --- |
| Wrong baseline | Avoided: the comparator remained the A0-locked historical SVD-UKF target |
| Proxy promotion | Avoided: unit and parity checks promote only bounded A1 engineering status |
| Missing stop condition | The initial integration failure stopped handoff until the exact test and complete suite passed |
| Unfair comparison | CPU and GPU used identical points, target, dtype, signatures, and compiled program |
| Hidden assumption | The approximate filter, four-coordinate mask, fixed remaining parameters, and unnormalized prior remain explicit |
| Stale context | `HEAD`, full live history, protected rows, entry/boundary hashes, source hashes, and artifacts were checked after repair |
| Environment mismatch | CPU was deliberately GPU-hidden; GPU used owner-designated trusted managed-session provenance |
| Artifact insufficiency | Both structured artifacts were strict-loaded and fully recomputed after the repair |

Audit status: `PASSED_FOR_A1_ENGINEERING_CLOSEOUT_AND_A2_PLANNING_ONLY`.

## Locked Target And Interfaces

| Component | A1 contract |
| --- | --- |
| Free dimension | `4` |
| Free indices | `(12,13,14,15)` in the locked 24-coordinate scalar chart |
| Free names | `latent_mean_weight.0.0`, `latent_mean_bias.0`, `observation_weight.0.0`, `observation_bias.0` |
| Fixed coordinates | Remaining `20` coordinates from the A0 fixture |
| Dtype | `float64` |
| Filter target | Historical SVD-UKF filtering log likelihood plus locked prior |
| Prior | Unnormalized `-0.5 * sum((free - truth_free)^2 / 4.0^2)` |
| Target semantic SHA-256 | `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Parameter-mask SHA-256 | `9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043` |
| Masked-posterior contract SHA-256 | `004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556` |
| Default execution | TensorFlow, `float64`, `jit_compile=True`, XLA enabled |

The production target exposes typed mask construction, free-to-full embedding,
full-to-free extraction, scalar value/score, and statically sized batch
value/score. Nonfinite free inputs use a graph-native deterministic reject
branch; finite filter failures remain loud. The module does not self-certify
HMC, predictive, calibration, or product authority.

## Frozen Boundary And History

| Field | Value |
| --- | --- |
| A0 anchor and final observed `HEAD` | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Live commits after anchor | None |
| Protected dependency rows | `23/23` matched the frozen entry hashes after repair |
| Excluded dependency rows | `28`; the repaired generic HMC module remains outside A1 target dependency equality |
| Entry artifact SHA-256 | `fc3c953155e32fbb76aaa10c7e1404ed9009b94b156dc941c91d57d1a4597700` |
| Scoped boundary SHA-256 | `a14e15dbbe8db5986400a1f986dbc67acc9e9eb9623b057565c7da90fbf4a8f3` |
| Scoped boundary aggregate | `c7c0384c31e0bd030005d887cf9c6dfb2e79402779691bfbfa0448343c8b65ee` |
| Protected aggregate | `3cfea11da5f415d8b0ae51b4a444763641ab4fdac7559705b0ac110486f78cc2` |
| Excluded aggregate | `ff70e1fc141823fdc1c133d08ab694ddf8cdaac975a6ded87655a230c845c23c` |

The one-time entry writer was not rerun. Later processes verified the frozen
entry and boundary read-only and independently enumerated anchor-to-current
history.

## Source And Review Bindings

| Role | Path | SHA-256 |
| --- | --- | --- |
| Production target | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py` | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| Lazy exports | `bayesfilter/nonlinear/__init__.py` | `9bfbe2a912b6465e8338d61c48c51b91b2b30d1f11912a543772e5901998de68` |
| Focused A1 tests | `tests/test_ssl_lstm_posterior_tf.py` | `9635074e50f47b321e946707770503480e43b2bad78d2963d155127569ac25ca` |
| Evidence harness | `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py` | `94d232114395438b743f4cc06ff7a5b806df28c82016d1af9d9bea4da7061440` |
| Golden signatures | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json` | `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34` |
| Historical comparator | `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py` | `fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28` |
| Accepted subplan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md` | `43a671b3ed9d651ea2d3c4622c5667da0128e91cd4a71d6d7c2ef25dc840cb72` |
| Accepted subplan review | `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md` | `ad79215c5f85d241999172453b371bc6e40da9bea705028e24b9278f69da8546` |
| Accepted golden review | `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md` | `e175d526689329de77fb4424d5e0db07ef67d64d337ee0d0c8e8241420f3edcc` |
| Implementation/CPU review | `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-implementation-codex-substitute-review-2026-07-11.md` | `eaf34b7f11855bd4b60ba73274d88898086a997dd4be3f8ebacfae27ea23aa27` |
| Repaired integration module | `bayesfilter/inference/hmc.py` | `d4e74475e2f5fe43f952746bc0641089314a5e4980ef1ca068740b91b89597dc` |

Claude remained policy-unavailable; no Claude process ran and no repository
content was sent. All review records are bounded `CODEX_SUBSTITUTE_REVIEW`,
explicitly weaker than Claude review.

## CPU Evidence

| Field | Value |
| --- | --- |
| Artifact | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json` |
| Exact file SHA-256 | `b6dc26637d584dbf6d62575a999af5cf43bb7bab35a5cf9eb6984d1cfaf6a068` |
| Log SHA-256 | `db53c9675604100562ccf2673e56577a42f59be5b580b3eb929019f0503863e2` |
| Evidence signature | `c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d` |
| Status | Generation `phase_a1_cpu_reference_passed`; post-repair exact `--verify` returned `phase_a1_artifact_verified` |
| Finite/reject rows | `10/10` finite and `3/3` reject rows passed |
| Historical value/score maximum residual | `0.0` / `0.0` |
| Eager/CPU-XLA value/score maximum residual | `1.9697381503647193e-09` / `5.6968261219481064e-09` |
| Finite-difference maximum absolute residual | `1.1806352007148746e-09` |
| Generation wall time | `748.8583243020112` seconds |

CPU run manifest: commit
`a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`; interpreter
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; conda environment `tfgpu`;
Python `3.13.13`; TensorFlow `2.20.0`; TensorFlow Probability distribution
`0.25.0`; NumPy `2.1.3`; `float64`; `jit_compile=True`; XLA enabled;
`CUDA_VISIBLE_DEVICES=-1`; deterministic ops on; oneDNN off; one
intra/inter-op and OMP thread; no model randomness; data version observation
SHA-256 `aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98`;
and trust basis `cpu_hidden_reference_exception_not_gpu_evidence`. The plan,
result, output, and log paths are the exact paths in the artifact manifest.
The CPU-hidden `cuInit` message is a framework anomaly only.

## Trusted GPU/XLA Evidence

| Field | Value |
| --- | --- |
| Artifact | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json` |
| Exact file SHA-256 | `1538032c6e0c9ea664ea92ce9ea334c92c916c13831fd08af69865435c822f6e` |
| Log SHA-256 | `0d7a30e868666196afc88d76d8928a04f6354eea04696575b4a005c2d0de5fc9` |
| Evidence signature | `077abbd5d5d8dc1068d99aba90fc8b6dd5b74001cda1dd1fe4428d13a0b4631c` |
| Status | Generation `phase_a1_gpu_xla_canary_passed`; post-repair exact `--verify` returned `phase_a1_artifact_verified` |
| Frozen finite points | `10/10` passed; GPU artifact contains no reject rows |
| Maximum CPU/GPU value residual | `0.0` |
| Maximum CPU/GPU score infinity residual | `3.552713678800501e-15` |
| Generation wall time | `97.10013170191087` seconds |
| Devices | Two physical and logical NVIDIA GeForce RTX 4080 SUPER GPUs plus CPU |
| Runtime policy | `float64`, `jit_compile=True`, XLA enabled, TF32 enabled and recorded |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

GPU run manifest: same commit, interpreter, package versions, deterministic
target, and data version as CPU; `CUDA_VISIBLE_DEVICES` unset; compiled output
placement verified on GPU; plan/result/output/log paths recorded in the exact
artifact. The only supported GPU status is
`GPU_XLA_CANARY_PASSED_AT_10_FROZEN_POINTS`.

## Local Checks

| Check | Result |
| --- | --- |
| A1 and repaired HMC `py_compile` | Passed |
| Forbidden benchmark/`tf.py_function`/principal-square-root production scan | No matches; expected `rg` exit `1` |
| Forbidden NumPy/`jit_compile=False` production scan | No matches; expected `rg` exit `1` |
| Non-vacuous tracked/untracked whitespace audit | Passed |
| Pre-repair complete suite | `1 failed, 74 passed`; preserved as the repair trigger |
| Focused repaired integration test | `1 passed` in `10.04s` |
| Final complete four-file suite | `75 passed` in `812.42s` |
| Post-repair protected dependency rehash | `23/23` matched |
| Post-repair CPU artifact exact recomputation | Passed with unchanged evidence signature |
| Post-repair GPU artifact exact recomputation | Passed with unchanged evidence signature |

## Repair Record

| Event | Classification | Repair and evidence |
| --- | --- | --- |
| Frozen-entry/live-history contradiction | Governance lifecycle defect | Owner authorized up to five exceptional rounds; recovery E1 made the entry immutable and live history independent; review agreed |
| Initial mandatory suite failure | Excluded integration defect, not A1 target failure | `hmc.py` used `tf.TensorArray` without a local binding; one function-local import was added |
| Focused repair check | Engineering repair validation | Previously failing XLA chain-order test passed |
| Final checkpoint | Conjunctive A1 closeout | `75/75`, CPU verify, GPU verify, hashes, protected rows, and live history all passed |

Four exceptional lifecycle-review rounds remain unused. The code repair did
not change a target, tolerance, frozen point, schema, baseline, or evidence
role. It did not run HMC.

## Candidate Versus Research Direction

The pre-repair integration candidate was rejected for closeout because one
mandatory test raised `NameError`. That failure was an implementation binding
defect, not evidence against the target, data, filter mathematics, artifacts,
predictive-validation design, NeuTra idea, or SSL-LSTM direction. The repaired
candidate passes A1's bounded engineering criteria. This still does not promote
the broader research direction scientifically; later oracle, sampler,
predictive, and calibration gates remain necessary.

## Post-Run Red Team

The strongest alternative explanation is shared implementation lineage: exact
historical replay and CPU/GPU parity could preserve a common model or
derivative error. Central finite differences reduce derivative risk but do not
establish the exact nonlinear likelihood, identification, posterior
correctness, or predictive adequacy.

What would overturn this A1 engineering result: any mutation of the frozen
target, mask, source, test, harness, boundary, or evidence artifacts without
restarting the affected checkpoint; a newly reproduced value/score mismatch;
or failure of the target on broader points under a later reviewed gate.

The weakest evidence is the ten-point design. It is sufficient for the
predeclared extraction canary, not target-wide correctness or HMC readiness.

## Exact A2 Handoff

A2 implementation remains forbidden until both conditions pass:

1. This exact A1 result receives a fresh hash-bound bounded material
   `CODEX_SUBSTITUTE_REVIEW` verdict of `AGREE`.
2. A dedicated A2 subplan is drafted from the actual A1 signatures and source
   contracts, states terminal-state/filter parity and forecast boundaries,
   limits its write set, preserves GPU/XLA defaults, identifies the later A3
   oracle prerequisites, and independently receives `AGREE`.

A2 may then implement typed terminal filtered-state extraction and stateless
multi-horizon path simulation. It may not run HMC, NeuTra, predictive
equivalence, calibration, model ranking, or application claims. Analytic LGSSM
oracle and statistical equivalence work belong to A3 unless the reviewed A2
subplan names only a non-promoting prerequisite fixture.

## Nonclaims

- Not posterior correctness, HMC validity/readiness/convergence, or NeuTra evidence.
- Not predictive equivalence, calibration, empirical adequacy, or model selection.
- Not a performance benchmark, sampler ranking, public API, default, product,
  release, or scientific result.
- Not full-parameter or expanded-mask support.
- Not a Zhao-Cui source-faithfulness result.
