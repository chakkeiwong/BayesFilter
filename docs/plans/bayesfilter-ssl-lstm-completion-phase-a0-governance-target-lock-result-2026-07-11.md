# Phase A0 Result: Governance, Target, And Artifact Lock

Date: 2026-07-11

Status: `PASSED_FOR_A1_IMPLEMENTATION_ONLY`

## Outcome

Phase A0 has produced one strict, replayable historical four-parameter scalar
SSL-LSTM target lock. Successful attempt 02 passed generation, immediate
fresh-process verification, a guarded mutable-governance provenance refresh,
and two unchanged final fresh-process verifiers. The final verifiers reproduced
the original immutable and signature aggregates exactly. A1 implementation
remains barred until a fresh hash-bound review accepts this exact final result
and the complete A1 entry preflight passes.

The lock is classified `extension_or_invention`. It preserves the historical
SVD-UKF approximate filtering target; it does not certify that target as an
exact nonlinear likelihood or scientifically adequate model.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept attempt 02 for A1 implementation entry only | Passed strict schema, tensor, signature, dependency, immutable, geometry, replay, mutable-provenance refresh, and final rehash gates | No active A0 runtime veto; exact final-result review remains a conjunctive external entry gate | Production extraction and GPU/XLA parity are untested | Obtain the focused hash-bound final-result review, then run the strict A1 entry preflight | Posterior correctness, HMC/NeuTra readiness, predictive equivalence, model adequacy, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for attempt 02 after the guarded mutable-provenance refresh and two unchanged final verifiers |
| Statistically supported ranking | None; no stochastic candidate comparison occurred |
| Descriptive-only differences | Phase 2S raw-to-regularized geometry residuals and framework startup messages only |
| Default readiness | Not assessed and not implied |
| Next evidence needed | Hash-bound final-result review, then A1 typed production extraction with CPU-XLA and trusted GPU/XLA gates |

## Separate Evidence Ledgers

| Ledger | Status | Evidence and boundary |
| --- | --- | --- |
| Engineering correctness | `passed_a0_historical_replay_only` | Strict lock validation and deterministic replay passed; production module does not yet exist |
| Numerical/sampler validity | `not_assessed` | Source-aware geometry metadata is internally reconstructed, but no sampler ran |
| Computational predictive equivalence | `not_assessed` | No forecast API, posterior draws, moments, MMD, or equivalence test ran |
| Synthetic generative calibration | `not_assessed` | No replicated data set or held-out calibration ran |
| Empirical model adequacy | `not_assessed` | The data are a locked synthetic fixture, not an application study |

## Locked Target Identity

| Component | Locked value or identity |
| --- | --- |
| Static dimensions | `T=30`, latent `1`, hidden `1`, observation `1`, augmented state `3`, full parameter dimension `24` |
| Free coordinates | indices `(12,13,14,15)`; `latent_mean_weight.0.0`, `latent_mean_bias.0`, `observation_weight.0.0`, `observation_bias.0` |
| Free truth/prior center | `(0.35,-0.08,0.65,0.05)` |
| Fixed coordinates | Remaining `20` entries of the locked full fixture |
| Dtype | `float64` |
| Likelihood | Historical `tf_ssl_lstm_svd_ukf_score` SVD-UKF filtering log likelihood |
| Prior | Unnormalized `-0.5 * sum((free - truth_free)^2 / 4.0^2)`; no parameter-independent constant |
| Full-fixture raw SHA-256 | `33b0814b86c5875e6746150762b8ae3b655e5bbcaa0bfd8df51488783bcb601f` |
| Observation raw SHA-256 | `aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98` |
| Truth/prior-center raw SHA-256 | `e46fb6877d89473071047938f170cef5c3d02b2c87ce7f9834d92c4040e16c2f` |

Exact likelihood settings are `std_floor=1e-4`, `alpha=1`, `beta=2`,
`kappa=0`, `placement_floor=0`, `innovation_floor=1e-12`,
`rank_tolerance=1e-12`, `spectral_gap_tolerance=1e-10`,
`fixed_null_tolerance=1e-10`, `jitter=0`, and
`allow_fixed_null_support=False`.

## Probe Replay

| Probe | Total value | Total score | Status |
| --- | ---: | --- | --- |
| `truth_free` | `-37.847429129540124` | `(0.04136499462054382,1.5569199957141024,0.25645839913165336,2.385670540067278)` | Passed decomposition, historical-anchor tolerance, and exact fresh-lock replay |
| `phase2s_center` | `-37.77528495512358` | `(-1.0897220625860626e-12,-1.5913695708413833e-11,-9.711197818118578e-11,-2.5740403732099626e-11)` | Passed decomposition, historical-anchor tolerance, and exact fresh-lock replay |

Historical JSON decimal anchors differed from fresh binary64 values by
`7.105427357601002e-15` at both scalar values. The verifier uses the reviewed
`8*eps64` scale-aware formula only for those historical comparisons. Fresh
attempt replay and all integrity checks remain exact.

## Signatures And Fingerprints

| Signature | SHA-256 |
| --- | --- |
| Target semantic | `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Implementation/execution | `650eeeb05ae94497b988373efee9759447211f8eff39d039e0c1eebf1ccb0a53` |
| Sampler geometry | `f9f9cbf0847bded113eb9eebcd6abecdf439a78d41f1e88035e79bc1a69c872a` |
| Forecast design | `9916a90e8eb0f87f3b41790a111cde12221314a77bf56cdedb97345d82dc1b07` |
| Signature aggregate | `af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc` |
| Dependency-manifest semantic aggregate | `9718ba393521486d1b63ae19c31c59e2ec636889002d2363c696523ac8ca5f9b` |
| Dependency-manifest exact file | `2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517` |
| Immutable attempt aggregate | `6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f` |
| Target-lock exact file after mutable-provenance refresh | `1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383` |
| Harness exact file | `e8bb6e8dbc861f9c63982e8ea4f67d2cfa4c6cf413ab9e5d5ec5763858af6954` |

The successful dependency closure contains `43` sorted local Python modules.
The immutable fingerprint binds them, critical roots, historical inputs, the
A0 harness, and the exact dependency-manifest bytes.

## Sampler Geometry Disposition

The Phase 2S center, scale, factor, covariance, and precision matrices are
historical sampler initialization/tuning context only. Source-aware checks
passed, including exact reconstruction of the regularized theta precision and
covariance. The raw-to-stored residuals
`7.503534252273347e-10` (covariance) and
`1.0000462680181954e-09` (precision) are explanatory because the source adds
`1e-9 I` and applies its eigenvalue/condition rule before inversion. Phase 2S is
not a certified global MAP or posterior covariance.

## Historical Artifact Disposition

| Artifact | SHA-256 | Disposition |
| --- | --- | --- |
| `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py` | `fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28` | Target-construction context, not production code |
| `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py` | `4dbc1a0d6a002476bedb24b2e023a37cb435362a39a467d67f8788f784dcd46a` | Untracked sampler-geometry context only |
| `docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json` | `a8ea5e1e2f41078a99d120939a9f5b9ac2dbfedea813a4e6fa70a05725c785b6` | Historical target context replayed, not promoted |
| `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json` | `8048fb04d440ca65c178aacfd7cdae20a93306e5a129edc7d35420f1e50dbbc5` | Untracked sampler-geometry context only |
| `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json` | `5394b6da96e57edb62bcce5d77048a70d6d130a12a7fe5db1359c77f0fc8fff4` | Diagnostic context forbidden as confirmatory baseline |
| Phase 2W/2X/2Y/2Z JSON artifacts | Bound individually in the lock | Failed/exploratory context, not posterior evidence |

No historical row is promoting. The failed independent-reference branch stays
blocked and Phase 2V stays one-chain CPU-hidden diagnostic context.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Git state | Dirty; `309` porcelain rows at final close-record drafting; unrelated user work preserved |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, Python `3.13.13` |
| Packages | TensorFlow `2.20.0`, TensorFlow Probability distribution `0.25.0`, NumPy `2.1.3` |
| Environment | `CUDA_VISIBLE_DEVICES=-1`, deterministic ops on, oneDNN off, intra/inter-op and OMP threads `1`, hash seed `0` |
| Device/JIT/XLA | Deliberate CPU-hidden, `jit_compile=False`, non-XLA historical reference exception |
| Data version | Exact observation tensor in the target lock; stateless generator seed `(20260708,2301)` |
| Generation start/end | `2026-07-11T12:46:05.255961+00:00` / `2026-07-11T12:46:28.109440+00:00` |
| Generation wall time | `22.85354214301333` seconds |
| Plan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md` |
| Result | This file |
| Structured outputs | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json`, manifest and logs beside it |

Exact generation and verification commands are the commands declared in the A0
subplan and embedded in the lock. Attempt-02 generation and immediate
verification exited `0`. After the guarded provenance-only refresh, the
unchanged verifier exited `0` at approximately `2026-07-11T23:34:43+08:00` and
again after downstream A1 hash rebinding at approximately
`2026-07-11T23:40:01+08:00`; both returned immutable aggregate
`6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f`
and signature aggregate
`af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc`.
TensorFlow emitted `CUDA_ERROR_NO_DEVICE` during framework startup despite
`CUDA_VISIBLE_DEVICES=-1`; this is a CPU-hidden startup anomaly only, not GPU
evidence or a machine/driver diagnosis.

## Failed Attempt And Repairs

| Event | Classification | Repair/decision |
| --- | --- | --- |
| Interpreter symlink mismatch | Harness provenance defect | Compare the invoked path without resolving its symlink; rereviewed |
| Raw Phase 2S cross-coordinate identities failed | Plan/schema/harness math defect | Reproduced the cited precision regularization and made raw residuals explanatory |
| Attempt 01 exact historical-value equality failed | Verifier-contract decimal-round-trip defect | Preserved the whole attempt; scoped `8*eps64` tolerance to historical anchors; rereviewed |
| Attempt 02 | Successful A0 candidate | Generation and immediate strict verifier passed |
| Final verifier found stale mutable A0-subplan provenance | Provenance-lifecycle defect, not target or immutable drift | Recorded the failed gate; refreshed only the stale descriptor from `330b4f1bfd0820700e2dcc91d982a63e5f086281bc4d4ee140879ab48ccdb53b` to `cabf1439c4702515e8591c1865db4cca5a1f143f3111cfab510ece26faebc947`; proved every other JSON field unchanged; rereviewed and reran the unchanged verifier twice |

Failed attempt 01 is immutable at
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/failed-attempt-01/`
with aggregate
`6dc7b4942a96cae4a10e77313b54eef60406bf8826ae872763f2c0d9a02e2f2c`.
It is not reused as successful evidence.

## Review Record

- Claude review remained policy-unavailable after informed user approval. No
  Claude process or liveness probe ran and no repository content was sent.
- Roadmap `CODEX_SUBSTITUTE_REVIEW` converged in Round 4; runbook review agreed
  in Round 1.
- A0 subplan Rounds 1-5 exposed material contract defects. The fifth-round
  lifecycle finding triggered a stop. The user authorized one focused recovery,
  which returned `AGREE`.
- Harness, geometry repair, and historical-anchor tolerance repairs each passed
  fresh bounded `CODEX_SUBSTITUTE_REVIEW` after visible repairs.
- The provisional A0 result review agreed that only the final verifier could
  proceed. The first final verifier then failed closed on the stale mutable A0-
  subplan descriptor. A fresh provenance-lifecycle review accepted a guarded
  one-leaf lock refresh; immutable and signature fields remained byte-identical.
- The terminal A1 plan review converged in Round 5. Fresh provenance-rebind
  reviews accepted exact A1 plan SHA-256
  `7de07b80aa059ec43cc688e8729acc0af5ef6dc9ba702bbe778483134e7e72cd`
  and golden SHA-256
  `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34`.
- Substitute agreement is weaker than Claude review and is not called Claude
  convergence.

Current exact final-document review status: `PENDING_EXTERNAL_HASH_BOUND_RECORD`.
Final verifier status: `PASSED_AFTER_GUARDED_MUTABLE_PROVENANCE_REFRESH`.

## Candidate Versus Research Direction

Attempt 01 was rejected as an evidence artifact because its verifier contract
was too strict for decimal-serialized binary64 anchors. That rejected attempt
did not invalidate the harness architecture after repair, target, data, model,
math, predictive-validation proposal, NeuTra candidate, or SSL-LSTM direction.
Attempt 02 passed the A0 runtime and immutable-evidence checks. The stale
governance descriptor was rejected and repaired without changing the target or
immutable attempt. This does not reject the SSL-LSTM research direction and
does not promote any later sampler or predictive claim.

## Post-Run Red Team

The strongest alternative explanation is that deterministic agreement merely
reproduces the same implementation lineage and could preserve a shared model or
derivative error. A1 therefore requires production-owned extraction, mask and
prior tests, central finite differences, CPU-XLA parity, and trusted GPU/XLA
parity. Even those tests would establish engineering preservation, not exact
nonlinear likelihood, identification, posterior correctness, or application
adequacy. Any immutable-member change before final handoff overturns this
provisional result and requires a fresh attempt.

## Exact A1 Handoff

A1 may begin only after all of the following are true:

1. this exact final result receives a bounded material read-only `AGREE`
   verdict that binds its current SHA-256;
2. `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md`
   independently receives `AGREE`;
3. the identical A0 `--verify` command has exit `0` immediately before handoff
   and returns immutable aggregate
   `6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f`
   and signature aggregate
   `af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc`;
4. the non-TensorFlow A1 document preflight independently confirms all current
   result, plan, golden, review, lock, manifest, harness, comparator, and `HEAD`
   bindings before any A1 source edit.

## Nonclaims

- Not a production implementation.
- Not posterior correctness, HMC validity/readiness/convergence, or NeuTra evidence.
- Not predictive equivalence, calibration, empirical adequacy, or a sampler ranking.
- Not GPU/XLA, public API, default-policy, release, or product readiness.
- Not a Zhao-Cui source-faithfulness result.
