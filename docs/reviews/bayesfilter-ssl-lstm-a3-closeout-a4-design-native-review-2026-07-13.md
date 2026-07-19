# Focused Native Review: SSL-LSTM A3 Closeout And A4 Design

Date: 2026-07-13 (Asia/Shanghai)

Review type: `FOCUSED_NATIVE_READ_ONLY_REVIEW`

Status: `AGREE_AFTER_REPAIR`

## Scope

Reviewed together because they form one A3-to-A4 evidence boundary:

- `docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py`;
- `docs/benchmarks/verify_ssl_lstm_predictive_validation_a3_2026_07_13.py`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md`; and
- `docs/plans/bayesfilter-ssl-lstm-predictive-validation-live-plan-2026-07-13.md`.

The review checked factual agreement with the canonical A3 artifacts and
receipts; adapter fail-closed behavior; baseline and evidence roles; statistical
error definitions; practical-margin boundaries; weighting and MGF claims;
validation/confirmation separation; feasibility; artifact coverage; and the
prohibition on A4, HMC, or NeuTra execution under A3 authority.

This was a native Codex review under the Tier 2 academic workflow. Claude was
not used or needed, and this record is not represented as external review or
scientific peer review.

## Findings And Repairs

| Severity | Finding | Repair | Focused recheck |
| --- | --- | --- | --- |
| Material factual | The first A3 result draft copied an incorrect numeric-configuration SHA-256 into the run-manifest table. | Replaced it with the CPU/GPU artifact value `5eb087faefaf40c1393dd844b0d037ac02c0ebee161fa3d949ce432ef7a016d2`. | Parsed both canonical artifacts and confirmed identical configuration bindings. |
| Material statistical | The first A4 draft conflated null false-material-difference error with false equivalence under a material alternative, and did not require enough true-equivalence `PASS` probability. An always-inconclusive design could therefore appear safe. | Separated simultaneous interval coverage, true-equivalence decision power, null false-material-difference rate, false-equivalence `PASS` rate, and material-difference power. Added family-specific exact-binomial bounds and vetoes. | Audited the four-state decision logic against `classify_predictive_evidence`: `PASS`, `MATERIAL_DIFFERENCE`, `INCONCLUSIVE_UNDERPOWERED`, and `INVALID_HARD_VETO` now have distinct calibration roles. |
| Moderate mathematical | “MGF-inspired weight” was underspecified and could imply that an MGF uniquely supplies an optimal weight. | Defined a bounded symmetric multivariate log-MGF feature grid `log E[exp(t^T z)]`; separated the feature map from a regularized long-run-covariance/GMM weight; retained characteristic/kernel MMD as the robust primary joint-law route. | Confirmed the plan states the MGF existence/tail condition, influence/stability vetoes, no clipping/winsorizing, and the Gaussian-kernel/characteristic-function relationship without claiming a unique optimal weight. |
| Moderate validity | “Valid replication count” could invite silent replacement of invalid stochastic replications. | Made any invalid replicate a repair trigger and prohibited silent discard or replacement. | Confirmed invalid evidence remains a promotion veto and cannot be converted to a favorable denominator. |
| Moderate boundary | A4 requires split-half ordinary-HMC calibration draws but this design draft does not authorize generating them. | Required the execution refresh to bind an existing sampler-valid calibration artifact or seek separate authorization for a calibration-only acquisition run; prohibited borrowing A5 confirmation draws. | Confirmed the design remains non-executable and cannot smuggle an HMC run across the A3-to-A4 handoff. |
| Moderate provenance | A final source-binding check initially compared the CPU artifact's adapter v1 hash with the post-CPU GPU-loader-repair v2 file and reported drift. | Reconstructed v1 byte-for-byte by removing exactly the two v2 hunks preserved in the execution record; documented that CPU binds v1 and GPU binds v2. | Reconstructed v1 compiled and matched `047449702edad16a0db7316ac7daf2d8a1b8b587bd6fc7ea4a8d0f85c952ab28`; current v2 matched the GPU-bound `9d614b69b1535278994eb1027a3048824faa137e6fbfc60768cb6ce2ec17a36a`. |

## Adapter Review

No remaining material adapter finding was identified.

- The runner changes governance bindings and manifest/schema adaptation without
  changing production mathematical code.
- Controlled-alternative construction and inferential validity remain hard
  checks; one-fixture detection is correctly explanatory rather than a hidden
  promotion veto.
- The verifier independently reconstructs formulas, continuous tensors within
  a scale-aware floating tolerance, exact categorical/integer schedules,
  decisions, compiler structure, source/configuration bindings, and CPU/GPU
  parity.
- GPU execution consumes the independently verified persisted CPU values and
  all four materialized resampling rows. It does not regenerate replay inputs
  from seed metadata.
- Raw HLO integrity remains exact inside each artifact; only TensorFlow
  process-local function IDs are normalized for fresh-process structural
  comparison.
- Generation and verification provenance are separated, and the final receipts
  bind the current verifier sources.
- The CPU/GPU adapter version split is explained by the bounded GPU-loader
  repair, and the exact compiling v1 source is reproducible from v2 plus the
  recorded two-hunk inverse diff at
  `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/tier2-generation-adapter-v1-to-v2.patch`.

Residual risk: the Tier 2 adapters intentionally reuse the historical A3
generation and replay cores. Shared conceptual errors remain possible despite
direct equation replay and independent verification. This is correctly stated
in the A3 post-run red team and is not hidden by CPU/GPU parity.

## Result And A4 Boundary Review

No remaining material result or plan finding was identified.

- The result distinguishes engineering passage from the actual
  `INCONCLUSIVE_UNDERPOWERED` identical-law decision.
- It treats missed variance, skew, and dependence alternatives as A4 repair
  triggers, not evidence against the research direction.
- It includes the required decision, inference-status, evidence-ledger,
  manifest, repair, uncertainty, red-team, and nonclaim records.
- A4 keeps horizon-specific mean and log-variance intervals co-primary and
  prevents aggregate weights from concealing a material horizon.
- Higher moments and covariance remain explanatory; independent-bank MMD is the
  omnibus inferential branch.
- Equal, diagonal inverse-variance, shrinkage/GMM, bounded MGF-inspired, and
  characteristic/kernel candidates have explicit roles and safeguards.
- Calibration nomination, fresh validation, confirmation, and audit seeds are
  separated.
- A4 has explicit continuation and resource stop logic but remains design-only
  until exact commands, budget, replication count, numerical ladders, and
  scientific material-effect labels are frozen.

## Verdict

`VERDICT: AGREE`

The reviewed A3 evidence supports `PASSED_FOR_A4_DESIGN_ONLY`. The A4 document
is a coherent design draft, not execution authority. No claim of SSL-LSTM
predictive equivalence, calibration, posterior correctness, HMC/NeuTra
readiness, superiority, model adequacy, or production/default readiness is
supported.
