# Corrected Neural-Force HMC Terminal Result

Second supersession note, 2026-07-18: the later native-tuning audit invalidates
all tuned runtime, seconds/ESS, speed, break-even, and nonlinear performance
claims inherited from the exact-gradient comparison repair.  They are
`UNSUPPORTED_PENDING_NATIVE_RETUNING`; see
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

Supersession note, 2026-07-18: the later exact-gradient comparison repair
supersedes this file's nonlinear performance-gap classification. PP-UKF,
PP-SGQF, and SIR-SGQF now have complete same-chart HNN-versus-exact-gradient
accuracy and descriptive performance-screen passes. STR-UKF has valid HNN
sampling and healthy matched-mechanics timing, but direct exact posterior
agreement remains unresolved after repeated exact-arm energy-health failures.
See `docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-terminal-result-2026-07-18.md`.

Decision: `PROGRAM_COMPLETE_TIER_A_FIVE_OF_FIVE_VALIDITY_TIER_B_EIGHT_REQUALIFICATION_BLOCKERS`.

The corrected position-only learned-force kernel passed the prospective
one-seed validity contract in all five posterior-target configurations that
had durable BayesFilter target and NeuTra chart artifacts. These five
configurations represent four model families. The matching zero-residual
control also passed all five. Eight additional historical configurations were
audited but not rerun because their selected frozen charts were not durably
reconstructible.

## Final Cell Ledger

| Cell | Model/filter posterior | Learned validity | Performance status | Minimum truth tail |
|---|---|---|---|---:|
| LGSSM-KF | exact Kalman likelihood | confirmed, one seed | descriptive screen pass | 0.06637 |
| PP-UKF | fixed UKF | confirmed, one seed | not demonstrated: matched ledger missing | 0.2062 |
| PP-SGQF | fixed SGQF | confirmed, one seed | not demonstrated: matched ledger missing | 0.2177 |
| SIR-SGQF | fixed SGQF | confirmed, one seed | not demonstrated: matched ledger missing | 0.3697 |
| STR-UKF | structural fixed UKF | confirmed, one seed | not demonstrated: matched ledger missing | 0.2777 |

The machine-readable count and evidence ledger is
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/final_cell_ledger.json`.
It counts filters as separate posterior configurations but not as separate
model families, and it never counts arms, seeds, training recipes, or tuning
candidates as models.

## Validity Evidence

Every tested learned arm and zero-residual arm passed:

- deterministic value-only endpoint parity against the complete transformed
  target, including the chart log-Jacobian;
- exact archived reconstruction of the joint potential-plus-kinetic
  Metropolis energy difference;
- retained modern R-hat at most `1.01`, bulk ESS at least `1000`, and tail ESS
  at least `400` with retained warm-up;
- the predeclared one-seed truth-tail ladder; and
- target-specific plain-HMC mean agreement where an admitted comparator
  existed.

The structural raw-coordinate comparator remained source-geometry blocked.
No comparator was fabricated. The structural result independently preserved
the deterministic completion and forbade artificial `k_t` process noise.

## Performance And Statistical Interpretation

LGSSM-KF alone has the complete matched same-chart true-gradient and
amortization ledger. Its learned arm passed the predeclared descriptive
performance screen: reuse seconds per minimum bulk ESS were `0.2483` versus
`0.6193`, and sampling-only values were `0.1098` versus `0.2279`. The
zero-residual arm was descriptively better than the learned arm, so residual
learning benefit was not demonstrated.

P4 and P5 do not contain a matched same-chart true-gradient cost arm or the
complete amortization ledger. Their four cells are therefore
`PERFORMANCE_NOT_DEMONSTRATED_MISSING_MATCHED_LEDGER`. Acceptance, observed
ESS, losses, and runtimes do not fill that gap. This classification does not
claim that these candidates are intrinsically slow.

| Inference question | Status |
|---|---|
| Hard veto screen | passed in all five tested configurations |
| Viable corrected learned-force candidates | five |
| Statistically supported arm ranking | none |
| Descriptive performance screen | passed only for LGSSM-KF |
| Descriptive-only differences | all one-seed acceptance, ESS, loss, and runtime differences |
| Default readiness | not established |
| Next evidence | matched performance ledgers, independent fixtures/seeds, or a separately planned fresh-chart Tier B campaign |

## Tier B And Knowledge Transfer

The historical matrix contains eight further configurations across six model
families: funnel, ill-conditioned Gaussian, German logistic regression,
NK-analytic, real NK, NK SVD-UKF, Rotemberg linear-Kalman, and Rotemberg
second-order SVD. Their historical result notes remain useful context. None
enters the corrected-kernel denominator because its exact selected chart state
could not be reconstructed.

P6 and P7 therefore closed all eight as `REQUALIFICATION_BLOCKED`. This is a
reproducibility and artifact-preservation failure. It is not a corrected-kernel
failure, a historical NeuTra invalidation, or a pooled DSGE failure. A future
fresh-chart campaign must be labeled as such and preserve versioned transport
checkpoints.

## Decision Table

| Decision field | Status |
|---|---|
| Program objective | achieved for durable Tier A matrix; Tier B honestly classified |
| Primary validity criterion | passed in five of five tested configurations |
| Main uncertainty | one fixture and one seed per tested configuration; incomplete cross-model performance comparison |
| Strongest alternative explanation | strong NeuTra charts, not learned residuals, provide most of the practical benefit |
| What would overturn the conclusion | parity/energy replay failure or independent valid runs showing systematic posterior disagreement |
| What is not concluded | calibration, universal validity, superiority, default readiness, or success on the eight blocked cells |

## Run Manifest

| Field | Value |
|---|---|
| BayesFilter git commit recorded by serious runs | `15170e1573d19b235d96f3ed3525fa2071f58320` |
| Environment | `tf-gpu`, TensorFlow 2.19.1, TFP 0.25.0 |
| Device policy | GPU/XLA/TF32 with memory growth and managed-session trust basis |
| Seeds | target-specific manifests under P3-P5 artifacts |
| Charged campaign wall time | `1.67` CPU hours, `2.703987` GPU hours |
| Output root | `docs/plans/artifacts/corrected-neural-force-hmc-20260717/` |
| Plan | `docs/plans/bayesfilter-hnn-surrogate-hmc-master-program-2026-07-17.md` |
| Result | this file |

The charged GPU total includes all P3-P5 smoke and serious runs recorded in the
budget ledger. P6-P8 launched no GPU experiment.

## Terminal Verification And Review

- `64` focused kernel, training, campaign, and five-target tests passed under
  explicit CPU-only execution in `406.74` seconds; the only warnings were two
  upstream TensorFlow Probability `distutils` deprecations.
- All campaign JSON ledgers passed `python -m json.tool`.
- The campaign modules and benchmark entry points passed `py_compile`.
- `git diff --check` passed for the program files.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` built the
  complete 438-page monograph. Existing unrelated duplicate-label and missing-
  citation warnings remain; the new chapter section introduced no build
  failure.
- Claude performed the required bounded read-only review of this result and
  returned `VERDICT: AGREE`, finding the tested/blocked count, filtering
  boundaries, one-seed limits, and validity/performance separation internally
  consistent.

Terminal review record:
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p8/terminal_review.json`.
