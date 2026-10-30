# Five-model precision result: final filter likelihood and analytical score

The owner-approved 1% threshold was applied to **FP32 covariance restoration** in the shared finite filter. It did not relax the factorization, mean, rank, finite-value, correction-cap, or FP64 guards. The completed experiment compared final log likelihoods and every analytical score coordinate from FP64, ordinary FP32, and FP32 with TF32 matrix operations, on byte-identical FP32-representable observations, parameters and particle designs. The primary result is mixed: ordinary FP32 passes the declared materiality screen for LGSSM, KSC SV, predator–prey and all measured range-bearing scopes, but fails SIR's score screen on one dataset. TF32 passes that screen for LGSSM and KSC SV, fails it for SIR, and cannot finish any measured range-bearing or predator–prey run.

This measures arithmetic sensitivity of the **finite covariance-guided defensive-mixture filter**, a prerequisite to the surrogate-HMC question. It is not a surrogate-HMC run. FP64 is the arithmetic comparator on the same rounded inputs, not an exact filtering oracle. The score is the runtime's analytical recursive derivative of its finite value program; the comparison does not independently prove that derivative formula or posterior validity.

## Experiment and decision rule

The [plan](ledh-five-model-tf32-2026-10-10.md) fixed N=1008, T=50, two saved independent datasets and eight particle designs each for the P44 LGSSM (state/observation dimensions 3/3), KSC SV (1/1), predator–prey (2/2), and SIR (18/9). Range bearing used T=20, dimensions 4/2, N=1008/2016/4032, three saved datasets, truth and difficult parameter points, and eight designs per case. Each scope used its own previously saved FP64 tuning controls **frozen as a diagnostic comparator**, not as FP32/TF32 tuning admission. All numerical evaluations used the same shared TensorFlow/XLA GPU filter and analytical score route. FP32 inputs were prepared once by FP64 generation followed by common FP32 rounding, including the FP64 comparison arm.

The predeclared screening limit for each dataset/point is RMS paired final-log-likelihood error at most `max(0.01, 0.1 × FP64 design SD)` **and** RMS Euclidean final-score error at most `max(0.01, 0.1 × FP64 score spread)`. This is a materiality convention; it is not a posterior-error theorem. Invalid or incomplete trajectories veto that precision arm. Acceptance/speed of failed prefixes and covariance residual alone never substitute for final likelihood or score. Relative-to-final-score percentages are reported separately because a literal 1% final-score rule differs from this predeclared materiality screen.

## Complete evaluations and final-output discrepancies

The table gives the **largest dataset/point RMS** in each scope, not a pooled average. A dash means no complete final value/score exists. Speed is the observed median complete FP64 time divided by median complete candidate time, excluding compile and invalid prefixes; it is descriptive single-machine evidence.

| Model, N | Precision | Complete / planned | Materiality cells | Worst RMS log-likelihood difference | Worst RMS score-vector difference | Observed speed / FP64 |
|---|---|---:|---:|---:|---:|---:|
| LGSSM, 1008 | FP32, TF32 off | 16/16 | 2/2 | 0.0000200 | 0.0000837 | 3.04× |
| LGSSM, 1008 | FP32, TF32 on | 16/16 | 2/2 | 0.00529 | 0.00789 | 3.30× |
| KSC SV, 1008 | FP32, TF32 off | 16/16 | 2/2 | 0.00102 | 0.00139 | 6.92× |
| KSC SV, 1008 | FP32, TF32 on | 16/16 | 2/2 | 0.00102 | 0.00139 | 6.73× |
| Predator–prey, 1008 | FP32, TF32 off | 16/16 | 2/2 | 0.000426 | 0.00525 | 3.66× |
| Predator–prey, 1008 | FP32, TF32 on | 0/16 | 0/2 | — | — | — |
| SIR, 1008 | FP32, TF32 off | 16/16 | 1/2 | 0.00669 | 35.15 | 2.42× |
| SIR, 1008 | FP32, TF32 on | 16/16 | 0/2 | 0.524 | 1452.89 | 2.51× |
| Range bearing, 1008 | FP32, TF32 off | 48/48 | 6/6 | 0.00409 | 0.12479 | 3.89× |
| Range bearing, 1008 | FP32, TF32 on | 0/48 | 0/6 | — | — | — |
| Range bearing, 2016 | FP32, TF32 off | 48/48 | 6/6 | 0.01412 | 0.15429 | 7.52× |
| Range bearing, 2016 | FP32, TF32 on | 0/48 | 0/6 | — | — | — |
| Range bearing, 4032 | FP32, TF32 off | 48/48 | 6/6 | 0.02009 | 0.13763 | 7.73× |
| Range bearing, 4032 | FP32, TF32 on | 0/48 | 0/6 | — | — | — |

All 208 FP64 and all 208 ordinary-FP32 measurements finished; TF32 finished 48 of 208. The terminal [audit](artifacts/ledh-five-model-tf32-20261010-01/terminal-audit.json) passed for all 624 planned measurements, complete scope/seed coverage, source identity, byte-identical paired inputs, GPU placement, graph stability, memory growth and every accepted trajectory's full-horizon validity. The complete per-cell [comparison](artifacts/ledh-five-model-tf32-20261010-01/report.json), per-model [summary](artifacts/ledh-five-model-tf32-20261010-01/summary.json), final likelihood/score [CSV](artifacts/ledh-five-model-tf32-20261010-01/final-likelihood-and-scores.csv), [execution record](artifacts/ledh-five-model-tf32-20261010-01/execution.json), and each worker's manifest and per-design traces preserve the numbers. The 21 workers completed without infrastructure retry in 2668 seconds of aggregate worker wall time, under the 7200-second cap. The separate 26 focused CPU engineering tests passed in 30.63 seconds; their [engineering artifact](artifacts/ledh-five-model-tf32-engineering-20261010-01/result.json) records deliberate GPU hiding.

**SIR is a score failure, despite small likelihood error.** For dataset 26100832, ordinary FP32 has score RMS error 35.15 versus the predeclared limit 4.21 (20.43% of FP64 score RMS magnitude); dataset 26100831 passes with score RMS error 0.654 versus limit 5.05. TF32 SIR is much worse: score RMS errors 1452.89 and 971.11 against limits 5.05 and 4.21; its likelihood RMS errors 0.524 and 0.406 also exceed the respective limits 0.094 and 0.068. The worst ordinary-FP32 design, seed 261008401, has final score-vector difference 97.97 while the final likelihood differs by −0.0189. Its per-step score increments differ by hundreds around steps 23–24 and partly cancel by T=50. The observed restoration residual there is tiny, about 2×10⁻⁷; relaxing that guard further cannot explain or cure the score difference. This localizes the symptom in accumulated score arithmetic, not its root mathematical or numerical cause.

For ordinary FP32 range bearing, every materiality cell passes. Even the largest score RMS error at N=4032, 0.13763, is 4.36% of its FP64 design score spread and 1.24% of the previously measured N1008-to-N2016 score-change RMS for that matched case. The hardest cases nonetheless violate a **literal 1% relative final-score** rule: 5 of the 18 range-bearing dataset/point/N cells have score RMS between 1.21% and 2.08% of FP64 score RMS magnitude. SIR dataset 26100832 also violates that rule. All complete arms' likelihood RMS discrepancies are below 1% of FP64 likelihood RMS magnitude. These relative percentages use score magnitudes as denominators, so they must not be conflated with either the 1% *covariance-restoration* guard or the predeclared design-spread materiality screen. The [per-cell report](artifacts/ledh-five-model-tf32-20261010-01/report.json) retains every numerator and denominator.

## Why TF32 stops in two models

The bounded [first-step guard probe](ledh-five-model-tf32-guard-localization-2026-10-10.md) reproduced the campaign's exact initial inputs and called the shared map, mixture-density and reset primitives. Its [result](artifacts/ledh-five-model-tf32-guard-20261010-01/guard-diagnostic.json) shows `map_valid=true`, `reset_valid=true`, finite value/score/state/tangent and a covariance-restoration error below 0.001 for the first range-bearing and predator–prey cases. Yet `density_valid=false`: the raw TF32 products `B Q Bᵀ` are asymmetric beyond the unchanged scale-relative `safe_cholesky` tolerance. For range bearing, branch-one/branch-two maximum relative asymmetries are 3.41×10⁻⁵/9.74×10⁻⁵ versus tolerance 3.81×10⁻⁶; 1008/789 particle covariance factors reject. For predator–prey, branch one reaches 4.52×10⁻⁶ versus tolerance 1.91×10⁻⁶; all 1008 factors reject. LGSSM, KSC and SIR's first steps pass the same density guard. This first-step probe explains the observed rejection but does not prove the full trajectories of an unmodified repaired TF32 route would be accurate. The earlier range-bearing result correctly identified the old restoration guard as a rejection; this experiment shows that loosening that guard alone leaves an independent factorization rejection.

The guard probe is explanatory, not a new filtering algorithm, and changes no accepted value or tangent. No new Cholesky threshold, covariance symmetrization, ridge, damping or TF32 covariance-product repair was introduced. A proposed repair would require its own no-harm and final-likelihood/score evaluation. At the successful scopes, TF32 adds little observed runtime benefit over ordinary FP32: LGSSM 2.214 to 2.040 seconds per complete evaluation, KSC 0.274 to 0.282, and SIR 6.763 to 6.503; these timings do not rank precision policies statistically.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the 1% FP32 restoration guard as an explicit precision-policy change | Actual-consumer 0.9% pass/1.1% fail boundary checks; 26 focused tests passed | Mean/rank/finite/FP64 checks remain active | This does not establish the accuracy of every FP32 score | Preserve the measured setting and test each downstream scope | HMC or posterior correctness |
| Ordinary FP32 is a viable precision candidate for LGSSM, KSC, predator–prey and these range-bearing cases under the declared *design-spread* screen | 2/2, 2/2, 2/2 and 18/18 cells respectively | Every final value and score exists | Some range score errors exceed 1% of score magnitude; controls were frozen FP64 choices | Scope-specific tuning, strict downstream score checks, then actual HMC if sought | A universal FP32 admission or statistical ranking |
| Ordinary FP32 SIR score is **not accepted** for this scope | 1/2 cells; second dataset score error 35.15 vs limit 4.21 | No run invalid, but score materiality fails | Cause of rare, large per-step derivative discrepancies | Localize score arithmetic on failing designs before HMC | That SIR value accuracy certifies its score |
| TF32 is **not accepted** across all five models | LGSSM and KSC pass; SIR fails materiality; predator–prey and range bearing lack final outputs | Cholesky symmetry rejection in two models | Whether selective precise covariance products and SIR score repair preserve speed/accuracy | Separate bounded numerical repair and full-consumer re-test if TF32 breadth is sought | That loosening another guard would be valid |
| The current five-model *precision experiment* is complete | All 21 planned jobs and terminal audit completed | No source, data, input, GPU or artifact invalidity | Finite datasets, designs and parameters; FP64 is not a nonlinear oracle | Report these limits before any separate surrogate-HMC campaign | Surrogate-HMC readiness |

| Inference question | Status |
|---|---|
| Hard veto screen | TF32 invalid for predator–prey and range bearing; SIR ordinary FP32 and TF32 fail the final-score materiality screen. |
| Statistically supported ranking | None: this design evaluates paired precision discrepancies and describes timing, not population-wide superiority. |
| Descriptive-only differences | All timing ratios, RMS errors, design spread estimates and extreme design discrepancies. |
| Default-readiness | Not established; the saved tuning artifacts are FP64 scope controls, and this is not HMC. |
| Next evidence needed | Resolve SIR score sensitivity; evaluate any TF32 covariance-product repair in the full filter, then use per-scope precision tuning and downstream posterior checks for HMC claims. |

The strongest alternative explanation for favorable scores is that these saved datasets and parameter points omit more precision-sensitive regimes; the strongest alternative explanation for the SIR anomaly is a localized analytical-tangent or branch-sensitivity problem rather than generic FP32 inaccuracy. A stable SIR score under a targeted recomputation or an independent checked derivative would overturn the interpretation that ordinary FP32 itself is generally unsuitable for SIR. No such recomputation was part of this experiment. The present result rejects only the measured precision candidates in the measured scopes, not the filtering algorithm or the surrogate-HMC research direction.
