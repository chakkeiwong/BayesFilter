# Independent SQMC mathematics and implementation audit

Audit date: 2026-09-25. Checkout: `BayesFilter-SQMC`, branch `sqmc-development`, inspected commit `f3995a06a467f16574f96bbc8a68ccbbc4e30dad`.

**Verdict: the recent transfer campaign is not mathematically or scientifically validated. Several campaign adapters compute the wrong target or the wrong derivative, and the four-route transfer includes a duplicated route. The shared implementation has useful passing correctness tests, but it also has a confirmed annealing regression. These findings reject the campaign's correctness and generalization claims; they do not reject SQMC as a research direction.**

This audit examined the recent transfer commits through `e6e5fc40`, their executable dependencies, the original LGSSM tuning and oracle comparison, earlier route-resolution claims, and the integration state through the commit above. It traced the consumers into the shared executor rather than judging the existence of capable helper functions. Runtime code has not been repaired or changed by this audit.

The [audit plan and evidence directory](../plans/artifacts/sqmc-independent-audit-20260925/plan.md) contains the command manifests, complete logs, deterministic reproducer, source copies, and structured findings. All executed numerical checks were deliberately CPU-only, with GPU devices hidden before importing TensorFlow. No new research campaign or GPU benchmark was run.

## Severity and disposition

| ID | Severity | Verified finding | Consequence |
|---|---|---|---|
| A1 | Critical | New generic and transfer model adapters omit parameter derivatives; an actual finite-program comparison returns zero analytical scores for nonzero finite differences | Their analytical-score claims are wrong |
| A2 | Critical | Generic P44 adapter squares covariance entries a second time and changes the initial law | Its Kalman comparison is not a same-target comparison |
| A3 | High | Transfer data observe the initial state before transitioning; the executor transitions before its first observation; the purported 3D baseline also changes H and dynamics | Transfer results do not replicate the frozen tuning target |
| A4 | High | Permutation ablation uses the same cap and selected controls as the ordinary permutation route | Transfer phases contain three distinct route computations, not four |
| A5 | High | Finite value completion replaces the planned score/oracle measurement | 112 completed cells do not answer score transfer or accuracy |
| A6 | High | “Fisher-scaled” and “induced HMC” metrics have incorrect mathematical interpretations; the former is a selection veto | Existing tuning cannot certify HMC suitability under those criteria |
| A7 | High | Non-finite executor return can be marked `valid=True` | Harness success is not a numerical-validity check |
| A8 | High | Three existing annealing tests fail because all `annealed_stages > 1` calls are rejected | An advertised shared-executor capability regressed |
| A9 | High, claims | Standard SQMC theorems are not a proof for the rank-permutation plus Contract-E/correction construction | Consistency, unbiasedness, global smoothness and HMC validity need separate evidence |
| A10 | High, claims | Earlier “equivalent routes” and HMC conclusions exceed their statistical and mathematical support | Preserve measurements, withdraw the stronger interpretation |
| A11 | Medium | Tuning accepts a configuration with only a majority of valid seeds; the campaign bypasses scope validation and lacks holdout proof | Latent selection weakness and ineligible promotion provenance |
| A12 | Medium | Missing local tuning evidence in the new worktree; missing historical Austria raw results at cited paths; eager/NumPy paths and quadratic ancestry work | Reproducibility and production-readiness remain incomplete |

“Critical” here means a decisive blocker for the claimed mathematical quantity, not a security classification. Confirmed latent bugs are distinguished below from failures observed in the saved campaign.

## A1. The new score adapters do not differentiate their own parameterized program

The [generic adapter](run_sqmc_generic_lgssm.py), lines 126–193, converts `theta` to NumPy, captures physical parameters as constants, creates a direction setter that its callbacks never use, and returns zero density/covariance tangents. Its transition tangent includes only propagation of an incoming state tangent. The [10D adapter](run_sqmc_10d_t120_tuned.py), around lines 164–206, and both transfer scripts repeat the omissions. The dimension-generic branch of [the tuner](run_sqmc_tuning.py), lines 350–441, does so too. The original five-parameter, 3D branch instead calls `diagonal_lgssm_canonical_model`; it must not be conflated with these broken adapters.

For transition mean \(f_\theta(x)=\Phi(\theta)x\), the total directional derivative is

\[
 df=\Phi\,dx+(d\Phi)x.
\]

The omitted second term is nonzero for the declared transition parameters. For an isotropic Gaussian density, with residual \(r=z-\mu\), dimension \(d\), and standard deviation \(\sigma\),

\[
 d\ell=-\frac{r^\top dr}{\sigma^2}
       +\left(\frac{\|r\|^2}{\sigma^3}-\frac d\sigma\right)d\sigma.
\]

Returning zero loses both state/mean dependence and scale dependence. Likewise \(Q=q^2I\) requires \(dQ=2q\,dq\,I\). The correct transition tangent at a fixed state `[1,2,3]` for the generic rho direction is approximately `[0.517008, 0.878914, 1.085717]`; its callback returned `[0,0,0]`.

The [deterministic reproducer](../plans/artifacts/sqmc-independent-audit-20260925/diagnostic_probes.py) rebuilt the generic adapter at perturbed parameters, keeping the particle inputs and noise fixed, and evaluated the actual canonical executor with Contract-E. At N=12, T=1, float64, with central-difference step `1e-5`, it found:

| Parameter | Returned analytical derivative | Same-program central difference |
|---|---:|---:|
| rho parameter | 0 | -1.0887155366 |
| log process scale | 0 | -0.5818450908 |
| log observation scale | 0 | -0.7013888658 |
| initial mean scale | 0 | 0 |

The finite value was `0.8425433588`. The fourth zero is another defect: that adapter never uses its declared initial-mean parameter. This comparison does not confuse Monte Carlo error with differentiation error: it differentiates the same finite program with the same realized inputs. A derivative through an intentionally frozen closure would indeed be zero, but that is not the parameterized model derivative the consumer claims.

The 10D runner also calls a single-direction executor once and saves a one-element score for a 12-parameter model; a full score requires all declared directions. Its `_oracle_score_10d_t120` is an explicit zero placeholder, not a Kalman oracle. Saved Phase 0 rows contain `[0.0]`.

Evidence: [deterministic-probes.json](../plans/artifacts/sqmc-independent-audit-20260925/deterministic-probes.json).

## A2. The generic P44 candidate and its oracle are different models

The target in `tests/highdim/test_p44_lgssm_exact_baseline.py::_physical_parts`, lines 40–63, defines covariance diagonal entries

\[
 Q_{ii}=e^{\theta_1}a_i,\qquad R_{ii}=e^{\theta_2}b_i.
\]

The generic adapter, lines 145–149, uses their squares. At the declared 3D parameter value:

| Quantity | Candidate diagonal | P44 reference diagonal |
|---|---|---|
| Q | `[0.026244, 0.039204, 0.054756]` | `[0.162, 0.198, 0.234]` |
| R | `[0.0144, 0.020736, 0.009216]` | `[0.12, 0.144, 0.096]` |

The reference raw initial law has mean `[0.04,-0.02,0.01]` and covariance diagonal `[0.6,0.8,1]`. The candidate generates standard-normal initial particles, uses identity initial covariances, and omits the mean parameter. The P44 reference adapter predicts the raw initial law once before passing it to its observation-first Kalman implementation. A repair must align this convention explicitly, rather than merely changing a covariance power.

The generic data generator has its own initial-law/timing mismatch. Separately, the generic P44 branch added to the tuner uses an isotropic RMS covariance scale for simulation but `exp(theta[1])` in the supplied transition log density. That density is not the Gaussian from which its pre-flow innovations were drawn when the component scale factors differ. This latter branch was statically checked; it is not evidence that the preserved original 3D tuning ran that branch.

Verdict: wrong relative to a P44 same-target value/score claim. Correcting only the zero tangents would still leave the oracle comparison invalid.

## A3. The transfer generator and executor disagree at time zero

`run_sqmc_horizon_transfer.py::_generate_lgssm_data`, lines 56–77, observes \(x_0\sim N(0,I)\) at its first time point and transitions only for `t > 0`. The corresponding dimension and 10D generators use the same convention. The shared executor, `ledh_canonical_score_tf.py`, lines 365–388, predicts and simulates a transition at every iteration, including `t=0`.

With identity observations, the generator's first-observation covariance is \(I+R\). The candidate's first-observation covariance is \(\Phi\Phi^\top+Q+R\). For \(\phi=0.95,q=0.6\), the latent variance is 1 versus 1.2625. This is an exact model discrepancy. Evaluating a different model on those observations can be mathematically defined, but it cannot be presented as matched-model replication.

The frozen tuner also uses a nonidentity 3-by-3 observation matrix and theta `[0.9,0.8,0.7,0.6,0.8]`; the transfer scripts use identity observations and dimension-dependent phi beginning `[0.95,0.90,0.85]`. Equal D, T and N therefore do not recover the original tuning target. The planned Phase 1 equivalence check was not executed.

## A4. The transfer ablation is a duplicate computation

The tuner, line 497, uses coordinate cap 0.98 normally and 0.97 for `is_ablation=True`. Both transfer scripts and the 10D runner use 0.98 for every route. They map both permutation labels to `hilbert_permutation_one_to_one`.

An executor-interception probe verified identical model tensor inputs, noise, ancestor uniforms and all executor options for the two labels at fixed controls. The original tuning artifacts show that those two labels selected identical controls too. The saved Phase 2 and Phase 3 values are **exactly identical for every matched permutation/ablation pair**: eight pairs in Phase 2 and sixteen in Phase 3.

Thus this is observed contamination of the comparison, not merely a possible future bug. A difference in timing between these identical calls is not evidence about the cap ablation.

Evidence: [tuning artifact inspection](../plans/artifacts/sqmc-independent-audit-20260925/tuning-artifact-inspection.json), [transfer record audit](../plans/artifacts/sqmc-independent-audit-20260925/transfer-record-audit.json).

## A5. The recorded experiment does not answer the stated question

The successful transfer record comprises 16 Phase 0 cells, 32 Phase 2 cells and 64 Phase 3 cells. Phase 0 saves zero singleton scores; Phases 2 and 3 explicitly execute `with_score=False` and save no score or oracle error. The earlier Phase 2 attempt failed all 32 cells on tuple unpacking and was followed by a successful retry. Its preservation is appropriate.

All 112 successful cells have finite saved values. That establishes completion of those diagnostics. It establishes neither value accuracy nor score accuracy, transferability of tuned controls, equivalence of algorithms, or absence of a need to tune. Changes in unreferenced raw log values across dimensions and horizons are not error measurements.

The [September 24 closeout](sqmc-campaign-final-summary-20260924.md) correctly narrows these claims and supersedes the Phase 0–4 overstatements. Its cautious conclusion is supported by this audit; the original generalization conclusion is not.

## A6. The Fisher and HMC metrics are mathematically wrong as labeled

The governing [score-metrics note](../plans/sqmc-oracle-principled-score-metrics-2026-09-11.md), lines 55–99, defines two quantities whose interpretations are incorrect. The tuner implements them at lines 540–556 and uses the first as a hard constraint at lines 230–255.

### Fisher information

For score \(s(y,\theta)=\nabla_\theta\log p_\theta(y)\), Fisher information is \(I(\theta)=E_\theta[ss^\top]\), under the usual regularity assumptions. A realized score magnitude is not that expectation, and taking another square root does not repair the substitution. For a scalar normal location model with variance 1, observed at \(y=\theta\), the realized score is zero while Fisher information equals 1.

The code reports `abs(error)/sqrt(abs(oracle_score))` away from zero, and **reports zero when the oracle component is close to zero regardless of the candidate error**. An error of 1 at that normal-model point is therefore reported as zero. This is a decisive counterexample to its information-scaled error interpretation. If Fisher information is actually available and nonsingular, an explicitly defined quadratic scale such as \(e^\top I^{-1}e\) can be meaningful; that is not what this code computes.

### HMC local error

The note writes the update as \(q_{t+1}=q_t+\epsilon\nabla\log p(q_t)\), which is not HMC leapfrog. For constant mass matrix M and force \(g=\nabla\log\pi\), the relevant first updates are

\[
 p_{1/2}=p_0+\frac\epsilon2g(q_0),\qquad
 q_1=q_0+\epsilon M^{-1}p_{1/2}.
\]

At the same initial position and momentum, replacing g by \(g+e\) produces

\[
 \Delta p_{1/2}=\frac\epsilon2e(q_0),\qquad
 \Delta q_1=\frac{\epsilon^2}{2}M^{-1}e(q_0).
\]

This derivation follows Neal's equations (2.28)–(2.30); see the [cached source](../plans/artifacts/sqmc-independent-audit-20260925/sources/neal-hmc-1206.1901v1.txt), lines 397–406. It does not support the implemented `epsilon * ||error|| / ||oracle_score||`. That expression also reports zero when the oracle norm is zero. Later trajectory error requires the evolving dynamics; the displayed first-step result is not a trajectory-error bound or a replacement HMC admission rule.

Cosine 0.9995 corresponds to an angle of approximately 1.812 degrees. “Direction correct to 0.05%” is not an established accuracy interpretation, and cosine/norm thresholds alone do not certify HMC usability. No downstream HMC claim follows from these metrics.

The selected original tuning controls may remain candidates. Their reported L2 comparisons are distinct quantities, and this audit does not prove those raw L2 measurements false. However, invalid metric interpretations affect the documented constraints and promotion argument. The purported HMC objective is also just a constant multiple of L2 when the oracle and epsilon are fixed.

Evidence: [exact counterexamples](../plans/artifacts/sqmc-independent-audit-20260925/metric-counterexamples.json). These are algebraic counterexamples, not stochastic rankings.

## A7. The harness does not respect the executor's failure signal

The shared executor, lines 698–722, aggregates numerical validity and can return a `-inf` value with a zero score on failure. This matters especially where assertions are removed by compilation. The transfer wrappers instead set `valid=True` after any non-raising return; see the horizon script, lines 249–256.

Injecting a `-inf` executor result reproduced `valid=True` in both the horizon and 10D wrappers. The saved successful records inspected here are finite, so this test does **not** establish that those particular records encountered an invalid reset. It establishes that their validity field is insufficient and that a failed future call could be counted as success. The repair should check the returned value, score when requested, and supported explicit status before admitting a cell.

## A8. Annealed execution has a confirmed regression

Three existing tests in `test_ledh_canonical_score_full.py` fail immediately with the explicit `annealed_stages=1` restriction in the executor, lines 231–236. The affected tests are:

- `test_annealed_telescope_score_matches_oracle`;
- `test_annealed_telescope_with_reset_score_matches_oracle`;
- `test_annealed_with_full_production_program_matches_oracle`.

Git blame attributes the restriction to `e7f2a88ec`, September 14, before the recent integration. The public signature still exposes annealing and the implementation docstring still describes it. This is not a merge-created mathematical error, but it is a real capability regression in the inherited work. The recent transfer runs used the supported single-stage path; their other defects are independent of this regression.

Evidence: [extended test log](../plans/artifacts/sqmc-independent-audit-20260925/extended-tests.log).

## A9. What the SQMC source theory does and does not establish

Gerber and Chopin, arXiv:1402.4039v5, Algorithm 3, sort the joint point rows by ancestor coordinate, Hilbert-sort the states, and apply the weighted empirical inverse CDF while preserving each row's innovation coordinates. The inspected author-maintained `particles` implementation, `core.py::resample_move_qmc`, lines 339–355, follows that structure. The cached code is a later author-maintained implementation, not claimed to be the exact code used for the original paper experiments.

The local joint-row sorting and inverse-CDF primitive follow those mechanics. The rank-permutation route is different: it assigns each ordered ancestor exactly once. Equal weights alone do not make inverse-CDF sampling of a randomized Halton coordinate a permutation. For N=12, dimension 3, seed 17, the implemented point set gave ancestor indices `[1,2,2,3,4,5,7,8,8,10,11,11]`, with only nine unique ancestors. A shifted systematic grid with equal weights is a different construction that can produce a permutation.

This observation does not make the permutation candidate intrinsically invalid. It makes it an adaptation requiring its own argument. Theorems 5–7 assume Algorithm 3 plus explicit regularity and point-set properties; Lemma 8 concerns unbiased normalizing constants for that construction. The Contract-E moment reset, higher-moment correction and deterministic rank pairing are not silently covered. Nor does an unbiased likelihood imply an unbiased log likelihood or score.

The paper's method, theorem assumptions, relevant consistency-proof passage, and numerical design were inspected. Its numerical study uses scrambled Sobol constructions; it is not a validation of this particular Halton/reset/correction implementation. No new asymptotic proof was attempted here.

For the local analytical score, Hilbert order and ancestor indices are discrete decisions. Holding realized indices fixed can describe a local branch derivative where the decisions stay unchanged. It does not establish global differentiability across ordering or ancestor-selection boundaries, nor unbiased differentiation of an expected likelihood. The passing same-branch derivative tests have that scope.

Source copies, identities and hashes: [SQMC paper](../plans/artifacts/sqmc-independent-audit-20260925/sources/gerber-chopin-sqmc-1402.4039v5.pdf), [author implementation](../plans/artifacts/sqmc-independent-audit-20260925/sources/particles-core.py), [source manifest](../plans/artifacts/sqmc-independent-audit-20260925/sources/sources.json). A broader citation-network survey was not needed for these concrete defects; no literature-exhaustiveness claim is made.

## A10. The older four-route statistical interpretation overreaches

The September 12 [resolution](../plans/sqmc-route-comparison-resolution-2026-09-12.md) turns paired intervals containing zero into “equivalent routes,” and turns the metrics above into acceptable HMC gradients. The [Austria report](../plans/sqmc-4route-comparison-final-report-2026-09-09.md) also compares a range of score norms with a typical within-route standard deviation and calls this statistical indistinguishability.

A confidence interval containing zero does not establish equivalence. Equivalence requires a scientifically justified margin and an appropriate interval/test relative to that margin. Similar gradient norms cannot establish similar gradient vectors. A larger estimated log likelihood is not automatically more accurate without a reference target or justified error criterion. Comparisons of dense and streaming execution additionally vary more than ancestry.

The analysis script implements paired bootstrap differences and checks common seed sets; that structure is useful. Its words `SUPERIOR` and `INFERIOR` refer only to a sign in raw value differences, however, not verified estimator accuracy. Its loader treats a missing `program_valid` as true. The four cited Austria raw files were not present at their documented paths in either the new or old SQMC worktree. This audit therefore did not reproduce their intervals or independently validate the Austria call chain; it audited the available analysis code and stated inference. It does not assert that the quoted numerical intervals are fabricated or numerically wrong.

## A11. Tuning and scope enforcement are incomplete at the consumer

The reusable `SQMCOrderingScope` and artifact matching tests passed. The transfer scripts do not call that validator: they read JSON `best_controls` directly. A successful unit test of a validator therefore provides no scope-enforcement guarantee for these consumers. Changed model, initial law, dimension, horizon, N and execution mode are new scopes. Phase 2 reuses tuning seeds 50001–50004; the 16-seed source tuning uses 50001–50016. No untouched claim validation is established by this reuse.

`ConfigEvaluation.valid` in the tuner is `valid_fraction > 0.5`, and objective means are computed on valid seeds only. The downstream nondominance routine filters on that property. Consequently a candidate with 9/16 valid seeds can be selected after seven failures are excluded. All grid entries in the four preserved tuning artifacts have valid fraction 1, so this is a confirmed latent selection defect, not an explanation of their particular winners. Hard numerical failure should not become survivor-conditioned scientific evidence without an explicit contract.

The artifact describes constraints as applied to a seed-aggregated mean, but the Fisher test actually takes the maximum component across valid seeds. This provenance inconsistency further argues for replacing the metric contract before a new tuning campaign.

## A12. Reproducibility and execution boundaries

The four original tuning JSON files survived in the old Claude worktree but were absent from the new worktree because they are ignored local evidence. They have now been copied byte-for-byte into the same relative paths in `BayesFilter-SQMC`, with [source/destination checksums](../plans/artifacts/sqmc-independent-audit-20260925/restored-tuning-artifacts.json). No existing file was overwritten with differing content. They remain ignored and uncommitted; a fresh Git checkout alone is still insufficient to reproduce the runs.

The transfer JSONs do not record a complete executable provenance chain: exact command, source identity, environment, verified device placement, JIT and memory growth are incomplete. The closeout correctly labels these float64 eager diagnostics. Their NumPy data/model construction and uncompiled calls do not satisfy the project's claim-bearing TensorFlow GPU/XLA path. The tuner also uses NumPy selection calculations despite the current runtime/selection policy. These issues block production promotion; they do not invalidate every diagnostic calculation merely because it used NumPy or CPU.

`inverse_cdf_ancestor_indices` constructs an N-by-N comparison matrix, and `_transition_ancestors` constructs another for the unique-ancestor diagnostic. Hence the present implementation cannot inherit the paper's O(N log N) complexity claim. This is a static scaling finding; no performance benchmark was run.

## What passed, and the call chains actually checked

The core receives the children, covariance tangents and weight tangents, carries ancestry consistently across states and covariances, sends the weight derivatives into Contract-E, and delegates higher-moment correction to the batched general implementation. The inspected flow determinant tangent uses the QR/triangular-solve form of \(d\log|\det J|=\mathrm{tr}(J^{-1}dJ)\). These are meaningful positive findings, subject to the fixture limits below.

| Check | Result | What it supports |
|---|---|---|
| SQMC primitives, tuning scope, supported full-score fixtures | 28 passed; 4 deliberately deselected in first batch | Ordering/ancestry mechanics, scope helper, finite-program derivatives and validity propagation on those fixtures |
| Additional flow, step, recursion and UKF tangents | 6 passed | Stage and recursion derivative parity |
| Existing annealed full-score fixtures | 3 failed | Confirmed unsupported multi-stage capability |
| Original canonical 3D LGSSM and Kalman oracle tests | 5 passed | Original adapter's tested parameter directions and small oracle checks |
| Independent Hilbert grids: 2D at 2/3 bits, 3D at 2 bits | All unique keys; adjacent Manhattan distance 1 | Stronger grid property beyond agreement between wrappers sharing one implementation |
| Independent campaign probes | Defects A1/A2/A4/A7 reproduced | Direct evidence at the actual consumer/core boundary |

Total existing tests run: **39 passed, 3 failed**. Deselections overlap across batches and are not extra tests. Total recorded numerical test/probe wall time was approximately 88 seconds against a 1,800-second CPU audit budget. The expensive production fixture, GPU/TF32/XLA execution, full-scale corrected scores, all nonlinear model source routes and all historical campaigns were not rerun. Passing a derivative test of a finite program does not prove that the finite program approximates the desired statistical model accurately.

The original LGSSM tests include explicit non-vacuity for process/observation covariance directions. This makes the new adapters' return to zero tangent placeholders particularly avoidable: an appropriate reusable model adapter and relevant tests already existed.

## Decision and next justified work

| Decision | Primary criterion status | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not use the transfer campaign as score-accuracy or HMC evidence | Failed: wrong/missing scores and mismatched targets | Reproduced derivative error, target mismatch and duplicate ablation | Correct performance after repair | Repair shared model/consumer wiring and run deterministic parity first | SQMC direction rejected |
| Retain the supported core as a repair starting point | Limited finite-program checks passed | Annealed capability fails | Full production scope and nonlinear fidelity | Restore or explicitly retire unsupported annealing; validate affected consumers | Entire core certified correct |
| Reopen the metric and tuning contract | Fisher/HMC interpretations invalid | Algebraic counterexamples; scope bypass | Candidate quality under valid criteria | Define information/error quantities correctly, require all mandatory diagnostics and fresh holdout separation | Existing L2 observations erased or a route ranked |

| Inference status | Audit conclusion |
|---|---|
| Hard veto screen | Wrong analytical derivative and wrong comparator are established; required scores are absent; duplicate ablation is observed |
| Statistically supported ranking | None established by this audit; historical Austria intervals not reproduced |
| Descriptive-only differences | Existing values, norms, runtimes and selected-control metrics remain descriptive within their actual targets |
| Default-readiness | Not established by this campaign or these CPU tests |
| Next evidence needed | Same-target Kalman checks, all-direction analytical derivative parity, distinct ablation wiring, valid exact-scope tuning and untouched multi-seed validation |

The repair order should be: define one model and timing convention; implement its analytical callbacks once; test every parameter direction against the same finite program; verify model equality with the exact Kalman oracle; enforce finite/status checks; restore the cap ablation; correct the mathematical metrics and scope/holdout contract; then perform only the smallest newly justified research run. Preserve the old outputs as evidence of the failed measurement process.

The strongest alternative explanation for some historical discrepancies is a changed model rather than poor SQMC performance. A corrected same-target experiment could overturn a negative performance impression. It cannot overturn the reproduced fact that the inspected callbacks omit derivatives. The weakest remaining audit coverage is production GPU behavior and the missing historical Austria raw evidence. The audit is complete for the inspected files and bounded checks, not a proof of every repository algorithm or historical result.
