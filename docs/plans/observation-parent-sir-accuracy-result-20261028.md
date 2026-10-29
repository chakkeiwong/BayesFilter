# SIR conditional-parent comparison, 28 October 2026

Status: implementation, numerical execution, analysis and manuscript complete.
All 1584 expected filter measurements are present, with no invalid filter rows.
Final document rendering and reporting provenance are recorded below.

## Question and implementation
The question is whether exact Gaussian child conditioning and predictive-
likelihood parent allocation improve log-likelihood and the three SIR score
components relative to the original three-branch proposal, at T=10,20,50.
The engineering default is now conditional_observation_weighted. It replaces
the global branch, retains defensive identity/local branches, and evaluates the
full mixture denominator. All supported scalar, directional, batched and
fixed-design consumers reach the shared analytical implementation. The original
policy remains explicit. Non-affine/non-Gaussian observation laws fail closed;
future-guided diagnostic runners explicitly retain their original policy.

This is the covariance-proposal route. It is not canonical LEDH admission or
HMC readiness. Categorical parents are locally fixed in the analytical score;
finite differences away from selection switches verify this finite-program
quantity, not equality to the exact model likelihood score.

## Completed engineering evidence
- Shared SIR Gaussian geometry agrees with the independent derivation.
- Default consumer parity, branch-stable finite differences, fixed-design parity,
  unsupported-model rejection and sharded/full checks: six tests passed (108.50s).
- Original SIR XLA recurrence versus independent directional filter: one test
  passed (18.77s).
- Future-guide rejection and explicit-original zero-guide parity: two tests
  passed (14.34s).
- GPU/XLA pilot: all six arms valid through T50; all 18 prefix rows preserved.
  T10/T20 prefix optimization checked independently; scaled score FD errors
  3.58e-9,4.51e-10,4.29e-9. Runtime226.19s.
- An initial memory-growth startup failure was repaired by configuring the GPU
  before scientific imports. A stopped-row parent-index diagnostic mismatch was
  repaired by masking rejected rows; accepted numerical calculations unchanged.

Compatibility follow-through: historical saved-control readers now interpret
missing proposal_policy as original_three_branch; any explicit saved policy
wins. This avoids silently changing saved tuning/force definitions. Scope checks
remain intact. Fifteen caller files received narrow migrations or CLI exposure;
before/after hashes and protected source copies are in compatibility-before.
The original call-chain and guided pilot regression passed (2 tests,11.12s).
No HMC tuner, sampling rule, or running comparison kernel changed.

## Reference meaning and limitations
The baseline is the saved Zhao-Cui adapted TT smoother, two independent fits
(rank20/fit17 and rank40/fit2), each with 100000 paths per horizon. Exact-target
importance reweighting retains the original proposal denominator. Log likelihood
is log-mean-exp of path weights; the score is their normalized analytical
complete-data score average. Joint densities and complete-data derivatives are
checked against independent canonical model callbacks. These are approximate
references, not exact oracles. Rank and fit seed vary together, so their agreement
cannot establish rank convergence.

The original 21 parameter/horizon cells have complete reference calculations.
All nominal and 0.8 observation-noise cells pass the declared overlap/agreement
screen. Noise1.25 passes T20/T50 but fails T10 (rank40 path ESS425). All +/-10% rate
shifts collapse path ESS to approximately1 and cannot support accuracy rankings.
All four prospective +/-0.1% rate changes pass both reference screens at all
three horizons. Thus20/33 cells have usable references; failed wide shifts are
retained. The local reference run completed24 rows in763.20s. Reference failure is a comparator limitation, not evidence
against any filtering method.

## Review and provenance
Plan: observation-parent-default-and-sir-accuracy-20261028.md. The recorded
skeptical audit checks baseline identity, fixed-data scope, proxy metrics,
conditional heuristic comparisons, uncertainty, stop conditions and budget.
The advisory Claude review returned a login error, hence no independent review
was completed. Authentication was left unchanged; the repository policy allows
execution after the documented local review and focused checks.

All result files live under
 docs/plans/artifacts/observation-parent-sir-accuracy-20261028/ . Per-run manifests
record source hashes, commit, commands, environment, seeds, device, memory policy,
data hash, controls and wall time. This FP64 GPU/XLA comparison is an untuned
precision diagnostic. It does not replace per-scope tuning or establish the
TF32 production route. One fixed dataset cannot establish performance across
newly generated SIR outbreaks.

Source audit during execution: the main campaign and local-reference kernels
match their launch hashes. The earlier pilot predates addition of the expanded
case traversal and masking of diagnostic parent indices on stopped rows. Every
pilot row is valid, so the mask cannot alter its numerical output; the shared
geometry and score checks remain applicable. The first reference run predates
the local-only CLI option; its numerical kernel is unchanged. These differences
are retained in each launch manifest rather than rewriting old provenance.
The final source snapshot is preserved under source-snapshot-final.

## Completed accuracy comparison

The observed score errors fall dramatically versus the original mixture, but
parent weighting is not uniformly better than conditional uniform parents.
Simpler controls also outperform the new default on some score components;
this is a promotion veto, not a footnote to the ESS improvement.

The table reports RMSE against rank40/fit2; all rankings below must also hold
against rank20/fit17. ESS is averaged over time and seeds. Score coordinates are
log multipliers of infection, recovery and observation noise. The likelihood
metric is log-likelihood error, not raw-likelihood relative error.

| T | Proposal | Mean ESS | Log-likelihood RMSE | Infection-score RMSE | Recovery-score RMSE | Noise-score RMSE |
|---|---|---:|---:|---:|---:|---:|
| 10 | Original | 381.94 | 0.961 | 1,290.56 | 539.05 | 347.34 |
| 10 | Conditional, uniform | 472.56 | 0.788 | 30.54 | 12.06 | 4.92 |
| 10 | Conditional, weighted | 733.81 | 0.832 | 38.61 | 17.09 | 4.35 |
| 20 | Original | 597.78 | 0.894 | 11,371.11 | 4,871.67 | 3,488.04 |
| 20 | Conditional, uniform | 651.62 | 0.812 | 31.53 | 6.92 | 3.90 |
| 20 | Conditional, weighted | 828.23 | 0.687 | 41.28 | 12.91 | 3.98 |
| 50 | Original | 796.73 | 0.854 | 18,648.59 | 7,905.32 | 9,593.82 |
| 50 | Conditional, uniform | 816.12 | 0.750 | 32.76 | 7.23 | 3.85 |
| 50 | Conditional, weighted | 911.87 | 0.655 | 44.42 | 13.08 | 4.95 |

All six arms completed at all 11 parameter points and all three horizons, using
eight paired streams (seeds281201–281208) and N1008. The 1566 main rows plus18
pilot rows exactly match1584 expected unique measurements. The maximum recorded
covariance-restoration error is9.222844127446702e-15. No filter validity veto
fired; this does not override the13 reference-overlap vetoes.

Across the20 usable reference cells, weighted conditional proposals have lower
observed score RMSE than original in20/20 cells for each component, and lower
log-likelihood RMSE in19/20. The exception is infection1.001,T10. These counts
are descriptive, conditional on two approximate references.

At nominalT50, the weighted infection-score RMSE is40.92–44.42 across references,
versus18646.67–18648.59 originally; recovery11.96–13.08 versus7904.64–7905.32;
noise4.95–5.02 versus9593.77–9593.82. Log-likelihood RMSE is0.654 versus0.854.
Every paired score error at this cell favors the weighted conditional mixture,
with exact sign p=.0078125, but the paired mean-squared-error intervals cross
zero. For infection against rank40 the mean reduction is3.47768e8 and95% CI
[-4.56538e8,1.15207e9]. Extreme original errors dominate the squared-error
variance. No weighted-versus-original cell passes the full predeclared paired
interval-and-sign criterion. The descriptive reduction is substantial; a
statistically supported MSE ranking is not established by eight streams.

Against conditional uniform parents, weighted parents have descriptively lower
log-likelihood RMSE in15/20 cells (uniform5); infection CW3/uniform13/fit-dependent4;
recovery CW2/uniform15/fit-dependent3; noise CW7/uniform13. Only two componentwise
comparisons meet the exploratory criterion for this pair:

| Case and horizon | Noise-score RMSE: uniform | Weighted | Supported direction under declared exploratory rule |
|---|---:|---:|---|
| Noise0.8,T20 |197.524–197.525|12.345–12.355|Weighted lower MSE; reduction CI about[10183,67544] for either fit, sign p=.0078125|
| Noise1.25,T50 |1.085–1.105|2.851–3.314|Uniform lower MSE; reduction CIs[4.381,9.527] and[6.838,12.685], sign p=.0078125|

There is no multiplicity correction and no claim of familywise superiority.
Other pairwise results are preserved in analysis01/summary.json.

## Heuristic comparison and scientific interpretation
At noise1.25,T50, identity-only noise-score RMSE is1.156–1.208 and conditional-only
uniform is1.157–1.161, both below weighted-mixture2.851–3.314. Thus simple
controls outperform the new default in a requested situation. Across80 usable
cell/quantity combinations, identity-only, uniform-only and weighted-only have
lower observed RMSE than CW against both fits in42,52 and32, respectively.
The complete conditional table and machine-readable promotion veto are retained.
These are descriptive vetoes under the predeclared conservative promotion rule;
a veto is not automatically a statistical ranking.

At nominalT50, weighted-only has ESS1008 but infection-score RMSE95.82–100.96,
versus31.91–32.76 for conditional-uniform. ESS measures concentration of the
current importance weights, not model-score error or accuracy of the incoming
cloud. The nominal mean ESS increases from381.94 to733.81 atT10 and796.73 to
911.87 atT50. The earlier saved stress case improved2.404 to392.512 (~163x), a
distinct experiment documented in the manuscript. Mean ESS across a longer
horizon includes easier later observations and must not be interpreted as an
intrinsic benefit of longer horizons.

Claimed target: log p_theta(y_1:T) and its parameter gradient. Actual filter
quantity: a finite-cloud, reset-dependent log normalizer and its analytical
branchwise derivative, with categorical parent indices held locally fixed.
Actual reference quantity: a fixed-path Zhao-Cui importance normalizer and its
normalized analytical complete-data score. The two are different numerical
approximations to the model target; neither is proven equal to it. Independent
finite differences verify the implemented local derivative, not population
score correctness. Two-fit agreement plus path ESS provides a usable conditional
comparison, not an exact oracle. Reference rank and fit seed change together.

## Decision table
| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep requested conditional-weighted engineering default |Implemented and tested; measured ESS improvement |No implementation/filter validity veto in this run; heuristic accuracy veto remains |Scope/precision dependence |Keep explicit original/uniform alternatives and historical-policy compatibility |Accuracy-optimal default or canonical admission |
| New conditional mixture versus original score accuracy |Lower observed RMSE in20/20 cells per component; full paired MSE criterion not met |No filter-invalid rows;13 reference cells vetoed |Heavy tails with8 streams; reference bias |More paired streams under a new bounded plan |Statistically established MSE superiority |
| Weighted versus uniform parent accuracy |Mixed; two exploratory noise-score findings in opposite directions |Simpler controls outperform CW in some cells |Multiplicity, tuning and categorical choices |Investigate derivative variability and scope-specific tuning |Uniform score improvement from weighting |
| Wide rate-parameter comparison |Not assessable: path ESS~1 for both saved references |Reference-overlap continuation veto for ranking these cells |Off-reference support |Fresh parameter-specific adapted TT fits |Failure of LEDH or conditional proposals |

## Inference status
| Item | Status |
|---|---|
| Hard veto screen |All1584 filter rows valid;13/33 reference cells fail overlap; broad accuracy promotion vetoed by simpler controls |
| Statistically supported ranking |Only declared cell-specific exploratory results; CW/original has no ranking meeting combined criterion |
| Descriptive-only differences |Large CW/original score reductions; most CW/uniform contrasts; ESS and runtime |
| Default-readiness |Owner-selected engineering default implemented; untuned FP64 experiment does not satisfy canonical/per-scope/TF32/HMC admission |
| Next evidence needed |Fresh wide-parameter references; more paired streams; independent datasets; per-scope tuning and production-precision checks |

## Post-run red-team review
The strongest alternative explanation for apparent accuracy is shared bias in
the two saved TT references; their conditional jackknives do not measure support
or fitted-proposal approximation bias. A fresh, well-overlapping higher-quality
reference could change the score ranking. The weakest stochastic evidence is
eight streams for heavy-tailed squared errors. Parent weighting improves weight
balance while changing ancestry and sensitivity propagation; its score effect
is genuinely parameter-dependent in this diagnostic. No tuning on these final
measurements occurred. The observed failures reject broad accuracy promotion
of the current untuned candidate, not the research direction. Fresh references
and increased replication are the discriminating next experiments.

## Execution and budget
Main campaign completed in4117.411160970805s; reference01 in1235.4757100089919s;
reference-local01 in763.2006204379722s; pilot02 in226.18919918499887s. Total
successful numerical time6342.276690602768s. Tests and conservatively charged
startup/review attempts add361.27s, giving6703.546690602768s before reporting
against the7200s cap. Reporting uses the reserved allowance; final accounting
is in budget-final.json. No N2016 extension or fresh TT training was launched.
Commands, environment, GPU growth verification and sources are in the four run
manifests. Main campaign status is complete; final six scientific source hashes
match their launch values. Peak TensorFlow allocator use was79876864bytes.

Figures and tables are in analysis01; the manuscript includes the conditioning
identity, correct full-mixture denominator, ESS mechanism, earlier stress result,
expanded protocol, nominal table, conditional comparison plots and limitations.
The current paper PDF is docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf.

## Document and delivery checks
The manuscript adds Sections 16.8–16.11 and updates the subsequent research
discussion to reflect the completed comparison. It preserves the earlier
equations and results, distinguishes the exceptional ESS stress case from the
new parameter grid, and states both the approximate-reference limitation and
the simpler-proposal accuracy failures. The new tables and figures were
visually inspected in the rendered PDF. Existing pdflatex was used after
latexmk was found unavailable; no package installation was needed. Build
commands, exit codes and log paths are in analysis01/document-build.json.

The numerical plan received a skeptical local audit before execution and a
post-run local review. Independent Claude review was unavailable because login
was required; it is not represented as completed. Final source and reporting
snapshots, document hashes and the unchanged scientific-source check are in
delivery-final/manifest.json. The budget ledger charges the entire reserved
300-second reporting allowance conservatively, for a total charge of
7003.55 seconds against 7200 seconds; that allowance is not a measured runtime.
No further numerical experiment or fresh reference training was launched.

## All-case ESS follow-up
A deterministic re-read of campaign01 and pilot02 confirms that weighted
conditional parents have higher time-mean and time-minimum ESS than uniform
conditional parents in every saved paired run: 160/160 pairs across the twenty
reference-usable cells, and 264/264 across all thirty-three cells. Each cell has
eight seeds. The seed-averaged time-mean ESS increase ranges from 6.6793% to
102.7241% among the twenty reference-usable cells. This comparison does not
require a usable Zhao-Cui reference. It does not establish every-time-step
dominance, superiority on future draws, or score-accuracy improvement.

The saved audit is delivery-ess-audit01/ess-pair-audit.json under the campaign
artifact root; it records exact source hashes, per-pair values and case means.
Section 16.11 now states these counts explicitly. This is reporting from
existing measurements, with no new numerical experiment. The earlier
delivery-final snapshot remains unchanged; the follow-up document build and
hashes are recorded separately in delivery-ess-audit01.

