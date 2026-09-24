# Remaining TT regression repair work

Follow-up: the code and frozen-target diagnosis in [the first-transition result](observation-tt-first-transition-root-cause-20260914-result.md) resolves the ambiguity in item 4. The target already is joint; separate block charts are mathematically valid. Grouped rank-three separation and a large remaining fitting gap are now supported by direct diagnostics. A coupled whitening map is not established as the required repair.

2026-09-14. This is a read-only numerical-artifact assessment, with documentation updates only; no new fit or filtering experiment was run. The selected SGQF–TT filtering lifecycle has been implemented and executed. That does not establish that the TT regression has been repaired. Original-model particle importance correction can produce acceptable filtering estimates while the fitted proposal remains inaccurate.

## Direct evidence from the executed fits

Source: `docs/benchmarks/artifacts/observation_aware_tt_complete_20260913/campaign-01/d4/tt_guided_fit_t*.json`. The fit error is the empirical relative RMS error of the square-root target on independent audit rows sampled from the standard Gaussian reference. It is not a posterior error, a population L2 certificate, or an importance-weight variance bound.

| Quantity | Observed value |
|---|---:|
| Initialization, t=0 audit relative RMS | 0.0303194921 |
| First transition, t=1 audit relative RMS | 0.4745265639 |
| Median over all 20 fitted steps | 0.4404874650 |
| Maximum over all 20 fitted steps | 0.4878598535 |
| t=1 training relative RMS | 0.4093122400 |
| t=1 validation relative RMS | 0.4666047536 |
| Steps selecting the largest tested L1 penalty, 0.001 | 18 of 20 |

The first transition changes the fit from four to eight continuous axes. The jump is a localization clue, not proof that rank, optimization, row coverage, coordinate dependence, or inherited retained-density error is the cause. Substantial training error also prevents attributing the problem solely to out-of-sample generalization. Selecting the L1 grid endpoint does not prove that a larger penalty would help.

The launch manifest fixes rank 3, degree 3, 1024 rows per split and four sweeps. `compiled_fitter` uses 128 proximal iterations per core and checks objective nonincrease; it does not record a stationarity certificate, conditioning spectrum or convergence ladder. Particle seeds vary, but the fitted TT uses one construction seed per time/method/dimension, so the four particle replications do not replicate fitting uncertainty.

## Missing experiments and implementation capabilities

1. **Separate local fit error from inherited error.** Freeze one transition's incoming retained density, charts and target. Check the scaled polynomial mass and density against independent integration of that same target before mixing or particle correction. Separately compare the incoming retained density with a filtering reference. The present particle-reference screen combines these effects and cannot identify the regression defect. The recorded t=1 failure may be used for diagnostic localization; it must not become a new untouched validation claim.

2. **Add and evaluate importance sampling of regression rows.** `fit_amplitude` currently draws every training/validation/audit design from the standard Gaussian reference r. It correctly fits the pulled-back target divided by r, but has no separate mixture row sampler s or r/s loss weighting. For amplitude a(z)=sqrt(gamma_pull(z)/r(z)), preservation of the original objective requires

   E_s[(r(z)/s(z)) (P(z)-a(z))^2] = integral (P-a)^2 r dz.

   The equality follows by cancelling s inside the integral, assuming s is positive wherever the integrand has mass. A defensive mixture can deliberately cover tails and adjacent-state dependence. Its complete density must enter the loss, and normalization/validation must use the matching measure. Changing the target to sqrt(gamma_pull/s) would define a different representation and requires a separate derivation. Final particle importance weights do not implement this training-row repair.

3. **Establish convergence and capacity.** On fixed local targets, record per-sweep objectives, proximal stationarity and design conditioning; then vary solver budget, rank, degree, row count and fitting seeds in a controlled order. Keep L1 selection on disjoint calibration/validation data. More capacity is a hypothesis, not an automatic repair, and objective nonincrease alone does not prove convergence.

4. **Test whether the coordinates leave a difficult adjacent-state dependence.** Current charts whiten current and previous states separately; they do not decorrelate the two-time joint. A conditional triangular affine map or an alternative ordering may help if the frozen-target diagnostic supports this explanation. Any map/order change must preserve the conditional sampler and retained-marginal calculation. CUT4, nonlinear maps and full higher-moment propagation are possible alternatives, not prerequisites established by current evidence.

5. **Demonstrate downstream benefit on fresh data.** Repeat fitting and filtering on independent observation sequences and construction seeds with separate calibration and untouched evaluation. Check local mass/density accuracy, conditional weight concentration and filtering/evidence accuracy against the constructed simple proposals, with uncertainty. The current four particle seeds on one sequence establish neither comparative benefit nor higher-dimensional scaling. The guided TT's conditional losses remain a promotion veto, not a rejection of the research direction.

## Decision and next action

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the completed lifecycle; regression repair remains unresolved | Exploratory particle-reference screens passed; local regression repair not established | Conditional heuristic losses block promotion | Optimization versus capacity, coverage, coordinates and inherited error | A frozen one-step localization diagnostic, followed by correctly weighted row-design and convergence comparisons on fresh calibration data | Accurate retained TT, reliable direct TT normalizer, comparative superiority, HMC or high-dimensional readiness |

Inference status: no new stochastic comparison was run. Values above are descriptive summaries of existing fit records. Hard implementation checks remain passed; no statistically supported ranking exists; default readiness remains false. MathDevMCP's full proof audit is still incomplete and is separate from the observed regression error.

Existing campaign usage remains one full launch, 99.574 of 2400 numerical seconds. This assessment does not launch or silently enlarge the campaign. A follow-on comparison needs its own frozen targets, calibration/evaluation split, evidence contract and budget within the user's authorized research scope.
