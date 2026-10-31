# Terminal particle-count follow-up — 18 October 2026

## Finding

**Both product consumers still lose descriptively to the unguided-global and
Gaussian-future-global heuristics at 128 particles. Neither is promoted.**
Larger clouds reduce their conditional score-coordinate-1 squared error, but
that finding does not establish a benefit from backward guidance over the
strong simple comparators, count convergence, or equal-compute superiority.

The final launch under
[the follow-up plan](../../ledh-particle-count-followup-20261018.md) completes
the three-attempt campaign. The independent terminal analyzer reports `PASS`:
1,536 rows, 9,216 trace checks, eight saved fits, 16 finite-difference checks
with maximum error `4.505907469365411e-08`, 73 source hashes and 18 input hashes.
The numerical reference and coverage checks pass. No numerical continuation
veto fired. The attempt limit, not a failed candidate, closes GPU execution.

## Conditional comparison

This is the KSC seven-mixture positive-future-tail diagnostic, with observations
`(0.2, 5)`, parameters `(1.5, 0)`, and horizon two. Eight previously saved fits
are held fixed. Each receives twelve fresh nested clouds with seeds
`862800 + 100*cluster + replicate`; the 64-particle cloud is the prefix of its
128-particle counterpart. Fit seeds are `822000` through `822007`. Chunks equal
particle counts in both arms. The preceding 32/64 study used different clouds:
its 64-particle observations must not be joined to this table as one nested
32/64/128 cohort.

The computed score is the analytical recursive derivative of the finite,
frozen-guide likelihood program. Finite differences check that derivative;
they do not establish equality with the model score. The independent numerical
reference gives log likelihood `-7.684108174460466` and model score
`[0.5404131211495837, 2.5653202234510557]`. Its converged grid comparison differs
by `1.7763568394002505e-15` in value and `4.440892098500626e-16` in score.

| Arm | Coordinate-1 MSE, N64 | Coordinate-1 MSE, N128 |
| --- | ---: | ---: |
| Unguided identity | 5.41760625 | 1.59984795 |
| Unguided global | 0.30749211 | 0.08088601 |
| Unguided local | 0.71537934 | 0.22007841 |
| Unguided equal mixture | 1.00631240 | 0.16220164 |
| Gaussian-future global | 0.24477466 | 0.08158302 |
| Gaussian-future equal mixture | 0.65554908 | 0.12238736 |
| Fitted-product global | 0.30973632 | 0.09867255 |
| Exact-product global | 0.31761546 | 0.10000959 |

The six simple arms were constructed before the run: identity removes flow;
global and local isolate the two flow choices; equal mixture removes mixture
selection; the two Gaussian-future arms supply simple lookahead with and
without selection. The artifact evaluates both coordinates at each count,
conditional on this tail situation. At N128, the four descriptive losses are
the two product arms against the two global heuristics in coordinate 1.
The heuristic screen therefore fails as a promotion veto only.

The fourteen predeclared contrasts use within-fit means of twelve paired cloud
differences, paired t intervals with seven degrees of freedom, and Bonferroni
95% simultaneous coverage. Negative differences favor the candidate. These
intervals assume independent groups and adequate small-cluster t calibration;
they are conditional on the saved fits and fixed observations.

| N128 candidate | Comparator | MSE difference | Simultaneous interval | Status |
| --- | --- | ---: | --- | --- |
| Fitted product | Same arm, N64 | -0.211064 | [-0.404694, -0.017433] | Conditional gain |
| Fitted product | Identity, N128 | -1.501175 | [-3.191743, 0.189392] | Unresolved |
| Fitted product | Global, N128 | 0.017787 | [-0.036774, 0.072347] | Unresolved |
| Fitted product | Local, N128 | -0.121406 | [-0.224768, -0.018044] | Conditional gain |
| Fitted product | Equal, N128 | -0.063529 | [-0.164298, 0.037239] | Unresolved |
| Fitted product | Gaussian global, N128 | 0.017090 | [-0.017362, 0.051541] | Unresolved |
| Fitted product | Gaussian equal, N128 | -0.023715 | [-0.106816, 0.059386] | Unresolved |
| Exact product | Same arm, N64 | -0.217606 | [-0.423695, -0.011517] | Conditional gain |
| Exact product | Identity, N128 | -1.499838 | [-3.182822, 0.183145] | Unresolved |
| Exact product | Global, N128 | 0.019124 | [-0.032900, 0.071147] | Unresolved |
| Exact product | Local, N128 | -0.120069 | [-0.218607, -0.021530] | Conditional gain |
| Exact product | Equal, N128 | -0.062192 | [-0.165324, 0.040940] | Unresolved |
| Exact product | Gaussian global, N128 | 0.018427 | [-0.025518, 0.062371] | Unresolved |
| Exact product | Gaussian equal, N128 | -0.022378 | [-0.106990, 0.062234] | Unresolved |

Thus four intervals favor the stated conditional changes, none establishes
harm, and ten are unresolved. No ranking over either strong global comparator
is supported. At N128, fitted and exact product coordinate-1 bias/variance are
`0.13376411 / 0.08077971` and `0.12900917 / 0.08336623`. Those decompositions
are explanatory, not an asymptotic-bias estimate or a tuning objective.

## Decision and inference status

| Item | Status |
| --- | --- |
| Decision | Reject promotion of both tested product candidates; retain guidance as a research direction. |
| Primary criterion | Four conditional gains: count increases and comparisons with unguided local. No established gain over the strong global heuristics. |
| Veto diagnostic status | Numerical, derivative, reference, coverage and provenance checks pass; four descriptive heuristic losses veto promotion only. |
| Main uncertainty | Fixed fits and observations, eight groups, small-cluster interval calibration, and unequal compute across particle counts. |
| Next justified action | Isolate the current-observation Gaussian/map approximation from the moment-restoring reset in a newly budgeted diagnostic. |
| Not concluded | No general improvement, count convergence, canonical LEDH conformance, scope tuning, default readiness or HMC readiness. |

| Inference class | Status |
| --- | --- |
| Hard veto screen | No numerical continuation veto; both candidates remain numerically viable. |
| Statistically supported ranking | Only the four stated conditional contrasts; none over the strong global comparators. |
| Descriptive-only differences | All other observed MSE, bias, variance and runtime differences; heuristic losses do not establish statistical harm. |
| Default-readiness | Not established. This is an FP64 GPU/XLA diagnostic exception, not the TF32 canonical route. |
| Next evidence needed | A controlled map/reset attribution, then fresh fits and datasets and an equal-compute comparison before broader claims. |

The exact future message retains error under the same consumer, so fitting
error alone cannot explain all residual error. This does not identify the
current-observation approximation, reset, flow discretization or finite-cloud
effects separately. The next repair must discriminate those explanations,
rather than enlarge fitting capacity by assumption. Existing exact-message
and strong global arms remain necessary comparators; raw-barycentric or
reset-disabled controls, if needed, would be historical/mechanism diagnostics
only, never canonical Contract E alternatives.

Post-run red team: the strongest alternative explanation for the conditional
gains is finite-cloud variation under this single fixed tail example. Two
counts do not establish a limit, and more particles cost more. Fresh-fit and
new-data uncertainty could overturn any broader ranking. A controlled
map/reset comparison that removes the exact-message residual would overturn
the current inability to attribute it. The weakest evidence is generalization,
which this study does not evaluate.

## Reproducibility, budget and refresh

The [run manifest](attempt-01/manifest.json) records commit
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, branch
`ledh-refactor-with-policy-fix`, dirty-source hashes, environment `tftwogpu`,
TensorFlow `2.20.0-dev0+selfbuilt`, data, seeds, commands and plan identity.
GPU execution uses FP64, XLA enabled and TF32 disabled. Memory growth is
verified before initialization on the RTX 4080 SUPER and RTX 5080. Allocator
current/peak bytes are `46080 / 8444672`; these are not whole-device reservation
or capacity claims. CPU reference and analysis deliberately hide GPUs.

Untuned diagnostic controls remain epsilon one, five Sinkhorn iterations,
16 flow steps, two correction steps, basis four, rate `0.1`, radius `0.25`.
Holding these fixed isolates this comparison; it does not make them tuned
defaults or confer canonical admission.

Commands run from the LEDH worktree:

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_particle_count_stress_diagnostic.py --followup --budget-seconds 500 --output docs/plans/artifacts/ledh-particle-count-followup-20261018/attempt-01
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/analyze_ledh_particle_count_stress_diagnostic.py --run docs/plans/artifacts/ledh-particle-count-followup-20261018/attempt-01
```

The GPU command uses trusted/escalated device access. GPU-process wall time is
`248.20482324197656` seconds; CPU reference and terminal analysis take
`0.5198429790325463` and `0.6317310389713384` seconds. Cumulative campaign use is
**894.3676919049467/1200 GPU-process seconds, 3/3 launches, and
4.615978229005123/180 CPU reference/analysis seconds**. The launch budget is
exhausted; the unused time cannot authorize a fourth launch.

The terminal [analysis](attempt-01/analysis.json), raw result, manifests and
logs preserve the evidence. Pre-refresh drivers are retained under
`source-before-refresh/`; the prior checkpoint is archived beside this note.
No numerical kernel or audited driver is changed during the terminal document
refresh. The revised manuscript builds without warnings (75 to 76 pages), and
rendered pages 69--71 pass local inspection. Exact build commands, hashes,
timings and the review are preserved under
[document-build/review.md](document-build/review.md). The masters and concise
checkpoint are refreshed by the agent after audit; this is not an unattended
service. Phase completion and candidate rejection do not require another
continuation request; resource and validity boundaries do.
