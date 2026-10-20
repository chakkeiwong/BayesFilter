# Skeptical review of the SIR tuning plan (2026-10-06)

Reviewed plan: docs/plans/ledh-sir-no-oracle-tuning-20261006.md

## Review questions

| audit question | finding | repair or limit |
|---|---|---|
| Is the baseline the actual comparator? | Yes. It is the current shared marginal-mixture evaluator at the same scope, data, design, route, dtype, and particle count. | The ancestor route remains diagnostic only. |
| Is a proxy being used as correctness? | The plan explicitly prevents ESS, CV, cap activity, and reset norms from being promotion criteria by themselves. | The conditional SIR predictive calculation and repeated value/score sweeps are required. |
| Is there an exact oracle? | No full filtering oracle is available. The one-step h diagnostic is exact only conditional on the realized pre/post clouds. | The result must not be described as exact SIR accuracy. |
| Are calibration and validation separated? | Yes. Candidate selection uses calibration; validation and confirmation observation and design seeds are held out. | Each partition has one observation dataset per horizon, shared across candidates for paired comparisons. Generalization across a population of datasets is untested. |
| Is horizon transfer hidden? | No. Each horizon has its own scope and report. | No setting is promoted across horizons without a new scope artifact. |
| Are safety changes treated correctly? | The actual guarded baseline keeps moment_safety enabled for every arm. | Its possible distortion is measured by conditional reset errors and cap diagnostics; successful completion alone does not establish non-harm. |
| Can the plan pass while harming protected models? | This is a material risk if shared controls or code are changed. | The SIR artifact is isolated and LGSSM/KSC/predator-prey value and score tests are a hard veto. |
| Is the score target the claimed score? | The candidate calls the shared analytical recursive score. | Existing finite-difference/marginal-weight tests must pass before campaign results are interpreted. |
| Is the compute bounded? | Candidate grid, seeds, horizons, and particle count are finite. | Stop on invalidity, missing artifacts, regression, or budget exhaustion. |
| Is the selection objective fair? | No mean likelihood is optimized. Value variance, each score-coordinate variance, and conditional reset error are screened separately. | Four calibration, four validation and eight confirmation designs provide limited uncertainty evidence; two paired heuristic seeds cannot support a ranking. |
| Could the implementation fork the numerical path? | The planned trace change only exposes arrays already produced by the canonical finite program. | Call-chain regression and exact trace-field checks are required. |
| Is there a stale baseline or historical result? | The baseline is the current dated campaign configuration, not pre-invalidation LEDH results. | Historical artifacts remain excluded from evidence. |

## Verdict

**PASS WITH LIMITS.** The plan is executable and answers a narrower question
than exact SIR correctness: whether a bounded, scope-specific control choice is
non-deteriorating and descriptively less variable under paired repeated
designs. The main unresolved limitation is statistical power and dependence on
one generated data set per partition and horizon. That limitation blocks a superiority,
default-readiness, HMC, or broad scientific claim but does not block the
planned diagnostic campaign.

## Final implementation audit amendment (controlling)

The first draft was not yet executable as reviewed: it used pairwise_steps=0
and safety off while the actual baseline is guarded_pairwise; it reused one
observation dataset across partitions; and a displayed reset-score identity
omitted D Z_mu. These were corrected before the serious run. The runner now
imports the exact arm settings, uses three observation datasets, retains all
three analytical reset derivatives, and checks the public endpoint. There are
four calibration, four validation, and eight confirmation design seeds.

The six candidates and eight-hour budget are explicit. Three heuristic
adversaries are evaluated on paired confirmation seeds and cannot influence
selection. The existing moment-safety cap stays enabled for every arm. Candidate
rejection is distinguished from experiment invalidity. Actual values, scores,
failed screens, and intervals must accompany conclusions. The main unresolved
issue remains absence of a full SIR oracle: lower variance and smaller local
reset error do not prove lower total error. Author self-review is recorded;
no independent peer-review claim is made. PASS for bounded diagnostic execution.
