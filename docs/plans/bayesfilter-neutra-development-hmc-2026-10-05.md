# Development validation of the fitted NAF maps

This continues the authorized source-and-fit study within its additional 24
GPU-process / 24 CPU-core hours. It changes no runtime default. At specification
time two two-mode final NAF maps have passed the existing coarse fit screen;
the remaining four fits are unobserved. All six prescribed fits must finish
and their numerical/source checks pass before this phase consumes their results.
The exact targets are the two development specifications, never the untouched
randomized final targets. Native-teacher transfer and final generalization remain
separate phases.

## Question, evidence and decisions

Can a frozen, exact-teacher NAF map that passes the preliminary distribution
screen support useful exact-target HMC in its latent coordinates? The primary
criterion is sequential retained physical-space convergence, information and
precision, followed by agreement with a new independent exact reference.
No density or 1,000-point score diagnostic substitutes for that result.

The numerical control is the same public tuning/retained call chain on a standard
two-dimensional Gaussian with an exactly identity configured IAF. Independent
exact iid samples check the feature/reference assessment. These are engineering
controls, not a tuned classical multimodal comparator or a method ranking.
Existing IAF fitting results and the original-author reference remain visible;
this phase asks whether the optional map is usable, not whether NAF is superior.

| Role | Quantity or condition |
|---|---|
| Promotion criterion within development scope | Shared sequential HMC checks plus fresh physical-reference agreement |
| Promotion veto | Failed fit screen, invalid inverse/density/score, nonfinite state/target/log acceptance, movement failure, failed R-hat/ESS/precision/reference check |
| Continuation veto | Broken common control, corrupt/mismatched checkpoint, unsupported public consumer, exhausted allocation |
| Repair trigger | No verified kernel in the funded search, warm-up or retained cap, candidate-only shape/geometry failure |
| Explanation only | KL, acceptance, finite extreme log-acceptance alerts, score-residual quantiles, runtime, coordinate stretching |
| Not concluded | Default readiness, algorithm superiority, full posterior correctness beyond checked quantities, teacher transfer, final-target or q20 transfer |

Every final NAF endpoint passing the existing fit screen is eligible, in target
then seed order (11, 37, 73). No descriptive metric ranks these maps. Ineligible
maps retain explicit failed-screen records. Every eligible map receives its own
kernel tuning scope; no epsilon or tuning artifact transfers across maps.
For each map, consider at most three independently verified kernels in declared
trajectory-diverse order, inherited from the existing qualification consumer.
Select the first that passes the sequential screen, then open the fresh reference
once. A failed reference does not trigger selection of another kernel using
that same reference. This is development evidence, not untouched confirmation.

## Numerical protocol and provenance

Use `tune_fixed_transport_hmc_kernel`, as required by the capability registry,
with fixed identity mass in latent coordinates and the exact transformed density
including its log Jacobian. The same four physical starts, sampled independently
from the exact development target, are inverse-mapped for all maps of that target.
This is an explicitly favorable benchmark initialization control, not evidence
for finding all modes. Generic discovery remains a native-teacher question.
Roundtrip and finite-value/score checks precede tuning.

The leapfrog grid (3, 5, 9, 13, 18, 25), initial epsilon 0.1, maximum epsilon 2,
per-L pilot and one refinement round are inherited tuning hypotheses, not
target-specific calibrated defaults. The public tuner must measure and freshly
verify every funded survivor. Use the public default acceptance policy, 128
measurement and 128 verification transitions, evidence rungs (1, 2, 4), at most
36 candidates and 120 native work units. The draw count doubles the older
64-draw development setting to reduce its short-block noise; this is an explicit
engineering choice with no nominal repeated-look coverage claim. The 120-work
cap extends the public controller's 100-work convenience allocation by its
20-work repair-reserve size to fund six primary pilots plus measurement and
verification with bounded repairs; an
exhausted search remains inconclusive. Pilot/tuning draws never enter estimates.

Use the shared sequential controller unchanged: four batched chains; at least
2,000 warm-up transitions; latest 1,000 window; maximum rank/folded split R-hat
<=1.05 for warm-up; retained R-hat <=1.01; 1,000-transition chunks; at most 10,000
warm-up and 10,000 retained transitions per chain. Archive every warm-up chunk
and exclude it from estimates. Require bulk/tail ESS >=400 and MCSE/SD <=0.03
for physical coordinates and the additional declared quantities, inherited from
the earlier development consumer. The precision ratio corresponds to roughly
1/(0.03^2)=1,111 effective observations for an ordinary mean, and is stricter
than the ESS floor. These are operational diagnostics, not convergence proofs.

Additional quantities are stable component log-odds, responsibility-weighted
standardized first and unique second moments, and radial-tail contributions from
the existing exact evaluator. Omit duplicate cross-products, since xy and yx
are the same statistic. They are continuous quantities, not binary
events. Responsibilities use exact mixture information only inside evaluation;
the learner/tuner see density/score. These features inspect weights, within-mode
shape and tails without demanding observations of essentially impossible valley
events. They do not certify every rare-event probability.

A new 32,768-row exact iid reference per map is generated on CPU, independently
of training, kernel selection and retained streams. Apply the existing operational
agreement rule: absolute discrepancy <=4 combined standard errors +0.03 reference
SD, while retaining the independent MCSE/SD requirement above. This is an
inherited resolution screen, not a simultaneous confidence statement. Exact iid
controls and an intentionally wrong-location control must demonstrate that the
assessment accepts the intended reference and rejects a clear mismatch before
claim-bearing development use. CPU sample generation uses two native TensorFlow
worker threads and records seeds, hashes and placement. GPU training/inference
uses TF/TFP/XLA and verified memory growth. FP64 remains this study's explicit
diagnostic/reference precision; TF32/default promotion is outside this phase.

Streams are prospective convenience labels: physical starts (target index+31000,
6101); reference (fit seed, 6201); tuning (fit seed, 6301); warm-up (fit seed,
6401+10*member ordinal); retained (fit seed, 6501+10*member ordinal), for
zero-based member ordinals 0, 1 and 2. The retained public bridge checks these
against all used tuning streams. Different maps have separate target/map scopes;
their common starts are intentional. Reference values are not consulted during
kernel selection. All arrays, manifests, source hashes and decisions go under
the existing versioned campaign output root in new `development-hmc-*` jobs.

## Pricing, stop conditions and skeptical review

First execute the Gaussian control with a 600-second worker ceiling. Then price
one complete eligible two-mode/seed-11 map with a 1,800-second ceiling, reserving
3,600 CPU-core seconds. These are engineering containment bounds, not sampler
stopping criteria. The previously observed NAF fit cost is not extrapolated to
HMC: inverse, compilation and downstream assessment costs differ. Use a 300-second
Gaussian-control and 900-second fitted-map tuning sub-budget, with the remaining
wall time for sequential checks and a
60-second finalization reserve. Price the remaining full development jobs from
the actual worker and native stage timings: their worker ceiling is the larger
of 1,800 seconds and twice the first worker's measured wall time. The factor
two is an explicitly conservative engineering allowance for target variation,
not a statistical runtime bound. Reserve the complete chosen scope
before launching the rest. Costs and failed attempts remain charged to both
nested campaign ledgers. If a common consumer defect appears, preserve it,
repair and regress it before continuing. A map failure leaves other eligible
maps and the planned geometry repair viable.

Skeptical review: good raw-flow fitting and reliable corrected sampling differ;
the first two passing maps still have large residual-score tails. The new phase
directly measures downstream behavior rather than inferring it from their KL.
It uses the public tuner and canonical sequential controller, with fixed latent
mass, scope-specific verification and fresh-reference isolation. The main
limitations are favorable exact-target starts, a finite candidate budget,
heuristic diagnostic thresholds and the absence of a method-comparison inference
plan. They forbid generic mode-discovery, superiority and final generalization
claims. A Gaussian control cannot prove difficult-target calibration; it only
guards the changed generic-target plumbing. The output must retain failures and
uncertainties rather than hiding them through selection or threshold relaxation.

## Implementation audit and execution

The active call chain is `validate_development_map` -> public
`tune_fixed_transport_hmc_kernel` -> verified retained member ->
`run_sequential_neutra_hmc`. The member reconstructs physical draws before
assessment and archival. Tuning membership uses only density/score and the
frozen map; exact-mixture responsibilities enter posterior evaluation only.
Focused CPU reference tests check scores against finite differences, the
identity control, feature indexing and the independent-reference positive and
negative controls on the Gaussian and both actual development mixtures. These
use separate 16,384-row iid samples and 32,768-row assessment references, with
seeds (target index+31000, 6202/6203); they never open the fresh posterior
reference. This exposes quantity-definition or rare-event assessment failures
even on independent draws before they can be misattributed to a learned map.
The GPU Gaussian control exercises the complete public call
chain. All qualifying routes are registered in the shared route ledger.

The prior fixed/fresh IAF, matched optimizer, unchanged-author and exact iid
controls remain reported separately for two- and three-mode development targets.
They are diagnostic comparators, not a statistically supported ranking. This
phase deliberately asks absolute sampler viability against an exact oracle;
it does not establish comparative efficiency against tuned classical HMC.

The unmeasured worker ceilings and finalization reserve are engineering bounds.
The grid, epsilon bounds, 36-candidate cap and 120-work allocation are inherited
or convenience hypotheses; pilot results and native accounting expose poor
search coverage. They cannot justify rejecting the research direction if the
search is exhausted. The 1e-8 FP64 start roundtrip threshold is inherited from
the current qualification consumer; final inverse validity retains its stricter
configured scalar tests. The fresh-reference size is inherited from the fitting
study and explicitly enters the combined uncertainty estimate. All these
choices are serialized before each run. Numerical and shared-control failures
stop dependent results; candidate convergence failures trigger candidate repair.

Execute using the existing approved wrapper:

```sh
bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh resume
```

The wrapper uses `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, GPU 1,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, two native CPU threads and verified memory
growth. The CPU checks hide all GPUs. A new source snapshot contains the code,
tests and this plan. `development-hmc-result.json`, per-member archives and
fresh-reference reports preserve the result. No active training source snapshot
is modified. Pending separately recorded CPU diagnostics are charged exactly
once into both ledgers after acquiring the shared master lock.

### Pre-run assessment repair from the exact iid controls

The first mixture iid controls failed only tail ESS for responsibilities:
two quantities on the two-mode target and one on the three-mode target.
R-hat, bulk ESS and mean precision passed. In floating point, a near-one
responsibility can become exactly one. If its empirical 95th percentile is one,
the indicator `r <= quantile(.95)` is constant and has undefined ESS. This
rejects correct independent samples, so it is a diagnostic defect, not evidence
against the map. Original reports are preserved in
`development-assessment-debug-r1`; no fitted-map HMC had run.

For component log densities including mixture weights, write a_j(x). We now
compute `ell_j=a_j-logsumexp(a_k, k!=j)` directly and use it for the ordered
responsibility diagnostics. In exact arithmetic, ell_j=log(r_j/(1-r_j)) is
strictly increasing: ranks and quantile-event membership are preserved.
Direct log computation avoids the near-one subtraction and rounding. Folded
R-hat is assessed on ell_j as a declared additional scale diagnostic; nonlinear
transformation need not preserve folded distances. The shared posterior
controller and its R-hat/ESS/MCSE thresholds are unchanged.

Actual responsibilities remain the quantities for mode-mass estimates, MCSE
and exact-reference agreement. Their original mean precision must also satisfy
MCSE/SD <=0.03 during cumulative retained sampling. No probability is clipped
or redefined, no feature is labeled binary incorrectly, and no undefined metric
is silently accepted. Tests must show finite ordered quantities at saturated
responsibilities, agreement of sigmoid(ell_j) with responsibilities where
representable, and positive/negative iid controls for both mixtures before the
HMC phase executes.

The final call-chain inspection also confirms the current public controller's
energy-report semantics: finite extreme log acceptance is an explanatory alert,
not a native divergence diagnosis or hard veto. The inherited -1,000 threshold
labels that alert only. Nonfinite log acceptance, states or targets and failed
movement remain hard failures. The earlier shorthand "energy failure" in the
table was ambiguous; this clarification follows the inspected shared numerical
policy and changes no runtime check. Large finite errors must be reported and
investigated alongside actual posterior diagnostics, without inventing a
divergence count from the historical field name.
