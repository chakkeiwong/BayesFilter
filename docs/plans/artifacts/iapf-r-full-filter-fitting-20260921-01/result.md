# Full-filter fitting and separate controller comparison

Fitting stage and the first controller dataset are complete. This independent
CPU R campaign executes the next steps already recorded in the master under the
owner's continuation request. The [fitting plan](../../iapf-r-full-filter-fitting-comparison-2026-09-21.md)
and [controller plan](../../iapf-r-controller-window-comparison-2026-09-21.md)
were reviewed before their respective experiments. Both share one 2400-second,
16-launch allocation; the previous campaign's unused balance is untouched.

## The constrained fit fails the fresh full-filter screen

All 32 constrained validation replicas completed, with converged fits and passing
Gaussian-limit tail checks. Nevertheless, the frozen nominal box fails the
predeclared practical-reference criterion on both fresh d20 datasets. QR passes
that criterion on both. This is a numerical-method result, not an optimizer
crash or an invalid likelihood identity. None of these cells reproduces the
separate literal paper SD/resampling pattern.

| Fresh dataset | Fit | Replicas | Mean Zhat/Z (95% bootstrap interval) | SD (95% interval) | Mean final N | Mean total seconds |
|---|---|---:|---|---|---:|---:|
| 89200020 | Constrained Eq15, box1 | 16 | .9593 [.7526,1.2290] | .5010 [.1587,.6809] | 1687.5 | 23.00 |
| 89200020 | QR | 16 | .9706 [.9141,1.0288] | .1203 [.0736,.1501] | 1000 | 4.32 |
| 89300020 | Constrained Eq15, box1 | 16 | 1.0283 [.8157,1.3005] | .5229 [.2393,.7512] | 1687.5 | 34.87 |
| 89300020 | QR | 16 | .9743 [.9258,1.0203] | .0997 [.0571,.1308] | 1000 | 5.61 |

The practical bounds were the mean interval within [.8,1.2], SD upper interval
<=.38 and mean N<=1500. Both constrained cells fail all three conditions. Their
mean intervals include1; these results do not establish bias. The much larger
observed SDs and runtimes are descriptive comparisons, not proof of population
variance ordering. Timing includes learning and the independent final filter;
it was measured with up to two CPU workers and is not a dedicated speed benchmark.

The cheap heuristic comparison supplies an additional conservative veto: the
constrained method has larger observed prefix MSE than FA-APF in both ordinary
and large-innovation situations on both datasets. All four paired difference
intervals include zero, so statistically established inferiority to FA is not
claimed. QR clears the observed conditional heuristic screen on these two cells.
All comparisons, intervals and the BPF/FA/SIS rows are in
[fitting-analysis/summary.json](fitting-analysis/summary.json). Their particle
counts are deliberately different, so these are not equal-cost comparisons.

Calibration used four replicas per arm at each of d5/d20. The widest box2 had
one tail-quality failure, at d20 replica2204/time47, relative margin-.020529.
Box.5 and box1 passed the numerical checks in both dimensions. The rule fixed
before calibration nominated box1, then .5, then2; it did not select by an
attractive sample mean. The narrower box's d20 mean was .6460 in its four runs.
An active constraint was explicitly allowed and reported. Rejecting box2's
Gaussian-limit check does not prove infinite variance for the actual positive-
floor guide. Calibration tables are preserved in calibration-analysis/.

## Saved guides explain the density-fitting concern

The shape diagnostic checked all64 saved validation guides,100 time points each,
against the exact future-likelihood Gaussian. Its identity/nonnegativity checks
and exact terminal observation-guide check pass. For Gaussian components,

KL(N(m*,V*) || N(m,V)) = .5 [log(det V/det V*) - d + tr(V^-1 V*)
                           + (m-m*)' V^-1 (m-m*)].

The covariance and mean terms isolate shape spreading from displaced centers.
The diagnostic excludes the positive floor; it is not a KL between the actual
mixture proposals and does not substitute for the likelihood comparison.

| Dataset | Fit | Mean component KL | Mean term | Covariance term | Mean log determinant inflation |
|---|---|---:|---:|---:|---:|
| 89200020 | Box1 | 2.2479 | 1.2097 | 1.0382 | 8.4106 |
| 89200020 | QR | .1850 | .0805 | .1045 | -.4721 |
| 89300020 | Box1 | 2.2248 | 1.2024 | 1.0225 | 8.3337 |
| 89300020 | QR | .1876 | .0832 | .1044 | -.4689 |

These descriptive differences are consistent with the previously derived
density-scale escape: minimizing sum(p_i-lambda*b_i)^2 can reduce the loss by
spreading p and lowering its density, even when its shape becomes less useful.
Compact bounds make the fit exist but do not make that objective identify the
best guide. Here both center error and covariance spreading remain appreciable
inside the constraints. This does not prove that every possible constrained
optimizer fails, or that the original authors used these settings.

The raw6400 rows and64-guide coverage record are in attempt13-saved-guide-shapes/;
guide-shape-summary.json retains the aggregates. No validation guide was used to
select new bounds or change any fitting setting.

## Controller follow-through

The independent extension changes the stopping-history window from k+1 to k,
with QR/floor8 fixed. It retains the earliest eligible iteration, doubling
window, threshold and fresh final filter. A deterministic test verifies the
default decision is identical to explicit window6, the window5 change acts at
the intended point, and both routes make an additional final filter call.
Consumer wiring and the existing12 regressions pass. The fitting workers used
their original captured core; the optional controller change is captured in a
separate source snapshot and does not alter those results.

The first fresh d80 dataset (89400080) completed all eight paired replicas and
all three heuristic baselines: 40 complete method/replica pairs, 10400 fit rows,
1600 tail rows and 4000 prefix records. All numerical and guide-quality checks
pass. The second fixed dataset (89500080) is unrun: the remaining 45.097347
worker seconds cannot cover its complete reservation. It remains required
evidence and is not waived because of the first result.

| Stopping window | Mean Zhat/Z (95% interval) | SD (95% interval) | Mean N | Mean total seconds | Practical screen |
|---|---|---|---:|---:|---|
| Six estimates, unchanged comparator | .9956 [.8766,1.1137] | .1824 [.0925,.2289] | 2000 | 29.00 | Fails N<=1713 |
| Five estimates, explicit extension | 1.0447 [.8603,1.2194] | .2846 [.1395,.3548] | 1000 | 22.32 | Fails mean-interval upper bound<=1.20 |

The shorter window used 1000 fewer particles in every pair. The paired mean
runtime difference is -6.675 seconds, bootstrap interval [-7.627,-5.879].
These intervals describe this dataset and machine; eight identical particle
differences do not establish that every future dataset will behave the same.
The paired terminal squared-error difference is .04374, interval
[-.00291,.09599], so the terminal error ordering is unresolved. Ordinary-innovation
prefix MSE is higher under the shorter window by .2002, pointwise interval
[.0493,.3742]; the large-innovation difference is .0968, interval
[-.0221,.2311]. This is evidence of an accuracy/cost tradeoff, not permission to
promote a method that fails its primary screen. The intervals are pointwise,
with no simultaneous multiplicity correction or cross-dataset inference.

Both controller arms clear the observed conditional heuristic screen in both
situations. All fixed-N BPF and SIS terminal ratios underflow when exponentiated:
their finite log ratios lie in [-2734.87,-2416.47] and [-16632.85,-13380.26].
Their displayed ratio means/SDs of zero are therefore not exact zero likelihoods
or zero theoretical variances. The retained log values show severe particle
degeneracy, and the prefix MSE calculation remains meaningful to machine
precision. FA's observed terminal mean ratio is .308; none of these eight-run
baseline samples establishes the baseline's mathematical bias. Baseline particle
counts differ and this is not a matched-cost comparison.

Attempt14 reached its 400-second cap after 26 complete pairs. Attempt15 used
175.279533 seconds to execute only the 14 missing pairs. Each method resets its
recorded seed; algorithm sources and regenerated observations/Kalman values were
unchanged. Complete-pair filtering, unique 40-pair identity and all diagnostic
counts passed before the derived cell was assembled. Both attempts, interrupted
work, source closures and the merge provenance are retained and charged. The
repair introduced no statistical selection and did not rerun an unfavorable pair.
The skip-path smoke passed before the repair. The terminal reporter subsequently
added explicit underflow counts and a machine-readable heuristic verdict; this
reporting repair does not change any experiment or interval.

Authoritative combined tables are in
[terminal-analysis-v2/summary.json](terminal-analysis-v2/summary.json) and
[tables.md](terminal-analysis-v2/tables.md). The earlier reports are preserved;
v2 adds underflow interpretation and heuristic verdicts. The frozen calibration
decision's validation_opened=false records its pre-validation timestamp, not
the campaign's terminal state.

## Decisions and inference status

| Decision | Primary criterion | Vetoes | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject box1 as the current practical reference | Fails both fresh d20 cells | Conditional heuristic vetoes; no numerical crash | Small-sample tails and unknown author choices | Preserve failure; diagnose shape; do not promote or scale this arm | All Eq15 optimizers or iAPF fail |
| Keep QR available for independent reference work | Passes both fresh d20 cells | No heuristic/tail veto here | Original objective differs; higher-dimensional controller issue remains | Preserve as fixed comparator for second d80 dataset | Exact paper replication or production readiness |
| Exclude widest tested box2 | Calibration feasibility veto | One Gaussian-limit tail-quality failure | Actual positive-floor variance not determined by this test | Preserve bound-sensitivity evidence | Infinite variance under the finite floor |
| Do not promote either controller arm at d80 | Six-estimate arm fails particle limit; five-estimate arm fails mean-interval limit | No fit/tail/heuristic veto on completed first dataset | Only eight pairs on one dataset; second dataset unrun | Execute fixed second dataset under a renewed adequate allocation | General failure of iAPF or proof of the extension's bias |

| Inference status | Finding |
|---|---|
| Hard veto screen | Box2 guide-quality veto; box1 fresh practical-screen failures; both d80 controller arms miss a primary condition; all completed validation/controller fits finite/converged |
| Statistically supported ranking | No general method ranking. Pointwise d80 intervals support reduced runtime/particles and increased ordinary-prefix error for the shorter window on this dataset; terminal-error interval spans zero |
| Descriptive-only differences | Fitting SD/runtime/N and guide-component KL differences; d80 SD differences and cross-dataset/general rankings |
| Default readiness | No default change; independent R reference only |
| Next evidence needed | Fixed second d80 dataset with enough compute, then reference decision; missing author choices/data, paper-scale replication and multivariate R/TF parity remain |

Engineering correctness: checked core identities and consumer wiring, plus12
regressions. Numerical validity: all fitting validation rows are complete,
source/data identities checked, and every validation tail row passes. Scientific
interpretation: the constrained reconstruction fails its declared practical
screen, while original-paper identity and literal replication remain open.

Post-run red team: a different constraint geometry, scale-invariant objective,
or unpublished author solver could behave differently. The strongest alternative
explanation for the observed variability is small-sample tail coverage; the two
fresh datasets and component-shape mismatch weaken, but do not eliminate, that
explanation. A frozen independently validated fitting rule with adequate tail
evidence could overturn this candidate-specific rejection. No tuning on the
failed validation datasets is authorized by these results.

The controller's strongest alternative explanation is that the first dataset
underrepresents the rare likelihood tail: reducing N may save work while making
that tail harder to observe. Its mean interval includes one, so bias is not
established. The independent second dataset and larger frozen replication are
needed to assess that tradeoff. Do not tune a new window on these results.

Run provenance, actual commands, seeds, R/CPU environment, source snapshots,
wall times and per-attempt checks are preserved in manifest.json. Full logs
remain beside each attempt. Total charged numerical work is 2354.902653/2400
summed worker seconds in 15/16 launches, leaving 45.097347 seconds and one launch.
The previous campaign's unused balance was not transferred. No worker remains.
All 30 captured source hashes and 24 merge-input hashes check, as do data, seed,
coverage and budget accounting. The 12 regression tests, direct oracle/identity
checks, controller wiring/default/fresh-final-call checks, skip smoke and
git diff --check pass. Terminal review was by the executor with executable
checks; no independent reviewer was launched or implied.

Next exact action: retain the frozen controller comparison and run dataset
89500080, IDs2601--2608, under an allocation sufficient for all 40 method pairs.
Measured first-cell method time is about550 seconds; allow roughly700 worker
seconds plus checkpoint overhead and split into resumable method/replica units.
That new allocation is not spent or assumed here. The current budget boundary
prevents the next full cell; the candidate failures do not reject the research
direction. Neither literal paper replication nor LEDH/KDM/GPU/HMC correctness
has been established by this independent R campaign.
