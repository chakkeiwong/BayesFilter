# M22: independent null size and complete-fit defect sensitivity

M16's 128-trial sequential null intervals were too wide for its declared
upper-0.10 screen. Analytic rank experiments showed weak sensitivity to small
posterior shifts. The remaining questions concern precision of the null-size
measurement and defects passed through the entire ordinary HMC procedure.
Repeating only an analytic sampler cannot answer the second question.

## Independent sequential-null continuation

Preserve the inspected Gandy--Scott Algorithm 3/Theorem 3.1 implementation and
M16's fixed Gaussian transition experiment: epsilon0.3, L5, power32, seven rank
draws, initial16384 independent anchors, three looks, multiplier2, bounded-radius
rank/two-sample tests and the independent Gaussian-energy test. The exact
baseline is the existing TF/TFP kernel. Each look draws a fresh complete
experiment. Numerical settings are inherited comparators, not new defaults;
source and official-code anchors remain those recorded in M16.

First run two development complete experiments per baseline/no-op arm on the
new snapshot, at most 180 CPU seconds. If numerical health and the look/stream
inventory pass, freeze **512 fresh experiments per arm** before execution.
At nominal probability 0.05 the normal 95% planning half-width is about 0.019;
report exact Clopper--Pearson intervals. The primary screen is unchanged from
M16: every experiment must be valid and each null rejection interval's upper
endpoint must be <=0.10. This is an operating-characteristic screen, not proof
that nominal size equals 0.05. Do not pool older-source evidence into the new
denominator, add repetitions until the screen passes, or treat an invalid fit
as a detected defect.

The old CPU cost was 2779 seconds for 128 trials with three arms. Two arms and
512 trials project roughly 7411 seconds before source/environment effects.
Reserve 10800 CPU seconds for the fresh null study, subject to the new pilot.
The estimate is inherited measured evidence, not a guaranteed runtime.

## Controlled errors through complete ordinary fits

Add a diagnostic-only normal-conjugate target translation with explicit
severity d. If the intended posterior is N(mu,s^2), compute the deliberately
wrong density `log p(q-d*s | data)` and its exact score. Its actual target is
N(mu+d*s,s^2), with unit translation Jacobian. Here
`s^2 = 1 / (tau^-2 + n*sigma^-2)` follows by completing the square in the
declared prior and likelihood. This is a changed posterior target, not merely
a score perturbation that a correct Metropolis correction might preserve.
The prior-data generator and independent assessor continue using the original
law. Serialize the control and severity in target/source identity.

Implement only for normal-conjugate data-bearing numerical routes. Reject an
unsupported target, missing data, missing/nonfinite severity or a reference
route that would ignore the control. Verify the translated mode, curvature,
density difference and exact zero-severity no-op independently. A public-dispatch
test checks that the mutated adapter reaches actual preparation. No production
tuner or numerical policy changes.

The first full-fit pilot has two datasets per arm and three **independent
complete fits per dataset** for baseline, no-op and d=0.5. Each fit starts fresh
ordinary preparation, searches the broad L grid, preserves all verified
candidates, preselects the first candidate ID at L=3, and runs the existing
posterior controller. One fixed first-chain terminal draw per complete fit
forms a rank; correlated draws or sibling candidates do not multiply the
sample size. Use the inherited tau2, sigma1, n6 model, M21 preparation/schedule
and MCSE tolerance0.05. Root seed2026092222 and separate declared arm streams
are convenience identifiers; truth never enters preparation.

Reserve 3600 CPU seconds for this 18-fit pilot. It tests activation, engineering
integrity and affordability; two ranks cannot establish sensitivity. After its
measured cost, freeze the largest justified complete-experiment inventory that
fits the remaining M22 allocation. M16 found that 128 datasets with three ranks
could detect a half-SD shift under its analytic comparator, but actual HMC costs
three fits per dataset. Quantify that cost honestly. If independent repeated
128-dataset experiments cannot support a lower-0.8 power bound within the
budget, perform only the separately declared affordable detection experiment
and leave **complete-fit power open**. Do not rename a nonrejection or one
detected error as a calibrated power result. A quarter-SD study requires its
own adequate evidence and is not silently covered by the half-SD result.

## Roles, repairs and execution audit

Invalid target arithmetic, stream reuse, source mismatch, corrupt evidence or
an unactivated mutation invalidates that experiment and triggers a localized
repair. Candidate, warmup and precision caps remain outcomes and missing-rank
causes. A valid rank-test rejection under baseline triggers target/stopping/
statistic diagnosis; a valid nonrejection under a defect triggers quantity and
power diagnosis. Neither changes tuning membership. Fix all known arithmetic
or routing defects before untouched confirmation.

M22's ceiling remains 43200 CPU and 14400 GPU worker-seconds. New GPU runs
require free permitted capacity, XLA and verified growth. CPU arms are explicit
reference/development exceptions with hidden GPUs and one intra/inter-op
thread. All pilot, failed and successful numerical worker time is charged.
Use new `m22-r1` directories, immutable source, exact manifests and the existing
validation suite executor. No raw M16 checkpoint is resumed on new code.

Skeptical review: a wrong force with the correct endpoint Metropolis target is
not automatically a wrong stationary law; the translation control instead
changes the declared posterior in a mathematically checked way. Analytic
statistical power is not whole-HMC power. The independent-look theorem does
not confer stopping-time coverage on retained chains. Pilot counts and rate
denominators are distinct, and the affordability decision precedes any fresh
confirmation outcome. Default promotion, general subtle-defect sensitivity
and universal posterior calibration remain outside these bounded results.
