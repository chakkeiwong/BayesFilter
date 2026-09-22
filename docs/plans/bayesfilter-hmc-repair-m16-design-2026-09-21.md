# M16: acceptance boundaries and diagnostic sensitivity

Refreshed after M15's terminal reconciliation. Routine wrapper implementation
and its passing focused tests are recorded in the engineering note. This design
uses at most 22,000 CPU-reference and 15,000 GPU worker seconds within the existing
campaign. The increase below repairs an underfunded draft; no total campaign
authorization changes. Preserve pilots, caps and failed attempts.

## Questions and evidence

Measure the public acceptance screen near both practical-band boundaries,
the size/power of the newly implemented independent-look Gandy--Scott wrapper,
and the sensitivity of M15's four-category SBC design to specified errors.
Comparators are M8's mostly distant acceptance points, its fixed-look powered
kernel test, and an independent analytic posterior sampler. These experiments
answer different questions and must not be pooled into an accuracy verdict.

Use actual public fixed-pair tuning with fixed identity mass, four dispersed
starts, separate measurement/verification streams and the inherited rungs 1/2/4.
There are no repairs or posterior draws in this experiment. Retain every
qualified pair and every inconclusive or rejected result. R-hat/ESS/MCSE have
no role in tuning. Independent exact stationary Gaussian anchors and endpoint
energy calculations check the reference acceptance; they never supply tuning
starts or admission decisions. Stationary acceptance and finite-window
fixed-start acceptance remain different quantities.

For the sequential test, every look has fresh anchors, insertion positions,
momenta, ties and Monte Carlo null draws. Test the actual frozen HMC kernel
and a deliberately reversed Metropolis ratio, alongside baseline/no-op arms.
A numerical or execution failure is invalid evidence, never defect detection.
The source-checked theorem is conditional on independent look vectors and
superuniform component p-values. This wrapper does not confer sequential
coverage on acceptance qualification, posterior stopping or arbitrary callbacks.

Primary engineering criteria are intact inventories/receipts, separate streams,
correct complete-experiment repetition, known control activation and complete
denominators. Numerical/reference disagreement or corrupt evidence stops the
affected comparison for repair. Primary statistical outputs are qualification,
null rejection and detection frequencies with exact binomial intervals.
The inherited defect sensitivity screen is a 95% lower detection bound of .8.
For baseline size, report whether the 95% upper bound is at most .10, an explicit
development precision screen for a nominal .05 procedure, not proof of exact
size or the theorem's assumptions. Family comparisons are pointwise and cannot
rank algorithms. Failure of power triggers a design repair or an honest
underpowered verdict; it does not count as failure of HMC.

## Acceptance nomination with an independent Gaussian calculation

For a standard two-dimensional Gaussian with identity mass and one leapfrog
step, each position/momentum pair is mapped by

```
M = [[1-epsilon^2/2, epsilon],
     [-epsilon + epsilon^3/4, 1-epsilon^2/2]].
```

Its determinant is 1, and `trace(M' M)=2+epsilon^6/16`. Write the two eigenvalues
of `M' M` as lambda and1/lambda, with lambda >= 1. Orthogonal invariance of the
four independent normal coordinates gives
`2 delta_H = (lambda-1) U + (1/lambda-1) V`, where U,V are independent
chi-square variables with two degrees of freedom. Therefore
`P(delta_H<0)=P(V>lambda U)=1/(1+lambda)`. The reversible, volume-preserving
Metropolis proposal pairs the positive/negative energy regions, so its mean
acceptance is twice this probability: `a=2/(1+lambda)` (the zero-step limiting
case gives one). Consequently

```
epsilon(a) = [64(1-a)^2 / (a(2-a))]^(1/6).
```

This is a local independent reference derivation, not a new runtime policy.
Use L=1 and a=.64, .65, .66, .74, .75, .76 to bracket and hit the inherited .65/.75
boundaries without assuming fixed-L monotonicity at other L. Check actual
TFP stationary reference means against this formula. For each six-point
pilot or fresh device family, use the two-sided normal Monte Carlo radius
`Phi^{-1}(1-.01/(2*6)) * SE`, giving a declared 99% familywise approximate
screen, plus the existing endpoint-energy tolerance
`1e-10*(1+abs(delta_H)+sum(p0^2+p1^2))`. The family level is a convenience
diagnostic choice, separate from qualification; the normal approximation
uses the predeclared 8192 independent bounded marks. A discrepancy triggers
inspection of exact reference marks and energy reconstruction before the
qualification frequencies are interpreted. It does not alter any tuning rule.

Run two GPU pilot searches per point, 100 seconds per point. After reviewing
costs and energy checks, run 32 fresh GPU searches per point at 650 seconds per
point and 64 CPU searches per point at 900 seconds per point. CPU jobs are
reference/non-XLA; GPU jobs use trusted XLA with memory growth. Base evidence
has 128 draws and independent stationary references have 8192 anchors, inherited
from M8. Pilots are excluded from fresh frequencies. The one-step experiment
isolates a boundary mechanism; it does not validate broad-grid preparation.

Synthetic CPU comparisons use known stationary Beta acceptance marks with
concentration 20, the same six means and persistence 0 or .9. Each cell repeats
256 whole controller experiments at 400 seconds, total 4800 seconds. Concentration
and persistence are explicit stress hypotheses inherited from earlier diagnostic
fixtures. Synthetic movement and known means test the screen arithmetic, not
numerical HMC. No nominal repeated-look confidence claim is made.

## Sequential and SBC power allocations

Repeat 128 independent sequential experiments on each device, using the M8
Gaussian epsilon .3/L5/K^32 test, 16,384 initial anchors, seven rank draws,
bounded-radius rank/two-sample tests and the independent Gaussian energy test.
The wrapper has alpha .05, at most three looks and sample multiplier 2 applied
once after look one. These are bounded design hypotheses, not copied mcunit
defaults. Its first component threshold is .05/(3*3), resolvable with 1999 null
simulations. Every experiment includes baseline, no-op and reversed-energy
arms. Before these fresh trials, run two independently seeded pilot experiments
per device at a 200-second ceiling, excluded from the fresh frequencies. The
fresh GPU cap is 4000 and CPU cap 3000 seconds. The draft's 2500/1800 caps were
underfunded: scaling M8's 465-second, 32-experiment two-arm fixed-look result
to 128 experiments and three arms already projects about 2790 GPU seconds,
before extra looks. The revised GPU cap adds about 43% to that projection;
the CPU cap allows the larger anchor count and extra arm relative to M8's
267-second, 32-experiment CPU baseline. Pilots must check those hypotheses.
Unused acceptance/power allocations may fund a documented local resource repair
without changing seeds, target or test definition.

SBC statistical-engine power uses the proper normal model tau=2/sigma=1/n=6 and
three observables: parameter, bounded radius and data log likelihood. Compare
correct outputs, ignored-data prior outputs and posterior location errors of
.25 or .5 posterior standard deviations. Cross dataset counts 32/128 with rank
draw counts 3/15, yielding eight designs; each repeats 256 independent complete
rank experiments at 600 seconds, total 4800 CPU seconds. The 32/3 designs match
M15's normal dataset/rank counts. Other cells are explicit sensitivity designs;
they do not add public HMC fits to M15 or imply power against every sampler bug.
The analytic sampler isolates the statistic's limitations. Full public
ordinary-SBC defect power remains a separate costly experiment if this design
shows inadequate sensitivity.

Reserve 1200 CPU seconds for the already-started engineering tests, 900 for
analysis, 1100 for localized repairs, with another 600 seconds unassigned.
The CPU numerical allocations are 5400+4800+3000+200+4800=18,200.
GPU allocations 600+3900+4000+200=8700 leave 800 for repair/analysis.
No new default or sampler ranking is an exit requirement.
Root seed 2026092196 has separate engine/device/pilot/fresh/cell domains. Exact
JSON designs, manifests, raw tests, inventories, tensors and run indices go to
`artifacts/hmc-repair-master-2026-09-16/m16-r1/` on a frozen reviewed source.

## Terminal refresh and skeptical audit

The proposed baseline now addresses both practical boundaries, rather than
copying M8's distant rates. The Gaussian derivation nominates step sizes only;
actual reference energy checks guard an implementation mismatch. Finite-start
screen rates cannot be called strict stationary-mean classification error.
Repeated p-values from cumulative samples would violate the source theorem;
the implemented fresh-experiment wrapper and tests prevent this reuse.
Arbitrary runtime failures cannot be reported as successful defect detection.
SBC with only three rank draws can have poor power; larger designs explicitly
measure that limitation rather than declaring an absence of detected discrepancy
to be validation. CPU/GPU sources and settings remain distinct.

M15 completed all 82 fresh SBC datasets, with all 246 planned fits available
after one preserved CPU resource continuation. No rank discrepancy was detected.
All 24 CPU stopping pairs were available, but eight reached the precision cap;
all warmup checks completed at their inherited minimum. These results do not
remove the need for boundary and power experiments, and do not justify a
shorter burn-in or a sequential-coverage claim. The M15 integrity audit found
no invalid artifact and leaves 91,734.13 CPU and 30,250.62 GPU seconds.
Revised M16/M17/M18 ceilings leave 51,734.13 CPU and 8750.62 GPU seconds
unreserved, so the resource repair is affordable without changing the campaign.

The audit found and repaired the draft's missing reference-discrepancy radius
and underestimated sequential-test cost. Frozen source identity is
`b5d2d662171ed61dab32a681693400abc1285ec5c13981a6118d1ce33624b54a`;
its wrapper has fresh-look, multiplicity, count-schedule and invalid-experiment
tests. Pilot/fresh seeds are disjoint. These checks support bounded execution;
they do not certify power before the experiment is run. Proceed with pilots,
inspect their numerical/reference checks and costs, then execute the fixed
fresh designs. A real defect or invalid reference triggers repair; poor power
is a result that must be preserved.

### Pilot cost repair before fresh numerical work

All six GPU boundary reference means agree with the derived values within the
declared familywise radius, and endpoint energy checks pass. Two-search
pilot costs were 35.46, 58.02, 83.10, 84.40, 34.54 and 39.66 seconds. Thus
the original 650-second cap for 32 searches would underfund the slow interior
points. Set fresh GPU caps to 1600 seconds per point: 32 searches times the
largest observed average cost plus approximately 18% margin. Keep all
predeclared fresh targets, seeds, counts and criteria. Extend CPU point caps
from 900 to 1800 seconds to allow independently compiled searches and avoid
using the faster synthetic controller as their runtime baseline.

The sequential GPU pilot completed two experiments in 34.09 seconds; the
4000-second fresh cap covers its linear projection with a substantial margin.
The CPU pilot took 76.74 seconds with extra null-arm looks; increase its
still-unstarted fresh cap to 6000 seconds, approximately 22% above the
76.74/2*128 projection. One of two pilot baseline experiments rejected and
both injected defects were detected. Those tiny pilot frequencies cannot
decide size or power; preserve them separately and run the fresh experiment.

The 20 synthetic/analytic-power jobs are already complete at a combined
277.79 CPU seconds, releasing their unused 9600-second ceiling. These actual
costs, the 76.74-second CPU pilot, 16 seconds of wrapper tests, 10,800 CPU
boundary seconds and 6000 CPU sequential seconds leave over 4800 seconds
inside the 22,000-second phase ceiling for analysis and localized repair.
GPU pilots consumed 369.29 seconds; fresh boundary 9600 plus sequential 4000
and an 800-second analysis/repair reserve fit a revised 15,000-second GPU
ceiling. After reserving M17/M18, the M15 ledger still leaves 3250.62 GPU
seconds unassigned. This amendment supersedes the earlier within-phase
resource table, not any scientific criterion. Use fresh r3 suite files.
