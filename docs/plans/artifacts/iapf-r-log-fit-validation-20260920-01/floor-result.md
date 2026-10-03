# The d80 failure was caused by the local positive floor

The smaller positive-floor candidate completes the previously failing
80-dimensional linear-Gaussian case and a fresh dataset without increasing
the controller limits. This establishes a viable optional R reference
candidate. It does not establish likelihood accuracy from four repetitions,
replicate the paper's results, or validate the TensorFlow/LEDH/KDM programs.

## What went wrong

The independent R implementation uses the paper's Gaussian-plus-constant
guide, but the paper does not specify the size of that constant. Our original
choice made the constant dominate the proposal at a difficult observation.
The regression solver remained well behaved while the particle proposal lost
almost all of its observation guidance. Increasing the iteration budget would
therefore have treated a symptom rather than the identified mechanism.

Write the transition as f(x)=N(x;a,Q) and the fitted guide as
psi(x)=N(x;m,V)+c. The proposal is exactly the mixture

    f(x) psi(x) / (J+c) = [J/(J+c)] q_G(x) + [c/(J+c)] f(x),
    J = N(m;a,Q+V),

where q_G is the Gaussian obtained by multiplying the transition and guide
densities. Thus the probability of discarding the Gaussian guidance is
p_floor=c/(J+c). A constant small relative to the guide's peak can still be
large relative to the overlap J between the guide and transition.

Our rule sets c=N(m;m,V) exp(-q/2), with q the chi-squared upper-tail quantile
at probability N^-p, originally p=2. Direct substitution gives

    log(c/J) = {log[det(Q+V)/det(V)]
                + (a-m)'(Q+V)^(-1)(a-m) - q}/2.

The determinant term grows with dimension; the displacement term grows when
the transition and guide disagree. The floor dominates when their sum exceeds
q. This equation proves the mixture mechanism for fixed inputs; the saved
replays establish that those inputs occurred in the failing run. The paper's
Eq.16 and explanation are in local text lines925-944; the formula for c is
our reconstruction, not a checked original-author choice. Nothing here proves
the author's implementation suffered this failure.

At observation93, two saved bad passes proposed every particle from f:

| Saved call / N | Mean floor probability | Floor draws | Original log error | Zero floor only at93 | Zero floors throughout |
|---|---:|---:|---:|---:|---:|
| 15 / 4000 | .9999966004 | 4000 | -37.697750 | -1.205137 | .657626 |
| 19 / 8000 | .9999942211 | 8000 | -36.045951 | -1.892848 | -.676411 |
| 20 / 8000 | .1649265335 | 1314 | -.178183 | -.207645 | -.032761 |

All three unmodified replays reproduce the saved likelihood exactly, with
prefix/ESS parity within serialization tolerances. At time93 the two bad
passes have ESS1.003 and1.006. The corresponding log-floor values are
-155.878/-155.620 while mean log overlaps are -174.609/-173.509. The mixture
identity therefore explains why the floor overwhelmed the guided component.
Interventions use the same initial RNG state, but altered mixture sampling
changes later draws. Their differences are mechanistic/descriptive evidence,
not paired estimates of an average treatment effect. Zero floors are diagnostic
limits and were not adopted.

## Positive-floor repair and checks

The candidate changes only the explicit floor-tail power from2 to8. It retains
positive c, the log-quadratic objective, diagonal guides, k5/tau.5/kappa.5,
N0=1000 and the original20-iteration/N16000 caps. The core and replication
runner are unchanged. The default remains paper_eq15 with power2.

Power8 was frozen from a calibration of mixture probabilities on300 saved
proposal clouds. Maximum probabilities for powers2/3/4/8/16 were respectively
1, .999999999999893, .999999943685828, 8.31552e-14 and9.77018e-50.
Eight was the first candidate below the declared1e-8 disturbance threshold;
likelihood errors did not select the power. This finite-cloud calibration is
not a uniform guarantee over all states or a default for other targets.

Seven healthy-case comparisons are exactly equal, including particles,
weights, ancestors, prefixes and likelihood. A separate Gaussian-limit check
tests tail integrability. With next guide covariance V_next and observation
precision C'R^-1 C, define H=C'R^-1 C+A'(Q+V_next)^-1 A (omit the future term
at the final observation). Up to constants and linear terms, the conditional
second-moment integral for a Gaussian-only guide is

    integral f(x) [g(x) J_next(x)]^2 / N(x;m,V) dx
      = constant * integral exp[-x' M x/2 + b'x] dx,
    M = Q^-1 + 2H - V^-1.

Positive definiteness of M establishes finiteness for each fixed ancestor
in this Gaussian limit. All1700 checked matrices pass the scale-relative
threshold100*d*machine_epsilon; the minimum relative eigenvalue margin is
.307997. This does not give a uniform full-filter variance bound or controller
convergence theorem. The positive floor remains part of the evaluated method.

## Completed repair runs

| Dataset | Repetitions | Adaptive passes / final particles | Mean likelihood ratio | Bootstrap95%, descriptive |
|---|---:|---|---:|---|
| d20,70000020 | 4 | 7 / 2000 | .975785 | [.912960,1.035728] |
| d20,71000020 | 4 | 7 / 2000 | 1.038341 | [.973042,1.103640] |
| d80,72000080 (repair pilot) | 2 | 8 / 2000 | 1.051459 | [.978388,1.124529] |
| d80,74000080 (fresh) | 4 | 8 / 2000 | 1.088298 | [1.008317,1.162667] |

Each completed controller uses a separate fresh final filter estimate. The
fresh d80 ratios are1.081940,.975797,1.105880,1.189576. The interval with four
observations is highly uncertain; its upper endpoint exceeds1.1, so these data
cannot close the stricter32-repeat accuracy screen used earlier. No bias or
superiority conclusion follows. All9000 fits pass the declared checks.

The d20 non-harm screen passes, but smaller errors are not universal: ordinary
prefix MSE rises by36.3% and25.6% on the two datasets; large-innovation MSE
changes by+3.0% and-38.2%. All changes stay inside the prespecified factor2
screen. This limited check does not establish statistical equivalence.

For each dataset, iAPF prefix MSE is below all three constructed comparator
values in both declared situations. At d80 the table is:

| Dataset / situation | iAPF | Fully adapted N5000 | BPF N10000 | SIS N10000 |
|---|---:|---:|---:|---:|
| Pilot / ordinary | .130870 | 1.146630 | 2.01241e6 | 4.45261e7 |
| Pilot / large innovation | .252080 | 1.141410 | 3.56917e6 | 8.71625e7 |
| Fresh / ordinary | .112178 | 5.621120 | 2.34106e6 | 4.82897e7 |
| Fresh / large innovation | .112717 | 5.291705 | 2.14002e6 | 4.20031e7 |

These are conditional heuristic veto checks at unequal budgets, not evidence
of a statistically supported ranking or greater efficiency. Exact Kalman
remains the accuracy reference.

## Decision and remaining evidence

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject original power2 d80 setting in this scope | Controller incomplete; floor collapses guidance | Reproduced | Original-author floor unknown | Preserve as failure evidence | Failure of original iAPF |
| Retain optional power8/log-fit candidate | Controllers complete; finite ratios | Healthy, tail, fit, non-harm and conditional screens pass | Few repeats; dataset dependence; tails | Frozen32-repeat validation on two untouched d80 datasets and d40 regression control | Accuracy, new default, published replication |
| Repair logging harness | Numeric exit code and both streams preserved | Focused regression passes | No remaining observed status fault | Keep retry provenance | Numerical rejection from attempt08 |

| Inference status | Finding |
|---|---|
| Hard veto screen | Power2 controller failure reproduced; no power8 numerical/fit/heuristic veto in completed runs |
| Statistically supported ranking | None |
| Descriptive-only differences | All repair ratios, intervals, prefix MSE and runtime comparisons |
| Default-readiness | Not established; explicit optional R reference |
| Next evidence needed | Multi-dataset32-repeat accuracy intervals, broader floor/guide tail behavior and source-setting resolution |

Engineering evidence: eight earlier focused tests pass; the final suite adds
a permanent R phase-log/status regression. Final command/log evidence is in
tests-floor-final.log and checks-floor-final.json. Source capture and complete
row-set/hash checks pass in each inspection.json. The numerical core remains
SHA256 c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0.

Attempt07 used90.618141 seconds. Attempt08 used53.397834 seconds and failed
because R log capture returned text instead of an exit code. It is a harness
failure, not a candidate failure. The corrected retry in attempt09 used
320.289619 seconds; combined repair time373.687453 is below600. Total campaign
use is1474.773800/1800 seconds. There were eight planned launches plus one
localized infrastructure retry, all under the same total time allowance.
325.226200 seconds remain, with no further launch allocated and no running job.
The prior terminal report is preserved in pre-floor-continuation/.

Post-run skeptical review: exact replay plus the analytic mixture identity
strongly supports the floor mechanism on these trajectories. Success after a
smaller floor does not prove that diagonal log fitting is reliable on arbitrary
datasets. The weakest evidence remains the four-repeat untouched d80 batch;
ordinary d20 MSE also increased descriptively. A failed untouched accuracy or
tail screen would reject advancement of power8, without invalidating the
diagnosis of the original floor. The paper's fitting objective, original data,
solver and floor details remain unresolved. TensorFlow comparison comes after
qualification of this independent reference, and LEDH/KDM scores remain open.
