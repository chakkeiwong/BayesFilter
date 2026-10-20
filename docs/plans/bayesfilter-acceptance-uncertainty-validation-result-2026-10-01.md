# Acceptance uncertainty: implementation complete, calibration failed

Date: 2026-10-01

The [reviewed plan](bayesfilter-acceptance-uncertainty-validation-plan-2026-10-01.md)
has been executed. The uncertainty arithmetic, adversarial tests, real-model
integration checks, BGS replay, and guide updates are complete. **The tested
batch-means/lugsail interval rule fails the declared statistical promotion
criteria and remains experimental.** It cannot issue tuning artifacts or alter
candidate membership. This is a completed development and falsification phase,
not a completed repair of stochastic tuning admission.

## What was implemented and checked

The TensorFlow/XLA primitive in
[`mcmc_uncertainty.py`](../../bayesfilter/inference/mcmc_uncertainty.py)
estimates ordinary and lugsail batch covariance while retaining the draw,
chain, and quantity axes. It reports discarded remainders, long-run variance,
MCSE, and effective-information estimates. Constant, nonfinite, and nonpositive
variance results remain unavailable; they are not clipped into apparently valid
uncertainty. Valid marginal variances do not establish positive semidefiniteness
of a multivariate lugsail covariance estimate.

[`hmc_acceptance_uncertainty.py`](../../bayesfilter/inference/hmc_acceptance_uncertainty.py)
analyzes Metropolis probabilities, keeping realized accept/reject frequency
separate. It requires an explicit fixed-kernel declaration and evaluates both
the supplied and doubled batch sizes. All six contrasts between four temporal
windows are evaluated within each chain. Simultaneous marginal intervals are
subtracted to construct contrast intervals; this avoids an independence
assumption for adjacent windows. Bonferroni allocation still depends on valid
marginal intervals, which this calibration does not establish. The approximate
Student-t reference is a local experimental choice, not a theorem about
lugsail estimates. A finite positive estimate is named
`variance_estimate_available`, not sufficient effective information.

The report wraps the existing health evaluator and preserves its decision.
Support rejections contribute zero Metropolis probability; invalid telemetry,
stuck/cyclic trajectories, and duplicate traces are exercised by tests. R-hat
remains reporting-only during tuning. Public v5/v6 admission semantics and
historical artifacts are unchanged. The only change to the existing v6
implementation in this work removes an unsupported use of “calibrated” in its
docstring.

The current tuning reference and the tuning chapter included by `docs/main.tex`
describe the experimental layer and its failed promotion. A standalone excerpt
of the chapter was compiled and visually inspected; its
[rendered PDF](artifacts/acceptance-uncertainty-validation-20261001/checks/guide-excerpt/excerpt.pdf)
and build log are preserved. The entire monograph was not rebuilt for this
change.

## Validation evidence

There are **171 passing checks** in two final groups: 44 new/related checks
(nine unrelated tests deselected) and 127 existing verification, candidate-set,
artifact, and public execution regressions. The checks include independent
test-only NumPy covariance formulas, chain permutations, correctly aligned time
reversal, remainders, negative lugsail estimates, near-constant traces,
opposing/common/single-chain drift, oscillation, support rejection, multiplicity
allocation, policy serialization, explicit fixed-kernel declaration, and
unchanged R-hat behavior. CPU graph/XLA equivalence is checked for the new
arithmetic. These tests establish engineering behavior on their fixtures;
they do not prove calibrated statistical admission.

Seven real fixed TFP HMC integrations exercise Gaussian, QR Kalman LGSSM,
near-unit LGSSM, nonlinear sigma-point SSM, mixture, noncentered funnel, and
analytically partially whitened funnel targets. These are small CPU integration
checks, not posterior-accuracy or GPU qualification runs. The funnel map is
analytic; no learned NeuTra transport is trained or promoted.

The executable calibration is
[`acceptance_uncertainty_validation.py`](../../bayesfilter/testing/acceptance_uncertainty_validation.py).
Its final [result](artifacts/acceptance-uncertainty-validation-20261001/run-03/result.json)
contains 20 generated cases evaluated by two estimators, with 256 independently
seeded four-chain trace sets per case, plus six bandwidth-diagnosis cells and
two search stresses. Method evaluations on the same generated traces are
paired; they are not additional independent replications. Earlier attempts
remain preserved and are not pooled into a larger sample.

Stationary fixtures use the bounded process
`a[t] = mu + amplitude*S[t]`, where the symmetric two-state chain has
`E[S[t]S[t+k]] = rho**k`. Thus
`Var(mean(a[1:n])) = amplitude**2/n * (1 + 2*sum((1-k/n)*rho**k, k=1..n-1))`.
This follows by summing all covariance terms in the mean and supplies an exact
finite-length variance comparator. Division by four gives the variance of the
average of four independent identically distributed chains. A normal interval
using that variance is still only an approximate interval, not an exact
finite-sample coverage oracle.

### Strong dependence defeats the proposed intervals

The table reports the final 8,192-draw-per-chain cases. “Joint coverage
delivered” requires the declared marginal intervals to cover together;
unavailable estimates count as nondelivery. Coverage is not conditioned on
successfully obtaining an estimate.

| Case | Estimator | Joint coverage delivered | Conflict reports |
| --- | --- | ---: | ---: |
| Stationary IID, mean .70 | Ordinary batch means | 249/256 | 0/256 |
| Stationary IID, mean .70 | Lugsail | 183/256 | 0/256 |
| Stationary, rho=.995 | Ordinary batch means | 5/256 | 63/256 |
| Stationary, rho=.995 | Lugsail | 20/256 | 37/256 |
| Opposing within-chain drifts | Ordinary batch means | Not a stationary coverage case | 256/256 |
| Opposing within-chain drifts | Lugsail | Not a stationary coverage case | 194/256 |

For the stationary rho=.995 case, lugsail's false-conflict frequency is 14.45%,
with a pointwise Wilson 95% interval of 10.67%–19.29%. This fails the declared
5% ceiling. Its interval delivery also fails badly. Some IID windows yield
nonpositive lugsail estimates, so unavailable estimates contribute to its
coverage-delivery failure as well. Twenty-seven method/case promotion checks
fail in the final report.

The cheap comparator set includes naive point-band nomination, IID-normal
intervals, the existing v5 raw screen, the optional v6 paired-chain screen,
and ordinary batch means. Results are retained separately by stationary,
persistent, drifting, and short-trace cases. On the persistent stationary case,
the v5 temporal screen fires in 249/256 replications and v6 in 1/256. But v6
misses all 256 opposing-drift cases: averaging the signed changes across
chains cancels the effect. This exposes a limitation of using that pooled
contrast to assess within-chain temporal stability. No overall ranking or
universal calibration follows from these individual screens.

At 512 draws, each of four windows contains 128 draws. The declared base batch
is 11 and its sensitivity batch is 22; the latter leaves five complete batches,
below the eight-batch requirement. All experimental results are therefore
insufficient under this allocation. That is a limit of this design, not proof
that every possible 512-draw diagnostic must fail.

The plan's unconditional coverage-delivery requirement therefore fails by
construction at this short allocation. Abstaining here is correct engineering
behavior, not a false statistical decision. The strongly dependent 8,192-draw
false conflicts independently rule out promotion, without relying on that
short-allocation delivery requirement. Future criteria must distinguish
erroneous decisions from budget-limited, explicitly inconclusive outcomes.

Fresh-seed bandwidth diagnosis does not rescue promotion. At base batch 128
and rho=.995, ordinary batch means delivers joint coverage in 86/256 cases
and reports 14/256 conflicts; lugsail gives 107/256 and 8/256 respectively.
Coverage remains inadequate. A separate comparison of overlapping contiguous
windows shifted by 1,024 draws changes decisions in 58/256 replications.
Those overlapping windows are paired sensitivity observations, not independent
replications or evidence for selecting whichever boundary looks favorable.

### Search and fresh verification

Each stress has 128 independent searches, eight candidates, three measurement
looks (512, 2,048, and 8,192), and a disjoint fresh verification bank. For the
outside-band persistent process (mean .855, rho=.995), naive point-band
nomination occurs in 19/128 searches. The experimental rule produces no
compatible nomination or fresh delivery, with or without the declared
multiplicity adjustment. Zero observed deliveries have a pointwise 95% upper
Wilson bound of about 2.91%, not a zero-error guarantee.

For viable IID mean .70, all 128 searches obtain an experimental nomination,
but only 99/128 obtain adjusted fresh delivery (97/128 without adjustment).
This exercises the fresh-bank path and exposes delivery loss. These counts
do not justify ranking the adjusted and unadjusted rules. Nomination and
verification have distinct seed banks; no sequential-validity claim is made
from approximate marginal intervals or from three inspected looks.

### BGS replay

All eleven saved raw log-acceptance tensors decode, and their historical mean
and decision reproduce. The primary `L=3`, epsilon
`0.00044697315517561615` tensor has SHA-256
`eb183221549211096c07f367d54e038f7d57c5d5d192018ab89442c0c31d32f8`.
All eleven experimental reports are insufficient under the declared 512-draw
allocation. The result records source paths and hashes; it neither changes
the historical decisions nor establishes that any BGS candidate is valid.
The live DSGE and state-space campaigns were not restarted or modified.

## Execution and reproducibility

The final [manifest](artifacts/acceptance-uncertainty-validation-20261001/run-03/manifest.json)
records commit `de80aaff5812ebfbed551977476c0868551a2c88`, source checksums and
snapshots, exact command, environment, seeds, timestamps, and artifacts.
The worktree contains unrelated changes; the source snapshots identify the
executed numerical code more precisely than the commit alone.

All commands used conda `tfgpu`, TensorFlow 2.20.0, CPU reference execution
with `CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`TF_NUM_INTRAOP_THREADS=4`, `TF_NUM_INTEROP_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`. No GPU budget was consumed. Calibration attempts
took 30.95, 39.70, and 48.97 wall seconds, about 120 seconds total, within the
declared two-hour wall and three-attempt caps. Test runtimes are separate:
40.50 and 18.03 seconds for the two final groups.

The final calibration command was:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout 3600 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  -m bayesfilter.testing.acceptance_uncertainty_validation \
  --output docs/plans/artifacts/acceptance-uncertainty-validation-20261001/run-03 \
  --replications 256 --wall-seconds 3500 \
  --bgs-summary /home/ubuntu/python/dsge_hmc/docs/experiments/bgs/2026-09-28-longer-evidence/results-512-round.json
```

For a repeat, choose a fresh output directory. The
[test log](artifacts/acceptance-uncertainty-validation-20261001/checks/bf-acceptance-final-tests.log),
[JUnit report](artifacts/acceptance-uncertainty-validation-20261001/checks/bf-acceptance-final-tests.xml),
and [existing-regression log](artifacts/acceptance-uncertainty-validation-20261001/checks/bf-acceptance-existing-regressions.log)
preserve final checks. The new tests are in
`tests/test_hmc_acceptance_uncertainty.py` and
`tests/test_hmc_acceptance_uncertainty_models.py`; existing regressions cover
verification, candidate tuning/artifacts, exact-pair fresh evidence, and
reporting-only R-hat.

## Decision and remaining repair

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not established |
| --- | --- | --- | --- | --- | --- |
| Keep the experimental arithmetic and tests | 171 checks pass | No observed engineering failure in these fixtures | Untested backends/shapes and targets remain | Retain reusable diagnostics and regression coverage | Statistical calibration from arithmetic tests |
| Reject default promotion of the tested interval rule | Coverage and false-conflict criteria fail | Statistical promotion veto fires | Dependence and variance-estimator instability | Resolve the uncertainty estimand and dependence strategy before a fresh calibration | Reliable admission for arbitrary short HMC traces |
| Preserve BGS historical decisions | 11/11 replay | New reports have insufficient batches | Candidate quality and initialization bias unresolved | Obtain new fixed-kernel evidence under a reviewed allocation | BGS qualification or rejection by this experiment |
| Complete this bounded development phase | All six phases delivered | No continuation-invalidating artifact failure | Admission repair remains open | Carry the explicit next task into the master program | Completion of the overall randomness repair |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Existing numerical/health vetoes remain; invalid variance estimates cannot supply experimental evidence. |
| Statistically supported ranking | None claimed across methods or HMC candidates. |
| Descriptive-only differences | Per-case detection, coverage delivery, boundary sensitivity, search delivery, and runtime, with pointwise replication intervals where applicable. |
| Default-readiness | Failed for the tested experimental rule; preexisting v5/v6 rules have the documented limitations. |
| Next evidence needed | A justified uncertainty target and dependence strategy, fresh-seed coverage/error/delivery calibration, then public-controller integration and independent verification checks. |

The next repair must first distinguish a stationary acceptance expectation
from the explicitly finite-start, finite-horizon expectation of repeated
fixed-kernel trials. For the former, batch-size adaptation or another
long-run variance estimator needs an explicit dependence-resolution criterion;
larger batches and a finite MCSE alone are insufficient. For the latter,
independent replications from a declared start distribution permit uncertainty
over trial summaries, but change the estimand and must not be advertised as
stationary calibration. Four differently initialized chains are not an
identically distributed replication sample by assertion.

Before public integration, the chosen strategy must pass new held-out
stationary/dependent and drift tests, quantify inconclusive delivery as well
as false decisions, and account for the actual look/candidate budget. Repairing
epsilon, changing preparation, or changing the candidate scope requires fresh
measurement under its own identity. Independent verification remains mandatory.
Only after that evidence should an admission policy be versioned and connected
to both public tuners with checkpoint/membership regression coverage.

The strongest alternative explanation for poor delivery is the deliberately
conservative simultaneous allocation and invalid lugsail windows, rather than
failure of all uncertainty estimation. It does not explain away false conflicts
in the stationary persistent case. New held-out coverage and false-conflict
results could overturn the rejection of a revised policy. The weakest part of
the current evidence is transfer from bounded synthetic traces and tiny model
integrations to long state-space HMC runs; no such transfer is promoted here.

## Git integration checkpoint

The publication check used a clean temporary checkout containing the acceptance
commit `70a6d7e96` and remote main `88297ad29`, merged as `56bab6c28` without
conflicts. This exposed dependencies that the original shared-worktree test
could not detect: the optional v6 comparator, its version-aware execution
configuration reader, and the state-space fixture adapters were still
uncommitted. The integration includes those prerequisites. The model checks
now call the same state-space factory directly, without depending on the
uncommitted campaign registry. Neither the model laws nor numerical defaults
change in this packaging repair.

The first collected merged run passed 190 checks but failed eight execution
setups because the old reader expected the optional v6 field in v5 payloads.
After including the version-aware reader and adding v5/v6 configuration
round-trip and common-drift tests, **201 checks pass in the merged checkout**.
These include both state-space adapter suites and the public candidate
execution regressions. The ignored local Sylvester binary was copied from
the shared checkout for these CPU checks; it is not part of the commit.

The [final merged-test log](artifacts/acceptance-uncertainty-validation-20261001/checks/bf-acceptance-merged-r2.log),
[JUnit report](artifacts/acceptance-uncertainty-validation-20261001/checks/bf-acceptance-merged-r2.xml),
and [validation record](artifacts/acceptance-uncertainty-validation-20261001/checks/merged-checkout-validation.json)
preserve the command, environment, source hashes, and result. The failed first
run remains alongside them. The calibration was not repeated; its failed
statistical promotion and the original 171-check development record above
remain unchanged. Unrelated shared-worktree campaign edits were excluded.
