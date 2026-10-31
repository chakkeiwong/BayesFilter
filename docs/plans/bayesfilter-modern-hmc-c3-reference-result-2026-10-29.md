# C3 independent MacroFinance reference: result and continuation

29 October 2026. The independent reference and GPU quantity checks passed
their declared screens. The seven-member GPU pricing stage started at
23:01:16 Asia/Shanghai. Posterior assessment and the master program remain
incomplete. The [C3 plan](bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md)
defines the target, measure, diagnostics, numeric provenance and continuation.

## Question and checked target

The question is whether a separately implemented covariance Kalman likelihood
and independent integration can assess the seven retained kernels on the
frozen 48-observation, nine-parameter MacroFinance target. The comparator is
the public factor-QR adapter at commit
f9291ee4437aa8c00dabbe331bffc6871b574c5c. The integration uses
z=theta/0.35 with a standard-normal prior; it includes the same prediction-first
timing, stationary initial law, location constants and parameter chart. No
chart Jacobian is added to the stated prior. HMC draws do not fit the proposal
or estimate the reference.

The claim checked by the numerical parity work is equality of these two value
programs for this model and their derivatives at the recorded test points.
Observed discrepancies are consistent with floating-point and finite-difference
error. This finite test does not prove equality throughout the parameter space.

## Results and repairs

| Stage | Result | Interpretation |
|---|---|---|
| Independent reference tests | Dense Gaussian timing, Lyapunov, row independence, measure and quantity tests passed | Numerical implementation checks |
| C3a public CPU/XLA parity | 28 points; max value error 2.842170943040401e-14; max chart error 2.6645352591003757e-15 | Same target at checked points |
| Central-difference scores | Max errors 3.76196e-7 at h=1e-5 and 9.44658e-8 at h=5e-6 | Both pass absolute 1e-7 plus relative 1e-8; not both below absolute 1e-7 |
| Reference01 | Import failure before integration | Frozen diagnostic helpers were missing from the clean sampler checkout |
| Reference02 | Max empirical envelope 0.1188137462 posterior SD; screen fails | Initial proposal was inadequate |
| Reference03 | All 31 quantities pass; max empirical envelope 0.002180955192 SD versus 0.02 limit | Reference is usable for the declared experimental agreement screen |
| GPU/XLA quantity preflight | 28 points, all 31 quantities; max error 1.4210854715202004e-14 | Runtime quantities match the independent definition |
| Focused regression suite | 36 passed in 1.36s | Reference, proposal, seed, identity and pricing mechanics |

Reference02 inherited a center and covariance from generating-parameter
geometry. These were a warm-start hypothesis, not a posterior fit. The first
calibration had importance ESS 13.77 out of 32768 and the second 86.31 out of
32768. Fitting a mixture to such concentrated weighted samples left poor
coverage in the final integration. This failed reference candidate did not
invalidate the likelihood or the seven HMC candidates.

Reference03 used seven independent optimization starts. All converged to the
same checked local solution; residual gradient maxima were below 5e-6. Two
finite-difference Hessian step sizes agreed and had positive curvature. The
proposal retained the standard-normal prior component and expanded local
components after each of two discarded calibration fits. Final validation
used eight independent scrambles at each of powers 14, 16 and 18. At power18
the observed importance ESS was about 183000 out of 262144 per scramble.
No HMC draws or prior final-validation draws entered the new fit.

The repaired proposal changed integration efficiency, not the target, prior,
scientific estimands or acceptance threshold. Reference02's pricing seed
overlapped one calibration component seed, although its final validation
streams were separate. Reference03 inventories fully disjoint pricing,
calibration and validation streams. Both earlier attempts remain preserved.

## Decision and uncertainty

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept numerical parity at checked points | Passed | No source drift or nonfinite required evidence | Finite set of points and finite-difference error | Use independent reference for experimental comparisons | Global analytic proof |
| Reject reference02 | Failed 0.02-SD envelope | Concentrated weights are a repair trigger | Proposal coverage | Preserve and repair proposal independently | HMC failure or invalid target |
| Retain reference03 | All quantities pass unchanged screen | No source/measure/quantity mismatch found | Unseen modes and empirical integration error | Price all seven kernels, then bounded posterior checks | Exact posterior, simultaneous coverage or release |

| Inference status | Finding |
|---|---|
| Hard veto screen | No parity/source/nonfinite veto remains in reference03; reference01 was a harness import failure |
| Statistically supported ranking | None; no HMC-method ranking was performed |
| Descriptive-only differences | Importance ESS, concentration, runtime and uncertainty reduction describe these attempts |
| Default-readiness | Not established; kernels remain experimental |
| Next evidence needed | Shared sequential posterior checks and agreement for every retained member; C4 estimator/tail reliability remains separate |

The strongest alternative explanation for apparent integration stability is
a mode missed by all local starts and sampled too rarely by the defensive
prior component. The repaired checks reduce some risks but do not prove global
coverage. A materially different independently stable reference, or repeat
integration outside the empirical envelope, would overturn the current
reference-use decision. The weakest evidence is calibration of the empirical
refinement-plus-randomization envelope, which is not a rigorous error bound
or simultaneous confidence interval.

## Reproducibility and compute

Full manifests, exact commands, frozen sources, seeds, data hashes, progress,
replication summaries and terminal results are under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/c3-reference-20261029-01`.
The parity, failed import, first integration and repaired integration cost
15.25544701400213s, 0.2317074550082907s, 59.72888099704869s and
225.33257259009406s respectively: total 300.54860805615317s, charged once.
These independent diagnostic processes deliberately hid GPUs and used the
tfgpu environment's NumPy/SciPy reference lane. Runtime sampling remains
TensorFlow/TFP on GPU with XLA.

GPU preflight artifacts and the active price stage are under
`N5/c3-posterior-20261029-01`. Preflight inclusive supervisor cost was
18.11339380289428s, within the preparation reserve; its child time is not added.
The price stage has a 7200s inclusive cap and preserves all seven identities.
Its measured terminal costs will determine the posterior reservation. The
separate SIR allocation is untouched. No tuning qualification was rerun.

## Posterior preparation audit and live continuation

The first posterior driver saved durable chunks but lacked a resume command.
Before any posterior sampling, this was repaired by delegating to the existing
cohort-continuation helper. It checks original member identity, policy, seeds,
quantities and committed chunk integrity; it carries terminal successes and
failures without resampling. The helper is frozen separately from the clean
numerical checkout. Active price01 continues on its original frozen driver.

The expanded CPU-only regression run passed **66 tests in 217.19s**, including
interrupted/uninterrupted equality, repeated interruption, corrupt/missing
checkpoint rejection and preserved terminal failures. Seven additional checks
against the exact frozen sampler source passed: actual shared configuration,
price target/reference scope, cohort delegation, charge accounting,
unaffordable/failing-cohort rejection, narrow program refresh and native
interrupted/uninterrupted equality. CPU devices were explicitly selected for
these mechanics tests; they are not GPU performance evidence.

At 23:15 Asia/Shanghai, L=3/.4 and L=5/.4 had completed 1000 transitions per
chain in 192.488s and 206.754s respectively, including validation and reporting;
L=9/.4 was running. These are descriptive work prices, not method rankings.
Trusted process inspection confirmed the pricing supervisor977947 and
worker977950 were alive. No error was recorded.

Continuation observer990814 started at23:14:31 and is waiting for all seven
prices and terminal settlement. It then validates the frozen scope, reserves
only an affordable maximum-count posterior tranche, and starts the prepared
shared-controller cohort. If checks fail it records the reason without
silently dropping members or changing criteria. The supervisor owns the
time cap, minute-by-minute status and30-minute capacity reassessment. The
observer updates the master and active checkpoint at launch and settlement.

The exact preparation, frozen sources, regression logs and live status are
under `N5/c3-posterior-20261029-01`, particularly
`posterior-preparation.json` and `continuation-01/status.json`. The overall
master remains incomplete; automated C3 execution does not close C4/C5/C6.
