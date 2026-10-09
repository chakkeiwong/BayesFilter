# NeuTra remedy: fitting studies and development HMC validation

## Current result: development validation complete

All six prescribed NAF fits completed after the localized inverse repair.
Five final maps passed the preliminary screen and all five subsequently passed
shared sequential HMC, precision and independent-reference checks. Three-mode
seed 11 failed the fit screen and was not admitted. This is evidence that the
optional NAF route can support useful corrected sampling on these two
exact-teacher development targets; it is not a faithful canonical IAF success,
full Gaussian whitening, generic mode discovery or a method ranking.

| Target / fit seed | Epsilon | L | Retained per chain | Maximum R-hat | Minimum bulk / tail ESS | Maximum original-quantity MCSE/SD | Fresh reference |
|---|---:|---:|---:|---:|---:|---:|---|
| calibration_random_two / 11 | 0.2 | 9 | 2000 | 1.001065 | 1222.0 / 1335.3 | 0.02746 | Passed |
| calibration_random_two / 37 | 0.1 | 18 | 2000 | 1.003620 | 1898.3 / 1883.2 | 0.02311 | Passed |
| calibration_random_two / 73 | 0.1 | 18 | 6000 | 1.001496 | 3450.2 / 3048.6 | 0.02925 | Passed |
| calibration_random_three / 37 | 0.1 | 9 | 4000 | 1.002987 | 1541.9 / 1311.5 | 0.02828 | Passed |
| calibration_random_three / 73 | 0.0707107 | 18 | 4000 | 1.002588 | 1833.5 / 1552.7 | 0.02721 | Passed |

Each selected kernel used 2,000 warm-up transitions per chain, archived and
excluded from estimates. On three-mode seed 73, the first kernel (epsilon .4,
L=3) failed the second warm-up chunk with nonfinite proposal target, momentum,
score and acceptance quantities. It produced no retained estimate. The next
predeclared, independently verified kernel (epsilon .0707106781, L=18) passed
all checks. The exact cause of the failed proposal arithmetic is not localized;
it is preserved as a kernel-specific numerical rejection, not silently erased
or treated as evidence that every kernel fails.

The retained runs still contain finite extreme log-acceptance alerts: 257, 201,
728, 314 and 284 respectively. The public controller labels these explanatory;
TFP does not expose a native divergence count here. These observations and the
1,000-point score tails show that full whitening has not been established,
even though the selected kernels pass this downstream scope.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Five maps viable in development scope | Sequential convergence, precision and fresh-reference checks passed | No hard veto for selected kernels | Exact teacher, favorable initial distribution, two targets | Separate native-teacher calibration/transfer study | Generic pipeline or q20 readiness |
| Three-mode seed 11 rejected | Final coarse shape screen failed | Fit-screen veto | Training variability | Preserve for later fit repair | NAF direction rejected |
| Three-mode seed 73 first kernel rejected | No retained estimate | Nonfinite proposed quantities in warm-up | Extreme-proposal arithmetic not localized | Use the independently verified alternative already tested; preserve failure | Global numerical correctness for all step sizes |
| Diagnostic repair accepted for this study | Exact iid and wrong-location controls passed | Saturation defect exposed before learned-map HMC | Folded log-odds is a declared diagnostic transformation | Retain original probability precision/reference checks | Threshold or target relaxation |

| Inference status | Result |
|---|---|
| Hard veto screen | One fit and one kernel rejected; five selected maps/kernels pass declared checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, required lengths, KL and score-tail differences |
| Default readiness | Not established; canonical IAF unchanged |
| Next evidence needed | Native-teacher source/accuracy checks, transfer, then a frozen procedure on untouched randomized targets |

The terminal audit verified source and artifact checksums, all five distinct
map-specific numerical bindings, identical physical starts within each target,
CPU reference placement, GPU memory growth and frozen-transport-only geometry.
See `campaign-r1/development-terminal-audit-r1.json`. The full pre-run check
passed 131 tests. A later controller repair makes completed resumes audit saved
evidence without rerunning tests, pricing or numerical workers; 17 focused
controller/route checks passed, and the actual terminal resume launched none.
The live route ledger also corrects an overly broad historical classification:
the September-24-named q20 wrapper delegates its current posterior call to the
shared controller; old map eligibility remains a separate policy. This ledger
clarification does not change the completed numerical source snapshot.

Five learned-map HMC jobs consumed 3166.067 GPU-process seconds, plus the 90.890-second Gaussian control. Current remaining balances are 81927.755904 GPU-process seconds (22.76 hours) and 84631.889878 CPU-core seconds (23.51 hours). All numerical, repair and test costs remain charged; no worker is active.

The strongest alternative explanation is exact-teacher access plus added NAF
capacity, with posterior starts drawn from the correct target. Failure under
native teachers or unseen geometries would overturn an inference of broader
transferability. The weakest evidence is the small target set and finite
operational screens. Native teacher transfer is the next justified phase;
its old FAB, Gabrié, AFT and CRAFT prerequisites must be inspected explicitly,
not relabeled as full source-faithful implementations because a student map
now works. The current native-bank harness also generates on GPU; serious
external sample generation needs a separately specified multicore CPU lane
under the repository policy. No native phase was silently launched through that
unrepaired route. The overall remedy campaign is not complete.


## Historical status before the NAF representation study

At the end of the initial IAF fitting and matched optimizer studies, no learned
map had passed the preserved distribution screen, and no posterior confirmation
or final random target had been run. The conditional representation check was
the next planned step. The records below preserve that sequence; the completed
NAF and development HMC results at the top of this report supersede its earlier
execution status. Untouched final randomized targets remain untested.

## Checked historical diagnosis

`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/history-audit-r1.json`
links all 48 October 2 fits to ancestors, optimizer restoration, heldout records
and posterior outcomes. It reproduces the 106 warm-up caps, 76 incomplete resource
outcomes, 36 retained caps and 8 numerical/health outcomes in the saved summary.
These are repeated map/kernel observations, not independent replications.

Five of the eight final joint-continuation maps passed their coarse coverage
screen. The oracle/seed-11 fits on both targets and the warped estimated-teacher
seed-37 fit did not. Thus neither "all joint fits lost coverage" nor "all failed
because of undertraining" is supported. Some apparently adequate coarse fits
still failed warm-up or retained-event precision. Budget stops do not prove a
numerical defect. The raw audit preserves missing or inconclusive evidence.

## Matched exact-teacher results

Two development targets, three independent fit seeds each (11/37/73), width 64,
three canonical IAF stages, batch 64, LR .001 and 8,192 forward updates were
matched across fixed 4,096-point banks and fresh iid batches. Each fresh parent
then supplied three 2,048-update branches with identical reset Adam. All 30
endpoints and prescribed 1,000-point diagnostics were saved. GPU/XLA execution,
memory growth, immutable source copies and artifact hashes passed the terminal
audit. FP64 is explicitly diagnostic; these are not TF32 promotion results.

Heldout forward-KL estimates in nats are descriptive:

| Target / seed | Fixed bank | Fresh data | Forward continuation | RKL continuation | Joint continuation |
|---|---:|---:|---:|---:|---:|
| Two / 11 | .112 | .092 | .121 | .701 | .108 |
| Two / 37 | .116 | .092 | .105 | 3.181 | .106 |
| Two / 73 | .138 | .120 | .081 | .336 | .080 |
| Three / 11 | .375 | .352 | .314 | 3.328 | .332 |
| Three / 37 | .364 | .345 | .315 | .870 | .288 |
| Three / 73 | .378 | .325 | .268 | .968 | .308 |

Fresh data gives lower observed FKL in all six pairs, but the seed count and
absence of a predeclared powered comparison do not establish superiority. All
maps still fail the distribution screen, principally conditional-shape/moment
checks. Accurate mode weights alone are insufficient. Pure RKL worsens forward
fit in every pair under this continuation allocation; it remains a diagnostic
branch instead of an automatic final stage. These results do not prove RKL
always fails, that joint loss cannot work, or that IAF lacks sufficient capacity.

The six complete workers took about 14.1 GPU-process minutes. The initial
342-second forecast per five-endpoint job was conservative: actual jobs took
about 141--144 seconds. Exact costs and cumulative remaining balances are in
the campaign state. A lower-LR continuation is the next discriminator; it is
not selected by ranking individual descriptive seed outcomes.

## Original-author reference

The unchanged Gabrié `flonaco.training.train` completed 1,500 iterations in an
isolated CPU reference process, using the paper's six coupling pairs, width 100,
depth three, 40 walkers/batch 400 and LR .005. The original local-ULA/global-MH
order, retries, clipping and optimizer were preserved. All previously unspecified
settings are recorded as source-default hypotheses in
`bayesfilter-neutra-author-reference-2026-10-05.md`. No package installation or
source compatibility patch was needed.

The full process took 170.02 wall seconds / 169.89 CPU-core seconds. The observed
raw-flow responsibility masses were (.31363,.68637), against (1/3,2/3). Forward
KL was .11083 and reverse KL .20457. Mass outside both radius-squared-8 component
ellipses was .04651, exceeding the target upper bound exp(-4)=.01832. These are
one-run descriptive results. The source controller can execute and keep both
modes while leaving a density-fitting error; the paper itself reports a residual
bridge. This is not exact Figure 5 reproduction, posterior confirmation or a
matched ranking against our IAF on different targets.

## Completed optimizer repair

Both LR arms restarted Adam from the same fresh forward parent and received
8,192 further forward updates on the same new exact samples. Each endpoint
therefore has 16,384 lifetime updates. All twelve endpoints failed the preserved
shape screen, although their losses and mode weights often look favorable.

| Target / seed | FKL at LR .001 | FKL at LR .0003 | Max shape z at LR .001 / .0003 |
|---|---:|---:|---:|
| Two / 11 | .07743 | .05104 | 7.459 / 5.642 |
| Two / 37 | .07214 | .04329 | 10.777 / 5.650 |
| Two / 73 | .09518 | .07424 | 8.341 / 7.118 |
| Three / 11 | .19026 | .29458 | 10.045 / 12.196 |
| Three / 37 | .22456 | .26930 | 8.586 / 11.855 |
| Three / 73 | .16000 | .24213 | 7.747 / 11.568 |

These observations oppose a universal instruction to lower the LR. The lower
rate is descriptively favorable on the two-mode target and unfavorable on the
three-mode target under this fixed effort. Neither choice delivered an admitted
map. All 82 mechanics/source/controller checks passed before this stage, and
the completed worker/source artifact audit has no errors. Actual repair cost
was 559.33 GPU-process seconds. After settling the author reference, retrospective
audit and readiness costs, 2,481.182329 GPU-process and 6,641.169138 CPU-core
seconds remained before representation pricing. The next diagnostic is the
existing DSF/NAF configuration, with its full-cost reservation and distinct
source/numerical limitations stated in the representation plan.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Controller mechanics | Passed 80 checks initially, 82 after optimizer-repair checks | No new numerical invalidity in first comparison | Full scientific quality separate | Continue authorized repair | Correctly trained NeuTra |
| Fixed/fresh and optimizer fits | All 42 endpoints fail coarse distribution screen | Shape/moment mismatch | Further optimization versus representation | Price and test configured DSF/NAF | Failure of native teachers |
| Original code | Complete prescribed training count | Raw-flow discrepancy remains | Single realization; source defaults differ from an exact figure reconstruction | Preserve baseline for attribution | Author/local equivalence or superiority |
| Full pipeline | Posterior criterion not reached | No map admitted | Transfer and posterior precision untested | Representation test if optimizer repair fails | Research-direction rejection |

| Inference status | Current conclusion |
|---|---|
| Hard veto screen | Local maps fail distribution checks; no newly observed numerical failure in completed jobs |
| Statistically supported ranking | None |
| Descriptive-only differences | KL estimates, residuals, mode masses and timings |
| Default readiness | Not established |
| Next evidence needed | Controlled optimizer/representation repair, then native-teacher transfer and untouched posterior confirmation |

Post-run skeptical check: the strongest alternative to a capacity limitation is
that the inherited LR/update allocation leaves substantial optimization error.
The matched LR repair reduces some errors without resolving the failure. A good coarse screen would nominate
a candidate, not prove posterior correctness; conversely a precise shape veto
must not be mistaken for a failure of six unexecuted sampling methods.

## Representation pricing and current execution boundary

The expanded 110-test suite passed, including the configured NAF's inverse,
parameter derivatives, source mapping and frozen-consumer tests. The two-block
GPU pricing worker completed 1,024 forward updates with finite gradients in
103.396 process seconds. Its warmed block took 29.011 seconds for 512 updates.
The configured roundtrip error was 4.76e-11 and logdet discrepancy 3.11e-15.
The optional NAF has 56,868 parameters versus 13,836 in the matched IAF. This
is a timing/validity diagnostic and says nothing about final NAF fit quality.

Pricing the six complete 16,384-update NAF recipes gives a reservation of
6,816 GPU-process seconds and 13,632 CPU-core seconds. After the pricing run,
2,377.786281 GPU-process seconds and 6,355.025211 CPU-core seconds remain.
The controller correctly stopped before launching an underfunded comparison.
The owner subsequently granted an additional 24 GPU and 24 CPU hours and
requested continuation, superseding the earlier 2/3-hour request. The increment
of 86,400 seconds per resource is recorded once in both nested allocation
ledgers. Cumulative study limits are 92,400 / 98,400 seconds, with
88,777.786281 GPU-process / 92,755.025211 CPU-core seconds remaining before
resumed execution. Prior charges remain intact. The complete NAF comparison is
funded and has resumed through the same wrapper. No permission rule blocked it.

The final endpoint audit checked all 42 completed IAF endpoints: every required
1,000-point diagnostic is complete and finite. Across their distinct training
arms, 233,472 updates were recorded and none was clipped. The current failure
therefore is not explained by clipping in these runs. Passing the finite probe
check is not a whitening certificate. The report and restart state preserve
native transfer and untouched posterior confirmation as outstanding work.

## Funded continuation: infrastructure record

The first resumed source check passed 109 tests and failed the cumulative-budget
fixture, which implicitly used the old live study limit. The fixture now sets
both of its synthetic limits explicitly; the production ledger continues to
retain prior charges. This is a test-fixture repair, not a numerical failure.
The failed check is preserved in `campaign-r1/check-r4` and charged. No NAF
scientific worker launched before this check; the unchanged comparison follows
a successful repeat.

The repeat (`check-r5`) passed all 110 tests. Refreshed pricing
(`representation-price-r2`) completed in 106.421 GPU-process seconds; the warmed
512-update block took 30.526 seconds. The same forecast formula gives a
1,190-second worker ceiling and a six-fit reservation of 7,140 GPU-process /
14,280 CPU-core seconds, approximately two hours elapsed on one GPU.
`naf-calibration_random_two-s11-r1` has launched on GPU 1 with verified memory
growth. GPU 0 and GPU 2 were occupied by other work. The controller executes all
six specified fits sequentially and preserves their two endpoints. These fits
are running; they have no final scientific result yet. Before charging this
active worker, the ledger has 88,671.365186 GPU-process / 92,296.888921 CPU-core
seconds remaining. Consult live state for subsequent charges and completion.

## Numerical failure and localized inverse repair

The first NAF worker did not finish the recipe. Its first 8,192 forward updates
were finite with no clipping. That saved endpoint has descriptive forward KL
0.0203632, reverse KL 0.0278063 and maximum feature z=5.91373, so it still fails
the existing shape screen. It is not the intended 16,384-update endpoint.
Continuation rejected update 2,828 and preserved Adam iteration 2,827; the worker
exited after 284.732 seconds against its 1,190-second cap.

The exact saved-branch replay reproduced the failure with finite parameters.
One scalar inverse returned invalid after 25 iterations, even though its loop
had just accepted the root. Final absolute residual 1.2116974090758958e-11
exceeded tolerance 1.2116901911200309e-11 by about 7.22e-17. Separate XLA
loop/final evaluations disagreed at the stopping boundary. The actual failing
checkpoint, batch and per-coordinate values are in
`campaign-r1/debug-naf-calibration_random_two-s11-r1-r1/`.

The numerical repair tightens internal stopping to half the requested residual
and bracket tolerances. Final tolerances, iteration cap, density, architecture,
data, loss and Adam settings remain unchanged. The compiled scalar regression
includes neighboring queries around the observed failure. A complete saved
branch replay is required before the full six-fit retry. This is evidence of a
local numerical implementation defect; it does not reject NAF or its objective.

| Decision | Primary criterion | Numerical veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| First NAF attempt | Full comparison incomplete | Reproduced inverse stopping failure | Quality after complete training | Regression, exact replay, full retry | NAF failure as a method |
| Forward checkpoint | Shape screen failed | Finite saved endpoint | Remaining optimization and geometry | Preserve as intermediate evidence | Correctly trained or whitened map |

The diagnostic used 108.398 GPU-process / 117.104 CPU-core seconds. Its source
and output are preserved separately from the failed training attempt; all costs
remain charged. Verification and retraining results follow below when complete.

The repaired source passed 111 focused tests. The saved lower-LR branch then
completed all 4,096 replay updates on GPU/XLA with finite objective/gradients and
zero clipping, passing the old update-2,828 failure. Independently, a CPU/XLA
reference loaded the exact pre-failure checkpoint and exact rejected batch,
without retraining. Its loss is finite (3.9505779802), gradient norm 3.5728583,
and three parameter-direction finite differences have maximum scaled error
2.12558e-7. This distinguishes the repair from merely escaping the failed state
through a different training trajectory. The CPU reference cost is 9.066 wall /
17.742294 CPU-core seconds, recorded in `inverse-repair-reference-r1/result.json`;
settle this external diagnostic into both ledgers once the active master releases
the lock. The full six-fit retry is running from the original prescribed starts
and datasets on the repaired source; its results remain pending.

The first repaired full fit, two-mode development target / seed 11, completed
all 16,384 prescribed updates in 458.222 GPU-process / 488.110 CPU-core seconds.
Its final lower-LR endpoint passes the existing coarse screen: FKL 0.0096423080,
RKL 0.0179236745, maximum feature z 2.86519, maximum responsibility discrepancy
0.00510703. The earlier forward endpoint still fails with z=5.91373. All four
training blocks were finite and no update was clipped. This makes the final
endpoint viable for further validation; it establishes neither architecture
superiority nor posterior correctness. The second fit (two-mode / seed 37)
started automatically; five full fits are still outstanding in this comparison.


## Active monitoring and downstream diagnostic repair

The first five repaired full NAF fits have completed normally. The three
final two-mode maps and three-mode seed 37 pass the coarse screen. Three-mode
seed 11 fails it at z=5.82252 despite lower descriptive FKL; its intermediate
checkpoint passed at z=3.66407. The final prescribed seed is still running.
No intermediate checkpoint is substituted for a failed final endpoint.

While training continued, the prospective downstream phase was implemented
under `bayesfilter-neutra-development-hmc-2026-10-05.md`. It preserves the
public fixed-map tuner, identity latent mass and canonical sequential sampling.
Initial focused checks found two pre-existing unclassified HMC-related files;
the route ledger now records their explicit reference/historical status.

More importantly, exact iid mixture controls exposed an assessment error before
any fitted-map HMC ran. Tail ESS was undefined for responsibilities rounded to
one, although R-hat, bulk ESS and mean precision passed. Computing a constant
95th-percentile event is not a valid convergence test for that numerical
representation. The local diagnostic now uses directly computed log-odds for
ordered quantities, with responsibilities retained for mode-mass estimates,
mean precision and exact-reference comparison. The plan derives the relation
and explains why folded distances are assessed on the declared transformed
quantity. This changes no target, sampler, probability estimate or shared
posterior threshold.

The updated 27 focused tests pass, including both mixtures' iid controls,
wrong-location rejection, score finite differences, saturation regression,
identity map, route registration, candidate-failure continuation and common-
control stopping. Tests and the causal diagnostic have separate preserved
outputs and pending CPU charges; the next master will settle them exactly once
after obtaining the shared lock. GPU downstream validation remains unexecuted
at this checkpoint.


## Complete six-fit representation result

All six full fits completed on the repaired source. The source/artifact audit
passed, all 98,304 updates were finite, and all twelve saved 1,000-point probes
are complete and finite. Three updates were clipped, one in each three-mode
fit; the pervasive clipping seen in the historical campaign is absent here.

| Development target | Seed | Final screen | FKL (descriptive) | Maximum feature z | Score residual median | Score residual maximum |
|---|---:|---|---:|---:|---:|---:|
| calibration_random_two | 11 | Passed | 0.00964231 | 2.86519 | 0.292144 | 536.53 |
| calibration_random_two | 37 | Passed | 0.00379385 | 2.63147 | 0.0999294 | 1156.96 |
| calibration_random_two | 73 | Passed | 0.0066021 | 3.3567 | 0.170852 | 748.232 |
| calibration_random_three | 11 | Failed | 0.0158166 | 5.82252 | 0.310533 | 1289.76 |
| calibration_random_three | 37 | Passed | 0.0114964 | 2.88452 | 0.302163 | 553.742 |
| calibration_random_three | 73 | Passed | 0.011084 | 2.86059 | 0.273106 | 690.815 |

Five final maps are eligible for the prospective downstream phase. The failed
three-mode/seed-11 endpoint remains rejected under the unchanged screen; its
passing intermediate map is not substituted. Small medians coexist with large
score tails. These results do not establish Gaussian whitening or usable HMC.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Representation comparison complete | Six prescribed full fits | No remaining numerical/source/probe veto | Coarse fit and derivative tails differ | Public fixed-map HMC on five eligible maps | NAF superiority or default replacement |
| Three-mode seed 11 rejected | Final shape screen failed | Candidate-only screen | Training variability and shape error | Retain failure for later fit repair | Failure of the whole direction |
| Posterior assessment repaired before HMC | Exact iid positive and wrong-location negative controls pass | Saturated responsibility tail-ESS defect resolved locally | Difficult-target HMC still untested | Gaussian full call-chain control, then per-map tuning | Posterior or q20 readiness |

| Inference status | Result |
|---|---|
| Hard veto screen | One final fit fails its shape screen; all twelve probes finite |
| Statistically supported ranking | None; three fit seeds and no ranking inference plan |
| Descriptive-only differences | KL, feature z and residual quantiles/maxima |
| Default readiness | Not established; canonical IAF unchanged |
| Next evidence needed | Actual corrected sampling, then separate teacher/generalization study |

The monitored training phase is complete. The next wrapper invocation settled
82.492066 CPU-core seconds of separately recorded diagnostics/tests once into
both ledgers. Before its new full checks and GPU work, 85,184.712062 GPU-process
seconds and 88,322.367898 CPU-core seconds remain. Session 41182 now owns the
development validation continuation. No new allocation or permission-rule change
was needed.

Post-run red team: the strongest alternative explanation for favorable density
results is extra capacity and exact-teacher access, rather than broadly better
learning or sampling. Actual HMC can overturn a favorable interpretation; the
weakest evidence remains the coarse single-endpoint screen and score-tail
measurement. Native teachers and untouched targets remain outside this result.


The immutable downstream source passed all 131 full checks. The Gaussian
full-call-chain control passed in 90.890 GPU-process seconds: its
first selected verified kernel (epsilon=1.6, L=3) passed after 2,000 warm-up
and 2,000 retained transitions per chain, with no hard veto and successful
fresh-reference agreement. This validates the exercised control path; it
does not establish learned-map quality. The first eligible two-mode/seed-11
NAF map is now running its own tuning scope and sequential assessment.


The first learned-map downstream result passed: two-mode development seed 11,
epsilon=0.2 and L=9, 2,000 warm-up and 2,000 retained transitions per chain.
The parent measured 654.537 GPU-process seconds for the full tuning/validation
worker, including shutdown. Eight kernels were independently verified; the
first declared representative passed sequential and fresh-reference checks.
Minimum bulk ESS is 1,221.98 and minimum tail ESS 1,335.25 among the shared
ordered quantities. Original-quantity maximum MCSE/SD is 0.0274582.
Estimated mode masses are 0.729625/0.270375, with MCSE 0.0121964 each;
the independent reference gives 0.734804/0.265196. These satisfy the declared
operational agreement screen, without establishing simultaneous coverage.

Finite extreme log-acceptance alerts remain: the second retained chunk records
124. The current shared controller explicitly treats these as explanatory,
not native divergence counts. Nonfinite, movement, convergence and precision
checks passed. The map supports this bounded downstream computation despite
imperfect whitening; its low forward KL did not eliminate all integration
stress. Four other eligible maps remain pending/running. The complete remaining
queue reserves 7,200 GPU-process / 14,400 CPU-core seconds, within the authorized
allocation; the first measured cost suggests about 44 minutes if the other maps
cost the same, but the three-mode runtime is still unmeasured.
