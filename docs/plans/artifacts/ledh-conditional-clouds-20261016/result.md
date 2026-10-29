# Conditional particle-cloud and particle-count result

Completed 17 October 2026 on `ledh-refactor-with-policy-fix`. The two future-guided
candidates fail the predeclared descriptive heuristic promotion screen: 35 of
64 candidate-versus-heuristic comparisons have larger observed log-process-SD
score MSE. Neither guided candidate can be promoted from this experiment.
All 832 evaluations are numerically valid. This rejects promotion of these
candidates; it does not invalidate the implementation or stop the research.

Doubling N from 32 to 64 lowers observed global-arm MSE in seven of eight
dataset/guide comparisons. Five paired, pointwise intervals exclude zero in
that direction. The stress cases contain both appreciable cloud variability
and mean error at N=32; both decrease descriptively at N=64. Future guidance
itself has no supported reduction in the primary score error in any of the
eight fixed-global-beta comparisons. One comparison instead supports harm
conditionally on its fixed dataset, subject to the exploratory interval limits.

## Question, estimator and scope

The [pre-run plan](../../ledh-conditional-cloud-localization-20261016.md)
asks whether the Gaussian score error reflects particle randomness, conditional
mean error, or both. The target is the exact Kalman **model likelihood score**.
The computed estimator is the total analytical derivative of the finite shared
covariance-mixture filter with fixed data, randomness, guide coefficients and
beta. These quantities are different at finite N; differentiating that program
correctly does not make it the exact model score.

The diagonal two-state AR model has T=5, drift (0.9,0.9), process SD 0.3,
observation SD 0.5, prior N(0,I), and predict-first timing. Each of the ordinary
and impulse regimes has two fixed final datasets and twelve independent
particle clouds per dataset. The impulse adds three marginal SD at observation
index 2, coordinate 0. The final data seeds are 317200/317201 (ordinary) and
318200/318201 (impulse). N=32 uses the exact prefix of each N=64 evaluation
cloud. Calibration is separate for each N/regime/guide, and validation/final
observations are disjoint from calibration. No score outcome selects a beta.

Eight arms produce 768 final rows plus 64 validation rows. The four heuristic
adversaries are the physical identity proposal, current-only global flow,
current-only local flow, and equal density mixture. The independently
current-calibrated mixture supplies the optimized classical comparator. Future
guidance is tested at fixed global beta and at the future-calibrated beta; a
current-only arm also uses that latter beta to isolate guidance. Global/local
beta means (0.1,0.9,0)/(0.1,0,0.9), not an unprotected single proposal.

The optional mixture's moment-restoring reset is not canonical Contract E
admission. OT stays at epsilon 1 and five Sinkhorn iterations; flow/correction
counts stay 16/2. Runs use FP64 GPU/XLA with TF32 off, an explicit reference
exception. They do not establish production TF32, scope-tuned defaults, HMC
readiness, nonlinear transfer or trained KDM quality.

## Conditional heuristic comparison

Entries are observed log-process-SD score MSE over twelve clouds, conditional
on the indicated dataset. I is identity; CG/CL/CE are current global/local/equal;
CC is current with its own calibrated beta; FG is future-global; CF/FF use
the same future-calibrated beta with current/future guidance respectively.
All arm, dataset and coordinate results are preserved in [decision.json](decision.json).

| Dataset | N | I | CG | CL | CE | CC | FG | CF | FF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary 0 | 32 | 0.2550 | 0.4012 | 0.6121 | 0.2166 | 0.2863 | 0.5647 | 0.3602 | 0.1769 |
| Ordinary 0 | 64 | 0.3498 | 0.1495 | 0.2066 | 0.1194 | 0.1336 | 0.1159 | 0.1288 | 0.2057 |
| Ordinary 1 | 32 | 0.1861 | 0.7620 | 0.1345 | 0.4292 | 0.1487 | 0.3894 | 0.3796 | 0.3797 |
| Ordinary 1 | 64 | 0.2752 | 0.0988 | 0.1565 | 0.1362 | 0.1005 | 0.4182 | 0.1717 | 0.1387 |
| Impulse 0 | 32 | 8.2289 | 3.3186 | 1.0701 | 1.3692 | 4.4865 | 2.9485 | 2.8916 | 1.3988 |
| Impulse 0 | 64 | 2.7711 | 1.2660 | 0.7728 | 0.6772 | 0.8561 | 1.2871 | 1.1406 | 1.7110 |
| Impulse 1 | 32 | 6.5270 | 5.0310 | 5.4820 | 3.9013 | 4.4269 | 5.8158 | 5.2032 | 7.0988 |
| Impulse 1 | 64 | 4.0818 | 1.1249 | 3.9465 | 1.4001 | 0.7333 | 1.7629 | 0.8290 | 1.2677 |

FG loses to at least one heuristic in seven of eight cells; FF also loses in
seven of eight. These are descriptive promotion vetoes by the stated plan,
not 35 statistically established inferiority findings. The current-calibrated
arm also has descriptive losses to simple heuristics, so it is a comparator,
not a newly promoted default. No observed minimum in this table is selected.

## Cloud variability and mean error

For errors e_r and R=12, let m be their mean and v their sample variance.
The sample identity is MSE=m^2+11v/12. Under independent clouds, v estimates
conditional variance, sqrt(v/12) is the estimated MCSE of m, and
m^2-v/12 estimates squared conditional bias without the finite-replication
upward bias. Negative estimates are retained. They reflect estimation noise,
not a negative true squared bias. The implementation and independent identity
test match this derivation.

| Dataset | N | Guide at global beta | Mean error | MCSE | Variance | Estimated squared bias |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary 0 | 32 | Current | -0.0018 | 0.1910 | 0.4377 | -0.0365 |
| Ordinary 0 | 32 | Future | 0.2160 | 0.2170 | 0.5651 | -0.0004 |
| Ordinary 0 | 64 | Current | -0.1470 | 0.1078 | 0.1395 | 0.0100 |
| Ordinary 0 | 64 | Future | 0.0057 | 0.1026 | 0.1264 | -0.0105 |
| Ordinary 1 | 32 | Current | 0.1374 | 0.2599 | 0.8107 | -0.0487 |
| Ordinary 1 | 32 | Future | -0.1020 | 0.1856 | 0.4135 | -0.0241 |
| Ordinary 1 | 64 | Current | -0.0503 | 0.0935 | 0.1050 | -0.0062 |
| Ordinary 1 | 64 | Future | 0.0411 | 0.1946 | 0.4544 | -0.0362 |
| Impulse 0 | 32 | Current | 1.3394 | 0.3723 | 1.6633 | 1.6553 |
| Impulse 0 | 32 | Future | 0.8375 | 0.4520 | 2.4514 | 0.4971 |
| Impulse 0 | 64 | Current | 0.3882 | 0.3184 | 1.2168 | 0.0493 |
| Impulse 0 | 64 | Future | -0.0301 | 0.3419 | 1.4031 | -0.1160 |
| Impulse 1 | 32 | Current | 1.5949 | 0.4755 | 2.7135 | 2.3174 |
| Impulse 1 | 32 | Future | 1.5314 | 0.5617 | 3.7862 | 2.0296 |
| Impulse 1 | 64 | Current | 0.6360 | 0.2559 | 0.7859 | 0.3390 |
| Impulse 1 | 64 | Future | 0.8352 | 0.3112 | 1.1622 | 0.6008 |

Ordinary-case mean errors are small relative to cloud variation. At N=32 the
stress data show positive mean errors alongside substantial variance. Increasing
N reduces both estimates in the stress cells, although residual mean error
remains in impulse dataset 1. Two counts and twelve clouds do not identify an
asymptotic bias floor, convergence rate, or population distribution of errors.
The data support a finite-particle explanation for part of the earlier error;
they do not prove that every remaining discrepancy is Monte Carlo noise.

## Paired uncertainty

Each interval uses 10000 paired bootstrap resamples of the twelve clouds,
conditional on one dataset. Intervals are pointwise and exploratory, with no
multiplicity correction. Resampling preserves guide, beta and N pairing.
Negative MSE differences favor future guidance or the larger count.

| Dataset | N | Future minus current, global beta (95% interval) | Same comparison, future-calibrated beta (95% interval) |
| --- | --- | --- | --- |
| Ordinary 0 | 32 | 0.1634 [-0.1290, 0.4684] | -0.1833 [-0.4081, 0.0224] |
| Ordinary 0 | 64 | -0.0336 [-0.1507, 0.0820] | 0.0769 [-0.0268, 0.2094] |
| Ordinary 1 | 32 | -0.3726 [-1.4230, 0.2868] | 0.0001 [-0.2507, 0.2142] |
| Ordinary 1 | 64 | 0.3195 [-0.0724, 0.9444] | -0.0330 [-0.1122, 0.0370] |
| Impulse 0 | 32 | -0.3701 [-2.6651, 2.5751] | -1.4928 [-3.9939, 0.8148] |
| Impulse 0 | 64 | 0.0210 [-1.6037, 1.8215] | 0.5704 [-1.0746, 2.6574] |
| Impulse 1 | 32 | 0.7848 [-0.9256, 2.5658] | 1.8955 [-4.1731, 8.4857] |
| Impulse 1 | 64 | 0.6380 [0.0906, 1.4092] | 0.4387 [-0.7679, 1.6122] |

All eight calibrated-beta guidance intervals include zero. Seven global-beta
intervals include zero; the last supports a conditional increase in error.
These results do not support a general ranking. Other score coordinates have
mixed exploratory signs: global guidance has one negative interval for each
drift coordinate, two positive intervals for the second drift coordinate, and
two negative/one positive intervals for log observation SD. They do not
override the primary coordinate or the heuristic veto.

| Dataset | N64 minus N32, current-global (95% interval) | N64 minus N32, future-global (95% interval) | Change in guidance contrast (95% interval) |
| --- | --- | --- | --- |
| Ordinary 0 | -0.2518 [-0.4789, -0.0144] | -0.4488 [-0.7589, -0.1859] | -0.1970 [-0.5417, 0.1492] |
| Ordinary 1 | -0.6633 [-1.7084, -0.0443] | 0.0288 [-0.3351, 0.4449] | 0.6921 [-0.1131, 1.8027] |
| Impulse 0 | -2.0525 [-4.8197, 0.2841] | -1.6615 [-6.2482, 1.5241] | 0.3911 [-2.5563, 2.8326] |
| Impulse 1 | -3.9061 [-8.4583, -0.5451] | -4.0528 [-7.9850, -0.6023] | -0.1468 [-1.2863, 1.1968] |

All four count-by-guidance interaction intervals include zero. There is no
evidence here that doubling N specifically rescues the relative guidance
effect. Count contrasts involving calibrated arms, saved in decision.json,
also change independently calibrated beta and are combined interventions.

## Engineering, numerical and mathematical audit

Engineering: 27 distinct focused tests pass, comprising 21 conditional
analysis/cloud tests and six shared-consumer tests. Checks include independent
moment algebra, known paired contrasts, corrupted or duplicate rows/clouds,
calibration scope/beta/certificate mismatches, modified observations, exact
N=32 prefixes, shared-filter wiring, harmless roundoff acceptance and material
moment-error rejection. The analysis verifies the full 832-row design and the
same source closure and controls across N. All 79 recorded runtime source
hashes still match the working tree at terminal inspection; the saved analysis
hash also matches. No numerical core was changed for this stage.
Terminal whitespace checks pass for tracked changes and all seven new stage
files; all ten local links in the new plan/result/checkpoint resolve.

Numerical: all eight calibrations converge with the declared certificates.
N=32 iteration counts are 11/15/14/17 and N=64 counts are 6/16/15/17 in
ordinary-current/ordinary-future/impulse-current/impulse-future order.
All 832 rows have valid diagnostics and there are zero roundoff-warning rows.
Largest reset mean residuals are 6.66e-15/6.88e-15 for N32/N64; largest
relative covariance residuals are 5.55e-15/4.50e-15. No tolerance was tightened
or relaxed. These residuals are observability, not score-quality evidence.

Mathematics: the mean/variance calculation is derived above and independently
tested. The consumer remains the shared analytical finite-program route;
the independent Kalman reference uses differentiation only as a diagnostic
oracle. The fixed-guide/fixed-beta derivative conditions agree with
`prop:bg-mixture-guided-score` in `ledh_backward_guidance.tex`. Its correctness
claim concerns that finite program, not equality with the Kalman score.
The conditional weight-variance objective does not establish a score-MSE
ordering. This stage supplies no new theorem and requires no theorem revision.
No new external MathDev review was performed; the earlier inconclusive
MathDev result is not upgraded by these tests.

## Decision and next action

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete conditional Gaussian localization; do not promote either guide | No primary guidance improvement established; several conditional count effects support lower error | Heuristic promotion veto; no numerical or artifact continuation veto | Twelve clouds, two datasets/regime, two counts, fixed Gaussian target | Advance to the master's target-specific nonlinear protocol and consumer audit; retain this result as a Gaussian control | Universal inferiority, asymptotic correctness/bias, canonical admission or KDM failure |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | All planned evaluations and validity checks pass; zero roundoff warnings |
| Statistically supported ranking | No population ranking. Exploratory conditional intervals support five count reductions and one primary guidance increase |
| Descriptive-only differences | Heuristic losses, bias/variance estimates, ESS/weight tails, other unseparated intervals and runtime |
| Default-readiness | Neither guided arm is eligible; current comparators are not promoted either |
| Next evidence needed | A target-specific nonlinear plan with a checked oracle, shared-consumer wiring, independent fitting/tuning and downstream score uncertainty |

The observed failure is failure to demonstrate a useful Gaussian guided
candidate, not evidence of a broken target, harness, derivative or artifact.
There is no reason to reopen the 1e-13 precision investigation. More Gaussian
particle counts or KDM capacity are not automatic prerequisites for continuing.
The next authorized master step is to inspect a nonlinear target's actual
model/reference/consumer call chain and specify its fitting protocol and
bounded comparison before training or claim-bearing runs. A richer KDM model
is justified only by that target's representation problem; this Gaussian
result alone does not justify it. No additional experiment is launched under
the completed two-count design.

Post-run red team: the strongest alternative explanation is poorly sampled
particle tails on only four datasets. Bootstrap intervals from twelve clouds
may understate tail uncertainty, and scope-specific beta can explain
calibrated-arm count effects. Fresh conditional replications could overturn
the observed guidance harm or descriptive minima. Conversely, a repeated
fixed-beta deficit would weaken that guidance construction. Neither outcome
would retroactively change this experiment's promotion veto.

## Reproducibility and budget

The numerical base is `ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus the
manifested dirty source closure in worktree
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Both manifests preserve commands, environment, TensorFlow version, data
version/seeds, source hashes, score engine, controls, plan and result paths:
[N32 manifest](n32-attempt01/manifest.json), [N64 manifest](n64-attempt01/manifest.json).
Each directory also retains calibration.json, datasets.jsonl, rows.jsonl and
result.json. Analysis settings and source hash are in decision.json.

Python is `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, with
`PYTHONPATH=.`, `TF_FORCE_GPU_ALLOW_GROWTH=true`, `TF_CPP_MIN_LOG_LEVEL=2`,
and `MPLCONFIGDIR=/tmp/bayesfilter-ledh-mpl`. GPU commands run with escalated
access and `CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`.
The RTX 4080 SUPER has memory growth verified before initialization. FP64
placement and XLA execution are recorded. Allocator peak bytes are 612864
and 2095104; these are TensorFlow allocator observations, not total GPU memory.

Executed runner arguments, using that interpreter/environment:

```
scripts/run_ledh_conditional_clouds.py --particles 32 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/n32-attempt01 --budget-seconds 800
scripts/run_ledh_conditional_clouds.py --particles 64 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/n64-attempt01 --budget-seconds 1400
scripts/analyze_ledh_conditional_clouds.py --n32 docs/plans/artifacts/ledh-conditional-clouds-20261016/n32-attempt01 --n64 docs/plans/artifacts/ledh-conditional-clouds-20261016/n64-attempt01 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/decision.json
```

The two GPU commands use `/usr/bin/time -f 'supervisor_wall_seconds=%e'` and
`timeout --signal=TERM --kill-after=5s`, with 820/1420-second supervisor caps;
complete logs are [gpu-n32-attempt01.log](gpu-n32-attempt01.log) and
[gpu-n64-attempt01.log](gpu-n64-attempt01.log). Internal times are 306.874 and
508.504 seconds; supervisor times are 281.55 and 467.95 seconds. Charge the
larger clock per run: 815.378/1800 GPU seconds, leaving 984.622 unused, with
two of four allowed launches used and no retry.

CPU-only verification deliberately hides GPU devices with
`CUDA_VISIBLE_DEVICES=-1`. Commands are `python -m pytest -q
tests/highdim/test_conditional_cloud_analysis.py
tests/highdim/test_guided_mixture_diagnostic_tf.py`, followed after added
artifact checks by `python -m pytest -q
tests/highdim/test_conditional_cloud_analysis.py`; see [cpu-tests01.log](cpu-tests01.log)
and [cpu-tests02.log](cpu-tests02.log). Tests take 35.95 aggregate wall seconds;
analysis takes 1.16 seconds ([analysis01.log](analysis01.log)). Conservatively
charge 45/180 CPU seconds including small terminal inspections, leaving 135.
No commits, pushes, environment changes, HMC runs or external-agent launches
occurred in this stage.
