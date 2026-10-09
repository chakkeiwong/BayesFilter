# Matched T=50 covariance-guided mixture comparison (2026-10-08)

## Result

The rerun used the old campaign scope: (T=50), (N=1008), FP64 TensorFlow/XLA, TF32 disabled, data seed 26100611, and four paired design seeds 261006101--104. The P44 three-dimensional LGSSM, KSC stochastic volatility model, predator--prey model, and SIR (d=18) used the same target and observations as the saved old runs. The old and new filters received identical initial and process-noise tensors for each seed; the new mixture additionally received fixed branch/ancestor uniforms required by its proposal. GPU 1 was used with verified memory growth.

The old filter replayed protected values and score vectors. Every comparison row completed all 50 steps and passed finite-value, directional-score, and validity checks. The table below is descriptive over four paired designs; standard errors and paired (t) intervals condition on one fixed dataset and the numerical reference.

| Model | Method | Mean likelihood | Mean \(|\ell-\ell_{ref}\)| | Mean score (L_2) error |
|---|---|---:|---:|---:|
| lgssm | old | -132.975242030 | 0.037781598 | 0.099477128 |
| lgssm | old_covariance_only | -132.975233260 | 0.037655567 | 0.100616519 |
| lgssm | new | -132.968394486 | 0.028170090 | 0.083998506 |
| ksc | old | -123.347057715 | 0.464825069 | 0.203998471 |
| ksc | old_covariance_only | -123.445629525 | 0.511588642 | 0.218406021 |
| ksc | new | -122.819548683 | 0.211916762 | 0.063659364 |
| predator_prey | old | -240.275854361 | 0.111998179 | 1.241568242 |
| predator_prey | old_covariance_only | -240.273322222 | 0.113256135 | 1.160679385 |
| predator_prey | new | -240.229966929 | 0.043259414 | 1.234030630 |
| sir_d18 | old | -1902.553778428 | 233.443220888 | 2925.095282127 |
| sir_d18 | old_covariance_only | -1893.723439238 | 224.612881699 | 4845.454420802 |
| sir_d18 | new | -1669.180420776 | 0.181067681 | 31.631943709 |

## Paired new-minus-old errors

| Model | Metric | Mean difference | SE | 95% paired interval (df=3) |
|---|---|---:|---:|---:|
| lgssm | absolute_value_error | -0.009611507 | 0.014927940 | [-0.057118874, 0.037895859] |
| lgssm | score_l2_error | -0.015478622 | 0.017242754 | [-0.070352759, 0.039395516] |
| ksc | absolute_value_error | -0.252908307 | 0.093283302 | [-0.549777406, 0.043960792] |
| ksc | score_l2_error | -0.140339107 | 0.092325797 | [-0.434160998, 0.153482783] |
| predator_prey | absolute_value_error | -0.068738765 | 0.045982311 | [-0.215075001, 0.077597470] |
| predator_prey | score_l2_error | -0.007537612 | 0.262624821 | [-0.843327005, 0.828251781] |
| sir_d18 | absolute_value_error | -233.262153207 | 25.911981506 | [-315.725643015, -150.798663399] |
| sir_d18 | score_l2_error | -2893.463338417 | 1322.072340123 | [-7100.887572560, 1313.960895725] |

A zero-containing interval means that this four-design diagnostic does not support a ranking. A negative mean difference is descriptively favorable because it is an error difference.

## Score-coordinate errors

| Model | Method | Mean absolute score error by parameter coordinate |
|---|---|---|
| lgssm | old | 0.076247838, 0.030533405, 0.039160256, 0.001706135 |
| lgssm | old_covariance_only | 0.078152946, 0.030663080, 0.039005033, 0.001705215 |
| lgssm | new | 0.073281716, 0.017254090, 0.029334164, 0.002257780 |
| ksc | old | 0.168648632, 0.061953364 |
| ksc | old_covariance_only | 0.177953702, 0.079095557 |
| ksc | new | 0.055307378, 0.023448475 |
| predator_prey | old | 0.670878205, 0.016816912, 0.002671120, 0.239993751, 0.626659567, 0.769408693 |
| predator_prey | old_covariance_only | 0.655320682, 0.014300041, 0.002376830, 0.239832662, 0.561417788, 0.689246339 |
| predator_prey | new | 0.711154304, 0.012224565, 0.002575941, 0.168608018, 0.573457181, 0.708001689 |
| sir_d18 | old | 2222.362496432, 1161.284442213, 906.213777960 |
| sir_d18 | old_covariance_only | 3943.990571988, 1840.571988109, 1518.409013732 |
| sir_d18 | new | 27.692954883, 10.742841257, 4.740189356 |

## Raw values and scores

| Model | Method | Seed | Likelihood | Score vector |
|---|---|---:|---:|---|
| lgssm | old | 261006101 | -133.009993533 | 1.808146758, 4.483398421, 1.570872872, -0.092810262 |
| lgssm | old_covariance_only | 261006101 | -133.009614602 | 1.810724780, 4.483279777, 1.570496493, -0.092809069 |
| lgssm | new | 261006101 | -133.009845389 | 1.764305480, 4.557605232, 1.433801759, -0.093624413 |
| lgssm | old | 261006102 | -132.937774125 | 1.782094304, 4.541151878, 1.484859419, -0.095473415 |
| lgssm | old_covariance_only | 261006102 | -132.937533606 | 1.784529046, 4.540706316, 1.484870134, -0.095473610 |
| lgssm | new | 261006102 | -132.909804483 | 1.754435291, 4.522433471, 1.504653432, -0.096241750 |
| lgssm | old | 261006103 | -132.937146739 | 1.690277002, 4.591168186, 1.455822695, -0.091195855 |
| lgssm | old_covariance_only | 261006103 | -132.937621781 | 1.687692741, 4.591194013, 1.456230940, -0.091196331 |
| lgssm | new | 261006103 | -132.976736531 | 1.783110001, 4.534887597, 1.523400054, -0.090929121 |
| lgssm | old | 261006104 | -133.016053722 | 1.594972706, 4.555515731, 1.526450266, -0.093288770 |
| lgssm | old_covariance_only | 261006104 | -133.016163052 | 1.594949299, 4.555444400, 1.526624711, -0.093286565 |
| lgssm | new | 261006104 | -132.977191539 | 1.605204318, 4.568732196, 1.476915073, -0.093646128 |
| ksc | old | 261006101 | -122.732999847 | -0.852342673, 1.118181455 |
| ksc | old_covariance_only | 261006101 | -122.836616322 | -0.871802284, 1.130561717 |
| ksc | new | 261006101 | -123.028727221 | -0.842998087, 0.965517916 |
| ksc | old | 261006102 | -123.633319930 | -0.898086065, 1.034317508 |
| ksc | old_covariance_only | 261006102 | -123.673636017 | -0.885723033, 1.051308707 |
| ksc | new | 261006102 | -122.788946563 | -0.829922587, 0.994227389 |
| ksc | old | 261006103 | -123.898072398 | -1.427737685, 1.067777158 |
| ksc | old_covariance_only | 261006103 | -124.005664190 | -1.453560609, 1.085951724 |
| ksc | new | 261006103 | -122.543298323 | -1.014665074, 1.042611572 |
| ksc | old | 261006104 | -123.123838686 | -0.810406969, 0.989250614 |
| ksc | old_covariance_only | 261006104 | -123.266601571 | -0.806106192, 1.014984183 |
| ksc | new | 261006104 | -122.917222624 | -0.879485112, 1.005684907 |
| predator_prey | old | 261006101 | -240.375422022 | 13.860510430, 3.498041962, 0.072386561, -10.625065891, -13.590844524, 16.507811405 |
| predator_prey | old_covariance_only | 261006101 | -240.384416885 | 13.878439211, 3.496654443, 0.072500566, -10.627590599, -13.619536366, 16.542833199 |
| predator_prey | new | 261006101 | -240.238643454 | 13.698054528, 3.485597817, 0.070541962, -10.377561181, -13.192056668, 16.031590242 |
| predator_prey | old | 261006102 | -240.087571452 | 13.826866616, 3.471980883, 0.065917535, -9.941696675, -12.250194825, 14.859809074 |
| predator_prey | old_covariance_only | 261006102 | -240.079991263 | 13.836243737, 3.477507031, 0.066363065, -9.966672130, -12.352461889, 14.986063055 |
| predator_prey | new | 261006102 | -240.225089425 | 13.801809227, 3.478157839, 0.067608572, -10.110056436, -12.588039943, 15.281572095 |
| predator_prey | old | 261006103 | -240.247078629 | 13.826432146, 3.456611765, 0.066332158, -10.158055751, -12.217870175, 14.825133422 |
| predator_prey | old_covariance_only | 261006103 | -240.244715141 | 13.861838981, 3.460565074, 0.066634981, -10.184018849, -12.286396755, 14.908663299 |
| predator_prey | new | 261006103 | -240.306311775 | 13.703868139, 3.462445118, 0.064561263, -10.108603510, -11.740229511, 14.247616325 |
| predator_prey | old | 261006104 | -240.393345339 | 13.806247582, 3.463788783, 0.070547613, -10.120771964, -13.383858746, 16.254765864 |
| predator_prey | old_covariance_only | 261006104 | -240.384165598 | 13.805764940, 3.462989291, 0.070004801, -10.073002475, -13.264993428, 16.108878512 |
| predator_prey | new | 261006104 | -240.149823063 | 13.955220486, 3.457703715, 0.067679532, -10.039902662, -12.645915992, 15.350771896 |
| sir_d18 | old | 261006101 | -1952.142760912 | -985.454592589, 683.522427325, 559.742750446 |
| sir_d18 | old_covariance_only | 261006101 | -1948.619777919 | 170.325656195, 477.179778711, 415.589543158 |
| sir_d18 | new | 261006101 | -1669.015808264 | 94.337386190, -46.133883343, -25.294092786 |
| sir_d18 | old | 261006102 | -1830.086278679 | -352.627612500, 459.122776831, -1209.137274472 |
| sir_d18 | old_covariance_only | 261006102 | -1880.290452285 | 7724.314280409, -3207.179513071, -1319.037236482 |
| sir_d18 | new | 261006102 | -1669.318241214 | 148.163008780, -57.648551817, -29.545947809 |
| sir_d18 | old | 261006103 | -1920.150336819 | 6355.545004192, -2815.582656772, 207.025633777 |
| sir_d18 | old_covariance_only | 261006103 | -1889.966473961 | 3964.210618722, -1803.817870850, 377.409823219 |
| sir_d18 | new | 261006103 | -1669.404735700 | 119.223022693, -39.016404324, -24.025897460 |
| sir_d18 | old | 261006104 | -1907.835737299 | -1020.720726034, 612.683070418, 1588.414248413 |
| sir_d18 | old_covariance_only | 261006104 | -1856.017052789 | -4092.213783039, 1874.110789803, 3901.064247337 |
| sir_d18 | new | 261006104 | -1668.982897925 | 75.849547720, -48.626200557, -37.291490748 |

## Reference and controls

LGSSM uses the exact Kalman reference; KSC uses the independently checked refined Gaussian-grid reference. Predator--prey and SIR use the saved (N=131072), four-replication bootstrap/Fisher reference from the same observations. Those nonlinear references retain finite-particle and replication uncertainty and are not exact oracles.

The old comparator is the October 6 guarded-pairwise, normal-quantile reset route: flow substeps 8, reset epsilon 102.4, 24 Sinkhorn and 12 balance steps, four marginal and four pairwise correction steps, trust radius 0.5, and the recorded model-specific importance policy. The old covariance-only arm keeps the reset but disables correction iterations. The new candidate uses 16 flow steps, 40 reset steps, epsilon 1, no optional moment correction, and a beta selected by the independent pilot objective at this (N,T) scope. The selected betas are listed in `comparison.json` manifests.

## Interpretation

- **LGSSM:** the new arm has lower mean likelihood error (0.0282 versus 0.0378) and lower score (L_2) error (0.0840 versus 0.0995). The paired intervals include zero, so this is descriptive evidence of parity, not a statistically supported improvement.
- **KSC:** the new arm is descriptively closer to the refined reference for both likelihood and score. The four-design intervals include zero, so the run establishes viability and a favorable diagnostic pattern, not superiority.
- **Predator--prey:** likelihood error is descriptively lower for the new arm (0.0433 versus 0.1120), while score (L_2) error is essentially unchanged (1.2340 versus 1.2416). The score paired interval is wide and includes zero.
- **SIR:** the old arm remains far from the saved nonlinear reference (mean likelihood error 233.44 and score (L_2) error 2925.10). The new arm is close in likelihood (0.1811) and much closer in score (31.63). The paired intervals for likelihood error exclude zero in this fixed-data diagnostic, while the score interval is wide and includes zero because the four score vectors vary substantially. This is evidence that the matched new run repairs the previous SIR failure under this scope, not a general SIR accuracy guarantee.

## Checks and limitations

The old protected replay passed for the P44 LGSSM, KSC, and predator--prey values/scores; the nonlinear saved rows also passed for their four design seeds. Every paired seed has identical initial/process input hashes across methods. No row was invalid. The old and new numerical controls are intentionally different because they are complete algorithm variants; the run does not isolate one code-line change. Only one dataset and four designs were used, and no default, HMC, production, or scientific-superiority claim follows. The full manifests, source hashes, calibration records, traces, raw rows, references, and campaign ledger are under `docs/plans/artifacts/ledh-matched-comparison-20261008-01/`.

## Decision and next action

| Decision | Primary criterion | Veto/uncertainty | Next action |
|---|---|---|---|
| Keep covariance-guided mixture as an optional diagnostic candidate | No invalid rows; matched errors are no worse descriptively in these four scopes | Four-design intervals are weak; nonlinear reference has finite-particle bias; controls differ | Repeat on independent T=50 data/design partitions before any default decision |
| Reject old baseline as a universal accuracy comparator for SIR | Old SIR error is orders of magnitude larger than the matched new arm | This is a candidate failure, not a rejection of all old-route use in other models | Preserve old arm as falsification baseline and investigate why its proposal collapses on SIR |
