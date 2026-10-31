# Optional Gaussian-mixture product moment repair — 18 October 2026

## Question and execution scope

Does retaining between-component uncertainty in the global proposal reduce
conditional KSC likelihood-score error? The previous saved-fit diagnosis
identified an omitted variance term, not its causal contribution to score
error. Implement a general-dimensional, batch-native Gaussian-product moment
routine with analytical directional derivatives, expose it as an opt-in shared
global-map consumer, and compare on fresh fit/cloud seeds. Work only in
`ledh-refactor-with-policy-fix`, checkout `ledh-iapf-execution-20260922`, base
`ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus preserved dirty changes.

This is the optional covariance-guided mixture, not admitted canonical LEDH.
The local positive-information maps, physical identity branch, marginal
proposal weights, reset and correction remain unchanged. Guide coefficients
are frozen during differentiation; Gaussian moments and component
responsibilities are differentiated. No runtime autodiff or new ridge.

## Mathematical specification

For Gaussian N(a,S) and fixed positive mixture sum_k pi_k N(c_k,R_k), define
P=S^-1, C_k=(P+R_k^-1)^-1, b_k=C_k(Pa+R_k^-1 c_k), and
alpha_k proportional to pi_k N(a;c_k,S+R_k). The normalized product has mean
b=sum alpha_k b_k and covariance V=sum alpha_k[C_k+(b_k-b)(b_k-b)']. These
are exact moments of that product, not exact filtering moments.

For one parameter direction, dP=-P(dS)P, dC_k=-C_k(dP)C_k,
db_k=dC_k(Pa+R_k^-1 c_k)+C_k(dP a+P da). Put B_k=S+R_k and
u_k=B_k^-1(a-c_k). Then dlogalpha_unnormalized_k=-u_k' da
+(u_k' dS u_k-tr(B_k^-1 dS))/2. Softmax centering gives d alpha.
Differentiate both sums for b and V, including changes of alpha, C and b_k.
Use checked SPD factors; invalid inputs/results reject, never silently ridge.
An inactive guide returns the input Gaussian and tangents exactly. The new
mode requires a mixture guide; unsupported guide types fail at configuration.

## Evidence contract and research intent

| Role | Predeclared requirement |
| --- | --- |
| Mechanism | Restore product mixture uncertainty in the shared global map |
| Primary comparators | Same fitted kernel with old positive-information consumer; unguided global |
| Primary quantity | Squared error of each of two scores versus independent numerical model reference, separately in each observation condition |
| Statistical criterion | 24 planned paired MSE contrasts: two repaired beta arms, two comparators, three conditions, two coordinates. Average paired losses over eight clouds within each of eight independent fit clusters; simultaneous two-sided 95% Student-t intervals, df=7, Bonferroni over 24. A negative upper endpoint supports a conditional improvement; a positive lower endpoint supports harm. Small-cluster normality is an explicit limitation. |
| Promotion criterion | Both comparisons support improvement in every condition/coordinate for an arm, with no heuristic or validity veto; otherwise no broad promotion |
| Promotion veto | Any conditional heuristic MSE loss; supported score harm; fit ineligibility; missing reference/derivative evidence |
| Continuation veto | Invalid target/reference, wrong density or derivative, nonfinite accepted result, corrupted/incomplete evidence, exhausted budget |
| Repair trigger | Candidate failure with valid computation motivates the next discriminating consumer/fit diagnostic, not rejection of the whole direction |
| Explanatory only | Fit loss, ESS, tails, moment residuals, runtime, observed bias/variance, exact-future guide controls |
| Not concluded | Original KDM training, full controlled SMC, canonical UKF/dual-cap/Contract-E admission, scope tuning, TF32 readiness, HMC, cross-model or population superiority |

Conditional situations are the existing fixed ordinary, negative-future-tail
and positive-future-tail T=2 datasets, at theta=(1.5,0), Q=1 and N(0,1)
predict-first prior, seven-mixture KSC observation target. Fresh Monte Carlo
seeds do not turn these already-inspected datasets into new holdout data.

Constructed heuristic adversaries, evaluated within each condition/coordinate:
identity proposal (no flow uncertainty), unguided global (current Gaussian
update), unguided local (conditional local information), unguided equal mixture
(diversification), Gaussian-future global and equal mixtures (one broad moment
approximation). Exact seven-component future guides with global/equal beta
are additional explanatory controls. All use identical paired clouds. A
descriptive heuristic loss vetoes promotion; it does not establish a
statistically supported ranking. No selection uses these final scores.

## Defaults and assumptions audit

| Choice / provenance | Justification and risk | Earliest check / status |
| --- | --- | --- |
| General Gaussian product, derived above | Retains all component moments; current Gaussian remains approximate | Independent dense 2D and batched reference, tangent finite differences / hypothesis |
| Old consumer stays default | Preserves existing finite program; omission already demonstrated | Default and inactive-path parity / baseline |
| No ridge or clipping | Exact SPD algebra defines this candidate; changing factors changes that object | Relative factor margins, invalid-input tests / reviewed mathematical choice |
| Kernel Nfit=256, R=16, bandwidth factor .5 | Selected on prior development loss, not current scores; finite fitting error remains | Fresh-fit eligibility, preserve all fits / frozen comparison hypothesis |
| N=32, T=2; chunks K=32 | Same prior scope, computationally bounded; weak long-horizon generalization | Exact reference and call-chain tests / comparator |
| OT epsilon1, five steps | Owner-requested untuned comparator, not optimum | Existing finite residual/validity traces / baseline |
| Flow16, correction2, basis4, rate .1, radius .25 | Same optional consumer comparison; not independently tuned | No-fire baseline tests, cap/moment trace checks / baseline |
| FP64 GPU XLA, TF32 off | Isolate consumer mathematics; not production TF32 evidence | GPU/XLA compile plus derivative checks, verified memory growth / reference exception |
| Eight fit clusters x eight clouds | Same bounded replication structure; small df limits inference | Paired cluster tables and simultaneous intervals / convenience with explicit limit |

## Execution, budget and stop conditions

1. Implement one shared TensorFlow routine and opt-in control. Focused CPU
   diagnostics test multidimensional/batched values and derivatives, SPD
   rejection, inactive parity, local-map preservation, and actual filter wiring.
2. Run one GPU/XLA comparison: 3 conditions x8 fits x8 clouds x12 arms =2304
   rows. Arms are the eight existing unguided/Gaussian/exact-mixture arms,
   fitted-kernel global/equal under the old consumer, and the same two under
   product moments. Freeze all choices before launch. Fresh fit seeds
   `821800+100*condition+cluster`; fresh cloud seeds
   `831800+1000*condition+100*cluster+replicate`. Check against prior seed
   manifests before launch. Fit once per cluster, shared by both consumers.
3. Recompute the independent reference and enumeration check on CPU, check
   complete-program frozen-fit finite differences at two step sizes for both
   repaired arms in all conditions, and audit complete row/seed/source coverage.
4. Record conditional paired intervals, heuristic verdicts, decision and
   inference-status tables. Update proposition/proof, MathDev audit, PDF and
   active masters/checkpoint. Failed candidates remain archived.

New stage budget: at most 900 cumulative GPU-process seconds and two launches
(second only for localized infrastructure repair); 180 CPU diagnostic/analysis
seconds, with up to60 seconds for references. Focused test and build time is
recorded separately, bounded to five minutes each. No earlier unused budget
transfers. GPU commands require trusted permissions and verified memory growth.
Python is `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; deliberately CPU
checks set CUDA_VISIBLE_DEVICES=-1. Run with PYTHONPATH=. and fixed thread limits.
Outputs use fresh directories under
`docs/plans/artifacts/ledh-mixture-product-20261018/`; commands, environment,
actual source hashes, seeds, wall time and device/allocator information go in
each manifest. Never overwrite an attempt. CLI and actual commands are recorded
in the stage checkpoint before launch. No data download or environment mutation.

## Skeptical pre-execution audit

PASS for bounded implementation and comparison. The failed fitted score study
motivates a specific omitted-variance repair; it does not justify larger KDM
capacity or use old final fits as confirmation. Reference target is explicitly
the seven-mixture model. Fitting loss cannot promote score quality. Comparator
clouds, beta, counts, observations and numerical controls match. Both original
and repaired consumers enter the same make_filter call chain, with analytical
derivatives and unchanged physical importance weights. Known datasets limit
claims to conditional fresh Monte Carlo replication. Required stop conditions,
paired uncertainty and simple adversaries are specified before launch.

Pre-mortem: the run could pass mechanically while product moments worsen
scores because the current-observation Gaussian is poor, smoothing information
misaligns with the reset, or kernel tails are inaccurate. Finite-program
derivatives, actual wiring tests, exact-message controls and independent
references distinguish implementation failures from candidate failure. None
of these alone proves the mathematical model score equals the finite estimate.
