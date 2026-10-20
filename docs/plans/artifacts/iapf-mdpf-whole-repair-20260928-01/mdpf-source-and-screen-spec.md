# MDPF source reconciliation and saved-trial screen

Active R3a, independent of iAPF outcomes. CPU-only independent author-reference
analysis, <=300 wall seconds, <=3 attempts after the documented harness repair review, charged within the parent CPU cap.
No optimizer update, GPU initialization, new training or checkpoint selection.

## Source findings

Appendix B, Eqs. (20)–(22), uses controlled scalar linear-Gaussian dynamics,
a bimodal linear-Gaussian observation likelihood, eight learned parameters
(A,B,C1,C2,c1,c2,w1,w2 constrained by v), fixed sigma/gamma, 1000 length-5
sequences, batch 64, final-time posterior NLL, separate resampling .05 and
posterior .5 bandwidths. Comparators are IWSG, IRG, truncated gradients and
exact mixture Kalman. This is a learning experiment, not one-step gradient
parity alone. Numerical true coefficients, perturbations, schedule and seeds
are not fully specified in the inspected text and have not been recovered.

The pinned author tree b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18 has no identified
Appendix B executable. Searched Python filenames and contents for linear,
Kalman, C1, gamma and bandwidth clues; matched routes do not implement it.
src/datasets/toy_problem_dataset.py:197–247 implements the nonlinear Cappe
model x/2+25x/(1+x^2)+8cos(1.2t), y=x^2/20 with Gaussian noise. It is WRONG
relative to an Appendix B replication claim. No exhaustive absence claim.

The completed bearings trial uses one sensor at (-5,0), concentration 50,
85% von Mises / 15% uniform contamination, bias hardcoded zero despite a
bias-concentration configuration field. Anchors:
src/datasets/bearings_only_dataset.py:12–59; trial attempt17/contract.json
resolved dataset_params. State has x,y,heading, with recorded skip 3.
Source initializes particles from the supplied true initial state plus noise
(kde_particle_filter.py:573–615). Use the identical saved initial cloud for
all baselines; no fresh latent initial information is introduced.
The source raw metric stores dx^2+dy^2+wrapped_dtheta^2. Reported RMSE is
mean_i sqrt(mean_t raw_metric[i,t]), not mean pointwise distance or
position-only RMSE.
Particle heading uses a circular mean (particle_transformer.py:112–140);
loss_functions.py:235–348 defines the squared error and square root;
full_sequence_evaluation_base.py:91–119 performs time averaging before the
square root, then averages trajectories. Retain that source metric and
also report position error with an explicit unit distinction.
Existing checkpoint optimized KDE NLL; no MSE refinement occurred.

## Frozen saved-trial screen and research intent

Question: does the one completed source NLL trial show a basic conditional
point-estimation failure against cheap, information-matched heuristics?
1. Initial-cloud persistence: no motion forecast; tests whether updates help.
2. Constant heading/speed: initial circular heading and training median
   displacement per observed step; tests simple motion exploitation.
3. Training marginal center/heading: fixed training means, no test fitting;
   tests whether the learned tracker beats ignoring observations.
These three are a cheap first screen. Fitted robust EKF/UKF and BPF density
comparisons remain required before closing the full heuristic gate.
No two-sensor triangulation applies.

Use all saved 5000 T=150 trajectories and saved particle/weight outputs.
Freeze heuristic parameters only from attempt16/datasets/training.pt.
Conditional situations: early 1–17, middle 18–50, late 51–150, and observation
residual within/outside the training 90th percentile of absolute wrapped bearing
error relative to truth. The latter is a diagnostic stratum, not a classifier
available to the predictor, and does not identify realized contamination labels.
Output time and trajectory means; do not pool away conditional failures.

Primary pass criterion is numerical validity and reproduced source per-step
RMSE within 1e-4 absolute error. Missing/mismatched metric/dataset, nonfinite
predictions, failed source metric identity or mismatched checkpoint hash veto
interpretation. Persist per-trajectory conditional errors and aggregates.
An observed heuristic loss is a point-estimation promotion veto, but differences
are DESCRIPTIVE: this is a post hoc screen of one training seed, no prespecified
inferential ranking. Passing cannot establish NLL quality or full heuristic
dominance. Test data must not tune any repair or select checkpoints.
The original NLL and MSE objectives must remain distinguished.

Expected failure: finite short-trained trajectory model drifts beyond 17 steps.
Repair trigger: conditional underperformance nominates later NLL/MSE and
classical-baseline work on fresh tuning data. It does not reject IWSG math.
Continuation veto: corrupt source outputs or invalid metric reproduction.
No conclusions about independent training-seed variation, Appendix B,
paper's 11-run comparisons, TF GPU parity or HMC.

## Skeptical pre-run review

PASS for this explicitly limited saved-output screen. Correct sensor geometry,
source metric, initialization information and training-only parameter fitting
are specified. Source NLL fit versus point-error screen is labeled. No proxy
can promote the model; source/Appendix gaps remain visible. The stronger
classical density baselines are outstanding rather than silently omitted.

## Localized harness repair review

Attempt03 failed before import: malformed newline literal; corrected and
preflight AST parsing added. Attempt04 failed the intended source identity
check because this diagnostic misread raw_rmse.pt as already square-rooted.
Source call-chain inspection localized the mismatch: save_data retains MSE,
get_value takes the square root of each trajectory's mean MSE, then averages.
No heuristic result was inspected or selected. Correct the diagnostic to this
source aggregation, preserving the 1e-4 pointwise norm tolerance and all data,
baselines, strata and nonclaims. This repairs the harness, not the model.

Skeptical review PASS for one additional localized retry, raising this phase
from two to three attempts within the unchanged 12-attempt tranche and 300
CPU-second phase cap. The two failed attempts consumed 2.539602756 seconds.
This is permitted by the active campaign repair policy; it changes no method,
scientific criterion, data, hardware, privacy boundary or total budget.
Preserve attempts03/04. A source-identity failure is a continuation veto for
interpretation until repaired; it is not evidence against MDPF.
