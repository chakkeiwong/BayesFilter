# Native pretraining cloud-law diagnostic

Authorized continuation, 13 October 2026. Replication only. This completes a
remaining source-consumer question after the saved-state boundary tests passed.
No algorithm settings or acceptance margins change.

## Evidence contract and intent

Question: does the TensorFlow native measurement-pretraining consumer generate
the same composed cloud distribution as the pinned author consumer? Comparator:
Younis source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, unchanged
WeightModelTrainer.do_forward_pass -> DistributionCloudGenerator.create_cloud.
TF endpoint: SourceWeightPretraining.native_evaluate. Author code runs on CPU
with CUDA hidden; TF runs GPU/XLA FP32 with TF32 off and memory growth.

Source anchors: paper local text D.1/D.2, lines 1584-1610; author
src/trainers/specific_trainers/weight_model_trainer.py lines 24-85 and 104-172;
native individual_training/config.yaml particle_cloud_params. The paper describes
random particle-cloud measurement pretraining; the release specifies two
successive random distributions. Native YAML, not JSON string-key reconstruction,
defines the comparator. Generic sampler law and replay tests already passed and
are not repeated as a substitute for this composed consumer check.

For each Cartesian coordinate, C=X+E with E~N(0,4), and P_j=C+U_j with
U_j~N(0,9). Thus E[(P_j-X)^2]=13 and
E[(P_j-X)(P_k-X)]=4 for distinct particles. Independence gives zero
cross-coordinate and center/conditional covariance. For the angle, both stages
use von Mises concentration k=1/(0.1+1e-8). From the density proportional to
exp(k*cos(theta)), its first circular moment is A=I1(k)/I0(k) by the defining
integrals; independent angle addition has first moment A^2. Conditional pair
angle differences also have cosine mean A^2. Test the sine means as zero.
These are local deductions from the source distribution, not paper claims.

Use unit01's complete saved validation inputs, four independent RNG passes,
each with batches 256/256/256/232. Five sparse true states per sequence give
20,000 cloud centers and 1,000,000 particles per backend. Compute a vector of
moments per cloud; standard errors use independent cloud centers as clusters,
never individual particles for marginal statistics. Fixed conditioning inputs
do not contribute sampling noise to the residual distribution.

Primary diagnostic screen: each theoretical moment and each independent
author-versus-TF moment difference lies within six estimated standard errors
plus 5e-6 (single) / 1e-5 (difference) FP32 arithmetic slack. This conservative
large-sample moment screen detects distribution/wiring errors; it is not a
population equivalence test or a downstream replication acceptance margin.
Predeclare the moments in the diagnostic before launch. Per-pass differences
and native losses are descriptive only; no quality ranking or tuning.

Hard harness/continuation vetoes: wrong source pin, invalid tensors, incomplete
coverage, mutable evaluation state, instrumentation altering the author result,
unexpected shape/retracing, missing artifacts, exhausted budget. A failed
moment screen is a repair/localization trigger, not research-direction rejection.
Do not rerun with new seeds merely to remove a statistical flag. Inspect its
first source/TF divergence. No full training campaign is authorized by this test.

Heuristic/reference set: analytic center law, conditional law, marginal law and
within-cloud covariance distinguish omitted centers, wrong scales, correlated
draws and angle wrapping mistakes. These direct moment oracles answer this
sampler question; an unrelated filter-quality ladder would not. Condition on
Cartesian versus circular coordinates, center versus conditional versus marginal
sampling, and full versus partial consumer batches.

## Defaults, review and budget

Source scales, particle count 50, sparse times and native source precision are
baseline choices from the pinned YAML, checked at setup. Four RNG passes and
six-SE screens are diagnostic convenience choices, not tuning defaults; weak
power against higher moments remains an explicit limitation. Unit01 inputs
cover actual angles and translations but cannot prove every dataset or precision.
The existing generic multi-concentration test supplies separate primitive evidence.

Skeptical audit: PASS. Independent native RNG streams are compared by law rather
than exact draws; common-cloud parity already exists. Cluster uncertainty avoids
pseudoreplication. Actual consumer endpoints are executed; author instrumentation
only retains computed centers/clouds and is checked against an uninstrumented
same-seed invocation. Passing cannot establish final trained-model quality, whole
KDM replication, performance superiority or bitwise independent trajectories.
No native optimizer or source algorithm is changed.

Cap: 1,200 CPU wall seconds and 1,200 GPU wall seconds, at most two attempts
with 600 seconds per attempt, debited from the existing campaign allocation.
Retry only localized infrastructure repair; preserve both attempts. No additional
allocation assumed. Output root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/pretraining-cloud-law-20261013/attemptNN.
The supervisor records source hashes, environment, seeds, exact commands,
elapsed wall time, decision and ordinary run provenance.

Exact trusted command, from /home/chakwong/BayesFilter:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_pretraining_cloud_law.py`.
Result note: docs/plans/controlled-smc-kdm-cloud-law-result-20261013.md.
