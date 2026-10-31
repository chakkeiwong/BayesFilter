# Appendix B exact mixture-Kalman reference

The KDM paper Appendix B equations (20)--(22), local text lines 935--978,
define linear-Gaussian dynamics and a two-component linear-Gaussian observation
likelihood. The released pinned repository search found no implementation of
this Appendix B mixture-Kalman experiment. Its toy_problem_dataset.py is the
different nonlinear Cappe model. Exact source coefficients, perturbations,
initial distribution/action generation, optimizer schedule and training seeds
remain unresolved. Do not silently supply those values for a replication claim.

This phase implements the exact posterior for arbitrary explicitly supplied
scalar coefficients and a fixed Gaussian initial state. It is a reference and
gradient control, not a learned filter, training replication or new default.

## Derivation and target

For each binary observation-component path z[1:H], use a conditional Kalman
recursion. Predict m-=A*m+B*a, P-=A^2*P+sigma^2. For component z, the observation
residual is e=y-C_z*m--c_z and S=C_z^2*P-+gamma^2. The posterior is Gaussian with
m=m-+P-*C_z*e/S and P=P-*gamma^2/S. Add log(w_z)+log Normal(e;0,S) to the path's
log weight. Start every enumerated complete path with zero log weight; it is
weighted by its full product of mixture probabilities during recursion. After
H steps logsumexp of the 2^H path weights is log p(y[1:H]|a[1:H]). Normalize
those weights once to obtain the exact terminal mixture. Each complete path
occurs exactly once. Do not normalize duplicated prefixes as distinct priors.

The paper has w1=1/(1+exp(v)), w2=1/(1+exp(-v)); use log-sigmoid(-v),
log-sigmoid(v), respectively. Terminal true-state loss is the negative log
density of the exact mixture, averaged over the data batch, as Appendix B
specifies. Differentiation is through this deterministic exact recursion, so
central finite differences are valid for this reference's parameter gradient.
They are not an oracle for stochastic IWSG resampling gradients.

## Evidence contract, defaults and skeptical review

Question: does the TF GPU/XLA batched exact reference compute this mixture and
its terminal likelihood gradient? Baseline is an independent standard-library
Python reference that dynamically branches and normalizes after each update,
plus ordinary Kalman filtering when the two observation components coincide.
Candidate enumerates paths up front and uses a single native TensorFlow time
loop across all data rows and mixture paths; no per-row map or pfor.

Use H5, batch four, initial N(.1,1.2), sigma=.3, gamma=.4,
theta=[A=.8,B=.4,C1=1,C2=-1.2,c1=.2,c2=-.3,v=.4], explicit fixed observations,
actions and terminal states in the diagnostic. These are constructed controls
chosen to exercise unequal weights, opposite slopes, nonzero offsets/actions
and heterogeneous data. They are NOT inferred author defaults. The collapsed
control sets C1=C2=1,c1=c2=.2. No learning or parameter selection occurs.

Primary pass: terminal evidence, density, mixture weights, means and variances
agree with the branching reference within 1e-9 absolute + 1e-9 relative;
collapsed-mixture values agree with Kalman; every parameter gradient agrees
with central differences at 1e-5 step within 2e-6 absolute + 2e-6 relative.
Weights sum to one, variances positive, and invalid variances fail closed.
Stable functions trace once. A failed identity or invalid accepted output
vetoes the reference and triggers repair. Missing inputs or budget exhaustion
stop the phase. Timing is descriptive and no training ranking is evaluated.

Heuristic adversaries are the collapsed single-component Kalman model and
direct branching mixture recursion; they are exact structural controls, not
empirical performance competitors. Exercise ambiguous/opposite-slope and
collapsed/unimodal situations separately. This phase makes no heuristic-
dominance claim for learned KDM; that remains an empirical requirement.

Pre-mortem: a complete-path implementation could double-count priors or reverse
the logistic mixture weights while producing finite results. The independently
branching recursion and asymmetric v catch both. Collapse could mask an error,
so it is only a second control. Exponential cost is bounded to H<=10 and is an
explicit limitation, suitable for the paper's H5 experiment only. The assumed
initial Gaussian/action timing remains a hypothesis for the unrecovered source
experiment; it is a fully specified target for this reference check.

Skeptical audit PASS for a reference-control implementation with these explicit
limits. Maximum two attempts and 120 GPU seconds total, charged within the
existing campaign/allocation. Execute after the current long Figure 4 job or
within its remaining allocation without concurrent numerical GPU jobs. Preserve
source, command, checks, fixture, device, memory-growth policy and summary in a
versioned attempt directory. No external messaging or new package environment.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase k1_mixture_kalman
