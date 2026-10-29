# Nonlinear structural Model B: fresh-data confirmation

## Question and scope

Test the owner's model, exactly:
m_t = rho m_(t-1) + sigma epsilon_t;
k_t = 0.55 k_(t-1) + beta tanh(m_t);
y_t = m_t + k_t + 0.30 eta_t.
Truth is (rho,sigma,beta)=(0.70,0.25,0.80). State dimension is two,
innovation dimension one, and observation dimension one. The purpose is to
test structural LEDH likelihoods and scores and their use in surrogate-force
HMC parameter inference. These equations are already Model B in the completed
23 October campaign; this run is a fresh-data confirmation, not a new model.

The equations do not specify an initial law or estimation scope. Preserve the
existing declared choices: independent m_0~N(0,0.25), k_0~N(0,0.20), where
0.25 and 0.20 are variances; estimate rho,sigma,beta and fix 0.55 and 0.30.
Use T=50, with T=2 for independent quadrature checks. Retain q=(atanh(rho),
log(sigma),log(beta)) and the existing explicitly synthetic normal priors
q~N((0.5,log(0.4),log(0.7)),diag(0.8^2,0.7^2,0.6^2)). These are modeling
assumptions, not implications of the displayed equations or literature priors.

## Research intent and evidence contract

Candidate: shared innovation-space LEDH, FP32, 4032 particles, frozen controls
and conditional FP64 reset repair, followed by target-specific NeuTra and the
shared adaptive HMC controller. Complete k deterministically for each sampled
innovation; never add artificial process noise to k or factor a full-rank Q.
For a selected parent j and innovation e, the importance weight is
phi(e) g(y|C(x_j,e)) / q(e|j). The parent selection probabilities cancel.
All three parameter sensitivities propagate together through the recurrence.

Comparators: refined Gauss-Hermite integration at T=2 and independently
replicated bootstrap/Fisher-score references at T=50. Construct cheap
adversaries: bootstrap filtering (direct simulation of the specified law), EKF
(local linearization), and cubature Gaussian filtering (nonlinear moment
approximation). Report them separately at truth, positive and negative local
parameter perturbations, and prior center. Their uncertainty and finite-N bias
must be explicit. Do not select controls by which competitor loses.

Primary engineering checks: equation/initial-law equality, structural completion,
analytical score checks and batch parity, finite outputs with all T steps,
fixed-target repeatability, and correct sampler diagnostics in q and physical
parameters. Short reference refinement must be smaller than discrepancies it
is used to interpret. Compare long-reference Monte Carlo uncertainty and
particle-ladder movement before judging LEDH differences. Score components near
zero are reported in absolute units, not unstable percentage differences.

Inference stopping: four chains, adaptive warmup (minimum 2000, maximum 10000,
existing readiness checks); cumulative retained-draw R-hat <=1.01 for every
mapped parameter, at most 20000 draws per chain or remaining budget. Target
acceptance 0.70; report the owner's preferred 0.65-0.75 band. ESS and finite
energy errors are explanatory diagnostics, not stopping criteria. Report truth
coverage of marginal 95% intervals without treating one dataset as a coverage
study. The MH endpoint for B is the frozen finite LEDH likelihood: passing
sampling checks establishes mixing for this approximation, not equality to
the exact nonlinear posterior.

Promotion vetoes: failed structural identity/derivative wiring, invalid values,
nonconvergence at limits, statistically supported material reference discrepancy
or heuristic inferiority. A candidate veto triggers the already budgeted
repair/diagnostic phase; it is not automatically a continuation veto.
Continuation vetoes: corrupted data/artifacts, changed target, invalid
independent reference, exhausted total budget, or an irreparable numerical
failure. No claim of superiority or global default change is planned.

## Execution, budget, and outputs

Use a fresh versioned root under docs/plans/artifacts/
structural-tanh-model-b-20261025-01. Preserve the old campaign untouched.
Reuse its worker/target/training/HMC implementations; add a focused controller
only where required to select Model B and fresh data. Source, data and design
identities must be regenerated for this new run; do not reuse trained weights
with an incompatible target identity.

1. Run focused regressions and exact-equation checks; generate independent
   calibration, validation and final datasets and particle designs.
2. Compute T=2 refined quadrature and T=50 bootstrap references with replicated
   seeds and a particle ladder. Preserve per-replication values and Fisher scores.
3. Check FP32/XLA 4032-particle production calls, all parameter rows, diagnostics,
   compile/runtime and memory. Existing Model B controls are a warm-start
   hypothesis, then calibrated on fresh data before final-data evaluation.
4. Run the existing target-specific NeuTra capacity/learning-rate/seed protocol,
   frozen-map checks and downstream pilot. Training loss alone cannot certify
   an inference method. Reuse procedures, not an unverified learned map.
5. Run adaptive HMC and summarize parameter intervals, R-hat, acceptance, ESS,
   numerical health and independent-reference limits. Refresh next-phase state
   after every worker; a failed candidate must not mark the campaign complete.

The owner authorized 48 GPU hours for Model A/B; recorded use is 11.52154.
This confirmation may use at most 12 additional GPU-worker hours initially,
plus at most two CPU-reference wall hours, within the remaining 36.47846 GPU
hours. All attempts, compilation and failures count. Allow transfer between
confirmation phases within this cap. Budget exhaustion is an incomplete result,
not permission to weaken criteria. No unrelated GPU work or new model search.

## Defaults, pre-mortem and skeptical review

FP32/4032 are owner choices; keep TF32 off and record conditional FP64 reset.
T=50 and initial law/priors come from the previous exact model definition;
prior predictive and short-path checks expose timing/variance mistakes.
Controls from the previous campaign are warm starts, not universal optima.
Learned architecture, optimizer, seed and budget choices retain the existing
target-specific protocol and its limits; fresh target validation is required.
Independent PF has resampling/path-score error: replication estimates Monte
Carlo variability but does not prove its bias is zero. Refine before using it
to accuse LEDH of error. A low R-hat could accompany a biased finite likelihood;
therefore preserve separate engineering, sampling and accuracy conclusions.

Review verdict: scope and mathematical target are coherent, with one important
correction to a naive rerun: the prior campaign's NeuTra map and selection
identity cannot be silently reused on fresh observations. The implementation
review must verify the new controller rebuilds these identities and uses the
existing adaptive sampler and diagnostic map. Execute reference and focused
checks first; record any controller-specific amendments before GPU training.

## Implementation review and launch record

2026-10-25: the focused controller is
`scripts/run_structural_tanh_confirmation.py`. It calls the existing workers
and shared adaptive HMC implementation. The only shared-worker changes are
optional preparation/reference scope and data-seed offset, binding this new
controller into source provenance, and recording already-computed step counts
and reset fallbacks in probes. Both probe repetitions must be valid and equal.
Old campaign files are untouched. All six datasets use a +200000 seed offset;
the final target's data, particle design, controls, prior, reset policy and
source identities are checked before training and HMC.

Skeptical audit: no wrong full-rank process model, artificial noise on k, reused
old learned map, stale identity, extra posterior validation stage, ESS stopping
criterion, or energy-error cutoff is introduced. Calibration loss nominates
controls but is not statistical evidence of superiority. Width/rate pairs
(18,0.001) and (36,0.0003), each at seeds 17 and 29, receive 256 accepted
updates; the nominated candidate extends to 1024 total accepted updates. This
limited paired capacity/rate search is inherited from this exact model, not
an architecture-optimality claim. Downstream HMC is the utility check.

The discrepancy table reports signed value/score errors, reference MCSE and
particle-ladder movement. Four MCSE plus ladder movement plus 1% of value
magnitude (score vector norm for scores, with unit floors) is a conservative
review flag only, not an exactness certificate or sampler stop. Conditional
heuristic errors are constructed for every probe; superiority remains unresolved
without replicated LEDH uncertainty. No candidate or algorithm default is
promoted by this run. This explicitly limits the promotion veto to promotion.

Focused CPU checks initially passed: 25 tests in 43.62 seconds, covering
completion and full derivatives, batch parity, independent references, reset
repair and restart policy. An additional exact-equation/data-law test and
manifest changes are checked before launch. TensorFlow 2.19.1 sees the intended
RTX 5080 with memory growth, FP32 and TF32 disabled. GPU 1 is excluded.

Exact launch (from `/tmp/BayesFilter-structural-model-ab-20261023`, escalated
for GPU access; all attempts remain inside the 12-hour worker cap):

```bash
/home/chakwong/anaconda3/envs/tf-gpu/bin/python scripts/run_structural_tanh_confirmation.py run --output docs/plans/artifacts/structural-tanh-model-b-20261025-01 --gpu-hours 12
```

Resume uses the same executable and script with `resume --output` and no new
budget. The run manifest records source snapshots/hashes, executable, command,
seeds and artifact locations; each worker records actual environment and wall
time. `state.json` and `next-phase.json` refresh during every worker.

## Execution checkpoint

The final focused numerical suite passed 26 tests in 38.87 seconds. The
controller suite subsequently passed eight tests, including both complete
and incomplete end-to-end orchestration outcomes and a regression for the
reporting repair described in
`structural-tanh-confirmation-reporting-repair-2026-10-25.md`. The repair moves
only diagnostic aggregation to the host after a GPU eager-cast failure; it
does not change filter values, scores, or transitions. Failed-attempt time
remains charged.

All six fresh datasets and independent references were produced. GPU fixed
target repeatability, training setup, and the shared XLA HMC integration
checks passed. The integration map inverse error was 4.44e-16 and its
log-determinant check error was 1.06e-11. These checks establish wiring and
numerical consistency at the tested inputs; they are not posterior mixing
evidence. The campaign subsequently completed on 25 October at 00:20 UTC.
The terminal state is SAMPLING_COMPLETE_ACCURACY_UNRESOLVED. Fresh controls
are 16 flow steps with 10 reset iterations at T=2 and 5 at T=50. Fresh NeuTra
training completed 1,024 accepted updates; four HMC chains completed 2,000
warmup and 1,000 retained draws each. Maximum mapped R-hat is 1.00745;
acceptance is 72.2%, 73.3%, 72.4%, 76.4% (pooled 73.575%). All three truths
lie in their marginal 95% intervals, without establishing repeated-sampling
coverage or exact-model posterior accuracy. Charged GPU-worker time is 6.157
hours of the 12-hour cap. The committed analysis.json and result.md preserve
the full numerical evidence. Documentation closeout is recorded in
structural-tanh-latex-update-2026-10-25.md. No further sampling or separate
validation phase is pending under this completed confirmation.
