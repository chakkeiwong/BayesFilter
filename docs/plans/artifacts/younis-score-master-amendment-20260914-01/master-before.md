# Master program: systematic investigation of KDM, LEDH, and model-score estimators

Date: 2026-09-14

## Purpose

This program investigates how to estimate the observed-data model score

\[
 S(\theta;y_{1:T})=\nabla_\theta\log p_\theta(y_{1:T})
\]

when a particle filter uses LEDH, KDM, GenUT, UKF, continuous resampling, or
optimal transport. The program is deliberately organised around the target
quantity. It does not treat a differentiable finite filter, a KDM expectation,
an unnormalised likelihood derivative, and the marginal score as interchangeable
objects.

The central research question is:

> Which proposal, score identity, resampling representation, and variance-
> reduction method gives the smallest scientifically relevant error for the
> model score at a declared compute budget, while remaining valid for the
> transition-support class of the model?

The program has two non-interchangeable tracks:

1. **Regular-transition track.** Transitions admit densities on a common
   reference measure. This is the setting in which model-corrected mixture
   proposals, Fisher identities, backward pair estimates, Rao--Blackwellization,
   and Younis-style KDM calculus can be tested.
2. **Degenerate-transition track.** Transitions are deterministic or singular
   conditional on an ancestor, as in the DSGE target. Ordinary ambient-space
   Gaussian-mixture density ratios and regular-transition smoothing arguments
   are not assumed valid. A support-aware score identity must be derived before
   implementation or comparison.

No result from the regular track may be presented as evidence that the
degenerate track is solved.

## Project implementation constraints

The master program inherits the repository's active implementation contracts.
The claim-bearing LEDH baseline uses Contract E--Chol and its declared
analytical total-derivative composition. Autodiff, finite-program JVPs, and
GradientTape may be used for parity and debugging, but they are not a
claim-bearing score route. Active transport rows use the repository's exact
divisor chunk policy, and TensorFlow/TFP with the configured GPU, TF32, XLA,
and memory-growth settings is the default execution target. NumPy is confined
to independent references, tests, and post-run diagnostics. A candidate that
violates one of these contracts is either repaired or labelled diagnostic; it
does not enter the score leaderboard by changing a CLI flag.

These engineering constraints do not decide the scientific question. A route
can satisfy the implementation contract and still estimate the wrong target or
lose to a simple heuristic. Conversely, a mathematically interesting reference
route can remain diagnostic until its backend and derivative contracts are
implemented.

## Program architecture

The implementation should be a typed experiment system with one coordinator
and separate scientific components. This keeps a KDM expectation gradient from
being combined with a marginal score without detecting the target change.

### Registries

The coordinator reads five repository-owned registries:

1. **Model registry:** model equations, data generator, support class,
   parameterisation, initial law, transition representation, observation law,
   and oracle provider.
2. **Target registry:** exact marginal score, finite-program derivative, KDM
   expectation gradient, unnormalised derivative, or another explicitly named
   target. Each target declares its measure, normalisation, and valid support
   class.
3. **Proposal registry:** bootstrap, adapted, LEDH, KDM, forward/backward, OT,
   twisted PF, iAPF, SGQF-guided, exploration mixture, or coordinate proposal.
   Each proposal exposes both its sampler and its evaluable density or mass
   correction.
4. **Estimator registry:** Fisher, backward-pair, IWSG, pathwise, hybrid,
   Rao--Blackwell, ratio, or control-variate estimator. Each estimator declares
   the quantities it requires and the target it returns.
5. **Tuning registry:** scope identity, calibration partitions, candidate
   controls, selected controls, and the untouched claim partition.

Registry validation must reject an experiment before tracing or sampling when a
proposal, estimator, target, and support class are incompatible. In particular,
an ambient Gaussian KDM proposal cannot be registered for a degenerate model
under a regular-transition score identity.

### Matrix generator

The matrix generator expands a declarative study into rows over model, regime,
horizon, particle count, proposal, score estimator, variance-reduction method,
dtype/backend, and seed. It first generates the baseline ladder, then all
single-factor and pairwise interactions, and finally the full factorial of
surviving candidates. This gives exhaustive coverage without allowing an
unvalidated option to acquire promotion status merely because it appeared in a
large pooled sweep.

Each row is immutable after tuning. The claim runner consumes a repository-issued
tuning artifact and refuses a missing, stale, cross-model, or cross-horizon
artifact. Calibration and validation runs never write into the claim directory.

### Execution lanes

Use distinct lanes for:

- oracle construction;
- fixed-cloud estimator comparison;
- full-filter proposal comparison;
- long-horizon and particle-count scaling;
- degenerate support derivation and fixtures; and
- report assembly.

The fixed-cloud lane isolates score-estimator variance. The full-filter lane
measures the combined effect of proposal, resampling, and score recursion. Their
outputs must not be merged into one score table without a lane label.

### Coordinator pseudocode

```text
load model_registry, target_registry, proposal_registry,
     estimator_registry, tuning_registry
validate_registries_and_supports()

for phase in dependency_order:
    matrix = generate_rows(phase, surviving_candidates, declared_scopes)
    for row in matrix:
        validate_target_measure_support(row)
        if phase.needs_tuning:
            tuning = tune_on_calibration_partition(row)
            validate_tuning_scope(tuning, row)
        for dataset in row.claim_datasets:
            oracle = oracle_registry[row.model].get(dataset, row.parameter)
            for seed in row.claim_seeds:
                result = execute_one_row(row, dataset, seed, oracle)
                write_immutable_result(result)
    decision = aggregate_with_paired_uncertainty(phase.results)
    apply_vetoes_and_heuristic_gate(decision)
    write_phase_note_and_checkpoint(decision)
    surviving_candidates = decision.viable_candidates
```

The coordinator may schedule independent rows in parallel, but each row must
retain the exact command, environment, device, random seed, tuning scope, and
source revision needed to reproduce it. Parallel scheduling must not change
random-number coupling within a paired comparison.

### Result schema

Every row writes machine-readable JSON plus a human-readable Markdown note with
at least:

```text
program_id, phase_id, row_id, git_commit
model_id, dataset_id, regime_id, parameter_id, horizon, particle_count
support_class, measure_id, target_id, score_kind, normalization_kind
proposal_id, resampling_id, estimator_id, variance_reduction_id
reset_id, derivative_id, tuning_scope_id, tuning_artifact_hash
dtype, backend, jit_compile, device, memory_policy, chunk_policy
seed, paired_seed_group, wall_time, peak_memory
oracle_id, oracle_status, score_error, value_error, diagnostics
hard_veto_status, heuristic_verdict, inference_status, artifact_paths
```

The aggregation layer computes paired confidence intervals, bias/variance/MSE
decompositions, and the decision table. It may produce plots and rankings only
after hard vetoes and target compatibility have passed.

### Factorial axes

The study matrix is the Cartesian product of the following axes, subject to the
support and phase gates:

| Axis | Levels to enumerate |
|---|---|
| Target | exact marginal score; finite-program derivative; KDM expectation gradient; unnormalised value/derivative; declared approximation |
| Proposal | bootstrap transition; EKF; UKF; LEDH; physical transition mixture; KDM; forward/backward mixture; twisted PF; iAPF; SGQF-guided; defensive mixture; support-aware coordinate proposal |
| Resampling/coupling | multinomial/systematic discrete resampling; continuous KDM mixture draw; OT coupling with preserved density; OT cloud plus explicit jitter; OT as a common-random-number coupling only |
| Covariance | model (Q); UKF covariance; KDM weighted covariance; LEDH local covariance; GenUT-restored covariance; SGQF projected covariance; calibrated hybrid choices |
| Score identity | finite scalar derivative; forward Fisher; backward pair/Fisher; Nemeth KDE/Rao--Blackwell; PaRIS; coupled-CPF debiasing; IWSG; pathwise; hybrid; ratio of unnormalised estimates; iterated-filtering sensitivity |
| Variance reduction | none; Rao--Blackwell; exact-mean control variate; reference-plus-residual; randomized QMC; coupled conditional-particle debiasing; randomized multilevel |
| Support | regular common-density; bounded/constraint regular; deterministic map; singular ancestor mixture; unresolved DSGE support |
| Evaluation lane | fixed-cloud estimator; full filtering recursion; long-horizon scaling; support derivation |

The generator records why each combination is omitted. “Not applicable” and
“not yet derived” are distinct statuses; silently dropping an incompatible row
would make the final option inventory incomplete.

## Non-negotiable target definitions

Every implementation and result must label which of the following it computes.

### Exact model score

\[
 Z(\theta)=p_\theta(y_{1:T}),\qquad
 S(\theta)=\nabla_\theta\log Z(\theta).
\]

For a regular state-space model, Fisher's identity is

\[
 S(\theta)=\mathbb E_\theta\left[
 \nabla_\theta\log \mu_\theta(X_0)+
 \sum_{t=1}^T\{
 \nabla_\theta\log f_\theta(X_t\mid X_{t-1})+
 \nabla_\theta\log g_\theta(y_t\mid X_t)\}
 \mid y_{1:T}\right].
\]

The identity requires the stated support and differentiation conditions. For a
singular transition, its density notation is not used until a valid
ancestor/innovation base measure has been supplied.

### Finite-program derivative

\[
 L_N(\theta;\xi)=\text{the scalar produced by one declared finite program},
 \qquad
 S_N^{\mathrm{fin}}=\nabla_\theta L_N.
\]

This derivative is a valid derivative of the finite program only when all
declared dependencies are differentiated. It is not automatically an estimate
of \(S(\theta)\).

### KDM expectation gradient

For a mixture \(m_\theta\) and integrand \(F_\theta\),

\[
 \mathcal J(\theta)=\int F_\theta(z)m_\theta(z)\,dz.
\]

With \(q_0=m_{\theta_0}\) held fixed while differentiating,

\[
 \mathbb E_{q_0}\left[
 \nabla_\theta F_\theta(Z)|_{\theta_0}
 +F_{\theta_0}(Z)\nabla_\theta\log m_\theta(Z)|_{\theta_0}
 \right]
 =\nabla_\theta\mathcal J(\theta)|_{\theta_0}.
\]

This is the scope of the Younis IWSG identity. If \(F=\log g(y\mid z)\),
the target is an expected conditional log likelihood. In general,

\[
 \mathbb E_m[\log g(y\mid Z)]
 \neq \log\mathbb E_m[g(y\mid Z)].
\]

### Unnormalised pair and normalised score

The program must retain separately

\[
 \widehat Z_N,\qquad \widehat D_N\approx\nabla Z,
 \qquad \widehat S_N=\widehat D_N/\widehat Z_N.
\]

Unbiasedness of the first two does not imply unbiasedness of the third.

## Evidence contract

The primary comparison question is conditional model-score accuracy for a fixed
dataset and parameter, averaged over independent particle randomisations. The
exact comparator is an analytic or numerically certified score oracle. When no
oracle exists, the result is restricted to identity checks, convergence
evidence, and descriptive comparisons; it is not called unbiased or accurate
without a derivation.

The primary promotion criterion is a predeclared paired error measure against
the oracle, such as squared Euclidean score error and a scale-normalised version,
with a confidence interval across independent datasets and particle seeds.
Required vetoes are:

- non-finite values, invalid supports, negative or zero proposal densities where
  the numerator has mass, failed derivative identities, or failed likelihood
  normalisation checks;
- a mismatch between the implemented target and the claimed target;
- missing initial-law, transition, observation, mixture, Jacobian, or total
  derivative terms;
- use of a regular-transition identity on a degenerate transition without a
  support derivation;
- data leakage from tuning or control-variate fitting into the claim run; and
- a failed heuristic-dominance check in any salient regime.

Explanatory diagnostics include bias/variance decomposition, ESS, weight tails,
ancestor entropy, mixture overlap, bandwidth sensitivity, covariance calibration,
OT marginal error, and runtime. They cannot replace the primary criterion.

Passing this program does not establish posterior correctness for an untested
model, HMC readiness, asymptotic unbiasedness, or superiority outside the
declared scope.

## Campaign envelope and stop rules

“Budget and time are not a problem” permits exhaustive coverage, but it does
not remove the need for a finite, reproducible campaign definition. The initial
claiming envelope is:

- at least 100 independent observation datasets per regular model/regime;
- 64 independent particle randomisations per dataset for the first claim run,
  with a maximum of 256 if the paired interval remains too wide;
- particle counts (N\in\{32,64,128,256,512,1024,2048\}) and horizons
  (T\in\{1,5,20,50,100\}) where the model supports them;
- all baseline and identity rows, all single-factor ablations, and the complete
  factorial only for candidates that pass the preceding phase gates; and
- no replacement of a failed artifact: each repair or retry gets a new,
  versioned output directory.

These values are an explicit planning envelope, not inherited scientific
defaults. A row may stop early when a hard veto fires, when its target identity
fails, or when an implementation failure has been reproduced three times with
the same cause. A viable row may stop replication when its predeclared paired
interval is sufficiently narrow for the decision question; otherwise it is
reported as descriptive-only at the maximum replication count. A candidate
that fails a promotion criterion but has a planned repair continues to that
repair; it is not silently discarded as evidence against the whole method.

The campaign root is a unique versioned directory under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`. Every launch records
the exact matrix manifest, git revision, environment, hardware, seeds, command,
wall time, and output hashes. The campaign is closed only after each planned
row is classified as promoted, viable-but-underranked, failed-with-repair,
failed-support, or unresolved-evidence.

## Skeptical pre-mortem

Assume the program produces an apparently favourable KDM or LEDH result. It
could still be misleading for the following reasons:

- the implementation differentiates the finite cloud while the claim is about
  the marginal likelihood; the earliest check is target metadata plus the
  analytic linear-Gaussian score;
- the KDM bandwidth or OT reset changes the propagated measure and the apparent
  gain is a target change; the earliest check is a physical-numerator,
  evaluable-denominator identity test;
- the score error is dominated by a random denominator, so a lower variance
  expectation gradient is irrelevant to the reported score; the earliest check
  is the separate \(\widehat Z_N\), \(\widehat D_N\), and ratio study;
- a missing initial or transition derivative is hidden by a fixed-cloud finite
  difference; the earliest check is a parameter-dependent initialization test
  and an independent Fisher identity;
- tuning or control-variate fitting leaks information from the claim set; the
  earliest check is a scope-bound manifest with disjoint data hashes;
- a complex route loses to a cheap UKF, adapted PF, or bootstrap method in a
  salient regime while its pooled mean looks favourable; the earliest check is
  the conditional heuristic table; and
- a regular-transition success is reported as a DSGE result; the earliest check
  is the support-class field and a mandatory derivation reference for every
  degenerate run.

Any one of these findings blocks promotion until the corresponding repair is
completed. A failed candidate remains useful evidence about that candidate; it
does not by itself terminate the broader program.

## Option matrix

Every candidate is implemented as a complete estimator with a target label.

| Family | Candidate | Intended role | Main question |
|---|---|---|---|
| Baseline | Bootstrap PF with complete Fisher-score recursion | classical reference | How much error comes from ordinary particle approximation? |
| Baseline | Locally adapted proposal with exact importance correction | tuned classical reference | Does simple observation adaptation explain any gain? |
| Baseline | EKF and UKF approximate-likelihood scores | cheap heuristic adversaries | Is low-order Gaussian moment information sufficient? |
| LEDH | Canonical LEDH-PF-PF-OT with analytical recursive score | project baseline | What does the existing finite algorithm achieve? |
| KDM | KDM proposal with physical transition numerator and evaluable proposal denominator | model-corrected proposal | Does KDM improve coverage without changing the target? |
| KDM | KDM expectation/IWSG gradient | declared expectation objective | What is its variance and bias for its own objective? |
| KDM/Rao--Blackwell | Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and observed information | direct regular-model score comparator | Does the published linear-cost kernel estimator improve horizon variance under its bandwidth and mixing assumptions? |
| Covariance | UKF per-particle covariance, KDM weighted covariance, and hybrid covariance | proposal-shape study | Does a better covariance estimate improve the score through proposal quality, or only change a finite approximation? |
| Fisher | Forward particle score with complete-data terms | direct score estimator | Can transition and observation score terms be propagated stably? |
| Smoothing | Backward pair/Fisher estimator | regular-transition only | Does ancestor averaging reduce score variance or finite-\(N\) bias? |
| Smoothing | Fearnhead--Wyncoll--Tawn sequential smoother | regular-transition only | How do linear- and quadratic-cost additive-functional smoothers compare? |
| Smoothing | PaRIS with explicit backward-draw count | regular-transition only | Does \(\widetilde N\ge2\) control path-degeneracy variance at the target horizon? |
| Debiasing | Coupled conditional particle filter with Rhee--Glynn correction | regular tractable-transition only | Can unbiased smoothing expectations and Fisher scores be obtained at acceptable random cost? |
| Rao--Blackwell | Analytically integrate tractable state blocks or ancestor indices | variance reduction | Which conditional integrations are exact and worthwhile? |
| Control variate | Exact-mean control variate for unnormalised derivative | variance reduction | Does known centering reduce variance while preserving expectation? |
| Control variate | Tractable reference plus residual | variance reduction | Does a separately corrected approximate model help? |
| Hybrid | Pathwise plus importance-weight gradient | gradient variance candidate | Can the two estimators be combined without changing the target? |
| Combination | Calibrated linear combination of two score estimators | bias--variance study | When do two biased but target-matched estimators have lower MSE? |
| Multilevel | Coupled \(N\)/bandwidth or randomized telescoping estimator | bias reduction | Is there a verified expansion supporting debiasing? |
| Proposal correction | Forward/backward mixture proposal with generative correction | proposal design | Can future-data coverage help while preserving the model measure? |
| Continuous likelihood | Malik--Pitt/DeJong-style continuous particle likelihood | continuous-resampling comparator | Which discontinuity is removed, and what finite-particle measure and derivative does the method target? |
| Simulation-only sensitivity | Ionides iterated filtering | unavailable-transition-density comparator | What bias, variance, and mixing cost arise as artificial parameter noise decreases? |
| Iterated APF | Guarniero--Johansen--Lee twisted/auxiliary filter | proposal-quality comparator | Does future-data twisting reduce normalizer and downstream score variance when the analytical score recursion is held fixed? |
| Proposal variance | Whiteley--Lee twisted particle filter | proposal-only variance control | Can normalizer variance be reduced without changing the physical target correction? |
| Proposal moments | Fixed structured Gaussian quadrature filter (SGQF) moments | high-dimensional proposal-design comparator | Do sparse-grid projected moments improve proposal coverage at an admissible cost without replacing the physical target? |
| Directional validation | Three-point, five-point fourth-order, and eleven-point cubic finite differences | oracle-calibrated score diagnostic | Which fixed positive \(h\)-ladder gives the best bias--variance tradeoff for each directional derivative? |
| Discretization | Jasra--Kamatani--Law--Zhou multilevel particle filter | coupled-level variance control | Does a score-level telescope exist for the chosen discretization and coupling? |
| Degenerate | Ancestor/innovation-coordinate estimator | unresolved support-aware route | Can a valid base measure and score identity be derived for DSGE? |

## Phase 0: specification and oracle ladder

### 0.1 Build a model catalogue

Use a minimum ladder of:

1. scalar and multivariate linear-Gaussian state-space models with exact Kalman
   value and score;
2. nonlinear regular-transition models with high-accuracy reference scores;
3. models with multimodal or strongly curved posteriors;
4. the existing repository models used by the LEDH adapters; and
5. the degenerate DSGE model, kept in a separate support track.

Record state dimension, parameter dimension, horizon, transition rank,
observation rank, support, analytic oracle availability, and the model's
parameter-dependent initial law.

### 0.2 Construct the oracle ladder

For each model, implement in order:

1. closed-form value and score where available;
2. deterministic quadrature or Rao--Blackwellised integration;
3. high-particle, replicated reference estimates with an error certificate;
4. an independent implementation using different code and random numbers; and
5. support-specific reference calculations for degenerate models.

An oracle is admitted only after value, directional derivative, finite-difference,
and parameter-coordinate checks agree within predeclared tolerances.

### 0.3 Freeze notation and target metadata

Every result records `target_id`, `measure_id`, `support_class`, `score_kind`,
`normalization_kind`, `proposal_id`, `reset_id`, `derivative_id`, and whether the
proposal is detached locally. A result with missing metadata is diagnostic only.

## Phase 1: identity and call-chain verification

Before measuring quality, verify each candidate against the scalar it claims to
differentiate.

1. Compare analytical derivatives with finite differences of the same fixed-
   random-design scalar.
2. Check all initial, transition, observation, mixture-weight, covariance,
   bandwidth, Jacobian, OT, and GenUT dependencies.
3. Verify that the claim-bearing endpoint calls the general implementation and
   has the required rank, batch, device, and dtype contracts.
4. For IWSG, test the proposition directly on a finite Gaussian mixture with
   known integrals and compare the estimator with the gradient of
   \(\mathcal J\), not with a marginal likelihood unless those are analytically
   identical.
5. For each model-corrected proposal, numerically verify that the proposal
   denominator is the actual density used to generate particles and that the
   numerator contains the physical model transition and observation factors.
6. For degenerate models, fail closed until the coordinate/base-measure identity
   is proved and tested.

## Phase 2: proposal and representation study

Hold the score estimator fixed and vary only the proposal representation. The
factorial factors are proposal family, lookahead construction, covariance
source, particle count, horizon, observation regime, and compute budget:

1. bootstrap transition proposal;
2. UKF/EKF adapted proposal;
3. LEDH flow proposal;
4. KDM with explicit component covariances;
5. OT-resampled KDM cloud;
6. forward/backward mixture proposal;
7. mixtures with a declared support-preserving exploration component; and
8. the Guarniero--Johansen--Lee iterated auxiliary particle filter (iAPF),
   where its learned \(\psi\)-twisting is used only to construct an auxiliary
   proposal.

The proposal family is crossed with the following covariance sources: physical
\(P^Q\), per-particle UKF, KDM weighted-mixture, LEDH/GenUT, SGQF projected,
and a calibrated convex hybrid. SGQF is included as a proposal-design arm
because the repository already exposes \texttt{tf\_fixed\_sgqf\_cloud} and
\texttt{tf\_fixed\_sgqf\_filter}; the current implementation is an
eager/Python-branch approximate lane, so it is not automatically a
claim-bearing high-dimensional filter. Its sparse-grid point count, memory,
moment positive-definiteness, and dimension/level scaling are recorded before
any score comparison. The SGQF moments parameterise \(q_t\), while the
physical transition and observation factors remain in the numerator and the
actual SGQF-induced proposal density remains in the denominator.

Twisted PF and iAPF are crossed as none, fixed one-step \(\psi\), and
iteratively fitted \(\psi\). The fitted lookahead is produced on a disjoint
pilot/calibration partition and frozen before validation. A twisted or iAPF
arm must report the ancestor law, state proposal, correction factor, support,
and whether the proposal density is joint or marginalized. It is not allowed
to substitute an approximate future likelihood for the physical target.

For regular transitions, compare both ordinary PF weights and full model-
corrected mixture weights. This separates proposal quality from target changes.
Measure conditional ESS, weight variance, coverage of high-posterior regions,
and score error at equal particle count and equal wall-clock cost.

The minimum proposal test set is: (i) linear-Gaussian/Kalman normalizer and
score with an exact model-correction identity; (ii) a nonlinear
regular-transition model with mode or tail stress; and (iii) a high-dimensional
SGQF scaling ladder. For every row, test proposal normalisation, support,
finite weights, target-preserving \(\widehat Z_t\), and the call chain from the
claim-bearing endpoint to the declared proposal. Primary proposal metrics are
normalizer MSE/variance, coefficient of variation, ESS and weight tails at
equal compute. Score MSE is secondary until those tests pass. A paired
uncertainty interval is required before a row is called better; otherwise it is
classified as viable or descriptive-only.

The focused proposal tests must cover (a) equality of the sampled proposal
density and the denominator evaluated at the sample, (b) normalization of the
twisted/iAPF correction on a finite discrete auxiliary example, (c) SGQF
cloud-weight and projected-moment invariants, (d) positive-definiteness and
conditioning of every covariance arm, and (e) a linear-Gaussian identity in
which every proposal recovers the same normalizer in expectation. These tests
precede the factorial campaign; a failed identity or support test excludes the
row rather than being averaged into its score result.

The iAPF is a proposal-quality experiment, not a replacement score identity.
Guarniero, Johansen, and Lee define a family of positive future-data twisting
functions \(\psi\), show that the corresponding \(\psi\)-auxiliary particle
filter preserves the marginal-likelihood normalizer after the change-of-measure
correction, and iteratively approximate the zero-variance \(\psi^*\) sequence
from particle output (their Sections 2--3 and Algorithms 2--4). In this
program, an iAPF arm may replace the UKF only at the proposal-construction
interface. The analytical recursive score, physical transition numerator,
proposal denominator, and support declaration remain unchanged. The arm is
therefore admissible only when the required transition integrals, proposal
sampling, and correction are evaluable. An ambient Gaussian implementation is
not a solution for a singular DSGE transition; a disturbance- or
manifold-coordinate version would require a separate support and derivative
derivation.

This is distinct from Ionides iterated filtering. Ionides augments the model
with an artificial parameter random walk and decreases its perturbation scale
to obtain a simulation-only likelihood-derivative approximation. It is a
separate comparator for implicit models, not a UKF replacement and not an
iAPF proposal module.

Twisting is also not synonymous with replacing the UKF. The UKF supplies
local moment geometry for a proposal (and, in canonical LEDH, the per-particle
covariance lifecycle). A twisted filter changes ancestor and transition
sampling through a positive look-ahead function \(\psi\), with a correction
that preserves the declared likelihood normalizer. The two mechanisms can be
combined: UKF/LEDH can provide local proposals while twisting supplies an
auxiliary look-ahead weight. A route that replaces the UKF with a twisted
transition must therefore be registered as a new proposal route and must
prove that its sampling law, normalizing integral, correction, and support are
evaluable. It remains a proposal-variance experiment; it does not change the
analytical score identity.

The OT study must include exact marginal-balance error, coupling entropy,
transport cost, sensitivity to Sinkhorn iterations, and whether the transported
cloud remains compatible with the density used in the importance correction.
OT is never treated as a resampling theorem by itself.

OT has five separate experimental roles that must not be conflated:

1. a coupling for comparing two particle systems with common random numbers;
2. a deterministic cloud rearrangement used only as a proposal heuristic;
3. a continuous resampling law whose density can be evaluated and corrected;
4. an OT cloud followed by an explicit, evaluable jitter kernel; and
5. a model-changing reset whose score is the derivative of a new finite scalar.

Roles 1 and 2 do not by themselves define a model-corrected proposal. Role 3
requires a density and support proof. Role 4 requires the jitter density and
Jacobian in the correction. Role 5 is allowed only under the finite-program
target label and cannot be compared to the exact score as though the reset were
measure-preserving.

The Sen--Thiéry--Jasra coupling belongs specifically to role 1. It couples two
particle-filter replicas while preserving their marginal algorithms, usually to
estimate a difference between parameters, models, or discretization levels.
For score work this can reduce the variance of a finite-difference or
multilevel difference when the replicas are positively correlated. It does not
reduce the variance of one LEDH score by itself, and averaging two positively
coupled copies is not a variance-reduction argument. The Jacob--Lindsten--Schön
coupled conditional-particle construction is a different route: its meeting
and Rhee--Glynn correction can debias a smoothing expectation under regular,
tractable-transition assumptions. Neither coupling construction supplies a
singular-DSGE score identity.

### UKF, KDM, and the covariance question

The UKF already provides a Gaussian moment approximation, but it does not make
the KDM mixture identity true and it does not supply a model-score correction
for an arbitrary nonlinear or singular transition. The covariance study must
therefore compare three distinct objects:

1. UKF covariance used only to construct a proposal;
2. KDM weighted covariance computed from the continuous mixture; and
3. a hybrid proposal that uses UKF moments for local propagation and KDM
   moments for mixture placement.

For each object, hold the physical model numerator and score estimator fixed,
then measure proposal coverage, importance-weight variability, and score error.
This identifies whether a covariance improvement helps because it places
particles better, because it changes the target measure, or because it merely
changes the finite-program derivative. No covariance comparison is interpreted
as evidence that UKF has been replaced as the filter unless the complete
filtering recursion and its target are explicitly changed and audited.

For covariance interpretation, keep the score code and physical numerator
fixed while swapping only the proposal moments. Record eigenvalue margins,
condition numbers, weight tails, and proposal-density agreement. A covariance
arm fails closed if it is not positive definite, if a caller-supplied
covariance does not match the generated density, or if it is used to claim
replacement of the UKF without a complete filtering derivation.

## Phase 3: score-estimator study on a fixed particle law

For each proposal, compare the following estimators using the same particle
cloud whenever possible:

1. derivative of the executed finite value program;
2. unnormalised likelihood derivative divided by the estimated normalizer;
3. complete-data Fisher score;
4. backward pair/Fisher score;
5. Nemeth KDE/Rao--Blackwell score and observed-information estimator;
6. Fearnhead sequential smoothing and PaRIS with an explicit backward-draw
   count;
7. coupled conditional-particle/Rhee--Glynn debiased smoothing score;
8. KDM IWSG gradient of a declared mixture expectation;
9. hybrid pathwise/IWSG gradient;
10. Rao--Blackwellised variants; and
11. exact-mean control-variate variants.

Report vector bias, covariance, MSE, componentwise error, and correlation of
errors across coupled estimators. Do not call a lower MSE an unbiasedness result.

Control variates require a known or separately corrected centering constant.
Estimate coefficients on calibration data only, freeze them, and test whether
the empirical variance reduction survives on untouched data. A fitted baseline
whose centering expectation is unknown is labelled a heuristic variance
reduction, not an unbiased control variate.

### Combining two imperfect estimators

If (A) and (B) estimate the same score (S), a combined estimator

\[
 C_\alpha=\alpha A+(1-\alpha)B
\]

has error

\[
 \mathbb E[C_\alpha-S]
 =\alpha b_A+(1-\alpha)b_B,
\]

and variance

\[
 \operatorname{Var}(C_\alpha)=
 \alpha^2V_A+(1-\alpha)^2V_B
 +2\alpha(1-\alpha)\operatorname{Cov}(A,B).
\]

The program must first establish that (A) and (B) target the same score.
Then estimate (alpha) on independent calibration replications, using a
predeclared loss or an exact control-variate relation, freeze it, and evaluate
on untouched replications. With an oracle, the MSE-optimal scalar coefficient
can be estimated directly. Without an oracle, a variance-minimising coefficient
does not generally minimise MSE because the unknown bias remains. A combination
of two biased estimators is therefore a candidate for variance reduction, not
an automatic debiasing method. The study must report the two component biases,
their covariance, the selected coefficient, and the combined estimator's
untouched error.

The comparison should include:

1. canonical LEDH score plus model-corrected KDM/Fisher score;
2. forward Fisher plus backward-pair score;
3. direct ratio score plus an unnormalised derivative residual; and
4. IWSG plus pathwise gradient only when both have the same declared
   expectation target.

If the component targets differ, the linear combination is a new target and
must be analysed as such. It cannot be described as a better estimate of the
original score merely because its numerical variance is smaller.

## Phase 4: normalisation and long-horizon study

This phase isolates the random-ratio problem.

1. Compare \(\widehat Z_N\), \(\widehat D_N\), and
   \(\widehat D_N/\widehat Z_N\) separately.
2. Vary particle count over a logarithmic ladder and retain paired random
   numbers across methods.
3. Vary horizon while holding model, dimension, and tuning scope fixed.
4. Measure denominator coefficient of variation, reciprocal-weight tails,
   score bias, and variance.
5. Compare direct posterior-score estimators with ratio estimators at equal
   compute.
6. Test coupled and randomized multilevel estimators only after an empirical or
   analytic error expansion has been established.

The conclusion of this phase must state whether an observed error is caused by
wrong derivatives, changed filtering measure, finite Monte Carlo integration,
or the random normalisation.

## Phase 4B: approximate consistency diagnostics

When an exact score oracle is unavailable, the program may study diagnostics
that are plausibly correlated with bias, but it must keep them separate from
the score criterion. Candidate diagnostics include:

- score estimates at (N,2N,4N) under coupled random numbers;
- Richardson-style slopes in (N) or bandwidth, when a fitted expansion is
  supported by the data;
- agreement between direct, Fisher, and finite-program derivatives;
- forward/backward consistency and ancestor-pair residuals;
- likelihood normalisation residuals and denominator stability;
- score directional identities under parameter perturbations; and
- replicate-to-replicate correlation between each diagnostic and measured score
  error on oracle-bearing models.

The diagnostic study must be calibrated on oracle-bearing models first. A
diagnostic can then be used as an explanatory warning or a continuation veto
under a declared threshold, but it cannot be promoted to a proof of low bias
on an oracle-free model merely because its correlation is high elsewhere. The
program must report the calibration model, held-out model, correlation
uncertainty, and the strongest counterexample.

Central finite differences of a particle log-likelihood are useful for this
diagnostic, but they are not automatically a bias estimate. Writing
\(\ell(\theta)=\log Z(\theta)\) and
\(b_N(\theta)=\mathbb E[\log \widehat Z_N(\theta)]-\ell(\theta)\),

\[
\mathbb E[\widehat s_{\varepsilon,N}]-\ell'(\theta)
=
\underbrace{\frac{\ell(\theta+\varepsilon)-\ell(\theta-\varepsilon)}
 {2\varepsilon}-\ell'(\theta)}_{O(\varepsilon^2)\text{ under smoothness}}
+
\frac{b_N(\theta+\varepsilon)-b_N(\theta-\varepsilon)}{2\varepsilon}.
\]

Common random numbers or Sen-style coupled resampling reduce the variance of
the difference, but they do not remove the log-normalisation bias, finite-
particle bias, or finite-difference truncation bias. If the same fixed random
design contains discrete resampling, the \(\varepsilon\to0\) limit may instead
differentiate a discontinuous finite program. Richardson extrapolation is
allowed only after an observed \(\varepsilon^2\) scaling law has been checked.

A symmetric local-polynomial fit is a useful multi-point version of this
diagnostic, but the number of points is not itself an accuracy guarantee. For
points \(\theta+k h\), the derivative is the slope of the fitted polynomial at
zero. With a symmetric stencil, a linear fit has
\[
 \widehat s_h=s+\frac{\ell^{(3)}(\theta)}{6}
   \frac{\sum_k k^4}{\sum_k k^2}h^2+O(h^4),
\]
and a quadratic fit has the same leading order because the quadratic term is
even and is orthogonal to the slope on the symmetric design. To cancel the
third-derivative term one must fit at least a cubic (or use the equivalent
five-point fourth-order stencil), giving \(O(h^4)\) truncation under the
required smoothness. More points can reduce observation-noise variance, but
they can also enlarge the stencil and increase the truncation constant or
amplify correlated particle noise. The implementation must therefore choose a
fixed symmetric grid, fit degree and span separately, and report both the
observed order in \(h\) and the replicate variance. The fitted derivative still
inherits the finite-particle/log-normalisation bias of the values being
regressed; regression does not turn it into an oracle.
An unconstrained normal draw for (h) is a poor default because draws near zero
divide particle noise by a small number; use a deterministic positive ladder
(or a distribution truncated away from zero), and use negative points as the
symmetric stencil rather than as independent random magnitudes.

A finite-difference score can be an exact control variate only when its
centering expectation is known (for example, an exact score from a tractable
reference model). In that case use
\(S_{\mathrm{cv}}=S-\beta(D-\mu_D)\), estimate \(\beta\) on calibration
replicates, and freeze it. If \(\mu_D\) is unknown and replaced by the
finite-difference estimate of the target score, the construction is a
calibrated heuristic or residual estimator, not an unbiased control variate.

## Phase 4C: fixed symmetric directional finite-difference ladder

This phase turns the finite-difference diagnostic into a reproducible test
arm. It is separate from the proposal factorial: keep the proposal, target
label, particle count, and random-number coupling fixed while changing only
the stencil. For a unit direction \(v\), test

\[
D_2(h;v)=\frac{\ell(\theta+hv)-\ell(\theta-hv)}{2h},
\]

\[
D_4(h;v)=\frac{\ell(\theta-2hv)-8\ell(\theta-hv)+8\ell(\theta+hv)-\ell(\theta+2hv)}{12h},
\]

and an eleven-point cubic least-squares fit on \(k=-5,\ldots,5\), whose
derivative at zero is the fitted linear coefficient. Under the required
smoothness, the expected truncation orders are two for \(D_2\) and four for
\(D_4\) and the cubic fit. A five-point quadratic fit or an eleven-point
linear/quadratic fit is retained only as an \(O(h^2)\) comparator; the number
of points alone does not provide fourth-order accuracy.

Use a deterministic positive ladder, for example
\(\mathcal H=\{h_0,h_0/2,h_0/4,h_0/8\}\), with \(h_0\) chosen in parameter
units after checking that every perturbed model is valid. Do not draw \(h\)
from an unrestricted normal distribution: near-zero draws divide particle or
log-normalisation noise by a small number. A truncated-away-from-zero random
ladder is an optional later arm, never the default. The negative evaluations
are the paired points in the same stencil and use common random numbers (or a
declared coupling) where the finite program permits it.

The calibration test has four parts:

1. On scalar and multivariate linear-Gaussian models, compare each directional
   estimate with the exact Kalman directional score over independent
   replicates. Fit the slope of log absolute error versus log \(h\) and verify
   the expected order before selecting a ladder.
2. On nonlinear regular-transition oracle models, repeat the curve while
   recording finite-particle bias, replicate variance, plus/minus covariance,
   non-finite weights, and the cost of each stencil.
3. Freeze the selected ladder on a disjoint validation partition and repeat
   the slope and variance checks. A ladder is not selected on the final claim
   paths.
4. For a \(d\)-parameter score, use coordinate directions or a full-rank
   direction matrix \(V\). If arbitrary directions are used, reconstruct the
   score from \(V^\mathsf T s\) and record its condition number; missing rank is
   a hard test failure.

The artifact records all perturbed parameter points, stencil weights, coupling
identity, observed order, replicate variance, and finite-particle/log-
normalisation diagnostics. Promotion requires finite values, valid support,
an observed order compatible with the claimed stencil on an oracle-bearing
model, and a predeclared uncertainty comparison. Failure of the order test is
a diagnostic veto for the ladder, not evidence that the underlying score
method is wrong. In oracle-free models this phase remains a parity or
calibration diagnostic and cannot be reported as a bias estimate or an exact
control variate.

The executable harness must also have focused tests for the exact three-point
and five-point weights, cubic-fit derivative extraction, rejection of zero or
sign-inconsistent ladder values, deterministic reuse of the plus/minus random
stream, and full-rank direction reconstruction. These are mechanics tests;
the observed-order and bias--variance curves are experiment-level tests on the
oracle ladder and must be stored under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`.

## Phase 5: regular-transition smoothing and KDM study

This is the only phase in which the ordinary Younis forward/backward smoothing
idea is eligible for model-score investigation.

1. Implement the physical-transition mixture proposal first, without positive
   bandwidth or OT, and verify the marginal likelihood identity.
2. Add KDM/LEDH proposal geometry while retaining the physical numerator and
   evaluable proposal denominator.
3. Add forward/backward mixture proposals with the required auxiliary density.
4. Compute adjacent-state pair expectations needed for transition scores.
5. Compare bootstrap, adapted, KDM, LEDH, and smoothed estimators against the
   same oracle.
6. Test whether any gain persists after retuning each method for its own scope.

The phase must distinguish a proposal that improves particle placement from a
score estimator that changes the target. Positive-bandwidth KDM is a changed
filtering approximation unless a limiting argument and the associated model
correction are supplied.

## Phase 6: degenerate-transition support program

This phase is a derivation program before it is an implementation program.

1. Inspect the manifold/constrained-state particle-filter literature and
   classify each method by its state-space reference measure and target. A
   fixed-manifold intrinsic smoother is admissible only as a special-case
   control: it must supply (or permit deriving) a transition density and
   backward kernel with respect to a common, parameter-independent manifold
   measure. Structural DSGE equilibrium and policy manifolds are generally
   parameter-dependent, so this branch cannot be promoted to the DSGE route
   without a separate moving-support derivation.
2. Write the DSGE transition in explicit ancestor/innovation coordinates.
3. Identify the base measure on which the conditional transition law is
   represented and determine whether its support depends on parameters.
4. Derive the complete-data score including the deterministic map and any
   Jacobian or constraint terms.
5. Determine whether distinct ancestors induce mutually singular supports and
   whether backward ancestor probabilities collapse.
6. Prove or disprove a valid marginalisation identity for the proposed
   coordinate representation.
7. Build a tiny exact fixture with symbolic or numerical integration.
8. Only after those checks, implement an estimator and compare it with an
   independent support-aware oracle.

Full-rank Gaussian kernels, ordinary ambient-space KDM densities, and regular-
transition PaRIS formulas may be used as stress tests, but cannot establish
validity for DSGE. If the support derivation fails, the scientific conclusion
is that the route is unresolved, not that a tuned approximation solves it.

The manifold literature supplies useful state-space machinery, including
geodesic or tangent-space sampling and implicit-constraint proposals. It does
not, by itself, supply the observed-data parameter score. A future intrinsic
Poyiadjis/PaRIS-style estimator would be a new theorem on a fixed manifold;
for DSGE, the required object is instead a moving-support or
disturbance-coordinate derivation. Manifold score-matching papers, which
estimate $\nabla_x\log p(x)$, do not answer this parameter-score question.

## Phase 7: heuristic dominance and practical comparison

For each salient regime, construct and evaluate cheap adversaries:

- linear-Gaussian: Kalman score;
- weakly nonlinear: EKF and UKF score;
- highly nonlinear: bootstrap PF and locally adapted PF;
- long horizon: larger-particle bootstrap and direct Fisher recursion;
- multimodal: mixture proposal without KDM score correction;
- degenerate: exact coordinate reference or a declared unresolved marker.

A complex method losing to a heuristic in a salient regime is a promotion veto,
even if it beats another complex method on average. The heuristic table is not
a tuning target.

## Phase 8: tuning, replication, and compute fairness

Each route receives a separate tuning scope binding model, target, horizon,
particle count, dimensions, dtype, backend, chunk policy, and all tunable
controls. Calibration, validation, and untouched claim data are disjoint.

The final comparison uses at least:

1. multiple independent observation datasets;
2. multiple particle randomisations per dataset;
3. paired seeds across methods where coupling is meaningful;
4. a particle-count and an equal-compute comparison;
5. uncertainty intervals for paired score error; and
6. preserved failed candidates and repair attempts.

One seed or one short chain is descriptive only. A candidate may be called
viable after passing hard screens, but a ranking requires uncertainty evidence.

## Phase 9: implementation and production audit

Before any default or HMC-facing recommendation, audit:

1. source-faithfulness to Younis and Sudderth where claimed;
2. support class and reference measure;
3. complete call chain from consumer to implementation;
4. analytical derivative route, with autodiff restricted to parity diagnostics;
5. TensorFlow/XLA, dtype, GPU memory-growth, and chunk-policy compliance;
6. tuning-artifact scope identity;
7. reproducibility manifest; and
8. rendered scientific documentation.

The canonical LEDH route remains governed by the repository's Contract E and
per-scope tuning policies. A new estimator can be scientifically promising
without replacing the canonical route until its own gates pass.

## Phase exit gates and dependency order

The phases are sequential where a later phase depends on a mathematical or
implementation identity, and parallel only where the dependency is explicit.

1. **Gate A, after Phase 0:** every model has a target record and an oracle
   status; no score-quality comparison proceeds without a valid comparator or
   an explicitly diagnostic label.
2. **Gate B, after Phase 1:** every candidate passes fixed-program derivative,
   support, positivity, and call-chain tests; failures return to implementation
   repair.
3. **Gate C, after Phase 2:** proposal comparisons show the physical numerator,
   proposal denominator, and resampling representation separately; otherwise
   the candidate is classified as a changed-target approximation.
4. **Gate D, after Phase 3:** estimator-level bias and variance are available on
   common clouds; only then may Phase 4 compare normalisation effects.
5. **Gate E, after Phase 4:** the dominant error source is classified as
   derivative, measure, Monte Carlo, or ratio error; unresolved classification
   blocks a claim of improvement.
6. **Gate F, after Phase 5:** regular-transition smoothing/KDM results pass an
   untouched retest and heuristic dominance; they remain confined to regular
   models.
7. **Gate G, after Phase 6:** the degenerate track has either a proved,
   executable support-aware identity or an explicit unresolved verdict.
8. **Gate H, after Phases 7--9:** only candidates with statistical uncertainty,
   complete manifests, scope-specific tuning, and implementation audit may be
   considered for default or HMC-facing status.

Phases 0--4 are the common foundation. Phases 5 and 6 then proceed as separate
regular and degenerate branches. Phase 7 can use results from either branch but
cannot merge their targets. Phases 8 and 9 are terminal validation and audit,
not sources of new tuning data.

## Required artifacts

Each phase produces:

- a plan and skeptical pre-mortem;
- a model/target/measure ledger;
- an oracle or oracle-status report;
- executable identity and call-chain tests;
- a run manifest containing commit, command, environment, hardware, seeds,
  wall time, tuning scope, and artifact paths;
- raw structured results and uncertainty calculations;
- a decision table separating hard vetoes, descriptive evidence, statistical
  ranking, and default readiness; and
- a red-team note stating the strongest alternative explanation, the result
  that would overturn the conclusion, and the weakest evidence.

The master result must include a matrix with one row per method, model, horizon,
particle count, and score kind. It must never combine KDM expectation gradients,
finite-program derivatives, and marginal scores in one unlabeled leaderboard.

## Decision rules

The program may promote a candidate only when:

1. its claimed target and computed quantity are equal by derivation or a tested
   declared approximation;
2. all support, positivity, normalization, and derivative vetoes pass;
3. the candidate clears the conditional heuristic-dominance table;
4. paired uncertainty supports the stated comparison;
5. the result survives an untouched retest under its own tuning scope; and
6. no conclusion crosses from the regular-transition track to the degenerate
   track without a new support derivation.

If a candidate fails, classify the failure as implementation, tuning,
diagnostic, finite-normalisation, support, or evidence failure. Rejecting one
candidate does not reject the research direction; rejecting the direction
requires evidence that the repairable alternatives and their target identities
have also failed.

## First executable tranche

The first tranche should be run on the linear-Gaussian model, because it supplies
an exact score and permits every estimator to be compared without support
ambiguity:

1. implement the target/oracle ledger and identity tests;
2. reproduce bootstrap, UKF, canonical LEDH, physical-mixture, KDM, Fisher,
   backward-pair, IWSG, and control-variate estimators;
3. use common particle clouds for estimator comparisons and independent clouds
   for full-filter comparisons;
4. sweep particle count, horizon, and bandwidth with separate tuning scopes;
5. quantify bias, variance, MSE, denominator variability, ESS, and runtime;
6. apply the heuristic-dominance table; and
7. write the first decision note before moving to nonlinear regular models.

The degenerate DSGE track begins in parallel with the support derivation in
Phase 6, but it does not inherit any numerical conclusion from this tranche.

## Literature coverage addendum (2026-09-14)

The literature gap audit in
`docs/plans/artifacts/particle-filter-score-literature-gap-audit-20260914/`
identified several named methods that must be made explicit before interpreting
the matrix as a complete score-computation study. The highest-priority direct
addition is the Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and
observed-information estimator. The existing `KDM` row is only a method label;
it does not yet specify Nemeth's kernel recursion, bandwidth, conditional
integration, or its regular-model asymptotic assumptions.

The smoothing branch must also name the PaRIS backward-draw parameter and the
Fearnhead--Wyncoll--Tawn linear-cost smoother. A generic backward-pair label
cannot expose the distinction between one backward draw, two or more draws,
mixing assumptions, path degeneracy, and O(N) versus O(N^2) computation. These
methods estimate additive complete-data functionals; their output is not
automatically the marginal score and their regular-HMM assumptions do not close
the degenerate DSGE branch.

The matrix must also include the coupled conditional-particle smoother of
Jacob, Lindsten, and Schön. Its Rhee--Glynn construction is a qualitatively
different answer from averaging two biased score estimates: under its meeting
and moment assumptions it debiases a finite-horizon smoothing expectation, and
the authors explicitly combine it with Fisher's identity to obtain an unbiased
score when the transition density is tractable. The estimator has random
runtime and its own variance/cost trade-off. It is a regular-model comparator,
not a singular-transition result.

The program should add continuous-likelihood particle filters (Malik--Pitt and
DeJong et al., with Flury--Shephard as a related econometric source) and
iterated filtering (Ionides et al.) as separate comparators. The first family
addresses discontinuous particle likelihood evaluation; it does not prove that
the derivative of a continuous finite particle program is the model score. The
second is designed for simulation-only models with unavailable transition
density and gives a vanishing-artificial-noise approximation with an explicit
bias, variance, and mixing trade-off. It is relevant only if a model's
transition can be simulated but its density or derivative cannot be evaluated
in the required base measure. When the transition density and its parameter
derivative are available, the ordinary Fisher/particle-score routes dominate
this comparator and iterated filtering adds no score capability. It must carry
a distinct target identifier from an exact fixed-parameter score.

Twisted particle filters should be an optional proposal-quality arm because
they preserve the likelihood through a change-of-measure correction while
optimising an asymptotic normalizer-variance criterion. Multilevel particle
filters should remain a discretization/coupling arm: their telescoping result
does not by itself yield an unbiased score or an unbiased ratio derivative.
The variational particle-objective literature (FIVO/AESMC/VSMC) belongs in the
target-boundary table so that an ELBO gradient is not reported as an observed-
data score.

These additions do not change the central support split. The inspected sources
provide no generic ambient-density KDM or PaRIS theorem for deterministic or
singular DSGE transitions. Phase 6 still requires a base-measure or
disturbance-coordinate derivation and its own score estimator before any
regular-track result can be transferred.
