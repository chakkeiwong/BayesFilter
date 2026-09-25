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

The applicability record distinguishes two non-interchangeable tracks:

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

**Active execution scope, amended 2026-09-14.** This program now executes
KDM/IWSG filtering and estimator combinations, proposal/covariance alternatives
(UKF, KDM, SGQF), twisting/iAPF, and coupled directional finite differences.
Regular models provide tractable reference tests for these mechanisms.
Regular-transition smoothing and degenerate/DSGE score derivations are assigned
to separate programs. Phases 5 and 6 and the corresponding literature rows
remain reference material; they are not generated as executable rows here and
are not prerequisites for the active KDM filtering study. Rhee--Glynn/JLS,
Nemeth, PaRIS, and other smoothing estimators are deferred with that work.
An imported result needs target, support, and provenance checks before use.

The changes below incorporate the mathematical corrections agreed in Claude's
follow-up. Claude's bounded Phase 4C review gave a REVISE verdict and marked
other sections unchecked; it did not approve those sections. The dependency,
tuning, and SGQF amendments are additional findings from this program review.

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

Every registered row also has an execution status: active, deferred, or blocked
on a named prerequisite. The matrix generator expands active eligible rows
only. Listing a method in the registries, factorial axes, or literature table
does not authorize its execution.

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
- externally supplied support/oracle evidence, imported read-only; and
- report assembly.

The fixed-cloud lane isolates conditional score-estimator variance. Repeat
over independent clouds to measure variation in the cloud itself; conditioning
on one cloud does not establish marginal-score accuracy. The full-filter lane
measures the combined effect of proposal, resampling, and score recursion. Their
outputs must not be merged into one score table without a lane label.

### Coordinator pseudocode

```text
load model_registry, target_registry, proposal_registry,
     estimator_registry, tuning_registry
validate_registries_and_supports()

candidate_status = initialize_active_candidates_and_predeclared_repairs()
for task in topological_order(active_tasks):
    matrix = generate_eligible_rows(task, candidate_status, declared_scopes)
    for row in matrix:
        validate_target_measure_support(row)
        verify_named_prerequisites(row)
        if row.is_quality_claim:
            tuning = prepare_and_freeze_scope_on_allowed_partitions(row)
            validate_tuning_scope(tuning, row)
            require_repository_issued_tuning_artifact(tuning)
        for dataset in row.declared_partition:
            oracle = oracle_registry[row.model].get(dataset, row.parameter)
            for seed in row.declared_seeds:
                result = execute_one_row(row, dataset, seed, oracle)
                write_immutable_result(result)
    decision = aggregate_under_declared_evidence_role(task.results)
    apply_vetoes_and_heuristic_gate(decision)
    write_phase_note_and_checkpoint(decision)
    update_candidate_status(decision)
    queue_predeclared_repairs_unless_continuation_veto_or_budget_exhausted()
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
decompositions, and the decision table. Diagnostic plots retain failed rows
with their status; rankings require passed vetoes, target compatibility, and
uncertainty evidence.

### Factorial axes

The active study matrix crosses the following axes, subject to compatibility
and prerequisite checks. The broader option table below retains deferred
literature coverage separately.

| Axis | Levels to enumerate |
|---|---|
| Target | exact marginal score; finite-program derivative; KDM expectation gradient; unnormalised value/derivative; declared approximation |
| Proposal | bootstrap transition; EKF; UKF; LEDH; physical transition mixture; KDM; fixed lookahead mixture; twisted PF; iAPF; SGQF-guided; defensive mixture |
| Resampling/coupling | multinomial/systematic discrete resampling with declared derivative target; continuous KDM mixture draw; OT with a derived correction or changed-target label; OT cloud plus explicit jitter; OT as coupling between replicas |
| Covariance | model (Q); UKF covariance; KDM weighted covariance; LEDH local covariance; GenUT-restored covariance; SGQF projected covariance; calibrated hybrid choices |
| Score identity | analytical finite scalar derivative; IWSG expectation gradient; analytical/IWSG hybrid; ratio of unnormalised estimates; finite-difference diagnostic |
| Variance reduction | none; conditional Rao--Blackwell; exact-mean control variate; independently calibrated estimator combination; declared coupling of finite differences |
| Support | regular common-density; bounded/constraint regular with checked support and differentiation conditions |
| Evaluation lane | fixed-cloud estimator with independent-cloud replication; full filtering recursion; directional diagnostics; long-horizon scaling |

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

Assume a common support, differentiability, integrable domination permitting
differentiation under the integral, and a positive sampling density wherever
the derivative integrand contributes. With \(q_0=m_{\theta_0}\) held fixed
while differentiating,

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
the oracle, such as squared Euclidean score error or a fixed scale-normalised
version. Choose the metric before selection and account for oracle uncertainty
when it is nonzero. Confidence intervals respect independent datasets and the
particle replicates nested within them; the latter are not independent
datasets. Any claim of improvement needs its predeclared uncertainty criterion,
including an effect-size margin or precision target where relevant.
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

Before each research run, record the main question, mechanism, expected
failure, promotion criterion, promotion veto, continuation veto, repair
trigger, explanatory diagnostics, and forbidden inference. Classify every
diagnostic by these roles. A failed candidate is not a continuation veto when
a planned repair addresses that failure. A model-score error claim and a
finite-program derivative claim require separate evidence even if they use
the same numerical estimates.

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
- particle counts \(N\in\{32,64,128,256,512,1024,2048\}\) and horizons
  \(T\in\{1,5,20,50,100\}\) where the model supports them;
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

Before a serious launch, its concise run plan must fill in the finite number
of eligible rows, maximum attempts, total wall/GPU-hour budget, hardware,
uncertainty precision needed, and stop conditions. The large grid above is a
coverage proposal, not an instruction to launch its Cartesian product. A pilot
determines feasible allocation and power; its outcomes cannot select claim
data. Any optional replication increase uses a predeclared sequentially valid
interval/stopping rule or a fixed terminal sample size. Budget exhaustion
leaves an unresolved result; it does not relax a scientific criterion.

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
  and an independent exact Gaussian derivative fixture;
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

Every active candidate requires a complete estimator and target label.
In this inventory, Fisher/backward smoothing (including Nemeth, PaRIS,
Fearnhead, JLS), iterated filtering, continuous-likelihood methods, multilevel
debiasing, variational objectives, and degenerate coordinates are deferred
literature/reference rows. Their presence does not create executable tasks.
KDM/IWSG filtering, covariance/proposal alternatives, twisting/iAPF, and FD are
active subject to their named prerequisites.

| Family | Candidate | Intended role | Main question |
|---|---|---|---|
| Baseline | Bootstrap PF with a declared value/score route | classical reference | How much error comes from ordinary particle approximation? |
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
5. the degenerate DSGE model as a deferred applicability record, without
   scheduling its support derivation or experiments in this program.

Record state dimension, parameter dimension, horizon, transition rank,
observation rank, support, analytic oracle availability, and the model's
parameter-dependent initial law.

### 0.2 Construct the oracle ladder

For each model, implement in order:

1. closed-form value and score where available;
2. deterministic quadrature or Rao--Blackwellised integration;
3. high-particle, replicated reference estimates with an error certificate;
4. an independent implementation using different code and random numbers; and
5. externally supplied support-specific references when applicable, with
   provenance and applicability checked before import.

Use the highest justified reference available; an unavailable closed form
does not block construction of a valid next rung. Replication at high particle
count bounds Monte Carlo uncertainty, not approximation bias by itself.
A numerical certificate must also bound relevant discretization/particle bias;
otherwise label the reference provisional and restrict the resulting claim.

An oracle is admitted only after value, directional derivative, finite-difference,
and parameter-coordinate checks agree within predeclared tolerances.

### 0.3 Freeze notation and target metadata

Every result records `target_id`, `measure_id`, `support_class`, `score_kind`,
`normalization_kind`, `proposal_id`, `reset_id`, `derivative_id`, and whether the
proposal is detached locally. A result with missing metadata is diagnostic only.

### 0.4 Prepare baselines and tuning before quality comparisons

Build and verify the exact Gaussian oracle and the applicable cheap baselines
before the first proposal or estimator comparison. For an active regular model,
the minimum ladder is bootstrap PF, an EKF/UKF approximation, a classically
adapted proposal where evaluable, canonical LEDH, and the proposed enhancement.
The Kalman oracle measures error in Gaussian fixtures; it is not a Monte Carlo
competitor that a particle method must beat. Define the heuristic adversaries
and salient regimes from Phase 7 now, but reserve their final evaluation as a
falsification check rather than a tuning objective.

Scope-specific tuning is a reusable prerequisite of every quality claim in
Phases 2, 3, 4, 4B, 4C, and 7; it does not wait until Phase 8. Bind model,
target, horizon, particle count, dimensions, dtype/TF32, backend, chunk policy,
proposal/estimator route, and all controls, including bandwidth, covariance
protection, twisting fit, finite-difference span/directions, and combination
coefficients. A horizon or particle-count change creates a new scope.
Partition calibration, validation/selection, and untouched claim observations
and particle streams in advance. Fit or choose controls on permitted partitions,
freeze them, and require the repository-issued scope artifact in every claim
row. Mechanics and debugging fixtures need no statistical tuning artifact but
cannot be used for quality claims.

For each material numerical default, record provenance, justification, likely
failure mode, smallest diagnostic, and status as baseline, hypothesis, or
reviewed choice. In particular, imported bandwidths, ridges, covariance floors,
lookahead families, step sizes, and conditioning limits are hypotheses until
justified for this scope. Phase 8 performs final replication and cost analysis.

## Phase 1: identity and call-chain verification

Before measuring quality, choose the identity test from the estimator's
declared mathematical target:

| Target | Required evidence |
|---|---|
| Analytical derivative of a fixed finite scalar | Compare with that same scalar at fixed randomness using diagnostic FD/autodiff where differentiable; include initial and recursive dependencies. |
| IWSG gradient of a mixture expectation | Derive the locally fixed-proposal identity with support and differentiation/integrability assumptions; compare repeated estimates with a known expectation gradient. Samplewise equality with a different pathwise estimator is not required. |
| Unnormalised value/derivative pair | Verify the sampling law and the stated expectation/derivative identities separately; declare whether \(\widehat D\) differentiates this particular \(\widehat Z\). |
| Estimate of the model score | Verify its own derivation and implementation, then measure oracle error and uncertainty. Neither pathwise parity nor low MSE proves unbiasedness. |

All applicable initial, transition, observation, mixture-weight, covariance,
bandwidth, flow-Jacobian, OT, and GenUT dependencies must be covered.
Parameter-dependent initial means and covariances need explicit sensitivity
fixtures. First compare the same finite scalar; separately compare the
estimator statistically with Kalman. A finite particle realization need not
equal the exact model score.

Require executable consumer-to-provider tests for rank, batch, dtype,
graph/XLA/device, and analytical-sensitivity contracts. For a model-corrected
proposal, verify the actual sampler/denominator law and physical numerator.
Support and positivity checks apply to the relevant density or covariance;
signed quadrature integration weights are not sampling probabilities.

The current source snapshot has an unresolved KDM wiring prerequisite:
the scalar executor in bayesfilter/highdim/ledh_canonical_score_tf.py
rejects trace and observation/post-reset callbacks requested by
ledh_younis_kdm_integrated_tf.py and ledh_younis_kdm_resampling_tf.py.
Recheck the source during the ongoing canonical repair and require an
executable endpoint regression before scheduling either KDM route. Static
inspection is evidence of this incompatibility, not runtime verification.
No alternative reduced LEDH lane may bypass it.

The same executor currently initializes state and covariance tangents to zero.
That is correct for fixed supplied inputs; it does not establish total
sensitivity through a parameter-dependent initial law. Test that outer
dependency explicitly before making the broader claim.

## Phase 2: proposal and representation study

Hold the score estimator fixed and vary only the proposal representation. The
factorial factors are proposal family, lookahead construction, covariance
source, particle count, horizon, observation regime, and compute budget:

1. bootstrap transition proposal;
2. UKF/EKF adapted proposal;
3. LEDH flow proposal;
4. KDM with explicit component covariances;
5. OT-resampled KDM cloud;
6. a fixed lookahead mixture proposal (backward-smoothing extensions deferred);
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
claim-bearing endpoint to the declared proposal. Proposal normalizer MSE is
the primary criterion for a normalizer-accuracy claim; variance alone suffices
only under a proved common expectation. ESS, coefficient of variation, coverage,
and weight tails explain or screen proposals under predeclared roles. A claim
of better score computation requires lower held-out oracle score MSE at the
declared budget. Proposal metrics cannot substitute for that criterion.
A paired uncertainty interval is required before a row is called better;
otherwise it is classified as viable or descriptive-only.

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
interface. The form of the analytical recursive score identity and the physical
target remain fixed. The implemented recursion must include derivatives of the
new ancestor probabilities, proposal density, twisting correction, and all
parameter-dependent normalizing integrals. Pilot-fitted controls may be frozen,
but parameter dependence in their declared evaluation is not silently detached.
The denominator and support record must describe the new sampling law.
If ancestor sampling is discrete, an expectation-gradient claim needs the
appropriate sampling-law terms; differentiating fixed ancestor labels is not
a replacement for them.
The arm is admissible only when the required transition integrals, proposal
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

Every UKF/KDM/SGQF provider must specify the full time-indexed filtering
lifecycle: incoming component means, covariances, and weights; prediction
through the model; conditioning on the new observation; outgoing moments;
and reassignment or transformation of persistent component state at resampling
and Contract E reset. Physical process covariance, proposal covariance, and
KDM bandwidth have distinct roles. Record the actual moments consumed by LEDH
and their analytical sensitivities, not only the provider's standalone output.

SGQF's signed integration weights must not be used as categorical probabilities.
If projected moments define a Gaussian or mixture proposal, validate its density
parameters separately and declare any protection that changes those moments.
Before its scaling study, require multistep Gaussian moment fixtures and an
executable provider-to-LEDH test across batch dimensions, dtype, graph/XLA,
device, and analytical sensitivities. An existing eager standalone filter
does not establish this integrated capability.

The OT study must include exact marginal-balance error, coupling entropy,
transport cost, sensitivity to Sinkhorn iterations, and whether the transported
cloud remains compatible with the density used in the importance correction.
OT is never treated as a resampling theorem by itself.

Record where the existing LEDH flow Jacobian enters the likelihood weights.
OT moment restoration is not a proof of preservation of the filtering law;
retain its full moments/weights/transport dependence in the total derivative.
No extra OT log determinant is inserted without a derivation for the actual
sampling law. Clarifying the existing flow determinant and deriving an OT
density correction are different tasks.

OT has five separate experimental roles that must not be conflated:

1. a coupling for comparing two particle systems with common random numbers;
2. a deterministic cloud rearrangement used only as a proposal heuristic;
3. a continuous resampling law whose density can be evaluated and corrected;
4. an OT cloud followed by an explicit, evaluable jitter kernel; and
5. a model-changing reset whose score is the derivative of a new finite scalar.

Roles 1 and 2 do not by themselves define a model-corrected proposal. Role 3
requires a density and support proof. Role 4 requires the actual jitter-mixture
density. A change-of-variables Jacobian is included only when the sampling
construction requires it; jitter around fixed transported centers does not
create a new OT determinant. Role 5 is allowed only under the finite-program
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

1. analytical derivative of the executed finite value program;
2. unnormalised likelihood derivative divided by the estimated normalizer;
3. KDM IWSG gradient of a declared mixture expectation;
4. hybrid analytical/IWSG gradients with an explicit expectation target;
5. Rao--Blackwellised conditional-integration variants; and
6. exact-mean control-variate and independently calibrated combination variants.

Fisher/backward-pair, Nemeth, Fearnhead, PaRIS, and coupled
conditional-particle/Rhee--Glynn rows are deferred to the separate smoothing
program. An available exact conditional expectation can still serve as a
small identity reference. Target-specific identity tests precede quality
comparisons; gradients of different objectives cannot be pooled.

Report vector bias, covariance, MSE, componentwise error, and correlation of
errors across coupled estimators. Do not call a lower MSE an unbiasedness result.

Control variates require a known or separately corrected centering constant.
Estimate coefficients on calibration data only, freeze them, and test whether
the empirical variance reduction survives on untouched data. A fitted baseline
whose centering expectation is unknown is labelled a heuristic variance
reduction candidate, not an unbiased control variate.

Exact centering preserves the expectation of the baseline estimator; a biased
baseline remains biased. Independent random centering also requires a proved
unbiased center and its variance contribution in the comparison. If
\(\widehat D=\nabla\widehat Z\) for the same positive finite scalar, then
\(\widehat D/\widehat Z=\nabla\log\widehat Z\) pathwise. Comparing these two
expressions tests derivative consistency, not ratio bias; compare their
expectation with \(\nabla\log Z\) to measure that bias.

Concretely, for a scalar LEDH score estimate \(B\), a KDM/IWSG statistic \(H\),
and its mean \(\mu_H=\mathbb E[H]\) under the actual joint experiment,
\(C_\beta=B-\beta(H-\mu_H)\) has \(\mathbb E[C_\beta]=\mathbb E[B]\)
for a frozen coefficient \(\beta\). Its variance is
\(\operatorname{Var}(B)+\beta^2\operatorname{Var}(H)
-2\beta\operatorname{Cov}(B,H)\). An independent unbiased estimated center
adds \(\beta^2\operatorname{Var}(\widehat\mu_H)\). A center derived conditional
on the incoming cloud must be valid for that cloud; repeat over independent
clouds to evaluate total error. The KDM mean identity, the centering mechanism,
and the correlation with LEDH error are three separate requirements.
If coefficient fitting reuses the random center, prove the required conditional
centering identity or use an independent center; unconditional unbiasedness of
the center alone does not justify multiplying it by a correlated coefficient.

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

and, for a scalar score component, variance

\[
 \operatorname{Var}(C_\alpha)=
 \alpha^2V_A+(1-\alpha)^2V_B
 +2\alpha(1-\alpha)\operatorname{Cov}(A,B).
\]

The program must first establish that (A) and (B) target the same score.
Then estimate \(\alpha\) on independent calibration replications, using a
predeclared loss or an exact control-variate relation, freeze it, and evaluate
on untouched replications. With an oracle, the MSE-optimal scalar coefficient
can be estimated directly. Without an oracle, a variance-minimising coefficient
does not generally minimise MSE because the unknown bias remains. A combination
of two biased estimators is therefore a candidate for variance reduction, not
an automatic debiasing method. The study must report the two component biases,
their covariance, the selected coefficient, and the combined estimator's
untouched error.

For vector scores the cross term is
\(\alpha(1-\alpha)[\operatorname{Cov}(A,B)+\operatorname{Cov}(B,A)]\).
With a fixed positive-definite error metric \(W\), put
\(\Delta=A-B\). The oracle MSE quadratic has

\[
 \alpha^*=-\frac{\mathbb E[\Delta^\mathsf T W(B-S)]}
                   {\mathbb E[\Delta^\mathsf T W\Delta]},
\]

when the denominator is positive. If it is zero, the estimators agree in this
metric almost surely. A convex-combination restriction projects this value
onto \([0,1]\). Estimate the coefficient on calibration data, check the
denominator and oracle uncertainty, freeze it, and evaluate it on untouched
data. Without an oracle or a derived error identity, minimising variance is
a different objective and generally does not identify this MSE optimum.

The active comparison should include:

1. canonical LEDH and an eligible model-corrected KDM score estimate, each
   explicitly targeting the same model score;
2. analytical finite-program and coupled FD estimates of that same finite
   program, kept separate from model-score accuracy claims;
3. LEDH with a KDM/IWSG statistic whose centering under the actual joint
   sampling law is known or independently estimated without bias; and
4. IWSG plus an analytical or diagnostic pathwise gradient only when both have
   the same declared expectation target.

An unnormalised derivative cannot be added directly to a score without a
derived scaling and centering relation. An oracle-trained bias prediction from
Phase 4B is a heuristic residual-correction arm and must pass held-out
model-score MSE; it is not an exact-mean control variate.

If the component targets differ, the linear combination is a new target and
must be analysed as such. It cannot be described as a better estimate of the
original score merely because its numerical variance is smaller.

## Phase 4: normalisation and long-horizon study

This phase isolates the random-ratio problem.

1. Compare \(\widehat Z_N\), \(\widehat D_N\), and
   \(\widehat D_N/\widehat Z_N\) separately.
2. Vary particle count over a logarithmic ladder and retain paired random
   numbers across methods.
3. Vary horizon while holding model and dimension fixed, with a new tuning
   scope and frozen controls for each horizon.
4. Measure denominator coefficient of variation, reciprocal-weight tails,
   score bias, and variance.
5. Compare eligible analytical/model-corrected score estimators with ratio
   estimators at equal compute, without launching deferred smoothing rows.
6. Test coupled finite differences and consistency ladders after their own
   prerequisites; randomized debiasing remains outside the active scope.

The conclusion distinguishes established derivative or measure errors from
evidence about Monte Carlo and random-normalisation error. If attribution is
unresolved, say so. This limits causal explanation; it does not by itself
invalidate a correctly measured held-out MSE comparison.

## Phase 4B: approximate consistency diagnostics

When an exact score oracle is unavailable, the program may study diagnostics
that are plausibly correlated with bias, but it must keep them separate from
the score criterion. Candidate diagnostics include:

- score estimates at \(N,2N,4N\) under coupled random numbers;
- Richardson-style slopes in \(N\) or bandwidth, when a fitted expansion is
  supported by the data;
- agreement between eligible analytical, IWSG, and finite-difference estimates
  after checking that the claimed targets agree;
- conditional mixture/normalisation identities; backward/ancestor-pair
  residuals are deferred with smoothing;
- likelihood normalisation residuals and denominator stability;
- score directional identities under parameter perturbations; and
- replicate-to-replicate correlation between each diagnostic and measured score
  error on oracle-bearing models.

The diagnostic study must be calibrated on oracle-bearing models first. A
diagnostic can then be used as an explanatory warning or a predeclared
promotion veto. A continuation veto requires evidence of invalid execution or
a separately justified stop condition, not merely correlation with error.
The diagnostic cannot prove low bias on an oracle-free model merely because
its correlation is high elsewhere. The program must report the calibration
model, held-out model, correlation
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

Common random numbers or Sen-style coupled resampling can reduce the variance of
the difference, but they do not remove the log-normalisation bias, finite-
particle bias, or finite-difference truncation bias. If the same fixed random
design contains discrete resampling, the \(\varepsilon\to0\) limit may instead
differentiate a discontinuous finite program. Richardson extrapolation is
allowed only after a relevant truncation/error expansion has been justified
and tested on the quantity being extrapolated. A noisy MSE slope is not that
test; Phase 4C separates deterministic truncation from stochastic error.

A symmetric local-polynomial fit is a useful multi-point version of this
diagnostic, but the number of points is not itself an accuracy guarantee. For
points \(\theta+k h\), the derivative is the slope of the fitted polynomial at
zero. With a symmetric stencil, a linear fit has
\[
 D_h\ell=s+\frac{\ell^{(3)}(\theta)}{6}
   \frac{\sum_k k^4}{\sum_k k^2}h^2+O(h^4),
\]
for deterministic \(\ell\) with bounded fifth derivative locally. A quadratic
fit has the same leading order because the quadratic term is
even and is orthogonal to the slope on the symmetric design. To cancel the
third-derivative term one may fit a cubic or use the five-point fourth-order
stencil. These have different weights and error constants on different grids;
both give \(O(h^4)\) truncation with five bounded continuous derivatives locally.
More points can reduce observation-noise variance, but
they can also enlarge the stencil and increase the truncation constant or
amplify correlated particle noise. The implementation must therefore choose a
fixed symmetric grid, fit degree and span separately, and report both the
deterministic order in \(h\) separately from stochastic bias and variance.
The fitted derivative still
inherits the finite-particle/log-normalisation bias of the values being
regressed; regression does not turn it into an oracle.
An unconstrained normal draw for (h) is a poor default because draws near zero
divide particle noise by a small number; use a deterministic positive ladder
(or a distribution truncated away from zero), and use negative points as the
symmetric stencil rather than as independent random magnitudes.

A finite-difference statistic can be an exactly centered control variate when
its own expectation \(\mu_D\) is known (or independently estimated without
bias). An exact model score is not automatically the expectation of a finite
stencil applied to random log-likelihood estimates. When the center is valid, use
\(S_{\mathrm{cv}}=S-\beta(D-\mu_D)\), estimate \(\beta\) on calibration
replicates, and freeze it. If \(\mu_D\) is unknown and replaced by the
finite-difference estimate of the target score, the construction is a
calibrated heuristic or residual estimator, not an unbiased control variate.

## Phase 4C: fixed symmetric directional finite-difference ladder

This phase tests whether a coupled finite-difference calculation is a useful
directional score estimate or calibration diagnostic. It has two distinct
questions: does the deterministic stencil implement its stated approximation,
and how accurately does that stencil, applied to random particle log-likelihoods,
estimate the model score? Passing the first test does not answer the second.
Keep the model, proposal, target label, particle count, horizon, and coupling
fixed within a comparison; selected tuning controls stay frozen at all its
perturbed parameter points.

### 4C.1 Stencils, smoothness, and deterministic mechanics

Let \(f_v(t)=\ell(\theta+tv)\), where \(\ell=\log Z\) and \(v\) is a
declared unit direction in the chosen parameter coordinates. Test

\[
D_2(h;v)=\frac{\ell(\theta+hv)-\ell(\theta-hv)}{2h},
\]

\[
D_4(h;v)=\frac{\ell(\theta-2hv)-8\ell(\theta-hv)+8\ell(\theta+hv)-\ell(\theta+2hv)}{12h},
\]

and an unweighted eleven-point cubic least-squares fit at \(j=-5,\ldots,5\).
Write its derivative as \(D_{\rm cub}(h;v)=h^{-1}\sum_j c_j f_v(jh)\).
With \(S_r=\sum_{j=-5}^5 j^r\), its weights are

\[
 c_j=\frac{S_6 j-S_4 j^3}{S_2S_6-S_4^2}.
\]

These follow by solving the odd block of the cubic normal equations.
A fit against dimensionless \(j\) returns a linear coefficient that must be
divided by \(h\); a fit against \(jh\) already returns the derivative in
parameter units. Use QR or SVD for fitted coefficients, not explicit
normal-equation inversion. Centered even polynomial terms do not change the
unweighted slope. Linear/quadratic symmetric fits remain second order;
eleven points alone do not produce fourth order.

A sufficient local assumption is \(f_v\in C^3\) with bounded third derivative
for \(D_2-f'_v(0)=O(h^2)\), and \(f_v\in C^5\) with bounded fifth derivative
for the five-point and cubic fourth-order statements. Four continuous
derivatives alone are insufficient. Taylor substitution gives

\[
 D_4-f'_v(0)=-\frac{f_v^{(5)}(0)}{30}h^4+o(h^4),\qquad
 D_{\rm cub}-f'_v(0)=-\frac{143}{90}f_v^{(5)}(0)h^4+o(h^4).
\]

The eleven-point fit spans \(5h\) in each direction; the five-point formula
spans \(2h\). Compare node spacing and total span separately rather than
interpreting more points as an automatic accuracy improvement.

Mechanics tests use exact polynomials and noise-free functions with known
derivatives, followed by deterministic Kalman log-likelihoods. Check stencil
moments, fitted-slope units, and a function with nonzero leading truncation
coefficient. Measure order only in a range above the roundoff/evaluation-error
floor. Exact cancellation on a polynomial is a passing algebraic check, not a
failed slope test. Any numerical slope tolerance must follow from the fixture's
error budget; no universal tolerance or condition-number cutoff is assumed.

### 4C.2 Particle bias, covariance, and the actual MSE

For a fixed dataset and parameter, let
\(L_j=\log\widehat Z_N(\theta+jhv;\xi_j)\), with the correct marginal
particle law at every point and a declared coupling across the \(\xi_j\).
Assume positive likelihood estimates, finite second moments of \(L_j\), and
write \(b_N(\vartheta)=\mathbb E[L_N(\vartheta)]-\ell(\vartheta)\).
For any of the fixed stencils, set

\[
 \widehat D_h=\frac{1}{h}\sum_j c_j L_j,\quad
 T_h=\frac{1}{h}\sum_j c_j\ell(\theta+jhv)-s_v,\quad
 B_{N,h}=\frac{1}{h}\sum_j c_j b_N(\theta+jhv),
 \qquad s_v=v^\mathsf T S(\theta).
\]

Taking expectations and then using the variance of a linear combination yields

\[
 \mathbb E[\widehat D_h]-s_v=T_h+B_{N,h},\qquad
 \operatorname{Var}(\widehat D_h)=\frac{c^\mathsf T\Sigma(h)c}{h^2},
\]
\[
 \operatorname{MSE}(\widehat D_h)
 =\bigl(T_h+B_{N,h}\bigr)^2+
   \frac{c^\mathsf T\Sigma(h)c}{h^2},
 \qquad \Sigma_{jk}(h)=\operatorname{Cov}(L_j,L_k).
\]

The squared bias includes \(2T_hB_{N,h}\); truncation and finite-particle bias
can cancel. Summing their squares separately is wrong. On oracle-bearing
models, deterministic evaluations give \(T_h\); repeated particle evaluations
estimate total bias, \(B_{N,h}\), and covariance with uncertainty. Away from
such oracles these terms generally cannot be identified separately.

Estimate MSE directly from replicated squared oracle errors. A squared
replicate-mean error is not itself an unbiased estimate of squared bias;
retain its finite-replication uncertainty or apply a justified correction.

There is no universal stochastic slope or U-shaped MSE curve. If the same
additive noise enters all \(L_j\), it cancels because \(\sum_jc_j=0\). With a
mean-square differentiable coupled process, the numerator variance can be
\(O(h^2)\), so division by \(h^2\) leaves bounded variance. With independent
noise of nonvanishing variance it instead grows as \(h^{-2}\). Discrete
resampling and other couplings can behave differently. Record
\(c^\mathsf T\Sigma(h)c\), total bias, variance, and MSE; report a plateau,
boundary minimum, or irregular curve if that is observed. No noisy error
slope is required to match deterministic stencil order.

### 4C.3 Calibration, held-out comparison, and full-score reconstruction

Use a finite deterministic positive ladder, such as
\(\{h_0,h_0/2,h_0/4,h_0/8\}\), whose span, lower bound, and length are chosen
for the parameter scale, dtype, model domain, and evaluation accuracy.
Every perturbed point must be valid and distinguishable in the execution
dtype. If a log-likelihood evaluation has error bounded by \(\delta_j\), the
corresponding derivative error is bounded by
\(h^{-1}\sum_j |c_j|\delta_j\); use such bounds or measured reference errors
to justify the numerical range. No fixed universal range of \(h\) is imposed.
An unrestricted normal draw for \(h\) is excluded because it can approach zero.
A distribution truncated away from zero is an optional later comparison.

For constrained or differently scaled parameters, declare the coordinates
before choosing directions and steps. If \(\theta=g(\eta)\), the computed
likelihood score in these coordinates is
\(S_\eta=(\partial\theta/\partial\eta)^\mathsf T S_\theta\); reconstruct in
those coordinates and transform with the appropriate nonsingular Jacobian
when an original-coordinate score is required. This is likelihood
reparameterisation, not a posterior-density Jacobian term.

Calibrate each stencil and coupling on independent particle replications of
Gaussian oracle models, then nonlinear regular oracle models. Preserve each
replication's entire vector of perturbed values to estimate the full
within-stencil covariance, including correlations across directions and
stencils. Common random numbers must preserve each perturbed filter's marginal
algorithm. Independent noise is a useful covariance reference arm.

Predeclare the directional or scaled vector MSE, cost accounting, selection
rule, and uncertainty method. Use disjoint calibration, validation/selection,
and final claim data and streams; freeze the selected stencil, \(h\), directions,
and coupling before claims. The selected calibration minimum is optimistic.
Do not require validation MSE to fall inside a calibration confidence interval.
Use paired differences in squared oracle error on the untouched set, with
dataset-level or hierarchical uncertainty for particle replicates nested in
datasets. Report all predeclared comparisons or account for multiple selection;
do not select a winner on the claim set.

For \(m\) directions in \(d\) coordinates, define
\(V=[v_1,\ldots,v_m]\in\mathbb R^{d\times m}\) and let
\(b\in\mathbb R^m\) contain their derivative estimates. Then
\(b\approx V^\mathsf T s\). If \(V\) has full row rank, the least-squares
solution is

\[
 \widehat s=\arg\min_s\|V^\mathsf T s-b\|_2^2
           =(VV^\mathsf T)^{-1}Vb.
\]

Solve by QR/SVD without forming the displayed inverse. Coordinate directions
provide a simple reference; fewer than \(d\) independent directions identify
only a projection. Report singular values, reconstruction residual, and
propagated error: for \(A=(VV^\mathsf T)^{-1}V\), bias is \(A\,\operatorname{Bias}(b)\)
and covariance is \(A\,\operatorname{Cov}(b)A^\mathsf T\). Rank failure or error
amplification beyond the declared dtype/accuracy budget vetoes a full-score
claim. A small overdetermined residual alone does not establish accuracy.
Any generalized least-squares extension must freeze its independently
estimated weighting rule and account for its estimation uncertainty.

Compare equal-particle and equal-compute performance separately, charging all
nonzero-weight function evaluations, directions, pilot fitting, and replication.
Report both total campaign cost and amortized per-evaluation cost. MSE times
runtime is not a universal objective: averaging \(R\) replicas gives
bias squared plus variance divided by \(R\), leaving the bias unchanged.

### 4C.4 Pass criteria and saved evidence

Deterministic algebra/order failures, invalid support, non-finite values,
incorrect coupling marginals, missing sensitivities in a claimed derivative,
or insufficient reconstruction rank are repair triggers and relevant validity
vetoes. Noisy slopes and the absence of a U-curve are explanatory observations,
not vetoes. A valid candidate that fails held-out MSE improvement is rejected
for promotion at that scope; continue any planned repair within budget.
Without a model-score oracle or a certified reference-error bound, this is
a consistency/parity diagnostic, not a measured bias correction.

Focused tests cover exact weights and moments, cubic-fit units, invalid steps,
finite-precision point collapse, coupling-stream reuse and marginal laws,
rectangular direction reconstruction, and error propagation. Campaign artifacts
save perturbed parameter points, weights, coordinate transform, coupling,
deterministic errors, replicate vectors, covariance, MSE uncertainty,
selection partitions, reconstruction diagnostics, costs, and the criterion/
veto/explanatory role of each diagnostic. Store them under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`.

## Phase 5 (deferred): regular-transition smoothing

The ordinary Younis forward/backward smoothing study belongs to the separate
regular-transition program. KDM filtering and IWSG remain active in Phases 2--4;
this deferred phase must not be a prerequisite for those experiments and no
smoothing row is generated by this master.

The external program should, in its own plan, implement the physical-transition
mixture before positive bandwidth/OT, add the auxiliary backward density,
compute adjacent-state expectations, and retune every method in its own scope.
It must keep proposal-placement gains separate from a score identity. Positive-
bandwidth KDM is a changed filtering approximation unless a limiting argument
and model correction are supplied.

## Phase 6 (deferred): degenerate-transition support program

This is a derivation program owned by the separate DSGE/support workstream,
before it is an implementation program. This master records applicability but
does not schedule these rows or transfer regular-track results to them.

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

- linear-Gaussian: bootstrap and locally adapted PF, with Kalman as the exact
  error oracle;
- weakly nonlinear: EKF and UKF score;
- highly nonlinear: bootstrap PF and locally adapted PF;
- long horizon: larger-particle bootstrap and a fixed-coupling central
  finite-difference reference of its likelihood;
- multimodal: mixture proposal without KDM score correction;
- no degenerate row is executed here; imported support-aware results require
  an applicability and provenance check.

A complex method losing to a heuristic in a salient regime is a promotion veto,
even if it beats another complex method on average. The heuristic table is not
a tuning target.

Construct three to seven concrete cheap competitors for each evaluated regime
from these families, with a rationale and the same target/budget accounting.
Oracle distance is the primary error measure; an exact oracle is not itself
a stochastic method to beat. Use paired uncertainty for loss comparisons.
A statistically supported loss to a heuristic vetoes promotion; inconclusive
differences block a superiority claim until the planned precision is reached
or the budget ends. Do not tune against the final heuristic evaluation.

## Phase 8: final replication and compute fairness

Tuning is already a prerequisite of each claim row under Phase 0. Each route
receives a separate tuning scope binding model, target, horizon,
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

| Task | Minimum prerequisite |
|---|---|
| Baseline preparation, oracle checks, tuning service | Phase 0 specification and bounded run plan |
| Phase 2 proposal quality | Gate A, applicable Gate B identities, actual proposal/correction law, scope tuning |
| Phase 3 estimator comparison | Gate A, estimator-specific Gate B, verified fixed-cloud law; Gate C for each new proposal used |
| Phase 4 normalisation study | Verified value/derivative pair and Phase 3 measurements for the estimators compared |
| Phase 4B consistency calibration | Oracle and eligible estimators; Phase 4C mechanics for FD diagnostics; no circular requirement to finish all error attribution |
| Phase 4C deterministic mechanics | Phase 0 exact scalar/derivative fixtures |
| Phase 4C stochastic/full-score study | Eligible value endpoint and proposal law, deterministic mechanics, valid directions, coupling checks, scope tuning |
| Phases 7--9 final comparisons/audit | Applicable mechanism gates and frozen scopes for the candidates actually compared |

These are dependencies for each row, not a requirement that every candidate
finish one numbered phase before any candidate enters another. A KDM wiring
block does not stop oracle FD mechanics or an already verified UKF comparison.
Neither Phase 4B nor Phase 4C requires the separately owned Phases 5 or 6.

1. **Gate A, after Phase 0:** every active model has a target record, baseline
   ladder, tuning scope, and oracle
   status; no score-quality comparison proceeds without a valid comparator or
   an explicitly diagnostic label.
2. **Gate B, after Phase 1:** each target-specific identity, support,
   positivity, sensitivity, and consumer-to-provider call-chain test passes;
   an IWSG identity is not replaced by finite-program parity. Failures return
   to implementation repair.
3. **Gate C, after Phase 2:** proposal comparisons show the physical numerator,
   proposal denominator, and resampling representation separately. Missing
   correction blocks a model-corrected claim. A separately defined, valid
   finite scalar may continue as a changed-target approximation; an invalid
   density or implementation cannot be repaired by relabeling it.
4. **Gate D, after Phase 3:** estimator-level conditional error is available on
   common clouds and total error over independent clouds; Phase 4 uses these
   verified estimators for normalisation comparisons.
5. **Gate E, after Phase 4:** \(\widehat Z,\widehat D,\widehat D/\widehat Z\)
   and their uncertainty are reported; derivative, measure, Monte Carlo, and
   ratio explanations are classified where evidence permits. Unresolved
   attribution limits causal language but does not erase a valid held-out MSE.
6. **Gate F, after Phase 4B:** consistency diagnostics have oracle calibration,
   held-out correlation uncertainty, and explicit explanatory/veto roles.
   Correlation is never a proof of low bias.
7. **Gate G, after Phase 4C:** deterministic mechanics, scale-aware stochastic
   calibration, coupling marginals, and full-rank reconstruction pass. A noisy
   MSE slope or absent U-curve is not a veto.
8. **Gate H, after Phase 7--9:** only candidates with statistical uncertainty,
   complete manifests, scope-specific tuning, and implementation audit may be
   considered for default or HMC-facing status.

Phases 0--4 are the active foundation. Phase 7 evaluates active regular
candidates against conditional heuristics. Phases 5 and 6 are deferred
external branches and cannot supply active claim evidence. Phases 8 and 9 are
terminal validation and audit, not sources of new tuning data.

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

1. the claimed target matches the computed quantity by derivation; a declared
   approximation retains its distinct target and reports its measured or
   bounded error against the model score;
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
2. verify noise-free three/five/eleven-point mechanics and rectangular
   directional reconstruction against exact derivatives;
3. admit bootstrap, UKF/adapted, canonical LEDH, and physical-mixture baselines
   only after their endpoint, support, initialization, and density tests pass;
4. use common clouds for conditional estimator comparisons, repeat across
   independent clouds, and preserve each full-filter arm's marginal law while
   coupling arms where valid for paired inference;
5. add KDM/IWSG, centered control variates, covariance providers, and twisting
   after their named prerequisites pass; defer integrated KDM rows while the
   trace/callback conflict remains;
6. calibrate particle count, horizon, bandwidth, and FD choices in separate
   scopes, then quantify held-out bias, variance, MSE, denominator variability,
   ESS, runtime, and the conditional heuristic table; and
7. write the first decision note before nonlinear regular and SGQF scaling
   studies, which each require their own scoped plans and prerequisites.

The deferred DSGE/support program is not launched in parallel by this tranche
and cannot inherit any numerical conclusion from it.

## Literature coverage addendum (2026-09-14)

This is a literature applicability inventory, not an expansion of the active
execution scope. Smoothing, iterated filtering, continuous-likelihood methods,
multilevel debiasing, and DSGE derivations remain deferred; their earlier
implementation requests are superseded by the active-scope amendment.

The literature gap audit in
`docs/plans/artifacts/particle-filter-score-literature-gap-audit-20260914/`
identified several named methods that must be made explicit before interpreting
the inventory as a complete score-computation study. One direct
regular-model comparator is the Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and
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

The literature inventory also includes the coupled conditional-particle smoother of
Jacob, Lindsten, and Schön. Its Rhee--Glynn construction is a qualitatively
different answer from averaging two biased score estimates: under its meeting
and moment assumptions it debiases a finite-horizon smoothing expectation, and
the authors explicitly combine it with Fisher's identity to obtain an unbiased
score when the transition density is tractable. The estimator has random
runtime and its own variance/cost trade-off. It is a regular-model comparator,
not a singular-transition result.

The inventory records continuous-likelihood particle filters (Malik--Pitt and
DeJong et al., with Flury--Shephard as a related econometric source) and
iterated filtering (Ionides et al.) as deferred comparators. The first family
addresses discontinuous particle likelihood evaluation; it does not prove that
the derivative of a continuous finite particle program is the model score. The
second is designed for simulation-only models with unavailable transition
density and gives a vanishing-artificial-noise approximation with an explicit
bias, variance, and mixing trade-off. It is relevant only if a model's
transition can be simulated but its density or derivative cannot be evaluated
in the required base measure. When the transition density and its parameter
derivative are available, the plug-and-play motivation does not by itself
justify this extra arm. No claim of statistical dominance follows without a
comparison. If investigated elsewhere, iterated filtering must carry
a distinct target identifier from an exact fixed-parameter score.

Twisted particle filters should be an optional proposal-quality arm because
they preserve the likelihood through a change-of-measure correction while
optimising an asymptotic normalizer-variance criterion. Multilevel particle
filters remain a deferred discretization/coupling family: their telescoping result
does not by itself yield an unbiased score or an unbiased ratio derivative.
The variational particle-objective literature (FIVO/AESMC/VSMC) belongs in the
target-boundary table so that an ELBO gradient is not reported as an observed-
data score.

These additions do not change the central support split. The inspected sources
provide no generic ambient-density KDM or PaRIS theorem for deterministic or
singular DSGE transitions. Phase 6 still requires a base-measure or
disturbance-coordinate derivation and its own score estimator before any
regular-track result can be transferred.

## Open prerequisites and amendment review

This amended master specifies the experiment; it is not evidence that the
filter integrations or the research campaigns have passed. Prerequisites are
attached to the affected rows, so unrelated verified work can proceed.

| Obligation | Current evidence and next check |
|---|---|
| KDM integrated/resampling endpoint | Static trace/callback conflict in the current scalar executor; recheck the ongoing repair and run endpoint wiring regressions before KDM rows. |
| Parameter-dependent initialization | Supplied initial-state/covariance tangents start at zero; demonstrate the full outer initialization sensitivity on a nonconstant fixture. |
| UKF/KDM/SGQF moment lifecycle | Specify prediction, observation conditioning, persistent component state and reset, then test the actual moments and sensitivities consumed by LEDH. SGQF integration remains unverified. |
| Twisting/iAPF | Derive and test the actual ancestor/state law, normalizing integrals, correction and derivative terms. Preserve paper/source anchors for any faithfulness claim. |
| KDM as a LEDH control variate | Supply the center under the actual sampling law, or an independent unbiased center, and include its estimation cost/variance; otherwise retain the heuristic label. |
| Companion manuscript | Synchronize the finite-difference smoothness, stochastic MSE, direction convention and covariance lifecycle before using it as the revised implementation specification. |
| Serious run | Fill the scoped evidence contract, defaults audit, partitioning, finite attempts/compute budget, exact environment/commands, and unique output root. |

The companion manuscript at
[ledh_younis_kdm_score.tex](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex)
still has the old four-continuous-derivatives claim and combined stochastic
slope check in its directional-FD section, and a prediction-only SGQF moment
sketch with a literal ",qquad". These need a localized manuscript amendment,
PDF build, and inspection; they must not override the corrected specification
here. The already documented ratio-bias witness does not need replacement,
and clarifying the existing flow determinant does not imply adding an OT
determinant.

The [amendment review](../reviews/younis-score-master-program-amendment-review-2026-09-14.md)
records the mathematical checks, remaining implementation/documentation
obligations, and review scope. Historical reviews remain preserved. Their
procedural gates do not supersede the current repository policy, and a review
or document check is not a MathDevMCP audit result.
