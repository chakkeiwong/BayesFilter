# A04: initialize the pair TT from the available SGQF joint

## Authority, question and scope

This is the executable phase-7 protocol of the
[master program](observation-aware-tt-repair-complete-program-20260913.md), under
H6's five-hour total authorization. The owner asks us to try SGQF initialization
and, if possible, preserve SGQF as a lower bound. The preceding explanations
are recorded in [the clarification](observation-aware-tt-sgqf-fit-clarification-20260914.md).
The master and checkpoint before this amendment are preserved with checksums in
`artifacts/observation-tt-sgqf-initialization-20260915-01/preservation.json`.

The question is whether starting with the known correlated SGQF Gaussian
reduces fitting error, and whether subsequent refinement preserves its accuracy.
The current implementation uses that joint to draw rows but starts fitting from
generic coefficients. We will distinguish error introduced by polynomial/rank
conversion from error introduced or removed by optimization. This refines the
planned same-target fitting diagnosis in phase 8. It does not introduce coupled
charts, change the target or claim a source-faithful Zhao–Cui method.

The initial diagnostic is limited to d1/d4, degree 3 and pair rank 3, on exposed
observations. Phase 8's defensive-mass calibration and phase 9's untouched
sequence comparison remain later work. This protocol cannot promote a filter.
Independent [protocol review](../reviews/observation-aware-tt-master-amendment-04-review-20260915.md)
returned AGREE before numerical execution, as requested by the owner.

## Mathematical construction

Let z be the previous state and x the current state. From incoming SGQF
N(m,P), transition x|z = N(Az,Q), and updated SGQF q_t(x), construct
q_G(z,x) = q_t(x) N(z; m + K(x-Am), P-KSK'), where S=APA'+Q and K=PA'S^-1.
This is the correlated Gaussian already used for regression-row generation.
The target remains g_t(x) f(x|z) times the saved previous TT marginal; a
separate Gaussian-incoming first-transition diagnostic isolates that difference.

In the existing separate affine charts, write the Gaussian joint as
q_G(w)=N(w;mu,C) and the reference as rho(w)=N(w;0,I). Its amplitude
a_G=sqrt(q_G/rho) has norm one in L2(rho). An exact finite polynomial
representation is not assumed. Compute its degree-3 Hermite projection
analytically and compress the coefficient tensor by pair TT-SVD at rank 3.
This is a bounded d<=4 initialization diagnostic, not a scalable production
implementation or an added fitting solver.

The coefficients can be computed without fitting: sqrt(q_G rho) = M N(h,V),
V=2(C^-1+I)^-1, h=(C^-1+I)^-1 C^-1 mu, and
log M=-log|C|/4+log|V|/2-mu'C^-1mu/4+h'V^-1h/2.
For normalized probabilists' Hermites, let e_alpha=E[H_alpha(W)] under N(h,V).
Choose an i with alpha_i>0 and beta=alpha-e_i. Then
e_alpha = h_i e_beta/sqrt(alpha_i)
 + sum_j (V-I)_ij sqrt(beta_j/alpha_i) e_(beta-e_j), with e_0=1.
The desired coefficient is M e_alpha. This follows by differentiating the
generating function exp(t'h + t'(V-I)t/2) and dividing by sqrt(alpha!).
Record the projection residual 1-sum(coefficients squared), the TT compression
residual, and total amplitude residual against the exact Gaussian. A negative
residual beyond 256*float64_epsilon*coefficient_count invalidates the
construction. Recurrence is ordered by total degree up to 2*d*3 (24 at d4).

TT-SVD and generic starts use different coefficient gauges. This is an
initialization comparison including its gauge, not a gauge-invariant proof
about L1. No perturbation of an exactly zero rank channel is silently added:
record deficient bonds and stop that case for diagnosis. The cases with a real
adjacent-state Gaussian should have populated bonds; constant/independent
Gaussian checks exercise projection only, not a fictitious rank-3 optimization.

## Evidence contract and diagnostic roles

All arms within a case share the physical target, rows, row weights, train-only
target scale, degree, rank and optimization budget. Use 1024 training rows,
4096 validation rows and 8192 final audit rows, with the existing mixture
s=0.2 rho+0.8 q_G and weights rho/s. Fresh row seeds are 815100, 815200 and
815300, offset by 1000*d+10*t; split offsets are 0, 100000 and 200000.
These are independent numerical-design replications on exposed sequences,
not independent observations or fresh scientific holdout evidence.

Construct these simple Gaussian comparators from the problem: product of the
two existing SGQF marginal charts (no temporal dependence); the predictive
Gaussian joint (transition dependence without the current observation); and
the updated SGQF joint (observation and temporal dependence). Compare them
with the saved TT, the unrefined SGQF-to-TT conversion, generic initialization
followed by fitting, and SGQF initialization followed by the identical fitting.
Report conditional results separately by dimension, time and incoming law.
Do not pool away the late failing targets.

Fit both initializations with L1 in {0, 1e-5, 1e-3}, 4 sweeps and 128 proximal
steps; select each family's L1 only by signed-amplitude validation relative RMS,
as in the current fitter. Also run 8 sweeps/256 steps at the selected L1 from
the SAME original initialization, not from the fitted result. This tests
remaining optimization work at fixed capacity. It is explanatory only and
cannot prove global optimality. Before fitting, give both initializations the
same positive training-only least-squares scalar normalization. Record that
scalar, the initial objective, train/validation/audit residuals, KKT, Gram
conditioning and core objective decreases. A nonpositive optimal scalar is
a rejected initialization, not a sign-flipping repair.

The primary diagnostic of density accuracy is the common-panel squared
Hellinger discrepancy H2=1-<a,b>_w/(||a||_w ||b||_w), where a is the
nonnegative candidate density amplitude and b is the target amplitude.
Self-normalization removes the unknown target mass. This is an empirical
quadrature estimate of Hellinger distance, not a certified population error.
All TT candidate densities use their exact polynomial mass and the inherited
5% defensive component; also report the raw conversion without that component
to isolate its effect. The current 5% remains an uncalibrated comparator,
not a default selected by this experiment.

An empirical SGQF safeguard chooses the lowest validation H2 among the exact
SGQF joint, the unrefined conversion and the two fitted families. Ties select
SGQF. The extended-budget fits are excluded from this selector. The unchanged
SGQF joint is always available, so selected validation H2 cannot exceed its
value. Freeze that decision before evaluating the audit panel. Report when
audit performance nevertheless worsens. No pointwise lower bound follows:
two normalized densities cannot have a strict everywhere ordering unless they
are equal almost everywhere. Nor does a validation safeguard guarantee unseen
accuracy, finite-particle performance or improvement over the true filter.

| Role | Criterion and consequence |
| --- | --- |
| Primary diagnostic | Common-target validation/audit H2, with conversion and refinement effects separated; nominate or reject this initialization only. |
| Promotion veto | Any material heuristic underperformance on a case is a veto on promoting this candidate; report it prominently. No filter promotion is attempted anyway. |
| Continuation veto | Wrong target, failed Gaussian/projection identities, invalid covariance, nonfinite values, corrupt input provenance, missing audit separation or exhausted total budget. Repair infrastructure within scope; never interpret invalid output. |
| Repair trigger | Projection loses accuracy, rank channels disappear, or fitting worsens the initializer. Localize conversion/capacity/optimization before any new method is proposed. |
| Explanatory | Signed fit RMS, KKT, Gram conditioning, objective decreases, effective rows, conversion residuals, runtime and defensive-component effect; none proves convergence or filter quality. |

## Cases, checks, commands and artifacts

Use the unchanged campaign-02 fixtures and saved retained marginals under
`docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/`.
For each dimension evaluate t=1 with Gaussian incoming SGQF, and t=1,17,18
with the actual saved preceding TT. Record source/fixture hashes and confirm
the charts agree. The chosen late times diagnose already observed failures;
they cannot validate a model selected on them. Gaussian-incoming comparisons
must not be mislabeled as the recursive TT target.

Before the research run, CPU-only reference checks (CUDA_VISIBLE_DEVICES=-1)
verify Gaussian coefficient moments against independent low-dimensional
quadrature, the identity Gaussian limit, exact coefficient reconstruction at
sufficient rank, joint conditional algebra, and explicit-initialization call
behavior. The real GPU/XLA run verifies the compiled initializer and fitter
against these reference outputs. Existing pair-consumer tests remain required.
Do not modify unrelated dirty numerical code.

Planned runner: `docs/benchmarks/diagnose_observation_tt_sgqf_initialization.py`.
Run in conda `tftwogpu`, float64, GPU/XLA, trusted/escalated access, verified
memory growth, TF_NUM_INTRAOP_THREADS=2 and TF_NUM_INTEROP_THREADS=1.
The exact command is the environment's Python followed by this script,
`--output-root docs/benchmarks/artifacts/observation_tt_sgqf_initialization_20260915/attempt-01`
and `--wall-budget-seconds 3600`. A bounded one-case GPU smoke precedes it.
The final script records its actual full command and all CLI defaults.
Use stable signatures and XLA for repeated fitting and coefficient recurrence;
one-time Gaussian setup, SVD, diagnostic evaluation and serialization are
explicit host/reference exceptions. No NumPy runtime and no pfor.

Artifacts: manifest with commit, dirty state, dependency hashes, seeds, input
hashes, environment, TF/GPU/growth/XLA settings, wall time and plan/result;
per-case JSON and coefficients; terminal decision and inference-status tables;
complete stdout/stderr; updated master and concise checkpoint.

## Assumptions, budget and pre-execution skeptical audit

The degree/rank/rows/solver schedule and L1 grid are frozen baseline settings,
not justified optima. Poor conversion or a budget response tests their failure
mode. The 20% row mixture is inherited with bounded importance weights; effective
sample size and split agreement expose insufficient coverage. Degree-3 projection
can discard important Gaussian dependence; its analytic residual measures that
before fitting. Saved TT targets inherit earlier error, so even excellent fits
cannot establish posterior correctness. Fresh independent sequences and the full
six-filter/reference ladder remain required in phase 9. L1 selection is scoped
separately to every d/time/target case and audit rows never select it.

Five-hour accounting is in
`artifacts/observation-tt-sgqf-initialization-20260915-01/budget.json`:
1200 seconds conservatively charged before 2026-09-14T17:59:16.768325Z,
then elapsed active wall time. Do not add the old campaign's unused allowance.
Within the remaining 16800 seconds, reserve at most 3600 design/review,
3600 implementation, 900 focused checks, 5400 aggregate numerical work and
2700 closeout (600 spare). At most three research launches including localized
infrastructure retries; unique directories, no overwrites. Numerical per-launch
cap 3600 seconds, aggregate 5400 seconds, always bounded by remaining total.
A failed scientific candidate is evidence, not an invitation to consume all
three attempts. Update the ledger after each substantive result.

Skeptical audit: the prior explanation confused knowing the Gaussian joint with
having initialized TT from it. This protocol corrects that baseline error and
measures conversion before optimization. It avoids treating KKT or fit loss as
filter promotion, controls targets/rows/budgets, exposes inherited assumptions,
and includes explicit stops. It could still mislead through finite-panel error
or inherited TT target error; independent audit rows and explicit limits address
those risks, while independent sequence filtering remains deferred. The audit
passes for a bounded initialization diagnostic, subject to independent review
and the stated executable validity checks.
