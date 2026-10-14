# Lorenz controlled-SMC root-cause audit and investigation plan

Status: forensic audit complete through 129, 2026-10-02; repair phases below are proposed and not yet implemented. Owner requested a careful code/math
trace, a detailed cause list, and a proposed repair/investigation plan.

## Immediate evidence contract

Question: which checked code operations explain invalid fitted precision,
extreme proposals, forward overflow, and independent-recursion disagreement?
Trace the actual safety harness -> LorenzRefinement -> shared fit/forward ->
Gaussian formulas -> RK4, and the corresponding author code and paper.

Use saved attempts 76, 111, 114, 121, 123 and 126. The comparator is the same
saved input, independently evaluated Gaussian identities and original author
code; earlier healthy refits are controls. All new numerical work is CPU-only
diagnostic/reference work. No new policy, tuning, random seed, or efficacy run.
Saved failing cases are debugging fixtures, not future validation holdouts.

The diagnostic must reproduce saved initial particles from the recorded
Gaussian parameters and noise, quantify mean versus covariance effects, check
algebraic identities, and preserve input hashes. A mismatch invalidates that
diagnostic's attribution and triggers localization, not relaxed tolerances.
Finite/shape/hash failures and the 30-second process limit stop the diagnostic.
Geometry, condition numbers, residuals and counterfactual decompositions explain
failure only; none promotes a repair or ranks methods. Full recursion checks,
source faithfulness and empirical replication remain separate requirements.

Budget: one numerical CPU diagnostic with at most one infrastructure retry,
at most 30 total process seconds, charged
to the existing campaign/tranche. Before this audit: 126/130 attempts,
45.915555 CPU seconds left in the tranche; 299.082491 CPU seconds and
116126.124294 GPU seconds left in the inherited allowance. No GPU allocation
or extension. The previous bounded spectral repair stage remains closed.
Output: the next versioned campaign attempt and this note; exact command,
source/input checksums, environment, timing and result remain in its manifest.

Assumptions: FP64 decoding matches saved tensors; P0=100 I and transition
precision=1000 I come from the checked Lorenz constructor and source model;
row 0 stayed active in the failed four-refit runs; saved standard normals are
the actual sampler inputs. Particle reconstruction is the early check of all
four assumptions. NumPy/SciPy are independent diagnostic tools only.

Skeptical audit before execution: a saved-cloud replay can identify arithmetic
and proposal geometry, but cannot establish general stochastic robustness,
causal attribution of all upstream errors, MATLAB/Octave equivalence, a valid
repair, or paper replication. Comparing coefficients alone can confuse a
constant policy gauge with a change in the proposal; inspect means, covariance,
log integrals and downstream quantities too. New repair proposals must separate
paper-specified operations from local choices. No proxy becomes an efficacy
criterion. These limits make the bounded diagnostic informative and valid.

## Findings and next phases

The detailed code/math findings and source anchors are in
[the audit result](controlled-smc-lorenz-root-cause-result-20261002.md).
Attempts128 and 129 pass 33 reconstruction/high-precision checks and 50 algebraic
checks respectively. All404 separately recorded fixed-time fits replay within
the diagnostic tolerance. No repair is admitted by this audit.

Infrastructure note: attempt127 consumed 0.007918 CPU seconds before any numerical
work. The legacy CPU dispatcher redirected the explicitly registered diagnostic
to an absent generic script. Repaired its routing and added an exact script
assertion; retry uses a new directory within the unchanged 30-second total.

## Follow-up attribution, specified before attempt129

Attempt128 passed all 33 diagnostic checks and reproduced404 fixed-time fits.
At refit 4/time 0, before any local projection, the proposal mean reaches 635
with minimum precision eigenvalue1.816; the initial regression singular ratio
is0.125 and its RMS residual0.00929. The covariance floor leaves that weak
precision unchanged. This motivates a frozen-target algebraic decomposition,
not another floor sweep.

One additional CPU diagnostic (15seconds per launch, at most one localized
infrastructure retry) separates current weights from the next-policy integral
and checks the absolute Bellman-target identity including the observation
quadratic. The shared aggregate cap remains30CPU seconds across both diagnostic
phases;127+128 consumed4.130385seconds. No new GPU work or randomness.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_root_cause_attribution

Pre-run skeptical audit: equality is checked on exactly the saved clouds and
next-time policies, using independent completed-square integrals and SVD.
Tolerance1e-6+1e-8*abs(reference) is inherited from the diagnostic128 and does
not replace stricter full-recursion admission. Decomposition checks must all
pass; source/data hash or algebraic failure invalidates attribution. Frozen
inputs can locate contributions but cannot prove an upstream counterfactual
or validate a new repair. This is an explanatory diagnostic, with no tuning,
promotion or statistical ranking.


## Terminal audit and corrected research question

Attempts127–129 charged6.850606 process seconds, CPU only, within the30-second
diagnostic allowance. Attempt127 was a dispatcher infrastructure failure;
128 and 129 passed. No numerical job is running. The former bounded spectral
stage remains closed without an admitted candidate. Its unused allocation is
not a license to repeat floor sweeps.

The new evidence changes the next question from “which covariance floor makes
the run finite?” to “can we reproduce the specified adaptive recursion and
prevent an accurately fitted local quadratic from inducing an unsupported
Gaussian proposal, without disguising an algorithmic change as replication?”
The observed mean 635 exists before time 0 projection. The future-policy term
supplies almost all the displacement. The implemented covariance floor can
improve condition number while preserving that mean failure. These are checked
findings. The upstream contribution of earlier projections remains unknown.

## Separate scientific targets

1. **Author replication:** preserve the published model, data, RK4 map, policy
   class, particle/refit schedule and source procedure. Record failures as
   replication outcomes. Do not silently replace normal-equation semantics,
   truncate ranks, add a trust radius or change the integrator and then label
   the result exact author-code replication. Native Octave is an executable
   source baseline; MATLAB numerical equivalence is not yet checked.
2. **Numerically faithful reformulation:** same mathematical target, evaluated
   with centered QR, stable integrals and checked eigenpairs. Demonstrate
   equivalence on matched inputs and account for source solver differences.
   Whole-recursion validation remains required, including114.
3. **Robust policy extension:** any new constraint that changes an accepted
   fitted policy is explicitly an extension. Its first acceptance question is
   non-harm and bounded, flagged behavior; empirical efficiency comes later.
   The paper's general projection suggestion does not specify our floor or
   trust-radius choices. This path cannot silently close target1.

These tracks preserve the user's replication objective. Developing a diagnostic
extension within the repair effort is not authority to redefine the empirical
replication target. Any material change of that target needs owner direction.

## Research intent and diagnostic roles

| Item | Predeclared meaning |
|---|---|
| Main question | Reproduce the source method and explain/fix local numerical defects; separately test whether a declared protection can make the adaptive proposal usable |
| Mechanism | Backward quadratic approximation, natural-parameter update and Gaussian propagation |
| Expected failure | Negative precision or a positive but extreme proposal; recursive arithmetic amplification; source/reference mismatch |
| Numerical pass criterion | Same-input identities and existing full-recursion tolerances pass; complete finite forward/backward execution; no silent fallback, clipping or target change |
| Safety-extension pass criterion | Identical outputs when the protection is inactive; bounded and explicitly flagged updates on the calibrated failure cases, including means/covariances and finite source RK4; independent downstream validation |
| Promotion veto | Invalid precision, nonfinite values, unexplained reference discrepancy, missing telemetry, silent algorithm changes, or heuristic underperformance for the claimed task |
| Continuation veto | Corrupt/unverifiable inputs, invalid comparator, unsupported hardware/backend, exhausted aggregate budget, or a change of scientific target requiring a new decision |
| Repair trigger | A candidate fails a validity or safety check; localize and continue the stated next phase if its assumptions and budget remain valid |
| Explanatory only | ESS, regression residual, condition number, fit-cloud coverage, mean displacement, individual runtime and frozen-case counterfactuals |
| Prohibited conclusion | A smoke or fixed-input pass proves full replication, statistical superiority, default readiness, HMC suitability or a flaw in controlled-SMC theory |

## Ordered repair and investigation phases

### R1. Preserve the first failed operation

Add ClassA telemetry to the actual consumer, preserving refit/time/replica
indices for the raw and accepted A,b,c, precision spectrum, mean, covariance,
fit center/scale, fit residual/rank, normalizer components, projection and
fallback branch. Record dimensionless mean displacement relative to the
previous covariance and the fitted cloud. Distinguish a rejected fitted policy
from the last valid policy; do not count a rejected/fallback update as a
successful adaptive refinement.

Propagate existing validity from `multivariate_twist` through `_forward` and
stop a rejected row before softmax/resampling or further RK4 work on invalid
values. Stop the backward loop after a failed required integral/factorization,
preserving the valid suffix and the first failure. If per-row continuation
requires a larger refactor, an explicitly reported batch stop is acceptable
for the initial diagnostic. No silent substitution may become an accepted
result. Assess validity only for the branch whose answer is selected in the
projected-integral helper; reproduce any unused-branch false veto first.

Verification: identical healthy outputs and ancestry with the same recorded
randomness; injected invalid precision/nonfinite lookahead terminates at its
first occurrence; stable trace counts and memory growth on the real GPU/XLA
consumer. An early finite-only check detects failure but cannot by itself
establish adequate proposal geometry. These guards alter failure handling,
not the accepted scientific object.

Deliverable: first-failure JSON plus a call-chain test and small no-fire
regressions. This phase is an engineering repair, not a filter admission.

### R2. Close the unresolved arithmetic/reference discrepancy

Start from 114, not a new stochastic run. Freeze the earliest common input
before the first A,b,c mismatch. Compare actual GPU/XLA, CPU graph and an
independent SVD/high-precision calculation, separating: RK4 output, old/new
Gaussian parameters, log integral, regression target, centered solve,
coordinate conversion and projection. Advance one backward step at a time
only after that operation agrees. Preserve the first divergence rather than
comparing only the final refit.

Compare the represented polynomial on independent points, induced means and
covariances, incremental weights and log normalizer as well as coefficients.
A constant policy gauge can cancel analytically; it must be demonstrated
through the complete normalizer bookkeeping, not used to excuse arbitrary
coefficient drift. Keep existing admission tolerances. If a revised numerical
error criterion is needed, derive it before interpreting the failing case and
validate it independently; do not widen tolerances to obtain a pass.

Test the algebraic identity confirmed in 129 as a numerical reformulation:
fit the absolute target `-log g - log f(psi_new_next)`, equivalently
fit only `-log f(psi_new_next)` and add the exact observation quadratic
analytically. This removes cancellation of the old
policy in exact arithmetic. It is not a remedy for global extrapolation.
Also check forward absolute Gaussian integrals: the previous stable correction
fix did not replace every expanded normalizer in the forward consumer.

Deliverable: a first-divergence record with 80-digit authority where needed,
a localized numerical correction if justified, and independent complete
backward-plus-forward validation. If the reference is wrong, repair it with
an independent derivation. A fixed-target coefficient replay alone cannot
close this phase.

### R3. Attribute upstream projection and evaluate one complete-update protection

First use frozen clouds to split observation curvature, future-integral
curvature and previous-policy cancellation at the earliest offending time.
Compare the raw fit, actual projection and candidate correction with identical
next-time inputs. Propagate a changed next-time policy one step backward to
measure its effect. Do not integrate an indefinite Gaussian just to manufacture
an “unprotected counterfactual”; restrict counterfactuals to valid distributions.
Report whether negative curvature is present before the relevant projection,
is amplified by it, or arises through changed cloud coverage.

The first extension candidate should scale the **whole** policy increment:

\[
A_\alpha=A_{old}+\alpha\Delta A,\quad
b_\alpha=b_{old}+\alpha\Delta b,\quad
c_\alpha=c_{old}+\alpha\Delta c,\quad 0\le\alpha\le1.
\]

Positive-definite precision is necessary but insufficient; feasibility-only
backtracking already failed in 78/79. Also constrain a calibrated measure of
Gaussian displacement, for example the exact divergence

\[
D_{KL}(N(m_\alpha,C_\alpha)\Vert N(m_o,C_o))
=\tfrac12[\operatorname{tr}(C_o^{-1}C_\alpha)
+(m_\alpha-m_o)^TC_o^{-1}(m_\alpha-m_o)-d
+\log\det C_o-\log\det C_\alpha].
\]

Evaluate this for the initial Gaussian and the declared transition conditioning
clouds; also report absolute/relative covariance and fixed-probe RK4 validity.
A finite set of probes is not a global guarantee over the state space.
At alpha0 the previous policy is retained. If only that choice is safe, record
an incomplete/rejected adaptation, not successful execution of four refits.

The trust budget, backtracking schedule, absolute covariance limits and choice
to disable protection are scientific hypotheses, not inherited defaults.
Calibrate them with model-trust/non-harm curves: predicted local approximation
versus actual future-integral values on heldout probes, Gaussian displacement,
and valid transition behavior. Do not select them to optimize Figure 6 variance,
filtering MSE or the heuristic table. No particular radius is justified yet.
Freeze the calibration before fresh validation.

A secondary candidate is a projection centered on an explicit anchor mu,
adding `(x-mu)'D(x-mu)` so global coefficient changes are
`Delta A=D, Delta b=-2D mu, Delta c=mu'D mu`. It preserves the added term's
value/gradient at mu, unlike an origin-centered curvature correction. This
does not guarantee global safety and should be investigated only if the first
candidate's measured failure motivates it. A prior-whitened precision lower
bound would control maximum covariance more directly than the existing
covariance lower bound, but still needs a mean constraint.

Deliverable: derivation, fixed-input parity, measured calibration, and an
explicit extension label. Reject a candidate for the specific question it
fails; a covariance floor's failure to control mean displacement is not proof
that covariance floors have no numerical use.

### R4. Validate the complete adaptive method on fresh inputs

Freeze all choices and use disjoint calibration and validation randomness.
The saved failed cases are permanent debugging regressions, not holdouts.
Validate complete recursion independently, then scalar versus batch and actual
GPU/XLA endpoint parity. Include forcing near the data-generating4.8801,
the difficult2.5 boundary, the opposite8.5 boundary, strongly observed and
weakly observed directions, and both inactive/active protection cases. Keep
the exact source forcing grid distinct from an added diagnostic at4.8801.

Construct these cheap adversaries and retain them without selecting against
them: BPF at the same particle count (no lookahead), observation-only APF
(exact Gaussian one-step adaptation), the last valid controlled policy
(tests whether the next update helps), and a cost-matched APF (tests whether
adaptation earns its computational cost). Report conditional results rather
than pooling the difficult forcing/activation regimes. For an additional
linear-Gaussian control, exact Kalman likelihood is an oracle. No exact
Lorenz oracle is presently established; do not label a large particle run exact.

Preserve the source claim: Figure 6 concerns normalizer-estimation variance,
not intermediate filtering-mean MSE. Include preparation and fitting cost in
efficiency comparisons. If filtering performance is claimed separately, use
its own conditional baseline/oracle checks. The existing filtering veto in
the RBF extension does not refute a different normalizer claim.

Predeclare multi-seed failure-rate reporting and simultaneous paired uncertainty
for any comparative claim before observing the final seeds. A small fresh
pilot can veto or nominate, never establish superiority. Do not silently drop
failed repetitions; show failure counts and the missing-estimate policy.

### R5. Resume empirical replication and the independent KDM work

Only after the applicable source/numerical track passes should the full
Figure 6 matrix launch. Benchmark the validated batched implementation and
forecast the entire matrix including source controls, fitting and compilation.
The previous scalar estimate37.23hours exceeds the approximately 32.26hours
left; enough compute is not yet established. Do not substitute a short smoke
for the requested matrix or spend the remaining allowance on another blind run.

Keep source-M16 RBF/Table2 separate from the all-positive-coefficient repair.
Its NNLS/stopping/top16 target needs its own exact procedure and independent
comparison before an empirical replication claim. KDM/IWSG needs an explicit
author training/data/evaluation protocol and target-specific objective,
architecture/optimizer search, seeds, heldout criteria and downstream tests.
Primitive values/gradients and sequential Kalman controls do not complete it.
Do not make KDM implementation depend unnecessarily on solving Lorenz.

## Budget, execution readiness and defaults

After129 the current tranche has 39.064949CPU seconds and 3174.823943GPU seconds;
the inherited allowance has 292.231885CPU seconds and 116126.124294GPU seconds.
Attempts129/130. These are balances, not a new grant.

Proposed next executable diagnostic tranche: at most six attempts,30 aggregate
CPU seconds and 300 aggregate GPU seconds, drawn from these balances. Within
that ceiling, reserve at most60GPU seconds for R1,120GPU seconds for R2 and
120GPU seconds for frozen-input R3 attribution; CPU references share30seconds.
A localized retry consumes the same totals. Before launching, implement the
minimal scripts, register these aggregate limits and a reconciled attempt
ceiling in the existing dispatcher, and record the exact commands. Expanding
an administrative attempt counter within the unchanged scientific/compute
contract does not require a new human approval; a material compute or target
change does. If a minimal probe exceeds its cap, preserve it and revise the
costed stage rather than extending a process silently.

R3 calibration and R4/R5 are conditional specifications, not implemented launch
commands or a promise that unknown repairs work. Their budgets and exact
commands must be filled after the preceding diagnostic gives a valid candidate
and measured cost. This user request asked for diagnosis and a proposed plan;
the completed execution here is the bounded forensic audit, not a new long
campaign. Continue to use the established dispatcher command prefix and
escalated GPU permissions, without unrelated allow-list changes.

| Material choice | Provenance and status | Failure risk and earliest check |
|---|---|---|
| d8,p6,H101,Q0.001I,C0=0.01I,R0.0001I and source RK4 | Published Section 6.2 and author constructor; replication baseline | Wrong model would invalidate every comparison; fixture/hash and source equation checks |
| F2.5,N512,four refinements in the forensic case | Saved76/121/126 failure; diagnostic fixture | Overfitting one failure; frozen diagnosis only, fresh source-grid validation later |
| FP64,TF32 off,GPU/XLA with growth | Current numerical validation baseline | Hidden GPU/precision or altered memory policy; escalated device/helper checks before run |
| Centered QR and independent SVD | Same full quadratic least-squares target; numerical candidate | Difference from source fallback or coordinate cancellation; matched-input and full recursion checks |
| Diagnostic tolerance1e-6+1e-8 abs(reference) |128/129 explanatory replay only | Could hide small propagated errors; never substitutes for existing admission checks |
| Covariance floor0.01, trigger0 or0.001 | Rejected local safety hypotheses | Does not control mean; checked spectrum and proposal geometry show the failure |
| KL/displacement limit and damping | Uncalibrated extension hypotheses, including the off setting | Arbitrary smoothing or suppressed adaptation; model-trust/non-harm curve before freeze |
| CPU NumPy/SciPy/mpmath | Independent reference exception only | Cannot be promoted to runtime; actual TF endpoint must pass paired checks |

## Skeptical review of the proposed phases

This plan uses the actual failed consumers and source mathematics. It preserves
the difference between a raw author implementation and a stabilized extension,
does not use residual/condition number/ESS as efficacy criteria, and diagnoses
the still-open114 mismatch before long comparisons. Its stop conditions separate
invalid evidence from a candidate rejection. It includes fresh/conditional
comparisons and cost limits, and keeps KDM and RBF scope visible. Unknown
protection thresholds remain explicitly unpromoted. The main remaining risks
are amplification from earlier local projections, uncertainty in exact MATLAB
solver semantics and inadequate cost for the complete matrix. These are
assigned concrete diagnostics rather than treated as settled explanations.
R1/R2 and frozen R3 attribution are ready for implementation; calibration and
empirical execution remain conditional on their findings.
