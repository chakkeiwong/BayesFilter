# C2 Mixture-UKF/APF Phase 5C-R: Replicated and Recursive Hybrid Validation

Date: 2026-09-04  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Prerequisite: Phase 5C hybrid mechanics pass  
Status: `EXECUTED_PASS_REPLICATION_INTEGRATED_READY`  
Scope: three-seed, two-step fixed-map hybrid representation validation  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Research question and boundary

Does the mechanically valid Hermite-plus-RBF representation remain finite and
useful when (a) the reference banks change and (b) the lagged moment-derived
coordinate map is rebuilt after each observation update?

This phase is a replication and recursive representation diagnostic.  It does
not test proposal ESS, posterior correctness, HMC readiness, an analytical
total gradient, or a production/default choice.  Three seeds and two
recursive steps are enough to expose obvious instability, but not enough to
support a statistical ranking of widths or a claim of generalization.

## Exact recursive finite program

For each seed, initialize the same C2 stationary cloud and assimilate the
first observation exactly as in the closed Phase 5 map probe.  At step
\(t=1,2\), with parent states \(x_{t-1,j}\) and normalized weights
\(w_{t-1,j}\), construct conditional transition moments
\[
 m_{t,j}=F x_{t-1,j},\qquad V_{t,j}=Q,
\]
then form the checked map \(x=m_t+L_tu\) from the weighted law of total
covariance.  The exact target used for fitting is
\[
 \gamma_t(x)=\sum_j w_{t-1,j}
       \mathcal N(x;m_{t,j},V_{t,j})g(y_t\mid x),
 \qquad
 h_{\star,t}(u)=\gamma_t(m_t+L_tu)|\det L_t|/\eta_4(u).
\]
The fitted square-root target is \(s_{\star,t}=\sqrt{h_{\star,t}}).  After
the representation diagnostic, generate the next finite cloud with the same
frozen standard-normal bank, evaluate the exact observation density, and
normalize its weights.  This is a deterministic finite recursion for the
diagnostic; it is not asserted to be an unbiased particle filter.

At every step compare the fitted Gram contraction
\[
 Z_{H,t}=\int h_t(u)^2d\mu(u)
\]
with the independent predictive-mixture identity
\[
 Z_{T,t}=\mathbb E_{J\sim w_{t-1},\,X\sim
 \mathcal N(m_{t,J},V_{t,J})}[g(y_t\mid X)].
\]
The predictive audit uses a separate stateless bank and is never reused for
fit rows.

## Representation arms

Carry only the two Phase 5C arms that passed the condition screen.  The broad
width-3 arm remains in the prior result as negative conditioning evidence and
is not refit here.

| Arm | Hermite degree | centers | width | RBF constant | role |
| --- | ---: | --- | ---: | :---: | --- |
| `hybrid_d6_w075` | 6 | `(-2,-1,0,1,2)` | 0.75 | no | surviving narrow candidate |
| `hybrid_d6_w150` | 6 | `(-2,-1,0,1,2)` | 1.50 | no | surviving moderate candidate |

Use paired seed families `(20260904,701)`, `(20260904,1701)`, and
`(20260904,2701)` for reference banks, with map seeds `(20260904,501)`,
`(20260904,1501)`, and `(20260904,2501)`.  For each family derive disjoint
train, holdout, shell/audit, predictive-audit, and recursive-cloud streams by
fixed offsets recorded in the manifest.  Use `N=128` carried rows,
`512/512/4096` train/holdout/audit rows, and horizon two.  Keep rank 2, two
ALS sweeps, ridge `1e-8`, the Phase 5 shell radius, and the exact C2 model
calls unchanged.

## Evidence contract

| Field | Declaration |
| --- | --- |
| Question | stability of the hybrid representation across seeds and lagged map rebuilds |
| Exact comparator | independent predictive-mixture `Z_T,t` at each seed/step |
| Primary validity gate | every accepted record has finite exact target/map, analytic mass/cross parity, finite fit, SPD mass, condition below `1e14`, and complete banks |
| Recursive diagnostics | per-step shell/central RMS, `log Z_H,t-log Z_T,t`, map eigenvalue/condition, map shift, cloud/observation ESS, cumulative error |
| Uncertainty | paired seed means, standard deviations, and predeclared small-sample t intervals; ranking is not promoted with `n=3` |
| Promotion criterion | none in this phase; a valid arm is only a candidate for a later integrated run |
| Hard vetoes | target/map mismatch, missing/disjoint banks, contraction mismatch, no valid arm at a seed/step, nonfinite/corrupt records, or exhausted budget |
| Candidate failure | an individual arm condition/fit or recursive screen failure when another arm remains valid |
| Nonclaims | no proposal ESS, posterior, HMC, gradient, default, superiority, or general-model claim |
| Artifact | fresh `phase5c-replication-attempt02/` with manifest, per-seed records, result, close note, MathDevMCP/Lean sidecars, and test capture |

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| three seed families | minimum bounded replication after one-bank pilot | insufficient power for ranking | per-seed intervals and raw records | diagnostic design |
| two recursive steps | first nontrivial lagged-map rebuild | long-horizon instability is missed | map shift and per-step residual | diagnostic design |
| fixed standard-normal cloud bank | preserves the Phase 5 map contract | finite-cloud Monte Carlo noise | bank hashes and predictive SE | frozen comparator |
| widths 0.75 and 1.5 | both survived Phase 5C condition screen | either remains ill-conditioned on another seed | pre-fit eigenvalue/condition | candidate hypotheses |
| width 3 excluded from refit | declared negative conditioning result | a different width could recover | preserve prior record; no new claim | reviewed scope |
| rank 2, two sweeps, ridge `1e-8` | paired Phase 5 settings | underfit or ridge bias | fit/holdout/condition | frozen diagnostic |
| exact predictive audit | transition/observation identity | finite-audit error | independent bank and SE | primary comparator |

The ridge remains a frozen Class-C choice.  No seed or recursive result may
retune it.  Any change to rank, sweeps, ridge, rows, target, or map defines a
new plan.

## Required implementation and audit checks

1. Keep the generic `HermiteRBFBasis1D` and existing `ProductBasis`/`FixedTTFitter`
   call chain.  A driver-side wiring check must require the 12-channel basis,
   one constant, and the hybrid family for every accepted arm.
2. Build one fixed-shape exact target kernel per recursive step; do not use a
   selected-component denominator or an approximate target.
3. Check analytic full mass, integral, and cross blocks against two independent
   TensorFlow Jacobi quadrature orders before each fit.  A failed arm is
   retained and classified; the phase continues only if another arm and the
   global target/map checks remain valid.
4. Run the Phase 5C hybrid, RBF, Hermite, recursive-map, and Phase 4-repair
   focused tests on CPU with GPUs deliberately hidden.
5. Run MathDevMCP structural slices and the existing scoped Lean certificate;
   preserve their non-proof boundaries.
6. Run one GPU/XLA attempt with memory growth configured before logical-device
   initialization.  Record the complete command, source hashes, device,
   seeds, wall time, and dirty-worktree identity.

## Skeptical plan audit

Disposition before execution: `PASS_FOR_BOUNDED_PHASE5C_REPLICATION`.

The exact target, map construction, representation, and predictive comparator
are inherited without a scientific change.  Fresh seeds are fixed before
execution, and the width-3 candidate is not silently dropped from evidence;
its earlier condition failure is preserved.  The phase has an explicit
per-arm candidate-failure rule so one bad width cannot masquerade as a broken
algorithm, and an explicit no-ranking rule so three seeds cannot be treated as
statistical proof.  The recursive cloud is a finite diagnostic rather than an
unqualified particle-filter claim.  A global target/map/bank/contraction
failure, no surviving arm, corrupted artifact, or exhausted budget is the
continuation veto.

### Pre-mortem

| Misleading outcome | Distinguishing check | Action |
| --- | --- | --- |
| seed variation is mistaken for improvement | raw paired records and t intervals | report descriptive only; add seeds before ranking |
| recursive map hides target mismatch | exact transition/observation checks at every step | invalidate the affected step and repair wiring |
| a surviving arm is nearly singular | pre-fit mass eigenvalue and fit condition | classify arm failure; do not promote |
| direct normalizer uses fit rows | predictive bank hashes and independent sampling path | invalidate artifact and rerun |
| map recursion looks stable only because the cloud is frozen | record map shifts and observation-weight shifts each step | treat as finite-cloud diagnostic; require larger recursive study |
| GPU metadata does not identify the lane | endpoint wiring and memory-policy checks | repair metadata and rerun in a fresh directory |

## Budget and close rules

Budget: one focused CPU regression and one GPU/XLA attempt, each at most one
hour; localized harness/metadata repairs may retry in fresh directories under
the unchanged contract.  No proposal ESS ladder, Student reference law, or
large horizon is opened here.

At close, classify engineering correctness, numerical validity, and scientific
interpretation separately.  Continue if at least one arm remains valid at each
seed/step and all global checks pass, even when another arm fails or the
replicated intervals overlap zero.  Stop only for a true continuation veto.
Write the decision table, inference-status table, per-seed/per-step records,
uncertainty summary, repair history, MathDevMCP/Lean limits, and post-run
red-team note before refreshing the master plan.

### Implementation pre-launch audit (2026-09-04)

The implemented driver passed the skeptical recheck before launch.  Its exact
target kernel accepts fixed-shape parent states, weights, observations, and
reference rows through one `tf.function` signature; the moment kernel is also
shared across seeds and steps, and the CPU smoke emitted no retracing warning.
The endpoint calls the audited Phase 5C hybrid fitter and records a wiring
identity.  No rank, width, ridge, map, or bank decision is selected from
holdout values.  The only continuous summaries used for interpretation are
per-seed descriptive intervals, not promotion criteria.  The target/map,
bank-separation, finite/SPD, and budget checks remain explicit; a candidate
arm failure cannot be upgraded to a continuation veto.  This audit therefore
passes for the bounded GPU launch.

### Contract refresh after attempt01 (2026-09-04)

The preserved `phase5c-replication-attempt01/` run used `AUDIT_ROWS=4096` in
the driver while its launch-time plan snapshot declared `1024`.  This is an
implementation/plan contract mismatch in the audit-bank size, not a target,
measure, map, or representation mismatch.  Attempt01 remains immutable
diagnostic evidence, but it is not the authoritative contract-matched run.
The plan is refreshed to `512/512/4096`; the larger independent audit bank
reduces the predictive-mixture Monte Carlo error and changes no target, map,
arm, seed family, budget class, or promotion rule.  A fresh CPU regression and
GPU/XLA attempt02 are required under this refreshed hash.  The retry is within
the bounded campaign because it repairs a localized harness/contract issue
without changing the scientific question.

### Execution close (2026-09-04)

Attempt02 is the authoritative refreshed run.  It passed on GPU/XLA with
`12/12` finite seed/step/arm records, `audit_rows=4096`, exact target/map and
quadrature checks, disjoint streams, and the executable hybrid endpoint check.
The focused CPU suite reported `27 passed, 2 warnings`; the driver compiled;
MathDevMCP JSON was valid; and the scoped Lean certificate exited `0`.  The
three-seed summaries remain descriptive, so no width is ranked or promoted.
The complete result and close record are under
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5c-replication-attempt02/`.

Disposition: `PASS_PHASE5C_REPLICATION`; no continuation veto fired.  Refresh
the integrated phase with both surviving widths as frozen comparator arms,
nominate width `1.5` only as a warm start, and evaluate proposal quality with
the exact complete-mixture denominator.  The Student TT reference-law route
remains deferred until its measure, mass, row law, and derivative are built as
one consistent route.
