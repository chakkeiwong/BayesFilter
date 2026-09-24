# C2 Mixture-UKF/APF Phase 4 Execution Plan

Date: 2026-09-03  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Status: `PHASE4_REPAIRED_VALIDITY_PROMOTION_VETO_PHASE5_READY`

## Research question

Does a fixed-size, smoothly data-dependent defensive proposal repair the weak
tail coverage of the Phase 3 local UKF mixtures while preserving the exact C2
finite target, complete APF/importance denominator, and frozen analytical
score?

This phase tests a proposal law, not a new target. It cannot establish
posterior correctness, unbiased likelihoods, universal efficiency, or a new
default.

## Fixed mathematical object

For each ancestor (j), Phase 3 supplies a local Gaussian mixture

\[
q^L_{t,j}(x)=\sum_{k=1}^{K}\pi_{t,j,k}
  \mathcal N(x;m_{t,j,k},P_{t,j,k}),
\qquad \sum_k\pi_{t,j,k}=1.
\]

The defensive component is a Student law with the UKF posterior mean and
covariance:

\[
r_{t,j}(x)=t_{\nu}\!\left(x;\,\mu_{t,j},
  \frac{\nu-2}{\nu}P_{t,j}\right),\qquad \nu>2.
\]

Let

\[
\kappa_{t,j}=e_{t,j}^{\mathsf T}S_{t,j}^{-1}e_{t,j},
\quad
\epsilon_{t,j}=\epsilon_{\min}+(epsilon_{\max}-\epsilon_{\min})
  \operatorname{sigmoid}\!\left(\frac{\kappa_{t,j}-c_0}{\tau}\right),
\]

where (e) and (S) are the UKF innovation and innovation covariance. The
proposal used for sampling and weighting is the complete density

\[
q_{t,j}(x)=(1-\epsilon_{t,j})q^L_{t,j}(x)+
           \epsilon_{t,j}r_{t,j}(x).
\]

The APF ancestor law remains

\[
a_{t,j}\propto \bar w_{t-1,j}\widehat\ell_{t,j},
\]

and a draw (J\sim a_t, X\sim q_{t,J}) receives the unchanged exact weight

\[
\widetilde w=\frac{\bar w_{t-1,J}f_t(X\mid x_{t-1,J})g_t(y_t\mid X)}
                   {a_{t,J}q_{t,J}(X)}.
\]

If a local or defensive label is sampled, that label is not used to replace
the denominator: every Gaussian component and the Student component remains in
the log-sum-exp. The Student scale is chosen so its covariance is (P_{t,j}),
but its heavier tails change the proposal density away from the center.

The first route freezes (\epsilon_{t,j}), UKF moments, component parameters,
random tensors, rows, and topology at the reference point. Consequently the
reported score is the analytical derivative of this frozen finite program. A
future adaptive-total route would have to differentiate the sigmoid, UKF
moments, Cholesky factors, and both proposal densities; it is outside this
phase.

## Candidate controls and calibration

The local component counts are (K\in\{1,2,4\}), with the Phase 3 fixed
Cholesky-column split for (K=2,4). The predeclared defensive ladder is

| ID | \\(\\nu\\) | \\(\\epsilon_{\\min}\\) | \\(\\epsilon_{\\max}\\) | \\(c_0\\) | \\(\\tau\\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tail_a` | 5 | 0.05 | 0.20 | D | 2D |
| `tail_b` | 8 | 0.05 | 0.20 | D | 2D |
| `tail_c` | 5 | 0.10 | 0.30 | D | 2D |
| `tail_d` | 8 | 0.10 | 0.30 | D | 2D |

Here (D=4) is the declared C2 state dimension. A separate stateless,
model-generated calibration bank (`N=64`, horizon 10, seed `(20260903, 7401)`)
selects the candidate with the largest calibration minimum ESS among rows that
pass all support, normalization, complete-density, and score-free proposal
checks. The claim observation sequence is never used for this choice. The
selected ID and all controls are then frozen for an untouched claim run.

The calibration also records epsilon range/entropy, Student and Gaussian log
density recomposition, component support, and observation response. Selection
is not repeated on claim branches.

## Evidence contract

### Hard vetoes

- target, conditioning, or observation mismatch;
- nonfinite or nonpositive Student/Gaussian scale;
- incomplete mixture denominator or selected-component shortcut;
- proposal or base-mass normalization failure;
- unsupported samples or nonfinite exact target/score;
- central finite-difference, non-JIT/XLA, or call-chain parity failure;
- no measurable observation response in UKF moments or epsilon;
- missing branch records, corrupted artifacts, GPU/memory-growth/XLA provenance
  failure, or exhausted budget.

### Promotion screen

The defensive candidate must have a paired 95% interval for the log minimum-ESS
ratio against K=1 that is positive with at least 10 of 12 contrasts positive,
and must not lose to a cheap adversary in the predeclared salient times. This
is only a mechanism nomination. With twelve branches, all continuous
comparisons remain descriptive unless a declared uncertainty calculation
supports a ranking.

### Explanatory diagnostics

Per-time ESS, maximum normalized weight, epsilon spread and entropy, Student
tail usage, log-likelihood, runtime, compile/retrace warnings, and proposal
overlap are explanatory unless the promotion screen explicitly names them.

## Scope and budget

The claim run is fixed at (N=8192), twelve paired branches, horizon 20, the
same observations/seeds/target as Phase 3, and the same RTX 4080 SUPER GPU/XLA
lane. It evaluates the three defensive local arms plus the eight Phase 3
comparators (eleven families, 132 records) only if the calibrated proposal
passes. If cost exceeds the remaining campaign allowance, preserve the partial
records and close with a bounded cost veto; do not silently reduce branches or
rows. Larger particle counts remain closed.

## Implementation contract

1. Add one repository-owned fixed-shape TensorFlow/XLA sampler that accepts
   Gaussian local components, a Student component, smooth epsilon rows, and
   explicit random tensors, and returns the complete log density.
2. Reuse the generic batched UKF kernel and the shared
   `FrozenProposalAPFProgram`; the C2 adapter may construct the transformed
   observation but may not duplicate the exact evaluator.
3. Add focused CPU tests for Student density normalization conventions,
   complete-mixture recomposition, epsilon observation response, support,
   frozen score finite differences, and XLA/eager parity.
4. Add a Phase 4 driver mode with an independent calibration artifact,
   checkpointed claim records, a manifest, decision/inference tables, and a
   post-run red-team note.

## Pre-execution skeptical audit

The plan was reviewed before implementation:

- The baseline ladder is unchanged and includes bootstrap, transformed Student,
  Gaussian hint, stationary, retained TT, and K=1/K=2/K=4 arms.
- The exact numerator and APF (w/a) correction are unchanged; only the
  proposal denominator and sampling law are extended.
- Calibration data are disjoint and model-generated; claim data cannot select
  (\nu), epsilon bounds, or gate temperature.
- Complete-mixture density and label permutation are explicit hard checks.
- The frozen score contract avoids an unimplemented adaptive-total derivative.
- Record coverage, GPU provenance, finite checks, and a six-hour total campaign
  budget are explicit stop conditions.
- A low ESS or candidate loss is a promotion veto/repair trigger, not a
  continuation veto. A target or score mismatch remains a true blocker.

Audit disposition: `PASS_FOR_BOUNDED_PHASE4_IMPLEMENTATION_AND_CALIBRATION`.

## Implementation and calibration close before GPU launch

The generic sampler, C2 adapter, driver mode, and focused tests are complete.
The CPU-only focused Phase 4 suite passed `8` tests, and the broader C2
regression passed `39` tests. A separate calibration pilot on the independent
stateless bank passed all four configurations across K=1, K=2, and K=4. The
selection rule chose `nu5_eps05_20` (`nu=5`, epsilon bounds `0.05` and `0.20`);
its worst calibration minimum ESS was `14.8526`. During the pre-launch audit a
comparator proposal-set key mismatch was found in the Phase 4 driver and
corrected; the regression was rerun after the repair. The first GPU claim run
then completed all twelve paired branches and eleven families (132 records).

## Post-run validity finding and repair contract

Five records from the first GPU claim attempt were marked invalid solely by
`apf_w_over_a_identity`. Their absolute residuals were
`2.3283064365386963e-10` through `9.313225746154785e-10`, while every other
required check passed. The checked expression is the floating-point
rearrangement

\[
 a+q+(\gamma-a-q)-\gamma=0.
\]

Because the realized log terms can have magnitudes of millions, an absolute
`2e-10` cutoff is dimensionally wrong for this cancellation diagnostic. This is
a localized numerical-validity diagnostic defect, not evidence that the
proposal law or exact target failed. The absolute residual remains recorded;
the repaired gate is the normalized backward error

\[
 \frac{|a+q+(\gamma-a-q)-\gamma|}
 {\max(1,|a|+|q|+|\gamma-a-q|+|\gamma|)}
 \le 32\,\varepsilon_{64}.
\]

The repair also recomputes the final normalized log weights through the
independent recurrence and compares them with the shared evaluator using the
same scale-aware bound. This adds an independent parity check rather than
loosening the target or proposal calculation. The target, observations, seeds,
proposal controls, rows, branches, hardware class, and promotion rule remain
unchanged.

The repair replay is a fresh, versioned GPU artifact with all 36 defensive
records (`K=1,2,4`, twelve branches); the 96 unchanged comparator records are
retained from the immutable first attempt by hash and are not silently
rewritten. Replay parity must match each replaced record's branch/program
identity and finite scientific outputs. A failure of the repaired backward
error or recurrence parity is a true Phase 4 validity veto. Passing validity
does not remove the already observed candidate-efficiency and heuristic
promotion veto.

The renderer reports evaluated and valid record counts separately, and computes
the predeclared paired log-minimum-ESS screen against K=1 plus the conditional
cheap-adversary table at zero-based `t=3`, `t=4`, the observed minimum-ESS time,
the largest mean UKF innovation-quadratic time, and the final time. These are
promotion diagnostics; they do not become tuning targets.

## Planned artifacts

Implementation and tests remain in the repository-owned modules and
`tests/highdim/`. The claim output will be a fresh directory under
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-.../`, with a
formal close note linked from the governing master plan.

## Repair execution close (2026-09-04)

The fresh replay `phase4-backward-error-repair-attempt01` completed 36/36
defensive records on the declared GPU/XLA lane. All replay records passed the
repaired finite-program checks, all 36 branch/program parity rows matched the
immutable parent, and the maximum normalized identity error was
`1.1102224395112269e-16` against the bound `7.105427357601002e-15`. The
maximum final-log-weight recurrence error was `2.9297083281690912e-15`.

The original five absolute-residual failures are therefore closed as a
localized numerical-validity diagnostic defect. The defensive arms still fail
the predeclared ESS/heuristic promotion screen (5/12, 4/12, and 4/12 positive
primary contrasts for K=1, K=2, and K=4 respectively); this is a
`candidate_failure`, not a continuation veto. The complete disposition,
source/fixture identity check, MathDevMCP result, Lean result, budget, and
next command are recorded in
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-backward-error-repair-close-20260904.md`.

The stale pre-repair `result.md` renderer is not used to reinterpret the raw
replay. Its machine `result.json`, branch file, parity file, and hashes remain
immutable evidence; the close note supplies the corrected metadata reading.
