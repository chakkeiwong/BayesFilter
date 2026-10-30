# Lorenz repeated-refinement and APF validation — 30 September 2026

## Question and source contract

Can the existing centered-QR Gaussian controlled-SMC consumer execute the
paper's adaptive repeated-refinement procedure, with the correct APF
initialization and comparator particle counts? This is an implementation
validation before Figure 6, not a stochastic performance comparison.

Primary sources inspected: published paper section 4.3 (policy accumulation),
section 6.2, equations (42)–(44), local published text lines 1149–1196;
pinned author commit 2696f6d611aae172d10102cb7bfbfd314bb6b88f:
Lorenz/cSMC_Resample.m:187–258 (backward fitting, additive quadratic policy,
strict ESS stopping), Lorenz/APF_VaryF.m:45–77 and96–107 (APF policy and
counts), and Lorenz/cSMC_VaryF.m:72–85 (N512 and maximum four refinements).

APF means initial policy psi_t=g_t. The source omits the observation Gaussian
constant; preserve its coefficient convention and restore the known prefix
normalization in reported likelihoods. Covariance, transition, and initial
Gaussian constants remain in the consumer. The algorithm must refine at least
once, and stop only after min_t ESS_t*100/N is strictly greater than90, or
after four refinements. APF comparison counts map refinements1,2,3,4 to
N1382,2867,4505,6144. These are source CPU choices, not matched GPU costs.

The existing centered QR and exact Gaussian-ratio recursion are the scoped
equivalent-objective numerical repair validated at32; do not revert to unstable
normal equations or claim raw author coefficient replay. Retain all fail-closed
rank and positive-precision guards. No damping, ridge, projection or other
objective change is introduced.

## Research intent and evidence contract

- Candidate: stable-signature TensorFlow/XLA controller calling the existing
  MultivariateGaussianControlledSMC.forward and .fit, with a native
  tf.while_loop over refinements. No copied scalar fitting or forward kernel.
- Comparator: independent NumPy/SciPy full-rank SVD reference, explicitly CPU
  diagnostic-only, extending the already checked reference at31. It must not
  be imported by the TensorFlow consumer.
- Cases: source101-observation dataset, d8, observation variance1e-4,
  initial covariance.01I, transition covariance.001I, N512; forcing2.5,4.8801,8.5.
  One predeclared common noise design across forcing values isolates wiring,
  not a population comparison. Maximum four refinements, source adaptive stop.
- Primary pass: unchanged numerical tolerance atol1e-7+rtol1e-9 for policy,
  particles, weights, logZ and ESS; exact ancestry and iteration count where
  the reference ancestry is away from rounding boundaries. Any ancestry
  mismatch is recorded as failure, not waived post hoc.
- APF validation: use the same forward consumer at all four source particle
  counts, central forcing, no refinement; compare full-horizon outputs to
  independent reference. Check APF initial conditioning against the closed
  Gaussian formula and strict90%/at-least-one/max-four boundary cases.
- Promotion veto: any nonfinite value, invalid precision/rank, wrong stopping,
  incorrect count mapping, unexpected retrace or parity failure blocks wider
  execution. A reference failure invalidates that fixture, not the scientific
  direction. Preserve a partial result and diagnose before dependent execution.
- Continuation veto: wrong data/target, corrupted artifact, invalid oracle or
  exhausted stage budget. Local harness failures may be repaired and retried
  within the same total budget. Do not relax scientific checks to obtain PASS.
- Explanatory only: per-iteration ESS, residuals, condition/precision margins,
  iteration counts and timing. Neither a higher ESS nor a successful
  controller qualifies a performance claim.
- Not concluded: Figure6/Table1 replication, variance reduction, cost parity,
  filtering superiority, general solver robustness, default or HMC readiness.
- Artifacts: versioned campaign attempt directories, binary reference arrays
  with shape/dtype/SHA256 metadata, summaries, source/input hashes and logs.
  CPU fixtures use explicit GPU hiding; GPU validation records growth, device,
  allocator peak, FP64/TF32off, JIT and trace counts.

Heuristic gate is reserved for empirical quality claims, which are excluded
from this phase. The next Figure6 design must construct and compare BPF,
source APF, cSMC and a declared deterministic filtering baseline where
filtering means are assessed; it must include uncertainty and full fitting
cost. This identity/parity check is not evidence that cSMC clears that gate.

## Defaults, audit and pre-mortem

| Choice | Provenance / status | Failure risk | Early diagnostic |
|---|---|---|---|
| Three forcing values | Source grid endpoints plus generating parameter; validation cases | Interior regimes remain untested | Report limited scope before full40-value matrix |
| One shared noise design, seed202609360 | Fixed-input diagnostic convenience | Does not estimate stochastic superiority | No ranking; full call-chain reference parity only |
| Centered QR, no ridge | Existing repaired objective checked at32 | Later refinements lose rank or positive precision | Fail closed at every fit; preserve minimum margins |
| FP64, TF32off | Source-equivalence reference exception | Does not establish default production precision behavior | Record exact backend and compare with SVD |
| ESS strictly above90%, at most4 | Paper/source, reviewed default within this replication | Off-by-one, premature stop, or unreported failed iteration | Boundary controls plus reference iteration counts |
| Source APF counts | Author CPU time matching | False equal-GPU-cost claim | Explicit source-count label and measured timings |
| Fixed-size graph buffers for5 stages | Engineering choice including initial APF | Unexecuted entries mistaken for evidence | Executed-stage mask and iteration count; compare only executed entries |

Skeptical audit: PASS for bounded engineering validation. The correct
comparator is the source APF, the stopping rule is source anchored, and no
proxy metric promotes the algorithm. The reference and candidate share the
declared model and random inputs but use distinct implementations/solvers.
A parity pass could still hide shared algebra error; the APF closed-Gaussian
check and earlier normalized Gaussian identities provide independent controls.
A failure could reflect conditioning or discontinuous resampling instead of
wrong recurrence; save per-stage policies, ancestry and numerical margins.

## Commands and budget

Within the already authorized inherited campaign, allocate at most180 CPU
seconds total to c1_lorenz_refinement_reference and450 GPU seconds total to
c2_lorenz_refinement_validation, at most two attempts each. Expand the current
CPU tranche300->500 seconds, which fits the inherited CPU469.391272 seconds
remaining after68; retain GPU18000 seconds and the inherited overall balances.
Increase the total attempt ceiling68->72 only for these bounded phases.
Before new runs, current use is CPU283.775664/GPU13890.532955 seconds.
No new48-hour grant, external compute or package changes.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_refinement_reference
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_lorenz_refinement_validation

Reference first; GPU phase depends on its PASS. Both depend on the prior
validated Lorenz source consumer. Close with a result note and refresh the
master/checkpoint before planning the empirical matrix.


## Attempt70 finding and bounded localization

Reference69 passed in13.003 CPU seconds. GPU70 completed all three forcing
cases and all four APF counts in76.881 seconds:130/144 checks passed.
All stopping counts (4,1,3), ancestry, forward-output comparisons and
closed-Gaussian APF checks passed at their declared tolerances. Fourteen raw
A/b coefficient comparisons failed (maximum A difference3.70e-7,
b difference4.11e-6). Attempt70 remains FAIL; no tolerance is relaxed and
Figure6 execution remains blocked.

Before changing the solver, diagnose identical inputs. New named phase
c1_lorenz_refinement_diagnosis, <=45 CPU seconds, one attempt, within the
current CPU500/GPU18000 allocation and ceiling72: decode the saved GPU
stage0 clouds/weights/policy with GPU intentionally hidden; rerun the existing
independent SVD backward recursion on those identical inputs. Compare both
original and refrozen coefficients to the GPU stage1 fit, plus the represented
quadratic, gradient and scaled curvature on the training cloud. Preserve the
first-divergence location, input discrepancy and raw data. This distinguishes
input amplification from solver/recursion mismatch without changing the target.

All magnitude comparisons here are explanatory diagnostics; no new promotion
criterion and no empirical run follows automatically. The diagnosis must be
finite, shape-correct, input-hash-bound and must emit all requested comparisons.
A remaining coefficient difference triggers numerical/representation diagnosis,
not an inference that controlled-SMC theory fails. A large input-amplification
effect can justify a new matched-input validation, but cannot erase70.
Skeptical audit PASS: same baseline objective, frozen inputs, bounded cost,
no stochastic ranking or post-hoc relaxation.


## Identical-input validation after localization, attempt72

Attempt71 localizes the first-refinement discrepancy to different forward
inputs: the three CPU/GPU clouds differ by at most8.0e-15 and log weights by
7.46e-10. Independent SVD fits on the actual GPU inputs satisfy the unchanged
raw-coefficient tolerance in all three cases (largest error/tolerance0.046).
Thus70 does not establish a QR-versus-SVD solver defect. It remains a failed
independent-trajectory coefficient comparison.

The discriminating next check is every executed fit, not only the first one.
Phase c1_lorenz_refinement_matched_validation uses the saved GPU inputs and
outputs of all eight refinements (4,1,3), and the independent SVD recursion.
Its primary criterion is elementwise abs(error)<=1e-7+1e-9*abs(reference)
for every A, b and c coefficient, without changing this tolerance. It also
requires finite inputs, full-rank fits, positive precision, and exactly eight
checked refinements. Require every existing non-coefficient check from70's
saved check records to have passed; preserve their source hash and the
original FAIL. Report represented-quadratic discrepancies only as explanatory
diagnostics, not replacement acceptance criteria.

This is an explicitly revised, modular validation contract: identical-input
fit parity plus the existing independent forward/APF/stopping parity. It can
establish scoped implementation agreement. It cannot establish raw coefficient
agreement between separately rounded trajectories, population performance,
Figure6 reproduction, statistical superiority or a new default. The old
failed comparison must remain visible in the resulting decision object.

Budget: one attempt,45 CPU seconds, within the existing CPU500/GPU18000
tranche and attempt ceiling72. GPU devices are deliberately hidden. No
algorithm, tolerance, dataset, random design, or accepted numerical protection
changes. Any coefficient failure blocks this revised validation and triggers
localization at the first failing refinement; it does not reject the method.
The input manifest binds69's fixtures,70's tensors and summaries, and this
plan. The result is preserved under attempt72 with the independent fitted
policies and checks. This diagnostic reference must never be imported by a
TensorFlow runtime lane.

Skeptical audit PASS:71 supplies a measured explanation for the changed
comparison design, every actual call-chain fit is checked, forward behavior
retains its original criterion, and no failed70 check is silently reclassified.


## Terminal outcome

Attempt72 passed70 checks over all8 fits; maximum raw coefficient error/tolerance
0.045633. Attempt70 remains FAIL. See controlled-smc-lorenz-refinement-result-20260930.md.
The next authorized step is the bounded Figure6 cost pilot, not empirical promotion.
