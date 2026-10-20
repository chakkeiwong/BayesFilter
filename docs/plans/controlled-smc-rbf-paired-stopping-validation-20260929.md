# RBF fitting with a shared finite stopping rule

This is a numerical implementation validation within the authorized controlled
SMC replication campaign. It continues attempts50–56; their results are frozen.
It does not change the least-squares objective, RBF dictionary, bandwidth,
top16 retention, Gaussian transition, corrected proposal or acceptance tolerances.

## Question and checked motivation

Can independent fresh-SVD and native TF/GPU/XLA QR implementations reproduce the
same complete fitted policy when both use stopping factor0.1, on particle seeds
not used in the two-case calibration?

Attempt54 rejected every GPU stopping factor against the fixed factor0.01
reference. Attempt55 reproduced that reference exactly, but showed that its own
top16 policy changes with stopping factor. Attempt56 compared equal factors:
factor0.1 passed both frozen cases, with maximum retained-log-policy difference
2.33e-10; factor0.01 failed both. This nominates a finite numerical procedure for
validation, not a more accurate fit, a statistical winner or a default.

The distinction matters: all settings approximate the same NNLS objective,
but different approximate coefficients need not define the same truncated policy.
The fixed0.01 comparison remains failed. Successful paired validation would
establish only reproducibility of the explicitly named factor0.1 procedure.

Source grounding was rechecked during58: author manuscript/supplement section6.1,
local text lines2817–2846, specifies natural-scale nonnegative least squares on
particle knots followed by selection of the M largest weights, and explicitly
warns about numerical stability. Pinned ToyMultimodal/cSMC_Resample.m
lines203–215 implement that sequence with lsqnonneg and descending sort.
Neither inspected passage supplies our stopping factor or a unique finite
coefficient representation. Factor0.1 therefore remains a local numerical
hypothesis. This is not evidence against the paper's controlled-kernel equations.

## Research intent and evidence contract

| Item | Predeclared meaning |
|---|---|
| Mechanism | Shared factor0.1 stopping, all512 NNLS columns, active capacity128 in TF, original KKT/feasibility guards, top16 source retention |
| Baseline | Fresh-SVD Lawson–Hanson independent reference at the same stopping factor, corrected pinned-author forward consumer |
| Scope | N512, M16, R1, bandwidth0.3 times sample SD,101 observations, one refinement, same published observed series |
| Fresh inputs | Particle seeds20261009/20261010; fresh TF forward seed20261011. These are new particle draws, not heldout observations or independent datasets |
| Primary pass criterion | Every existing full-horizon fit, retained-policy, Gaussian-integral and forward replay check passes; exact ancestry retained |
| Numerical tolerances | Prediction, target, retained log policy/integral atol1e-7/rtol1e-6; objective atol1e-8/rtol1e-6; forward existing atol1e-6/rtol1e-8; no relaxation |
| Promotion vetoes | Any failed consumer comparison, nonfinite, feasibility/KKT/rank/iteration/capacity failure, missing retained component or invalid fresh forward |
| Continuation vetoes | Missing or mismatched inputs/stopping factor, invalid independent reference, corrupt artifacts, budget exhaustion, failed GPU growth/XLA contract |
| Repair trigger | A reproducible first divergence; preserve coefficients and targets and distinguish solver branching from recursive amplification |
| Explanatory only | RSS, coefficients, support overlap, runtime, graph trace count, allocator peaks; none substitutes for the consumer comparisons |
| Nonconclusions | Literal original-author solver parity, agreement with fixed0.01 reference, globally unique NNLS coefficients, Table2 replication, superiority, HMC/default readiness |
| Result | Versioned attempt57/58 directories and controlled-smc-rbf-fitting-execution-result-20260929.md |

If reference creation fails, do not launch the GPU consumer. If the paired full
consumer fails, reject the factor0.1 hypothesis for this scope and preserve it as
fresh validation evidence. Do not fit a replacement factor to these seeds.
A changed regularizer, objective or pruning rule would require a separately
classified method experiment; it cannot silently satisfy this validation.

## Assumptions and skeptical review

Factor0.1 is a calibration hypothesis from attempt56, not a source default.
It may fail on a different near-degenerate active set; the independent full202
fits are the first check. Fresh-SVD and fresh-QR retain different numerical
factorizations, which is intentional independent validation, not proof of exact
arithmetic equivalence. Stable source ordering and top16 selection remain fixed.
The author proposal correction remains explicit and its prior18-check validation
is retained. Full NNLS columns are never reduced to an arbitrary prefix.

The main pre-mortem is that excellent absolute predictions could again hide
large low-density log errors or divergent top16 policies. Both policy and
integral comparisons remain primary. A full pass could still mislead if called
empirical replication: only one observed series, one noise setting and one
refinement are covered. Independent particle seeds do not support ranking.

This is a computational-equivalence test, not a method-performance comparison.
Existing analytical nonnegative controls, BPF input trajectories and corrected
fixed-policy replay are harness controls. For a later empirical RBF comparison,
construct BPF512 (naive), the source cost comparator BPF6553, and a converged
one-dimensional quadrature filter (oracle); compare conditionally in each
observation-noise regime. Those unrun comparisons cannot support this stage.

Skeptical audit PASS: baseline and finite stopping rule now match; original
criteria remain fixed; fresh particle seeds test the nominated hypothesis;
candidate rejection is separate from direction rejection; there is no claim of
method-quality improvement or permission to bypass the old failed full phase.
No ridge, damping, clipping or coefficient-pruning change is introduced.

## Commands and bounded execution

The current allocation remains CPU300/GPU18000 seconds. At attempt56 it has
CPU39.500691/GPU6682.495116 seconds left. This stage uses at most39 additional
CPU seconds and900 GPU seconds, one attempt per phase. Raise the attempt ceiling
from56 to58 solely for these two specified phases; the original campaign balance
and resource allocation do not increase. Reference50 cost34.874853 CPU seconds;
full GPU51 cost398.489946 seconds. New particle draws may be slower, so the caps
are real stop conditions rather than promises of completion.

Run with the existing approved dispatcher prefix, from the repository root:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_rbf_paired_reference
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_paired_validation
```

The CPU reference explicitly hides GPU devices and uses NumPy/SciPy only as
an independent reference. The candidate runs escalated, float64, TF32 off,
GPU memory growth verified, XLA on and stable signatures; no NumPy runtime,
pfor or Python loop over scientific time. Both manifests preserve Git/source
hashes, input hashes, exact commands, environment, seeds, wall time and results.

## Validation outcome and bounded localization

Attempt57 passed all202 SVD fits. Attempt58 completed without an exception,
with one trace each for fit, forward and NNLS, but failed10/30 comparisons.
Maximum retained-log-policy difference83.1302, Gaussian-log-integral
difference0.106316 and forward-logZ difference0.0104804 reject the paired0.1
hypothesis for these fresh particle seeds. This is candidate rejection, not
evidence that the paper's algorithm cannot work.

The next action is saved-data localization only: c1_rbf_paired_diagnosis,
one CPU-only attempt capped at10 seconds. It compares57's fixed fixture with58's
saved tensors at every reverse-time step and saves the first divergent inputs.
Criterion for a usable diagnosis is exact reconstruction (error<=1e-10) of the
reference top16 policy from its exported coefficients. Broken reconstruction,
nonfinite input, missing tensors or budget exhaustion veto interpretation.
Prediction, coefficient, policy and parent-integral errors are explanatory;
this diagnostic cannot promote the filter or nominate a replacement factor.
No optimization or new random data is used. The reference remains factor0.1.

Skeptical audit PASS: the failed fresh validation is preserved; tracing is
not mistaken for mathematical correctness; the diagnostic evaluates the actual
consumer and correct previous-parent integral; it neither tunes on validation
seeds nor changes objective, truncation, reference or tolerances. Prior identical
diagnostic cost3.52 CPU seconds. CPU allocation has19.293691 seconds left.
Raise the attempt ceiling58 to59 only, leaving CPU300/GPU18000 and the inherited
campaign totals unchanged.

Exact command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_rbf_paired_diagnosis

Artifacts: attempt59_c1_rbf_paired_diagnosis under the existing versioned root.
Read its first_divergence rows before designing any subsequent repair.
