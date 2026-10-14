# Conditional solver semantics: terminal result, 5 October 2026

The declared Octave baseline is now reproduced exactly on all 128 saved
regression systems. An explicit implementation of its conditional solve,
linked to the same LAPACK library as Octave, reproduces every coefficient
bit for bit. The same branches using SciPy's bundled library pass only 64
of the 128 coefficient screens. This localizes a numerical-library
dependence on these ill-conditioned inputs. It does not complete the
TensorFlow fitting translation.

[Plan](controlled-smc-p0-solver-semantics-20261005.md);
[preceding fitting result](controlled-smc-p0-fitting-forward-result-20261005.md).
All original attempts and failed comparisons remain preserved.

## Question, baseline and source meaning

The source Lorenz fit forms a Gram matrix and right-hand side and applies
matrix left division: .localresources/code/heng-controlled-smc/Lorenz/cSMC_Resample.m:193.
The locked Octave runtime selects Cholesky/DPOCON, then LU/DGECON when
required, then DGELSD when its reciprocal-condition estimate satisfies
1 + rcond == 1. DGELSD receives rcond = -1. The inspected runtime anchors
are .localresources/code/octave-6.4.0-dMatrix.cc:1405–1567,1614–1628,2144–2219;
the rank convention is documented in the
[LAPACK routine](https://www.netlib.org/lapack/double/dgelsd.f).

The comparator is attempt94's ordinary Octave solutions and cached branch
types, applied to all four matrix/right-hand-side crossovers from attempt92:
32 regressions per arm, 128 systems in total. The predeclared localization
screen is componentwise absolute error <= 1e-7 + 1e-9 * abs(reference),
together with branch agreement. It is an explanatory deterministic screen,
not a downstream non-inferiority margin.

## Executed comparisons

| Attempt | Numerical library | Branch agreement | Coefficient screen | SVD fallback rank |
|---|---|---:|---:|---|
| 150 | SciPy bundled OpenBLAS 0.3.30 | 128/128 | 64/128 | 45 for all 60 fallback systems |
| 151 | Octave-linked system LAPACK 3.10.0 / OpenBLAS 0.3.20 | 128/128 | 128/128, all bitwise equal | 45 for all 60 fallback systems |

There are 68 positive-definite branch solves and 60 fallback solves. The
source's cached label Singular does not mean that any of these fallback
solutions discarded a singular direction. Adding a rank cutoff or ridge
would change this baseline. The measured LAPACK epsilon is
1.1102230246251565e-16; the runtime condition and rank conventions must be
preserved explicitly.

Attempt151 verified linkage against /usr/bin/octave-cli, tested the adapter
on a known SPD system first, and saved the resolved library path and SHA-256.
The library is /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3,
SHA-256 426b4bf536065cc63ef879444dad76dbc9f85e36e5cde906269acd6f177c5151.
The adapter assumes the checked Linux x86-64 LP64 Fortran ABI. NumPy,
SciPy and ctypes are independent CPU reference tools here; the TensorFlow
runtime does not import this adapter.

The explicit-inverse condition estimate selected the same branches on
these 128 systems. That is a bounded observation, not proof that it is
DPOCON/DGECON or permission to substitute it in an admitted runtime.

## Root-cause interpretation

The earlier GPU experiment, attempt149, made 32 valid fits and 32 valid
forwards across the raw normal-equation and centered-QR implementations.
Both had 0/16 strict fitted-policy/downstream matches. Feeding the author's
policies into the identical forward endpoint passed all 16 controls. The
fit and forward kernels each traced once. Thus the checked discrepancy
starts with fitting; no XLA failure, retracing problem or additional
numerical invalidity was observed in that experiment.

Attempts150–151 separate solve-branch selection from library arithmetic:
identical saved inputs and matching branches still disagree across the two
libraries, whereas the Octave-linked library reproduces the source exactly.
The previously measured Gram conditioning near 1e15 explains why small
arithmetic changes can create large coefficient changes. This is evidence
for conditional solver semantics and floating-point sensitivity in the
declared baseline. It does not quantify each mechanism's contribution to
the complete adaptive particle trajectory.

All fallback systems are numerically full rank under the source convention.
The computed target is the same unregularized normal-equation solve, with
its runtime-dependent numerical algorithm. This check does not establish
that Octave and the authors' original MATLAB runtime choose identical
branches or coefficients.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept the independent conditional reference on these inputs | 128/128 branches and coefficients, bitwise equality | Linkage, input hashes, SPD smoke and diagnostics passed | Other matrices and MATLAB runtime remain unchecked | Use this reference to specify and check the TF branch/condition/rank behavior | Complete fitting or recursive parity |
| Keep TF replication open | Existing fits are valid but fail the strict matched-draw screens | No extra invalidity observed; no candidate admitted | Ill conditioning and resampling make coefficient/trajectory closeness a weak final criterion | Freeze conditioning-aware P0 checks and independently calibrated P1 margins before qualification | Statistical inferiority, paper failure or a need to repair the author's algorithm |

| Inference status | Finding |
|---|---|
| Hard veto screen | No corrupt input or unexpected library failure; the current TF strict comparison failed |
| Statistically supported ranking | None; these are deterministic localization checks |
| Descriptive-only differences | Policy errors and downstream logZ differences from attempt149 |
| Default readiness | Not established; neither TF solver is newly promoted |
| Next evidence needed | Actual TF fitting/forward checks against the verified reference, followed by untouched downstream no-worse qualification |

Post-run skeptical review: sharing Octave's library is intentional for
isolating runtime arithmetic, not an independent proof of least-squares
accuracy. Exact equality could not establish a GPU port or statistical
quality. A mismatch on a new source-valid regression, failure to reproduce
the conditional branch, or evidence that the claimed consumer bypasses the
checked solver would overturn a broader implementation claim. The weakest
remaining evidence is full adaptive fitting and downstream qualification.
Candidate disagreement does not reject controlled SMC as a research method.

## Execution, accounting and next stage

Exact commands were:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase p0_solver_semantics
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase p0_solver_system_reference

Each attempt has a frozen source snapshot, command, input hashes, environment,
seeds, base Git commit, full log, summary and manifest. CUDA was intentionally
hidden before imports, and CPU BLAS used one thread. Charged CPU wall time:
0.214888912 seconds for attempt150 and 0.415395864 seconds for attempt151.
Attempt149 charged 168.896124636 GPU seconds. The active numbered allocation
now retains 151727.535367 CPU seconds and 2812.436807 GPU seconds (46.87
minutes); routine CPU software checks are recorded separately.

Artifacts under artifacts/controlled-smc-kdm-replication-20260928/:

- attempt149_p0_fitting_forward/: 32 fit/forward outputs and source controls.
- attempt150_p0_solver_semantics/: all 128 diagnostics and SciPy configuration.
- attempt151_p0_solver_system_reference/: all 128 diagnostics, linkage,
  library identity and coefficients. exact-equality-check.json is a derived
  post-run check with original and candidate hashes; frozen outputs are unchanged.
- p0-terminal-review-20261005-03/: final focused software review, 53 tests and14 subtests passed; CPU3.073748 seconds. Status/progress snapshots verify no running job and the active allocation.

The dispatcher now reports the current phase instead of allowing the old
source failure to hide newer work. A completed diagnostic cannot report a
verified solver merely because its process-level status is PASS: it must
also have the scientific agreement verdict. Progress now reports the smaller active compute balance separately from inherited balances. No numerical TF core was changed.

The bounded reference plan is complete. The next implementation stage must
bind the TensorFlow condition estimator, Cholesky/LU/SVD branches, rank rule
and all 128 reference systems before fitting/forward and P1 qualification.
Do not activate E0, invent a truncation floor or convert this diagnostic
screen into the final no-worse criterion. KDM training/source-specification
gaps remain separate; no numerical worker is left running.
