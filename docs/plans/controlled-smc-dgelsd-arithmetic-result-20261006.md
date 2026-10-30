# DGELSD source arithmetic: completed bounded diagnostic

6 October 2026. The decisive difference is localized to the divide-and-conquer
bidiagonal least-squares solver used by the Octave reference. It is not explained
by outer range scaling, tiny-diagonal flooring, splitting, or the workspace
query on the three inspected cases. Both replications remain incomplete.

## What was compared

The actual baseline is the author's unchanged MATLAB-language source executed
in Octave6.4.0 with system LAPACK3.10.0. It is not a captured run of the author's
original MATLAB installation. The author polynomial normal equations are at
`StateSpaceModels/Lorenz/cSMC_Resample.m:193`; the paper's least-squares target
is Section4.2 equations20–21. The installed library checksum agrees with the
preceding229-system reference and ILAVER reports3.10.0. No numerical backend
or author setting was changed.

[Plan](controlled-smc-dgelsd-arithmetic-20261006.md).
Successful artifact:
`artifacts/controlled-smc-kdm-replication-20260928/dgelsd-arithmetic-20261006-03/`.
The manifest, rows.jsonl, arrays.npz and downloaded versioned LAPACK source
preserve the executable comparison. NumPy and ctypes occur only in the explicit
independent CPU reference; no TensorFlow runtime imports this diagnostic.

DGELSD reproduces all three saved coefficient vectors exactly. The decomposed
DGEBRD → DORMBR(Q) → DLALSD → DORMBR(P) chain reproduces all three singular
spectra bitwise and predictions to at most1.887e-10. Its coefficients are not
bitwise identical: maximum deviations are1.091e-11,1.819e-12 and2.183e-11.
The source of this small decomposition-rounding difference remains unresolved.

Two earlier executions failed the initially required exact coefficient check.
The known-solution control difference was4.44e-16; the first saved case differed
by1.09e-11. Those failures and the pre-third-run diagnostic amendment are
preserved. The final diagnostic supports an approximate mechanism explanation,
not a claim that the full source call chain has been recreated bitwise.
All ten solver arms passed the known-solution control at the planned1e-10 bound.

## Discriminating result

The installed defaults are SMLSIZ=25, MNTHR=72 and DGELSD cutoff2^-53.
For a45-by45 matrix, no preliminary QR runs. None of the three cases triggers
outer range scaling, a tiny diagonal floor, or a bidiagonal split. The floor
and split intervention arms therefore produce exactly the unmodified QR-arm
behavior. These source protections exist but are not the explanation here.

Keeping the same Householder bidiagonal reduction and right-hand-side
projection, replacing only DLALSD's divide-and-conquer route with its own
small-problem QR route gives:

| Zero-based time | Source rank / smallest singular ratio | QR rank / ratio | Maximum polynomial difference from source |
|---|---|---|---|
|20|45 /3.94588626e-16|44 /7.79606406e-17|0.203840924|
|25|45 /2.32921632e-16|44 /8.01716353e-17|0.579450272|
|28|45 /1.14074077e-15|45 /1.90187849e-16|0.450073321|

These prediction separations exceed the decomposed-chain uncertainty by over
a billionfold. The same-library DGESVD/DGESDD explicit solves have analogous
ranks and errors. Earlier TensorFlow errors were0.5794 maximum and0.4508 at
time28. This supports a solver-algorithm explanation; it does not prove that
each TensorFlow error has the same instruction-level cause.

There is also a default mismatch that a careless comparison would miss:
DGELSS with rcond=-1 uses DLAMCH('P')=2^-52, while DGELSD/DLALSD uses
DLAMCH('E')=2^-53. DGELSS consequently reports rank44 even at time28.
The explicit SVD and reconstructed-QR arms use the DGELSD cutoff, so the main
intervention is not confounded by that default difference.

Versioned source anchors:
[LAPACK DGELSD](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/SRC/dgelsd.f)
lines378–498: range scaling and decomposition;
[DLALSD](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/SRC/dlalsd.f)
lines237–244,299–307,367–442: cutoff, bidiagonal scaling, flooring and solver;
[DGELSS](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/SRC/dgelss.f)
lines384,504–506: its different default cutoff. Tractable copies and hashes
are in the successful artifact.

## Verdict and remaining work

The claimed mathematical target is the author's quadratic least-squares fit.
The computed source quantity is a particular floating-point solution of rounded
normal equations using Octave's conditional solver. Different stable solvers
can compute different policies on these nearly singular matrices. The observed
difference is not evidence that the paper's least-squares derivation is wrong.
No filter-quality ranking follows from these fixed-matrix residuals.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Source wrapper and inputs verified|Direct DGELSD exactly reproduces three saved answers|No input, library, query or finite-control failure|Original MATLAB runtime not reproduced|Retain this explicit Octave baseline|Original-author hardware/runtime equality|
|D&C mechanism localized|Same reduction plus QR changes ranks/predictions at the observed scale|Exact decomposed coefficient criterion failed; amended diagnostic screen passes|Few-ULP call-chain difference; inner D&C instruction not localized|Use this cause when assessing cost of a faithful TF operation port|A TF repair exists or has passed|
|Full replication remains incomplete|No autonomous/P1 acceptance evidence added|Current forced-branch TF candidate remains incompatible|Downstream non-inferiority and failures|Define conditioning-aware P0 invariants and independently calibrated P1; continue source/training specifications|No-worse, default, robustness or HMC status|

| Inference status | Evidence |
|---|---|
|Hard veto screen|All direct source and finite controls pass; exact decomposed-chain reproduction fails and is reported|
|Statistically supported ranking|None; these are three chosen deterministic systems|
|Descriptive differences|All residuals, ranks, singular ratios and prediction errors above|
|Default readiness|None; diagnostic code only|
|Next evidence needed|Autonomous matched fitting/failure accounting, fresh source calibration and untouched downstream comparisons|

The next repair choice must distinguish reproducing the source operation from
qualifying an algebraically equivalent implementation. A full TensorFlow
DGELSD port is nontrivial; merely selecting Cholesky/SVD, changing a rank cutoff,
or increasing XLA accuracy has already failed to reproduce source behavior.
Centered QR remains an unqualified equivalent-algebra candidate. Neither
coefficient matching nor a lower training residual alone establishes no-worse
filter behavior. Do not relax a tolerance to accommodate this candidate.

## Execution accounting and review

The three CPU executions used0.515962503,0.265127803 and0.265057773 seconds
including process startup, total1.046148079 seconds, charged once to the active
CPU allocation. GPU use was zero. Remaining:151422.744040 CPU seconds and
702.756496 GPU seconds. Numbered dispatcher attempts remain160; these three
bounded reference executions have separate receipts. No process remains running.

Post-run red-team: the strongest alternative is an unmeasured tiny discrepancy
in decomposed source arithmetic. Its prediction effect is over a billion times
smaller than the QR intervention, and spectra match exactly; this weakens that
alternative for the large gap but leaves bitwise reconstruction unresolved.
The weakest generalization is extrapolation from three selected systems to
the full recursive filter. An autonomous matched comparison can overturn that
generalization. Review here was local self-review, not an independent reviewer.
