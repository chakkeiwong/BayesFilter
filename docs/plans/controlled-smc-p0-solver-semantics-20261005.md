# Conditional source-solver semantics check

Continuation of attempt149, within the existing replication allocation.
Question: does an explicit implementation of the Octave branch sequence,
using independent CPU LAPACK routines, reproduce its saved solutions on
identical matrices? This is a prerequisite check for translating that sequence
to TensorFlow, not a new runtime backend or a numerical protection.

Source anchors: author Lorenz/cSMC_Resample.m:193; local
.localresources/code/octave-6.4.0-dMatrix.cc:1405–1567,1614–1628,2144–2219.
The source tries Cholesky/DPOCON, then LU/DGECON if required, and then DGELSD
when the condition estimate rounds 1+rcond to1. DGELSD receives rcond=-1.
The [LAPACK DGELSD source](https://www.netlib.org/lapack/double/dgelsd.f)
documents the resulting machine-precision rank convention. No hand-selected
cutoff, ridge, truncation threshold, or forced branch may replace these rules.

Comparator: attempt94 ordinary Octave solutions and cached branch types,
using all128 matrix/RHS pairs from attempt92 (32 saved regressions, four
crossovers). Main localization screen: branch agreement and coefficient
agreement within the unchanged 1e-7+1e-9 absolute-reference screen on every
pair. This is not a statistical no-worse criterion. Report residuals, effective
ranks, condition estimates and exact-inverse condition estimates as explanations.
The latter is a hypothesis for a future implementation, not the source rule.

Assumptions: SciPy's LAPACK may use different libraries or rounding from
Octave. Record versions and configuration; backend disagreement is a possible
negative result, not evidence against the author algorithm. NumPy/SciPy are
explicit independent-reference tools here, never imported by a TF consumer.
No heuristic-superiority or downstream/default claim is tested. Source code
and prior matched forward controls define this replication comparator.

Skeptical audit: identical saved matrices separate arithmetic construction
from solve behavior. All128 pairs are retained, with no selection on errors.
Missing/corrupt inputs, nonzero unexpected LAPACK errors or missing branch
diagnostics invalidate the check. Failed coefficient agreement continues the
investigation; it does not authorize changing the author method or relaxing
the comparison. The strongest alternative explanation is differing BLAS/LAPACK
arithmetic on ill-conditioned matrices.

Phase p0_solver_semantics: one attempt, at most90 CPU seconds from the current
151728.165652-second CPU balance, zero GPU seconds. Reallocate the unused
attempt150 bookkeeping slot; an additional infrastructure retry would require
an explicit bounded plan update, not a new compute grant.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase p0_solver_semantics

Output: fresh attempt150_p0_solver_semantics with frozen sources, input hashes,
manifest, complete log, per-matrix diagnostics, library configuration and
summary. CUDA is hidden before imports; single CPU BLAS thread.
If all source branches and coefficients agree, implement and check the same
branch sequence in TF next. Otherwise localize branch versus backend arithmetic
before selecting an implementation. Full cSMC/KDM replication remains open.


## Prespecified follow-up after attempt150

Attempt150 matched every branch but only64/128 coefficient vectors. All60
DGELSD fallback solves retained rank45. The source baseline uses system
OpenBLAS0.3.20/LAPACK3.10; SciPy uses OpenBLAS0.3.30. Therefore a source-branch
repair alone is not yet a demonstrated numerical repair.

Next discriminating check: rerun the explicit same branch sequence using
the exact shared LAPACK library linked by /usr/bin/octave-cli, through an
independent diagnostic ctypes adapter. Verify linkage, hash the library, test
a small known SPD system before touching saved matrices, and preserve all
per-matrix branches/ranks/coefficient differences. The LP64 Linux Fortran ABI
is an explicit platform assumption; a call error, wrong linkage or invalid
SPD smoke invalidates this diagnostic. No runtime TF dependency is added.

Primary criterion remains all128 branches and coefficient screens; also report
bitwise equality. If this passes, backend arithmetic is isolated on these
saved inputs. It does not demonstrate a GPU translation, a need to copy the
CPU library into runtime, or statistical inferiority of a GPU solver. If it
fails, inspect exact source-versus-adapter operations before any port change.

Phase p0_solver_system_reference, one attempt151, at most90 CPU seconds,
GPU0. This is a new discriminating reference arm, not an unchanged retry.
Extend bookkeeping ceiling150→151 within the unchanged CPU allocation.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase p0_solver_system_reference

Review: changes the numerical-library comparator only, preserves matrices and
criteria, checks the actual loaded library. It answers the remaining backend
explanation without attempting an author algorithm repair. Detailed logs,
library configuration, original fixtures and new summary are retained.

Terminal state, 5 October 2026: this bounded diagnostic plan has finished; see [controlled-smc-p0-solver-semantics-result-20261005.md](controlled-smc-p0-solver-semantics-result-20261005.md). Full replication remains incomplete. New launches require the next bounded implementation/qualification contract within the remaining allocation.
