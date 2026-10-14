# Source DGELSD arithmetic audit

Authorized replication-only continuation, 6 October 2026. The next question is
why the author's full-rank DGELSD results disagree with TensorFlow SVD on the
identical saved normal equations. This audit does not attempt to improve the
author's method. It consumes at most 120 CPU seconds per execution, three
executions including repairs (360 CPU seconds total), no GPU time, from the
existing CPU allocation. It is a bounded reference diagnostic outside the
160-attempt dispatcher, with its own manifest and CPU accounting.

## Research intent and evidence contract

The baseline is the saved Octave coefficient vector and the already verified
linked system LAPACK library, not SciPy's separately linked LAPACK. Inspect
the matched LAPACK 3.10.0 source and measure the installed library version.
Use saved offset times20,25,28, chosen in the preceding result for the rank
and prediction discrepancies; they are diagnostic cases, not independent
replications. Add a well-conditioned, known-solution 45-dimensional control.

Test three explanations: an erroneous wrapper/workspace query; differences
in bidiagonal reduction or singular-vector application; and DGELSD's internal
scaling, tiny-diagonal flooring and divide-and-conquer solve. Reconstruct the
DGELSD call chain with the same library (DGEBRD, DORMBR, DLALSD, DORMBR), then
change only the bidiagonal solver in labeled diagnostic arms. Compare DGELSS
and explicit DGESVD/DGESDD on the same Gram matrix to separate generic SVD
semantics from the particular source routine.

The primary audit criterion is exact reproduction of saved source results by
DGELSD and by the reconstructed source call chain. Known-solution controls
must pass at 1e-10 absolute coefficient error. Query mutation, library/input
hash changes, failed LAPACK calls, or non-finite controls invalidate the
audit and trigger a localized repair. Finite differences in other solver arms
are explanatory evidence, not a continuation veto or a replication verdict.
Record ranks, full spectra, bidiagonal arrays, flooring/split locations,
coefficient and polynomial-prediction differences, residuals, and backward
errors. A mechanistic explanation requires an executable intervention, not
source inspection alone. If the reconstructed chain is not exact, stop its
causal interpretation and preserve that failure before any further port work.

This is not a no-worse test, a calibration of final tolerances, a reason to
alter the author comparator, or a claim that lower residuals improve the
filter. P0 conditional arithmetic checks remain separate from P1 no-worse
evidence. Numerical failure intrinsic to the author is an accepted baseline
outcome. Original iAPF and enhancements remain deferred. The known-solution
control is the cheap falsification comparator for this arithmetic question;
filter heuristic rankings are not applicable to this diagnostic.

## Sources and assumption audit

The paper's backward least-squares fit is Section4.2 equations20–21; the author
code `StateSpaceModels/Lorenz/cSMC_Resample.m:193` forms normal equations.
Octave6.4.0 `dMatrix.cc:2144–2219` calls DGELSD with rcond=-1. LAPACK3.10.0
DGELSD lines378–498 implement outer range scaling, bidiagonalization and the
DLALSD solve. DLALSD lines237–244 select machine epsilon; lines299–307 scale
the bidiagonal entries; lines367–371 floor tiny diagonals; lines374–442 split
subproblems and select QR or divide-and-conquer solves.

| Choice | Provenance/status | Risk and smallest check |
|---|---|---|
| Exact saved G,h and beta | Prior verified author fixture, baseline | Hash and exact DGELSD check before interventions |
| System LP64 LAPACK | Octave-linked baseline | Library checksum, ILAVER, ABI and known-solution controls |
| Float64; single BLAS thread | Author reference precision; fixed diagnostic environment | Query and direct-source parity checks |
| rcond=-1, installed SMLSIZ and MNTHR | Runtime source defaults, baseline | Record ILAENV and DLAMCH; do not tune thresholds |
| Large SMLSIZ arm | Explicit hypothesis forcing the routine's small-problem QR path | Diagnostic only; never silently replace the source default |
| Flooring/splitting interventions | Exact source formulas, mechanism hypotheses | Save changed indices; compare both before and after |
| NumPy/CPU | Explicit independent-reference exception | No import by TensorFlow runtime; CUDA hidden before imports |

Pre-mortem: several backward-stable algorithms can disagree on nearly singular
normal equations. A smaller residual or matching rank cannot establish the
source policy or scientific superiority. Even a perfect explanation here does
not complete autonomous branch selection, full fitting, failure matching or
P1 qualification. Avoid mistaking endless reproduction of LAPACK rounding for
the user's final no-worse target.

Skeptical audit: passed before execution. The baseline and selected cases are
preserved; solver arms isolate operations on identical inputs, include a
known-solution control, and cannot promote a runtime default. Version and
query checks address environment/wrapper errors first. Explicit bounds and
invalidity checks prevent an open-ended numerical campaign.

## Command and artifacts

`/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_lorenz_dgelsd_reference.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/dgelsd-arithmetic-20261006-01`

The script sets CPU-only and thread settings before imports. Preserve a full
log, source copies and checksums, per-arm arrays, summary and manifest in a
fresh output directory. Record wall time and subtract it from the active CPU
balance. Result note: `controlled-smc-dgelsd-arithmetic-result-20261006.md`.
Next action is determined by measured cause, not by which arm has the lowest
residual. Any TensorFlow candidate needs its own control and source-compatibility
check before full fitting or remaining GPU time is committed.

### Execution amendment after the first two diagnostic checks

The exact-coefficient check failed: the control differed by 4.44e-16 and
time20 by 1.09e-11, with singular values identical bitwise in both. These are
preserved failed checks, not a completed causal identification. The third
execution records all arms if the source coefficient vector still matches the
saved author exactly, the reconstructed source spectrum is bitwise identical,
and the reconstructed polynomial meets the previously frozen localization
screen `1e-7 + 1e-9*abs(reference)`. Exact reconstruction is reported separately.
This permits a bounded approximate mechanism diagnostic; it does not claim
bitwise call-chain reproduction, change P0/P1 acceptance, or license a runtime
candidate. A gap comparable with this reconstruction error remains unresolved.
The reason for the residual call-chain rounding difference is not established.
Skeptical re-audit: direct-source and spectrum anchors remain strict; record the
reconstruction error beside every intervention. Only separations demonstrably
larger than that error can support a mechanistic inference.
