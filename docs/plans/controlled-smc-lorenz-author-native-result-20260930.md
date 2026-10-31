# Direct author-code Lorenz result — 30 September 2026

All six direct author-code cases passed. We can execute the original adaptive
Lorenz implementation. The observed TensorFlow failure does not establish that
the author's source cannot be replicated, and the next step must compare the
two implementations before introducing another safeguard.

## Checked execution

Attempt86 runs the unchanged cSMC_Resample.m, RungeKutta4.m and
Systematic_Resampling.m with the setup read directly from cSMC_VaryF.m,
on bundled SimulatedData.mat. The author checkout has no tracked modifications;
all five executed source/data hashes agree with the preserved snapshot.
The reference uses Octave6.4.0 and the existing pinned statistics package,
CPU only, with GPU deliberately hidden. MATLAB itself was not tested.

| Forcing | Seeds | Completed refinements | Valid finite outputs and positive precisions |
|---|---|---|---|
| 2.5 | 20260929,20260930 | 4,4 | Both pass |
| 4.8801 | 20260929,20260930 | 1,1, from the original ESS stopping rule | Both pass |
| 8.5 | 20260929,20260930 | 3,3, from the original ESS stopping rule | Both pass |

The six cases completed in13.042128388 CPU wall seconds. This is a small source
execution check, not the4000-run Figure6 matrix or its matched-cost APF comparison.
Its timing is not directly comparable with the earlier GPU pilot, which includes
different computation and compilation. Every source return, including policies,
clouds and ancestry, is preserved for the next matched-input comparison.
At F2.5 the minimum precision eigenvalues over every retained stage/time are
89.7692 and90.4015. The original normal-equation solves nevertheless emitted
660 nearly-singular-matrix warnings across the six cases; warnings explain
numerical sensitivity and do not by themselves invalidate the finite results.

Attempt85 completed its first source run but failed when our reporting wrapper
called jsonencode, unavailable in this Octave version. That was our harness
failure. Its3.473717547 CPU seconds and original source result remain preserved.
The localized MAT/SciPy serialization repair left every author algorithm
unchanged; the total90-second diagnostic cap was unchanged.

## Why our work diverged

The earlier unchanged-source reference
(docs/benchmarks/reference_lorenz_consumer.m:25–28) imposed one refinement at
F4.8801. Later four-refinement validation used a new independent SVD
implementation and the centered-QR TensorFlow implementation. It did not
complete the full adaptive native-source comparison. We allowed porting and
numerical stabilization to advance before closing that missing baseline.

The source fits the raw quadratic basis using normal equations:
.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz/cSMC_Resample.m:191–193.
The local eligible fitting route standardizes the basis and uses QR, then
evaluates the recursion in centered coordinates. These target the same
unconstrained least-squares problem in exact arithmetic, but their numerical
behavior on a nearly singular raw design is not identical. The original
GPU raw-normal-equation attempt also failed coefficient parity. Stabilizing
that solve was motivated by evidence, but agreement with an SVD reference
establishes an intended-objective calculation, not exact author numerics.

The checked local failure77 is real: its third-refinement precision has an
eigenvalue approximately−1730, reproduced by independent SVD on identical
inputs. A quadratic least-squares minimum need not yield an integrable Gaussian
twist: the required matrix is Q^(-1)+2(A_old+Delta_A), which must be positive
definite. This requirement is explicit in published §6.2 p2924 and footnote4.
The source's multivariate quadratic branch (213–215) has no active positivity
constraint; its covariance-projection calls (56,98) are commented out.
This explains why invalid fits are possible; it does not establish that the
source follows the same failing trajectory.

We also use different random generators. Equal integer seeds do not give
identical Octave and TensorFlow draws. Thus86 cannot decide whether the
difference is primarily random-cloud sensitivity, solver arithmetic, or another
port discrepancy. The projections/backtracking introduced in78/80 are our
explicit repair candidates, not an executed full-quadratic author safeguard.
Their failures must not be described as failures of the original source.

## Decision and next work

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Direct native source works on these cases | Six of six cases valid | No nonfinite/complex/precision veto fired | Other seeds; MATLAB/Octave and RNG differences | Preserve this baseline and extend source replication under a bounded specification | Full Figure6 replication or general numerical robustness |
| Local GPU candidate remains rejected |77 invalid precision;82 failed repaired-policy correctness | Existing vetoes remain | First causal difference from original execution not yet matched | Replay the preserved86 source clouds/draws in the source-numerics and centered-QR paths; locate the first difference | That all divergence is caused by XLA or that the paper is wrong |
| Stop premature safeguard design | Original adaptive baseline now available | Source comparison still incomplete | Whether any change to the original method is needed for the replication target | Compare on identical inputs before choosing a numerical repair | A new default or abandonment of controlled SMC |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Six native cases pass; local candidate failures remain real |
| Statistically supported ranking | None |
| Descriptive-only differences | Counts, logZ, conditioning warnings and timing |
| Default readiness | Not assessed; this is an independent CPU source reference |
| Next evidence needed | Matched-input source/port comparison, then the full source experiment with its APF comparator and uncertainty |

The immediate sequence is: original-source execution as baseline; matched-input
port comparison; separately labeled numerical repairs if a verified discrepancy
requires them. Both whole replications remain incomplete.

Post-run skeptical review: the strongest alternative explanation is that86
uses different random clouds from failed76. This prevents attributing that
failure entirely to the QR port or entirely to the author method. A literal
source run on the failed inputs would discriminate. Source paths, hashes, setup,
CPU status and all six outputs were checked; no independent reviewer was launched.
No network metadata lookup was needed: the technical paper and pinned source
are local. No literature-wide or citation-prestige inference is made.

Evidence: artifacts/controlled-smc-kdm-replication-20260928/
attempt86_c1_lorenz_author_native/{manifest.json,summary.json,command.log,
author-F*-seed*.mat}. The governing plan is
controlled-smc-lorenz-author-native-audit-20260930.md.
