# q20 remaining training error: source and mathematical audit

Owner question: explain why the repaired maps still have large 1,000-point
Gaussian score residuals by tracing the executed code and mathematics.

Question: which remaining mechanisms are established by the actual objective,
architecture, optimizer state, stopping logic and saved observations? Compare
the exact frozen training source and parent checkpoint with its repair arms,
using their preserved histories and full post-training reports. Current
workspace code is a separate implementation version, not execution evidence.

The pass criterion is an anchored derivation and trace linking each explanation
to source or saved measurements. Diagnostics on saved data are explanatory,
not candidate ranking or HMC admission criteria. Mismatched identities, missing
data and incompatible definitions invalidate the affected inference. A map's
poor fit is a repair trigger, not rejection of NeuTra. No causal attribution to
clipping, capacity or loss is established by descriptive differences alone.

Inspect the actual reverse-KL scalar and its parameter derivative, Gaussian
score-residual identity, autoregressive flow/permutation/caps, affine scaling,
gradient clipping and Adam migration, RNG streams and actual stopping rules.
Use standard-library saved-data summaries where sufficient. Mathematical
claims require local derivations; inspect original technical sources if an
external result is needed. Explicitly distinguish objective correctness,
optimizer convergence, transform capacity and downstream usefulness.

No new training, target evaluation or GPU computation is in this audit. The
remaining diagnostic allocation is 7.711412891243526 seconds and is not expanded
by this request. Bound saved-data analysis to the five retained checkpoints,
their history/results and three completed 1,000-point banks; no parameter or
seed search. Summarize full recorded training intervals rather than choosing a
window based on its outcome. Preserve exact paths/hashes and scripts in
`docs/plans/artifacts/q20-training-math-audit-2026-09-23/`. The final report will
include uncertainty, repair implications and the smallest discriminating next
checks, without claiming those unrun checks passed.

Skeptical audit: reverse KL and Gaussian-score residual are different quantities;
small loss changes cannot establish stationarity or whitened geometry. Every
parameter-gradient and parameter-cap inference must account for Adam scaling,
composition and off-diagonal Jacobian terms. Repaired arms share inherited
states, so they are not independent replications. Finite differences previously
checked selected points only. Partial batch-128 evidence is not a full bank.
This read-only audit can answer mechanism questions within those limits; it
cannot identify a unique causal remedy without another controlled run. Audit
passes with these boundaries.

Saved-bank localization: partition squared residual energy by its four latent
coordinates. Calculate first derivatives for an infinitesimal affine
precomposition `T(exp(a)*z+b)` directly from saved scores: at a=b=0,
`dL/da_j=mean(-(g_j-z_j)*z_j-1)` and
`dL/db_j=mean(-(g_j-z_j))`. These test simple missing affine corrections, not
stationarity of the original network. Report descriptive standard errors, no
formal ranking or convergence test on the previously inspected bank. Project
each residual onto the five-dimensional span of 1 and the four z coordinates
by ordinary least squares; its in-sample energy fraction separates constant/
linear residual structure from remaining variation without fitting a transport
or evaluating a new target. Basis dimension is derived from d=4, not tuned.
Use the existing 128-update checkpoint cadence for all history blocks, retaining
all blocks. Do not infer late convergence from a whole-tranche before/after
improvement label.

If needed to distinguish unused nonlinear capacity from a simple affine error,
reconstruct only the saved transform forward pass in a diagnostic standard-
library script. Apply the exact degree masks, tanh scale formula, reversal and
outer affine from the frozen source. Check its layer-scale means against the
saved TensorFlow reports before interpreting it. Measure empirical affine-fit
fractions of the physical proposal coordinates and hidden activation ranges;
these are proposal/parameterization diagnostics, never posterior evidence.
No target evaluations or parameter changes are introduced.

Completed: the [result](bayesfilter-q20-training-math-audit-results-2026-09-23.md)
records the executed-source trace, RKL derivation, coordinate localization,
independent saved-transform reconstruction, full history blocks, optimizer
migration arithmetic, and terminal skeptical review. Both scripts stayed
within this saved-data scope and made no target calls. The principal new
finding is an almost-affine learned map alongside predominantly nonlinear
residual error. Initialization and optimizer explanations remain hypotheses;
neither more training nor added depth has been established as a sufficient
remedy. No training parameters, scientific defaults, or compute allocations
were changed by this audit.
