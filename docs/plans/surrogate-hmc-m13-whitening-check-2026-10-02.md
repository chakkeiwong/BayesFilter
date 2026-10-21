# M13 frozen-whitening diagnostic, 2026-10-02

Question: does the saved affine transform adequately remove linear scale and
correlation differences, and does the later process-scale excursion indicate
problems a single fixed covariance cannot describe? The user asks how to check
whitening. This is a bounded, CPU-only analysis of existing evidence, not new
HMC, a validation phase, retuning, or a change in sampling rules.

Target: the frozen M13/data1704/T20 importance-likelihood posterior and its
actual endpoint-corrected deterministic proposal field. Load the checksum-bound
public tuning artifact used by the final run, its nomination center/factor,
and all 12,750 saved post-warmup draws per chain. Keep the 9,000-draw stopping
point as a separately labeled historical summary. Exclude warmup from posterior
covariances; inspect the covariance-adaptation schedule separately. Verify the call chain from the
public frozen chunk runner through the affine adapter and proposal-field
factory; execute inverse/forward and factor/covariance parity checks.

Evidence contract: for theta = c + L z, empirical covariance in z is
L^{-1} Cov(theta) L^{-T}; if L L^T matched posterior covariance exactly, this
would equal identity. Compare marginal SD, correlation, covariance eigenvalues,
and condition numbers before and after the actual frozen transform. Identity
is the reference geometry, not a retrospective pass/fail threshold. Summarize
both histories and each chain; disagreement limits confidence in a pooled
estimate. Record mass-adaptation budget, shrinkage, chosen step size and
leapfrog count. Compare acceptance and squared movement
by starting-state process-scale region. Use pooled 5th and 25th percentiles
only as descriptive boundaries. Conditioning on proposed or accepted endpoint
states would bias the acceptance comparison and is forbidden.

All diagnostics are explanatory or possible repair triggers, never additional
promotion/continuation vetoes. Bad checksums, affine mismatch, nonfinite draws,
or inconsistent sample counts invalidate this diagnostic. No new whitening is
selected or installed from retained draws. No speed or method ranking, exact
likelihood equivalence, convergence proof, or all-model conclusion follows.
There is no new method comparison; heuristic-adversary ranking is inapplicable.

Default/assumption audit: identity covariance is the exact algebraic reference
for affine whitening. Prefix/full covariances and region boundaries are noisy
sample estimates, not known posterior geometry; chains and epochs expose this
risk. Conditional covariance alone is affected by truncation and is not a local
Hessian. The saved 128/256 mass windows, 10% prior shrinkage, and short trajectory
ladder are inherited hypotheses being inspected, not defaults being promoted.
Within-chain movement and acceptance cannot distinguish target curvature from
proposal-force mismatch. No Hessian or force-fidelity claim is supported here.

Skeptical pre-run audit: source/tuning artifacts identify the frozen transform;
9,000 later warmup steps must not be mistaken for continued covariance
adaptation. The diagnostic must use the saved factor, not refit and then claim
identity covariance. The observed posterior may be non-Gaussian and incompletely
explored, so affine covariance quality cannot explain every mixing failure.
The above controls address wrong baselines, hidden adaptation assumptions, proxy
promotion, and sample-selection bias. Proceed with these limitations.

Budget: one saved-data CPU diagnostic, maximum five minutes, one localized
infrastructure repair if needed. GPUs intentionally hidden before TF import;
TensorFlow/TFP compute statistics. Non-XLA graphs are an explicit offline
diagnostic exception. Matplotlib creates raw/whitened scatter plots. Preserve
prior evidence. Output: docs/plans/artifacts/surrogate-hmc-m13-whitening-check-20261002-01/.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python followed by that
directory's analyze.py. Save command log, report, plot, run manifest and result
note. Remaining sampling allocation is unchanged and expired.

Execution note: attempt 01 stopped before producing results because the reporting
script supplied the wrong array layout to the inverse triangular solve. The
localized repair flattens chain/draw axes and checks against the actual runtime
adapter. It also keeps time lags within each chain and conditions acceptance
on transition starting states. No sampler implementation changed. Attempt 01
and its log remain preserved; the repaired output directory ends in -02.

Completed: attempt 02 exit 0, 1.981 seconds after imports.
See artifacts/surrogate-hmc-m13-whitening-check-20261002-02/result.md.
Affine parity passes; incomplete global whitening and severe regional proposal
inefficiency are descriptive findings. No causal attribution or policy change.
