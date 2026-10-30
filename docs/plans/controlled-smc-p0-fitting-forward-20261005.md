# P0 fitting followed by the actual forward consumer

Parent: controlled-smc-kdm-continuation-readiness-20261005.md. Authorized by
the owner's instruction to execute that plan. Replication only; E0 excluded.

## Question and exact comparator

Do the existing raw-normal-equation and centered-QR fitting consumers, on the
same 16 source clouds, produce domain-valid policies and reproduce the next
source forward stage when supplied the source's exact random draws?

The baseline is the pinned author Lorenz/cSMC_Resample.m:187–233 in
.localresources/code/heng-controlled-smc, commit
2696f6d611aae172d10102cb7bfbfd314bb6b88f. It forms the quadratic design,
solves (D'D) beta = D'Value, adds the quadratic coefficients to the policy,
and recurses through Gaussian integrals. The inspected paper's Section 4.2,
equations 20–21, and Section 6.2, equations 43–44, specify least-squares
quadratic policy refinement. Section 6.2 explicitly conditions tractability
on positive-definite proposal covariances. No new projection is permitted.

Attempt147 preserves six original source cases with exact unchanged outputs
and RNG: F=2.5,4.8801,8.5, seeds20260929/20260930, d8,N512,T100. Their 22
forward stages contain16 transitions between successive fitted policies.
Use every transition, not a selected favorable subset. Attempt148 establishes
fixed-policy forward agreement. This phase adds the missing fit→forward
consumer chain, without repeating single-regression/matrix crossover studies.

## Evidence contract, defaults and audit

Call the existing MultivariateGaussianControlledSMC.fit with each source's
old A,b,c, x and incremental log weights. Compare its resulting policy on both
the old source cloud and the next source cloud. For domain-valid results,
call the existing forward endpoint with the next source stage's draws.
Also run the next source policy through that same forward endpoint as the
control. Every source-control output must pass the existing fixed-policy
screen, with exact ancestry; otherwise invalidate the affected experiment.

The diagnostic screen is |error| <= 1e-7 + 1e-9*|source| for x, log incremental
weights, cumulative logZ, ESS and policy evaluation, with exact ancestry
reported separately. These inherited diagnostic thresholds can identify a
mismatch, but are not P1 non-inferiority margins. A same-draw trajectory
disagreement after resampling is not by itself a statistical quality ranking.
Coefficient identity and smaller regression residuals are explanatory only.

Primary completion criterion: all32 fits and their downstream outcomes are
accounted for, every source control passes, all required diagnostics and
provenance are saved, and every scientific endpoint traces once. A PASS
execution status cannot mean replication acceptance. Emit a separate verdict
per solver: additional invalid fit/forward outcome, downstream mismatch, or
scoped matched-input screen passed. Never admit a solver from its fit residual.
Domain failure on a source-valid case blocks that candidate in this scope;
it does not reject the research direction or stop comparisons for other cases.

Raw normal equations reproduce the literal equation but not Octave's known
conditional minimum-norm fallback. Centered QR preserves the algebraic
least-squares objective but need not reproduce Octave's finite-precision
truncated solution. Both are hypotheses, not admitted replacements. No solver
is selected from this diagnostic. Their source-policy controls expose a wrong
baseline, wrong draw order, changed endpoint or input-shape error early.

Defaults: float64 matches source double; TF32 disabled; XLA and stable input
signatures remain on; no precision projection or first-failure guard candidate
is enabled. Six cases and32 fits come from complete existing capture, not
power calculations. No new RNG or training occurs. Eager host reductions and
serialization are diagnostic reporting only. Numerical consumers stay on GPU.
The original source method is the required comparator for this replication
question; heuristic superiority is outside scope and cannot be concluded.

Skeptical audit passes for localization: target, source normalization, source
settings and real consumer chain are fixed. Its chief limitation is that a
single refit from source clouds does not test an autonomous adaptive trajectory
or statistical non-inferiority. An algebraically equivalent fit may differ
because the source solves nearly singular normal equations. Distinguish those
findings rather than relaxing thresholds after seeing results.

Stop for corrupt/source-mismatched inputs, failing source controls, missing
diagnostics, unexpected retracing, wrong device or exhausted budget. A failed
candidate triggers source-compatible solver/recursion localization using these
saved outputs, not a full author rerun or an E0 protection. Before changing a
solver, specify and check its rank/branch semantics against the existing
Octave matrix fixtures; do not tune a threshold to these downstream results.

## Execution and allocation

Phase p0_fitting_forward, at most two attempts (second only for a localized
infrastructure repair), at most600 seconds per attempt and900 GPU seconds
total from the existing2981.332932-second active balance. No new compute grant.
Extend attempt bookkeeping from148 to150; retain the same campaign caps.
No other old phase is reactivated. Routine CPU checks are recorded separately.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase p0_fitting_forward

Use trusted GPU access and verified preinitialization memory growth. The
supervisor snapshots code, plan and inputs; saves manifest, logs, summary,
per-case progress, policy/forward tensors and result.md in a fresh attempt149
directory. Observe progress during execution; do not automatically retry a
scientific mismatch. Result note: controlled-smc-p0-fitting-forward-result-20261005.md.

Terminal state, 5 October 2026: this bounded diagnostic plan has finished; see [controlled-smc-p0-fitting-forward-result-20261005.md](controlled-smc-p0-fitting-forward-result-20261005.md). Full replication remains incomplete. New launches require the next bounded implementation/qualification contract within the remaining allocation.
