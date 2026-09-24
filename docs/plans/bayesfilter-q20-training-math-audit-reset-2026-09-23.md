# q20 training mathematical audit: continuation note

Owner request: explain the remaining poor training by tracing code and math.
Completed source/saved-data analysis is in
[the result](bayesfilter-q20-training-math-audit-results-2026-09-23.md), under
[the plan](bayesfilter-q20-training-math-audit-plan-2026-09-23.md).

Use the frozen `/tmp/BayesFilter-q20-training-repair-20260923-r1` source for
the actual repair histories. The ordinary master was repaired to continue
training and try alternatives; the recent maps instead came from a separate
fixed-tranche repair controller. Do not revive the obsolete claim that the
ordinary master still skips all HMC nominees.

New evidence: the depth-four physical proposal coordinates have 99.866–99.999%
of their saved-point variance explained by affine functions of the base draw.
Its second latent residual coordinate accounts for 89.02% of squared residual,
and an affine fit explains only 0.573% of that coordinate's residual energy.
The RKL first-gradient formula is supported, but finite-network stationarity
does not guarantee pointwise whitening. Output-scale attenuation remains.
Clipping was 0–0.68% in the intervention tranches; it cannot alone explain
their remaining error.

Correct the endpoint interpretation: `continuing_improvement` compares parent
to final map. It does not establish improvement at the final updates. The
depth-four loss falls early then is flat/noisy. Endpoint per-layer gradient
noise, actual update sizes, paired late-checkpoint progress, and effective
nonlinear capacity are the next discriminating measurements. No unique cause
or selected repair is established; there is no candidate ranking or sampler
promotion.

No target evaluations, new training, or GPU processes were launched for this
audit. Budgets remain 143975.28597232018 campaign seconds and
7.711412891243526 diagnostic seconds. The prior question requesting two more
diagnostic minutes remains unanswered; this audit did not treat it as approval.
The batch128 post-training report still has 440/1,000 points. Preserve its
partial output and the three completed reports. Runtime code, saved maps,
and unrelated worktree changes were not edited by this audit.
