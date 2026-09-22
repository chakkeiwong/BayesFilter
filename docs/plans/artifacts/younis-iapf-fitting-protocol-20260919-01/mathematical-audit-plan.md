# Mathematical explanation of the completed iAPF score campaign

Question, 2026-09-19: what was tested, which failure mechanisms are proved,
and what would be required to prove the cause of dataset 1900's remaining
observed UKF loss? This is a read-only mathematical and saved-result analysis,
not another filter experiment or promotion run.

Inspect the actual caller, Fisher statistic, fitted-twist transition, backward
fit objective, stopping controller, and saved source closure. Read the locally
stored Guarniero--Johansen--Lee paper's Propositions 2--4, Algorithms 3--4,
Section 4 and relevant appendix. Derive statements in the executed model's
notation. The paper's likelihood-variance result must not be relabeled as a
score-variance guarantee; local bounded relative-shape fitting and added score
controls must be named as adaptations.

Evidence contract: reproduce archived per-replicate MSE and its exact sample
variance/mean-error decomposition. Compute the descriptive standard error of
sample MSE on 1900 to expose uncertainty, not to make a new significance claim.
Use exact rational arithmetic to verify a two-region counterexample showing
that a small relative-shape residual need not control importance-weight
variance. Check the old CV bound and the zero-mean control expectation identity
algebraically. Saved-data arithmetic and counterexamples establish only the
stated identities/limitations; they cannot prove a causal attribution on 1900.

Skeptical audit PASS: distinguish observed failure of a declared screen from
population risk; preserve the resampled-particle versus independent-importance
sampling distinction; distinguish exact backward information from the recursive
fit's approximate next twist; distinguish positive-floor protection from
precision; no new seeds, filter calls, fits, GPU launches or retuning. Source
mismatch, missing records or inconsistent score arithmetic veto any relevant
implementation claim. Report those before interpreting it.

Use Python standard-library reporting only, at most 30 seconds, charged to the
existing CPU allowance. Preserve the command in `mathematical-checks.py` and
output in `mathematical-checks.json`; write the explanation and derivations in
`mathematical-explanation.md`. This is diagnostic artifact analysis, not runtime
code. Numerical results are immutable. No independent review is claimed.
