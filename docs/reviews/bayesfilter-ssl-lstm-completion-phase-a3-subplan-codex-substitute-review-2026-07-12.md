# Phase A3 Forecast Oracle And Statistics Subplan Codex Substitute Review

Date: 2026-07-13

Review class: `CODEX_SUBSTITUTE_REVIEW`

Status: `VERDICT_AGREE`

Reviewed exact path:

`docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md`

Accepted subplan SHA-256:

`67ee503a15f5e7a81ca2a37e52cc6b60264c1cff89ff5cff1a9fddd3187161c4`

Claude remained policy-unavailable. The trusted one-path Claude Opus gate was
rejected before process creation or disclosure by the environment's
external-data policy. No A3 content was sent. These fresh bounded Codex reviews
are explicitly weaker than Claude and are not Claude convergence.

## Round 1

Verdict: `REVISE`.

Material findings:

1. The quadratic MMD estimator's IID/unbiased assumption conflicted with the
   canonical MCMC and forecast-cluster dependence, and common-random-number
   exclusion was incomplete.
2. An ordinary centered/studentized bootstrap of the quadratic MMD U-statistic
   was not justified near equality because the statistic is degenerate under
   the null.
3. Feature-family intervals and the separate MMD interval lacked joint alpha
   control for conjunctive `PASS`.
4. Resampling the empirical distribution of only two or four chain identifiers
   was not inference-admissible; mechanics-only and inferential modes needed a
   hard boundary.

The scalar LGSSM formulas, phase boundary, provisional A3 versus frozen A4
distinction, and sampler/scientific nonclaims were otherwise sound.

## Visible Repair

- The quadratic statistic is now IID-unbiased only for verified IID oracle
  fixtures and a dependent descriptive U-form otherwise.
- The biased V-form remains separately labeled and explanatory.
- Common-random-number MMD rows are excluded from inference.
- Decision-bound MMD uses a separate cross-chain linear kernel contrast with
  four independent chains per arm, two disjoint chain pairs, independent arm
  banks, complete forecast-replication clusters, and chain-stratified block
  inference.
- Exact/near-null and boundary coverage checks use predeclared replication,
  slack, and exact binomial uncertainty contracts.
- Joint alpha admission requires
  `feature_alpha + mmd_alpha <= total_alpha`.
- The bootstrap keeps chains/disjoint chain-pair sequences as fixed strata and
  does not resample the tiny empirical distribution of chain IDs.
- Mechanics-only, non-admissible MMD, invalid alpha, same-chain contrast, and
  shared-bank MMD rows cannot emit `PASS`.

## Round 2

No material findings. The repaired plan consistently propagates the estimator
roles, cluster semantics, null-valid linear-MMD interval, joint error control,
small-chain boundary, tests, artifacts, nonclaims, and A4 handoff. The LGSSM
derivation remains correct. A3 constants remain test fixtures rather than A4
calibration defaults. No sampler run or scientific claim is authorized.

Final verdict: `AGREE`.

## A2 Binding Refresh

After the A2 terminal-trace parser repair, the A3 entry table was refreshed to
bind the current A2 CPU/GPU artifacts, focused tests, verifier, and bounded
implementation/trace review. A fresh exact-path review found no material
finding. The refreshed bindings preserve the fail-closed entry sequence; the
LGSSM derivation, dependent/descriptive quadratic-MMD boundary, independent
cross-chain linear-MMD inference, forecast-cluster semantics, joint alpha
control, small-chain veto, A4 calibration ownership, and forbidden claims
remain consistent.

VERDICT: AGREE
