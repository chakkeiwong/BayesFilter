# Zhao-Cui Bounded-Reference GenUT Austria Reset Memo

Date: 2026-08-06

## Current state

- Empirical GenUT particle moments are not used as the teacher.
- Direct physical third/fourth moments from the Lane-B squared TT are wrong:
  the TT uses an algebraic map from bounded `u` to unbounded local `r`, and the
  positive defensive reference density gives divergent high physical moments.
- The implemented repair uses independent Zhao-Cui retained samples, exact
  `log p_TT - log q` correction weights, bounded-coordinate standardized
  moments, and analytical normalized-weight tangents from the issued marginal
  score.
- The fixed 64-sample T1/T2 teacher artifact passes strict reload. Its ESS is
  `63.869/64` at T1 and `63.834/64` at T2.
- The one-seed GPU/XLA ladder failed its complete hard-veto screen. The diagonal
  arm crossed the bounded chart; the uncapped arm missed the parameter-0
  absolute score-FD tolerance. Cap 2 was finite and passed all executed FD
  coordinates but is not promoted or statistically ranked.
- The three-seed run was deliberately not executed. The two serious-launch
  budget was exhausted by one pre-arm memory-order harness failure and the
  completed hard-veto smoke.
- Post-run current code exactly bypasses the teacher when all shape steps are
  zero. The preserved smoke predates this bypass and binds its own source hash.

## Canonical artifacts

- Plan:
  `docs/plans/bayesfilter-zhao-cui-bounded-reference-genut-austria-t1-t2-plan-2026-08-06.md`
- Result:
  `docs/plans/bayesfilter-zhao-cui-bounded-reference-genut-austria-t1-t2-result-2026-08-06.md`
- Teacher:
  `docs/benchmarks/artifacts/zhao_cui_bounded_reference_genut_austria_t1_t2_20260806/teacher-attempt01-n64/manifest.json`
- GPU/XLA smoke:
  `docs/benchmarks/artifacts/zhao_cui_bounded_reference_genut_austria_t1_t2_20260806/smoke-attempt02/result.json`

## Next justified action

Write a fresh plan for a target-specific bounded-teacher control repair. Use a
larger independent teacher or quantify teacher seed-split uncertainty, calibrate
diagonal/pairwise strengths and caps on disjoint data, nominate cap 2 only as a
warm start, and preserve untouched validation. Do not start T3/T20 or HMC work
until the bounded-coordinate, finite-program score, and restoration gates pass.

## Nonclaims

No exact physical moment, exact TT contraction, T20 improvement, posterior
correctness, HMC/NeuTra readiness, statistical superiority, or default-readiness
claim is supported.
