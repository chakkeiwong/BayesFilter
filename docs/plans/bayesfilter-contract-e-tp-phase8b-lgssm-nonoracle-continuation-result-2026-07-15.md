# Contract E--TP Phase 8B LGSSM Non-Oracle Continuation Result

metadata_date: 2026-07-15
status: PASS_LGSSM_NONORACLE_HANDOFF
entry_plan: `docs/plans/bayesfilter-contract-e-tp-phase8b-nonoracle-continuation-and-nonlinear-chart-handoff-2026-07-15.md`

## Result

Exact bounded future-window likelihood features were tested at look-ahead
lengths `2,4,8`. These use future observations from the fixed offline dataset
but not the whole remaining horizon. They are appropriate as offline likelihood
features for the current HMC research target, not for strictly online filtering.

At `T=10`, windows 2 and 4 failed the frozen value/score screens. Window 8
passed and was the only candidate advanced to `T=50`.

| Window | T | Value difference | Largest score relative error | Sign reversal | Decision |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 10 | `0.0058602183` | `0.152989687` | none | fail |
| 4 | 10 | `0.0010569396` | `0.163927101` | none | fail |
| 8 | 10 | `-0.0004638756` | `0.003048063` | none | pass |
| 8 | 50 | `-0.0008675320` | `0.007361860` | none | pass |

The controlling `T=50` artifact is
`docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t50_order5_lookahead8_attempt1_20260715/result_aggregate.json`.
All five score directions pass same-scalar FD with maximum relative error
`1.93e-8`; all 49 charts are fixed, strictly positive, and full rank.

## Verdict

The one-step progressive score interaction is insufficient, but a bounded
eight-observation continuation feature is sufficient for the frozen LGSSM
center through `T=50`. This is a target-specific starting hypothesis for
nonlinear models, not a transferable default: every nonlinear row must prepare
and test its own finite continuation program at declared short prefixes.

Phase 8B LGSSM gate: `PASS_NONORACLE_LOOKAHEAD8_HANDOFF_TO_NONLINEAR_PREPARATION`.
