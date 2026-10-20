# Replication-first scope recovery — 2026-09-23

The owner's current clarification is to establish the iAPF and KDM/IWSG filter replications before proceeding with their extensions or broader LEDH/score comparisons. The previous answer described the umbrella research programme too broadly.

## Recovered instructions

The exact saved session is /home/chakwong/.codex/sessions/2026/09/17/rollout-2026-09-17T02-20-45-01a0ab73-2601-7192-887a-99a5e6c2e795.jsonl. Only selected user messages were inspected.

- 2026-09-17 16:47:48 UTC (September 18 Hong Kong): “create a plan to do the comparison for these two methods. Review the plan and execute.” The immediately preceding user message identifies iAPF and KDM and asks about open-source references.
- 2026-09-20 02:28:49 UTC: the owner requested a complete R reference and comparison with the paper so that the paper results are replicated first, before using it to debug BayesFilter.
- 2026-09-20 02:35:19 UTC: the owner selected the first linear-Gaussian study, starting with bounded runs and scaling toward published settings.

The recovered historical wording explicitly commissions a two-method reference comparison and then paper-level iAPF replication. It does not establish that a full KDM learned-filter paper-replication plan was commissioned or completed. The current clarification establishes the desired two-filter priority.

## Plans and actual completion

| Document | Scope | What it establishes |
|---|---|---|
| docs/plans/younis-kdm-score-master-program-2026-09-14.md | Broad score/proposal research master | Umbrella programme, not evidence of completed replications |
| docs/plans/younis-iapf-kdm-reference-comparison-2026-09-18.md | Public-source comparison of shared iAPF and KDM/IWSG operations | Executed component comparison, explicitly excluding full learned-model replication |
| docs/plans/iapf-r-paper-replication-2026-09-20.md | Independent R reference and paper Section 5.2 first study | Explicit paper-replication plan; original-method replication remains unresolved |
| docs/plans/bayesfilter-ledh-younis-kdm-phase4b-resampling-reference-plan-2026-09-08.md | Full-mixture IWSG combined with LEDH/OT/GenUT | Local integration study, not a standalone original-filter replication |

The September 18 result at docs/plans/artifacts/younis-iapf-kdm-reference-comparison-20260918-01/result.md reports 66 passing KDM checks after matching author epsilon conventions. It explicitly excludes complete author training and the outer adaptive iAPF experiment. The code reference is the original Younis/Sudderth repository; the public R iAPF code is independent, not verified original-author code.

The iAPF adaptive ladder and terminal review explicitly say the score and QR reconstructed fitters do not close Equation 15 or full-paper replication. Running many replicates does not change that identity.

No dedicated complete standalone KDM paper-replication plan was identified in the inspected relevant plan inventory. KDM component parity cannot fill that gap.

## Next justified action

Reconcile the umbrella master into two explicit replication milestones: (1) paper-grounded iAPF filtering and the selected first experiment; (2) author-grounded KDM/IWSG filtering with an explicit end-to-end scope and acceptance criteria. Keep known source differences and unresolved author choices visible. Integration, estimator modifications and precision optimization cannot substitute for either milestone.

This note is a scope/status correction, not a new experiment authorization or compute allocation. No runtime code, master plan, existing experiment budget or saved result was changed.
