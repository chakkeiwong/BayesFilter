# Algorithm, results and writing audit

20 October 2026. Documentation closeout against merged commit `6d9e036c2`.
The campaign implementation and terminal SIR evidence were already committed
as `0318dc6f2` and merged with remote main. This update adds no numerical run,
budget allocation, default change or release qualification.

## Scope and decision

The official modern-HMC chapter now describes the executed experimental v8
procedure: independent finite-horizon acceptance trials, fresh verification,
retention of every qualified member, permitted repairs and separate warm-up,
posterior and reference assessment. Its tables contain the completed SGQF and
SIR cohorts, including precision failures, reference failures and resource
censoring. The tuning chapter, diagnostics chapter and API reference agree.

The pre-execution skeptical review is in
[the integration plan](../../plans/bayesfilter-modern-hmc-sir-documentation-integration-2026-10-20.md).
It identified the main risks as using descriptive cohort counts to rank methods,
mistaking tuning qualification for posterior accuracy, overstating the audit,
and losing independent changes while merging the shared workspace. The final
revision addresses those risks without changing the scientific criteria.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Commit and publish the documentation update to the authorized Git remote | Algorithm/code correspondence, complete cohort accounting and consistent documentation checked | No lost prior displays/citations; no undefined references/citations; changed pages readable; regressions pass | Model review cannot establish a human reader's assessment of the prose | Continue the master's separately scoped validation phases | No sampler ranking, precision calibration, new default or release completion |

## Writing and source review

Applied the files in `~/python/claudecodex/policies/` (the installed directory
is `policies`, rather than `polices`):

- `global-scientific-coding-agent-policy.md`, especially reader-facing prose,
  numerical provenance and statistical evidence discipline;
- `humanizer-ai-writing-patterns.md`, used as a diagnostic list subordinate to
  scientific meaning; and
- `scholarly-literature-audit-policy.md`, especially the separation of source
  results, local derivations and implementation evidence.

Read the modern-HMC chapter and the changed tuning/diagnostics passages. Rewrote
stale future-tense recommendations, administrative metaphors and an implication
that a local NUTS comparator should be run. The authors' historical NUTS results
remain literature context; BayesFilter execution remains excluded. Replaced
vague audit wording with its actual scope. The independent diagnostic audit
reproduced streams, means and batch/control arithmetic, but did not independently
repeat all TensorFlow ESS and R-hat calculations. The SIR closeout checked saved
decisions, accounting and hashes, not an independent statistical recomputation.

Existing citations, source-version qualifications and mathematical derivations
were retained. No new literature finding is introduced, and this is not a fresh
MathDevMCP audit or a new literature-completeness certification. The existing
source and symbolic-audit records apply to the unchanged derivations. Human
reader assessment of naturalness remains pending; the author/model review and
successful builds are not a substitute for it.

## Numerical grounding and preservation

The finite-horizon acceptance description follows `hmc_acceptance_trials.py`
and the shared candidate-set controller. The reference formula is taken from
`modern_full_model_consumer.py`; the SIR functionals follow the reference
compatibility driver and frozen designs. All cohort thresholds are identified
as recorded experimental choices, not general numerical defaults.

Reconciled all eight method/target rows against the compact evidence here.
Original and copied SHA-256 values match for seventeen evidence files. SIR has
24 verified members, 23 posterior passes and 21 joint posterior/reference
passes; one member is resource-censored. Inclusive numerical charges sum to
`197315.48680924764` seconds. The chapter retains the earlier nine-grid MAMS
failures, the missed-mode mixture diagnostic and both interval studies, so
successful later cohorts do not erase contrary evidence.

Protected baselines are the merged versions of the three chapters. The
machine-readable [checks](documentation-checks.json) compare complete display
blocks and citation commands after whitespace normalization, not just counts.
No prior display, citation command or label was removed or changed.

| Chapter | Equation/align blocks, before → after | Bracket displays | Citation commands | Labels, before → after |
| --- | ---: | ---: | ---: | ---: |
| Modern HMC | 114 → 116 | 2 → 2 | 34 → 34 | 133 → 139 |
| Tuning interfaces | 9 → 9 | 8 → 8 | 4 → 4 | 13 → 13 |
| Diagnostics | 6 → 6 | 0 → 0 | 3 → 3 | 7 → 7 |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Preserve failed precision/reference deliveries and the resource-censored member; none loses acceptance/health-based tuning membership. |
| Statistically supported ranking | None. Frozen cohorts have unequal proposals/costs and lack matched independent-fit replication. |
| Descriptive-only differences | Member pass counts, retained lengths, reference screens and worker times. |
| Default-readiness | Not established; repeated-look MCSE coverage remains unresolved. |
| Next evidence needed | Matched model/independent-fit confirmation, precision/stopping calibration and actual MacroFinance/DSGE consumer qualification under priced plans. |

## Builds and rendered review

From the isolated checkout's `docs` directory, ran:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bayesfilter-latex-20261020/chapter modern_hmc_survey.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bayesfilter-latex-20261020/monograph main.tex
```

Both builds passed: the chapter is 46 pages and the monograph is 758 pages.
Neither has undefined references or citations. The standalone chapter has no
overfull/underfull hbox or hyperref warnings. The full monograph still reports
222 overfull and 767 underfull hboxes and three hyperref warnings; this task
does not certify the layout of every other chapter.

Inspected an overview of all 46 chapter pages, then detailed rendered pages
32 and 39–43 for the interval study, algorithm, settings, reference equation,
results and conclusions. Inspected monograph PDF pages 478–479, 539–541, 688
and 690 for the related tuning/diagnostics passages, algorithm and results
table. The inspected additions have no clipping or equation/table overlap.
The algorithm remains on a readable dedicated float page. Page numbers here
are PDF page indices, not printed chapter page numbers.

The expanded post-merge suite passed **356 tests**. The final documentation
contract rerun passed **15 tests in 2.75 seconds**. Both deliberately hid GPUs;
these are engineering/document checks, not new GPU performance evidence.
Commands and environment are in [integration checks](integration-checks.md),
and exact summaries/log hashes are in `documentation-checks.json`.

## Post-review limits

The strongest alternative interpretation of the successful cohorts is that
frozen geometry and proposals work for these particular finite-filter targets
without implying reliable adaptation on new fits. The chapter states this
limit. A matched replicated comparison or a failed independent reference could
change the scientific conclusion. The weakest evidence remains precision at
the selected stopping time and transfer to actual consumers. Neither the prose
rewrite nor the Git merge resolves those scientific questions.
