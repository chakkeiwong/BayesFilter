# Claude handoff: tempered reverse-KL transport ensemble

Date: 2026-08-28  
Status: `TWO_STAGE_REVIEW_COMPLETED_PLAN_FINDINGS_ADJUDICATED`

Completed responses:

- mathematical review: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-math-review-reply-2026-08-28.md`
  (`AGREE`);
- implementation-plan review: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-plan-review-reply-2026-08-28.md`
  (`REVISE`); and
- disposition and active-plan repair:
  `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-review-adjudication-2026-08-28.md`.

## Purpose

Review the corrected mathematical proposal first, then review the implementation
plan. Keep the stages separate so each first request names exactly one path.
Do not edit files or run experiments. Write each response to the distinct reply
path below so mathematical and implementation findings cannot be confused.

The central correction is that proposal-aware adaptive replay remains
conditionally valid mathematics but is withdrawn as the primary
high-dimensional NeuTra foundation. The replacement retains fresh-IID-Gaussian
reverse-KL training, uses multiple transport laws as a categorical ensemble,
continues them through a proper temperature bridge, freezes them as exact HMC
coordinate charts, and uses replica exchange. Exact stationarity is not a mode-
discovery or finite-mixing guarantee.

## Stage 1: mathematical review

Send this prompt first:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex.
Do not edit, run commands, launch agents, or review the whole repo. Question:
Are the propositions and proofs mathematically correct under their stated
assumptions, and are the limits stated strongly enough? Scrutinize especially
the mixture reverse-KL identity and its gradient interpretation, the separated-
region optimum for alpha, the beta=0 diversity correction, the proper bridge,
fixed versus state-dependent chart mixtures, replica-exchange detailed balance,
the exact cold marginal, and the distinction between invariance and discovery.
Classify any issue as WRONG, MISSING ASSUMPTION, OVERCLAIM, or CLARITY ONLY and
give exact line anchors. End with VERDICT: AGREE or VERDICT: REVISE.
```

Write the response to:

`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-math-review-reply-2026-08-28.md`.

If the verdict is `REVISE`, repair and re-audit the mathematics before sending
Stage 2. Do not let an implementation plan normalize a mathematical defect.

## Stage 2: implementation-plan review

After Stage 1 is `AGREE`, send this prompt:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md.
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does this plan faithfully and feasibly implement the accepted tempered
reverse-KL transport-ensemble mathematics without reintroducing particle
circularity or confusing exactness with discovery? Audit wrong baselines,
unfair target-call accounting, proxy metrics promoted to gates, hidden numeric
defaults, beta=0 lineage collapse, improper tempering, alpha/gamma/mode-mass
confusion, per-beta/chart tuning identity, canonical sequential-HMC integration,
TensorFlow/GPU/batching/XLA compliance, stop versus repair conditions, and
whether the artifacts would answer the research question. Give exact line
anchors and distinguish blockers to implementation from blockers only to a
serious campaign. End with VERDICT: AGREE or VERDICT: REVISE.
```

Write the response to:

`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-plan-review-reply-2026-08-28.md`.

## Reviewer boundaries

- Do not infer that zero MathDevMCP findings certify the proofs; eleven
  obligations were not checkable and the full rigor reporter hit a tool error.
- Do not demand posterior samples or an importance-weighted particle measure as
  an input to the reverse-KL training objective.
- Do not accept an arithmetic average of maps as a mixture distribution.
- Do not accept the current diagnostic pure-power replica-exchange module as
  the proper-reference multi-chart implementation.
- Do not accept loss, whitening, acceptance, or swap rate as posterior
  convergence or mode-mass evidence.
- Do not reject the overall direction merely because an optional joint-mixture
  arm or one candidate lineage fails.
- Treat missing component count, ladder, ESS/MCSE target, and serious compute
  budget as deliberate serious-campaign blockers, not routine implementation
  defects, unless the plan accidentally uses them before they are frozen.
