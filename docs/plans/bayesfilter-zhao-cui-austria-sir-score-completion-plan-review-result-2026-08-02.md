# Zhao-Cui Austria SIR Score Completion Plan Review Result

Date: 2026-08-02

Status: `PASS_AFTER_REVISION_FOR_BOUNDED_EXECUTION`

## Decision

The plan is executable after material revisions. The user explicitly requested
thorough review and execution after the plan disclosed that it replaces the
trained-TT normalizer with a new frozen importance-filter scalar. That request
accepts this finite-program target replacement only.

The revised plan is mathematically coherent for its narrow claim. For a
literal theta-independent branch, the carried derivative
`D[t-1, ancestor] = grad_theta log W[t-1, ancestor]` is the correct term, and
the likelihood-weighted branch-score average is the derivative of the declared
log-mean-exp multi-branch scalar. This does not differentiate the TT trainer or
the historical `sum_t(log(sirt.z_t)-const_t)` value.

## Material Findings And Repairs

| Finding | Severity | Repair |
|---|---|---|
| The generic frozen evaluator observes its first stored state, but Zhao-Cui's Austria event order is `x0` followed by `y1:y20`. | Blocking if reused | Bind the existing source-order evaluator, whose initial correction has no observation and whose time recursion consumes exactly `y1:y20`. |
| The plan proposed independent tuning scopes by final horizon even though a sequential proposal at time `t` depends only on `y1:yt`. | Material | Train and freeze one T20 chain step by step, then test identical prefixes at T1/T2/T3/T5/T10/T20. |
| FP32/TF32 was the terminal claim hypothesis despite a five-significant-digit score requirement. Existing FP32 diagnostics use materially looser tolerances. | Blocking for the requested tolerance | Make FP64 GPU/XLA the claim dtype. Keep FP32/TF32 as parity/performance evidence only. |
| Score relative tolerance was `5e-5`, looser than the user-selected five-significant-digit material rule. | Material | Cap both deterministic `score_atol` and `score_rtol` at `5e-6`; calibration may tighten but not loosen. |
| Branch count and proposal veto were unresolved. | Material | Freeze eight branches with 2/4/8 growth reporting, `ESS/N >= 0.10`, maximum normalized weight `<=0.10`, finite weights, and finite corrections at every time/branch. Treat these as viability vetoes, not score proof or ranking evidence. |
| The no-Python-loop rule covered only the online evaluator. | Material relative to the user's XLA direction | Apply it to every claim-owned numerical training, compilation, tuning, selection, and runtime path. Python remains only at static configuration and artifact I/O boundaries. |
| Existing Austria learned-TT and retained/TTSIRT proposal paths use Python optimizer, axis, time, or microbatch loops. | Blocking for those proposal implementations | Use exact locally optimal diagonal-Gaussian conditionals as rank-one squared-TT/analytic-KR components. Test the origin component first, then an exactly scored fixed 27-component guide mixture. Keep learned higher-rank fitting as a fallback only after an XLA-native repair. |

## Source Audit

```text
decision: sufficient for direct source classification, not a literature-completeness survey
metadata_date: 2026-08-02
seed_papers: Zhao and Cui, JMLR 2024
source_support_summary: Algorithm 2/3, Section 6.3, and pinned author full_sol.m and eg3_sir/mainscript.m inspected
citation_venue_summary: not required and not checked; no novelty or literature-completeness claim
backward_snowball_summary: not required for the direct operation mapping
forward_snowball_summary: not checked
quarantined_sources: none known; formal retraction/erratum search not performed
top_omission_risks: later parameter-score extensions may exist but cannot change what the inspected Austria experiment computes
claim_support_gaps: no inspected source derives the repository external three-coordinate Austria score
next_required_actions: preserve extension_or_invention classification and exact paper/author-code anchors
what_is_not_concluded: literature completeness, novelty, or source-faithful Austria parameter inference
```

The checked paper fixes `kappa_j=0.1` and `nu_j=18` for Austria and states that
the goal is inference of the 18-dimensional state. The author Austria script
sets `d=0`. Therefore the external parameters
`(log_kappa_scale, log_nu_scale, log_obs_noise_scale)` and their score are a
BayesFilter `extension_or_invention`. Squared-TT, marginalization, and KR
conditional operations may be source-grounded individually; the assembled
frozen score route is not source-faithful Austria parameter inference.

## Independent Review Availability

The user-requested Fable review was attempted through the approved bounded
read-only wrapper:

1. Sonnet health probe from the repository was blocked by Claude workspace
   trust; the neutral-directory retry returned `CLAUDE_PROBE_OK`.
2. The bounded plan-read probe returned no output.
3. The one-path Fable plan review returned no output.
4. A fixed-token Fable health probe returned no output.

No Fable verdict is claimed. Reviewer unavailability is advisory under the
active repository policy and does not override the mathematical/source audit,
focused checks, evidence contract, or stop conditions.

## Execution Gate

Proceed in bounded order. Phase 0-2 must first prove target identity, manual
same-scalar score identity, graph-native XLA control flow, forbidden-op absence,
and the FP64 five-significant-digit gate. Proposal tuning and later horizons may
start only after those gates pass. A failed candidate triggers the planned
repair; a wrong target, invalid measure, failed tail derivative, invalid
reference, exhausted budget, or absence of any viable T1/T2/T3 proposal is a
continuation veto.

Passing closes only
`PASS_T20_ZHAO_CUI_DERIVED_FROZEN_FINITE_SCORE`. It does not establish exact
physical likelihood, source-faithful Austria parameter inference, arbitrary
theta correctness, HMC readiness, posterior correctness, default readiness,
production readiness, or superiority.
