# Audit: structural SVD final execution path with tool gates

## Date

2026-05-06

## Plan reviewed

- `docs/plans/bayesfilter-structural-svd-12-phase-implementation-plan-2026-05-05.md`
- `docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`

## Audit stance

Pretending to be another developer, I re-audited the twelve-phase plan with
the execution path and local tools in mind.  The earlier plan was directionally
right, but too soft about ResearchAssistant and MathDevMCP.  The final plan
correctly promotes them into evidence gates for the phases where prose-only
reasoning would be risky.

## Tool findings

ResearchAssistant status:

- Local workspace is available in read-only/offline mode.
- Parser workflows report ready, including PDF text ingestion support.
- Broad local summary searches for Julier/Uhlmann unscented-transform support,
  SVD/matrix-backprop support, and NUTS support returned no matching local
  paper summaries during this audit.
- Interpretation: the execution plan must not assume local literature support
  already exists.  Phase 1 must produce a source-support table and use
  `source_missing` where support is absent.

MathDevMCP status:

- Tool matrix supports LaTeX search, derivation-backed claim routing, and
  document-code consistency workflows.
- Search found BayesFilter structural labels including:
  - `def:bf-structural-state-partition`;
  - `asm:bf-approximation-labeling`;
  - `prop:bf-structural-ukf-pushforward`;
  - `eq:bf-structural-ukf-prop-deterministic-completion`.
- Typed obligation routing for
  `prop:bf-structural-ukf-pushforward` located the proposition but routed the
  obligation to human review because assumptions and measure-theoretic notation
  are not enough for backend certification.
- Interpretation: MathDevMCP is useful for locating, decomposing, and auditing
  obligations, but cannot replace human derivations for the pushforward law.

## Completeness check

The final execution plan covers the required twelve phases:

1. source and derivation audit;
2. code reuse and document-code audit;
3. BayesFilter structural sigma-point core;
4. exact Kalman and degenerate linear spine;
5. generic structural fixtures;
6. MacroFinance adapter and analytic derivative classification;
7. DSGE adapter integration;
8. model-specific DSGE completion evidence;
9. derivative and Hessian safety gate;
10. JIT and static-shape production gate;
11. HMC validation ladder;
12. documentation, provenance, and release gate.

The plan also adds a necessary Preflight phase for workspace hygiene.

## Dependency audit

The dependency order is sound:

```text
Preflight
  -> source/support and derivation evidence
  -> code reuse decisions
  -> BayesFilter core value gates
  -> client adapter classification
  -> model-specific residual gates
  -> derivative and JIT gates
  -> HMC ladder
  -> release documentation
```

This ordering blocks the two main failure modes:

- implementing a cleaner version of the old full-state structural bug;
- running HMC before value, residual, derivative, and compiled-target evidence
  exist.

## Issues found in the earlier plan

1. ResearchAssistant and MathDevMCP were mentioned as optional helpers rather
   than hard gates.
2. The earlier plan did not specify what to do when ResearchAssistant finds no
   local source support.
3. The earlier plan did not specify what to do when MathDevMCP routes a claim
   to human review instead of backend certification.
4. The earlier plan did not separate source-support status from derivation
   status in the reset memo.
5. The earlier plan did not make the immediate next action narrow enough:
   Preflight plus Phase 1 should be the next execution pass, not backend code.

The final plan fixes these points.

## Remaining risks

| Risk | Mitigation in final plan |
| --- | --- |
| Local literature index lacks needed summaries | Use `source_missing`; add/review sources in a separate approved workflow before literature claims. |
| MathDevMCP cannot certify measure-theoretic pushforward claims | Record `human_review_required`; write explicit derivation and acceptance tests. |
| Client code contains useful but semantically model-owned logic | Phase 2 classifies client-owned code and blocks migration of economics. |
| Rotemberg/SGU first-order bridges are over-promoted | Phase 8 requires second-order/pruned and residual evidence before promotion. |
| SVD/eigen gradients pass simple tests but fail near repeated values | Phase 9 requires spectral-gap stress tests. |
| HMC smoke tests are mistaken for convergence | Phase 11 requires multi-chain diagnostics for convergence claims. |
| Dirty workspace files are accidentally staged | Preflight requires scoped staging and reset-memo status. |

## Decision

Approve
`docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`
as the final execution plan.

After the tool-gated Phase 1 addendum exists, autonomous execution may continue
through the BayesFilter-local value and adapter gates while preserving the
promotion blockers.  The execution pass should therefore run:

1. Preflight hygiene;
2. Phase 1 source and derivation audit;
3. Phase 2 code reuse/doc-code audit;
4. Phases 3--7 value and adapter gates;
5. Phases 8--11 as blocker evaluations unless their required evidence exists;
6. Phase 12 documentation/provenance validation.

Backend rewrites, DSGE nonlinear promotion, derivative certification, compiled
target claims, and HMC convergence remain blocked unless their phase-specific
evidence gates pass.
