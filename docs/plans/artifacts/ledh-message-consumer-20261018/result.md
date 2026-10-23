# Message/consumer continuation result — 18 October 2026

## Decision

The comparison completed with a passing terminal audit, but none of the four
product-consumer candidates is promoted. This is a candidate-promotion veto,
not a continuation veto. The exact future message reduces the product
consumer's squared error relative to the fitted message descriptively in four
of six global-condition/score-coordinate cells, but the repaired consumer
still does not beat the unguided global proposal with statistical support.

The result rejects the claim that fitting error alone explains the LEDH score
error. It does not reject backward information, KDM, or the research direction.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain as diagnostic candidates; withhold promotion | No supported gain in 36 simultaneous contrasts | All four candidates lose at least one heuristic comparison; numerical validity passes | Eight saved-fit clusters and finite-particle error | Paired particle-count stress diagnostic | No default readiness, general superiority or rejection of backward guidance |

## Execution and validity

- Branch: `ledh-refactor-with-policy-fix`; commit:
  `ea1018de1da55ed02a29a8d60f08a6827fce8a42`.
- The GPU run used 24 saved fits, 192 fresh common-random-number clouds and
  2,688 rows: three conditions, eight fit clusters, eight replicates and 14
  arms. The fitted and exact messages used the same product-moment consumer;
  the old consumer was retained as an attribution control.
- The independent CPU reference agreed with the enumeration/grid reference
  to `1.8e-15` in value and `4.5e-16` in score. All 16,128 trace checks and
  48 finite-difference checks passed; the largest finite-difference error was
  `1.68e-8`. No continuation veto fired.
- Inference is conditional on the saved fits and fixed datasets. The eight
  fit clusters provide a small-cluster approximation with seven degrees of
  freedom; it is not fresh-fit or population-generalization evidence.

## Score results

The primary quantity is squared error against the checked KSC model-score
reference. Representative global-arm MSEs are:

| Condition / coordinate | Unguided global | Fitted product | Exact product |
| --- | ---: | ---: | ---: |
| ordinary / 0 | 0.000304 | 0.000222 | 0.000211 |
| ordinary / 1 | 0.033379 | 0.040861 | 0.043290 |
| negative tail / 0 | 0.001200 | 0.000763 | 0.000701 |
| negative tail / 1 | 0.065069 | 0.049778 | 0.047906 |
| positive tail / 0 | 0.030444 | 0.050670 | 0.054089 |
| positive tail / 1 | 1.055393 | 2.024746 | 1.822145 |

Across the 36 predeclared paired contrasts, zero intervals supported a gain
and zero supported harm. All contrasts remain unresolved under the
Bonferroni-adjusted 95% cluster intervals. The heuristic screen nevertheless
found losses for every candidate: 12 for fitted/global, 23 for fitted/equal,
10 for exact/global and 16 for exact/equal. These are promotion vetoes.

## Attribution

The positive-tail second score coordinate is the clearest stress case. Exact
messages reduce product/global MSE from `2.024746` to `1.822145` relative to
the fitted message, but exact product/global remains worse than unguided
global (`1.055393`). Relative to the old consumer, the exact-message product
consumer changes MSE by `-0.247737`, while the fitted-message consumer changes
it by `-0.832199`; the resulting message/consumer interaction is `+0.584462`.
These interaction values are descriptive attribution, not significance tests.

The evidence supports the following diagnosis: fitting error matters, but
the current product-moment/reset consumer and finite-particle variability
remain material. Increasing fitting capacity is therefore not the next
repair.

## Inference status

| Question | Status | Meaning |
| --- | --- | --- |
| Hard validity veto | Pass | References, finiteness, traces, derivatives and provenance pass. |
| Statistically supported ranking | None | The 36 simultaneous intervals support no gain or harm. |
| Descriptive differences | Present | Exact messages have lower observed MSE in four of six global cells than fitted messages. |
| Default readiness | Not ready | Heuristic losses veto promotion; no default changes. |
| Next evidence | Required | Test the finite-particle stress hypothesis at larger `N`. |

## Red-team and next action

The strongest alternative explanation is that the positive-tail score error is
a finite-particle variance/bias problem rather than a message or consumer
problem. The next bounded repair is a fixed-control particle-count diagnostic
at `N=64`, using fresh paired clouds and the same exact/fitted messages,
consumer, reference and stress conditions. It must not tune on the opened
evaluation data or change canonical LEDH defaults. If the larger-`N` run does
not reduce the stress error, the next repair is consumer/reset design rather
than more KDM fitting.

## Accounting

Attempt 01 used `485.430` GPU-process seconds, `0.690` CPU reference seconds
and `1.069` CPU analysis seconds. The campaign has `714.570` GPU seconds
remaining within its three-attempt/1,200-second envelope. The complete
manifest, rows, result and analysis are in
`docs/plans/artifacts/ledh-message-consumer-20261018/attempt-01/`.
