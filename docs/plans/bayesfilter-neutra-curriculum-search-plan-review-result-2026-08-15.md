# NeuTra curriculum search plan review and mechanics test result (2026-08-15)

## Outcome

The scientific plan for target-specific NeuTra curriculum selection is written,
reviewed, and supported by a tested generic policy layer. No scientific training
campaign was launched in this step.

The review found one material flaw in the first draft: loss improvements from
group probes descended from different parent checkpoints are not directly
comparable enough to select the final curriculum. The plan and implementation
were revised so beam search only nominates a terminal candidate set. A separate
paired, equal-total-budget full-protocol tournament on fresh selection data is
mandatory before freezing a protocol for a final run.

## Implemented mechanics

`bayesfilter/inference/neutra_curriculum_search.py` provides:

- prerequisite-constrained variable-group definitions;
- replicated equal-budget probe aggregation;
- one-sided lower uncertainty bounds for local group nomination;
- bounded beam width, depth, and total probe-call accounting;
- common-parent state hash and incoming-loss enforcement by replicate;
- preservation of accepted sequences and the complete final beam;
- explicit budget-exhaustion and no-passing-group statuses; and
- paired full-protocol uncertainty-set selection with equal update budgets and
  common selection partitions.

The module is a pure policy layer. It imports neither TensorFlow nor NumPy and
contains no target/model names. Model adapters remain responsible for actual
GPU/XLA training and for producing immutable probe observations.

## Evidence contract status

| Item | Status |
|---|---|
| Research question | Preserved: select model-dependent curriculum scientifically rather than transfer a fixed order. |
| Baseline ladder | Cold joint, tuned cold joint, historical fixed staging, searched curriculum. |
| Probe criterion | Replicated held-out loss improvement per update with a predeclared lower uncertainty bound. |
| Cross-branch selection | Repaired: a separate paired full-protocol tournament is required. |
| Promotion criterion | Untouched target-specific predictive-output distribution equivalence. |
| Proxy promotion | Forbidden: search loss nominates only. |
| Audit leakage | Forbidden through disjoint probe, protocol-selection, final-training, and predictive-audit partitions. |
| SSL-LSTM claim | Not evaluated and not supported by this mechanics test. |

## Skeptical review result

| Risk | Verdict |
|---|---|
| Fixed schedule merely renamed as generic | Repaired by adapter-supplied groups plus prerequisite-constrained search. |
| Unequal probe work | Fail-closed checks require exact declared updates and replicate counts. |
| Different parent checkpoints compared as peers | Repaired by common-parent enforcement locally and the full-protocol tournament globally. |
| Beam search silently declares a winner | Repaired: representative sequence is bookkeeping only; complete final beam is retained. |
| One seed chooses the order | Replicated uncertainty rule required. |
| Selection loss becomes scientific correctness | Vetoed by the mandatory untouched predictive audit. |
| Complex protocols receive more final budget | Full protocols must share an exact update budget. |
| Audit data influence search | Partition identities are distinct by contract. |

Review verdict: the plan is coherent for a Gaussian/banana control campaign.
It is not yet an SSL-LSTM execution plan because the SSL-LSTM variable-group
adapter, compute budget, and predictive-audit binding still require a separate
target-specific review.

## Test result

Focused search-policy tests cover graph validation, uncertainty rejection,
beam behavior, common-parent evidence, exact probe budgets, budget exhaustion,
nonfinite rejection, paired protocol selection, equal full budgets, and paired
selection partitions.

Combined verification:

```text
56 passed
```

Python compilation, lazy public API import, target/backend-name scan, and
`git diff --check` also passed. The warnings in the broader suite are existing
TensorFlow AutoGraph/Gast deprecation warnings.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the generic curriculum-search policy | Focused mechanics and regression tests | No engineering veto | Real probe callbacks are not yet exercised | Build Gaussian/banana adapters and canary | Scientific effectiveness |
| Require full-protocol tournament | Review found local cross-parent scores incomparable | Initial design flaw repaired | Practical loss tolerance needs calibration | Predeclare tolerance in the control campaign plan | Statistical superiority |
| Do not launch SSL-LSTM search yet | SSL-LSTM groups and budget are not reviewed | Target-specific protocol requirement unmet | Appropriate group ownership and search scale | Inspect SSL-LSTM adapter after controls | SSL-LSTM readiness |
| Use Gaussian/banana as first controls | Exact-law validation already exists | No harness veto known | Search may still select poor curricula | Run bounded GPU/XLA control campaign | Universal generalization |

## Inference status

| Evidence question | Status |
|---|---|
| Hard veto screen | Mechanics passed; no scientific target run occurred. |
| Statistically supported ranking | None. |
| Descriptive-only differences | None generated in this step. |
| Default readiness | Not supported. |
| Next evidence needed | GPU/XLA Gaussian/banana curriculum search, full-protocol tournament, fresh final rerun, and exact-law audits. |
