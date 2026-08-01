# P6 Subplan: Standard Target Requalification

Phase objective: requalify funnel, ill-conditioned Gaussian, and German
logistic-regression targets under the corrected BayesFilter kernel.

Entry conditions: Tier A closes; exact target definitions and datasets are
locally available; portability audit passes; 18-GPU-hour ceiling is frozen.

Required artifacts: one BayesFilter target adapter and identity per
configuration; source-to-local parity; per-target training/tuning/sampling/cost
records; historical-versus-current status ledger.

Required checks/tests/reviews:

- analytic checks for Gaussian/funnel where available;
- dataset and preprocessing identity for logistic regression;
- source target-value/score parity before HNN work;
- current sampler health policy and target-appropriate posterior checks;
- truth-tail only where generating truth exists; otherwise use a declared
  analytic/reference comparator without inventing truth evidence.

Evidence contract: P6 is a fresh corrected-kernel requalification. Historical
NeuTra status is context, not the result.

Forbidden claims/actions: no automatic import of historical passes, no truth
claim for real data, no cross-target force, and no ranking from three cells.

Exact P7 handoff: all three cells are confirmed, failed, or
requalification-blocked with target/source evidence and remaining budget.

Stop conditions: unavailable local target/data source, identity mismatch that
needs scientific redesign, or P6 budget exhaustion. Such issues block the cell,
not P7 or Tier A close.

Phase-end duties: run checks; write P6 result; refresh P7 from actual portable
routes and costs; review P7; continue if no real blocker.

## 2026-07-18 Execution Refresh And Skeptical Audit

P5 closed with both Tier A cells independently passing. P6 retains a maximum
of six GPU wall-hours per cell and one repair attempt per cell. The exact
historical sources are locally readable under `/home/chakwong/python`; source
availability alone is not admission. Each cell must first pass the following
cheap ladder before training or sampling:

1. Freeze the exact source target, dimension, dtype, transform, data, and
   preprocessing identity from the historical result and its implementation.
2. Implement or identify a BayesFilter TensorFlow scalar-value and score route
   with XLA enabled; compare value differences and gradients at fixed probes.
3. Reconstruct and replay the exact frozen NeuTra transport. If the checkpoint,
   architecture, transform convention, or preprocessing cannot be reproduced,
   classify only that cell `REQUALIFICATION_BLOCKED`.
4. Run an endpoint-parity and one-proposal corrected-kernel smoke before any
   training. A target mismatch, non-finite endpoint, or unverifiable transform
   is a cell-local continuation veto.
5. Only then run the target-specific force protocol and retained sampler.

The frozen historical configurations are paper-scale dimension 100 funnel,
paper-scale dimension 100 ill-conditioned Gaussian, and the 51-dimensional
gamma-scales German-credit logistic posterior. Those dimensions are binding
unless the historical evidence identifies a different exact selected cell.
The synthetic targets use their analytic target laws as reference checks. The
German cell uses the locally preserved `german.data-numeric` bytes and exact
official-style preprocessing; it has no generating-truth claim.

### Research Intent And Evidence Contract

| Field | Binding P6 rule |
|---|---|
| Question | Can the corrected BayesFilter kernel be requalified on each exact historical target and frozen chart? |
| Candidate | target-specific frozen scalar residual force in the reconstructed chart |
| Baselines | zero-residual corrected kernel and same-chart true-gradient HMC; analytic checks for synthetic cells and preserved posterior reference for German |
| Promotion criterion | endpoint parity, energy identity, modern retained diagnostics, and target-appropriate reference agreement all pass |
| Promotion veto | non-finite/status failure, energy mismatch, R-hat above 1.01, bulk ESS below 1000, tail ESS below 400, or failed analytic/reference check |
| Continuation veto | target/data/transform identity cannot be reconstructed, shared corrected kernel fails, or cell/phase budget is exhausted |
| Repair trigger | local adapter, serialization, XLA, or harness defect with unchanged target and budget |
| Explanatory only | force loss, acceptance, ESS magnitude above threshold, runtime, and one-seed arm differences |
| Nonclaims | no ranking, universal reliability, calibration, or German truth recovery |
| Artifact | unique directory below `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p6/<cell>/` |

### Default And Assumption Audit

| Choice | Provenance / status | Failure mode and early diagnostic |
|---|---|---|
| Historical target dimension and preprocessing | historical selected configuration; binding identity, not a new default | wrong posterior; source/hash and fixed-probe parity before training |
| Historical NeuTra chart | warm-start hypothesis only | stale or irreproducible transform; checkpoint/architecture replay and transformed-value parity |
| P2 force recipes | candidate grid only, never transferred winner | capacity or scaling failure; heldout force diagnostics nominate but cannot promote |
| Four chains and modern thresholds | master program and current BayesFilter policy | false pass from legacy diagnostics; run shared rank/folded R-hat and ESS code |
| One data/sampler seed | owner cost-bounded diagnostic policy | seed-specific viability; label one-seed result and do not rank arms |
| GPU/XLA/TF32 with memory growth | repository execution policy | environment mismatch; trusted device/XLA smoke records settings before serious run |

Audit verdict: `PASS_AFTER_REFRESH`. The earlier subplan was under-specified:
it could have treated a newly constructed target or an unavailable historical
chart as a requalification. The ladder above makes identity reconstruction a
precondition, separates reference checks from promotion, and permits an honest
cell-local blocker. No P6 serious command may run before its cell passes steps
1-4.
