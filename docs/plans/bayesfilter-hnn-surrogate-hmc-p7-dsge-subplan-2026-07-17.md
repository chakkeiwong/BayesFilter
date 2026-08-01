# P7 Subplan: NK And Rotemberg Requalification

Phase objective: requalify the NK-like analytic, small/real NK, small/real NK
SVD-UKF, Rotemberg linear-Kalman, and Rotemberg second-order SVD configurations
without importing the historical invalid delayed-acceptance kernel.

Entry conditions: P6 closes; local source/model/data artifacts are readable;
each target is classified as analytic, exact Kalman, or named approximate
filter posterior; 30-GPU-hour ceiling and per-configuration early stops freeze.

Required artifacts: source/target/transform/filter identity ledger; BayesFilter
adapter or bounded bridge per configuration; parity evidence; corrected-kernel
training/tuning/sampling/cost results or precise requalification blockers.

Required checks/tests/reviews:

- inspect historical target and HNN source but port only scalar-position-force
  ideas, never historical delayed-acceptance execution;
- value/score/transform/Jacobian parity at fixed points;
- deterministic filter endpoint replay and correct filter-posterior label;
- current modern diagnostics and telemetry; retain qualified historical labels
  until fresh evidence passes;
- configuration-local cost ceiling and early stop after a decisive blocker.

Evidence contract: each result applies only to its exact solution/filter target.
No DSGE family pass can substitute for another configuration.

Forbidden claims/actions: no package/environment mutation without approval; no
fresh noisy filtering endpoint; no missing stage one; no broad "DSGE works"
claim; no promotion of a timed-out or marginal historical result by prose.

Exact P8 handoff: every Tier B configuration is classified with an evidence
path and nonclaim; remaining artifacts are reproducible; no unspent mandatory
repair remains within budget.

Stop conditions: target/source unavailable, bridge invalid, required
environment mutation, or budget exhaustion. Stop affected configurations and
continue synthesis.

Phase-end duties: run checks; write P7 result; refresh P8; review P8; continue
if no real blocker.

## 2026-07-18 Execution Refresh And Skeptical Audit

P6 demonstrated the relevant Tier B preservation failure: an exact historical
target can remain reconstructible while its selected frozen NeuTra chart does
not. The master program requires an admitted target-specific chart when one is
the basis of historical inclusion. Target-specific training in this program
means training the new scalar residual force inside that chart; it does not
authorize retraining or substituting a new NeuTra chart and calling the result
a requalification.

Each P7 cell must therefore pass this cheap ladder before implementation or GPU
work:

1. Identify the exact selected historical result, target/filter/solution
   preflight, frozen constants, parameter transform, dtype, and data.
2. Locate the durable selected chart state, including every affine and learned
   transport parameter needed to reproduce `T`, `T^{-1}`, and its log-Jacobian.
3. Replay the target signature, chart round trip, transformed scalar, and score
   at fixed probes using the historical implementation.
4. Identify an executable deterministic BayesFilter endpoint and score bridge
   for the same target. Approximate-filter cells must preserve their exact
   named filter semantics and frozen constants.
5. Only after steps 1-4 pass may a BayesFilter adapter, endpoint-only parity
   smoke, residual-force protocol, and corrected-chain run begin.

Missing final JSON alone is not necessarily fatal when a complete, hash-bound
training/replay state survives and the exact selected chart can be reconstructed.
Conversely, a launch summary that merely names missing replay/training outputs
does not reconstruct them. A fresh chart, nearby model, different solution
order, different filter, or re-created training run is a new campaign and must
not be substituted here.

### Research Intent And Evidence Contract

| Field | Binding P7 rule |
|---|---|
| Question | Can corrected neural-force HMC be freshly tested in each exact historical DSGE NeuTra chart and target? |
| Candidate | newly trained target-specific scalar residual force inside the replayed historical chart |
| Baselines | zero residual and true-gradient HMC in that same chart; exact same-target raw HMC where admitted |
| Promotion criterion | exact target/chart/endpoint parity plus energy, sampler-health, reference, and cost gates |
| Promotion veto | target/filter/transform mismatch, non-finite/status failure, energy failure, or modern retained diagnostic failure |
| Continuation veto | selected chart state or deterministic endpoint bridge is unavailable, or the cell/phase budget is exhausted |
| Repair trigger | local adapter, serialization, XLA, or harness defect with unchanged target/chart and budget |
| Explanatory only | residual loss, acceptance, runtime, and one-seed ESS differences |
| Nonclaims | no pooled DSGE claim, filter exactness, ranking, default readiness, or replacement of a missing chart |
| Artifact | unique cell directory below `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p7/` |

### Default And Assumption Audit

| Choice | Provenance / status | Failure mode and early diagnostic |
|---|---|---|
| Historical selected chart | master-program admitted-chart rule; binding identity | launch summary points to deleted state; require actual readable tensors and replay |
| Historical target/filter | named Tier B configuration; binding identity | nearby DSGE route substituted; hash preflight, constants, transform, and value/score probes |
| P2 residual recipe grid | warm-start candidates only | target-specific undercapacity; heldout screen and downstream corrected-chain gate |
| Preserved HMC settings | nomination only | legacy diagnostics or tuning mismatch; fresh current-policy tuning |
| One seed | owner cost policy | seed-specific result; exact one-seed label and no arm ranking |
| GPU/XLA/TF32 with memory growth | repository policy | device or compilation mismatch; trusted preflight before serious launch |

Audit verdict: `PASS_AFTER_REFRESH`. The earlier P7 text did not distinguish
historical chart replay from fresh chart retraining and could have converted a
preservation failure into a different experiment. The refreshed ladder closes
that path and keeps a missing cell local to P7.
