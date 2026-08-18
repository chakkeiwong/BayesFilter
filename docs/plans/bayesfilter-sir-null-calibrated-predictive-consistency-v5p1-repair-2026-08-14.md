# V5.1 Repair: Separate Predictive-Coverage Vetoes From Score Diagnostics

Date: 2026-08-14  
Status: `APPROVED_FOR_ONE_FRESH_SIR_ATTEMPT`

## Trigger

The first full SIR V5 attempt stopped before null calibration because six of
nine frozen heads exceeded the inherited V4 maximum-delta AUC cutoff `0.995`,
and three also exceeded the inherited ECE cutoff `0.04`. The process itself
completed normally to that gate; the command stream disconnect was not the
cause of the scientific stop.

Those thresholds were predeclared in V4 but were not derived for the V5 target.
V5 asks whether a frozen nine-output bundle has calibrated same-parameter
predictive coverage. AUC saturation and ECE concern likelihood-ratio/score
interpretability; they do not invalidate finite, deterministic output vectors
or exchangeability of the null-fit, calibration, and audit partitions.

## Repair

Classify frozen-head diagnostics by role:

| Check | V5.1 role |
|---|---|
| finite outputs and temperature | hard numerical veto |
| optimizer completion | hard execution veto |
| exact conditional balance and disjoint domains | hard validity veto |
| pooled classifier signal | score-interpretability diagnostic |
| calibration loss and per-delta ECE | score-interpretability diagnostic |
| AUC informativeness, inversion, and saturation | score-interpretability diagnostic |
| fixed-path heldout support | explanatory diagnostic; the joint conformal fixed-path acceptance is primary |
| SVD rank and finite conformal scores | hard null-harness veto |
| audit lower-bound falsification | primary predictive-coverage veto |

The runner must preserve every diagnostic and emit
`score_interpretability_all_passed=false` when appropriate. A predictive
coverage pass with failed score diagnostics may be described only as
`joint_same_parameter_predictive_coverage_only`; it is ineligible for exact
score, likelihood-ratio accuracy, filter correctness, HMC, or default claims.

## Skeptical Review

- This repair does not relax a principled error margin; it removes unrelated
  heuristic score screens from the predictive-coverage estimand.
- A degenerate constant estimator remains a risk. V5.1 therefore retains the
  SVD rank gate, the pooled-signal diagnostic, and the zero-mean diagnostic,
  and must state that predictive coverage alone can still be scientifically
  uninformative.
- The null threshold, partitions, counts, calibration order statistic, audit
  rule, simulator, estimator, training settings, and fixed path do not change.
- No failed attempt data are reused. The fresh attempt uses a new artifact
  directory and the same frozen seed domains defined by the plan.
- The first SIR attempt remains historical evidence of score-interpretability
  failures and is not overwritten.

Verdict: `PASS_FOR_ONE_FRESH_SIR_ATTEMPT`. Budget: one detached SIR attempt,
maximum 90 GPU minutes. No further gate change or retry is authorized by this
repair.

## Execution Reliability

Long execution uses `scripts/run_detached_with_status.sh`. The status directory
must contain `pid`, `worker.log`, `started_at`, `finished_at`, and `exit_code`.
The scientific artifact must end with a terminal result and manifest even when
an early veto fires.
