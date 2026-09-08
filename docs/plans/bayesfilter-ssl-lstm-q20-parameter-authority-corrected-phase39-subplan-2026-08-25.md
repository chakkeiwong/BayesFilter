# Corrected Parameter-Authority Phase 39 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 38 finite checkpoint receipts; unchanged target and measure  
Version: `v2.1-training-measure-bound` (historical diagnostic)  
Status: `PASS_MEASURE_SEPARATION_SPLIT_DEFECT_IDENTIFIED_HISTORICAL_V2_1`  
Local cap: 300 s

## Question

Are the persistent NeuTra residuals plausibly explained by a mismatch between
the weighted empirical measures used for train, validation, and audit, or do
the partitions have comparable theta support and target/proposal ratios?

## Evidence contract

Read only the existing N=256 M0 pilot tensors and the deterministic split
specified by the Phase 38 traces. Compute, separately for train, validation,
and audit:

- normalized theta mean and covariance;
- effective sample size fraction and maximum normalized weight;
- negative/positive sign counts for the declared mode axis;
- finite target/proposal log-density range and log-ratio range; and
- the affine train-measure oracle residual, where applicable.

The report is explanatory. It cannot establish target coverage, mode
discovery, posterior correctness, or IID whitening. It must not alter weights,
select a checkpoint, or tune a model. It must preserve the exact source hashes
and split counts and fail closed on a measure/signature mismatch.

## Hard gates and stop rules

Hard gates are source receipt validity, `theta_R4`, target signature equality,
finite tensors, and exact partition disjointness. A malformed source is a
repair trigger in a fresh output root. A valid report with poor overlap is not
a continuation veto; it selects a support/data-generation repair. A valid
report with comparable partitions but persistent residuals selects an
objective/architecture repair. Only loss of the declared theta target or
unrepairable source corruption stops the continuation.

## Pre-mortem and defaults

The 12-row validation and audit partitions may make covariance estimates noisy;
report counts and do not use them for statistical ranking. The N=256 bank was
nominated by root count, not a target-coverage theorem. All thresholds remain
diagnostic hypotheses. The output must be a unique root under
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase39-measure-separation/`.

## Execution and refresh

Implement a TensorFlow/standard-library diagnostic reporter if needed; NumPy
is permitted only at an explicit diagnostic boundary and is not required here.
Record command, environment, source hashes, git state, wall time, decision
table, inference-status table, red-team alternative, and next action. Refresh
the continuation plan after the receipt before changing the objective or
generating a new bank.

Receipt: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase39-measure-separation-attempt3/`.
The active repair is v2.2 root-group stratification; no v2.1 checkpoint or
split result is silently upgraded.
