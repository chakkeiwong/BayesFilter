# SSL-LSTM NeuTra/HMC Unified Pipeline Plan

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `COMPLETED_WRAPPER_IMPLEMENTATION`

## Objective

Provide one repeatable Python entry point for the existing q-specific pipeline:

```text
NeuTra training (two streams)
        -> transformed-target HMC preflight/tuning
        -> retained four-chain HMC with sequential checkpoints
```

The orchestrator must preserve the current stage boundaries and must not make a
vetoed training transport eligible for HMC by copying or rewriting artifacts.

## Research Question And Evidence Contract

The wrapper answers an engineering question: can a declared q, batch size,
parameter file, and set of stage budgets execute reproducibly through the
existing training/HMC artifact contracts? It does not establish posterior
correctness, convergence beyond the retained-HMC screen, predictive
equivalence, or scientific validity.

Exact comparator: the existing three stage scripts and their current source
contracts. The wrapper adds no numerical estimator and no new promotion metric.

Promotion/continuation gates:

- training must return two result receipts with status `ADMITTED`;
- HMC tuning must return `KERNELS_FROZEN`;
- retained HMC is launched only after the preceding gate passes;
- nonzero child exit, missing/malformed summary, resource stop, or hard veto
  stops the wrapper and writes a structured pipeline summary.

Explanatory diagnostics: stage wall times, child commands, logs, git/device
metadata, underlying stage statuses, and artifact paths.

Nonclaims: the wrapper does not rank batch sizes, prove predictive equivalence,
or turn one-seed/short-chain outcomes into convergence or posterior claims.

## Interface

```text
python docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py \
  --q {1|2|5|10|20} --batch-size 100 \
  --params-json <parameter-file> \
  --output-root <fresh-run-root> \
  --training-cap-seconds <cap> \
  --hmc-tuning-cap-seconds <cap> \
  --retained-hmc-cap-seconds <cap> \
  --authorize-material-run
```

`--resume` reuses only the child scripts' own resumable artifacts. A fresh
output root is required otherwise. `--mode contract-smoke` constructs no GPU,
target, HMC, or child process and validates the command contract only.

## Skeptical Audit

- Wrong baseline: no new baseline; each child is the current repository script.
- Proxy promotion: training loss and saturation remain child diagnostics; only
  the existing `ADMITTED`/`KERNELS_FROZEN` gates control handoff.
- Missing stop: child caps plus nonzero/malformed-artifact stops are explicit.
- Unfair comparison: q, batch size, params, seeds, and stage commands are
  recorded in the manifest; the wrapper changes no child defaults.
- Hidden environment: child stdout/stderr, Python executable, git state, and
  stage paths are recorded.
- Boundary safety: the wrapper never edits or synthesizes a Phase 3/4 receipt.

Audit decision: `PASS_FOR_WRAPPER_IMPLEMENTATION`; no material GPU execution
is part of this implementation step.

## Artifacts

- `pipeline-summary.json`: final wrapper status, stage records, commands,
  source state, and handoff decision;
- `logs/*.log`: live child output for each launched stage;
- child-owned training, tuning, and retained-HMC artifacts under the output
  root's stage directories.

## Checks

- contract-smoke mode does not launch children;
- invalid/nonpositive caps and missing parameter/output paths fail closed;
- a synthetic stage-result fixture verifies that vetoed training prevents HMC;
- Python compilation, focused wrapper tests, and `git diff --check` pass.

## Close Record

Implemented:
`docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py`.

Checks:

- contract smoke: passed;
- focused wrapper tests: `4 passed`;
- Python compilation: passed;
- `git diff --check`: passed;
- material GPU/HMC execution: not launched.

The wrapper is intentionally a subprocess orchestrator rather than a numerical
monolith. This gives one repeatable command while preserving the existing
training, HMC-tuning, and retained-HMC source/artifact boundaries.
