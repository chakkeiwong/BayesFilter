# Phase IE3 result: linear-Gaussian recovery

## Date

2026-05-16

## Status

Exit label: `ie3_linear_gaussian_recovery_passed`.

IE3 executed the clean-room linear-Gaussian recovery fixture, produced one schema-valid `linear_gaussian_recovery` JSON row for phase `IE3`, and recorded both replicated bootstrap PF descriptive diagnostics and deterministic EDH special-case recovery against an analytic Kalman reference.

## Skeptical Audit Before Execution

Audit outcome: pass.

Checks applied before execution:

| Check | Result |
| --- | --- |
| Wrong baseline | Clear. The trusted comparator is the analytic Kalman one-step reference from the phase-owned fixture. |
| Proxy metric promoted improperly | Clear. PF particle-count comparisons are descriptive only and not treated as a promotion ranking or convergence proof. |
| Missing stop conditions | Clear. Schema failure, missing uncertainty fields, or EDH tolerance failure would have blocked the phase. |
| Unfair comparison | Clear. PF and EDH were both compared to the same analytic fixture summary. |
| Hidden assumptions | Clear after restricting EDH exactness to the linear-Gaussian special case and recording CPU-only import proof. |
| Stale context | Clear. IE0 and IE2 readiness artifacts were present and the write set stayed inside the allowed lane. |
| Environment mismatch | Clear for this bounded CPU-only run; the runner records `CUDA_VISIBLE_DEVICES=-1` before scientific imports. |
| Artifact/question mismatch | Clear. The produced JSON row, Markdown report, and manifest directly answer the IE3 recovery question. |

## Research-Intent Ledger

Main question:

- Can a clean-room one-step linear-Gaussian fixture recover the analytic Kalman posterior with deterministic EDH equality in the special case, while replicated bootstrap PF summaries remain bounded descriptive engineering evidence?

Candidate/mechanism under test:

- analytic scalar linear-Gaussian fixture;
- bootstrap/SIR particle filter with seed list `[101, 102, 103, 104, 105]` and particle ladder `[64, 256]`;
- deterministic EDH affine special-case recovery.

Promotion criterion:

- the JSON row is schema-valid, the CPU-only manifest proof is present, and EDH posterior mean/variance recovery matches the analytic reference within stated deterministic tolerances.

Promotion veto:

- missing analytic comparator, missing PF uncertainty fields, missing seed policy, missing CPU-only proof, or schema-invalid artifact.

Continuation veto:

- the IE2 schema cannot represent PF uncertainty or EDH deterministic tolerance fields.

Repair trigger:

- non-finite analytic quantities, PF summaries with missing replication accounting, or EDH absolute-tolerance failure.

What must not be concluded:

- no production `bayesfilter/` validation;
- no nonlinear EDH exactness claim;
- no statistical ranking or asymptotic PF convergence claim;
- no DPF-HMC, DSGE, MacroFinance, banking, or production-readiness claim.

## Artifacts Produced

- JSON result: `experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json`;
- Markdown report: `experiments/dpf_monograph_evidence/reports/linear-gaussian-recovery-result.md`.

Codex audit adjustment:

- The first IE3 executor output used `not_source_dependent` for the row-level
  source-support class.  Codex corrected the runner and regenerated the JSON so
  the row now uses `bibliography_spine_only`, matching IE1 because the
  diagnostic is tied to classical SMC and EDH source families plus local fixture
  reasoning.

## Key Observations

Analytic fixture values:

- predictive mean: `0.315000000000`;
- predictive variance: `1.291000000000`;
- posterior mean: `-0.037630822312`;
- posterior variance: `0.332586543991`;
- one-step log likelihood: `-1.483609255109`.

Bootstrap PF descriptive summaries:

| Particles | Mean abs. error | Variance abs. error | Log-likelihood abs. error | MCSE status |
| --- | ---: | ---: | ---: | --- |
| 64 | 0.044946453465 | 0.066146906800 | 0.111824277715 | reported across 5 seeds |
| 256 | 0.021578810033 | 0.020995960579 | 0.035728463261 | reported across 5 seeds |

Interpretation:

- the 256-particle arm is descriptively closer to the analytic reference than the 64-particle arm on the recorded summaries;
- this supports a bounded engineering sanity check only;
- the result does not support a statistical ranking or convergence-rate claim.

EDH special-case recovery:

- posterior mean absolute error: `3.469446951953614e-17`;
- posterior variance absolute error: `0.0`.

Interpretation:

- this is deterministic special-case algebraic evidence for the linear-Gaussian fixture only.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | Pass |
| Primary criterion status | Passed |
| Promotion veto status | Not triggered |
| Continuation veto status | Not triggered |
| Main uncertainty | PF summaries use five fixed seeds and are descriptive only; no uncertainty-backed ranking is claimed. |
| Next justified action | Proceed to IE4 only if the same clean-room schema, manifest, and evidence-discipline rules are preserved. |
| What is not concluded | No production, nonlinear, or bank-facing claim follows from IE3. |

## Inference-Status Table

| Row | Status |
| --- | --- |
| Hard veto screen | Passed for this fixture: schema valid, analytic comparator present, CPU-only proof present, EDH tolerance passed. |
| Statistically supported ranking | Not established. |
| Descriptive-only differences | PF 256-particle summaries are descriptively better than PF 64-particle summaries. |
| Default-readiness | Not applicable. IE3 does not set any production default. |
| Next evidence needed | Additional controlled phases must validate their own fixtures and keep uncertainty discipline before any broader interpretation. |

## Run Manifest

- git commit: `0684d6fe4350664c34826e83df59b2413d7be89c`;
- branch: `main`;
- command: `python -m experiments.dpf_monograph_evidence.runners.run_linear_gaussian_recovery`;
- environment: Python `3.13.13`, NumPy `2.1.3`;
- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES=-1`, GPU hidden before scientific imports;
- seeds: `[101, 102, 103, 104, 105]`;
- wall-clock cap: `30` seconds;
- output artifacts:
  - `experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json`;
  - `experiments/dpf_monograph_evidence/reports/linear-gaussian-recovery-result.md`.

Codex verification:

- `python -m py_compile experiments/dpf_monograph_evidence/fixtures/linear_gaussian.py experiments/dpf_monograph_evidence/runners/run_linear_gaussian_recovery.py` passed;
- `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_monograph_evidence.runners.run_linear_gaussian_recovery` passed;
- `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_monograph_evidence.runners.validate_results experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json` passed.

## Post-Run Red-Team Note

Strongest alternative explanation:

- the EDH equality result could reflect only the chosen closed-form linear-Gaussian algebra rather than a more general implementation property.

What would overturn the current conclusion:

- a schema-valid rerun showing broken CPU-only manifest proof, missing uncertainty accounting, or EDH tolerance failure.

Weakest part of the evidence:

- PF comparison evidence is intentionally short and descriptive, so it cannot defend a stochastic ranking beyond this bounded sanity check.
