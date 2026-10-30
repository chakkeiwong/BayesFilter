# M13 exact-input correction localization — 2026-09-24

This is the next repair diagnostic in the existing surrogate-HMC closure
master, not a new scientific direction. The campaign question remains whether
the canonical analytical DCTR force can drive useful surrogate-HMC proposals
with a separately declared endpoint posterior. No sampler runs in this stage.

Controls-07 has two nominal calibration rows failing correction at time zero.
The saved source closure still matches current code. Unmetered quick probes
used partition zero instead of saved partition six and are excluded from
claims about that failure. Controls-08-recovery records a conservative reserve
of 1,000 CPU and 420 GPU seconds for those probes; those are bookkeeping
charges, not measured usage. Controls pool remaining before this stage:
352.445851 CPU and 2,503.083982 GPU seconds. Global remaining:
8,977.436151 CPU and 191,695.410031 GPU seconds.

## Evidence contract and skeptical audit

Question: does changing arithmetic precision remove the saved correction
failure while preserving the exact realized inputs and canonical call chain?
The baseline is controls-07 calibration arm zero: attempt three, seed 1703,
design partition six, N=256, T=20, nine nominal parameter points. Read the
saved points and controls, reconstruct the deterministic fixture, and verify
its data, settings, source closure and input hashes before evaluation. The
claim partition (3701) remains reserved.

Run the same canonical batch value/analytical-score implementation in FP32
with TF32, FP32 without TF32, and FP64 with TF32 disabled. FP64 inputs and
coefficients must be exact lifts of the FP32 inputs, including initial, process
and observation covariances. Regenerating an FP64 target directly would change
some coefficients and is therefore the wrong comparator for this question.
Keep all reset, correction, caps, LM, chunk and observation-law settings fixed.

| Role | Criterion |
| --- | --- |
| Diagnostic completion | All three arms preserve provenance and emit all per-row numerical statuses, moments and correction diagnostics. |
| Continuation veto | Source/input mismatch, unflagged nonfinite value/force/moments, wrong device or resource limit. |
| Repair trigger | Typed correction failure; finite outputs with implausible standardized moments; precision-dependent validity. |
| Explanatory only | Numerical differences, ESS, condition estimates, feasibility margins and elapsed time. |
| Promotion | None: this diagnostic selects no controls, endpoint, precision default or HMC configuration. |

The cheap comparator set is the unchanged default arithmetic, no-TF32
arithmetic, and an FP64 reference, evaluated separately on previously failed
and previously healthy rows. These are arithmetic comparators, not a sampler
performance comparison. The initial two low-ESS rows motivate a hypothesis
of weighted-moment overflow after covariance collapse. A healthy FP64 result
would support precision sensitivity, not prove that hypothesis or repair the
geometry. Capture the full diagnostics before proposing a mathematical fix.
Saved-run numerical deviations are warnings; a different realized input is
an invalid comparison, not a numerical tolerance discrepancy.

| Choice | Provenance and reason | Failure mode and early check | Status |
| --- | --- | --- | --- |
| FP32/TF32/XLA | Owner default; existing canonical consumer | Arithmetic overflow; typed statuses and FP64 comparison | Reviewed execution default |
| Nine nominal rows | Saved controls-07 failures and healthy comparators | Stress/tails untested; make no scope-wide claim | Diagnostic sample |
| Controls and designs | Saved calibration arm zero | Regeneration drift; exact hashes before tracing | Frozen comparator |
| FP64 lift | Local exact-representability argument | Recomputed coefficients change problem; tensor equality tests | Reference only |

Pre-mortem: a run could pass because it silently uses another partition, drops
correction diagnostics, or changes covariances. Provenance guards and focused
tests address these risks. It could fail because of compiler or resource costs;
retain each completed arm separately and classify a resource stop accordingly.
Skeptical audit passes for this diagnostic: no proxy is a promotion criterion,
comparators share realized inputs, the source closure is unchanged, and the
result can distinguish arithmetic sensitivity from an invariant failure.

## Execution and artifacts

One attempt. Master review cap: 50 CPU seconds / 60 wall seconds. Worker cap:
240 CPU seconds / 420 GPU wall seconds; total reservation 290 CPU seconds.
This fits the existing controls and global budgets. Use the approved tftwogpu
interpreter, trusted GPU access, verified memory growth, and XLA. Master review
is CPU-only with devices intentionally hidden. No external review is required.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-correction-localize --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-09
```

The master preserves source snapshots, run manifest, command logs, receipts
and accounting. Each arithmetic arm writes its complete diagnostics before
the next arm begins. The report records exact saved-input binding and per-row
comparison. The result note must distinguish engineering completion, numerical
validity, and the still-unrun surrogate-HMC test. A typed candidate failure
triggers the next bounded repair in the master; it does not reject the research
direction or justify stopping while an authorized repair fits the budget.
