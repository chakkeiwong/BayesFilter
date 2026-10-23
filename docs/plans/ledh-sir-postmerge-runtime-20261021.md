# Diagnose the SIR merge slowdown

**DISQUALIFIED, 23 October 2026:** the owner rejects the N64 workload. Earlier
measurements and conclusions cannot be used as evidence or baselines. See the
[N1008 replacement](ledh-sir-n1008-runtime-20261023.md).

Completed21 October2026. The predicted QR dependency is confirmed in optimized
HLO; eight alternating blocks completed with exact same-source tie-outs and no
invalid results. Report:
`artifacts/ledh-sir-postmerge-runtime-20261021/attempt-01/result.md`.
Charge412.984804 GPU-process seconds;14657.101235 remain in the parent budget.

Active question: why did the frozen replay become slower, particularly plain
LEDH and the conditional Gaussian endpoint? This is an engineering diagnostic
on exposed inputs, with no guide training, tuning, or scientific promotion.

The comparator is the actual `before-source` snapshot in
`artifacts/ledh-sir-postmerge-replay-20261021/attempt-01`; the candidate is its
`merged-source` snapshot. Keep the original controls, input generation, seed,
FP64/TF32-off GPU/XLA route, analytical derivative, memory growth, and all
validity monitors. Do not change OT `(1,5)` or the algorithms to obtain timings.

## Evidence contract and intent

- Mechanism under test: newly consumed flow validity flags retain the per-step
  QR determinant calculation even for endpoints whose density does not use that
  determinant. Previously XLA could eliminate the unused calculation. The
  physical mixture at beta=.9 already needs it and is the mechanistic control.
- Primary diagnostic: inspect optimized HLO from the actual consumer endpoints
  for live flow QR operations, then measure unchanged endpoints in alternating
  before/after blocks after compilation. HLO dependency evidence establishes
  retained work; timing alone cannot assign all overhead to one code change.
- Baselines/controls: plain beta=0, conditional beta=1, physical beta=.9. These
  are the existing three simple endpoints, not a new learned-method comparison.
  Evaluate each separately; do not pool them into a ranking.
- Validity veto: import/source/input mismatch, any invalid/nonfinite result,
  missing diagnostic, unexpected retracing, or disagreement with the archived
  same-source output. These stop interpretation, not justify deleting checks.
- Repair trigger: a localized diagnostic harness or serialization failure can
  be repaired within budget. Candidate slowdown is not a continuation veto.
- Explanatory only: device/CPU activity, endpoint serialization time, graph size,
  compile time, and warm runtime distribution. Timing intervals from one exposed
  dataset do not establish a general SIR speed ratio or scientific superiority.
- No conclusion about score accuracy, canonical admission, default readiness,
  guide efficacy, or numerical-parity repair is authorized by this diagnostic.

## Defaults, audit, and execution

Use dataset 263043001, replicate 0 (particle seed 283030000), reset seed
263040002. These reproduce an existing replay input; they are convenience
choices for compiler/runtime diagnosis, not representative performance samples.
Verify same-source loss/value/moment agreement before interpreting timing.
Measure zero direction (the main replay workload) and direction 0 (nonzero
analytical-score control) without recompiling. Both are dynamic inputs to the
same stable-signature compiled endpoint. Capture the original trace outputs.

Warm all three endpoints in two isolated, persistent worker processes, one per
source. Request runs serially in eight paired blocks, alternating which source
goes first. Each block runs all three endpoints, in rotating arm order, once
per direction. Synchronize by materializing every returned tensor. Preserve the
full result and separate execution/materialization from Python serialization.
Keep every raw timing, cold duration, manifest, HLO, source hash and log.

Premortem: sequential load can exaggerate speed differences; alternating blocks
and the physical-mixture control reduce that confounding. The second resident
process may affect absolute timings; no simultaneous kernels are requested.
An unused graph operation is not runtime evidence: use optimized HLO, not the
unoptimized TensorFlow graph. Guards may be the reason work survives; do not
disable them and label that as a valid speed fix.

Skeptical audit passed 21 October 2026: actual dirty snapshots are the baseline;
same-source output checks precede timing interpretation; compiled consumer
dependencies address the proposed cause; zero and nonzero directions are
distinguished; all numerical settings and safeguards are fixed. No proxy is a
scientific promotion criterion. No new hardware or external service is used.

Budget: at most 1,800 GPU-process seconds including initialization and any
localized retry, charged to the previously authorized remaining 15,070.086040
seconds. Stop on exhaustion, invalid evidence, or a source mismatch. Each launch
uses a new directory under `artifacts/ledh-sir-postmerge-runtime-20261021/`.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/diagnose_sir_postmerge_runtime.py --repository <this-worktree>
--output <unique-attempt>`. The supervisor records exact worker commands and
environment. GPU access is escalated; memory growth is verified before GPU
initialization. Command buffers retain the replay's explicit WSL exception.
Result: `<unique-attempt>/result.json`, plus a terminal `result.md` with decision
and inference-status tables. Update the concise active checkpoint afterward.
