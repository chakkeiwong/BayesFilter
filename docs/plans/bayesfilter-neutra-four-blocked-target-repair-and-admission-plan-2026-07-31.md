# NeuTra Four Blocked Target Repair and Admission Plan

Date: 2026-07-31
Status: `EXECUTED_TERMINAL_REPAIR_RESULTS_2026-07-31`

## Objective

Repair and re-evaluate the four currently blocked NeuTra targets:

- `SVX-SGQF`: no frozen SGQF level passed filter admission;
- `KSC-UKF`: the component principal-square-root UKF and Gaussian moment
  collapse failed the dense KSC reference gate;
- `SVX-ZC`: the current fixed adjacent-state wrapper is a monograph-defined
  approximation whose numerical admission was not yet established;
- `PP-ZC`: the current fixed-variant extension has implementation evidence but
  lacks a registered NeuTra target contract.

The objective is to produce target-specific admission evidence. A lane may
remain blocked if the repaired mechanism still fails its declared gate. No
NeuTra training, HMC, leaderboard, or default-readiness claim is allowed for a
lane without a terminal target-admission artifact.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can any of the four blocked target mechanisms be repaired into a valid, target-bound value/score route eligible for later NeuTra work? |
| Candidates | Higher-order exact-SV SGQF; a target-preserving KSC Gaussian-sum/UKF repair; fixed-HMC adaptations of the Zhao-Cui TTSIRT source recursion for actual SV and predator-prey. |
| Baselines | Existing blocked route and its preserved terminal artifact, plus the independent dense reference for the scalar-SV lanes. No completed model's settings are promoted as defaults. |
| Primary promotion criterion | All target-specific finite/value/score/status, reference, event-order, identity, and device gates pass. Zhao-Cui routes must also pass paper/source-anchor classification and avoid the generic retained tensor-product grid. |
| Promotion veto | Nonfinite values/scores, status failure, dense-reference disagreement, failed same-program score check, target/hash/order drift, missing route identity, failed source anchor, retained-grid fallback, missing GPU/XLA/memory-growth provenance, or output collision. |
| Continuation veto | Harness invalidity, target corruption, unavailable required source/code, exhausted bounded repair budget, or no candidate remains after the predeclared ladder. |
| Repair trigger | A localized numerical, quadrature, component-collapse, source-route, identity, serialization, or device failure under the unchanged target and budget. |
| Explanatory diagnostics | Runtime, ESS, residuals, approximation gaps, component counts, fit residuals, and observed score differences unless explicitly promoted by this plan. |
| Nonclaims | No exact likelihood claim for approximate filters, no source-faithful claim for an unassembled author route, no posterior/HMC convergence, no superiority, no cross-model transfer, and no default readiness. |

## Evidence Contract

### `SVX-SGQF`

Use the existing source-order exact transformed-SV data and independent dense
reference. Extend the SGQF ladder beyond the failed levels `(2,4,6,8)` with
predeclared levels `(10,12,16,20,24)` and retain the existing reference/order
checks. A level passes only when all status gates, prefix dense value error,
full value error, and full score error meet the frozen thresholds. The ladder
must write a fresh artifact and may issue a target identity only after an
independent posterior recomposition check.

### `KSC-UKF`

Do not relax the existing dense-reference thresholds. First run a CPU
diagnostic comparing the current component-collapse UKF against a bounded
target-preserving repair: retain a bounded Gaussian mixture through each
observation update with deterministic component pruning/merging, or reject
that hypothesis if its complexity or score is not bounded. The repaired route
must still be the KSC seven-component observation target and must report all
component/status diagnostics. Only a candidate that passes the dense value and
score gates may proceed to GPU/XLA admission.

### `SVX-ZC` and `PP-ZC`

Use the checked Zhao-Cui paper and pinned author source. Classify every route
operation as `source_faithful`, `fixed_hmc_adaptation`, or
`extension_or_invention`. A fixed-HMC adaptation may freeze author-route
randomness, ranks, schedules, and fitted tensors, but may not be called the
author's adaptive route. The repaired implementation must use source-order
TTSIRT operations, parameter/state dependence, proposal-density correction,
and no generic retained tensor-product grid. It must pass finite scalar/score,
same-program derivative, branch identity, and CPU/GPU/XLA tie-out gates. If
the assembled route remains an extension, it remains a candidate diagnostic
and does not enter the production NeuTra registry.

## Scope And Default Audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
| --- | --- | --- | --- | --- |
| SGQF levels `(10,12,16,20,24)` | Extension of the failed P2 level ladder | More points may not repair mixed-moment error or may exceed bounded runtime | CPU prefix/full ladder with dense reference | reviewed repair hypothesis |
| KSC mixture retention/merge | Target-preserving repair of the failed single-Gaussian collapse | Exponential component growth, unstable merge score, or no dense accuracy gain | T=1/T=2 component-count and score tests before T=1000 | repair hypothesis only |
| Fixed-HMC adaptation for Zhao-Cui | Paper Algorithm 3 and pinned `SIRT/TTSIRT` source | Freezing can change the adaptive author target; assembled value may remain extension | operation-classification manifest and source-order replay | candidate, not production default |
| FP64 CPU reference and FP32/XLA GPU canary | repository GPU/default policy and prior admission harnesses | device roundoff, missing memory growth, XLA drift | CPU finite/reference gates before GPU | binding execution policy |
| No transfer of prior tuning | per-scope tuning policy | inherited controls can produce false admission | scope hash and fresh calibration/validation artifacts | binding |

## Phases

1. **Inventory and source audit**
   - Preserve prior artifacts and record exact failed margins.
   - Inspect the Zhao-Cui paper technical sections and pinned author sources.
   - Confirm the four lanes are not executable in the registry until admission.
2. **SVX-SGQF repair and admission ladder**
   - Add the extended level ladder and fresh artifact root.
   - Run the independent dense reference on `/CPU:0` inside the trusted GPU/XLA
     admission harness, preserving the original device split and thresholds.
   - Issue identity only if every frozen gate passes.
3. **KSC-UKF repair feasibility**
   - Implement or reject the bounded mixture-retention repair after T=1/T=2
     mechanics checks.
   - Run the full dense-reference admission ladder only for a finite repaired
     candidate.
4. **SVX-ZC fixed-branch numerical admission**
   - Bind actual transformed-SV transition/observation/prior callbacks to the
     source-order TTSIRT compiler.
   - Freeze only declared HMC-adaptation controls and write source-classified
     route identity.
   - Run mechanics, derivative, and CPU/GPU/XLA tie-out checks.
5. **PP-ZC target-contract admission**
   - Replace the generic retained-grid route with the fixed source-order
     TTSIRT/APF branch only if its operations and target identity are explicit.
   - Preserve the extension classification for any assembled finite scalar not
     present in the author source.
   - Run mechanics and device tie-out checks; do not call this production
     admission if it remains `extension_or_invention`.
6. **Conditional NeuTra execution**
   - Promote only lanes that have terminal target admission artifacts.
   - For each promoted lane, create a new scope-specific tuning artifact using
     disjoint calibration/validation data, then run target-specific batched
     GPU/XLA training and the shared sequential HMC controller.
   - If no lane passes, stop with four terminal repair results and do not train.
7. **Terminal review and reset memo**
   - Record the decision table, inference-status table, hard vetoes, descriptive
     differences, remaining blockers, and next evidence needed.

## Compute And Attempt Budget

- Phase 1 and source audit: CPU, at most 30 minutes.
- `SVX-SGQF`: up to three CPU ladder attempts and one trusted GPU canary, at
  most 60 minutes total.
- `KSC-UKF`: one T=1/T=2 feasibility attempt, one bounded full CPU attempt,
  and one trusted GPU canary only if CPU admission passes, at most 90 minutes.
- Each Zhao-Cui lane: one mechanics implementation attempt, one CPU replay,
  and one trusted GPU/XLA canary, at most 90 minutes per lane.
- Conditional NeuTra training/HMC: zero budget until target admission passes;
  each admitted lane requires a fresh bounded plan and measured throughput.
- Every attempt writes to a fresh versioned output root and never overwrites
  historical evidence.

## Skeptical Plan Audit

- **Wrong baselines:** preserved failed routes remain baselines; no completed
  model or KSC surrogate is silently treated as truth for another lane.
- **Proxy promotion:** finite mechanics, residuals, ESS, and GPU parity do not
  by themselves establish target correctness or HMC readiness.
- **Missing stops:** every lane has a bounded ladder, fresh root, CPU-first
  gate, GPU prerequisite, and no-training boundary.
- **Unfair comparisons:** scalar-SV candidates use the same observations,
  parameter points, dense reference, event order, and score convention.
- **Hidden assumptions:** extended SGQF levels, mixture pruning, frozen
  randomness, and route classification are recorded as hypotheses with early
  diagnostics.
- **Source-faithfulness risk:** Zhao-Cui implementation evidence is separated
  from source-faithful claims; the author paper and code anchors are required.
- **Artifact mismatch:** commands emit structured result, manifest, identity,
  and source-hash artifacts; stale historical artifacts cannot satisfy current
  admission.

Audit verdict: `PASS_FOR_BOUNDED_REPAIR_AND_ADMISSION_ONLY`. No target-specific
training or HMC launch is authorized by this plan until a lane passes its
target-admission gate.

## Planned Artifact Root

```text
docs/plans/artifacts/bayesfilter-neutra-four-blocked-target-repair-20260731/
  svx-sgqf/
  ksc-ukf/
  svx-zc/
  pp-zc/
```

## Execution Closeout

The bounded repair phases are complete. The terminal result and reset point
are recorded in:

- `docs/plans/bayesfilter-neutra-four-blocked-target-repair-result-2026-07-31.md`
- `docs/plans/bayesfilter-neutra-four-blocked-target-repair-reset-memo-2026-07-31.md`

No NeuTra training or HMC was launched. `KSC-UKF` passes the CPU/reference
filter screen with the mass-preserving clustered Gaussian-sum repair, but its
trusted GPU/XLA canary remains pending because the final platform permission
reviews timed out before process creation. The other three targets remain
blocked by their unchanged numerical or target-contract gates. The historical
source-route mismatch interpretation for `SVX-ZC` and `PP-ZC` is superseded by
the owner authority in `docs/main.tex`; historical artifacts retain their
original wording but are not active veto definitions.
