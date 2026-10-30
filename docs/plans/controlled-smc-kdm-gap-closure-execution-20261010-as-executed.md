# Controlled SMC and KDM replication gap closure
Updated 10 October 2026. Authoritative parent:
`controlled-smc-kdm-replication-master-program-20260928.md`.

## Question and evidence contract
Replication only: numerical behavior no materially worse than the checked
author source under matched data/protocol/evaluation. No Task3/E0, author
algorithm repair, or default promotion. Source-intrinsic failures and tails
remain acceptable. cSMC comparators: Heng commit2696f6d611aae172d10102cb7bfbfd314bb6b88f
via attempt139 source lock. KDM: Younis b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18
via reviewed complete-training artifact.

Primary cSMC criteria: zero extra TF invalid on source-valid pairs; Q2 mean
likelihood1% / variance5% margins with supported uncertainty and the existing
author-tail interpretation. Inadequate variance precision is inconclusive,
not evidence of degradation. KDM: matched training/evaluation, existing
0.01-nat NLL / 1% RMSE margins with uncertainty across independent training units.
Small exact replays and single-unit point estimates cannot qualify replication.

| Diagnostic | Role / action |
|---|---|
| Corrupt tape, unequal source controls, missing artifact | Continuation veto: repair harness |
| Budget exhaustion or unmatched protocol | Continuation veto: stop allocation |
| Extra TF invalid on source-valid | Promotion veto and repair trigger; continue localization |
| Component/gradient/Adam/ancestry mismatch | Repair trigger / explanatory, not direction rejection |
| Source-only or shared failure/extreme tail | Explanatory; preserve and classify |
| Retracing/device fallback/memory growth absent | Engineering veto for that execution |
| Supported bound beyond margin | Candidate qualification failure |
| Inadequate precision | Inconclusive; assess feasibility |

No bitwise trajectory prerequisite, superiority, HMC, new default, or author
correctness conclusion. The heuristic-dominance gate for improvement claims is
outside this author-only replication comparison; no improvement is claimed.

## Budget, artifacts and stop conditions
Reserve **1200 CPU / 3600 GPU wall seconds**, maximum **two supervised attempts**,
within the existing campaign balance (last recorded about220884 CPU /
118257 GPU seconds; verify before launch). Already executed CPU C0 work
25.767 seconds counts. Failed attempts count. Children use remaining-time limits;
terminate only supervisor-owned process groups.

Unique root:
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/`.
Manifests preserve commit, actual command, environment, CPU/GPU, memory growth,
XLA/TF32, seeds, source/data/fixture hashes, wall time, logs, plan/result paths.
Never overwrite prior evidence. Terminal charges enter reference-diagnostics.
This is a localization reserve, not authorization for unspecified long training.

## Assumption audit
| Choice / provenance | Risk and early check / status |
|---|---|
| cSMC FP64, TF32 off: existing reference lane | Not TF32 production evidence; source-policy forward control |
| Centered QR: existing Lorenz port | Coefficients may differ; fit-policy/downstream comparison; frozen candidate |
| Source-normal-equations diagnostic | Octave backslash conditionally changes solver; 14/101 singular branches; not automatic exact author port |
| KDM four updates, batch64,T17,N25: source training shapes | Small diagnostic only; native/replay controls with evolving gradients/moments |
| Fresh Adam, fixed source learning rates | Not mature checkpoint/scheduler certification; record zero-state identity |
| Shared realized sample/initial tensors plus proposal noise | Isolates deterministic learning, not autonomous random-transform parity; exact source replay |
| Consecutive development rows384:640 | Avoid constant gradients; never qualification data; hash/mask/shape controls |
| Existing FP32/FP64 replay tolerances | Diagnostic only; preserve full errors and first failing update; no tuning to pass |
| Host case/update loops | Orchestration; TF update batch-native XLA graph, trace1 and GPU checks |

## Skeptical review and pre-mortem
Corrected three material draft defects before execution: source capture is CPU;
candidate failures are repair triggers, not automatic continuation stops;
Octave regression is conditional backslash, not just raw solve.
No wrong comparator/proxy promotion remains in the bounded executable stage.
A misleading PASS would promote deterministic replay as downstream equivalence:
prevent through K1. A misleading FAIL would call chaotic ancestry or author
tails defects: preserve controls and agreed margins. Near-zero angular
normalization may explain KDM gradients before Adam; inspect earliest operation.
Larger phases need a concrete statistical design, never guessed long runs.

## C0: component localization — execute
CPU fit trace and six-case author capture completed; all22 stages have unchanged
results and final RNG. Run GPU/XLA fit-to-forward diagnostic, source-policy
controls, both explicit solver arms, policy values, log weights, ESS, ancestors
and first differing times. Persist computed conditioning diagnostics.
Execution PASS does not mean solver-specific screen passed. A raw
normal-equation rejection does not reject centered QR.

## K0: coupled evolving updates — execute
Map actual TF-trained development checkpoints back to the source for each
MDPF/A-MDPF variant with exact values/shapes. Four consecutive development
batches in FP32/FP64, retaining evolving parameters and Adam moments.
Native/replayed source FP32 sequence must match exactly at every update.
TF consumes identical batches, realized sample tape and proposal noise.
Save losses, outputs, raw/clipped gradients, norm, parameters and moments;
report first failing check/update. Fresh-state coupled updates only:
mature optimizer state, scheduler/pretraining remain separate tests if needed.
No numerical protection or optimizer precision is silently changed.

## C1/C2/K1: conditional continuation
C1 targets F2.5 and four variance-ratio cells labeled4.346154,5.576923,
5.730769,7.115385; preserve exact forcing from saved inventory before capture,
seeds202610101:202610103. Record first-divergence and conditioning, weights,
ESS and ancestors. Add sampling/resampling arms only if C0 localizes there,
one mechanism at a time. Candidate failure triggers this repair, not abandonment.
C2 assesses heavy-tail precision feasibility first; require untouched-seed
sample size and uncertainty subplan before qualification.
K1 requires at least three independent matched training units, exact source
schedule/evaluation, frozen seeds/batch order, commands, cost estimate and
disjoint final evaluation. If under-budgeted, report it; never promote one unit.
These phases are conditional on localization and a concrete executable design.

## Exact command and approval scope
```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01
```
Run this literal supervisor with escalated GPU access. It performs GPU preflight,
CPU-source and GPU/XLA children, unique outputs, caps and logs. CPU children hide
GPU before import. Add only this narrow script prefix to the three existing
diagnostic rules. Rules cannot override managed permission or guarantee reload.
Terminal note: decision and inference-status tables, strongest alternative
explanation, actual results; refresh master and concise checkpoint.
