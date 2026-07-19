# SSL-LSTM NeuTra Phase 1 No-Oracle Design Result

Date: 2026-07-14

Status: `NO_ORACLE_DESIGN_VALID`

## Objective And Evidence Contract

Question: without a trustworthy SSL-LSTM posterior oracle, can exact
implementation invariants, prospectively independent NeuTra/HMC replications,
sensitivity probes, frozen predictive comparisons, and repeated synthetic
calibration provide bounded evidence that can falsify an invalid candidate?

Primary pass: the design has disjoint evidence roles, can detect seeded
implementation and sampler/predictive failures on known controls, and never
requires an approximate SSL-LSTM method to be called truth.

Vetoes: seed overlap, shared mutable transport state, circular tolerance
selection, an approximate method labeled an oracle, missing independent
replication, or an unfrozen claim ceiling.

Explanatory only: cross-replication parameter differences, loss, acceptance,
runtime, local Hessians, high moments, quantiles, and sensitivity-probe
differences unless a later prospective phase promotes a named quantity.

Nonclaim: a pass cannot establish absolute posterior correctness, complete
mode/tail coverage, NeuTra superiority, or application validity.

Preservation: this document is the Phase 1 design/result artifact. Each later
run must record its realized role code, full stateless seed, parent artifact,
target signature, and non-overwriting output path.

## Skeptical Audit

| Challenge | Resolution |
| --- | --- |
| Two wrong samplers agree | Agreement is only cross-replication stability. Dispersed starts, prior/shell/tail probes, analytic controls, and repeated generative calibration remain separate falsification routes. |
| Training seed becomes validation seed | Role codes below are disjoint and checked before execution. |
| Loss becomes promotion criterion | Loss is explanatory; downstream exact transformed-HMC admission is required. |
| Ordinary HMC becomes truth | It is a comparator only if independently admitted under the same sampler gates. |
| Analytic fixture is overgeneralized | Gaussian, curved-ridge, and LGSSM controls validate code only on their own known problems. |
| Predictive agreement hides posterior error | The claim ceiling forbids posterior correctness and complete mode/tail coverage. |
| Tolerances fitted to outcomes | Phase 8 freezes tolerances from controls before Phase 9 confirmation outputs are opened. |

Audit decision: `PASS_NO_ORACLE_DESIGN`. The design answers the stated bounded
question and preserves the absence of posterior truth.

## Seed And Namespace Contract

Every TensorFlow stateless seed is the pair `[20260714, role_code]`. Replicate
and rung offsets are added only inside the reserved interval shown. No seed may
cross a row, even when an earlier run fails.

| Role | Reserved role-code interval | Initial assignments |
| --- | --- | --- |
| Analytic-control generation | `1100..1199` | Gaussian `1101`; correlated Gaussian `1102`; curved ridge `1103`; seeded-defect controls `1110..1119` |
| Transport initialization/training | `2100..2199` | independent replications A/B `2101`, `2102`; repair replications begin `2110` |
| Training validation base noise | `2200..2299` | A/B `2201`, `2202`; never optimizer batches |
| Transformed-HMC tuning | `3100..3199` | A/B `3101`, `3102`; tuning rungs use offsets `+10..+19` |
| Retained NeuTra-HMC sampling | `3200..3299` | A/B `3201`, `3202`; never tuning transitions |
| Dispersed prior/shell/tail probes | `3300..3399` | prior `3301`; shell `3310..3317`; tail `3320..3327` |
| Predictive-design calibration | `4100..4199` | analytic/null/alternative banks `4101..4149` |
| Blinded predictive confirmation | `4200..4299` | independent arms A/B `4201`, `4202`; bootstrap `4250..4279` |
| Fresh-seed audit | `4300..4399` | retraining A/B `4301`, `4302`; HMC `4321`, `4322`; forecasts `4341`, `4342` |
| Synthetic dataset generation | `5100..5199` | dataset replicate `i` uses `5100+i`, with the final count fixed before execution |
| Synthetic fit/sampling/forecast | `5200..5599` | disjoint subranges fixed in the Phase 11 plan; never reuse dataset-generation seeds |
| Application evaluation | `6100..6199` | rolling origins and forecast banks fixed in Phase 13 |

The four historical A4 starts remain deterministic named inputs, not random
seed outcomes. Later additional starts are generated only in the `3300` probe
namespace and cannot replace an unfavorable historical start.

## Independence Contract

Replications A and B must have separate:

- trainable variables, optimizers, checkpoints, training noise, and validation
  noise;
- frozen payloads, topology/tensor hashes, HMC tuning seeds, retained seeds,
  and final states;
- forecast innovation banks and bootstrap clusters; and
- artifact directories and failure records.

They may share only the locked target, mathematical implementation, prospective
configuration family, analytic-control definitions, and decision thresholds.
Choosing B after seeing A is forbidden; both initial configurations are frozen
before material A training begins.

## Sensitivity And Falsification Probes

| Probe | Construction | Role |
| --- | --- | --- |
| Original four starts | Exact A4 fixed dispersed starts | Required chain/start sensitivity; never delete or replace a failed start |
| Prior probes | Stateless draws from the locked prior convention | Search for separated attraction regions; explanatory/repair trigger |
| Shell probes | Center plus/minus fixed coordinate and principal-direction radii chosen before training | Expose local ridge and transport-coverage failures |
| Tail probes | Larger fixed radii with finite target/value preflight | Expose saturation, logdet, nonfinite, or tail-undercoverage failures |
| Transport replay | Map the same frozen probes through forward/inverse and transformed score | Exact engineering veto |
| Cross-replication forecasts | Same frozen forecast statistic with independent innovation banks | Stability/equivalence evidence under later prospective criteria |

Probe radii and the prior draw count must be frozen in Phase 4 before training
outcomes are inspected. A probe can veto engineering validity or trigger a
coverage repair; it cannot prove exhaustive mode coverage.

## Known-Problem Analytic Controls

1. A diagonal Gaussian detects objective sign, shift, scale, batch reduction,
   and affine pullback errors.
2. A correlated non-diagonal Gaussian detects `J` versus `J^T`, covariance,
   and mixing-linear errors.
3. A normalized curved-ridge target with an explicitly evaluable log density
   and score detects failures hidden by affine controls. Numerical integration
   or direct sampling on this separate target may check its own fixture only.
4. The existing scalar LGSSM forecast oracle checks forecast propagation and
   predictive statistics on the LGSSM, never the SSL-LSTM posterior.
5. Seeded defects include reversed logdet sign, reversed component order,
   transposed pullback, shared-seed leakage, duplicated chains, and widened
   post-hoc tolerances; every defect must fail its intended gate.

Phase 2 owns transport controls 1-3 and the logdet/order/pullback defects. Phase
3 owns trainer objective/reduction controls. Phases 7-9 own duplicated-chain,
seed-leakage, and tolerance-mutation controls.

## Comparison And Claim Ceiling

| Evidence | Permitted conclusion | Forbidden conclusion |
| --- | --- | --- |
| Exact target/transport invariants pass | Implemented value/score/change-of-variable contract passes tested fixtures | Posterior or sampler correctness |
| Each NeuTra replication passes sampler gates | Each chain set is admitted under prospective diagnostics | Complete convergence or mode coverage |
| A and B agree under prospective uncertainty | Independent replications are stable for named functionals/predictive laws | Posterior truth |
| Predictive confirmation passes | Named one-to-ten-step predictive laws meet frozen practical-equivalence criteria | Parameter equality or posterior equality |
| Repeated synthetic calibration passes | In-class uncertainty is calibrated for the tested generator/configuration | Real-data adequacy or general validity |
| Ordinary HMC is admitted and agrees | An independently admitted classical comparator is consistent on named quantities | Either method is an oracle or superior |

## Result And Handoff

| Decision | Primary status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| `NO_ORACLE_DESIGN_VALID` | Role separation, independence, falsification probes, controls, and claim ceiling are frozen | No circularity, seed-overlap, fictitious-oracle, or missing-replication veto found | Two wrong methods can still agree outside tested probes; this is retained as the evidence ceiling | Implement Phase 2 dense-IAF mathematical closure | Absolute posterior correctness, complete mode/tail coverage, scientific validity, or readiness |

No sampler, GPU job, training run, forecast comparison, or posterior-reference
calculation was executed in Phase 1.
