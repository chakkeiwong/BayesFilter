# SSL-LSTM NeuTra Capacity Repair Plan

Date: 2026-07-15

Status: `R2_CAPACITY_ONLY_NOT_NOMINATED_CAPACITY_PLUS_SCHEDULE_REPAIR_TRIGGERED`

Implementation update, 2026-07-15: the separate `(32,32)` capacity family,
strict frozen-artifact label, paired diagnostic harness, and focused tests are
implemented. The strict `(4,4)` source-procedure preset remains unchanged.
After repairing six native-review findings, `62` focused tests, compilation,
and `git diff --check` pass. The durable review is
`docs/reviews/bayesfilter-ssl-lstm-neutra-capacity-32x32-native-review-2026-07-15.md`.

R2 result update, 2026-07-15: both historical streams crossed the `0.05`
saturation cap at step 100 under the inherited `0.01` initial learning rate
(`0.21745` for A and `0.11719` for B). The valid GPU/XLA diagnostic stopped
after `1,224.85` seconds. Capacity expansion alone is not nominated. This
triggers a prospective `(32,32)` plus lower-initial-rate repair; it does not
justify reverting to `(4,4)`, running full confirmation, or rejecting NeuTra.
See `bayesfilter-ssl-lstm-neutra-capacity-32x32-diagnostic-result-2026-07-15.md`.

This revision supersedes the reviewed learning-rate-only repair. The owner
identified the material design error: translating Rotemberg's width-equals-
dimension launcher convention into `(4,4)` preserved operator structure but
collapsed representational capacity. The prior Claude review remains valid
only for the superseded schedule plan and does not approve this revision.

## Research Intent And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Main question | Does expanding each of the three SSL-LSTM IAF conditioners from `(4,4)` to `(32,32)` repair seed-B saturation and moderate-shell support under otherwise unchanged training? |
| Exact baseline | Immutable completed `(4,4)` source-procedure A/B results: seed A SHA-256 `6b0b5ff525e9081870b707784715b6e31e1c8f47d8fec59ff7b96a1bc7bc8186`; seed B SHA-256 `3cfd5f1d936c99d1f42e4d7f5b4900da9d49403bcac902f55e912ac7b04ab40c`. Seed B is the failure-bearing paired comparator; seed A checks that the repair does not merely exchange which stream fails. |
| Candidate mechanism | Three dense autoregressive IAF stages with `(32,32)` ELU hidden layers, the same MADE degree masks, reverse-coordinate mixing, fixed prior-center translation, `s_max=1`, initialization scale `0.02`, standard-normal base, reverse KL, batch 480, 5,000-step paper schedule `0.01/0.001/0.0001`, Adam epsilon `1e-7`, and per-variable clipping at 10. Width is the only intended algorithmic change. |
| Classification | `ssl_lstm_capacity_dense_iaf`, a BayesFilter capacity adaptation of the Rotemberg/SGU operator structure. It must not be serialized or described as exact source-procedure parity. |
| Primary R2 diagnostic criterion | On exact historical A/B streams through step 1,200: every executed step is finite; saturation is at most `0.05` at every 100-step checkpoint; and at terminal step 1,200 both shell radius is at most `4.30` and the paired heldout one-sided 95% upper bound is below zero. Passing nominates fresh confirmation only. |
| Primary full-confirmation criterion | Two fresh independent 5,000-step seeds both pass the original material gates unchanged. This is outside the initial R2 execution boundary and requires a refreshed timing/resource authorization. |
| Hard veto | Strict source preset is changed; width adaptation is mislabeled source parity; target/chart/mask/stage/mixing/translation/objective/schedule drift; nonfinite state; resume/reload mismatch; corrupt artifact; CPU fallback; missing XLA evidence; or resource overrun. |
| R2 repair-nomination veto | Either historical stream exceeds saturation `0.05` at a 100-step checkpoint, or fails terminal shell/heldout gates. This stops before fresh confirmation but does not reject NeuTra generally. |
| Explanatory only | Loss, stage saturation, shell trajectory, gradient/clipping trajectory, parameter count, runtime, and all continuous `(4,4)` versus `(32,32)` differences. |
| Nonclaims | No retrospective checkpoint selection; no seed-A-only promotion; no posterior correctness, HMC readiness, predictive equivalence, superiority, default readiness, exact paper/source fidelity, or general NeuTra conclusion. |
| R2 artifact root | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/capacity-32x32-diagnostic/` |
| Result note | `docs/plans/bayesfilter-ssl-lstm-neutra-capacity-32x32-diagnostic-result-2026-07-15.md` |

## Skeptical Pre-Execution Audit

Audit decision: `PASS_CAPACITY_ISOLATION_AFTER_SUPERSEDING_LR_ONLY_REPAIR`.

| Challenge | Resolution |
| --- | --- |
| Wrong baseline | Compare against immutable completed `(4,4)` A/B artifacts, not the earlier one-stage ablation. |
| Hidden confounding | Preserve original learning rate, objective, batch, steps, stage count, masks, mixing, scale bound, initialization scale, seeds, target, and gates; change hidden widths only. |
| Proxy promoted | R2 can nominate full confirmation only. It cannot produce a main-lane transport or support HMC claims. |
| Post-hoc stream selection | Both historical A and B streams run. Earlier checkpoints cannot be selected, frozen, or promoted. |
| Missing stop | Finiteness is checked continuously; saturation at every 100 steps; shell and heldout at terminal step 1,200; any R2 veto stops before full confirmation. |
| Stale review | The prior `AGREE` applied to a learning-rate-only design and is explicitly superseded. This capacity revision gets a focused native implementation/plan audit before GPU execution. |
| Environment mismatch | Focused engineering tests are CPU-hidden; R2 is trusted GPU/XLA, TensorFlow `float64`, TF32 enabled, soft placement disabled. |
| Artifact mismatch | Fresh root, exact source hashes, configs, seeds, histories, states, device/HLO evidence, and result JSON are required. |

Pre-mortem: a wider network could still saturate because `0.01` is too hot;
that would nominate a later capacity-plus-schedule repair rather than prove
capacity irrelevant. It could avoid saturation but underfit by step 1,200;
the terminal heldout gate detects obvious failure, while only full 5,000-step
confirmation can assess the completed candidate. It could pass finite probes
while missing unknown modes; later transformed-HMC, replication, predictive,
and calibration gates remain necessary.

## R1: Engineering Repair

1. Preserve `dsge_paper_neutra_config` and all direct source-parity tests.
2. Add a named SSL-LSTM capacity preset whose only algorithmic difference is
   `hidden_layers=(32,32)`.
3. Give frozen artifacts a distinct capacity-adaptation procedure label.
4. Test topology, parameter count, masks, schedule, objective/update behavior,
   target binding, exact resume, serialization/reload, and rejection of width
   or procedure-label mutations.
5. Run the existing source-parity suite plus new capacity tests CPU-hidden.

## R2: Paired 1,200-Step GPU/XLA Diagnostic

Use the exact historical A/B initialization, training, and validation streams.
Run fresh `(32,32)` training sequentially to step 1,200 with immutable
checkpoints every 100 steps. At every checkpoint record overall and per-stage
saturation, moderate-shell radius/worst label, validation loss, gradients,
clipping, and state identity.

Timing rules are binding:

- any nonfinite training/validation state is an immediate hard stop;
- saturation above `0.05` at any 100-step checkpoint is an R2 nomination veto;
- shell radius is explanatory before step 1,200 and tested against `4.30` only
  at terminal step 1,200; and
- paired heldout step-1,200-minus-step-0 one-sided 95% upper bound is tested
  against zero only at terminal step 1,200.

No R2 state is a nominated transport. The runner may serialize a diagnostic
payload only to test exact reload, clearly labeled `diagnostic_only`.

## Resources

The owner directed implementation and retesting. This plan interprets that as
authorization for R1 and the smallest discriminating R2 only, not two new full
5,000-step seeds. R2 has a shared cap of `9,000` trusted GPU-seconds (`2.5`
GPU-hours), at most `4,500` seconds per stream, with sequential stopping. Do
not start B unless at least `4,500` seconds remain. Unused time does not
authorize full confirmation, another width, schedule tuning, HMC, or forecasts.

## Handoff

- `R2_CAPACITY_REPAIR_NOMINATED`: both paired diagnostics pass; estimate full
  runtime and request/confirm a prospective two-fresh-seed budget.
- `R2_CAPACITY_REPAIR_NOT_NOMINATED`: preserve evidence and diagnose whether
  saturation remains optimization-driven; do not silently lower the LR.
- `INVALID_EVIDENCE`: repair implementation/runtime/artifact validity before
  interpreting capacity.

Only two fresh full seeds passing unchanged gates can reopen Phase 5.
