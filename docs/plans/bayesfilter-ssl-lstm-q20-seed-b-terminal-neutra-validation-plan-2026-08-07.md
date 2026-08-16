# SSL-LSTM q=20 Seed-B Terminal NeuTra Validation Plan

Date: 2026-08-07  
Status: `PASS_FOR_BOUNDED_EXECUTION`

## Research Intent Ledger

| Field | Prospective definition |
| --- | --- |
| Main question | Did the clean seed-B fixed-budget continuation produce a NeuTra transport that supports a valid, converged fixed-HMC sampler under the repository sequential policy? |
| Candidate under test | Seed-B continuation checkpoint 4000, optimizer step 6250, selected by the disjoint 500-row selection bank and clear on its untouched 500-row audit and finite support/round-trip checks. |
| Mechanism | Tune a fresh fixed identity-mass HMC kernel in seed-B NeuTra coordinates over `L=(5,10,15,20,25,3)`, then run the selected kernel through `bayesfilter_neutra_sequential_hmc_v1`. |
| Expected failure mode | The trained transport may still have poor HMC geometry, target-invalid states, unstable energy behavior, or chains that do not converge within policy caps. |
| Tuning criterion | The public BayesFilter fixed-transport tuner nominates a seed-B-bound kernel after a fresh screen and 64-transition verification. Tuning only nominates; it does not establish convergence. |
| Sequential promotion criterion | At least 2,000 discarded warm-up transitions per chain; latest 1,000 warm-up transitions have maximum rank/folded R-hat `<=1.05`; then at least 1,000 retained transitions per chain have maximum rank/folded R-hat `<=1.01`, bulk ESS `>=400`, and tail ESS `>=400` in both NeuTra and mapped model coordinates. |
| Promotion veto | Invalid target status; non-finite state, target, proposed target, score, log acceptance, or energy difference; available positive native divergence; any chain not moving; or per-chain chunk acceptance probability outside `[0.35,0.95]`. |
| Continuation veto | Input/tuning/kernel identity mismatch; `L=1`; non-XLA or visible-GPU CPU worker; worker/affinity failure; corrupt archive; warm-up/retained cap failure; or wall-cap refusal. |
| Repair trigger | No tuned kernel or a sequential veto triggers a seed-B kernel/transport/initialization diagnosis. It rejects this candidate, not NeuTra generally, unless target or math invalidity is established. |
| Explanatory diagnostics | Acceptance inside the allowed bound, continuous R-hat/ESS values, runtime, RSS, movement magnitude, and finite energy-error tails. |
| Must not be concluded | Posterior correctness, model adequacy, scientific truth, superiority, robustness across seeds, or default readiness. |

## Evidence Contract

- Question: the main question in the ledger.
- Exact baseline: the completed seed-B terminal checkpoint at continuation update
  4000. Seed A is excluded because its training proposal stream encountered a
  target-validity veto and its recovered state conditioned on a replacement
  batch.
- Primary criterion: the sequential promotion criterion above. Reverse-KL loss,
  tuning acceptance, and the prior 500-row audit are not promotion criteria.
- Hard vetoes: only the promotion and continuation vetoes in the ledger.
- Explanatory only: all continuous loss, acceptance within bounds, runtime,
  energy-tail magnitude, R-hat before admission, and ESS before admission.
- Nonclaim: even a sequential pass establishes sampler convergence diagnostics
  for this fixed target/transport, not posterior/reference correctness.
- Artifact root:
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1/`.

## Input Evidence

The seed-B run completed 4,000 continuation updates with terminal optimizer
step 6250. The terminal checkpoint had the lowest fixed 500-row selection mean
loss, `40.60126788755288`. Its untouched 500-row audit mean was
`40.67435728533879` with standard error `0.0675201446000665`. The selected
support probe was finite and the maximum round-trip residual was
`6.829004511048463e-15`.

The result status is `GPU_CONTINUATION_COMPLETED_CANDIDATE_NOMINATED` with
`vetoes=[]`. Its v1 result schema does not explicitly contain a
`target_validity_failures` counter. Completion under the historical finite-only
route implies that no invalid status reached that route, because such a row
would have aborted it; this is not represented as explicit zero-event
telemetry. The new tuning and sequential routes preserve and gate target status
directly.

## Frozen Execution Protocol

### Phase 1: Seed-B Fixed-HMC Tuning

- CPU/XLA FP64, GPUs hidden before TensorFlow import.
- Four chains in every public tuner arm.
- Grid `(5,10,15,20,25,3)`; `L=1` is forbidden.
- Concurrent assignments preserve the measured allocation:
  `L5:6`, `L10:6`, `L15:8`, `L20:16`, `L25:16`, `L3:6`, plus CPU 127 for the
  supervisor.
- Fresh 2026-08-07 tune, screen, and verification seeds.
- Public tuning configuration: target acceptance `0.70`, candidate band
  `[0.65,0.75]`, budgets `(8,16,32)`, 16-result fresh screen, and 64-result
  fresh verification.
- Maximum phase wall time: 43,200 seconds. One material attempt; a localized
  harness retry may use only remaining time under the same cap.
- If no candidate is nominated, stop. Do not transfer seed A's `L=10` kernel.

### Phase 2: Sequential Fixed HMC

- Begins only from the hash-bound Phase-1 kernel.
- CPU/XLA FP64, GPUs hidden before TensorFlow import.
- Four persistent chains, each using eight cores; CPU 32 supervises.
- Each immutable chunk contains 500 transitions per chain. Chain state and
  deterministic random stream continue across chunks.
- Warm-up minimum/window/maximum: 2,000/1,000/10,000 per chain.
- Retained minimum/maximum: 1,000/10,000 per chain.
- Every warm-up chunk is archived but excluded from posterior estimates.
- All numerical, target-status, movement, divergence, and acceptance-bound
  gates apply to every chunk.
- Maximum phase wall time: 86,400 seconds. Before each later chunk, the measured
  slowest prior chunk plus 25% margin and a 600-second archive reserve must fit.
- Cap refusal preserves completed chunks and is classified
  `UNDER_BUDGETED_PARTIAL`; sample minima and thresholds are never reduced.

## Default And Numeric Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Seed-B checkpoint 4000 | Measured clean nomination; candidate baseline | Only completed continuation seed with no result veto | v1 event telemetry is not explicit | New tuner traces direct status; first invalid status vetoes |
| Six-`L` grid | Prior target-specific q=20 tuning design; baseline hypothesis | Screens trajectory length instead of transferring seed A's kernel | Grid may miss a viable `L` | No nomination is a repair trigger, not NeuTra rejection |
| Tuning numeric budgets | Inherited public q=20 tuner; reviewed baseline | Reproduces the prior target-specific nomination procedure with fresh seeds | Short screens are noisy | 64-transition fresh verification and later sequential gates |
| Identity NeuTra-coordinate mass | Public fixed-transport tuner policy; hypothesis | Tests whether the learned transport regularized geometry | Residual correlations may require a different fixed metric | Warm-up and retained convergence failures diagnose this |
| Four chains | Repository sequential minimum; reviewed default | Supports rank/folded multi-chain diagnostics | Four starts may miss modes | Posterior/reference validation remains separate |
| Initial chain states | Inherited modest dispersed NeuTra starts; warm-start hypothesis | Provides deterministic separation near the nominal latent region | Initial dispersion may be inadequate | Preflight value/score/status plus warm-up readiness |
| 500-transition chunks | Measured prior q=20 route; reviewed execution choice | Archives bounded progress and gives regular gates | Feedback arrives only after a costly chunk | Forecast before each subsequent chunk |
| Acceptance `[0.35,0.95]` | Existing q=20 sequential gate and user-directed acceptance bound | Rejects unusably sticky or degenerate fixed kernels | A single chunk may cross by finite-run noise | Report as candidate veto only; no broader target claim |
| R-hat and ESS thresholds | Owner sequential-HMC policy; reviewed default | Required sampler convergence screen | Passing is not posterior truth | Mandatory posterior/reference follow-up |
| 12-hour tuning cap | Measured prior grid completed in 7.49 hours; derived cap | Covers observed grid wall with margin | Seed-B geometry may tune more slowly | Supervisor preserves per-arm completion and stops at cap |
| 24-hour sequential cap | Prior `L=10` chunk measured about 3.0 hours; bounded hypothesis | Allows at least the six chunks needed for the minimum if new `L` is no more costly | Higher `L` or slower kernel can remain under-budgeted | First real chunk establishes the forecast; no weakened minimum |

## Skeptical Pre-Execution Audit

1. Wrong baseline: corrected. The campaign uses seed B's selected terminal state,
   not seed A, an old best state, the recovered seed-A proposal, or a checkpoint
   filename's controller label.
2. Proxy promotion: corrected. Loss and tuning acceptance can nominate only;
   sequential R-hat/ESS plus hard health gates decide sampler admission.
3. Missing stops: corrected. Target status, finite values, divergence,
   acceptance bounds, movement, identity, XLA/device, worker, archive, phase
   caps, and policy caps are explicit.
4. Unfair comparison: this is a single-candidate validation, not a method or
   seed ranking. Every tuning arm uses the same public procedure and fresh
   seed family.
5. Hidden assumptions: architecture, selected checkpoint, mass, grid, seeds,
   initial states, chunk sizes, thresholds, topology, and caps are classified
   above.
6. Stale context: corrected. Seed B's terminal result and checkpoint were read
   directly on 2026-08-07; seed A's target-validity veto is not ignored.
7. Environment mismatch: HMC sampling is an explicit CPU/XLA validation lane;
   every process hides GPUs before TensorFlow import. This does not change the
   repository GPU default for NeuTra training.
8. Artifact adequacy: checkpoint/tuning/kernel hashes, complete tuning arms,
   every warm-up and retained sample/trace shard, seeds, chain state, affinity,
   XLA/device evidence, progress, and terminal summaries answer the question.
9. Misleading success: R-hat/ESS can pass for the wrong posterior. Passing this
   campaign therefore triggers an untouched posterior/reference check rather
   than a correctness claim.
10. Misleading failure: a tuning, initialization, metric, resource, or kernel
    failure rejects only the current candidate/protocol unless target or math
    invalidity is independently established.

Audit verdict: `PASS_FOR_BOUNDED_EXECUTION`.

## Localized Execution Repair

The r1 tuning phase completed in `28222.05524148399` seconds and nominated
`L=3`, step size `0.8115211181271775`, with fresh verification mean acceptance
probability `0.6881377960324893`. The first sequential preflight then stopped
before sampling because the wrapper used a new `:sequential` transformed-target
scope while the kernel was correctly bound to the tuning scope
`:claim_tuning_grid6`. Since target scope is part of the transformed-adapter
signature, the preflight rejected the mismatch.

This was an integration identity error, not target, transport, tuning, or HMC
evidence. The wrapper now preserves the exact tuning target scope. The repair
changed no checkpoint, target, transport, kernel, seed, sampler, threshold,
hardware class, or compute budget. After the correction, 26 focused tests and
the real seed-B sequential preflight passed. The already completed tuning
artifact is retained; only the previously unstarted sequential phase is resumed
under its original 86,400-second cap and a new service identity.

## Verified Commands

Focused tests:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_q20_seed_b_terminal_neutra_validation.py \
  tests/test_neutra_sequential_hmc.py \
  tests/test_ssl_lstm_q20_chart_a_l10_sequential_hmc.py \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation.py
```

Observed before launch: `46 passed`.

End-to-end supervisor command (the supervisor runs the phase commands below and
starts sequential HMC only after a seed-B-bound tuning nomination):

```bash
CUDA_VISIBLE_DEVICES=-1 taskset -c 126 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_terminal_neutra_validation_supervisor_2026_08_07.py
```

Tuning preflight and material command:

```bash
CUDA_VISIBLE_DEVICES=-1 taskset -c 127 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_terminal_six_l_tuning_2026_08_07.py \
  --mode preflight \
  --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1/tuning

CUDA_VISIBLE_DEVICES=-1 taskset -c 127 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_terminal_six_l_tuning_2026_08_07.py \
  --mode supervisor --cap-seconds 43200 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1/tuning
```

Sequential preflight and material command after tuning nomination:

```bash
CUDA_VISIBLE_DEVICES=-1 taskset -c 32 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_terminal_sequential_hmc_2026_08_07.py \
  --mode preflight \
  --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1/sequential-preflight

CUDA_VISIBLE_DEVICES=-1 taskset -c 32 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_seed_b_terminal_sequential_hmc_2026_08_07.py \
  --mode run --cap-seconds 86400 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1/sequential
```

## Result Requirements

The terminal result must contain a decision table, inference-status table,
engineering/numerical/scientific ledgers, serious run manifest, and post-run
red-team note. It must say which hard vetoes fired, whether the candidate
remains viable, which differences are descriptive only, whether any ranking is
statistically supported (none is planned), and what evidence is still required.
