# Phase 8 C1 strict-backend cost-rescue subplan

Date: 2026-08-30  
Status: `PASS_PARITY_AND_COST_PILOT`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`  
Prior feasibility result:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-c1-result-2026-08-30.md`  
Localization result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-backend-localization-result-2026-08-30.md`

## Scope boundary

The original C1 allocation (`5,400` seconds) is closed and is not being
silently reopened. This is one new, separately recorded repair arm under the
user-authorized remaining campaign budget. It changes only the principal
square-root backend from `compiled_custom_op` to the existing
`tensorflow_eigh_strict` implementation. It does not change the target,
bridge, data, theta measure, score authority, transport architecture, candidate
counts, confirmation streams, or promotion criteria. C2--C5 and Phase 9 remain
closed until this arm produces a complete feasibility receipt and a refreshed
reviewed subplan.

## Research intent and evidence contract

| Field | Contract |
|---|---|
| Question | Does the strict device-side eigen backend preserve the q=20 target semantics and complete the frozen K=2 two-batch cost pilot within a bounded wall budget? |
| Baseline | The same q=20 target and bridge using `compiled_custom_op`; prior attempts timed out before a complete two-batch receipt. |
| Candidate | `tensorflow_eigh_strict`, with all other target and runner settings frozen. |
| Primary pass criterion | A parity receipt passes, followed by `PASS_PHASE8_COST_PILOT` with B=8 and B=32, finite values/scores/status, immutable checkpoint replay, learned-map reliability, route scan, XLA/memory-growth receipts, and allocator peak at most 4 GiB. |
| Hard vetoes | Target or bridge identity drift, value/score/status parity failure, nonfinite or invalid rows, wrong backend, checkpoint/reliability failure, forbidden row mapping/pfor, missing GPU memory-growth receipt, allocator cap violation, overwrite, or timeout. |
| Explanatory diagnostics | Timing, loss, gradient norms, residuals, occupancy, and selected batch size. They cannot establish whitening, mode discovery, convergence, or superiority. |
| Nonclaims | No backend default promotion, Gaussian IID pullback, mode-discovery guarantee, posterior correctness, HMC readiness, statistical ranking, or high-dimensional scaling. |

### Required parity before cost

Use the existing bounded q=20 GPU-native-eigh parity harness, which compares
separately compiled XLA programs on the same fixed two-row batch and then
checks a strict-backend batched update. The frozen tolerances are value
absolute `1e-8`, score absolute `1e-7` or scaled relative `1e-7`, and equal
status codes. The two-row check is a nomination diagnostic, not a proof for
all future batches; the cost manifest must still retain the backend identity.

### Cost pilot

Run the existing Phase 8 runner with `K=2`, the frozen compact-high recipe, one
compile plus four steady updates per chart, both `B=8` and `B=32`, and the
original validation bank size `256`. Keeping `validation-size=256` avoids
turning the faster backend into a reduced-bank claim. The runner's predeclared
selection rule remains operational only: choose B=32 only when finite, peak
allocation is at most 4 GiB, and median steady update time is no more than four
times the B=8 median; otherwise nominate B=8.

## Budget and artifacts

This arm has a fresh command-wall cap of `2,700` seconds: `900` seconds for
the parity receipt and `1,800` seconds for the cost pilot. The numbers are
bounded campaign allocations, not performance estimates. A timeout consumes
this arm's allocation and closes the hypothesis; it does not authorize an
unbounded retry. No Phase 9 reserve or untouched confirmation stream is read.

Fresh output root:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/
```

Parity output:

```text
.../backend-parity/attempt-01-native-eigh/
```

Cost output:

```text
.../cost-pilot/attempt-01-native-eigh-n256/
```

Every attempt must retain a manifest or timeout/failure record, Git state,
exact command, conda/Python/TensorFlow versions, target and bridge signatures,
backend, device, memory-growth receipt, XLA/TF32 settings, seeds, wall time,
and output paths. No existing directory may be overwritten.

## Commands

Parity (GPU 0, memory growth before TensorFlow import):

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
timeout --signal=TERM --kill-after=60s 900s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_gpu_native_eigh_localization_2026_07_31.py \
--mode gpu \
--output-root docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/backend-parity/attempt-01-native-eigh
```

Cost pilot (repository-default launcher, no idle probe or per-run Luna gate):

```text
BAYESFILTER_PHASE8_MODE=cost-pilot \
BAYESFILTER_PHASE8_TIMEOUT_SECONDS=1800 \
BAYESFILTER_PHASE8_PRINCIPAL_SQRT_BACKEND=tensorflow_eigh_strict \
BAYESFILTER_PHASE8_VALIDATION_SIZE=256 \
BAYESFILTER_PHASE8_OUTPUT_ROOT=docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/cost-pilot \
BAYESFILTER_PHASE8_ATTEMPT_LABEL=attempt-01-native-eigh-n256 \
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

The cost command is run only after parity passes. If parity fails, write a
failure result and stop; do not use timing to excuse a semantic mismatch.

## Skeptical pre-execution audit

| Risk | Control |
|---|---|
| Faster backend changes the target | Same-input value/score/status parity and frozen target signature are hard prerequisites. |
| Tiny parity is overinterpreted | The parity result only nominates; the full cost manifest retains separate backend identity and no scientific promotion follows. |
| Old C1 budget is silently reopened | New date-stamped root, fresh 2,700-second cap, and explicit scope boundary. |
| Reduced validation hides cost | Cost uses the frozen 256-row validation bank. |
| A timeout is mistaken for a mathematical failure | Classify it as backend feasibility evidence only and close this arm. |
| Memory or service boundary is confused with science | Pre-import growth, one visible GPU, allocator telemetry, and no idle probe are recorded separately. |
| Partial checkpoints are promoted | Only a complete manifest can pass; partial outputs remain diagnostic evidence. |
| Successful pilot triggers premature sampling | C2--C5 and Phase 9 stay closed pending a new reviewed decision. |

Audit verdict: `PASS_FOR_ONE_PARITY_THEN_ONE_BOUNDED_COST_PILOT`.

## Repair record before execution

Parity attempt 1 failed before constructing either target because the legacy
harness imported BayesFilter modules before configuring TensorFlow memory
growth. The failure is preserved at
`.../backend-parity/attempt-01-native-eigh/failure.json`. The import order was
repaired in the harness, and `py_compile` plus `git diff --check` passed. The
fresh retry uses `attempt-02-native-eigh`; no output is overwritten and no
backend or scientific contract changes.

The repaired parity retry (`attempt-02-native-eigh`) passed in
`67.73723304306623` process seconds. It recorded value maximum absolute
residual `1.8835777382264496e-10`, score maximum absolute residual
`1.4963432715120462e-9`, scaled score relative residual
`1.480718774225973e-9`, and equal target signatures/status codes. The strict
trainer completed three XLA updates and HLO extraction; its warm-update median
was `4.224995819968171` seconds and TensorFlow peak allocation was
`692393216` bytes. These are parity/feasibility receipts only. The cost-pilot
command below is now the only remaining action in this subplan.

The cost pilot completed in `261.52175762609113` seconds with status
`PASS_PHASE8_COST_PILOT`. Both B=8 and B=32 reliability receipts passed, peak
TensorFlow allocation stayed below 4 GiB, and the frozen rule selected B=32.
The full decision is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-result-2026-08-30.md`.
This closes the new feasibility arm; it does not authorize C2 or Phase 9 by
itself.

## Between-step repair and stop rules

After parity, classify any failure as serialization/harness, numerical/backend,
resource, or target. Repair a localized receipt defect once within the same
cap and fresh attempt directory. A semantic mismatch, invalid target, or cap
exhaustion closes this arm. After a complete cost receipt, update the parent
program, write a result note, and refresh the next subplan before any training
search. No confirmation stream is consumed here.
