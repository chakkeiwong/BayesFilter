# SSL-LSTM q=20 performance and whitening next plan

Date: 2026-09-02  
Status: `CLOSED_N0_N2_NO_NOMINATION_N3_BLOCKED`  
Predecessor: `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-result-2026-09-02.md`
Governing authority: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`
Reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-02.md`

## Purpose and boundary

The predecessor proved that q=20 target batching is useful but did not prove a
semantically valid grouped HMC implementation, and it did not explain the
large fresh-chart pullback residual. This plan has two independent work
streams. Stream A establishes exact per-candidate transition equivalence before
any batching change. Stream B runs a target-specific training ladder to
separate under-training from capacity or objective failure. The terminal M3
replay and Phase 9B remain closed.

## Evidence contract

| Item | Definition |
|---|---|
| Target | Frozen q=20 bridge signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, strict float64 diagnostic backend |
| Baseline | Existing serial reusable HMC runner and independent reverse-KL trainer |
| Stream A primary gate | For a fixed candidate, seed, initial state, step size, leapfrog count, and target-call sequence, grouped and scalar transitions produce identical state, accept/reject, log-accept, and call-count receipts within declared floating-point tolerance |
| Stream B primary gate | On disjoint calibration/validation banks and three independent seeds, a candidate must have finite/status-clean updates and at least two of three seeds must reduce the aggregate held-out pullback score RMS by at least 10 percent; this is a nomination screen, not a convergence claim |
| Validation pairing | The held-out validation seed is identical across arms for each seed index; calibration and training seeds remain arm-specific and disjoint from validation |
| Vetoes | Target/signature mismatch, non-finite status, memory-growth failure, changed kernel semantics, seed coupling, validation leakage, or missing artifacts |
| Nonclaims | No global whitening, mode discovery, posterior correctness, convergence, scalability, or superiority claim from this plan |

## Defaults and assumptions

The existing static batch-size specialization is retained as the baseline. A
single fixed diagnostic batch shape is used per compiled fixture so trace counts
remain interpretable. Stream A uses a counter-based per-candidate random-number
allocation; this is a new hypothesis and must be checked against the scalar
transition rather than assumed equivalent. Stream B starts with the existing
architecture as a warm start and tests update count, learning rate, gradient
clip, hidden width, and stage count in a small reviewed ladder. No setting is a
new default until target-specific multi-seed evidence supports it. The 10
percent/2-of-3 rule is a provisional diagnostic hypothesis chosen to distinguish
no movement from a measurable local repair; it is not a published tolerance.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest check | Promotion status |
|---|---|---|---|---|
| Static batch-size specialization | Existing target implementation | Shape changes can trigger retracing or alter graph semantics | Trace counts and fixed input signatures in N1/N2 manifests | Baseline |
| Per-candidate folded seeds | New N1 hypothesis | Grouped TFP may consume seeds differently from scalar calls | State/gradient/target/log-acceptance equivalence fixture | Hypothesis only |
| Three seeds and 12 updates | Bounded diagnostic budget | Slow optimization or seed variability is missed | Per-seed held-out residual trace | Diagnostic ladder |
| 10% and 2-of-3 nomination rule | Provisional separation threshold | Noise can reject a small repair or overstate a short one | Replication with uncertainty intervals | Not a default |
| Shared validation bank per seed index | Comparability requirement identified during audit | Arm-specific banks confound cross-arm differences | Manifest seed-map invariant | Required for comparison |
| Batch 32, beta 0.5, clip 10 | Target-specific ladder hypothesis | Optimization geometry may be poorly conditioned | Finiteness, gradient, and residual telemetry | Not promoted |
| GPU 0, XLA, TF32, memory growth | Repository execution policy | Device or allocator mismatch invalidates run | Trusted device/memory receipt | Policy default |

## Pre-mortem

The campaign could appear successful while still being misleading if arms used
different held-out banks, if high acceptance masked negligible movement, or if
the short ladder overfit one seed. The seed-map check, movement/residual
telemetry, and three-seed held-out screen address those risks; the first GPU
attempt demonstrated why the bank check is necessary. It could fail for an
implementation reason if TFP's grouped stateless-seed semantics differed from
the scalar kernel, if a graph retraced for changing shapes, or if GPU memory
growth was applied too late. The exact N1 fixture, trace receipt, launcher
policy, and route scan distinguish those failures from a candidate-quality
failure. A finite short run would still not establish whitening, convergence,
or mode discovery.

## Phases and repair points

### N0/R0: contract and source audit

Re-read the predecessor result, inspect the exact HMC transition and target
interfaces, write the seed/call-count convention, and run focused tests. The
harness writes the N0 receipt as `run_start.json` and the final manifest under
the versioned attempt directory. Stop before the ladder if the scalar
transition cannot expose the required receipt.

### N1/R1: exact grouped-transition fixture

Run `docs/benchmarks/run_ssl_lstm_q20_performance_whitening_next_2026_09_02.py`;
the harness executes N1 after N0 and records it in `n1_grouped_transition`.
The `--cpu-n1-only` option exits after N1 for the CPU control, while the full
GPU command continues into N2. Use four candidates, one chain each, fixed
`L=3`, fixed step size 0.20, and explicit per-candidate folded stateless
seeds. Compare (a) the
batched TFP transition, (b) the scalar transition with the same per-candidate
seeds, and (c) a `tf.while_loop` row-loop control that calls the scalar graph.
Record states, gradients, accept/reject values, expected target-call counts,
and trace counts. The row-loop control is a semantic control only. If the fast
batched TFP result differs from the scalar receipt, classify that as a design
boundary and do not integrate it.

### N2/R2: target-specific training ladder

Use fresh, disjoint calibration and validation latent banks. Test exactly these
three arms, each at three independent seeds and 12 updates: A `(16,16)`, two
stages, learning rate `1e-3`; B `(16,16)`, two stages, learning rate `3e-4`;
C `(32,32)`, three stages, learning rate `3e-4`. Batch size is 32, beta is
0.5, and gradient clipping is 10. Record initial/final held-out residuals,
per-update validity and gradient norms, training time, and allocator readings.
Acceptance remains explanatory only. The held-out validation bank is shared
across arms for each seed index, while calibration and training streams remain
arm-specific. The exact command and output root are bound below.

### N3/R3: conditional implementation repair

Only if N1 passes, add the smallest batch transition wrapper with stable
signatures and focused equivalence tests. Only if N2 nominates a candidate,
write a separate reviewed training plan; do not alter the active default or
reuse validation data for tuning. Any failed candidate triggers a fresh
repair/refresh entry, not a relaxed gate.

### N4/R4: closeout

Write a result note with decision and inference-status tables, uncertainty
limits, exact manifests, and a post-run red-team paragraph. A new replay plan
must state its changed schedule and total budget explicitly. Preserve all prior
artifacts.

## Exact execution commands

N0/N1/N2 use the diagnostic harness and launcher:

```text
BAYESFILTER_NEXT_MAX_SECONDS=500 BAYESFILTER_NEXT_ATTEMPT_ID=n0-n2-02 BAYESFILTER_GPU_ID=0 \
bash scripts/run_ssl_lstm_q20_performance_whitening_next_gpu.sh \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n0-n2-02-gpu
```

The CPU-only source/analytic fixture is:

```text
BAYESFILTER_NEXT_ATTEMPT_ID=n1-cpu-05 CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_performance_whitening_next_2026_09_02.py \
  --cpu-n1-only \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n1-cpu-05 \
  --max-seconds 300
```

## Budget and execution boundary

The bounded diagnostic budget was 900 GPU seconds and 300 CPU seconds, with a fresh
versioned output root below
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/`.
GPU runs must use memory growth before TensorFlow import. No package changes,
network access, destructive Git actions, or external messages are included.
The user's 2026-09-02 request authorized execution within this unchanged
bounded contract. It does not authorize M3 replay, Phase 9B, package changes,
network access, destructive Git actions, or a new hardware/privacy boundary.

## Skeptical review

The plan was reviewed on 2026-09-02 against the predecessor result and the
terminal master. It avoids the old M3 baseline, separates exact-equivalence
gates from speed proxies, declares validation separation and seed count, and
provides repair/refresh points. The audit also checks wrong baselines, proxy
promotion, unsupported numeric thresholds, seed/data leakage, static-shape
retracing, target-call accounting, GPU memory policy, output collisions, and
the possibility that TFP cannot provide per-row RNG equivalence. Verdict:
`PASS_M3C_BOUNDED_CONTINUATION`.

The post-run audit found that the first GPU attempt assigned a different
held-out validation bank to each arm. That attempt is preserved but is invalid
for cross-arm comparison. Before the claim-bearing rerun, the harness was
repaired to use one held-out validation seed per seed index across all arms;
the repaired manifest records the common seed map. This is a comparability
repair, not a change to the target, method, or nomination rule.

## Execution ledger and repair/refresh

| Phase | State before launch | Required closeout |
|---|---|---|
| N0/R0 | Complete | Source/seed audit, focused regression, and fresh receipts |
| N1/R1 | Complete; fast path rejected | Exact grouped/scalar/row-loop comparison; no integration on mismatch |
| N2/R2 | Complete; no nomination | Common-bank three-arm, three-seed ladder and held-out classification |
| N3/R3 | Blocked by declared entry conditions | Requires exact grouped equivalence and/or a separately reviewed semantic design, plus a nominated candidate for training follow-up |
| N4/R4 | Complete | Result, inference-status table, red-team note, reset memo, and master refresh |

## Closeout receipt

The authoritative result is
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`.
The repaired GPU manifest is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n0-n2-02-gpu/run_manifest.json`
(hash `a73661f50e2b559e54549c950ccaadaa0b027c1e9dbbdecd74971f692e68f4e4`),
and the fresh CPU N1 manifest is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n1-cpu-05/run_manifest.json`
(hash `649b10aaa9a00f4d5df8645c2f3abca89b24001861afc2425a0a52c51193d78c`).

N1's fast grouped transition was finite but not equivalent to the scalar
receipt; the row-loop control was exactly equivalent. N2 had 9/9 finite
candidates and 12/12 valid updates per candidate, but A, B, and C each had
zero of three seeds meeting the `>=10%` score-RMS nomination threshold. No
default or active route changed. N3 is therefore blocked and no new GPU
command is authorized by this plan. The two GPU attempts consumed
`834.5439817190636` seconds of the 900-second diagnostic budget; the remaining
budget is not an authorization to alter the contract or reopen M3.
