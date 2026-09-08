# SSL-LSTM q=20 NeuTra adaptive replay A/B campaign

Date: 2026-08-24

Status: `EXECUTED_SCREEN_COMPLETE_NO_PROMOTION`

This plan amends and operationalizes the route/assumption preflight required by
the adjudication plan
`bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
It is the execution plan for a paired screen of Candidate A and Candidate B.
It does not promote either route to a default, HMC, posterior, or scientific
claim.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does content-independent fixed-law replay (A) or frozen-before-draw adaptive replay with decaying stale influence (B) provide a viable q=20 NeuTra training signal under the same target and compute budget? |
| Candidate A | At every update, maintain a fixed-capacity buffer of whole blocks drawn from one frozen full-support known-density Gaussian-mixture proposal. Each slot is refreshed by an independent content-independent coin. |
| Candidate B | At every one-based update `u=1,2,...`, freeze the current transport, draw a fresh block from a known-density mixture of the fixed proposal and the frozen transport density, and combine that fresh block with stale blocks using deterministic `lambda_u = lambda_0/u^p`, `p>1`. |
| Comparator | The existing fixed six-bank normalized SMC replay is retained as historical context only. It is not an A/B arm and is not a proof-bearing estimator. |
| Mechanism isolation | A and B use the same target, proposal geometry, architecture, block size, optimizer, per-seed validation construction, paired seeds, and update ladder. The only intended difference is proposal adaptation and stale-buffer semantics. This bounded screen does not create an independent audit block. |
| Primary screen criterion | Both arms complete the route preflight and produce finite, status-valid, batch-native training artifacts without violating their declared replay semantics. |
| Promotion veto | Missing or non-finite proposal density, target status failure, support failure, failure to freeze `phi_t` before B's draw, content-dependent A refresh, non-summable B schedule, scalar/row-mapped target evaluation, or artifact/hash failure. |
| Repair trigger | A finite implementation or resource failure triggers a bounded harness repair and rerun under the same target/method/data/hardware/budget. A tail/support or envelope failure triggers proposal repair or theorem-scope downgrade, not silent threshold relaxation. |
| Explanatory diagnostics | ESS, maximum normalized weight, log-weight tails, latent moments/covariance/dependence, loss, gradient norms, refresh counts, adaptive mixture fraction, and wall time. These do not prove whitening, mode discovery, convergence, or superiority. |
| Nonclaims | No finite-sample unbiasedness claim for the screen estimator, no optimizer-convergence theorem for Adam, no global strong-monotonicity claim for the symmetric dense IAF, no mode-discovery proof, no HMC or posterior claim, and no statistical ranking from one or two seeds. |

## Exact mathematical routes

Let `r_0(theta)` be the fixed two-component Gaussian mixture whose means and
covariances are read from the frozen q=20 geometry artifact. Let
`q_phi(theta)` be the current transport density
`rho(T_phi^{-1}(theta)) |det D T_phi^{-1}(theta)|`.

Candidate A uses

```text
m_A(theta) = r_0(theta).
```

Candidate B freezes `phi_u` before drawing at one-based update `u` and uses

```text
m_{B,u}(theta) = (1-alpha) r_0(theta) + alpha q_{phi_u}(theta),
```

with stale coefficient `lambda_u = lambda_0/u^p`, `u=1,2,...`, and the same
mixture density evaluated by log-sum-exp. A transport draw is
generated as `theta=T_{phi_t}(z)`, `z~N(0,I)`. Every block stores its proposal
definition, allocation, source seed, target signature, and evaluated
`log(m(theta))`; no caller-supplied density is trusted without recomputation.

The screen combines per-block self-normalized log weights into a numerically
stable weighted loss. This is deliberately an empirical finite-block screen:
it is **not** the unnormalized estimator in Theorem 1. A theorem-bearing follow-
up would need an unnormalized SMC-U/known-density implementation and a
step-size/optimizer analysis. Candidate B's `lambda_t` is nevertheless checked
against the stated summability condition as a route-semantic diagnostic.

For A, the refresh coin is generated independently of block values, weights,
target values, and ancestry. For B, the current transport state hash is taken
before and after block generation and must match; stale blocks are never used as
the persistent fresh gradient lane.

## Baseline and assumption audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| Two-region Gaussian mixture `r_0` | Existing q=20 geometry artifact and prior canary | Full support and exact density are available; isolates replay semantics | Missed material modes or extreme ratios | Held-out support and log-ratio tails | Reviewed baseline, not posterior authority |
| Block size and buffer capacity | Initial bounded-screen hypothesis | Keeps target evaluation batched and GPU memory bounded | Too little proposal coverage or excessive variance | ESS and independent-block variability | Screen hypothesis; not a default |
| Adam and current dense IAF | Existing trainer path | Keeps A/B comparison implementation-matched | Constant-step Adam has no Theorem 2/2A guarantee; symmetry copies remain | Paired seed runs and local symmetry inventory | Comparator implementation, not theorem setting |
| Per-block self-normalization | Existing trainer contract and numerical stability | Prevents overflow and makes A/B loss scales comparable | Finite-block ratio bias | Compare raw log-normalizer and ESS; no unbiasedness claim | Explicit screen estimator |
| `lambda_t` schedule | Theorem 2A qualitative requirement | `p>1` makes `sum eta_t lambda_t` finite for bounded constant screen step proxy | Schedule may be too weak/strong for finite training | Emit partial sum and tail bound | Reviewed semantic diagnostic |
| GPU/XLA and memory growth | Repository owner policy | Required for serious NeuTra training | Launch or allocator failure | Trusted GPU preflight before training | Required execution policy |

## Evidence contract

**Question.** Under a common q=20 target and architecture, can both replay
semantics be implemented and screened without hidden density or sampling
assumptions?

**Comparator.** Paired A/B arms, plus the historical fixed normalized-SMC
baseline as descriptive context only.

**Primary pass/fail.** Route preflight and finite batch-native training pass for
each arm; all required provenance and semantics are present in the manifest.

**Veto diagnostics.** Nonfinite target/proposal/weights; invalid target status;
support or density mismatch; B state changed during draw; A refresh correlated
with block contents; non-summable schedule; scalar target fallback; stale or
colliding output root; GPU memory-growth/XLA failure for a serious GPU arm.

**Explanatory diagnostics.** ESS, tails, whitening moments, loss, gradient
norms, and runtime. They can nominate a repair but cannot establish Gaussianity,
convergence, or superiority.

**Nonconclusion.** A passing screen does not admit HMC or establish posterior
correctness. HMC is a separate future phase requiring a frozen transport and
the canonical sequential controller.

**Artifact.** A versioned root under
`docs/plans/artifacts/ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/`
containing the reserved/retry roots, the passing `r1-gpu` preflight and
per-arm results, a run manifest, and a result note with decision and
inference-status tables.

## Skeptical plan audit (completed before execution)

| Audit question | Finding | Disposition |
|---|---|---|
| Is the baseline answering the stated question? | The old runner reuses rows 0:600 and therefore tests neither adaptive arm. | New runner owns A/B semantics; old runner remains context only. |
| Is a proxy being used as promotion evidence? | Whitening, ESS, and loss are tempting proxies for Gaussianization. | Classified as explanatory only; primary screen is route/finite correctness. |
| Are stop/veto conditions explicit? | Prior plan lacked an executable distinction between route failure and poor geometry. | Separate hard vetoes, repair triggers, and nonclaims above. |
| Is A's refresh law actually fixed? | Replacing individual SMC rows or using target-dependent eviction would violate the theorem route. | Refresh whole known-density blocks with independent coins only. |
| Is B's adaptive density bound to the frozen state? | An evolving transport evaluated after the draw would make the proposal density ambiguous. | Freeze/hash `phi_t` before draw and recompute the mixture density. |
| Is the estimator theorem being overstated? | Existing trainer self-normalizes finite blocks. | Screen is explicitly empirical; no Theorem 1 unbiasedness claim. |
| Is `lambda_t` summability enough for Adam? | No; constant-step Adam is outside the theorem's stated stochastic-approximation regime. | Record summability as semantic evidence only and prohibit theorem promotion. |
| Are symmetry and mode claims controlled? | Dense tanh IAF has hidden-unit permutation/sign symmetries; global condition (30) is unavailable. | Restrict any interpretation to local/empirical diagnostics. |
| Are data partitions and artifacts independent? | The bounded screen uses disjoint preflight/validation seeds but has no independent audit partition, so it cannot support a generalization claim. | Record the missing audit partition as a nonclaim; use a fail-closed output root and reserve an independent audit block for any later promotion study. |
| Can the command answer the question within budget? | A full HMC campaign would consume budget without testing replay semantics. | Execute preflight and bounded training screen only; HMC explicitly out of scope. |

The audit passes after these repairs. No known material flaw remains that would
make the A/B screen meaningless before implementation.

## Execution phases and budget

The user-authorized incremental budget is 64,800 seconds (18 hours). The
campaign reserves 3,600 seconds for closeout and diagnostics, leaving at most
61,200 seconds for this screen. The numerical values below are bounded-screen
hypotheses, not performance defaults.

1. **Phase 0, documentation and static checks:** no GPU; verify plan hashes,
   Python compilation, Markdown parsing, and existing artifact hashes.
2. **Phase 1, CPU route preflight:** generate small calibration blocks for A
   and B, verify density identities, freeze semantics, statuses, support, tail
   and summability diagnostics. Cap: 1,800 seconds.
3. **Phase 2, trusted GPU paired screen:** run A and B with paired seeds,
   batch-native target evaluation, XLA, verified memory growth, fresh output
   roots, and bounded update ladders. Stop after the first paired seed if the
   route screen fails; add the second paired seed only if both first arms pass
   and the remaining budget is sufficient.
4. **Phase 3, result adjudication:** write decision and inference-status tables,
   classify failures, and explicitly state whether the result affects only the
   candidate or the research direction. No HMC launch is part of this plan.

## Exact implementation command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_adaptive_replay_neutra_ab_2026_08_24.py \
  --phase screen --device 1 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/r1
```

The CPU preflight uses `--phase preflight --cpu-only` and hides GPUs before
TensorFlow import. Both commands refuse an existing output root.

The exact trusted GPU screen command executed for the terminal result was:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_adaptive_replay_neutra_ab_2026_08_24.py \
  --phase screen --device 1 --rows-per-block 64 --updates 24 \
  --hidden-width 16 --stages 2 --learning-rate 0.0003 \
  --max-seconds 54000 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/r1-gpu
```

The reserved `r1` root contains the first preflight/smoke attempts; localized
shape repairs were isolated in `r1-retry-01`, `r1-retry-02`, and
`r1-retry-03`. The passing trusted GPU preflight and terminal screen are under
`r1-gpu`. No root was reused or overwritten.

## Stop and interpretation rules

Stop an arm immediately on any hard veto. A failed arm is a candidate or
implementation failure unless the harness, target, data, or artifact is shown
invalid. Do not rank surviving arms from descriptive metrics alone. The next
justified action after a clean screen is a separately reviewed theorem-bearing
estimator implementation or a frozen-transport HMC plan, not an automatic
promotion.

## Execution record

The CPU route preflight and repaired smoke passed. The trusted GPU preflight and
paired screen completed for both arms and both paired seeds. The terminal result
is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-result-2026-08-24.md`,
with machine-readable artifacts under
`docs/plans/artifacts/ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/r1-gpu/`.
The result is `SCREEN_COMPLETED_NO_PROMOTION`: the route implementation gates
passed, but whitening and HMC/posterior promotion gates were not met or were
not authorized.

Terminal red-team review corrected the one-based `lambda_u` notation, recorded
the exact `r1-gpu` command/root, and clarified that this bounded screen has no
independent audit partition. These are documentation and provenance repairs;
they do not change the executed target, code, data, budget, or interpretation.
