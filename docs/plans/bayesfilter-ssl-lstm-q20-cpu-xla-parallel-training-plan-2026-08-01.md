# q=20 CPU-XLA Parallel Seed-A/Seed-B Training Plan

Date: 2026-08-01
Status: `COMPLETED_CPU_XLA_DIAGNOSTIC`

Preflight status: two-seed XLA topology smoke passed on 2026-08-01. The
4,000-update campaign completed on 2026-08-01. Terminal result:
`docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-parallel-training-result-2026-08-01.md`.

## Research Intent

Run the existing q=20 `(32,32)` NeuTra training protocol for seed A and seed B
concurrently, using two disjoint 25-worker CPU lanes. Each lane uses a batch of
100 rows (`25 workers x 4 rows`), TensorFlow/XLA target evaluation, and XLA in
the parent transport/optimizer graph. The run is a CPU diagnostic comparison,
not GPU/default/HMC evidence.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Scientific question | Do two independently seeded q=20 CPU-XLA streams complete the 4,000-update adaptive training screen with finite artifacts and heldout/support/audit checks? |
| Exact baseline | Existing q=20 synthetic target, `(32,32)` three-stage dense IAF, `batch_size=100`, `lr=4e-4`, initialization scale `0.01`, clip norm `10`, and the existing seed definitions |
| Mechanism | Parent and worker `jit_compile=True`; 25 persistent workers per seed, 4 rows per worker; seed A and B run concurrently on disjoint CPU ranges |
| Primary criterion | Each seed reaches the declared plateau stop or 4,000-step maximum and writes a finite terminal result with checkpoints, support, and heldout audit |
| Hard vetoes | Visible GPU; wrong CPU affinity; configured cores above 50; nonfinite target/score/loss/gradient/support/audit; RSS above 64 GiB per child; missing/corrupt terminal artifact; child failure |
| Explanatory diagnostics | Loss trajectory, learning-rate repairs, runtime, XLA compile cost, thread count, RSS, worker skew |
| Promotion veto | Any seed fails its declared heldout paired-loss or support screen |
| Nonclaims | No posterior correctness, convergence, HMC readiness, transport promotion, architecture ranking, GPU equivalence, or repository default change |
| Artifacts | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/{seed-a,seed-b}/` |

## Parallel Resource Contract

- Seed A child: CPUs `0..24`, 25 workers, 4 rows per worker.
- Seed B child: CPUs `25..49`, 25 workers, 4 rows per worker.
- Combined configured compute cores: exactly `50`.
- Expected peak RSS: approximately `2 x 18.9 GB`, below the 256 GB host
  capacity and the per-child `64 GiB` veto.
- Each child has an independent worker pool, checkpoint root, progress file,
  launch manifest, and terminal summary.
- The supervisor does not share trainer state, checkpoints, validation data, or
  random streams between children.

## Budget And Stop Contract

- Per-child campaign cap: `40,000 s`; the supervisor wall cap is `40,000 s`.
- The measured XLA `25x4` warm update is `6.867 s`; 4,000 updates therefore
  require roughly `27,500 s` before validation/support/audit overhead.
- The cap is a maximum, not an expected-use claim. A plateau stop may return
  unused time. The supervisor waits for both children and records each exit.
- No child is resumed from the historical non-XLA 2,000-step artifacts.

## Skeptical Pre-Execution Audit

- Wrong baseline: no. Both children use the same target, model, controls, and
  code; only the declared random seed and CPU range differ.
- Proxy promotion: no. Loss and screen status remain diagnostic; they cannot
  promote CPU training to the repository GPU default.
- Missing stop condition: no. Each child has a 4,000-step cap, 40,000-second
  budget, finite checks, and final artifact requirements.
- Resource contention: addressed by disjoint `taskset` ranges and a supervisor
  check that both ranges are subsets of the host affinity.
- Stale artifacts: addressed by a fresh versioned root; no historical root is
  overwritten or resumed.
- Misleading success: even two passing CPU streams do not establish posterior,
  HMC, or GPU equivalence.

Audit decision: `PASS_FOR_BOUNDED_PARALLEL_CPU_XLA_DIAGNOSTIC`.

## Preflight Smoke

Command:

```bash
python docs/benchmarks/run_ssl_lstm_q20_cpu_xla_parallel_supervisor_2026_08_01.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/smoke-2step \
  --cap-seconds 600 \
  --debug-stop-after-steps 2
```

Result: `COMPLETED` in 92.02 s. Both children exited with code `0`, reached
`CPU_DEBUG_SMOKE_COMPLETED` at program step 2, used the declared disjoint
affinities (`seed-a`: CPUs `0..24`; `seed-b`: CPUs `25..49`), and emitted
TensorFlow `Compiled cluster using XLA` receipts. This is initialization and
topology evidence only; it does not assess 4,000-update training quality.

Smoke artifacts:

- `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/smoke-2step/summary.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/smoke-2step/seed-a/seed-a/debug-smoke-result.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/smoke-2step/seed-b/seed-b/debug-smoke-result.json`

## Commands

```bash
timeout 40000 python \
  docs/benchmarks/run_ssl_lstm_q20_cpu_xla_parallel_supervisor_2026_08_01.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1 \
  --cap-seconds 40000
```

The command completed with aggregate status `COMPLETED`. Seed A stopped at
step 2,250 after `plateau_after_lr_repair`; seed B stopped at step 3,250 after
the same declared controller condition. Both terminal child results passed
the CPU-XLA diagnostic screen with no vetoes. See the result note for the
decision and inference-status tables.
