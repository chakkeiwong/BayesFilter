# Kalman QR Gradient Benchmark Restart Audit

Date: 2026-07-17

Status: `LANE_CLEAN_RESTART_READY`

Scope: the LGSSM Kalman QR analytical/autodiff gradient benchmark lane only.
Use the commit containing this memo as the restart anchor.

## Restart Anchor

The detailed scientific and engineering handoff remains:

`docs/plans/bayesfilter-kalman-qr-gradient-benchmark-reset-memo-2026-07-15.md`

Its conclusions, retained artifact paths, remaining gaps, recommended
persistent-pool experiment, and nonclaims remain current. No Kalman source,
test, benchmark plan, result, or terminal evidence changed after the closeout
commit `32695caa35c6e660f2fe6ed515bcb2b90123dc7f`.

Current inherited repository main before this audit:
`ffaaaf903354e095da126dbfa47878c34717c5b8`.

## Generated-File Audit

The Kalman lane satisfies the requested invariant:

- authored implementation, harnesses, tests, plans, result notes, reset memo,
  and terminal claim-supporting structured evidence are tracked;
- retry directories, worker logs, HLO text, progress journals, generated
  per-method Markdown/payload mirrors, failed attempts, and superseded run
  roots are ignored by the scoped Kalman rules in `.gitignore`;
- `git diff --name-only` and `git ls-files --others --exclude-standard` show no
  Kalman lane path;
- no Kalman benchmark or worker process is running.

The shared repository worktree is not globally clean. At audit time it has 188
untracked files plus tracked modifications belonging to the active
SSL-LSTM/NeuTra/HMC, nonlinear SVD, and related lane. Those include authored
Python modules, tests, plans, reviews, and scientific artifacts. They were not
committed, ignored, modified, or deleted here. A repository-wide
`all files tracked or ignored` claim must wait for that lane's owner to classify
and close its work.

## Current Result

- GPU memory growth remains required and the tested allocator peaks remain
  small (`302.5 MiB` in the full lattice and `131.1 MiB` in the matched run).
- All compared CPU and GPU routes use XLA; the fast process route is not a
  non-XLA comparator.
- Native CPU `B=16` remains the tested throughput bottleneck.
- Persistent 16-process `B=1` sharding and GPU native `B=16` remain viable.
- GPU versus process throughput at `(D,P,T,B)=(30,150,12,16)` remains
  unresolved under six variable shared-GPU blocks.

## Next Work

Open a Tier-2 plan for persistent-pool lifecycle integration in the real LGSSM
gradient caller. Compare `k={4,8,16}` persistent CPU workers with GPU native
`B=16` at one small and one representative larger workload. Measure startup,
dispatch, steady execution, cleanup, RSS, parity, and GPU thermal/clock/power
telemetry separately.

Do not rerun additional `tf.map_fn` or `tf.vectorized_map` wrappers without a
new lower-level row-parallel mechanism.

## Restart Check

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_gradient_scaling_lattice.py \
  tests/test_kalman_qr_cpu_throughput_comparison.py \
  tests/test_kalman_qr_cpu_xla_formulation_shootout.py \
  tests/test_kalman_qr_matched_cpu_process_gpu.py
```

## Nonclaims

No repository-wide clean-worktree claim, universal XLA/compiler failure,
universal GPU or process superiority, equal-cost ranking, persistent-pool
readiness, HMC/posterior readiness, default change, production readiness, or
scientific validity.
