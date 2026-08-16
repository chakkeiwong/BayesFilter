# BayesFilter Clean Reset Memo

Date: 2026-08-16
Branch: `main`

The terminal four-model GenUT artifact is preserved in
`docs/benchmarks/artifacts/genut_four_model_leaderboard_rerun_20260816/attempt04/`.
All 16 legacy dual-cap cells passed finite-value, finite-score, program-valid,
and residual gates. This is viability evidence only; it does not establish
exact nonlinear score correctness, unbiasedness, NeuTra readiness, HMC
readiness, or default promotion.

The next scientific gate is the current-scope LGSSM Kalman-oracle ladder in
`docs/benchmarks/run_genut_lgssm_oracle_validation_20260816.py`. Do not start
claim-bearing NeuTra/HMC until that gate is executed and reviewed. LGSSM and
KSC-SV current data scopes have regenerated hashes; old controls are warm
starts, not promoted defaults.

Every worktree file must be tracked or intentionally ignored. Track authored
source, tests, scripts, documentation, literature ledgers, mathematical notes,
compact claim/promotion receipts, and terminal result Markdown. Ignore local
literature downloads, build products, Python caches, binary backups, generated
LaTeX/PDF files, checkpoints, raw tensors/arrays, JSONL streams, private
payloads, mutable progress state, PIDs, tmux markers, logs, status directories,
and temporary run directories. No generated file is deleted by this reset.

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` for GPU work. Build the
custom op against the active TensorFlow environment and configure memory growth
before logical-device initialization.

Restart sequence: verify no visible untracked files; run custom-op/GPU/XLA
regressions; run the LGSSM Kalman oracle; review before nonlinear diagnostics;
keep LM/trust-region evidence separate from legacy dual-cap evidence; keep
NeuTra/HMC admission closed until exact-score and posterior gates pass.

The terminal comparison is documented in
`docs/plans/bayesfilter-genut-four-model-leaderboard-rerun-result-2026-08-16.md`.
This memo is a restart boundary, not a claim that remaining score/HMC gates
are complete.
