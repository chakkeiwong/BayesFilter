# SSL-LSTM NeuTra DSGE-Procedure Parity Repair Result

Date: 2026-07-15

Status: `DSGE_PROCEDURE_PARITY_ENGINEERING_PASSED_MATERIAL_TRAINING_BUDGET_REQUIRED`

## Outcome

BayesFilter now implements a source-matched transfer of the plain NeuTra
procedure in the cited local Rotemberg/SGU `dsge_hmc` implementation for the
locked four-coordinate SSL-LSTM target: three dense IAF stages with `(4,4)`
ELU hidden widths, reverse-coordinate mixing between stages, and a fixed
identity-scale translation to the prior center. Training uses standard-normal
base draws, reverse KL, batch 480, 5,000 steps, Adam at `0.01` with epsilon
`1e-7`, schedule boundaries `[999,3999]`, independent per-variable norm
clipping at 10, and score matching off. This is local-source procedure parity,
not a general claim of fidelity to the NeuTra paper or literature.

Direct tests against the actual sibling `dsge_hmc` implementation at commit
`d94566c9f70b3143e599a56eba7cb461ff2bda88` pass for explicit-tensor forward
maps, log determinants, every trainable gradient tensor, and one Adam update.
Mutation, chart/signature, exact-resume, six-component serialization, reload,
and failed-restore non-mutation tests also pass.

The source-hash-matched trusted GPU/XLA canary passed. It measured a
post-compile mean step time of `2.9719` seconds and maximum of `3.0085`
seconds. Conservatively using the maximum gives `15,042.7` seconds (`4.1785`
GPU-hours) per 5,000-step seed and `30,085.3` seconds (`8.3570` GPU-hours) for
two seeds, before compile, validation, checkpoint, serialization, and
contention overhead. The next phase should request a **10 GPU-hour contingency
cap** for the complete A/B pair: that is 19.7% above the raw `8.3570`-hour step
estimate to accommodate two compile/warmups and unmeasured validation,
checkpoint, serialization, and ordinary timing variation. Ten hours is a
prospective resource cap, not a measured minimum or evidence that both runs
will finish. No material training was launched here.

Earlier timing receipts `timing-canary.json`, `timing-canary-r2.json`, and
`timing-canary-r3.json` are superseded because relevant source changed after
they ran. They are historical timing diagnostics only. The authoritative
receipt is `timing-canary-r4.json` with SHA-256
`d20e1219026dcd3b62b2218b1978ec6979647d41022062db578ac766fdefb001`.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close engineering parity repair | Passed direct source parity, mutations, serialization/reload, and GPU/XLA canary | No engineering or execution veto | Procedure parity does not establish learned transport quality | Freeze this preset for a separately budgeted two-seed material phase | Paper fidelity generally, posterior correctness, HMC readiness, or scientific validity |
| Do not launch material training | Resource gate correctly stopped execution | Existing one-hour Phase 4 budget is closed; no new material budget was authorized | Full-run overhead and stochastic training outcome | Request a 10 trusted GPU-hour contingency cap, 19.7% above the raw step estimate, with independent A/B seeds and sequential stopping | Candidate rejection, success, superiority, or default readiness |
| Keep affine candidates as controls | Historical affine A/B remain viable controls only | No control hard veto | Linear controls cannot answer nonlinear geometry | Carry them into later matched comparisons after nonlinear admission | Nonlinear-phase completion or evidence affine is scientifically better |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Engineering parity and final GPU/XLA canary passed; no material candidate was run |
| Statistically supported ranking | None; no stochastic method comparison occurred |
| Descriptive-only differences | Compile time, step times, loss, gradients below hard validity thresholds, and comparison with superseded canaries |
| Default readiness | Not established |
| Next evidence needed | Two complete independent 5,000-step runs, prospective candidate gates, exact frozen-target preflight, transformed HMC, replication, and predictive validation |

## Review And Repairs

Claude performed bounded read-only plan and implementation reviews. The plan
review exposed missing hidden-width, target-chart, and direct-source parity
bindings. The implementation review exposed nonfinite-gradient handling,
serialization signature drift, partial mutation on failed restore, and legacy
schedule/config ambiguity. Each was repaired and covered by focused tests;
the final Claude verdict was `VERDICT: AGREE`. The durable review summary is
`docs/reviews/bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-claude-review-2026-07-15.md`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` (dirty worktree preserved) |
| DSGE source commit | `d94566c9f70b3143e599a56eba7cb461ff2bda88` |
| CPU test command | `CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_neutra_reverse_kl_training.py tests/test_neutra_dsge_procedure_parity.py tests/test_dense_iaf_neutra_artifact_loader.py` |
| GPU canary command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_dsge_parity_timing_canary_2026_07_15.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/procedure-parity-repair/timing-canary-r4.json` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; `float64` |
| Device policy | trusted GPU, XLA JIT on, TF32 enabled, soft placement disabled |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| GPU canary seed | `[20260715,4099]` |
| Canary compile/warmup | `109.2263` seconds |
| Measured steps | `3.0085`, `2.9931`, `2.9484`, `2.9514`, `2.9580` seconds |
| Canary wall time | `181.9562` seconds |
| Receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/procedure-parity-repair/timing-canary-r4.json` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-repair-plan-2026-07-15.md` |
| Result | this file |

The receipt binds these SHA-256 values: trainer
`211d93ec5bd228ae444b814d891af9a7714c5f3026e52ad0fcf29d992b3469ae`,
artifact loader
`f3dbb9ad2f750679f2d67b63bb8cce4db2c16622907512a701de799abf9194ec`,
direct parity test
`c6b15ab6eb09505700ef9fc65b3216c7e9c625e84be323c5690d4c1731293e12`,
runner
`26077ed39f9a586214532a6b29ca83097c669b22197bc01e916030b81ab57ab5`,
and locked target
`6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667`.

## Post-Run Red Team

The strongest alternative explanation is that exact procedure transfer still
inherits a procedure poorly suited to the SSL-LSTM posterior. Parity removes
the implementation-regression explanation; it does not show that reverse KL
will cover all important modes or ridge geometry. A canary can compile and
remain finite while a full run saturates or misses support. That is why loss
movement and timing are explanatory only, and why the next phase must retain
the prospective support, saturation, reload, and exact-target vetoes for both
independent seeds.

What would overturn this closeout is a reproducible mismatch with the cited
`dsge_hmc` forward, gradient, optimizer, schedule, or composition semantics,
or a source-hash mismatch in the authoritative receipt. No such mismatch is
present in the bounded evidence. The weakest remaining evidence is scientific,
not engineering: no full learned transport, transformed chain, or predictive
law has yet been evaluated.
