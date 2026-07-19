# SSL-LSTM NeuTra DSGE-Parity Material Training Launch

Date: 2026-07-15

Status: `COMPLETED_SEED_INSTABILITY_REPAIR_REQUIRED`

Completion update, 2026-07-15: the detached program completed both 5,000-step
seeds in `31,881.50` charged seconds. Seed A was nominated; seed B completed
without a hard veto but failed the moderate-shell and scale-saturation screens.
No material runner remains active. The final result is
`docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-result-2026-07-15.md`.

The authorized material program is running as a detached user service:

| Field | Value |
| --- | --- |
| Unit | `bayesfilter-ssl-lstm-neutra-material-20260715.service` |
| PID | `4184336` |
| Start | `2026-07-15T10:30:38+08:00` |
| Working directory | `/home/ubuntu/python/BayesFilter` |
| Environment | conda `tfgpu`; `CUDA_VISIBLE_DEVICES=1` |
| Physical device | GPU 1, NVIDIA GeForce RTX 4080 SUPER |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_dsge_parity_material_training_2026_07_15.py --program-output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training` |
| Structured start receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/program-start.json` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-plan-2026-07-15.md` |

Trusted runtime evidence observed before handoff:

- TensorFlow created the intended RTX 4080 SUPER device;
- the journal contains `Compiled cluster using XLA!`;
- a five-sample `nvidia-smi dmon` window showed 84-85 percent GPU 1 SM use;
- seed A wrote valid immutable checkpoints at steps 100 and 200;
- step-200 file SHA-256 is
  `997718b9442b431a4e30a29eb3ad06023cb521a84038df8d5528718ea39feb93`;
- step-200 trainer step and Adam iteration are both 200; and
- its recomputed state hash exactly matches the stored
  `9cf8fc662580412bc23de08555291cb6551173a2bf0854c73dde4ef762394ddb`.

Checkpoint 100 was written at `10:39:09+08:00`; checkpoint 200 was written at
`10:44:34+08:00`. The 325-second interval is about 3.25 seconds per step and
projects seed A inside its 18,000-second cap, subject to later timing variation
and finalization overhead. This is a resource projection only.

Monitoring commands:

```bash
systemctl --user status bayesfilter-ssl-lstm-neutra-material-20260715.service --no-pager
journalctl --user -u bayesfilter-ssl-lstm-neutra-material-20260715.service -n 100 --no-pager
find docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training -maxdepth 2 -type f
```

Do not edit the active runner, its material tests, the material plan, the
historical probe helper, the parity trainer/loader/target, or the direct parity
test while the service is running. Their hashes are checked again before a
candidate can be admitted.

This launch record remains runtime provenance, not the candidate decision.
The final program artifact has SHA-256
`3b2b1a27c4b9af4d4f10026a111d03ed464c42b3d4f3a4f0a8a93217686c230d`.
No posterior, HMC, predictive, ranking, superiority, readiness,
paper-fidelity, or scientific claim is made.
