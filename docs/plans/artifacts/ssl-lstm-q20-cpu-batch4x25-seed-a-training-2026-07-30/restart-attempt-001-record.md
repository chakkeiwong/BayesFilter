# q=20 CPU Seed-A Resume Attempt 001

Date: 2026-07-30
Status: `RESUME_LAUNCH_RECORDED`

## Recovery Decision

The reboot left a valid latest joint checkpoint at seed-A program step 250.
The checkpoint and sibling progress receipt passed the launcher's joint
checkpoint validation after a minimal infrastructure repair: JSON-decoded
stream seed lists are compared canonically with the in-memory stream record.
This repair changes no target, trainer, optimizer, seed, topology, stop rule,
budget, or evidence criterion. Attempt 000 remains preserved as historical
infrastructure provenance and is not relabeled.

## Budget Arithmetic

- Launch timestamp: `2026-07-29T18:12:26.314607+00:00`
- Trusted prior-boot shutdown boundary: `2026-07-30T03:45:33.099045+08:00`
- Source: `journalctl -b -1`, `systemd-shutdown` final SIGTERM/shutdown record
- Measured elapsed duration: `5586.784438 s`
- Charged prior wall time (rounded upward): `5587 s`
- Cumulative cap: `31500 s`
- Remaining outer timeout: `31500 - 5587 = 25913 s`

The shutdown boundary is used instead of the smaller checkpoint elapsed value,
so the campaign cannot undercharge work done after the last checkpoint.

## Validated Checkpoint

- Path: `seed-a/checkpoint-0250.json`
- File SHA-256: `6d6e043090a3e3a4a5ab5f7337146927f2c3cdbbc6e60268ba47d262cd5211fd`
- Joint checkpoint hash: `c4a1f70eabdc717206afc4336114574d3e87fed9c26f7c45a9eeb0278b5b30ac`
- Program/trainer step: `250`
- Controller status: `running`
- Best step: `250`
- Progress history steps: `[0, 250]`

## Source State

- Repository `HEAD` at inspection: `882679796e8ee684b6b020b7cd84e3cfc1d92d58`
- This is a direct child of the memo's `eae7955bbe8a8970328a162a1504e8b04b1ad57c`; its committed change only adds the reboot memo.
- Working-tree repair: `docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py`
- Repaired launcher SHA-256: `989d37dedd182dc16a242ca18615d0841925148e61704567d5176d2d1ffcaa61`
- Active plan, pool, and target hashes match attempt 000; no changes were made to them.
- Focused checks: `32 passed` (`tests/test_ssl_lstm_q20_strict_cpu_training.py` and `tests/test_neutra_training_control.py`)

## Exact Command

```bash
timeout 25913 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a \
  --cpu-processes 25 \
  --batch-per-process 4 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 31500 \
  --resume-checkpoint \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30/seed-a/checkpoint-0250.json \
  --prior-wall-seconds 5587
```

## Evidence Contract

The existing plan remains authoritative: completion requires the declared
plateau stop or 2,000 program updates plus finite final support and heldout
audit checks. Nonfinite values, resource violations, corrupted artifacts,
source/config mismatch, or exhausted cumulative budget veto continuation.
This CPU-only stream remains diagnostic and cannot authorize HMC, seed B,
transport promotion, posterior claims, or a change to the GPU NeuTra default.
