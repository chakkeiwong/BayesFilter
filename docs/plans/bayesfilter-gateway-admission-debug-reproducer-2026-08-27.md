# Gateway Admission Debug Reproducer

Date: `2026-08-27`  
Scope: permission-review admission only; no scientific result.

The HTTP 503 is returned by the elevated permission reviewer before the shell
creates a process. A normal program cannot force that response from inside the
process. This diagnostic separates command admission from Python, TensorFlow,
GPU, and the Phase 52 arguments.

[`scripts/gateway_admission_repro.py`](/home/ubuntu/python/BayesFilter/scripts/gateway_admission_repro.py)
never starts a child process and never writes a file. Its `shape` mode treats
arguments after `--` as uninterpreted strings and reports their count, byte
size, and SHA-256 digest.

Run each item as a separate elevated request through the same gateway:

1. `true`
2. `/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/gateway_admission_repro.py --mode python`
3. `TF_FORCE_GPU_ALLOW_GROWTH=true timeout 5s /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/gateway_admission_repro.py --mode sleep --seconds 2`
4. The exact Phase 52 argument shape, without TensorFlow. Use the same tokens
   as the real command, but pass them to `shape` after `--`:

```bash
PHASE52_ARGS=(
  --pilot-root-1 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-01
  --pilot-root-2 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-02
  --pilot-root-3 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-03
  --pilot-root-4 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-04
  --pilot-root-5 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-05
  --pilot-root-6 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-06
  --fixture-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired
)
TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true timeout 60s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/gateway_admission_repro.py \
  --mode shape -- "${PHASE52_ARGS[@]}"
```
5. CPU-hidden TensorFlow import: `CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true timeout 60s /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/gateway_admission_repro.py --mode tensorflow_cpu`
6. Tiny trusted GPU/XLA probe: `TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true timeout 60s /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/gateway_admission_repro.py --mode tensorflow_gpu`
7. The unchanged Phase 52 boundary command from its subplan.

The `shape` request should use the same `--pilot-root-*`, `--fixture-root`,
and `--output-root` tokens as the real command, but it must not invoke the
boundary. Keep the reviewer request ID, HTTP status, exact command, and
timestamp for every rejection. A reviewer rejection produces no probe JSON;
that is different from a process exit code.

Interpretation:

| Observation | Likely boundary |
|---|---|
| `true` is rejected | Permission-review service is unavailable globally. |
| Controls 1-3 pass, `shape` is rejected | Argument length, tokens, or risk classifier. |
| `shape` passes, CPU TensorFlow is rejected | Python/TensorFlow import policy or environment classifier. |
| CPU TensorFlow passes, GPU probe is rejected | GPU/device admission or GPU risk policy. |
| GPU probe passes, Phase 52 is rejected | Runner path, long timeout, output paths, or command-specific rule. |
| Phase 52 starts | The earlier 503 was transient; validate its normal artifacts. |

The probes are bounded diagnostics only. They do not run the SSL-LSTM target,
create particle artifacts, train NeuTra, run HMC, execute LEDH, or change a
repository default. A successful probe says only that one admission shape was
accepted; it says nothing about Phase 52 scientific validity.

## Observed isolation result

As of `2026-08-27T17:47:53+08:00`, the following requests are accepted by the
elevated path:

| Request | Observed result |
|---|---|
| `true` | admitted; exit `0` |
| `timeout 1s /bin/true` | admitted; exit `0` |
| project Python `-c` | admitted; exit `0` |
| probe interpreter-only mode | admitted; `PASS_GATEWAY_PROBE` |
| full Phase 52 argv shape, `60s` timeout | admitted; 19 argv entries, 1,245 bytes |
| full Phase 52 argv shape, `24000s` timeout | admitted; same argv digest |
| CPU-hidden TensorFlow import | admitted; TensorFlow `2.20.0`, no GPU visible |
| tiny TensorFlow GPU/XLA operation | admitted; two RTX 4080 SUPER devices, memory growth verified |
| real Phase 52 runner with `--help` | admitted; imports and GPU initialization completed, exit `0` |

The exact Phase 52 scientific command was previously rejected before process
creation five times. The reviewer reported HTTP 503 from
`https://llm.visioncoder.ai/responses` with request IDs:

- `1a16dba2-c8e7-4cd7-92f6-b9730a005d40`
- `d50ea25a-2ac9-4544-bf5d-1dc6f42d6925`
- `28f58517-fbb8-46f1-bdd6-cb759015ac57`
- `0beff352-928c-47e2-a2f3-608991aae42d`
- `550d0b2c-3a01-4b12-8f48-cf6885228ef0`
- `ca16e829-1eaf-4dd9-8767-c6713c752e7d`

At that point every reduced request passed, so the only reproducer was the
exact scientific command. A clean retry after switching gateways produced the
sixth 503 request ID above, before process creation. The standalone Bash
reproducer below subsequently narrowed the failing request substantially.

## Standalone Bash Reproducer

A repository-independent Bash script now provides a substantially smaller
reproducer at `/tmp/gateway_admission_repro_standalone.sh`. It has no BayesFilter
paths or imports and writes no artifacts. Its Bash-only `basic`, `shape`,
standard-Python, and CPU-hidden TensorFlow modes all start and pass. Submit this
GPU mode as one elevated gateway request:

```bash
PYTHON_BIN=/home/ubuntu/anaconda3/envs/tfgpu/bin/python TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true timeout 60s /tmp/gateway_admission_repro_standalone.sh tensorflow-gpu
```

On `2026-08-27`, that standalone command was rejected before process creation
with the same HTTP 503 from `https://llm.visioncoder.ai/responses`, request ID
`3fe18c6e-53ab-414a-b923-2d260650df26`. This is the smallest currently checked
reproducer that fails while its neighboring modes pass. Because the gateway
rejects before Bash starts, the result produces no script output and does not
initialize TensorFlow or a GPU.
