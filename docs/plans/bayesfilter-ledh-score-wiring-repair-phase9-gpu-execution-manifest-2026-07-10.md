# Phase 9 Trusted GPU/XLA Execution Manifest

Date: 2026-07-10

Status: `FROZEN_REVIEW_PENDING_NO_GPU_COMMAND_AUTHORIZED`

## Authorization Boundary

This manifest freezes commands before any Phase 9 GPU result is observed. It
does not authorize execution. A fresh bounded substitute review must return
`VERDICT: AGREE` before the trusted `nvidia-smi`, TensorFlow/XLA probe, or any
Gate B command runs.

Gate B is a compile/device/correctness preflight only. After Gate B, write and
review its result before Gate C. After a row's full-time seed-`81120` prefix,
write and review that row result before running seeds `81121..81124`. These
review boundaries do not permit changing frozen settings after observing a
result; any change requires a revised plan and remains diagnostic.

## Frozen Identity

| Field | Value |
| --- | --- |
| Repository | `/home/chakwong/BayesFilter` |
| Python | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python` |
| GPU selection | physical GPU index `0`; `--cuda-visible-devices 0`; logical `/GPU:0` |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| XLA | `tf.function(..., jit_compile=True)` with no non-JIT fallback |
| Precision | `float32`, TF32 enabled |
| Seeds | `81120,81121,81122,81123,81124`; one seed per runtime process |
| Transport | `active-all`, streaming plan, full AD, `manual_streaming_finite_sinkhorn_stopped_scale_keys`, 10 Sinkhorn iterations, annealing scale `0.9`, convergence threshold `1e-3` |
| Nonlinear Sinkhorn epsilon | `1.0` |
| Memory budget | maximum per-seed reset score peak `<=14000 MiB` |
| Runner | `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py` |
| Exact command generator | `docs/benchmarks/build_ledh_phase9_gpu_command_manifest.py` |
| Exact command JSON | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json` |
| Artifact root | `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla` |
| Log root | `docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla` |
| Plan | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Gate A result | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-a-harness-result-2026-07-10.md` |
| Phase result | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md` |

Every nonlinear shard records and the aggregator recomputes SHA-256 hashes of
the shared runner, active row adapter, shared score contract, fixed-SIR helper,
streaming/core TensorFlow transport implementations, and annealed transport
implementation. A code edit after a shard invalidates it for aggregation.

The exact command JSON is the executable source of truth. It contains 10 Gate
B runtime commands, 36 Gate C runtime commands, 40 Gate D runtime commands,
and 5 aggregate commands (`91` total), with complete argv, score-reference, output,
Markdown, log, row, seed, `T`, and `N` fields. Before execution, verify it is
current:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/build_ledh_phase9_gpu_command_manifest.py --output docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json --check
```

Execute only the selected entry's `shell_command`, from repository root, after
its predecessor gate passes. Do not hand-substitute a template or regenerate
the JSON after a GPU result without a revised review. The runner records and
validates the exact-command JSON hash in every shard.

## Frozen Row Matrix

| Row | Full `T` | Full `N` | `row/col/particle` chunks | FD step | FD atol | FD rtol | Gate C prefixes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| fixed-SIR | 20 | 10000 | `1024/1024/512` | `1e-3` | `1e-2` | `5e-2` | `1,5,20` |
| predator-prey | 20 | 10000 | `512/512/512` | `1e-4` | `5e-3` | `5e-3` | `1,5,20` |
| actual-SV | 1000 | 10000 | `512/512/512` | `1e-4` | `5e-3` | `5e-3` | `4,50,250,1000` |
| generalized-SV | 1008 | 10000 | `512/512/512` | `1e-4` | `5e-3` | `5e-3` | `4,50,252,1008` |
| KSC-SV | 1000 | 10000 | `512/512/512` | `1e-4` | `5e-3` | `5e-3` | `4,50,250,1000` |

Gate B reuses the existing tiny fixture `T/N` only: fixed-SIR, actual-SV,
generalized-SV, and KSC-SV use `T=1,N=4`; predator-prey uses `T=1,N=2`.
Transport modes, epsilon, iterations, and admitted chunk sizes remain frozen.
Gate B memory is a compile/preflight screen, not an `N=10000` memory result.

## Trusted Preflight

Run only after the fresh review agrees. Both commands require trusted or
escalated GPU execution.

```bash
mkdir -p docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla
```

```bash
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv,noheader > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/nvidia-smi-preflight.txt 2>&1
```

```bash
MPLCONFIGDIR=/tmp CUDA_VISIBLE_DEVICES=0 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -c "import datetime as dt, json, os, platform, subprocess, sys; from pathlib import Path; import tensorflow as tf; trust='owner_designated_managed_session_visible_gpu_trusted'; physical=[str(d) for d in tf.config.list_physical_devices('GPU')]; logical=[str(d) for d in tf.config.list_logical_devices('GPU')]; f=tf.function(lambda x: tf.linalg.matmul(x,x), jit_compile=True); x=tf.ones([2,2],tf.float32); y=f(x); record={'schema_version':'bayesfilter.ledh.phase9.gpu_preflight.v1','timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'working_directory':str(Path.cwd()),'python_executable':sys.executable,'python_version':platform.python_version(),'tensorflow_version':tf.__version__,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_status_short':subprocess.check_output(['git','status','--short'],text=True),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_trust_basis':trust,'physical_gpus':physical,'logical_gpus':logical,'jit_compile':True,'tf32_execution_enabled':bool(tf.config.experimental.tensor_float_32_execution_enabled()),'output_device':str(y.device),'output':y.numpy().tolist()}; record['preflight_pass']=bool(physical and logical and 'GPU' in record['output_device'].upper() and record['tf32_execution_enabled']); Path('docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gpu-preflight.json').parent.mkdir(parents=True,exist_ok=True); Path('docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gpu-preflight.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(record,indent=2,sort_keys=True)); assert record['preflight_pass'], 'trusted TensorFlow GPU/XLA preflight failed'"
```

Stop if either command fails, if GPU 0 is absent, if the XLA output is not on
GPU, if TF32 is not enabled, or if the JSON is missing/invalid. `nvidia-smi`
memory is explanatory only.

## Gate B Literal Commands

Create the artifact/log directories before the first row command:

```bash
mkdir -p docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla
```

Run score-only first for a row. Inspect its terminal JSON against the score
stop rule before running the matching FD-only command. Long-command stdout and
stderr go directly to the named log so the shell exit status remains the
Python runner's; the JSON remains the canonical evidence.

### fixed-SIR

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row fixed-sir --stage score-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-score.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-fixed-sir-t1-n4-score.log 2>&1
```

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row fixed-sir --stage fd-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-fd.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-fixed-sir-t1-n4-fd.log 2>&1
```

### predator-prey

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row predator-prey --stage score-only --batch-seeds 81120 --time-steps 1 --num-particles 2 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-score.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-predator-prey-t1-n2-score.log 2>&1
```

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row predator-prey --stage fd-only --batch-seeds 81120 --time-steps 1 --num-particles 2 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/predator-prey-t1-n2-seed81120-fd.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-predator-prey-t1-n2-fd.log 2>&1
```

### actual-SV

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row actual-sv --stage score-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-score.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-actual-sv-t1-n4-score.log 2>&1
```

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row actual-sv --stage fd-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/actual-sv-t1-n4-seed81120-fd.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-actual-sv-t1-n4-fd.log 2>&1
```

### generalized-SV

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row generalized-sv --stage score-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-score.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-generalized-sv-t1-n4-score.log 2>&1
```

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row generalized-sv --stage fd-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/generalized-sv-t1-n4-seed81120-fd.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-generalized-sv-t1-n4-fd.log 2>&1
```

### KSC-SV

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row ksc-sv --stage score-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-score.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-ksc-sv-t1-n4-score.log 2>&1
```

```bash
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row ksc-sv --stage fd-only --batch-seeds 81120 --time-steps 1 --num-particles 4 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/ksc-sv-t1-n4-seed81120-fd.md > docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/gate-b-ksc-sv-t1-n4-fd.log 2>&1
```

## Gate B Time Bounds

Allow 15 minutes per tiny score or FD process. Poll the atomic JSON and log at
intervals no longer than 60 seconds. If the bound is reached, send an interrupt
to the process and require the runner's terminal `failed` JSON; if no terminal
JSON results, classify `HARNESS_TERMINAL_ARTIFACT_FAILURE` and stop all rows.

## Gate C Exact Expansion Rule

The frozen exact-command JSON contains every executable Gate C command. The
following forms and table are a human-readable cross-check only; do not execute
them by hand. Execute one JSON `shell_command` at a time.

Score form:

```text
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row <ROW> --stage score-only --batch-seeds 81120 --time-steps <T> --num-particles 10000 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/<ROW>-t<T>-n10000-seed81120-score.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/<ROW>-t<T>-n10000-seed81120-score.md
```

FD form:

```text
MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row <ROW> --stage fd-only --batch-seeds 81120 --time-steps <T> --num-particles 10000 --device-scope visible --cuda-visible-devices 0 --device /GPU:0 --expect-device-kind gpu --score-reference-json docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/<ROW>-t<T>-n10000-seed81120-score.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/<ROW>-t<T>-n10000-seed81120-fd.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-c/<ROW>-t<T>-n10000-seed81120-fd.md
```

Exact expansions:

| Row token | `T` values, in order | Parser-frozen chunks |
| --- | --- | --- |
| `fixed-sir` | `1,5,20` | `1024/1024/512` |
| `predator-prey` | `1,5,20` | `512/512/512` |
| `actual-sv` | `4,50,250,1000` | `512/512/512` |
| `generalized-sv` | `4,50,252,1008` | `512/512/512` |
| `ksc-sv` | `4,50,250,1000` | `512/512/512` |

The parser supplies the frozen transport, chunks, precision, thresholds,
source artifact, and memory budget. The emitted manifest records every value.
Before each command, create `gate-c` and a matching log path. Add shell
`> <exact-row-stage-log> 2>&1` only as logging; do not alter Python arguments.

Time bounds:

| Rung | Score bound | FD bound |
| --- | ---: | ---: |
| `T<=5` | 30 minutes | 45 minutes |
| `5<T<=50` | 90 minutes | 3 hours |
| `50<T<=252` | 6 hours | 18 hours |
| `T>252` | 30 hours | 90 hours |

Long bounds reflect admitted value compiles near 18-20 minutes and the multiple
value-only FD calls. They are resource ceilings, not expected runtimes and not
promotion criteria. Poll at intervals no longer than 60 seconds.

## Gate D Exact Expansion Rule

The frozen exact-command JSON contains every Gate D and aggregate command. The
rules below explain the expansion and are not a license for manual
substitution.

Gate D starts only after the same row's full-time Gate C seed `81120` score and
FD artifacts pass and receive a bounded result review. For seeds
`81121,81122,81123,81124`, use the Gate C score and FD forms with:

- `T` equal to the row's full `T`;
- output directory `gate-d/<ROW>`;
- `--batch-seeds <SEED>`;
- filenames containing `seed<SEED>`;
- the same row's full-time score JSON as that seed's FD reference.

Run each seed score, inspect, then its FD. Never run the five seeds in one
process. Use the full-time Gate C bounds for each Gate D process.

After all five score and FD pairs exist for one row, run exactly one offline
aggregate command using comma-separated paths in seed order `81120..81124`:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py --row <ROW> --stage aggregate --batch-seeds 81120,81121,81122,81123,81124 --time-steps <FULL_T> --num-particles 10000 --device-scope cpu --expect-device-kind cpu --score-shards <SCORE_81120>,<SCORE_81121>,<SCORE_81122>,<SCORE_81123>,<SCORE_81124> --fd-shards <FD_81120>,<FD_81121>,<FD_81122>,<FD_81123>,<FD_81124> --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-d/<ROW>/<ROW>-full-five-seed-aggregate.json --markdown-output docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-d/<ROW>/<ROW>-full-five-seed-aggregate.md
```

The aggregate command is deliberately CPU-hidden and read-only with respect to
runtime shards. It must fail if code/source hashes no longer match, any seed is
missing/duplicated/unexpected, any singleton FD fails when recomputed, the
aggregate FD fails, or the maximum per-seed peak exceeds `14000 MiB`.

## Per-Command Stop Decisions

### Score-only

Proceed to matching FD only when all hold:

- process exits `0` and terminal JSON says `artifact_status=completed`;
- `jit_compile=true`, `gpu_trust_basis` matches, precision is float32/TF32,
  physical/logical/output devices are GPU, and code/source hashes validate;
- singleton seed, row, target, theta, transport, chunks, `T`, and `N` match;
- objective, likelihood, score, and per-seed score are finite and consistent;
- reset-memory before/after fields are finite and present;
- `score_memory_budget_pass=true` and peak `<=14000 MiB`.

At Gate C/D `N=10000`, require `n10000_memory_pass=true`. A score-only failure,
nonfinite value, missing terminal artifact, wrong provenance, XLA/device error,
or over-budget peak stops that row before FD and before its next rung. Shared
harness invalidity stops all rows.

### FD-only

Proceed to the next rung only when all hold:

- process exits `0`, terminal JSON is completed, and the referenced score file
  hash matches;
- value outputs are GPU XLA float32/TF32 under the same frozen identity;
- every parameter entry is finite and internally consistent;
- recomputed singleton pass is `max_abs<=atol OR max_relative<=rtol` under the
  row's frozen thresholds.

FD failure is a repair trigger and stops that row before the next rung. Do not
change the step, tolerance, precision, target, or transport after seeing it.

### Aggregate

Full row passes only if every singleton score/FD pair and the arithmetic-mean
aggregate pass, the maximum per-seed reset score peak is within budget, and
`validate_ledh_score_artifact(..., require_admitted=True)` succeeds. A blocked
aggregate is candidate rejection or incomplete evidence, not research-direction
rejection and not evidence against another row.

## LGSSM Separate Lane

LGSSM remains on
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`; it is not
accepted by the five-row nonlinear aggregator. The existing score-only artifact
`docs/plans/ledh-lgssm-n10000-t50-compact-score-memory-2500-2026-07-10.json`
is trusted harness precedent with a `719.671 MiB` reset peak, but it was emitted
from commit `36409ad0e4b8704214b93b64588b54062e1a1cbe` and lacks same-scalar FD.
It therefore remains non-admitted.

After nonlinear Gate B review, the only frozen LGSSM follow-up initially
allowed is a separate full-row seed-`81120` FD-only diagnostic using that exact
score reference and the original frozen `2500/2500/2500` chunks, epsilon `0.5`,
step `1e-3`, atol/rtol `5e-3`, float32/TF32, full AD, compact sensitivity, and
GPU/XLA. Because the existing LGSSM reference validator is weaker than the new
nonlinear validator and no merge helper exists for split score/FD artifacts,
that FD result cannot by itself admit LGSSM. A reviewed LGSSM-specific
content/hash-binding repair is required before seeds `81121..81124` or LGSSM
aggregation. This is a stated lane limitation, not permission to relabel the
nonlinear harness.

## Evidence Interpretation

- Tiny and prefix runs are nomination/continuation screens only.
- Sub-budget memory and runtime differences are descriptive only; no row or
  configuration ranking is supported.
- Passing hard screens means viability under the screen, not superiority.
- No run in this manifest establishes posterior correctness, HMC readiness,
  exact nonlinear likelihood correctness, native actual-SV correctness for
  KSC, runtime superiority, or broad scientific validity.
