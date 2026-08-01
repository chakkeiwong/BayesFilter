# Phase 9 Gate B Result: Nonlinear Trusted GPU/XLA Preflight

Date: 2026-07-11

Status: `GATE_B_REVIEWED_FOUR_ROWS_AUTHORIZED_FOR_GATE_C`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Fixed-SIR, actual-SV, generalized-SV, and KSC-SV remain viable after the tiny trusted GPU/XLA score-and-FD screen. Predator-prey is rejected from the current ladder because its frozen production-precision FD screen failed. | Four rows passed terminal status, trusted GPU/XLA/TF32 provenance, finite compact score, reset-memory presence, prepared-input identity, and the frozen absolute-or-relative FD rule. Predator-prey passed score execution but failed FD. | No shared continuation veto fired. Predator-prey has a row-local correctness veto: `max_abs=0.3162194490 > 0.005` and `max_relative=1.0 > 0.005`. | Obtain a bounded review of this Gate B result. If it agrees, Gate C may begin only for the four passing rows, one exact prefix command at a time. | No full-row score admission, `N=10000` memory result, runtime or statistical ranking, HMC readiness, posterior correctness, exact nonlinear likelihood correctness, native actual-SV correctness for KSC, or scientific superiority. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can each nonlinear compact score and its same-scalar value-only FD comparator execute at the frozen tiny fixture under trusted GPU/XLA/TF32 with valid provenance and terminal artifacts? |
| Exact comparator | Central finite differences of the row-matched value-only scalar using the same prepared tensor fingerprint, score-reference hash, target, theta coordinates, transport settings, precision, and fixed seed. |
| Primary criterion | Passed for fixed-SIR, actual-SV, generalized-SV, and KSC-SV. Failed for predator-prey. |
| Promotion vetoes | Predator-prey triggered the frozen FD veto. No row triggered nonfinite, wrong-device, wrong-trust, wrong-source, memory, prepared-input, or terminal-artifact veto in the final common-identity run. |
| Continuation vetoes | None. Earlier graph-extraction defects were bounded, repaired, reviewed, and covered by all-row XLA tests. The final predator-prey failure is row-local and does not invalidate unrelated rows. |
| Explanatory only | Tiny reset peaks, compile/call times, objectives, score magnitudes, and sub-threshold error differences. |
| Artifact | This result plus the ten final live Gate B JSON shards, Markdown summaries, logs, archived failed attempts, and reviewed repair artifacts. |

## Final Row Results

All final shards use seed `81120`, `float32`, TF32 enabled,
`jit_compile=True`, logical `/GPU:0`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`. Gate B peaks are tiny
compile/preflight diagnostics and are not `N=10000` memory evidence.

| Row | Tiny shape | Score | Reset peak MiB | FD max abs / atol | FD max rel / rtol | Frozen rule | Gate B decision |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| fixed-SIR | `T=1,N=4` | `[-9.4539299, 3.5755198, 5.4389696]` | `80.0473633` | `0.1514778 / 0.01` | `0.0170332 / 0.05` | Pass by relative tolerance | Viable for reviewed Gate C |
| predator-prey | `T=1,N=2` | `[-183.5921021, -1.3080407, -0.1401510, 26.5619144, 8.4396877, -11.9722939]` | `0.0375977` | `0.3162194 / 0.005` | `1.0 / 0.005` | Fail both branches | Rejected from current ladder |
| actual-SV | `T=1,N=4` | `[-0.0602190, -0.4003552]` | `0.0422363` | `0.0011738 / 0.005` | `0.0191198 / 0.005` | Pass by absolute tolerance | Viable for reviewed Gate C |
| generalized-SV | `T=1,N=4` | `[0.0217674, 0.0305396, -0.0365843]` | `0.0415039` | `0.0009666 / 0.005` | `0.0416118 / 0.005` | Pass by absolute tolerance | Viable for reviewed Gate C |
| KSC-SV | `T=1,N=4` | `[-0.0469415, -0.4008480]` | `0.0412598` | `0.0008873 / 0.005` | `0.0031044 / 0.005` | Pass both branches | Viable for reviewed Gate C |

Passing a branch of the declared OR rule is a pass; it is not a claim that the
other error measure is favorable. No thresholds were changed after observing
results.

## Claimed And Computed Quantities

| Row | Claimed target | Quantity actually computed | Relationship and verdict |
| --- | --- | --- | --- |
| fixed-SIR | Score of the realized finite-`N` fixed-SIR LEDH scalar in log-scale coordinates. | Compact forward sensitivity and float32 central FD of the same prepared-input value scalar. | Relative error passes the frozen rule; viable at Gate B. |
| predator-prey | Score of the realized finite-`N` additive-Gaussian predator-prey LEDH scalar in physical coordinates. | Compact forward sensitivity and float32 central FD at step `1e-4` of the same prepared-input value scalar. | They differ beyond both frozen tolerances. The current candidate fails. Whether the main cause is score error, float32 FD resolution, or their interaction is not established. |
| actual-SV | Score of the transformed `log(y^2)` actual-SV scalar. | Compact forward sensitivity and same-scalar central FD. | Absolute error passes. This is not an exact native actual-SV likelihood result. |
| generalized-SV | Score of the raw source-route prior-mean generalized-SV scalar. | Compact forward sensitivity and same-scalar central FD. | Absolute error passes. |
| KSC-SV | Score of the KSC log-chi-square Gaussian-mixture surrogate scalar. | Compact forward sensitivity and same-surrogate central FD. | Both error branches pass. This is not native actual-SV likelihood evidence. |

## Predator-Prey Veto Detail

| Parameter | Score | Finite difference | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| `r` | `-183.5921021` | `-183.4869385` | `0.1051636` | `0.0005728` |
| `K` | `-1.3080407` | `-0.9918213` | `0.3162194` | `0.2417505` |
| `a` | `-0.1401510` | `0.0` | `0.1401510` | `1.0` |
| `s` | `26.5619144` | `26.5884399` | `0.0265255` | `0.0009976` |
| `u` | `8.4396877` | `8.2778931` | `0.1617947` | `0.0191707` |
| `v` | `-11.9722939` | `-11.8255615` | `0.1467323` | `0.0122560` |

The `a` perturbation producing exactly zero FD at float32 step `1e-4` makes
finite-difference resolution a plausible alternative explanation, but that is
explanatory only. The reviewed criterion still fails. A different step or
precision arm would require a revised, predeclared diagnostic plan and could
not retroactively pass this Gate B candidate.

## Engineering Correctness Ledger

- Trusted `nvidia-smi` and TensorFlow/XLA preflight passed on NVIDIA GeForce RTX
  4080 SUPER, driver `591.86`, TensorFlow `2.19.1`, TF32 enabled, output on
  `/GPU:0`.
- The final five score shards are terminal `completed` artifacts and pass the
  runner's own raw-score validator.
- Fixed-SIR, actual-SV, generalized-SV, and KSC-SV FD shards are terminal
  `completed` artifacts and pass the runner's own raw-FD validator.
- Predator-prey is a terminal `failed_fd` artifact and is intentionally
  rejected by the raw-FD validator.
- Each FD shard references the exact SHA-256 of its score file and has the same
  prepared-input fingerprint.
- All final live shards bind cross-row review SHA-256
  `efa1f5300f87673223a7080767ce0f23586d662a9342427928db94ddc7e1739b`.
- Earlier fixed-SIR and predator-prey graph-extraction failures are preserved in
  separate attempt directories. No archived shard is mixed into the final set.

## Numerical Validity Ledger

- Fixed-SIR, actual-SV, generalized-SV, and KSC-SV pass the predeclared
  same-scalar FD screen.
- Predator-prey fails the predeclared same-scalar FD screen and cannot proceed
  to the current prefix ladder.
- All scores and FD values are finite and execute on GPU/XLA at the tiny shape.
- No `N=10000` prefix or full-time numerical validity result exists yet.

## Scientific Interpretation Ledger

- Gate B is engineering and tiny numerical-screen evidence, not posterior or
  scientific validation.
- The predator-prey candidate failure does not reject the compact-score research
  direction and does not invalidate fixed-SIR or any SV row.
- The four passing rows are viable candidates for the next reviewed screen; they
  are not ranked and are not established as superior.
- The actual-SV and KSC-SV rows compute different explicitly named targets and
  must not be conflated.

## Artifact Hashes

| Row | Score JSON SHA-256 | FD JSON SHA-256 |
| --- | --- | --- |
| fixed-SIR | `0f8a71afdce3571e952c061f036efd66923c08c42392b08999d1fb72c7631f18` | `1e1c3ceeb9e793298903b6853db463abfca13d41e73fb0126037807cdfe759be` |
| predator-prey | `82eb75a8710a6c4219419b5f9c14f670e371554a3c7943a2a3fb5e03f1c28f5c` | `738c59f9967ec86dfc09be7bfb315e4cc9fdfc04a22cec95292527405f1b3127` |
| actual-SV | `ad697c4bd06749cab022281b128b772e61286c0c8a506de623e49b0f03ea6e55` | `c5cda3c38e05564aa011168353b7f519bb1ce676b02127401f2fa618c41c6a92` |
| generalized-SV | `db0891693d55faa77e0dac689b26bf8d8394d802dd724847ff5ba3849a27a7d8` | `4fa4f8280b5d0f2d4ced2f8c048a79c0f358d16ab41874fd22d0020ec5a327ce` |
| KSC-SV | `832d183455de798c6794594a87e4c6b5630dc36201b7000e399342b5077cdbf6` | `2fc17704717383352a06362a951b7d3b8b38c1c06a9aa7d2cf712d0e2c04b641` |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the recorded dirty worktree and scoped repairs |
| Commands | The ten literal `gate_b_commands` in `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json`; score then FD per row |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; Python `3.11.14`; TensorFlow `2.19.1` |
| CPU/GPU status | Trusted GPU 0; NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; `16376 MiB`; TF32 enabled; XLA JIT; managed-session trust basis |
| Data version | Five admitted nonlinear source value artifacts dated 2026-07-07, each SHA-256 bound in its shard |
| Random seeds | Singleton seed `81120` for every Gate B process |
| Wall time | Final artifact elapsed times: score `18.37/23.25/15.05/14.89/14.40s`; FD `11.97/9.76/9.64/8.91/9.33s` in row order |
| Output artifacts | Ten live Gate B JSON shards, ten Markdown summaries, ten logs, preflight JSON/log, and archived repair attempts under the Phase 9 artifact roots |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

The preflight `nvidia-smi` snapshot reported `2833 MiB` used. That process-level
number is explanatory only and is not a score-memory criterion.

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Predator-prey fails FD. No hard veto is supported for the other four rows at Gate B. |
| Statistically supported ranking | None. One tiny fixed-seed screen cannot support a ranking. |
| Descriptive-only differences | Scores, objectives, compile times, tiny peaks, and sub-threshold FD differences are descriptive only. |
| Default-readiness | No new default-readiness conclusion. Gate B is insufficient for full-row or production admission. |
| Next evidence needed | Reviewed Gate C `N=10000` prefix ladders for the four passing rows; a separately reviewed diagnostic plan if predator-prey FD resolution or score correctness is to be investigated. |

## Post-Run Red Team

- Strongest alternative explanation: predator-prey's failure may be dominated
  by float32 central-difference cancellation at step `1e-4`, especially for
  `a`, rather than an incorrect compact recurrence. Current evidence cannot
  distinguish those causes.
- Result that would overturn the row decision: a predeclared, reviewed
  production-relevant correctness check that resolves the discrepancy without
  changing the claimed scalar. It would not retroactively change this run.
- Weakest evidence for passing rows: one seed and a tiny shape; scale-dependent
  XLA, memory, and numerical failures remain possible at `N=10000` and longer
  `T`.

## Review Boundary

The first Gate B result review returned `VERDICT: REVISE` because future Gate C
shards would not bind this result or its authorizing review. The runner now
records and validates both paths and SHA-256 hashes; independent path/hash
mutations are rejected. Fresh iteration-2 re-review returned `VERDICT: AGREE`.

Gate C is authorized only for fixed-SIR, actual-SV, generalized-SV, and KSC-SV,
in ascending prefix order with score-before-FD stop checks. Predator-prey, Gate
D, aggregation, and LGSSM remain blocked. A full-time Gate C pass still requires
a separate row-result review before Gate D.
