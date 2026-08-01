# Zhao-Cui Coupled Nonlinear TTSIRT-APF Rung-2 Result

Date: 2026-07-22

## Decision

`PASS_ENGINEERING_RUNG2` for the bounded synthetic `d=24,T=3,N=512` GPU/XLA
scope. The assembled route remains `extension_or_invention`, not a
source-faithful Zhao-Cui implementation.

The candidate is the frozen fitted-TTSIRT conditional with a reference-parameter
weight-aware predictive auxiliary genealogy. The uniform fitted arm remains an
explanatory comparator and failed the high-dimensional ESS screen.

## Evidence Contract

| Item | Contract and result |
| --- | --- |
| Question | Can a coupled nonlinear 2D block proposal compose to a finite, non-collapsed `d=24` fixed-branch APF with an analytical score of the same scalar? |
| Exact comparator | Exact Gaussian conditional with predictive and uniform auxiliary arms, sharing observations, particles, random uniforms, dtype, and genealogy scope. |
| Candidate | Gaussian-quantile coordinate map, deterministic Legendre projection TT-SVD, L1-aware density refinement, paired-core retained marginal, frozen predictive auxiliary, uniform online candidate proposal density correction. |
| Primary promotion screen | Candidate predictive ESS fraction `>=0.20`; passed at `0.2545734942`. |
| Veto screens | Conditional log-density RMS `<=0.75`; same-scalar score/FD max error `<=0.05`; cross-order log-normalizer spread `<=0.05`; finite values; inverse roundtrip `<=1e-4`; GPU placement and memory growth. All passed. |
| Explanatory only | Runtime, compile time, log-weight spread, exact-arm comparisons, q95/q99/max tails, and one-seed continuous metrics. No ranking or superiority claim. |
| Artifact | `docs/benchmarks/artifacts/zhao_cui_coupled_nonlinear_ttsirt_apf_rung2_20260722/gpu_attempt01/result.json` and `result.md`. |

## Selected Arm

| Field | Value |
| --- | --- |
| Scope | `d=24`, 12 replicated 2D blocks, `T=3`, `N=512` |
| Degree / TT rank cap / realized ranks | `6` / `12` / initial `(1,7,1)`, adjacent `(1,7,12,7,1)` |
| Coordinate map | Gaussian-quantile, scale `0.22`, location `(0.8,0.2)` per block |
| L1 selection | `0.0` selected; positive arms `1e-6` and `1e-5` were evaluated and did not meet the required `0.005` validation-KL improvement margin |
| Validation KL maximum | `0.0026409190` |
| Untouched audit KL maximum | `0.0049578141` |
| Cross-order log-normalizer spread | `0.0085649892` |
| Conditional log-density RMS | `0.3014470436` |
| Candidate predictive ESS fraction | `0.2545734942` |
| Candidate score/FD max error | `0.0001934841` on GPU (`0.0002686977` CPU) |
| Inverse roundtrip max error | `2.9788e-6` |
| GPU allocator current / peak | `1,111,552` / `1,741,981,952` bytes |
| Wall time | `119.63 s` GPU; `114.69 s` CPU precheck |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep v3 candidate as a viable engineering arm | ESS `0.2546 >= 0.20` | All declared hard screens passed | One seed and short horizon; predictive auxiliary frozen at reference theta | Multi-seed and longer-horizon rung under a fresh scope | No statistical superiority, posterior correctness, or production readiness |
| Reject v1 algebraic/random-core arm | ESS `0.00195`, conditional RMS `1707.8` | Candidate veto | Tail mismatch versus optimization contribution | Preserve as failed evidence; do not reuse settings | Not a rejection of the research direction |
| Reject v2 uniform-genealogy arm | ESS `0.1166` | Candidate ESS veto; other screens passed | Genealogy/auxiliary mismatch | Use predictive auxiliary repair, now v3 | No claim that predictive arm is statistically better |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for v3; v1 and v2 failures preserved as candidate evidence |
| Statistically supported ranking | None; one seed and no uncertainty interval |
| Descriptive-only differences | Predictive vs uniform ESS, runtime, log-weight spread, and arm values |
| Default-readiness | Not assessed; this is an optional synthetic extension arm |
| Next evidence needed | Multi-seed replication, longer `T`, scope-specific retuning, and a valid extended-state/pseudo-marginal treatment before HMC claims |

## Source Classification

Checked Zhao-Cui paper and pinned author source anchors:

- Eq. 13 squared-TT defensive density: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539`; `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/SIRT.m:74-85`.
- Proposition 2 / KR and paired-core marginal: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:592-670`; `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:19-85`; `eval_cirt_reference.m:43-100`.
- Frozen source sampling/settings: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:890-924`; `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-43`.

The following are local extensions: synthetic nonlinear model, Gaussian-
quantile map, reordered `(previous,current)` compiler, fixed quadrature
TT-SVD initializer, numerical grid inverse, predictive auxiliary genealogy,
block composition, APF scalar, and analytical score wiring. They are all
classified `extension_or_invention`. The squared-TT and paired-core operations
are source-faithful operations only; they do not upgrade the assembled route.

## Failure And Repair Ledger

| Attempt | Result | Classification | Repair |
| --- | --- | --- | --- |
| v1 CPU `attempt01` | Conditional RMS `1707.8`, ESS `0.00195`; algebraic map sampled extreme tails | Candidate failure, not harness failure | Replace algebraic reference-tail route and random-core initialization |
| v2 CPU `attempt02` | Proposal fit repaired; uniform fitted ESS `0.1166` | Candidate auxiliary failure | Add frozen reference-parameter predictive auxiliary genealogy |
| v3 CPU `attempt03` | All CPU screens passed; predictive ESS `0.2546` | Viable engineering candidate | GPU/XLA claim run |
| v3 GPU `attempt01` | All screens passed on `/GPU:0` with verified growth | Engineering rung pass | Fresh multi-seed/longer-horizon scope, not default promotion |

## GPU Policy Evidence

The GPU artifact records:

- `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import;
- `configured_before_logical_device_initialization=true`;
- `/physical_device:GPU:0` growth verified `true`;
- no whole-device preallocation;
- TF32 enabled and XLA compiled on `/GPU:0`;
- allocator current/peak bytes.

The escalated pre-launch occupancy gate observed the shared RTX 4080 SUPER at
`2,497 MiB` used, `13,549 MiB` free, and `36%` GPU utilization, with no process
reported by `nvidia-smi pmon`. A post-run escalated release check observed
`2,497 MiB` used, `13,549 MiB` free, and `26%` utilization, again with no
process reported by `pmon`. The occupancy snapshots are recorded here; they
were not serialized into the run's `result.json`.

Memory growth is not a hard cap. The run used the reviewed shared-device
occupancy gate and did not configure a logical-device memory limit.

## Dirty-Worktree Provenance

The run manifest binds Git commit
`ee346978741cf306167bb5a3e11c8aded506d593` and records a dirty worktree. The
candidate files were untracked at run time, so the handoff supplied and this
closeout rechecked these SHA-256 values before editing the plan text:

| Artifact-time file | SHA-256 |
| --- | --- |
| `bayesfilter/highdim/zhao_cui_coupled_nonlinear.py` | `8f9dfedc203e7ad5d87763ad4413ab444d63ace5361877bec96ae6195b25d0fe` |
| `docs/benchmarks/run_zhao_cui_coupled_nonlinear_ttsirt_apf_rung2.py` | `c799bc0fd77e311a096c344644127263871eb0f060f94bc19e47702b0bcd2901` |
| `tests/highdim/test_zhao_cui_coupled_nonlinear.py` | `f6946cadbfb2c464364c022eaa7b102bf50ba38c7d92429885a78dffb96fa437` |
| Pre-closeout plan used by the run | `78cbdf2b21bef875f1bcc8121b60e41b5e23b9693eb1cada93e7faed58da0332` |

The plan hash is intentionally the pre-closeout value. Its current content
differs only by terminal-status and evidence-consistency corrections made
after the GPU artifact was written; the model, runner, and focused test were
not modified during closeout.

## Nonclaims And Open Risks

- No source-faithful Zhao-Cui filter claim.
- No Austria SIR, NAWM, HMC convergence, posterior correctness, or production KR claim.
- No exact unbiased randomized-estimator or pseudo-marginal claim; the branch is frozen and defines a deterministic approximate finite target.
- No statistical ranking or superiority claim from one seed.
- No cross-block dependence claim; the 24D target is twelve replicated coupled 2D blocks sharing genealogy.
- Predictive auxiliary construction is frozen at the reference parameter and may require refresh or an extended-state treatment for parameter exploration.
- The numerical KR closure remains a diagnostic grid-CDF/bisection route, not production closure.

## Verification

Focused terminal suite: `26 passed` across the coupled model, frozen APF
compiler, source-route slice, and TensorFlow memory-policy tests. Static Python
compile and `git diff --check` passed. Existing unrelated worktree changes were
preserved.
