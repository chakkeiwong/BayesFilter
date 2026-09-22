# Corrected Parameter-Authority Phase 52 Result

Date: `2026-08-28`  
Version: `v3.4-fresh-paired-uncertainty-replication`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-subplan-2026-08-26.md`  
Status: `PASS_V3_4_FRESH_PAIRED_REPORT_CANDIDATE_REJECTED`

## Question and computed quantity

Phase 52 asked whether the frozen two-mode proposal geometry nominated by
Phase 51 reproducibly reduced between-bank variability relative to the frozen
isotropic-support proposal. The target remained the batch-native q=20
SSL-LSTM density in `theta in R^4`; the 60-dimensional UKF state remained
internal to target evaluation.

For metric `m`, arm `a`, and the six fresh banks, the report computed

`R_m(a) = max_i T_m(a,i) - min_i T_m(a,i)`

and `D_m = R_m(geometry) - R_m(support)`. A deterministic 20,000-replicate
bank bootstrap estimated a finite-design 95% percentile interval for each
`D_m`. The primary criterion required the upper endpoints for
`theta_mean_0`, `negative_mode_fraction`, and
`covariance_offdiag_max_abs` all to be nonpositive.

The quantity computed is a six-bank range comparison. It is not a posterior
distance, a Gaussianity test, a population ranking, or an HMC convergence
diagnostic.

## Hard-gate evidence

| Gate | Result | Evidence |
|---|---|---|
| algebra fixture | passed | `PASS_V3_4_FRESH_PAIRED_FIXTURE`; support and geometry non-symmetric MH corrections checked at beta zero and one |
| fresh inputs | passed | six distinct attempt-02 pilot receipts with root seeds `5101` through `5106` and M0 seeds `5201` through `5206` |
| target and measure | passed | target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; `theta_R4` |
| pairing | passed | all three arms in each replicate used the same initial tensor and resampling stream |
| proposal correction | passed | q remained the annealing base; each mutation arm evaluated its own proposal density at both MH endpoints |
| finite mutation | passed | all 18 arm receipts have status `PASS_V3_4_MUTATION_ARM`; no invalid candidate was accepted |
| provenance | passed | all pilot, fixture, source, Phase 50, and Phase 51 hashes verified; the two audited pilot-runner hashes differ only by one trailing blank line |
| GPU/XLA policy | passed | two RTX 4080 SUPER devices, pre-initialization memory growth, TF32, and XLA recorded |
| managed-session trust | passed | device and run manifest record `owner_designated_managed_session_visible_gpu_trusted` |
| report integrity | passed | `PASS_V3_4_FRESH_PAIRED_REPORT`; CPU intentionally hid GPUs; fixed bootstrap seed `[20260826, 52052]` |

No engineering, target, measure, numerical, device, or artifact veto fired.

## Scientific result

| Metric | Support range | Geometry range | Difference | 95% interval | Upper <= 0? |
|---|---:|---:|---:|---:|---|
| `theta_mean_0` | `1.278988` | `0.956447` | `-0.322541` | `[-0.895794, 0.182382]` | no |
| `covariance_offdiag_max_abs` | `2.222133` | `0.719168` | `-1.502965` | `[-1.620721, -0.555580]` | yes |
| `negative_mode_fraction` | `0.090887` | `0.047780` | `-0.043107` | `[-0.046140, -0.011081]` | yes |
| `root_count` | `11` | `9` | `-2` | `[-4, 6]` | no |
| `weighted_ess_fraction` | `0.220190` | `0.042712` | `-0.177478` | `[-0.181442, -0.013805]` | yes |

The primary criterion failed because the `theta_mean_0` upper endpoint is
positive. The terminal branch is therefore
`fresh_geometry_uncertainty_incompatible`. The frozen geometry is not
retained as a spread-reduction nominee.

Several descriptive findings remain useful. Geometry had lower covariance
off-diagonal values than support in all six paired rows, and its weighted ESS
fraction was higher in all six rows. Those observations do not rescue the
predeclared conjunctive criterion. A compact covariance can reflect
underdispersion, and high ESS can coexist with a biased or mode-incomplete
particle cloud. The true posterior mode mass and moment vector are not known
from this experiment.

## Failure classification

| Ledger | Verdict |
|---|---|
| engineering correctness | passed |
| numerical validity of the declared finite kernel | passed |
| current frozen candidate | failed the primary uncertainty screen |
| tuning | not evaluated; `rho=0.50`, `kappa=2.0`, equal mode weights, and inherited representatives remain hypotheses |
| broader mode-aware proposal idea | unresolved |
| target or research harness | not invalidated |

This is a candidate rejection, not evidence that all mode-aware proposals,
particle mutation, adaptive replay, ETPF, GenUT, or NeuTra are impossible.
It does show that Phase 51's three-bank descriptive pattern was not strong
enough to promote the frozen configuration.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| retain the q=20 theta target and three-arm harness | pass | no engineering veto | posterior truth remains unknown | retain as a diagnostic harness | posterior correctness |
| retain frozen geometry as a spread-reduction nominee | fail | scientific promotion veto | `theta_mean_0` interval crosses zero | reject this frozen nominee | population inferiority or impossibility |
| tune geometry on the six Phase 52 banks | forbidden | held-out-data reuse veto | tuning would overfit the claim rows | preserve banks as holdout evidence | an optimized geometry result |
| promote IID whitening or HMC | veto | no Gaussian-law, posterior-agreement, or HMC evidence | particle diagnostics are not a whitening theorem | keep NeuTra/HMC closed | IID Gaussianity or HMC readiness |
| launch another serious phase now | blocked | remaining-budget veto | remaining pool cannot fund disjoint calibration and validation | close this campaign and scope a future plan | exhaustion of the research direction |

## Inference status

| Evidence class | Status |
|---|---|
| hard veto screen | passed |
| viable candidates | the target/harness remains viable; the frozen geometry nominee does not |
| statistically supported ranking | none; the bootstrap is a predeclared six-bank finite-design diagnostic |
| descriptive-only differences | covariance, mode-mass, ESS, root, and theta-mean ranges and paired rows |
| default readiness | not ready |
| next evidence needed | independent calibration, a frozen redesigned proposal, untouched validation banks, and reference-anchored downstream agreement |

## Run manifest and budget

The successful boundary used commit `70bda6863582a563405346934cb30d4813b9ea90`
in a dirty worktree, Python `3.13.13`, TensorFlow `2.20.0`, and the
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python` environment. Memory growth was
verified on both visible GPUs before logical-device initialization. XLA and
TF32 were enabled. The report was deliberately CPU-only with
`CUDA_VISIBLE_DEVICES=-1`.

Boundary command:

```bash
TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase52_2026_08_26.py --pilot-root-1 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-01 --pilot-root-2 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-02 --pilot-root-3 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-03 --pilot-root-4 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-04 --pilot-root-5 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-05 --pilot-root-6 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-06 --fixture-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired
```

Report command:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase52_2026_08_26.py --fixture-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture --boundary-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/report
```

| Item | Wall seconds |
|---|---:|
| measured pre-boundary campaign use | `36993.01111694591` |
| ordinary-sandbox GPU-visibility diagnostic | `2.24441798` |
| trusted pilot-provenance diagnostic | `4.020615832` |
| successful boundary | `23316.68617718201` |
| report | `0.3806818010052666` |
| conservative campaign use | `60316.343009740929` |
| remaining from `64800 s` | `4483.656990259071` |

The measured Phase 51 three-bank boundary alone used `5558.9085 s`; the
successful Phase 52 six-bank boundary used `23316.6862 s`. The remaining
pool cannot fund a fresh, disjoint calibration/validation redesign. This is a
real under-budgeted next-phase blocker.

## Artifacts and hashes

- Boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired/result.json`
- Boundary SHA-256: `515abb20bd4737b684f8a77995f06628a03c976f6c767c51c0e177f54a16e83d`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/report/result.json`
- Report SHA-256: `bdacf7d0276fc9f554900d750f0fef06f68675b13e34d89ac69eb130bb0dd7f6`
- Boundary runner SHA-256: `3c8ca24aef245bd2f831008a144443ee3c48a4caa4365b3b8c0d5b4e9d25fca3`
- Reporter SHA-256: `8124dae6918ac8323c5cc10cb5468004e57058210a0f2285007c17e84de8bab5`
- Parent-plan SHA-256 at execution: `3f455a6504d77c50aa4447219194b9b884d12d34092e74eccecb9e343d7d4cff`

The machine-readable report is authoritative for exact rows, tensor receipts,
source hashes, intervals, and nonclaims. The parent plan is updated only after
the terminal report, so its later closeout hash is not retroactively substituted
for the execution-time hash.

## Red-team note

The strongest alternative explanation is that local two-mode covariance
improved finite proposal overlap and weight balance while equal mode weights,
inherited representatives, and local curvature failed to represent the global
target. The weakest evidence is the six-bank range estimand, which is
tail-sensitive and cannot support a stable population ranking. Evidence that
could overturn this conclusion would require a separately budgeted protocol
with disjoint calibration and validation banks plus downstream agreement to an
independent target-level reference.

No new MathDevMCP invocation was needed for this empirical phase. The passing
algebra fixture checks the implemented finite MH identity; it is not a proof
of posterior accuracy, exhaustive mode exploration, or IID whitening.

