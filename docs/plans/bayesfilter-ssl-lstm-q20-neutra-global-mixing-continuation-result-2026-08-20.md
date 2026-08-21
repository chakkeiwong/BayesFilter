# SSL-LSTM q=20 NeuTra global-mixing continuation result (2026-08-20)

Plan: docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-plan-2026-08-20.md

Status: TERMINAL_HMC_NO_CANDIDATE_ADMITTED_NO_PREDICTIVE

## Outcome

The fresh bounded HMC continuation completed cleanly on the trusted GPU. Both
predeclared frozen transport/kernel pairs reached complete four-chain,
2,000-draw-per-chain fixed-kernel verification, and both failed the required
maximum raw-target-coordinate rank-normalized split/folded R-hat gate of
<= 1.01. The terminal HMC artifact reports HMC_NO_CANDIDATE_ADMITTED and
NO_HMC_CANDIDATE_ADMITTED.

This is a candidate/kernel-pair rejection, not a target, harness, frozen
transport, or NeuTra research-direction rejection. The canonical sequential
controller was not entered because it requires an admitted fixed kernel.
Predictive validation was correctly not run and no predictive artifact root was
created.

## Claimed target and quantity computed

| Item | Verdict |
|---|---|
| Claimed target | Retained draws from one frozen target-specific weighted dense-IAF transport and one common exact fixed-HMC kernel that passes tuning verification and then bayesfilter_neutra_sequential_hmc_v1, including retained parameter/sign R-hat, ESS, and direct per-chain cross-sign-transition gates. |
| Quantity actually computed | Exact transformed-target fixed-HMC tuning and full raw-coordinate verification for seed 2 at L=5 and seed 3 at L=3, each with four initialized chains, 64 discarded verification burn-in steps, and 2,000 verification draws per chain. |
| Relation | The computed quantities are prerequisite candidate-verification evidence, not the claimed canonical sequential posterior. Both prerequisite gates failed, so neither is eligible for posterior or predictive use. |
| Target/engineering identity | The launch artifact records immutable prior evidence checks, target and adapter identities, direct canonical route classification, float64 XLA execution, TF32 disabled, and verified memory growth before logical-GPU initialization. |
| Not established | Sequential warm-up readiness, retained R-hat/ESS, per-chain cross-sign transitions, posterior correctness, mode weights, predictive equality, sampler or transport superiority, model adequacy, scientific validity, production readiness, or default readiness. |

## Candidate verification results

| Frozen pair | Verification evidence | Hard gate | Decision |
|---|---|---|---|
| Transport seed 2, L=5 | Step 0.2460072308515237; mean acceptance probability 0.7339184979522049; finite samples, target values, scores, and target statuses; rank R-hat max 1.0786399742045318; folded rank R-hat max 1.0875996310350042. | Folded R-hat for the observation-weight coordinate exceeds 1.01. | TUNING_NO_VIABLE_KERNEL; reject this exact pair. |
| Transport seed 3, L=3 | Step 0.2627342396519737; mean acceptance probability 0.7706182130120045; finite samples, target values, scores, and target statuses; rank R-hat max 1.0920614707067127; folded rank R-hat max 1.1020661342469682. | Folded R-hat for the observation-weight coordinate exceeds 1.01. | TUNING_NO_VIABLE_KERNEL; reject this exact pair. |

The acceptance values are explanatory only and do not override failed R-hat.
Both verification traces recorded a finite max_abs_log_accept_ratio proxy of
1e100 and raised the configured explanatory alert. The TFP kernel did not
expose a native divergence count, so this result does not claim zero
divergences. No candidate ranking is statistically supported: there were only
two ordered viability attempts and both were rejected by the hard screen.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit seed 2, L=5 | Failed | Modern folded R-hat 1.0875996310350042 > 1.01. | Whether a changed, independently reviewed kernel/tuning protocol would mix the slow coordinate. | Preserve this pair as failed verification evidence. | Seed-2 transport failure or NeuTra failure. |
| Admit seed 3, L=3 | Failed | Modern folded R-hat 1.1020661342469682 > 1.01. | Same; no broad transport comparison exists. | Preserve this pair as failed verification evidence. | Seed-3 transport failure, relative ranking, or direction failure. |
| Run canonical sequential HMC | Not eligible | No selected kernel exists. | Sequential behavior is unobserved. | Require a fresh admitted kernel in a new campaign. | Warm-up, ESS, or cross-sign traversal passed. |
| Run predictive endpoint | Not eligible | HMC status is not HMC_ADMITTED_FOR_PREDICTIVE. | Posterior predictive behavior is unobserved. | Keep predictive work closed until a future HMC admission. | Equality, equivalence, adequacy, or any horizon decision. |

## Inference status

| Status row | Finding |
|---|---|
| Hard veto screen | Supported for both pairs: full raw-coordinate modern rank/folded R-hat verification failed. |
| Viable candidates | None admitted in this campaign. |
| Statistically supported ranking | None. The ordered schedule was an affordability policy, not a comparison, and both pairs failed hard gates. |
| Descriptive-only differences | Acceptance, step size, runtime, and R-hat magnitude differ between attempts but do not support a ranking. |
| Default readiness | Not established. No posterior bank exists and neither candidate passed its promotion screen. |
| Next evidence needed | A new reviewed target-specific protocol must establish a different answer-bearing candidate path and then pass the canonical sequential evidence gates. |

## Separate evidence ledgers

### Engineering correctness

The continuation used the checked immutable r1 training and HMC graphs and a
new runner with the active route-ledger entry bound directly to
bayesfilter_neutra_sequential_hmc_v1. The launch and manifest record one
visible logical GPU selected from physical GPU 1, TF_FORCE_GPU_ALLOW_GROWTH=true,
memory growth applied before logical-device initialization, float64, XLA JIT,
and TF32 disabled. The fresh HMC root was new, and its SHA-256 inventory has 15
entries with zero rehash mismatches.

The result is sufficient to say the executed candidate-verification harness
completed under its declared engineering conditions. It is not evidence that
the conditional post-selection consumer parity check passed, because no kernel
was selected and that condition was not reached.

### Sampler validity

Both candidate verifications had finite state, target, score, log-acceptance,
and target-status telemetry, but each failed the required modern
rank-normalized split/folded R-hat screen. Therefore each candidate is rejected
before mechanics or canonical sequential sampling. No claim about native
divergence rate can be made because that TFP kernel did not expose it; the
finite proxy alert is explanatory only.

### Scientific interpretation

The scientific question remains open. This campaign gives direct evidence that
the two exact frozen transport/kernel pairs did not meet the specified
common-kernel verification threshold from the chosen initialized chains. It
does not show that the posterior lacks global mixing under another exact kernel,
that either transport is globally inadequate, or that the SSL-LSTM target is
wrong. The two candidate rejections are repair triggers for a future targeted
investigation, not evidence to abandon the research direction.

## Run manifest and provenance

| Field | Value |
|---|---|
| Command | TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 timeout 61200s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py --device 1 --incremental-campaign-cap-seconds 64800 --predictive-reserve-seconds 3600 --time-cap-seconds 61020 --training-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen --prior-hmc-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc |
| Environment | tfgpu, Python 3.13.13, Git commit 5699dafec23de9549a8092bec638997e7973593c, dirty worktree preserved. |
| GPU execution | Requested physical selector 1; one logical GPU visible; trusted basis owner_designated_managed_session_visible_gpu_trusted. |
| Numerical mode | float64, XLA enabled, TF32 disabled, TensorFlow GPU memory growth verified before logical initialization. |
| Fresh campaign budget | 64,800 s; HMC external/internal caps 61,200 s/61,020 s; predictive reserve 3,600 s was not borrowed. |
| HMC wall | 23,280.603976539 s (6.467 h), completed below both HMC caps. |
| Completion | 2026-08-20T01:56:44.208987+00:00. |
| Launch-bound plan SHA-256 | 8dfaaab35c4f62bdb2d92e22dd55f63fda83d9e6799ab7c277b5ca1667a8559a. |
| Runner SHA-256 | eeef1880cb26a7649ccf76230b909518fa1ca4a3e94e3bbc35e38de654d57723. |
| HMC result SHA-256 | 9092b82d25f8e63d1708c63c7d48284ef3c55a5edc3e225833ba505f0b65e706. |
| HMC manifest SHA-256 | c662e56370fe6dd111916232dfede31ac169ce214d03a95431583fe9ca6a92d6. |
| HMC inventory SHA-256 | d874cf937fe6b7cca69be6c3aa0274ad3ca3ba80a0d409a5a943370425f5a14e. |

The candidate receipts, tuning results, launch receipt, manifest, terminal
result, and inventory are under
docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc.

## Verification

- The HMC process exited normally with the terminal status above; it did not
  hit the shell timeout or an internal resource stop.
- The fresh HMC inventory contained 15 files and every SHA-256 entry was
  recomputed after completion with zero mismatches.
- The expected r2/predictive directory is absent, consistent with the
  predeclared HMC admission gate.
- Post-run CPU-hidden focused suite: 48 passed across the shared
  fixed-transport tuner and SSL-LSTM q=20 continuation tests. The 188 warnings
  are TensorFlow Probability/Gast deprecations, not test failures.
- Prelaunch focused tests passed: 20 fixed-transport-tuner tests, 34 route and
  q=20 tests, and 106 tests in the expanded focused set. Two unrelated broad
  test failures were preserved as pre-existing absence of the ignored P0
  registry; they did not exercise this route.

## Failure classification

| Class | Status | Evidence |
|---|---|---|
| Harness failure | No | The process completed and emitted internally consistent terminal receipts. |
| Immutable-input or artifact failure | No | Required prior identities validated; 15 r2 inventory entries rehash correctly. |
| Environment failure | No | The manifest records the required GPU, XLA, dtype, TF32, and memory-growth conditions. |
| Resource failure | No | Neither candidate was refused and the process finished below the HMC cap. |
| Candidate/kernel failure | Yes | Each full verification failed the specified modern folded R-hat threshold. |
| Target or research-direction failure | No | Only two scoped frozen pair configurations were adjudicated. |
| Predictive failure | Not applicable | Predictive was not authorized because HMC admission did not occur. |

## Post-run red team and resume boundary

The strongest alternative explanation for the shared observation-weight R-hat
failure is slow mixing that might respond to a materially different fixed-kernel
or target-specific tuning protocol. That explanation cannot make either failed
<= 1.01 gate pass. Another explanation is that 2,000 verification draws are
insufficient for the slow coordinate, but the fixed protocol explicitly made
that check a promotion boundary rather than an estimation of an asymptotic
claim.

Evidence that would overturn the terminal non-admission is a fresh reviewed
campaign in a new output root that establishes an answer-bearing candidate and
then passes all canonical sequential gates. The weakest evidence here is
breadth: only two new frozen pairs were tested and neither reached sequential
sampling. The strongest evidence is the clean classification: both declared
candidate verifications completed and failed their actual hard gate, while all
unreached phases remain correctly labeled unrun.

Do not rerun into the r2 root, reinterpret its unused wall allowance as an
automatic retry, use either failed pair as a posterior source, or create a
predictive artifact. A future continuation needs its own reviewed evidence
contract, target-specific tuning/default audit, budget, output root, and a
deliberate decision about the changed candidate mechanism.
