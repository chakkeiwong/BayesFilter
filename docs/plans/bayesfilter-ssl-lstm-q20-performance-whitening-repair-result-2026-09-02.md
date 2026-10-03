# SSL-LSTM q=20 performance and transport-score repair result

Date: 2026-09-02  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-plan-2026-09-02.md`  
Status: `CLOSED_WITHOUT_CORE_REPAIR`

## Outcome

The bounded diagnostic completed successfully on GPU 0 in 131.399 seconds.
It establishes three separate facts. First, the q=20 target has a useful
batching opportunity. Second, the target value/score program is internally
consistent on the checks performed. Third, a fresh reverse-KL chart is still
very far from Gaussianization after eight updates. The result does not justify
retrying the terminal M3 replay or changing the active HMC route.

## Evidence and commands

The claim-bearing execution receipt is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-01-gpu/run_manifest.json`
with manifest hash
`aeabd201ba7b03f2d986c6920c68ad35d8f023b215ffcf068e82e0e40cc9e340`.
The repaired CPU analytic receipt is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-03-cpu/run_manifest.json`
with manifest hash
`0d1e45cc3806dd0e5b15f940fbda1df5e8d218bf8367fbb437157cdf6158f608`.
The final post-closeout CPU smoke receipt is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-04-cpu/run_manifest.json`
with manifest hash
`b36d762b63819948ebd602c447a5fdf3507d981681cea4d4d75cf85ece4b46fa`.

GPU command:

```text
BAYESFILTER_PERF_WHITENING_ATTEMPT_ID=attempt-01 \
bash scripts/run_ssl_lstm_q20_performance_whitening_diagnostic_gpu.sh \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-repair-2026-09-02/attempt-01-gpu
```

The run used commit `54201f5cd925ed15036bad8156606b812d53b045`, TensorFlow
2.20.0, float64 strict `tensorflow_eigh`, TF32 enabled, one visible RTX 4080
SUPER, and verified memory growth. The source hashes and full environment are
in the manifest.

## Phase results

### E0/R0: source and evidence audit

The terminal M3 canary remains the governing historical result and was not
resumed. The active route scan passed with no occurrences of
`tf.map_fn`, `tf.vectorized_map`, `GradientTape.jacobian`,
`GradientTape.batch_jacobian`, or `pfor` in the four inspected runtime files.
There is no `ThreadPoolExecutor`, `ProcessPoolExecutor`, or distribution
strategy in the active candidate/HMC path. Four HMC chains are represented as a
batched tensor on the selected GPU; candidate, chart, and scope loops are
serial Python orchestration. Focused tests passed: `92 passed` in 40.36 s.

The diagnostic harness itself had two localized repairs before the clean
receipt: its repository import path was made explicit, and its analytic grouped
fixture was changed to cache one compiled graph per row count instead of
creating a `tf.function` inside each call. This distinction is important: the
fixture defect was repaired; the production bridge's finite static batch-size
specialization remains an intentional design choice.

### E1/R1: performance and grouping

The target timing stage evaluated equal total row counts after warm-up:

| Call shape | Rows | Steady total (s) | Seconds/row | Trace receipt |
|---|---:|---:|---:|---|
| batch 8 | 16 | 2.4897 | 0.1556 | 1 |
| batch 16 | 32 | 3.0031 | 0.09385 | 1 |
| batch 32 | 64 | 4.0608 | 0.06345 | 1 |
| serial batch 4 | 32 | 8.0508 | 0.2516 | static specialization |

For the equal 32-row comparison, batch 16 is 2.68 times faster in total and
batch 32 is 3.97 times faster per row than the serial batch-4 calls. These are
descriptive timings from one bounded run, not a superiority claim. The finite
trace counts show bounded static specialization. The TensorFlow retracing
warning is explained by the intentional sequence of batch sizes (and the
underlying component cache has the same policy); it is a performance concern to
profile, not evidence that every surrounding operation should be XLA compiled.

The analytic grouped-HMC fixture produced finite, moving samples and one trace
per row-count graph. On GPU, grouped elapsed time was 2.7070 s versus 2.6476 s
for the serial fixture. The grouped call used a different random-stream and
candidate partition, so samples, accept/reject states, and target-call
accounting were not an equivalence receipt. The measured 2.2 percent slowdown
in this tiny fixture is therefore diagnostic only. `integration_allowed` is
false.

### E2/R2: score authority and fresh chart

The exact affine Gaussian chart gave centered log-density RMS
`4.44e-16`, maximum pullback score row norm `3.19e-16`, and per-coordinate RMS
at approximately `1e-16`. This validates the diagnostic formula on an exact
known case.

The q=20 finite-difference check was finite for beta values 0, 0.5, and 1.
At beta 0 the central step `1e-4` had maximum absolute error
`1.28e-14`; the smaller `1e-5` step rose to `7.11e-14` through cancellation.
At beta 0.5 and 1, the `1e-4` step gave maximum relative errors `3.46e-6` and
`6.92e-6`, respectively, while the `1e-5` step again became less stable.
This supports the conclusion that the implemented score is differentiating the
same value program to finite-difference precision on the tested points.

The fresh `(16,16)` two-stage tanh chart used a held-out latent bank of 64 and
eight batch-native updates at beta 0.5:

| Diagnostic | Initial | Final |
|---|---:|---:|
| centered log-density RMS | 151.8167 | 147.4252 |
| maximum score-residual row norm | 2909.7892 | 2972.6954 |
| score RMS, coordinate 1 | 139.0364 | 137.4716 |
| score RMS, coordinate 2 | 486.32498 | 490.20836 |
| score RMS, coordinate 3 | 214.2852 | 207.2915 |
| score RMS, coordinate 4 | 14.4108 | 14.3415 |

All updates were finite and the training graph traced once. The mixed small
changes and the still-large residuals are diagnostic evidence of an unresolved
optimization, capacity, or objective/chart issue. Eight updates and one seed
cannot distinguish those explanations.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep target score path | Affine identity and finite differences | Pass | Only finite banks were tested | Use the path for a separately scoped training ladder | Global score correctness |
| Integrate grouped HMC | Exact stochastic/semantic equivalence | Not met | Batch RNG semantics and candidate accounting | Build a deterministic per-candidate transition fixture | Any HMC speedup |
| Claim whitening | Pullback residual near zero on fresh data | Fail diagnostically | Under-training versus chart limitation | Run target-specific capacity/optimization ladder | Posterior correctness or mode discovery |
| Retry M3/Phase 9B | Valid repaired schedule and open master gate | Blocked by governing master | Full replay cost remains unresolved | Refresh a bounded performance plan | Convergence, superiority, scaling |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No launch, finiteness, memory-growth, route-scan, or affine-identity veto fired. Grouped-HMC equivalence veto did fire. |
| Statistically supported ranking | None; one GPU run and one seed do not support ranking. |
| Descriptive-only differences | Batch timing, grouped/serial wall time, loss, and residual changes. |
| Default readiness | Not established. No core default changed. |
| Next evidence needed | Exact grouped-transition equivalence, multi-seed target-specific training ladder, and a new resource budget before any replay. |

## Interpretation and red-team check

The strongest alternative explanation is that the fresh chart is simply
under-trained: eight updates are a localization run, not a convergence study.
The opposite possibility is a capacity or objective limitation; the present
data cannot separate them. The finite-difference result makes a gross score
implementation error less likely, but does not prove the global Jacobian or
target program correct. The batching result may reduce target-evaluation cost,
yet the prior full-replay measurement shows that serial candidate/HMC
orchestration remains the dominant schedule cost. A successful future run could
therefore still fail to whiten even after it becomes faster.

## Repair and refresh decision

E3 was not entered because the grouped path lacks the required equivalence
receipt. No optimizer reset, seed policy, target, HMC kernel, or default was
changed. The final post-closeout analytic smoke also passed, but it does not
change the E3 decision. The next bounded plan is
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`.
It changes the question and budget explicitly; it does not retry M3. Phase 9B
remains closed.
