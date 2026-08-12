# SSL-LSTM q=20 seed-B mode-occupancy and predictive diagnostic plan (2026-08-09)

## Objective

Determine whether the 4,000 retained seed-B NeuTra/fixed-HMC draws entered both
observation-weight half-spaces containing the two known target-only stationary
basins, and determine whether a fixed
representative from either basin generates a complete-path output law that the
existing energy diagnostic distinguishes from the synthetic true-control law.

This is a diagnostic of one retained archive and two fixed parameter points. It
does not estimate integrated posterior mode masses or a posterior-predictive
mixture and cannot, by itself, prove why the posterior-mean predictive law was
rejected.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Did each retained chain enter both observation-weight half-spaces containing the known MAP representatives, and is either representative's simulator not distinguished from the true-control simulator at the five fixed horizons? |
| Retained candidate | The authenticated 4-chain, 1,000-retained-draw-per-chain seed-B archive from the passed sequential screen. |
| Basin representatives | The highest-log-density stationary endpoint with positive observation weight and the highest-log-density stationary endpoint with negative observation weight in the target-only multistart MAP artifact. |
| Comparator | Synthetic true-control parameter `(0.35, -0.08, 0.65, 0.05)`. |
| Expected failure mode | All chains agree within the positive basin, so low R-hat fails to reveal missing negative-basin coverage. |
| Region-coverage diagnostic | Map every retained NeuTra draw to physical coordinates and record observation-weight sign, range, quantiles, and sign transitions per chain. The positive MAP is in the positive half-space and the negative MAP is in the negative half-space. |
| Predictive criterion | Separately for each basin representative and each `T in (10,20,30,50,100)`, reject equality with truth iff the balanced-label whole-path energy permutation p-value is `<0.01`. |
| Promotion criterion | None. This is explanatory diagnosis, not candidate promotion. |
| Promotion veto | None. A rejected fixed-MAP simulator law is not a posterior or sampler veto. |
| Continuation veto | Invalid source hashes/schema, wrong retained shape/count, nonfinite mapped draws or paths, invalid MAP endpoint, failed forecast status, failed energy geometry, missing XLA provenance, visible GPU in the CPU exception, corrupt artifact, or wall-cap breach. |
| Repair trigger | No retained state in the half-space containing one known MAP motivates a separately planned multimode-initialized HMC campaign; both representative laws being distinguished motivates likelihood/filter/model or finite-data-posterior investigation rather than blaming mode coverage alone. |
| Explanatory diagnostics | Per-chain observation-weight half-space counts, ranges, quantiles, sign-transition counts, energy statistics, p-values, mean shifts, and runtime. |
| Must not conclude | Integrated mode probability, posterior correctness, cross-mode mixing, equality of infinite-horizon DGPs, that a MAP represents its whole mode, that a mixture of modes fails or passes, or that multimodality is the sole cause of the posterior-mean predictive rejection. |

## Evidence contract

1. Retained draws come only from
   `docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential/`.
   The archive must report `SEQUENTIAL_SCREEN_PASSED`, exactly 1,000 retained
   transitions per chain, four chains, 2,000 excluded warm-up transitions per
   chain, and hash-valid retained tensors of shape `[500,4,4]` for each of two
   chunks.
2. The exact frozen seed-B transport is reconstructed through
   `docs/benchmarks/ssl_lstm_q20_neutra_seed_b_terminal.py`; its target signature
   must be `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`
   and its base adapter signature must be
   `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3`.
3. The MAP endpoints come only from
   `docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json`.
   An eligible endpoint has finite log density, score infinity norm `<=1e-5`,
   and four finite coordinates. The positive and negative representatives are
   selected by observation-weight sign and maximum recorded log density within
   that sign.
4. Region coverage is defined only by the sign of physical coordinate 2,
   `observation_weight.0.0`. The positive MAP has weight `+0.589425...`; the
   negative MAP has weight `-0.587697...`. The artifact reports whether retained
   states entered each corresponding open half-space. It does not assign formal
   basin membership: another basin could occupy either half-space, a basin can
   have tails crossing zero, and unrecorded HMC leapfrog states are not audited.
5. Half-space fractions and transition counts are descriptive because retained
   HMC draws are autocorrelated. Zero retained states in the negative half-space
   establishes zero observed coverage there, not zero posterior probability or
   zero trajectory-level visits.
6. Each representative-versus-truth test uses 1,000 independent complete paths
   per arm, 9,999 independently generated balanced-label permutations, raw
   shared output coordinates, and the biased whole-path energy V-statistic.
   The Monte Carlo p-value is `(1 + exceedances) / 10000`.
7. The ten tests are separate diagnostics. There is no joint test, combined
   p-value, multiplicity adjustment, or global pass/fail decision. A fixed-MAP
   non-rejection is not an equivalence claim.
8. Forecast and energy kernels use TensorFlow XLA. CPU-only execution with
   `CUDA_VISIBLE_DEVICES=-1` is an explicit diagnostic exception. It supplies no
   production-target, GPU, or performance evidence.

## Numeric provenance and default audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| 4,000 retained draws | Measured archive content | Use all available retained evidence | Autocorrelation makes raw proportions look more certain than they are | Report descriptive counts only, per chain |
| Two MAP representatives | Measured target-only stationary endpoints | Existing independent evidence for two known basins | More basins may exist; MAP does not summarize mode volume | State two-known-representative limitation |
| Observation-weight half-spaces | Repaired explanatory diagnostic after the `r1` canary falsified raw nearest-center assignment | Directly records whether retained states entered the region containing each sign-separated MAP | Does not define formal basin membership or inspect intermediate leapfrog states | Report raw minimum/maximum, quantiles, sign counts, and transitions |
| MAP representatives | Reviewed convenience choice | Available fixed points with checked stationary scores | Representative law need not equal within-mode posterior predictive law | Explicit fixed-point nonclaim |
| `n=1000`, horizons, `alpha=0.01` | User-fixed in preceding diagnostic | Direct comparability with posterior-mean result | Multiple related tests can be overinterpreted | No joint decision or equality claim |
| 9,999 permutations | Inherited reviewed diagnostic setting | P-value resolution `0.0001` at a 1% threshold | Monte Carlo uncertainty near threshold | Report exceedance count and strict rule |
| 1,200-second campaign cap | Derived from measured 166.7 s for one five-horizon suite, doubled for two representatives, plus >3x loader/compile headroom | Bounded run with ample measured headroom | Shared CPU load can breach cap | Stop without scientific interpretation on breach |

The MAP representatives, half-space region diagnostic, and fixed-MAP predictive
tests remain explanatory diagnostics. They are not promoted defaults.

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. The occupancy source is the exact retained archive under question; predictive comparison uses the synthetic generating control. |
| Proxy promoted? | No. Half-space counts report only observed region coverage, and fixed-MAP energy tests concern only those simulator laws. Neither establishes basin membership or posterior correctness. |
| Missing stop condition? | No. Source/schema, shape, finite, XLA, device, forecast, energy, artifact, and 1,200-second cap vetoes are explicit. Statistical rejection does not stop later rows. |
| Unfair comparison? | No. Each test uses the same simulator, horizon, sample size, backend, dtype, and independent seed banks for its two arms. |
| Hidden assumptions? | Exposed: two known sign-separated representatives, half-space rather than basin classification, MAP-as-fixed-representative, iid simulated paths, and separate finite-horizon tests. |
| Stale context? | No. The prior posterior-mean paths are not reused. Existing target, archive, and MAP artifacts are hash-bound at execution. |
| Environment mismatch? | CPU/XLA is an explicit diagnostic exception; GPU is hidden before TensorFlow import. |
| Could the run pass while misleading us? | Yes. A mode representative may resemble truth even if its within-mode distribution does not, and a multimodal mixture may behave differently from both representatives. These are nonclaims. |
| Could the run fail for engineering reasons? | Yes. Source drift, transport reconstruction, shape, or forecast validity failure is separated as a continuation veto, not scientific evidence. |
| Would artifacts answer the stated diagnostic? | Yes. They directly show retained coverage of the two MAP-containing half-spaces and fixed-representative whole-path test outcomes, while explicitly not answering integrated mode mass, formal basin occupancy, or sole causation. |

The first `r1` canary falsified the proposed raw nearest-MAP partition: it labeled
1,376/4,000 draws as nearest the negative MAP even though all 4,000 observation
weights were positive. Other coordinates dominated Euclidean distance, so those
labels could not support negative-basin occupancy. No material predictive run
was launched. The `r1` canary is preserved as repair evidence; it is ineligible
for basin-occupancy interpretation.

Audit verdict after repair: **PASS FOR THE NARROW DIAGNOSTIC**. The design cannot
prove formal basin occupancy or sole causation, but it can establish whether any
retained state entered the half-space containing the known negative MAP and can
compare both fixed representative laws without rerunning HMC. No material plan
flaw remains within that narrower scope.

## Execution

Artifact root:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2/`

Phases:

1. Focused source-contract tests.
2. Repaired canary: authenticated archive/transport/MAP reconstruction, complete
   half-space coverage calculation, and one true-versus-true `n=32`, `T=20`, 999-
   permutation mechanics check; cap 300 seconds.
3. Campaign: repeat authenticated inputs and write the occupancy artifact, then
   run positive-MAP-versus-truth and negative-MAP-versus-truth at all five
   horizons; cap 1,200 seconds.
4. Write terminal result and reset memo with decision and inference-status
   tables.

Exact commands:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic.py tests/test_two_sample_energy_tf.py tests/test_ssl_lstm_complexity_predictive_tf.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py --mode canary
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py --mode campaign
```

## Interpretation

- No retained state in the half-space containing one known MAP supports the
  hypothesis that low R-hat reflected agreement without observed coverage of
  that MAP's region. It does not prove the other basin has material integrated
  mass or that no leapfrog trajectory crossed the boundary.
- A representative-versus-truth `p < 0.01` means that fixed simulator law is
  distinguished at that horizon. `p >= 0.01` means not distinguished at 1%; it
  does not establish equivalence.
- If only the sampled representative is distinguished, explicit multimode
  initialization becomes the next smallest sampler diagnostic.
- If both representatives are distinguished, missed mode coverage alone is not
  an adequate explanation; finite-data posterior displacement,
  likelihood/filter approximation, and model/output construction remain live.
- No outcome estimates the predictive law of a posterior mixture with unknown
  mode weights.
