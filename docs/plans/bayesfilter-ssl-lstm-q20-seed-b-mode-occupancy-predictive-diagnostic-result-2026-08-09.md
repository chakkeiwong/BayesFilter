# SSL-LSTM q=20 seed-B mode-region and predictive diagnostic result (2026-08-09)

## Verdict

The retained seed-B NeuTra/fixed-HMC archive did not cover the observation-weight
half-space containing the known negative MAP: all 4,000 retained states had a
positive observation weight, with pooled range `0.128007` to `1.205762`. Thus the
low retained R-hat is compatible with all four chains agreeing without observed
coverage of that known competing region.

Missed negative-region coverage is not an adequate standalone explanation for
the posterior-mean predictive rejection. The fixed positive-MAP simulator and
the fixed negative-MAP simulator were each distinguished from the synthetic
true-control simulator at every tested horizon. All ten energy permutation tests
had zero exceedances among 9,999 permutations and Monte Carlo p-value `0.0001`.
Both representatives also had descriptively positive overall output-mean shifts
of about `+0.255` to `+0.288` relative to truth.

This result does not test the posterior-predictive law within either mode or a
mixture over modes. It therefore does not prove that recovering the negative
mode cannot help, nor does it prove that multimodality caused no sampler error.
It does show that merely substituting the negative MAP for the sampled positive
region does not repair the fixed-parameter predictive discrepancy.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed coverage target | Whether retained states entered both observation-weight half-spaces containing the two known sign-separated MAP representatives. |
| Quantity computed | Observation-weight sign, range, quantiles, and sign-transition counts for every one of the 4,000 mapped retained states. |
| Relation | Equal to the stated retained-state half-space coverage target; different from formal basin occupancy, trajectory-level crossing, or integrated posterior mode mass. |
| Claimed predictive target | At each fixed horizon, equality of the complete-path simulator law at one fixed MAP representative and at the synthetic true control. |
| Quantity computed | Biased whole-path energy statistic and balanced-label Monte Carlo permutation p-value using 1,000 independent paths per arm and 9,999 permutations. |
| Relation | Equal to the stated fixed-representative finite-horizon equality-test target; different from a within-mode or mixture posterior-predictive test. |
| Source anchors | Plan, runner, retained archive, MAP artifact, transport loader, forecast source, and energy source hashes are recorded in the structured artifacts. |
| Not evaluated | Formal basin boundaries, leapfrog intermediate states, integrated mode masses, within-mode predictive mixtures, full posterior-predictive mixture, and independent posterior correctness. |

## Region coverage

The known stationary representatives were:

| Representative | Physical parameter | Target log density |
|---|---|---:|
| Positive MAP | `(0.733114, 0.172732, +0.589425, 0.158921)` | `-37.553174` |
| Negative MAP | `(0.446676, -0.241318, -0.587697, 0.119890)` | `-37.603473` |

Every retained state remained in the positive observation-weight half-space:

| Chain | Draws | Negative weights | Sign transitions | Minimum | Median | Maximum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,000 | 0 | 0 | 0.231420 | 0.598183 | 1.149258 |
| 1 | 1,000 | 0 | 0 | 0.198267 | 0.605854 | 1.161130 |
| 2 | 1,000 | 0 | 0 | 0.128007 | 0.607223 | 1.205762 |
| 3 | 1,000 | 0 | 0 | 0.131249 | 0.605644 | 1.182661 |
| Pooled | 4,000 | 0 | N/A | 0.128007 | 0.604353 | 1.205762 |

This establishes no observed retained-state coverage of the half-space
containing the negative MAP. It does not establish zero probability for that
mode or prove that no unarchived leapfrog state crossed zero.

## Predictive diagnostics

Each row used `n=1000` complete paths per arm, 9,999 permutations, and a strict
per-test decision rule `p < 0.01`. These are ten separate tests; no joint test,
combined p-value, familywise decision, or multiplicity adjustment was computed.

| Representative | T | Energy statistic | Exceedances | p-value | Overall mean shift | Decision |
|---|---:|---:|---:|---:|---:|---|
| Positive MAP | 10 | 0.211489 | 0 | 0.0001 | +0.287686 | Distinguished |
| Positive MAP | 20 | 0.244778 | 0 | 0.0001 | +0.256175 | Distinguished |
| Positive MAP | 30 | 0.298869 | 0 | 0.0001 | +0.253681 | Distinguished |
| Positive MAP | 50 | 0.425118 | 0 | 0.0001 | +0.265087 | Distinguished |
| Positive MAP | 100 | 0.590230 | 0 | 0.0001 | +0.263133 | Distinguished |
| Negative MAP | 10 | 0.194762 | 0 | 0.0001 | +0.273389 | Distinguished |
| Negative MAP | 20 | 0.272699 | 0 | 0.0001 | +0.272479 | Distinguished |
| Negative MAP | 30 | 0.345219 | 0 | 0.0001 | +0.271213 | Distinguished |
| Negative MAP | 50 | 0.407367 | 0 | 0.0001 | +0.260448 | Distinguished |
| Negative MAP | 100 | 0.559458 | 0 | 0.0001 | +0.255460 | Distinguished |

At the tested fixed points, the two modes display the same qualitative
predictive problem: a positive output shift and rejection against truth at all
five horizons. Differences between positive- and negative-MAP statistics are
descriptive only and do not support a ranking.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Record absent retained coverage of the negative-MAP half-space | 0/4,000 retained states had negative observation weight | Archive hashes, shape, finite transform, and source identity passed | Sign half-space is not a formal basin boundary; intermediate leapfrog states were not archived | If sampler coverage remains the question, plan chains initialized and tuned in both known regions | Zero negative-mode posterior mass or proof of no trajectory crossing |
| Reject both fixed MAP representatives as predictively matching truth at all tested T | All 10 equality nulls rejected at 1%, each with p=`0.0001` | Forecast status, finite paths, energy geometry, XLA, CPU hiding, artifact hashes, and wall cap passed | MAP points do not represent within-mode posterior mixtures; finite-data posterior may legitimately differ from truth | Diagnose the target/data/filter relationship and, if warranted, run a multimode-aware posterior predictive check rather than another plug-in MAP test | Posterior incorrectness, model inadequacy, or impossibility of repair by a posterior mixture |
| Do not attribute the predictive failure solely to missed multimodality | Negative-region coverage was absent, but the negative MAP fixed simulator also failed all five tests | No run-invalidating veto | Unknown integrated mode masses and within-mode predictive laws | Separate sampler coverage repair from likelihood/model/output diagnostic work | Multimodality is irrelevant or HMC is correct |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the repaired `r2` canary, all ten material rows, source bindings, retained mapping, forecast validity, and artifacts. The `r1` nearest-MAP occupancy interpretation is invalid and preserved only as repair evidence. |
| Statistically supported ranking | None. No uncertainty-supported comparison ranks the two MAP representatives. |
| Statistically supported decisions | Each of the ten fixed-representative equality nulls is rejected by its predeclared permutation test at 1%. |
| Descriptive-only differences | Half-space fractions, sign-transition counts, observation-weight ranges/quantiles, energy-statistic magnitudes, representative-to-representative differences, output-mean shifts, and runtime. |
| Default-readiness | Not evaluated. This diagnostic cannot promote a sampler, transport, model, or inference default. |
| Next evidence needed | Multimode-initialized and independently tuned chains if cross-region posterior coverage is required; within-mode or multimode posterior-predictive simulation if posterior predictive validity is the target; target/filter/data checks if fixed-point output displacement remains unexplained. |

No viable fixed-MAP representative remains under the tested equality screen. No
statistical ranking between them is supported. The current evidence rejects
these two fixed simulator laws, not the entire NeuTra/HMC research direction.

## Engineering, sampler, and scientific ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Passed focused tests, authenticated input reconstruction, finite/shape checks, XLA execution, CPU-only device check, row and occupancy receipt hashes, and bounded completion. |
| Numerical/sampler validity | The earlier sequential R-hat/ESS screen remains passed for the sampled region. This diagnostic establishes no retained negative-half-space coverage; native divergence remains unavailable in the source HMC artifact. |
| Scientific interpretation | Both fixed MAP simulator laws differ from truth at all tested horizons. This does not determine whether the finite-data posterior, approximate target, filtering likelihood, predictive simulator, or model is responsible. |

## Repair history

The initial `r1` canary used a forced raw-Euclidean nearest-MAP partition. It
labeled 1,376/4,000 positive-weight draws as nearer the negative MAP because
other coordinates dominated the distance. That label was wrong relative to a
negative-region occupancy claim. The material campaign was stopped before
launch, the plan was revised, and `r2` used only directly observed
observation-weight half-space coverage. The `r1` canary remains diagnostic-only
evidence of the falsified partition and must not be used for basin occupancy.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` with shared dirty worktree recorded |
| Focused tests | `21 passed` before `r1`; repaired source/energy tests `10 passed` before `r2` |
| Canary command | `CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py --mode canary` |
| Campaign command | `CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py --mode campaign` |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU | CPU-only diagnostic exception; `CUDA_VISIBLE_DEVICES=-1`; no visible TensorFlow physical GPU; 8 CPU threads |
| XLA | `jit_compile=true`; runtime emitted successful XLA cluster compilation |
| Data version | Retained summary SHA-256 `279bfbdfa244dcba28ec63c6cc1168be0273bb3558c70a70a1eae701b7e73165`; MAP artifact SHA-256 `e646428f54bc569d52b572525f2b3b2d555792d9c1e5cd3cd27e643a2e8cd48e` |
| Seeds | Deterministic representative/horizon seed banks recorded in each row receipt |
| Canary wall time | `36.038 s` under `300 s` cap |
| Campaign wall time | `210.002 s` under `1,200 s` cap |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-plan-2026-08-09.md` |
| Structured result | `docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2/summary.json` |
| Occupancy artifact | `docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2/occupancy.json` |
| Row artifacts | Ten `plus/minus-t*.json` receipts and permutation tensors under the `r2` artifact root; all receipt hashes verified |
| Historical repair evidence | `docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r1/canary.json` |

## Post-run red team

The strongest alternative explanation is that fixed MAP points are poor
representatives of their modes. A properly weighted posterior mixture, or even
the within-negative-mode predictive distribution, could differ substantially
from the negative MAP simulator. The result that would overturn the narrow
conclusion is a hash-valid rerun showing retained negative weights or a corrected
mapping from archived NeuTra states to physical parameters. The weakest part of
the scientific evidence is the jump from fixed representative tests to any
claim about whole-mode or whole-posterior predictive behavior; that jump is not
made here.

