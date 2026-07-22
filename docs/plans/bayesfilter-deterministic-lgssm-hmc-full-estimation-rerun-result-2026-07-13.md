# BayesFilter Deterministic LGSSM HMC Full-Estimation Rerun Result

Date: 2026-07-13  
Status: `COMPLETE_PASS_SINGLE_FIXTURE_FULL_ESTIMATION_RECOVERY_SCREEN`  
Plan: `docs/plans/bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-plan-2026-07-13.md`

## Outcome

The fresh deterministic `T=120`, 18-parameter lower-triangular LGSSM campaign
completed end to end. The corrected rank-normalized tuning verifier selected a
fixed kernel, a separate-seed serious HMC run passed the predeclared
all-parameter convergence and ESS gates, the retained archive passed provenance
and finiteness checks, and all 18 parameters passed the predeclared
single-fixture recovery screen.

This result supports the narrow statement that the corrected tuner and frozen
HMC execution path successfully estimated this fixed synthetic fixture. It does
not establish calibration, estimator robustness, sampler superiority, GPU
readiness, production readiness, or default readiness.

## Research Question And Answer

| Item | Recorded answer |
| --- | --- |
| Main question | After repairing tuning to use `max(rank-normalized split R-hat, folded rank-normalized split R-hat)`, can a fresh fixed kernel pass independent serious sampling and recover all 18 raw parameters on the fixed `T=120` fixture? |
| Candidate | Fresh tuning attempt index `2` (third attempt), frozen before serious sampling. |
| Exact comparator/baseline | The same deterministic LGSSM target, fixture, parameterization, thresholds, and CPU/XLA implementation used by the reviewed fresh-run plan; no sampler ranking was attempted. |
| Primary criterion | Target, corrected tuning, serious convergence, retained integrity, and every parameter recovery row jointly pass. |
| Promotion vetoes | Invalid target/score, nonfinite transition or archive, XLA fallback, modern R-hat above `1.01`, bulk ESS below `1000`, tail ESS below `400`, provenance mismatch, or any recovery distance above `3` posterior SD. |
| Result | Passed. No hard veto fired. |
| What is not concluded | Calibration, coverage, generality, robustness, superiority, GPU behavior, production readiness, or default readiness. |

## Review Record

The plan received a local skeptical audit before execution. That audit checked
the baseline, diagnostic roles, stop conditions, seed separation, environment,
historical-artifact leakage, and whether the planned artifacts could answer the
research question.

Claude was available for the initial bounded read-only review. A health probe
returned `CLAUDE_PROBE_OK`, and the bounded review identified
`FEASIBILITY_MISSING_HARDCODED_IDENTITY_REPAIR`. The plan was revised to add an
isolated fresh V3 config/controller/campaign route while preserving V1/V2.

A post-revision Claude recheck was rejected at the execution environment's
external-disclosure boundary. It was not retried indirectly. No post-revision
Claude convergence claim is made. Local implementation review, compilation,
focused tests, static XLA scans, actual-target smoke, serious execution, and
independent recovery supplied the remaining evidence. Codex remained
supervisor/executor and Claude was never treated as execution authority.

## Phase Results

| Phase | Result | Key evidence |
| --- | --- | --- |
| 0. Integration gaps | Passed | Fresh V3 path, builder, campaign launcher, recovery evaluator, no-overwrite checks, and V1/V2 preservation implemented. |
| 1. Preflight | Passed | Immutable original preflight artifact `sha256:590d87c85313a89cbae151926eda9d670caf3503c5803f7e9681117659034ac3`; tuning and serious seeds differ. |
| 2. Fixture and target | Passed | Fixture artifact `sha256:bc5cf901f1a98788b74a3144be188c4dccb19b868106f814a1e55417c1b98d66`; spectral radius `0.62`; Lyapunov max residual `1.39e-17`; independent target/finite-difference checks `3 passed`. |
| 3. Geometry and mass | Passed | No diagonal fallback; condition number `1.9641`; precision/covariance identity error `6.66e-16`; positive finite eigenvalues. |
| 4. Corrected tuning | Passed | Attempt index `2`; acceptance `0.7078`; 1,250 verifier draws; rank R-hat `1.00643`; folded R-hat `1.00865`; combined `1.00865`; no hard vetoes. |
| 5. Actual-target smoke | Passed after localized repair | Two workers, two chains each, one XLA trace per worker, finite `(8, 4, 18)` archive, exact config/replay provenance. Smoke R-hat/ESS remained explanatory only. |
| 6. Serious sampling | Passed | Burn-in passed at 2,000 per chain; retained sampling passed at 4,000 per chain; no extension or retry required. |
| 7. Final recovery | Passed after localized reporting repair | Independent diagnostics exactly agree with terminal diagnostics; every recovery row passed; worst distance `1.65139` posterior SD. |
| 8. Closeout | Passed | This result record, run manifest, immutable artifacts, decision tables, inference table, and post-run red team are complete. |

## Repair Record

Two procedural implementation defects were found during execution. Neither was
a tuning, target, transition, convergence, or recovery-math failure.

| Repair | Failure boundary | Change | Verification |
| --- | --- | --- | --- |
| Phase 5 replay provenance serialization | Both smoke workers completed finite burn-in and retained transitions with no hard vetoes, then `run_phase7` raised `KeyError` before writing the private archive. | `validate_phase7_v3_inputs` now returns the already-validated `private_replay_artifact_hash`. | Regression plus full focused suite; repaired smoke passed. Repair manifest artifact `sha256:7350ce08490f92d16b11f7b153f304a3bb06d3c0c392ffd442165be0d66f898d`. |
| Phase 7 result-path serialization | Recovery verified the archive, recomputed diagnostics, checked exact terminal agreement, and built recovery rows, then a relative path was compared to the absolute repo root before result serialization. | Resolve `config.path` before recording its repo-relative representation. | Absolute/relative path regression plus full focused suite; recovery-only rerun passed. Repair manifest artifact `sha256:ba42c964de9e9c196a6594c9d71008ce5e55fb019297b97364fc5801a27c8aab`. |

The original failed smoke is preserved at
`/tmp/bayesfilter-full-estimation-smoke-20260713`:

| Artifact | SHA-256 |
| --- | --- |
| Failed smoke result | `d151d38bc01d2c2d27bc852d993db7aa9de4e70301b0cf2c2de34ed3bb940c84` |
| Failed smoke progress | `de76cb017367d0f9ee5fa33c92d500ac423f98d234a9fa89571347970c287579` |

The repaired smoke is preserved at
`/tmp/bayesfilter-full-estimation-smoke-20260713-repair1`:

| Artifact | SHA-256 |
| --- | --- |
| Passed smoke result | `e01471b664946626af55df2bfae3c9a74c3ffa23b1af715c27583424c3d21fec` |
| Passed smoke progress | `32078b6c14b1739aa055eb6301e0e074ee32b47fe78b5785cbd4a72e55c5a629` |
| Smoke retained archive | `f446322b79cf2fe345193d54ce491f4ca62c7a254c3758e55754d3706ce90c6c` |

## Serious Sampling Evidence

The frozen kernel used serious root seed `(20260713, 701)`, independent of the
tuning seed `(20260709, 501)`. There was no retuning, reseeding, chain removal,
manual thinning, or candidate retry after serious execution started.

| Gate | Threshold | Observed | Status |
| --- | --- | --- | --- |
| Burn-in draws per chain | Initial `2000`; cap `16000` | `2000` | Passed at first check |
| Burn-in combined modern R-hat | `<=1.01` | `1.0090077069` | Passed |
| Burn-in minimum bulk ESS | `>=1000` | `1761.1982` | Passed |
| Burn-in minimum tail ESS | `>=400` | `1220.7061` | Passed |
| Retained draws per chain | Initial `4000`; cap `40000` | `4000` | Passed at first check |
| Retained combined modern R-hat | `<=1.01` | `1.0061667293` | Passed |
| Retained minimum bulk ESS | `>=1000` | `5561.2834` | Passed |
| Retained minimum tail ESS | `>=400` | `3104.4038` | Passed |
| Finiteness and hard vetoes | All finite; none | All finite; `[]` | Passed |
| Private archive | `(4000, 4, 18)`, exact config/replay provenance | Verified | Passed |
| Implementation source stability | No drift during serious run | Unchanged | Passed |

The worst retained R-hat was the folded rank-normalized split R-hat for
`log_q3`. It remained below the threshold. The lowest bulk ESS was for
`log_r3`, and the lowest tail ESS was for `log_q3`; both remained above their
thresholds.

## Recovery Results

The recovery evaluator independently reloaded all 16,000 retained samples,
recomputed modern rank-normalized diagnostics, and required agreement with the
serious terminal under the declared serialization tolerance.

| Parameter | Truth | Posterior mean | Posterior SD | Mean MCSE | Error / SD | Posterior / prior SD | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `a11_raw` | 0.927469 | 0.762124 | 0.229940 | 0.003046 | 0.7191 | 0.4599 | Yes |
| `a22_raw` | 0.639716 | 0.772940 | 0.208768 | 0.002561 | 0.6381 | 0.4175 | Yes |
| `a33_raw` | 0.368799 | 0.246548 | 0.144979 | 0.001598 | 0.8432 | 0.2900 | Yes |
| `a44_raw` | 0.190507 | 0.294187 | 0.145634 | 0.001288 | 0.7119 | 0.2913 | Yes |
| `a21_raw` | 0.568539 | 0.365469 | 0.310513 | 0.003740 | 0.6540 | 0.5175 | Yes |
| `a31_raw` | -0.293893 | -0.827025 | 0.351728 | 0.003702 | 1.5157 | 0.5862 | Yes |
| `a32_raw` | 0.423649 | 0.795032 | 0.356025 | 0.003513 | 1.0431 | 0.5934 | Yes |
| `a41_raw` | 0.173138 | -0.085817 | 0.214387 | 0.001868 | 1.2079 | 0.3573 | Yes |
| `a42_raw` | -0.232682 | -0.178996 | 0.247344 | 0.002247 | 0.2170 | 0.4122 | Yes |
| `a43_raw` | 0.325294 | 0.818821 | 0.374967 | 0.003847 | 1.3162 | 0.6249 | Yes |
| `log_q1` | -1.203973 | -1.378136 | 0.144974 | 0.001863 | 1.2013 | 0.4142 | Yes |
| `log_q2` | -1.347074 | -1.543280 | 0.118813 | 0.001236 | 1.6514 | 0.3395 | Yes |
| `log_q3` | -1.514128 | -1.627407 | 0.149237 | 0.002270 | 0.7591 | 0.4264 | Yes |
| `log_q4` | -1.714798 | -1.740517 | 0.134947 | 0.001964 | 0.1906 | 0.3856 | Yes |
| `log_r1` | -2.120264 | -1.957687 | 0.304933 | 0.004067 | 0.5332 | 0.8712 | Yes |
| `log_r2` | -2.207275 | -2.246895 | 0.272005 | 0.002817 | 0.1457 | 0.7772 | Yes |
| `log_r3` | -2.302585 | -2.230884 | 0.319020 | 0.004239 | 0.2248 | 0.9115 | Yes |
| `log_r4` | -2.407946 | -2.413829 | 0.315970 | 0.004100 | 0.0186 | 0.9028 | Yes |

The worst recovery distance was `log_q2` at `1.65139` posterior SD, below the
fixed threshold of `3.0`. Posterior/prior SD ratios ranged from `0.28996` to
`0.91148`. These contraction values are explanatory only.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept this frozen kernel for this completed fixture result | Passed end-to-end | No hard vetoes; serious R-hat/ESS, archive integrity, and all recovery rows passed | One favorable synthetic data realization; truth equals prior and geometry center | Close this campaign and plan a separate offset-truth confirmation | Calibration, robustness, superiority, GPU/production/default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed. No divergence/nonfinite, target, XLA, provenance, diagnostic-cap, source-drift, or recovery veto fired. |
| Viable candidates | The one predeclared frozen kernel remained viable and passed serious sampling. |
| Statistically supported ranking | None. No competing samplers or stochastic candidates were ranked. |
| Descriptive-only differences | Runtime, per-parameter ESS margins, posterior contraction, quantiles, MCSE values, and relative recovery distances are descriptive for this run. |
| Default readiness | Not established. This result is insufficient for a default change. |
| Next evidence needed | Offset-but-stable truth, fresh data seed, geometry initialized without truth access, multi-fixture replication, and uncertainty-aware aggregation. |

## Evidence Ledgers

| Ledger | Verdict | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASS` | Fresh V3 route, no-overwrite behavior, replay-bound archives, one XLA trace per worker, source-integrity terminal, `77 passed`, compilation, and static scans. |
| Numerical/sampler validity | `PASS_FOR_THIS_RUN` | Independent serious seed, four chains, 4,000 retained draws per chain, modern R-hat `1.00617`, bulk ESS `5561`, tail ESS `3104`, finite archive, no hard vetoes. |
| Scientific interpretation | `PASS_SINGLE_FIXTURE_RECOVERY_ONLY` | All 18 recovery distances are below `3` posterior SD; maximum `1.65139`. Favorable centering prevents a robustness claim. |

## Run Manifest Summary

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Python | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11`, version `3.11.14` |
| Environment | conda `tf-gpu`; TensorFlow `2.19.1`; TFP `0.25.0` |
| Device | Deliberate CPU-only, `CUDA_VISIBLE_DEVICES=-1` |
| Numerical route | TensorFlow `float64`, XLA JIT enabled, no non-JIT fallback |
| Workers/chains | 2 persistent workers, 2 chains per worker, 4 chains total |
| Threads per worker | intra-op `8`, inter-op `1`, OMP `8`, OpenBLAS `1`, MKL `1` |
| Seeds | simulation `(20260709,301)`; geometry `(20260709,401)`; tuning `(20260709,501)`; serious `(20260713,701)` |
| Serious wall time | `136.563 s` |
| Serious implementation inventory | `sha256:b75fad67f85923b6a0f7ec2eabdab4283892e4880de56a8a2cd5a09443892491` |
| Transition identity | `sha256:119a77708dc8c8255fa6d79cab4ed8e3b715d47b123bd648be530efb707d3df0` |
| Serious execution identity | `sha256:8dfeeb3aa161029480a413753ed3f892304e2c1da58481da94e9d779032e5a5d` |

The serious manifest froze driver SHA-256
`fdfb069840d1e80e7fd8c959dae3c84a3647968747eab4894c383bcc66e49bfb`.
The post-sampling recovery path-only repair changed that driver to
`2678c7b788b6b5f11a7cedbfb92462c4fa82f2326a607f7647ef822921e747c3`.
The serious terminal reports that every frozen source remained unchanged during
sampling. The later drift is explicitly scoped to recovery result-path
serialization and is bound by the supplemental repair manifest.

## Commands Actually Run

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 \
  docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py \
  --config docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json \
  --phase7-config docs/benchmarks/configs/multidim_lgssm_full_estimation_phase7_2026_07_13.json \
  --stage burnin_sampling --phase7-smoke \
  --phase7-output-dir /tmp/bayesfilter-full-estimation-smoke-20260713-repair1

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 \
  scripts/run_hmc_full_estimation_campaign.py \
  --config docs/benchmarks/configs/multidim_lgssm_full_estimation_phase7_2026_07_13.json \
  --campaign-root docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/phase7_campaign

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 \
  docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py \
  --config docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json \
  --phase7-config docs/benchmarks/configs/multidim_lgssm_full_estimation_phase7_2026_07_13.json \
  --stage final_recovery
```

The final-recovery command was first attempted with the old driver and failed
only at path serialization. It was rerun once after the bounded repair. Tuning
and serious sampling were not rerun.

## Artifact Index

| Artifact | Artifact hash or file SHA-256 |
| --- | --- |
| Original preflight manifest | artifact `sha256:590d87c85313a89cbae151926eda9d670caf3503c5803f7e9681117659034ac3`; file `e77401afd1a32aa14a7d64a80bfd81b8e5ebec7dd6d5e946a834f59883dfa192` |
| Fixture | artifact `sha256:bc5cf901f1a98788b74a3144be188c4dccb19b868106f814a1e55417c1b98d66`; file `d6b0ba10250f45913257055ad14f052361e3bb664bdaf95547a268396a37d276` |
| XLA target gate | artifact `sha256:be5b29759b30980233e527eab20f3ace5342a923a415c16233d9e767bf9c6b0a`; file `edd4f79923095595cbce78804cb1bf6bbedc72d205b86c36f92cce36235a689f` |
| Geometry | artifact `sha256:66f37340c9ce131152691797676d52ce77ba3decd08a3b2f64780de29d1a5f73`; file `bd9b086d60df518b4410c3348b1fd93663fc8dc861428d77b4098aa1e118a87d` |
| Mass | artifact `sha256:2e41adfdebb47e9b949a675671c12ad1261588d6932c27c3c795724abaa355ad`; file `54549c9156821536bc4780f0406a7716b0d3fa39a5b5900fa2893cbef2968a95` |
| Corrected tuning | artifact `sha256:f104f02e83bf8fa10405209771dd5b9afdda7786dd053fc31e7b554bc4e03546`; file `c5dfc95cf7ea2b4a0116c364fa0dd39d3c12f6524ab1a75eb83b60cefe44f58e` |
| Private replay | artifact `sha256:eab345c411a1279676588efc24993499c90e18c227b6a46391b98d03b067f318`; file `9ea3221a519ef0930be6029a35c0e266799e5055efc3f43ea4b188fa89cfd0d9` |
| Phase 5 repair manifest | artifact `sha256:7350ce08490f92d16b11f7b153f304a3bb06d3c0c392ffd442165be0d66f898d`; file `676af4b24e86c2e3975aabcb83ed7a455ee014b2466881e3eb50dd7aad61ca93` |
| Serious run manifest | artifact `sha256:6d498c295305f36b8374a71a9e14c9233b1d217544b45778853d1c3c91d070f9`; file `3f7df1e4247f1583989ee43d45d4fe53c8abb8c2dd8b83561053630fcfc1f146` |
| Serious controller result | artifact `sha256:7c852c35fb6850f33cc7316df2de26e7d1b168d614cd63e369a4fabb7bfe3552`; file `ee8ed0e1cb18f99e36e25873d545a80d00c088a7a368ab94eda9bec6f0b4e2fd` |
| Serious retained archive | file `1b0c05d4ea2981b1be179040d3a52039f05efe6c5b163f9bf7bba64ce2068920` |
| Serious campaign terminal | artifact `sha256:23cc2a0ce515a31e41a4bff6ba671e115191ff08d2358cfee126547f0570b4e5`; file `44b03edf07ac63506ed202195a0e872450cb3f26a3e8cd23046041cf446df309` |
| Phase 7 reporting repair manifest | artifact `sha256:ba42c964de9e9c196a6594c9d71008ce5e55fb019297b97364fc5801a27c8aab`; file `5792d5fbd332de7f8be6d20730e818f1ee9a935a6cb7c393f51d8d3257be43df` |
| Final recovery | artifact `sha256:777b79fd038cfbf18cca66788569e7784b971782bdc0c206600b53398ca73a3d`; file `bcc6e71a1067dc648758a5aac9c87ef7e94fdd4b1ac53d5601ef4e9fdf6741b5` |

## Verification

Final focused command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q \
  tests/test_hmc_full_estimation_campaign.py \
  tests/test_deterministic_lgssm_hmc_tuning_driver.py \
  tests/test_deterministic_lgssm_hmc_phase7_tf.py \
  tests/test_hmc_convergence.py \
  tests/test_hmc_fixed_size_chunk_runner.py
```

Result: `77 passed, 2 warnings`. The warnings are the existing TFP
`distutils.version` deprecations. Python compilation, `git diff --check`, and
the scoped scan for `jit_compile=False`, `--no-jit`, and runtime
`GradientTape` also passed.

## Post-Run Red Team

| Question | Assessment |
| --- | --- |
| Strongest alternative explanation | The fixture is unusually favorable because truth, prior center, and initial geometry center coincide. Recovery may be substantially easier than in ordinary estimation. |
| Weakest part of the evidence | One synthetic realization and one serious sampling seed cannot establish calibration or robustness. The `3 SD` recovery screen is a declared fixture check, not a coverage theorem. |
| What would overturn the narrow conclusion | Recomputing the retained archive with the same diagnostic definition and obtaining a gate failure, discovering a target/provenance mismatch, or reproducing the run and finding the recorded immutable artifacts inconsistent. None occurred here. |
| What would support a broader conclusion | A reviewed offset-truth campaign with a fresh data seed, geometry initialized without truth access, multiple fixtures/seeds, and uncertainty-aware aggregation. |

## Final Handoff

This campaign is closed. No phase remains in the current plan. The next
scientifically justified work is a new, separately reviewed offset-truth and
multi-fixture confirmation plan. It must not reuse this single favorable
fixture as evidence of robustness or default readiness.
