# SSL-LSTM Direct Visual Validation Result

Date: 2026-07-18  
Decision: `PASSED_VISUAL_PACKAGE_CONFIRMATION_CLOSED`

## Scope and Boundary

The bounded package used only the first 64 draws of segment 0 from each
admitted Phase-7 chain. The Phase-8 pilot had already opened and permanently
excluded these draws; segment 1 and indices 64--511 were hash-checked but not
deserialized. No new HMC/NeuTra run was performed and the G/H confirmation
boundary remains closed.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Primary integration evidence | Authenticated pilot-prefix loading, frozen transport mapping, trusted GPU/XLA ten-step forecasts, finite fan/moment summaries, five figure pairs, and receipt `visual-validation-result.json` |
| Hard veto screen | Passed: hashes, segment identity, shape/orientation, finite mapped parameters, finite forecasts, terminal covariance statuses, disjoint innovation banks, and artifact writes |
| Visual interpretation | Four launch traces and fan charts show descriptive initialization/predictive behavior; they are not a correctness test |
| Moment diagnostic | Maximum absolute mean difference `0.08497` at horizon 7; maximum absolute log-variance difference `0.27390` at horizon 5. Approximate 95% Bonferroni block-normal bands exclude zero only for log variance at horizon 5; this is explanatory and not a material-difference decision |
| Controlled calibration | Locked synthetic audit binding passed. Across 22 displayed families, minimum coverage was `0.93359`, minimum required-decision rate among gated families was `0.99023`, maximum false/boundary rate was `0.00326`, and invalid rate was zero. These are controlled-law operating characteristics, not retained-chain evidence |
| Statistical ranking | None; G/H differences are descriptive only |
| Nonclaims | No posterior oracle, posterior correctness, G/H equivalence/material difference, sampler ranking, model adequacy, or default readiness |

## Numerical Summaries

The forecast banks contain 512 paths per arm (four chains, 64 draws per
chain, two replications). The pooled observation means by horizon were:

| Horizon | G mean | H mean | G variance | H variance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03075 | 0.03019 | 0.72734 | 0.79356 |
| 2 | -0.00693 | 0.03452 | 0.74389 | 0.73390 |
| 3 | 0.14740 | 0.07245 | 0.82605 | 0.75009 |
| 4 | 0.02890 | -0.03841 | 0.74680 | 0.78278 |
| 5 | -0.00039 | 0.00548 | 0.70704 | 0.92982 |
| 6 | 0.04054 | 0.01020 | 0.77710 | 0.81790 |
| 7 | 0.07624 | -0.00873 | 0.77616 | 0.87938 |
| 8 | 0.08654 | 0.03440 | 0.82297 | 0.78793 |
| 9 | 0.05153 | 0.00773 | 0.80703 | 0.77792 |
| 10 | -0.00501 | 0.02347 | 0.71073 | 0.70391 |

The values are descriptive estimates from a confirmation-excluded pilot
prefix. They should not be read as estimates from all 512 retained draws.

## Figures and Artifacts

| Figure | Files |
| --- | --- |
| Transformed-coordinate launches | `ssl-lstm-launch-traces-z.png`, `ssl-lstm-launch-traces-z.pdf` |
| Physical four-parameter launches | `ssl-lstm-launch-traces-theta.png`, `ssl-lstm-launch-traces-theta.pdf` |
| Predictive fans | `ssl-lstm-predictive-fans.png`, `ssl-lstm-predictive-fans.pdf` |
| Horizon moment differences | `ssl-lstm-moment-differences.png`, `ssl-lstm-moment-differences.pdf` |
| Controlled calibration operating characteristics | `ssl-lstm-controlled-calibration.png`, `ssl-lstm-controlled-calibration.pdf` |

All paths are under
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/`.
The machine-readable receipt records SHA-256 hashes for every figure and
source binding.

## Run Manifest

| Field | Value |
| --- | --- |
| Command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_direct_visual_validation_2026_07_18.py --output-dir docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation --wall-cap-seconds 900` |
| Environment | conda `tfgpu`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0`; Pillow renderer |
| Device/JIT | NVIDIA GeForce RTX 4080 SUPER; TensorFlow XLA; float64; TF32 enabled |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Seeds | G `(20260718,5101)`; H `(20260718,5201)`; disjoint from calibration/evaluation domains |
| Wall time | `289.9988` seconds; cap `900` seconds |
| Plan | `docs/plans/bayesfilter-ssl-lstm-direct-visual-validation-plan-2026-07-18.md` |
| Result receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/visual-validation-result.json` |

The receipt SHA-256 is
`c3f8e3233eaeb427bb0b9bd5d86184da30f634e6fca75230487f5fdd539c2c3c`.
It binds the pre-execution/repair plan hash
`c24b17fbf2148e779a79c8488db169e00cc3dcaff76b67e02d9a7d12adabc450`.
The plan's later execution-close paragraph changed the current plan hash but
did not change the runner, inputs, seeds, outputs, figures, or receipt.

## Checks

CPU-hidden focused tests: `10 passed`. The existing target-adapter suite also
passed (`5 passed`). Python compilation and `git diff --check` passed for the
new runner and plans. Figure files were opened and visually inspected; no
orientation, truncation, or renderer failure was found.

After the ladder supervisor repair, the final combined visual, ladder, and
target-adapter suite reported `23 passed`. The warnings were upstream
TensorFlow AutoGraph/Python-3.13 deprecations, not test failures.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit visual package | All integrity and forecast hard checks passed | No hard veto | Only 64 confirmation-excluded draws per chain | Use figures for exposition and proceed to independent complexity ladder | G/H predictive equivalence or posterior correctness |
| Keep confirmation closed | Pilot boundary was respected | No authority to inspect suffix | Formal retained-chain covariance remains untested | Require a separate reviewed confirmation plan | Any G/H material/equivalence claim |
| Retain controlled calibration figure | Locked audit receipt passed | No controlled-audit gate failure | Controlled laws do not establish model adequacy | Cite as procedure calibration only | Retained-chain validity |

## Post-Run Red Team

The strongest alternative explanation for the apparent overlap is that the
pilot prefix is short and both arms target the same four-parameter chart. A
future suffix-based formal comparison could differ. The conclusion would be
overturned by any input hash drift, prefix-boundary violation, or an independent
bank/provenance mismatch. The weakest evidence is therefore external
generalization from this small, confirmation-excluded prefix.
