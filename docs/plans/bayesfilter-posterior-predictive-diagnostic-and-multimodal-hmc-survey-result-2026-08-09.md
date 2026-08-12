# Posterior-predictive diagnostic and multimodal-HMC survey result (2026-08-09)

## Outcome

The diagnostic change and literature survey are complete.

1. The new diagnostic samples one empirical-posterior archive row independently
   with replacement for every output path, simulates exactly one conditional path
   from each selected row, and compares that posterior-predictive complete-path bank
   with an independent true-parameter bank.
2. Deterministic analytic tests cover a unimodal Gaussian posterior predictive and
   an unequal-weight bimodal Gaussian-mixture posterior predictive. Strong shifted,
   wrong-mixture-weight, and complete-mode-collapse alternatives are detected.
3. The SSL--LSTM runner fails closed unless its posterior artifact excludes warm-up,
   binds passed sampler diagnostics, has the exact target and physical draw tensor,
   and states that relative multimodal weights are resolved (or not applicable for a
   unimodal posterior). It uses `replication_count=1` and has no posterior-summary
   fallback.
4. The standalone 11-page LaTeX survey builds without undefined citations,
   undefined references, or overfull boxes. It distinguishes mode discovery,
   cross-mode transitions, and relative mode weights.
5. No real q=20 posterior-predictive run was performed. The current seed-B archive
   remains ineligible: all 4,000 retained states have positive observation weight,
   while a known near-equal-density stationary point has negative observation weight,
   and relative mode weights have not been resolved.

## Implementation artifacts

| Artifact | Role | SHA-256 |
|---|---|---|
| `bayesfilter/testing/posterior_predictive_tf.py` | Generic TensorFlow posterior-predictive path construction and energy-test composition | `b3ee0627ee3049dae59326fc11eaa0281e8d658355f721a4a18015ed091d6ccf` |
| `bayesfilter/testing/two_sample_energy_tf.py` | Whole-path TensorFlow/XLA energy permutation diagnostic; exact host-side symmetry projection repairs XLA pairwise roundoff | `0f9aa24f307fe8ae7da13ed31604553b96bf7f8ea7fe5fdba3a95a776f2de27b` |
| `docs/benchmarks/run_ssl_lstm_q20_posterior_predictive_energy_diagnostic_2026_08_09.py` | Fail-closed future q=20 runner | `18e95ae6a1112f36348cf7e365b1da3d1c1a173c547e9e2764fd0b7101355f94` |
| `tests/test_posterior_predictive_tf.py` | Generic mechanics and analytic unimodal/multimodal validity tests | `4b82beb50c9e504197912373bd07f536c6343bace3514f01ac55caf9b3ac9a08` |
| `tests/test_ssl_lstm_q20_posterior_predictive_energy_diagnostic.py` | SSL archive, provenance, batching, and no-fallback contract tests | `b2a7741a2dc08d0b8dad0084827f74c67416bce808c56fad0533ec042f215769` |
| `tests/test_two_sample_energy_tf.py` | Energy statistic, permutation, XLA parity, and exact-symmetry regressions | `7ea58aaa93b1a0722f470fff4b0a69b8127fc3db00375e2e170e76df813ba6d2` |

## Survey artifacts

| Artifact | Status | SHA-256 |
|---|---|---|
| `docs/surveys/multimodal_hmc_survey.tex` | 550-line standalone survey source | `5520106f6aeb546bae4c6d192884011217bdf5afc24f73000370ce2f5bd1b491` |
| `docs/surveys/multimodal_hmc_survey.bib` | Bibliography | `a435e0ba97aa5b3a5c353a780339d78c185f572d95df07b562e250d20879b2ff` |
| `docs/surveys/multimodal_hmc_survey.pdf` | Clean 11-page build | `76fe7720589fb983bce0d26a1b82cb11d1320785b09e041f6b235c5ea4ce5343` |
| `.localresources/papers/multimodal_hmc/CORPUS_AUDIT.md` | Source identities, hashes, sections inspected, code status, and provenance repair | `32637d5003b522ad06db4e0d7298a47324137a6aa274ce326b966d908194c08d` |

The corpus contains local PDFs and extracted text for ordinary HMC geometry,
replica exchange, AIS/tempered transitions, continuous tempering, geometric
tempering, Wormhole HMC, transport-map MCMC, NeuTra, SMC samplers, symmetric
tempered HMC, and thermostat-assisted continuous tempering. An initially
mislabeled Wormhole download was replaced by arXiv:1306.0063v2. A Neal 1996
download that was HTML rather than PDF is explicitly quarantined and is not treated
as inspected literature; technical tempered-transition claims are anchored to the
fully inspected Section 7 of Neal (2001).

## Test evidence

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_q20_posterior_predictive_energy_diagnostic.py \
  tests/test_posterior_predictive_tf.py \
  tests/test_two_sample_energy_tf.py \
  tests/test_ssl_lstm_complexity_predictive_tf.py
```

Result: **35 passed in 77.25 seconds**. The 7,189 warnings are existing
TensorFlow AutoGraph/`gast` deprecation warnings for Python 3.15 compatibility;
there were no test failures or numerical warnings.

The test process intentionally hid GPUs. TensorFlow/XLA was used where the APIs
declare `jit_compile=True`; this is CPU-only diagnostic evidence, not GPU evidence.

Survey build command:

```bash
cd docs/surveys
latexmk -pdf -interaction=nonstopmode -halt-on-error multimodal_hmc_survey.tex
```

Result: clean 11-page PDF, 258,012 bytes, PDF 1.5; no undefined citations,
undefined references, or overfull boxes.

## Run manifest

| Field | Value |
|---|---|
| Git commit at verification | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` |
| Verification date/time | `2026-08-09T22:40:52+08:00` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu` |
| Python | 3.13.13 |
| TensorFlow | 2.20.0 |
| TensorFlow Probability | 0.25.0 |
| CPU | 2 x AMD EPYC 7742, 128 physical cores / 256 threads |
| GPU status | Intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`; no GPU scientific run |
| XLA status | Enabled for the default energy-distance kernels and exercised by focused tests |
| Data version | Analytic generated fixtures; no real SSL--LSTM posterior campaign |
| Random seeds | Stateless two-word seed domains rooted at `20260809`, enumerated in tests and runner |
| Test wall time | 77.25 seconds |
| Plan | `docs/plans/bayesfilter-posterior-predictive-diagnostic-and-multimodal-hmc-survey-plan-2026-08-09.md` |
| Result | This file |

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Diagnostic mechanics accepted | All generic, analytic, SSL contract, and energy tests passed | No nonfinite, shape, seed-alias, one-to-many path, unresolved-weight, or XLA-symmetry failure | Fixed-seed tests are regression fixtures, not a Type-I calibration study | Use the runner only after an eligible posterior authority exists | Equality, HMC correctness, or posterior correctness |
| Survey accepted | Primary sources inspected; clean LaTeX build | Mislabeled and invalid downloads repaired/quarantined | The survey is bounded, not exhaustive, and methods were not benchmarked locally | Plan a known-mixture comparison before implementing a real-target sampler | A promoted multimodal HMC default or performance ranking |
| Real q=20 diagnostic blocked | Required posterior artifact is absent | Relative multimodal weights unresolved | Mode completeness and relative posterior mass | Establish global multimodal evidence using exact tempering plus AIS/SMC evidence | That seed B represents the posterior mixture |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Diagnostic implementation and survey build passed; current seed-B posterior authority failed eligibility for a real run |
| Statistically supported ranking | None; no stochastic method comparison was run |
| Descriptive-only differences | None generated in this task; literature performance reports were not promoted to BayesFilter evidence |
| Default readiness | Not ready; the new diagnostic is a validated diagnostic path, while no multimodal HMC method is a new repository default |
| Next evidence needed | Known-weight, equal/unequal-scale mixture fixtures; exact parallel-tempered fixed HMC baseline; symmetric-schedule THMC candidate; independent AIS or annealed-SMC mass evidence with uncertainty |

## Negative-result classification and post-run red team

The inability to run the real diagnostic is not an implementation failure and does not
reject posterior-predictive assessment. It is an upstream evidence failure: the only
available retained archive does not cover the known competing mode, so its empirical
mixture is wrong relative to the requested posterior-predictive target unless that
mode's mass is proved negligible. The new runner correctly treats this as a veto.

The strongest alternative explanation for the apparent SSL--LSTM multimodality is a
target, coordinate, or gradient implementation mismatch. Both stationary points must
therefore be evaluated with the same canonical target and derivative authority before
interpreting them as posterior modes. Evidence that would overturn the present
blocker is either (a) a checked derivation/artifact showing the negative-weight point
is not a valid target mode, or (b) a globally validated sampler/weighted method that
resolves its posterior mass.

The weakest remaining evidence is mode completeness. Parallel tempering, AIS, and
SMC can all miss an undiscovered region. Their agreement is valuable only when their
discovery routes, initialisations, and failure modes are sufficiently independent.
