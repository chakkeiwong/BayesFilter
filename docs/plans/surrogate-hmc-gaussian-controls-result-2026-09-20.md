# Gaussian mechanism controls: retained result

Exact, half-strength and rotational proposal fields each completed four chains
with 2,000 retained draws per chain. All three comparisons with the analytic
Gaussian posterior are **INCONCLUSIVE**. Zero force failed the public tuner's
fresh-chain R-hat check and produced no retained draws. These results do not
reject surrogate HMC or establish posterior agreement: some uncertainty checks
failed, and several simultaneous intervals exceeded their equivalence margins.

The research goal remains testing whether imperfect full canonical DCTR fields
support endpoint-corrected parameter inference across repository models. C01 is
a mechanism control with no DCTR computation. Its target has precision
diag(1,4,9), and its fields are Qq, 0.5Qq, Qq+0.2Aq and zero. The antisymmetric
matrix A and the reversibility derivation are in the
[frozen plan](surrogate-hmc-gaussian-controls-2026-09-20.md). The rotational
field is not a gradient field. All arms use the exact Gaussian endpoint.

## Implementation and retained evidence

All 36 actual GPU/XLA qualification checks passed, including graph parity,
momentum-flip involution, independent rows, deterministic replay and exact
per-chain endpoint/energy correction. Each field used separate public tuning
and verified retained admission. Warm-up and verification draws were excluded.
All 24 retained chunks passed independent archive hash inspection. There were
no divergences, force fallbacks, numerical endpoint failures or rounding flags.
The relevant 54 CPU tests passed; an earlier run with two stale expectations
was repaired and its log preserved. CPU tests intentionally hid GPU devices.

| Field | Public tuning | Retained draws per chain | Final result | Maximum parameter R-hat | Minimum bulk / tail ESS | Intervals inside margins | Qualified MCSE functions |
|---|---|---:|---|---:|---:|---:|---:|
| Exact | Passed | 2,000 | INCONCLUSIVE | 1.00044 | 2,868 / 4,640 | 9/15 | 13/15 |
| Half-strength | Passed | 2,000 | INCONCLUSIVE | 1.00329 | 785 / 1,303 | 0/15 | 13/15 |
| Rotational | Passed | 2,000 | INCONCLUSIVE | 1.00094 | 3,185 / 4,483 | 9/15 | 11/15 |
| Zero | Fresh R-hat failed | 0 | TUNING_REJECTED | No retained diagnostic | N/A | N/A | N/A |

The interval counts describe geometry only. Every retained arm has
`diagnostic_valid=false`, so these counts do not establish equivalence. No
interval lay wholly outside its margin. All retained parameter R-hat/ESS,
function R-hat and MCSE sensitivity checks passed. Failures were negative
lag-one correlations of the largest batch means outside the absolute limit
0.3. There are only ten batches per chain. Exact-force examples were -0.314
for mean w2 and -0.528 for CDF90 w2. Finite-sample variability of this veto
needs separate calibration; it does not justify retrospective relaxation.

Interval width is a separate limitation. Exact-force CDF50 w0 had difference
0.00075 and interval [-0.03734, 0.03884], wider than its 0.02 margin. Analytic
expectations remove quadrature uncertainty, not Monte Carlo or floating-point
error. The protocol covers four arms, fifteen functions and two looks, using
batch means under a stationary-chain CLT assumption.

Acceptance across chains ranged from 0.752 to 0.7625 for exact force, 0.704 to
0.7325 for half-strength and 0.7505 to 0.764 for rotational. Maximum absolute
energy errors were approximately 5.78, 5.40 and 5.20. These are descriptive.
Zero force passed its other fresh verification checks; the rejection applies
to this tuning configuration, not every possible zero-force configuration.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C01 attempt 1 | Equivalence unresolved at planned limit | Some MCSE checks failed; zero tuning rejected | Interval width and correlation-veto calibration | Preserve both looks; diagnose the veto using independent known-reference draws | Posterior bias or surrogate-HMC success/failure |
| Continue all-model work | Mechanism checks passed | No corruption or implementation veto observed | Model coverage and cost | Prepare further scoped consumers | DCTR or all-model validation |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Zero tuning failed fresh R-hat; retained arms failed some MCSE checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, ESS, energy, timing and interval counts |
| Default-readiness | Not established; conditional heuristic dominance was NOT_RUN |
| Next evidence needed | Calibrated uncertainty diagnostics and adequate prospective precision; actual model-specific DCTR inference and conditional heuristics |

The strongest alternative explanation is a conservative, poorly calibrated
uncertainty veto combined with insufficient precision. Exact force itself
did not pass, so this evidence cannot isolate an effect of force approximation.
A prospective protocol that resolves known-reference and then model posterior
agreement could overturn this conclusion. No extra draws, looks or changed
criteria are applied to this closed attempt.

## Reproducibility and cost

Execution used commit 8a4cffa2fd085f43ab5710c24d6e7b48770aea0d plus recorded
source changes, tftwogpu, FP64 GPU/XLA, GPU
GPU-68251639-fe82-8f81-3ccc-2953c32e805b and verified memory growth. Manifests
and worker reports preserve exact commands, seeds, source snapshots, reference
arrays and device settings. Peak inference allocator usage was 12,523,776
bytes. Forecast cost was 559.112 GPU seconds and 558.429 CPU-worker seconds.
Actual qualification and inference charged 161.404 GPU seconds; the complete
C01 tranche including tests, inspection and restart charged 219.996 CPU-worker
seconds. The restart reported the closed result without new sampling.

Workers exited 0; master exit 2 means all-model coverage remains incomplete.
Whole-campaign charges are **8,276.791 GPU seconds and 8,677.214 CPU-worker
seconds**, leaving 218,523.209 and 45,322.786 seconds under the 63/15-hour caps.

Evidence: [qualification](artifacts/surrogate-hmc-all-models-20260920-c01-qualify-01/P3-C01-qualify/report.json),
[retained report](artifacts/surrogate-hmc-all-models-20260920-c01-inference-01/P3-C01-inference/report.json),
[manifest](artifacts/surrogate-hmc-all-models-20260920-c01-inference-01/run-manifest.json),
[accounting](artifacts/surrogate-hmc-all-models-20260920-c01-inference-01/accounting.json),
[inspection](artifacts/surrogate-hmc-all-models-20260920-c01-inspection-01/inspection.json),
[restart](artifacts/surrogate-hmc-all-models-20260920-c01-closed-01), and
[tests](artifacts/surrogate-hmc-all-models-20260920-c01-tests-02).
