# GenUT Three-Model Simple Feasibility Result

Date: 2026-07-22
Status: `aggregate_suite_status_revoked_reduced_sir_mechanics_only`
Plan: `docs/plans/bayesfilter-genut-three-model-simple-feasibility-plan-2026-07-22.md`
Artifact: `docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt01/`

Owner correction: `N=96` is below the required `N>1000` numerical-test scope.
All numerical comparisons in this note are historical mechanics diagnostics and
must not be used as current feasibility evidence. The replacement run uses
`N=1008` under `attempt02_n1008`.

Further correction: the reduced-SIR phase is an artificial boundary-stress
fixture, not an actual model. This aggregate cannot support a three-model suite
claim; the existing Chapter 18b structural model was the omitted actual target.

## Outcome

The FP32/TF32/GPU/XLA GenUT value and hand-derived recursive score executed for
all three short-prefix targets at `T=10`, `N=96`, one seed. All outputs were
finite, all transport residuals were below `1e-4`, and each recursive score
matched a central finite difference of the identical finite GenUT scalar within
the predeclared 5% scaled tolerance.

That engineering pass is not an accuracy pass:

- generalized SV is the cleanest feasibility result: its value differs from
  the fixed-branch diagnostic by `0.03452`, and its three score differences are
  `(0.03529, 0.02820, -0.002371)`;
- KSC mixture SV has close value (`-0.03431` difference) and close `z_gamma`
  score (`-0.00735` difference), but its `log_beta` score differs by `0.20207`;
- reduced SIR is executable and internally differentiated, but is not
  accurate enough at `N=96`: the value differs from the dense reference by
  `-0.91192` and the observation-noise score differs by `2.96241`.

The result supports continuing GenUT testing for all three targets. It does not
support leaderboard admission, a default change, or a ranking. SIR needs a
particle/tuning repair before a claim run; KSC needs a focused score ladder;
generalized SV is ready for a small tuning and seed/particle ladder.

## Value and score comparison

All score coordinates are in the target parameter charts shown in the table.
The Zhao-Cui generalized-SV route is diagnostic, not an oracle. KSC SGQF and
UKF target the declared KSC surrogate, not the exact native-SV likelihood.

| Target and parameter order | Route | Value | Score |
|---|---|---:|---|
| reduced preclip SIR `(log_kappa, log_nu, log_obs_noise)` | GenUT | -11.7710438 | `(-0.00594604, 0.08329159, 5.17564487)` |
| reduced preclip SIR `(log_kappa, log_nu, log_obs_noise)` | dense manual-score reference | -10.8591241 | `(-0.00460581, 0.06417235, 2.21323426)` |
| generalized SV `(z_gamma, log_tau, mu_over_tau)` | GenUT | -15.9853563 | `(-0.09018072, -0.12664723, 0.01989034)` |
| generalized SV `(z_gamma, log_tau, mu_over_tau)` | fixed-branch Zhao-Cui diagnostic | -16.0198730 | `(-0.12547017, -0.15484276, 0.02226093)` |
| KSC SV `(z_gamma, log_beta)` | GenUT | -19.9852524 | `(-0.69982243, 0.81165051)` |
| KSC SV `(z_gamma, log_beta)` | fixed SGQF | -19.9509416 | `(-0.69247488, 0.60957816)` |
| KSC SV `(z_gamma, log_beta)` | principal-square-root UKF | -19.9509416 | `(-0.69247488, 0.60957816)` |

## Same-scalar score audit

Finite difference is diagnostic only. The runtime score is the manual
recursive derivative of the same finite GenUT value program.

| Target | Maximum scaled recursive-score/FD error | Gate |
|---|---:|---|
| reduced preclip SIR | 0.0207403 | pass (`<=0.05`) |
| generalized SV | 0.000523806 | pass (`<=0.05`) |
| KSC mixture SV | 0.000174463 | pass (`<=0.05`) |

These checks show that the reported GenUT score gaps are not explained by a
missing derivative in the implemented finite scalar at the tested point. They
do not prove that the finite scalar is an accurate approximation of the target
likelihood.

## Numerical and execution diagnostics

| Target | Max mean residual | Max row residual | Max column residual | GenUT compile + main run | Allocator peak |
|---|---:|---:|---:|---:|---:|
| reduced preclip SIR | 3.8803e-5 | 1.1362e-7 | 3.3677e-6 | 4.436 s | 151,019,008 bytes |
| generalized SV | 5.9605e-8 | 1.1269e-7 | 4.1444e-7 | 2.374 s | 214,697,472 bytes |
| KSC mixture SV | 2.9802e-8 | 1.4063e-7 | 1.0431e-6 | 2.914 s | 168,688,128 bytes |

The complete sequential campaign took `54.02` seconds, mostly because the
independent FP64 dense SIR and fixed-branch generalized-SV comparators took
`10.71` and `20.22` seconds. TensorFlow recorded `/device:GPU:0`, XLA emitted a
compiled CUDA cluster, TF32 was enabled, and memory growth was configured and
verified before logical-device initialization.

## SIR correction and historical evidence warning

The pre-run target audit found two defects in the earlier reduced-SIR GenUT
setup:

1. `reduced_sir_candidate_adapter` advanced one RK4 substep of size `0.005`
   per observation interval, while the declared model interval is `0.02` and
   requires four such substeps. The adapter now uses a graph-native four-step
   `tf.while_loop`, with parity checked against the formal preclip model.
2. `docs/benchmarks/run_genut_sir_zhaocui_one_seed.py::_dataset` clips the
   initial susceptible draw before the first transition. The formal latent
   preclip law instead has `x_0=z_0` and clips susceptible `z_t` only for
   `t>=1`. This campaign generates its observations directly with
   `LatentPreclipSIRSSM.simulate_from_standard_normals`.

Consequently, prior reduced-SIR GenUT artifacts produced before these repairs,
including `genut_sir_feasibility_20260722` and
`genut_sir_zhaocui_one_seed_20260722`, remain historical mechanics evidence but
must not be used as corrected same-target value/score evidence. This campaign
still does not test the canonical clipped Austria `J=9`, `d=18` leaderboard row.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep generalized-SV GenUT as a viable experimental candidate | passed short-prefix feasibility | finite, FD and residual gates passed | one seed, `N=96`, Zhao-Cui is not an oracle | target-specific tuning, then `N`/seed ladder | accuracy, ranking, full-horizon validity |
| Keep KSC GenUT as a viable experimental candidate with score repair required | value is close; `log_beta` score gap remains | finite, FD and residual gates passed | finite-particle/reset bias versus comparator approximation | tune and run an `N` ladder focused on both score coordinates | exact-SV correctness, superiority |
| Do not advance reduced SIR directly to a claim run | mechanics pass; value and observation-noise score agreement are poor | harness and recursive derivative valid; no execution veto | `N=96` and untuned controls may be inadequate near clipping/nonlinearity | tune and run an `N` ladder against the dense reference | Austria-SIR result, GenUT rejection, leaderboard readiness |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | No execution, target, finiteness, recursive-derivative, transport, GPU, or XLA veto fired in this corrected campaign |
| Statistically supported ranking | None; one seed and one short prefix per target |
| Descriptive-only differences | All value, score, runtime, and memory differences |
| Default readiness | Not evaluated and not supported |
| Next evidence needed | Target-specific tuning; particle and seed ladders with uncertainty; untouched longer/full-horizon tests |

## Post-run red team

- Strongest alternative explanation: generalized-SV and KSC value agreement at
  `T=10` may be local to this seed and prefix; the generalized-SV comparator is
  itself approximate.
- Result that would overturn the favorable feasibility reading: score or value
  gaps that do not contract under target-specific tuning and increasing `N`, or
  failure on untouched seeds and longer horizons.
- Weakest evidence: one seed, warm-start controls transferred across models,
  and `N=96`. No confidence interval or statistical ranking is available.
- SIR-specific interpretation: the current candidate failed accuracy at this
  scope; it did not invalidate the target, harness, recursive score, or broader
  GenUT direction. The predeclared repair is a tuned particle ladder.

## Verification

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_genut_three_model_adapters.py
3 passed

python -m py_compile docs/benchmarks/run_genut_three_model_simple_feasibility.py bayesfilter/highdim/cubature_genut_adapters.py
passed

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 2513 MiB, 34 %

TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_genut_three_model_simple_feasibility.py
status: diagnostic_feasibility_pass_all_three
wall time: 54.0202 seconds
```

The aggregate JSON is
`docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt01/result.json`.
