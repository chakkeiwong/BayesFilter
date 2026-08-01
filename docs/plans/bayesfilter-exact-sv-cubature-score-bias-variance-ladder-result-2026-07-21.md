# Exact-SV Cubature Score Bias/Variance Ladder Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`

> **Correction, 2026-07-22:** The observation fixture was not generated from
> the SV model.  The claimed persistent SV score bias, its per-time
> localization, the accuracy gates, and the antithetic SV interpretation are
> wrong relative to an SV scientific claim and revoked.  Numerical conditional
> calculations remain reproducible but scientifically irrelevant to SV
> performance.  Only engineering evidence survives.  See
> `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.

## Outcome

The reviewed `T=50` scalar Cubature campaign completed on GPU/XLA for
`N={250,500,1000,2000}`, with fresh tuning at every scope and 16 untouched
common particle seeds. A separately tuned `N=1000` antithetic arm also
completed. All engineering gates passed.

The exact dense reference is:

```text
value                         = -96.441698858790
score[probit_gamma]           =   1.279444117397
score[log_beta]               =  12.945755924126
```

Changing the dense grid from order 257/radius 8 to order 401/radius 10 changed
the value by about `1.0e-10` and either score by at most `1.0e-9`. Dense score
increments sum to the total within `3.6e-15`. The oracle refinement gate passed.

## Standard Cubature Ladder

Value intervals are coordinate-wise 95% intervals. Score intervals are the
predeclared Bonferroni familywise 95% intervals over the two score coordinates.
The SD column is individual complete-run variability; interval width concerns
the 16-run mean.

| N | Value error mean (SD) | Value 95% CI | Gamma error mean (SD) | Gamma familywise CI | Beta error mean (SD) | Beta familywise CI |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | -0.08224 (0.33937) | [-0.26308, 0.09860] | 0.03231 (0.17833) | [-0.07870, 0.14331] | -0.00059 (0.38257) | [-0.23873, 0.23755] |
| 500 | 0.00145 (0.30309) | [-0.16006, 0.16296] | 0.05528 (0.15923) | [-0.04384, 0.15439] | -0.07743 (0.35380) | [-0.29767, 0.14280] |
| 1000 | 0.01508 (0.13327) | [-0.05593, 0.08609] | 0.09637 (0.11499) | [0.02479, 0.16795] | -0.05383 (0.21081) | [-0.18506, 0.07739] |
| 2000 | 0.02186 (0.10895) | [-0.03619, 0.07991] | 0.08330 (0.06774) | [0.04114, 0.12547] | -0.11152 (0.14134) | [-0.19950, -0.02354] |

The `N=2000` value interval includes zero with half-width `0.05805`, so its
value gate passes. Both score intervals exclude zero, so the predeclared score
gate and overall promotion gate fail.

The previous `N=1000` post-run gamma signal replicated under completely new
claim seeds: the earlier mean error was approximately `0.09654`, and this run
found `0.09637`. That agreement is descriptive because the second campaign was
not designed as an independent-data replication, but it makes a serialization
or one-seed explanation implausible.

## Particle Scaling

Individual-run SD decreased substantially from `N=250` to `N=2000`:

| Quantity | SD at N=250 | SD at N=2000 | Ratio | Ideal `sqrt(250/2000)` |
|---|---:|---:|---:|---:|
| Value error | 0.33937 | 0.10895 | 0.321 | 0.354 |
| Gamma score error | 0.17833 | 0.06774 | 0.380 | 0.354 |
| Beta score error | 0.38257 | 0.14134 | 0.369 | 0.354 |

This is strong descriptive evidence that larger `N` reduces single-run
variance at approximately the ordinary Monte Carlo rate over the full ladder.
It does not show that larger `N` reduces mean error. The paired bootstrap for
`abs(mean error at N=2000) - abs(mean error at N=250)` included zero for value
and both scores:

| Quantity | Observed change | Paired bootstrap 95% interval | Supported improvement? |
|---|---:|---:|---|
| Value | -0.06038 | [-0.23041, 0.04958] | No |
| Gamma score | 0.05099 | [-0.02804, 0.09193] | No |
| Beta score | 0.11093 | [-0.11037, 0.14525] | No |

The ladder compares separately tuned scopes. `N=250,500,1000` selected
`epsilon=2`, 8 Sinkhorn steps, ridge `1e-5`; `N=2000` selected the same epsilon
and ridge but 4 Sinkhorn steps. Therefore the endpoint comparison is the
performance of each tuned finite program, not a pure fixed-control `N` effect.
The `N=2000` selection difference was driven by a tiny two-validation-seed
value advantage; it is a limitation, not evidence that four steps are superior.

## Antithetic Ablation

The separately tuned `N=1000` antithetic estimator averaged complete runs at
`Z` and `-Z`. It selected `epsilon=2`, 8 Sinkhorn steps, and ridge `1e-5`.

| Quantity | Standard SD | Antithetic SD | SD ratio | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Value error | 0.13327 | 0.04701 | 0.353 | [0.226, 0.776] |
| Gamma score error | 0.11499 | 0.06686 | 0.581 | [0.395, 0.883] |
| Beta score error | 0.21081 | 0.10857 | 0.515 | [0.342, 0.819] |

All three bootstrap upper endpoints are below one, supporting variance
reduction for this fixed-data scope. But antithetic averaging did not repair
accuracy:

| Quantity | Mean error | 95% interval used for the gate |
|---|---:|---:|
| Value | 0.03971 | [0.01466, 0.06476] |
| Gamma score | 0.10142 | [0.05980, 0.14303] |
| Beta score | -0.06792 | [-0.13550, -0.00034] |

Reducing variance made the systematic discrepancies easier to detect. The
antithetic estimator is also a different finite scalar, so this result does not
retroactively change the standard estimator.

## Per-Time Localization

At `N=2000`, cumulative gamma error becomes persistently positive around
zero-based time 29 (observation 30). Cumulative beta error becomes persistently
negative around time 44 (observation 45). The largest mean increment errors are
distributed across observations near 6, 21, 26, 35, and 48. The discrepancy is
therefore accumulated through the finite filter and repeated resets rather than
created solely by the final update.

This supports Fable's transferable mechanism: process-path variation and
finite-cloud differences propagate through later likelihood increments. It
does not establish that Contract E restoration itself is wrong. Sinkhorn and
reset residuals remained small, and same-scalar derivative checks passed.

## Engineering Ledger

| Check | Result |
|---|---|
| GPU/XLA/TF32 placement | Pass for all claim rows |
| TensorFlow memory growth | Configured and verified before initialization |
| Finite value and score | Pass for all claim rows |
| Maximum row residual | `1.53e-6` at `N=2000` |
| Maximum column residual | `4.08e-5` at `N=2000` |
| Score-increment sum residual | At most `3.38e-6` across all scopes |
| Calibration recursive-versus-FD error | At most `0.00938`, below `0.05` |
| Focused pre-run tests | `44 passed` |
| Total campaign wall time | `128.963 s` |
| Maximum claim-row allocator peak | `67,532,032` bytes at `N=2000` |

The allocator peak is a TensorFlow process measurement, not a hard memory cap.
Every scope wrote and then consumed a scope-specific frozen tuning artifact.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject score promotion for the current scalar Cubature candidate | Fail: both `N=2000` familywise score-error intervals exclude zero | Engineering and dense-oracle vetoes all pass | One fixed observation sequence; separately tuned controls confound a pure `N` mechanism | Diagnose deterministic finite-filter bias before testing more models | No rejection of Cubature/GenUT research direction or Contract E canonical route |
| Retain antithetic averaging as an experimental variance option | Pass: all paired-bootstrap SD-ratio upper endpoints are below one | Finite, GPU, residual, and tuning gates pass | It changes the finite estimator and exposes nonzero means | Keep as optional ablation after bias repair | No accuracy repair or default promotion |

## Inference Status

| Question | Status |
|---|---|
| Hard veto screen | Pass: harness, dense oracle, GPU/XLA placement, finiteness, residuals, and tuning provenance are valid |
| Statistically supported ranking | Antithetic variance is lower than standard at `N=1000` under the paired bootstrap; no supported `N=250` to `N=2000` mean-accuracy improvement |
| Descriptive-only differences | Intermediate-`N` mean changes, runtime, extrema, tuning near-ties, and approximate `1/sqrt(N)` SD scaling |
| Default readiness | Fail; Cubature remains experimental and Contract E remains canonical |
| Next evidence needed | A fresh-data fixed-control mechanism experiment that separates finite-cloud bias from control selection and reset/transport contributions |

## Negative-Result Classification

- Implementation failure: no evidence.
- Tuning failure: not proved, but the two-seed, value-only selection is weak and
  the changed `N=2000` Sinkhorn count is a material alternative explanation.
- Diagnostic failure: no evidence; dense score refinement and score-increment
  summation passed.
- Evidence against the scientific idea: the current candidate's score accuracy
  is weakened, not the broader Cubature/GenUT direction. Variance reduction is
  viable, while mean-score accuracy needs a mechanism repair.

## Post-Run Red Team

The strongest alternative explanation is that per-scope value-only tuning
selected controls that are poor for score bias, particularly the four-step
`N=2000` route. The claim seeds cannot now be used to select another arm.
A fresh-data, predeclared fixed-control comparison would distinguish this from
an intrinsic finite-reset bias. The conclusion would be overturned if that
experiment centered both score errors at large `N` or if an independent dense
implementation contradicted the refined oracle. The weakest evidence is the
two-seed tuning ranking; the strongest evidence is the stable dense oracle and
the 16 untouched complete-run score intervals.

## Artifacts

- Plan: `docs/plans/bayesfilter-exact-sv-cubature-score-bias-variance-ladder-plan-2026-07-21.md`
- Result: `docs/benchmarks/artifacts/cubature_exact_sv_score_ladder_20260721/attempt01/result.json`
- Manifest: `docs/benchmarks/artifacts/cubature_exact_sv_score_ladder_20260721/attempt01/run_manifest.json`
- Dense oracle: `docs/benchmarks/artifacts/cubature_exact_sv_score_ladder_20260721/attempt01/dense_reference.json`
- Per-scope tuning and claim artifacts: same output root, named `tuning_*.json` and `claim_*.json`
- Result SHA-256: `d43109a7e70c416c62dd9c16984cfb49e43377cba67eeced666513f735eddebd`
- Git commit recorded by the run: `0fff464ab456b72a010007e552c1e2d761624afe`
