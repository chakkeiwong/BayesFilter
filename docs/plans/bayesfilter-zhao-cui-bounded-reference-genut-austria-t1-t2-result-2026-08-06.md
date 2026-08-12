# Zhao-Cui Bounded-Reference GenUT Austria T1/T2 Result

Date: 2026-08-06

Plan: `docs/plans/bayesfilter-zhao-cui-bounded-reference-genut-austria-t1-t2-plan-2026-08-06.md`

Status: `STOPPED_CURRENT_CANDIDATE_FAILED_HARD_VETO`

## Verdict

Zhao-Cui can be used as an independent teacher without empirical GenUT particle
moment targets. The implemented sampled T1/T2 teacher is finite, has exact
proposal-correction weights, and carries analytical tangents from the issued
Zhao-Cui marginal score. The bounded-feature GenUT composition also compiles
and executes on GPU/XLA.

The current inherited-control candidate does **not** pass the predeclared T1/T2
diagnostic. The diagonal-only arm leaves the bounded chart and is invalid. The
uncapped pairwise arm is finite but fails the parameter-0 absolute score-FD gate
(`0.10731 > 0.08`). Cap 2 is finite and all three score coordinates pass the
prospective FD thresholds, but one seed cannot rank or promote it. The planned
three-seed run was therefore not executed.

## Claimed and computed quantities

| Item | Result |
|---|---|
| Intended teacher | T1/T2 Zhao-Cui retained squared-TT marginal moments and total parameter tangents |
| Exact physical raw moments | Rejected as mathematically wrong for Lane B: the algebraic inverse and positive defensive reference density give divergent physical high moments |
| Quantity actually computed | Self-normalized 64-sample Zhao-Cui retained-marginal standardized moments in bounded reference coordinates, with exact TT/proposal correction and marginal-score weight tangents |
| GenUT value/score | T2 finite particle program on the sealed Lane-B observations, with a parameter-linear sampled-teacher target and manual total JVP |
| Equality verdict | The teacher estimator and its JVP are internally correct on focused FD tests; it is different from exact contracted or physical raw moments |
| Classification | `extension_or_invention` |

## Teacher evidence

Artifact:
`docs/benchmarks/artifacts/zhao_cui_bounded_reference_genut_austria_t1_t2_20260806/teacher-attempt01-n64/manifest.json`

| Diagnostic | T1 | T2 |
|---|---:|---:|
| Samples | 64 | 64 |
| Importance ESS | 63.869 | 63.834 |
| Minimum log correction | -0.11206 | -0.10314 |
| Maximum log correction | 0.12452 | 0.14766 |
| Maximum bounded `|u|` | 0.99799 | 0.99982 |
| Maximum marginal-score magnitude | 24.6645 | 36.3587 |

Teacher setup took `57.49 s` on deliberate CPU-only execution with
`CUDA_VISIBLE_DEVICES=-1`. High ESS shows that the retained grid proposal
closely covers the TT marginal for these fixed rows. It does not remove the
64-sample Monte Carlo uncertainty in third/fourth moments.

Strict identities:

- T1 parent:
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`;
- T2 parent:
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`;
- T1 child:
  `5a006e8f55423cb08e6b3b1b08443c6ac8fb3af1c637ff48c20ed7941cae0603`;
- T2 child:
  `17e33778c558e62972eb5bfe342e297520ab1475b3722602aeab7827c60cf263`.

## GPU/XLA smoke

Artifact:
`docs/benchmarks/artifacts/zhao_cui_bounded_reference_genut_austria_t1_t2_20260806/smoke-attempt02/result.json`

Hardware/environment:

- `/GPU:0`: NVIDIA GeForce RTX 4080 SUPER;
- TensorFlow `2.19.1`, FP32, TF32 disabled, XLA compiled;
- verified memory growth before logical-device initialization;
- `T=2`, `N=1008`, common seed `98601`;
- sealed observation hash:
  `c99064071b6613557227c6148b7353b6cde54b7462f9193dd708429328321f25`;
- smoke wall time: `67.47 s`.

| Arm | Valid | Value | Score | Pair objective | Pre/post cap RMS | FD verdict |
|---|---|---:|---|---:|---|---|
| no shape | yes | -64.43648 | `[-7.32343, 2.77082, -6.22562]` | 0.91696 | 0 / 0 | not run |
| teacher diagonal | no | N/A | N/A | 0.17493 | 0 / 0 | boundary veto |
| teacher pairwise uncapped | yes | -64.42596 | `[-6.72200, 2.86276, -6.16124]` | 0.07693 | 11.2897 / 11.2897 | fail parameter 0 absolute gate |
| teacher pairwise cap 8 | yes | -64.42596 | `[-6.85488, 2.87301, -6.16555]` | 0.07921 | 11.3091 / 6.5311 | not run |
| teacher pairwise cap 2 | yes | -64.42636 | `[-6.96894, 2.91915, -6.16836]` | 0.09324 | 10.2448 / 1.9629 | pass all coordinates |

Same-program score finite differences:

| Arm | Parameter | Absolute residual | Normalized residual | Gate |
|---|---:|---:|---:|---|
| uncapped | 0 | 0.10731 | 0.01596 | fail absolute `<=0.08` |
| uncapped | 1 | 0.00208 | 0.00073 | pass |
| uncapped | 2 | 0.00331 | 0.00054 | pass |
| cap 2 | 0 | 0.01474 | 0.00212 | pass |
| cap 2 | 1 | 0.00853 | 0.00292 | pass |
| cap 2 | 2 | 0.01526 | 0.00247 | pass |

The value differences from no shape are only about `0.010-0.011`, far below
what could support a likelihood-improvement claim from one seed. The score
changes are material, especially parameter 0, but there is no external exact
score authority at this T2 target and no statistical replication.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not run three-seed ladder | No promotion criterion; implementation smoke only | Diagonal boundary and uncapped score-FD veto fired | Sampled teacher size and inherited correction strengths | Freshly plan a control-repair diagnostic centered on cap 2 with larger teacher setup | No ranking, T20 benefit, HMC, posterior, or default readiness |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Failed for the complete candidate ladder |
| Viable candidates | no-shape baseline, pairwise uncapped/cap8/cap2 are finite; only cap2 has the executed FD pass |
| Statistically supported ranking | None; one seed only |
| Descriptive-only differences | Values, scores, residuals, cap activity, and runtime |
| Default readiness | No |
| Next evidence needed | Larger independent teacher, fresh target-specific strength/cap calibration, untouched three-seed then multi-seed validation, and T3+ teacher sequence |

## Engineering checks

Focused pre-run suite:

```text
38 passed, 2 warnings
```

New target-JVP, affine-restoration JVP, and adapter checks:

```text
4 passed, 2 warnings
```

A small end-to-end bounded-teacher finite program passed same-program FD with
absolute residuals `3.69e-5` and `1.45e-4`.

After the smoke, review found that the zero-step baseline unnecessarily passed
through the bounded transform, causing normalized displacement `9.13e-6`.
Current code bypasses the teacher exactly when all shape steps are zero and a
new exact-parity regression test covers it. The smoke artifact binds the older
source hash and is not silently upgraded.

## Negative-result classification

- Implementation failure: exact eager 36-core high-order contraction was not
  viable; the sampled independent-teacher repair was implemented instead.
- Candidate failure: the inherited diagonal and uncapped controls fail stated
  hard gates.
- Tuning failure remains plausible: cap 2 passes the executed mechanics/FD
  checks, so the result does not reject bounded Zhao-Cui teaching.
- Scientific-idea rejection: unsupported. T2, one seed, and a 64-sample teacher
  cannot reject the Zhao-Cui teacher direction.

## Post-run red team

Strongest alternative explanation: the failures are caused by transferring
empirical-target strengths to a bounded Zhao-Cui target and by noise in a
64-sample high-order teacher, rather than by the teacher concept.

Result that would overturn the stop: a fresh plan with larger independently
generated teacher samples and calibration/validation-separated correction
controls passes the boundary and same-program FD gates on an untouched run.

Weakest evidence: the teacher moments use only 64 fixed samples and the smoke
has only one particle seed. The high ESS establishes proposal coverage, not
high-order moment precision.
