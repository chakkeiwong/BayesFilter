# P2 Target-Repair Skeptical Audit

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `PASS_FOR_SVX_SGQF_ADMISSION_LADDER_SVX_ZC_BLOCKED`

## Decision

P2 may implement and run a bounded target-admission ladder for `SVX-SGQF`.
`SVX-ZC` remains cell-local `TARGET_BLOCKED`: its current factorized fixed-grid
wrapper is an `extension_or_invention` and no production-admissible fixed
source-route implementation exists. That blocker does not stop SGQF work.

No HMC, NeuTra training, recipe screen, or model-cell promotion beyond target
and value/score admission is authorized by this audit.

## Repaired Assumptions

| Risk | Finding | Repair |
| --- | --- | --- |
| Prior invention | P0 had no parameter prior; using an arbitrary Gaussian would redefine the posterior. | Use Zhao-Cui Section 6.2 synthetic prior: independent physical `gamma,beta ~ Uniform(0.1,0.9)`. |
| Chart mismatch | Historical BayesFilter uses `(Phi^-1(gamma), log(beta))`, while author code maps both physical parameters with `0.1+0.8 Phi(u)`. | Freeze the source-grounded two-probit chart and include its complete Jacobian; this induces independent standard-normal density in `u`. |
| Data drift | P0 described seed 81101/T=1000 but had not frozen data. | Replay the existing P8 manifest row and require raw-observation serialized SHA-256 `5e2423149e4f59eb588ccc7f16ec6d9ee984ccc4710a3ae07a3dbcf5c37db748`; truth is explanatory only. |
| Silent SGQF level | Level 2 is a historical convenience and existing tests show a level ladder. | Screen levels 2, 4, 6, and 8 before selecting the smallest passing level. |
| Wrong baseline | The dense T=1000 order-401 recurrence is unnecessarily expensive and not XLA-oriented. | Use dense exact reference on a frozen T=20 prefix plus full-T=1000 level-convergence comparisons. |
| Python target loops | Current exact-SV SGQF wrappers use Python time/axis loops and eager checks. | Implement one scalar-panel batch-native TensorFlow `tf.while_loop` value/score/status adapter; no scalar fallback or callback. |
| Proxy promotion | Level convergence alone could hide common bias. | Require both prefix dense-reference agreement and full-horizon successive-level agreement; score must also match centered finite differences on the same fixed branch. |
| Enhanced family gap | Only the plain dense-IAF learned family is implemented. | Record `UNAVAILABLE_CAPABILITY_NOT_EXECUTED`; do not fabricate a second family. Target/filter admission continues, but cell-level candidate rejection remains impossible. |

## Frozen Admission Contract

Question: can one graph-native fixed-SGQF route define the source-grounded
synthetic exact-transformed-SV posterior on the frozen T=1000 data without
target, score, batching, or XLA drift?

Candidate levels: `2, 4, 6, 8`. Reference level: `10`, explanatory only for
full-horizon convergence. Dense prefix: first 20 raw observations, Legendre
order 401, radius 8.

Parameter audit points in source chart:
`(-1,-1)`, `(-1,1)`, `(0,0)`, `(1,-1)`, `(1,1)`, and the transformed physical
truth. These are fixed before results.

Primary level rule: choose the smallest candidate that simultaneously has:

- dense-prefix maximum absolute per-observation log-likelihood gap `<= 1e-3`;
- dense-prefix maximum absolute score gap to centered finite difference
  `<= 1e-5`;
- full-horizon maximum absolute per-observation value gap to both the next
  candidate and level 10 `<= 1e-4`; and
- full-horizon maximum absolute score-coordinate gap to both the next candidate
  and level 10 `<= 1e-3`.

If no level passes, `SVX-SGQF` remains `TARGET_BLOCKED` and the observed ladder
is diagnostic evidence, not permission to loosen thresholds after inspection.

Hard vetoes: dataset hash mismatch; any zero raw observation; wrong prior/chart;
nonfinite or nonpositive variance; batch/permutation failure; value/score
mismatch; callback or Python sample/time loop in active target; XLA failure;
target identity substitution; or incomplete independent recomposition.

Not concluded: SGQF is exact, superior, calibrated, HMC-ready, trainable by the
current recipe, or suitable for another dataset/model/filter.

## Compute And Commands

CPU-hidden implementation tests and dense prefix reference are bounded to four
CPU-hours. The XLA/full-horizon level ladder is bounded to the P2 cell-admission
bucket and must use trusted GPU with memory growth. Output roots are fresh under
`phase-p2/SVX-SGQF/target-admission/attempt-<nn>-<timestamp>/`.

The next implementation result must freeze exact commands before running the
level ladder. A successful ladder may issue a typed target identity and advance
only `SVX-SGQF` to `POSTERIOR_IDENTITY_ADMITTED`; HMC remains a later rung.

Frozen CPU check:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl pytest -q tests/test_exact_sv_sgqf_neutra_target.py tests/test_neutra_campaign.py
```

Frozen trusted ladder command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/run_multimodel_neutra_p2_svx_sgqf_admission.py --output-root docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p2/SVX-SGQF/target-admission/attempt-01-<UTC_TIMESTAMP>
```
