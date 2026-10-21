# M10 local likelihood reference result

The sine-transition, quadratic-observation M10 reference passed all eight
declared checks. This prepares a distinct model for testing surrogate HMC;
it does not establish a full-prior parameter-posterior reference or test HMC.
The source initialization, observation scaling, transition ordering and
finite TensorFlow constant conversions were retained.

The four grids and36 calibration/validation probes gave maximum final mesh
and domain log-likelihood changes1.78e-15 and4.44e-16. Their relative score
changes were6.63e-16 and1.23e-15. Maximum omitted-mass indicator was4.44e-16,
posterior edge mass3.75e-29, independent linear Kalman error7.11e-15 and
centered finite-difference relative score error4.87e-11. Across six nonlinear
checks, Hermite64-to128 refinement was at most4.37e-9; order128 agreed with
the state-grid likelihood within1.07e-14. These are measured refinement and
agreement errors, not rigorous error bounds.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Qualify local likelihood reference | All eight checks pass | No finite/domain/derivative/independent-integration failure | Values outside the3x3 probe scope, especially normal-prior tails, remain unchecked | Qualify full DCTR consumer and full-prior posterior integration | HMC correctness, posterior accuracy or all-model coverage |

| Inference status | Finding |
|---|---|
| Hard veto screen | Local reference screens pass; no retained HMC evaluated |
| Statistically supported ranking | None; this is numerical reference qualification |
| Descriptive-only differences | Numerical refinement discrepancies listed above |
| Default-readiness | Not established |
| Next evidence needed | Full-prior posterior functional error allowances, actual DCTR mechanics, scoped calibration and public HMC tuning |

The worker was run with tftwogpu, CUDA_VISIBLE_DEVICES=-1, TF threads2/1 and
BAYESFILTER_PRELOAD_CUSTOM_OP=0, using the exact command in
`artifacts/surrogate-hmc-all-models-20260920-m10-reference-01/run-manifest.json`.
That root preserves data, four grid results, source snapshots and strict JSON
reports. It used14.375272 CPU seconds and10.092366 wall seconds. The3 numerical
tests passed using12.769667 CPU seconds. Master/dependency tests then passed30
checks using9.218621 CPU seconds; a focused accounting follow-up passed5 checks
using0.236455 CPU seconds. These consume the M10 allocation.

The master stage `--scope M10 --stage P3-M10-reference` was also executed in
`artifacts/surrogate-hmc-all-models-20260920-m10-reference-master-01` and reused
the checked reference, using6.017333 CPU seconds. Execution completed; the
master correctly returned its incomplete-program status and kept every HMC
comparison NOT_RUN. No reference calculation or retained chain was repeated.
Accounting combines standalone, nested master and test costs without charging
the nested worker twice. Numerical source/data changes invalidate evidence
reuse; a scientific reference failure requires a prospective repair plan.

Post-run red team: both quadratures share physical parameter interpretation,
so agreement cannot independently prove that mapping; the explicit source
mapping test and linear limit address it separately. The weakest evidence is
coverage outside the local probes. A tail-sensitive posterior check can
invalidate use of this grid as a global endpoint even though this local stage
passed. That would trigger reference repair, not rejection of surrogate HMC.
