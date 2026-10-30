# M10 reference quadrature repair result

The replacement quadrature passed all148 parameter/data comparisons, including
the tails that invalidated the uniform grid. This qualifies the declared
reference screen. It does not yet qualify a normalized model posterior or
establish surrogate-HMC accuracy. The goal remains testing an imperfect full
canonical DCTR force with endpoint-corrected HMC.

The [plan](surrogate-hmc-nonlinear-quadrature-repair-2026-09-21.md) integrates
the same two-observation sine-transition/quadratic-observation likelihood,
retaining both observation roots. Four levels produced592 likelihood/score
evaluations on frozen calibration and validation data. Claim data was untouched.
Maximum20/32 Legendre differences were9.886e-10 in log likelihood and1.908e-7
in total noise derivatives, below1e-6 and1e-5. At fixed20 Legendre nodes,
64/128 Gaussian-convolution differences were2.823e-12 and5.397e-11. Agreement
with the qualified local state grid was within3.220e-15 and6.218e-15. Both
convolution formulas agreed across the switch within2.221e-15. The largest
relative inverse residual was2.019e-16. No numerical veto fired.

Two focused tests passed, including polynomial integration and finite-difference
derivatives of the whole finite integral. The first test attempt exposed an
XLA DT_VARIANT failure in differentiated Newton loops under a conditional;
fixed16-iteration graph unrolling repaired compilation without changing the
iteration or falling back from XLA. Both test attempts remain archived.

Execution used TensorFlow FP64/XLA, CPU-only with GPU deliberately hidden,
two intra-op threads and one inter-op thread. The command was
`scripts/qualify_surrogate_hmc_nonlinear_quadrature.py --output docs/plans/artifacts/surrogate-hmc-all-models-20260921-m10-quadrature-01`
under the tftwogpu Python environment. Execution consumed65.667024707 CPU-worker
seconds and40.135758275 wall seconds; preparation tests consumed20.287408819
CPU-worker seconds. Total85.954433526/200 CPU seconds; GPU0. Full commands,
source snapshots, seeds, data hashes,592 rows,148 comparisons and accounting
are in the [run manifest](artifacts/surrogate-hmc-all-models-20260921-m10-quadrature-01/run-manifest.json)
and [report](artifacts/surrogate-hmc-all-models-20260921-m10-quadrature-01/report.json).

| Decision | Primary criterion | Numerical vetoes | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Reference repair screen passes | All148 refinements and local overlaps pass | None observed; inverse and switch checks pass | Probe coverage and posterior integration error | Qualify normalization, posterior functions and parameter tails | Full-prior reference or surrogate-HMC correctness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Uniform-grid full-prior candidate failed; replacement passes its screen |
| Statistically supported ranking | None; this is deterministic integration validation |
| Descriptive differences | Timings and numerical errors, no sampler ranking |
| Default-readiness | Not established |
| Next evidence needed | Posterior normalization/functions/tail bounds, scoped DCTR controls, public tuning and retained comparisons |

Post-run red team: two refinements can share an error. Independent convolution
refinement, local state-grid agreement and finite differences reduce that risk
but do not cover every parameter pair or normalization. The weakest evidence is
the untested space between and beyond probes. A failed full-prior integration
would overturn the reference's eligibility without rejecting surrogate HMC.
