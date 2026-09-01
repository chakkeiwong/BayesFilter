# Corrected Parameter-Authority Phase 37 Result

Date: 2026-08-25  
Continuation version: `v2.1-training-measure-bound`  
Status: `PASS_SUPPORT_LADDER_HARD_GATES_NEUTRA_SUPPORT_STRESS_ROLE_LIMITED`

## Scope

Phase 37 tested the smallest support hypothesis retained by the Phase 36
adjudication: whether fresh parameter-space banks at `N=64,128,256` improve
theta-measure support enough to change the downstream NeuTra boundary result.
The target remained the batch-native q=20 SSL-LSTM target in
`theta in R^4`. The 60-dimensional UKF state and 20-dimensional innovation
remained internal target-evaluation quantities. No replay, HMC, canonical
LEDH, or posterior calculation was launched.

The initial aggregate attempt failed before interpreting data because the
reporter used the stale calibration key `particles`; the runner expects
`particle_count`. The failed root is preserved at
`phase37-support-ladder/aggregate/`. The repaired aggregate is at
`phase37-support-ladder/aggregate-attempt2/` and is the only aggregate used
below.

## Skeptical pre-run audit

The ladder used one fixed target signature, the two fixed protocol hashes, the
same theta density convention, the same beta schedule, and fresh roots for
each size. The N values and seeds were hypotheses, not defaults. ESS, mode
fractions, masses, and root counts were classified as descriptive diagnostics;
only finite/status/shape/protocol gates could pass the phase. The downstream
NeuTra screen was role-limited and could not promote whitening.

The audit found no baseline, measure, or artifact defect after the aggregate
repair. It did find the expected risk that larger banks can add distinct
roots without representing the target geometry. That risk is carried into
the next objective/checkpoint repair rather than hidden by the ladder result.

## Receipts

| Item | Receipt |
|---|---|
| N=64 pilot | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n64/` |
| N=128 pilot | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n128/` |
| N=256 pilot | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256/` |
| aggregate | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/aggregate-attempt2/` |
| N=256 identity screen | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/neutra-identity/` |
| N=256 affine screen | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/neutra-affine/` |

All pilot arms reported target signature
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, measure
`theta_R4`, and protocol hashes
`270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067`
(C0) and
`a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631`
(M0). Each pilot had finite target/proposal terms, valid status, a reached
`beta=1`, and `[N,4]` rows.

## Support ladder

| N | Arm | ESS fraction | weighted negative-mode fraction | log mass estimate | unique terminal roots | wall seconds |
|---:|---|---:|---:|---:|---:|---:|
| 64 | C0 | 0.854829 | 0.170476 | -33.625718 | 27 | 101.87 |
| 64 | M0 | 0.914811 | 0.318019 | -34.042833 | 25 | 101.87 |
| 128 | C0 | 0.976364 | 0.494749 | -34.904778 | 73 | 193.31 |
| 128 | M0 | 0.874032 | 0.604848 | -33.277635 | 47 | 193.31 |
| 256 | C0 | 0.972233 | 0.460054 | -34.680543 | 143 | 348.25 |
| 256 | M0 | 0.952283 | 0.530069 | -34.688825 | 122 | 348.25 |

The N=256 M0 root was nominated for the downstream screen because it had the
largest retained root count. That is a nomination rule, not a statistical
ranking. The single seed per size and changing particle count do not support
an uncertainty-calibrated comparison.

## Downstream N=256 boundary

Both identity and train-measure-bound affine screens passed all engineering
gates: GPU memory growth was verified on both visible RTX 4080 SUPER devices,
XLA was enabled, the optimizer batch had 232 rows in `R^4`, target/status
checks were finite, and transport/log-determinant round trips were below
`2e-15`. No HMC was launched.

The final validation diagnostics at step 200 were:

| Precondition | Arm | validation loss | max abs latent mean | max abs off-diagonal covariance |
|---|---|---:|---:|---:|
| identity | compact | 8.892229 | 1.614304 | 0.539865 |
| identity | wide, low LR | 9.168333 | 1.794594 | 0.597478 |
| affine, exact train measure | compact | 6.060105 | 0.970912 | 0.772473 |
| affine, exact train measure | wide, low LR | 6.701205 | 1.193196 | 0.864401 |

The affine training oracle itself was exact to floating-point precision
(maximum mean residual `5.55e-17`, covariance residual `1.33e-15`). That
proves only the finite empirical train-measure construction. It does not
prove that the validation or target law is Gaussian. The affine arm's lower
loss is descriptive and is not a superiority claim.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain theta-space support path | all size roots satisfy common measure/status gates | no continuation veto | support may still be geometrically incomplete | continue with a frozen N=256 objective/checkpoint repair | no posterior or mode theorem |
| Nominate N=256 M0 for repair | largest retained-root count under identical protocol | promotion veto remains open | root count is not target coverage | use it as a fixed diagnostic bank only | no superiority |
| Admit identity/affine as NeuTra candidates | engineering boundary passes | whitening promotion veto remains open | held-out residuals persist | evaluate validation-selected checkpoints on untouched audit rows | no IID Gaussian or HMC claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for all six pilot arms and both N=256 boundary arms |
| Statistically supported ranking | none; one seed per size and no paired uncertainty analysis |
| Descriptive-only differences | ESS, mode fractions, masses, root counts, losses, and moment residuals |
| Default readiness | not ready |
| Next evidence needed | a predeclared, disjoint validation/checkpoint rule and untouched audit diagnostics |

## Red-team and stop classification

The strongest alternative explanation is that the fixed weighted forward-KL
objective fits the empirical training bank but does not identify the held-out
target geometry; a second is that the 12-row validation set is too small for
stable checkpoint selection. The result that would overturn the next repair
hypothesis is persistent audit failure after a frozen validation-selected
checkpoint on a fresh bank or a direct common-support failure.

This phase did not invalidate the target, harness, or corrected measure. It
found no continuation veto. Poor whitening remains a promotion veto and a
repair trigger. The next phase therefore tests objective/checkpoint selection
without changing the target, measure, proposal, or canonical LEDH boundary.

## Nonclaims

- No IID-Gaussian whitening claim.
- No exhaustive mode-discovery or posterior-correctness claim.
- No SMC-U authority admission, HMC readiness, canonical LEDH status, or
  default promotion.
