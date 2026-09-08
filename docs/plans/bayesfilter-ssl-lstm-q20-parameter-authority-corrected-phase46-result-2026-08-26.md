# Corrected Parameter-Authority Phase 46 Result

Date: 2026-08-26  
Version: `v2.8-support-envelope-diagnostic`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase46-subplan-2026-08-26.md`  
Status: `PASS_V2_8_SUPPORT_ENVELOPE_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 46 tested whether a third independently generated `N=512` theta bank
falls inside the scalar support envelope of the first two N=512 banks under
the unchanged M0 proposal law. The declared target remained the batch-native
q=20 SSL-LSTM target in `theta in R^4`. The 60D UKF state remained internal.
No NeuTra trainer, objective, architecture, proposal schedule, whitening
criterion, HMC route, or canonical LEDH route changed.

This was a read-only support diagnostic. The authority cloud is the frozen
training measure; authority, A, B, C, N512-a, N512-b, and N512-c were all
retained as finite audit clouds. No cloud was pooled or used for training.

## Execution and hard gates

The fresh N512-c pilot used seed `(20260826,4606)` and calibration size 128.
It passed `PASS_THETA_MEASURE_PILOT`; its M0 and C0 receipts have the exact
target signature, `theta_R4` measure, frozen protocol hashes, and distinct
tensor hashes. The CPU-hidden report passed with status
`PASS_V2_8_SUPPORT_ENVELOPE_REPORT` and wall time `0.5605123529676348 s`.

The report recomputed every stored M0 proposal log density from the retained
geometry and obtained maximum absolute residual `0.0` for all seven banks.
All loaded tensors were finite and `[N,4]`; the report refused absolute or
parent-traversal paths and refused to overwrite its unique output root.

| Gate | Result |
|---|---|
| target and measure | passed; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, measure `theta_R4` |
| M0/C0 protocol | passed; M0 `a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631`, C0 `270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067` |
| bank independence | passed; seven pilot receipts and required tensor hashes distinct |
| proposal recomputation | passed; max residual `0.0` for every bank |
| finite support metrics | passed; all pairwise values finite |
| GPU boundary | not applicable; report intentionally hid GPU and is diagnostic-only |

## Support receipt

| Source | Rows | Roots | ESS fraction | Negative-mode mass | theta mean[0] |
|---|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 |
| bank A | 256 | 103 | 0.801812 | 0.756588 | 3.550030 |
| bank B | 256 | 128 | 0.946687 | 0.517590 | 1.013180 |
| bank C | 256 | 125 | 0.975794 | 0.565503 | 0.877022 |
| N512-a | 512 | 248 | 0.927380 | 0.403469 | 1.446191 |
| N512-b | 512 | 233 | 0.968359 | 0.501739 | 0.587732 |
| N512-c | 512 | 241 | 0.878600 | 0.503522 | -0.896535 |

The predeclared scalar envelope branch was
`n512_c_outside_two_bank_scalar_envelope`. N512-c was outside the N512-a/b
interval for ESS, negative-mode mass, theta mean[0], and several target,
proposal, and log-ratio endpoints. The branch is descriptive; it does not
turn those fields into promotion thresholds.

Selected pairwise N512 coordinate-box intersection-over-union values were
`0.660484` (N512-a vs N512-b), `0.489971` (N512-a vs N512-c), and `0.475673`
(N512-b vs N512-c). Weighted fractions of N512-c rows inside the N512-a and
N512-b coordinate boxes were `0.997977` and `0.997977`, respectively, while
weighted nearest-neighbor means were `0.853161` and `0.992964`. Coordinate
boxes can overstate support in four dimensions, so these are explanatory
only and must not be read as common-support evidence.

## Decision and inference status

| Decision | Primary criterion | Status | Veto/limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target | pilot/hash/measure/finite gates | pass | none | retain parameter authority | posterior correctness |
| promote IID whitening | finite support envelope | veto | finite empirical clouds are not a population law | keep whitening closed | IID Gaussian law |
| change objective | support diagnostic | defer | no uncertainty-supported downstream comparison | test invariant mutation/rejuvenation | objective superiority |
| admit HMC/canonical LEDH | density and downstream gates | veto | role-limited support evidence | keep routes closed | HMC/LEDH readiness |

| Inference class | Status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none |
| descriptive differences | persistent N=512 bank-to-bank support variability |
| default readiness | not ready |
| next evidence | a valid invariant mutation/rejuvenation comparison under the same target |

## Interpretation and repair

The exact proposal-log recomputation rules out stale stored proposal values as
the explanation. The three N=512 draws nevertheless have materially different
finite support summaries, while all use the same proposal family and identity
mutation. This is consistent with finite resampling noise and the absence of
rejuvenation; it does not prove that the proposal density is globally wrong.

The next smallest discriminating artifact is a target-invariant mutation
diagnostic: compare the existing identity-mutation reference with a declared
theta-space Metropolis rejuvenation kernel at each tempered stage. The kernel
must use the same target/status API, preserve the tempered target by its
Metropolis-Hastings acceptance ratio, and remain role-limited until its finite
mode/support and downstream gates pass.

## Red team and nonclaims

The strongest alternative explanation is that coordinate boxes exaggerate
overlap and that all N512 banks share the same proposal-support bias. The
weakest evidence is finite box/distance summaries in only four dimensions.
A separately generated invariant mutation route with stable bank-to-bank
behavior would weaken the no-rejuvenation explanation; it would still not
prove IID whitening or posterior correctness.

No common-support theorem, population limit, IID Gaussian law, posterior
correctness, normalizer, exhaustive mode discovery, HMC convergence,
canonical LEDH validity, superiority, or default promotion follows.

## Artifacts and manifest

- Pilot: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase46-support-envelope/fresh-n512-c/`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase46-support-envelope/report/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase46_2026_08_26.py`

The report manifest records TensorFlow 2.20.0, intentionally hidden GPU,
CPU-only diagnostic status, command, source hashes, dirty-tree state, and
artifact roots.
