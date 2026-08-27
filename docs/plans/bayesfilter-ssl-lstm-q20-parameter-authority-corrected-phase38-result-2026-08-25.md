# Corrected Parameter-Authority Phase 38 Result

Date: 2026-08-25  
Continuation version: `v2.1-training-measure-bound`  
Status: `PASS_CHECKPOINT_SELECTION_AUDIT_RECEIPT_REPAIR_TRIGGERED`

## Question and scope

Phase 38 tested whether a validation-selected checkpoint reduces the
untouched audit moment residual for the nominated N=256 M0 theta bank. The
target, proposal density, train/validation/audit split, affine construction,
architecture arms, and GPU/XLA boundary were unchanged. Selection used only
the validation rows. The audit rows were evaluated only after selection.

This phase is a finite optimization diagnostic. It is not a transport
admission, posterior calculation, IID test, HMC run, or canonical LEDH route.

## Skeptical audit and implementation repair

The first version of the proposed checkpoint phase could not have answered its
own question because the existing Phase 37 receipts stored only the terminal
state. The runner was repaired to record audit moments at explicitly requested
checkpoints while leaving the optimizer and selection data unchanged. The
repaired runner and reporter compiled, refused output overwrites, and emitted
complete manifests. Both fresh GPU/XLA traces passed all boundary gates.

Receipts:

- identity trace: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/identity-trace/`
- affine trace: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/affine-trace/`
- selection report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/report/`

The fixed rule was

`S_t = validation_loss_t + validation_latent_mean_max_abs_t + validation_latent_covariance_max_abs_offdiag_t`

for `t in {1,5,10,25,50,100,150,200}`, with smallest-step tie breaking.

## Result

| Precondition | Arm | selected step | selected validation score | selected audit mean | selected audit covariance | terminal audit mean | terminal audit covariance |
|---|---|---:|---:|---:|---:|---:|---:|
| identity | compact | 200 | 10.792858 | 0.222255 | 0.524386 | 0.222255 | 0.524386 |
| identity | wide, low LR | 200 | 11.647487 | 0.340518 | 0.544977 | 0.340518 | 0.544977 |
| affine, exact train measure | compact | 200 | 7.310311 | 0.503940 | 0.675621 | 0.503940 | 0.675621 |
| affine, exact train measure | wide, low LR | 150 | 8.178224 | 0.276358 | 0.397779 | 0.354079 | 0.601880 |

The affine wide arm is the only arm for which validation selection changed the
checkpoint. Its audit covariance residual decreased from `0.601880` to
`0.397779`, and its audit mean residual decreased from `0.354079` to
`0.276358`. These are descriptive changes from one bank and one seed, not a
statistically supported improvement. The residual remains incompatible with a
claim of exact or approximate IID Gaussian whitening under the declared
evidence standard.

The identity arms and affine compact arm selected step 200, so checkpoint
selection did not repair their terminal audit result. The train-to-validation
loss gaps at the terminal step were also nonzero (identity compact about
`1.02`, identity wide about `1.38`, affine compact about `1.15`, affine wide
about `1.39`), which is evidence of empirical-measure mismatch or finite
support/optimization error, not a posterior theorem.

## Decision tables

### Engineering, numerical, scientific ledgers

| Ledger | Status | Evidence | Limit |
|---|---|---|---|
| Engineering correctness | pass | both traces passed target/status, shape, finite, XLA, memory-growth, and round-trip gates | no HMC or production route |
| Numerical validity | pass for finite receipts | checkpoint fields are finite and audit ordering is explicit | no density-identification proof |
| Scientific interpretation | repair trigger | one arm shows descriptive checkpoint benefit; residuals persist | no IID, posterior, mode, or superiority claim |

### Decision table

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
|---|---|---|---|---|
| Keep corrected theta path active | target/measure/common support and receipts remain valid | no continuation veto | audit train/validation/audit measure separation, then design objective/support repair | target correctness |
| Do not promote current NeuTra candidates | audit residuals remain material | whitening promotion veto | obtain a larger or independently replicated audit/support diagnostic | HMC readiness |
| Retain validation checkpointing as optional diagnostic | rule is auditable and changed one arm | no engineering veto | use only with fresh selection/claim partitions in future runs | universal early-stopping rule |

### Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for both fresh traces and reporter |
| Statistically supported ranking | none; one bank and one seed |
| Descriptive-only differences | checkpoint index, losses, train/validation gaps, audit moments |
| Default readiness | not ready |
| Next evidence needed | explicit empirical-measure separation/support diagnostic and a separately reviewed objective or data-generation repair |

## Red-team and stop classification

The strongest alternative explanation is not simply “too many optimizer steps.”
The selected checkpoint helped one arm, but the persistent train/validation/audit
differences can arise because the fixed 256-row empirical bank and its
normalized weights do not represent the target geometry between partitions.
Another possibility is that the weighted forward-KL objective is not aligned
with the desired global transport criterion. The current audit set has only 12
rows, so its residuals are themselves noisy diagnostics.

The next smallest artifact is a read-only measure-separation report over the
existing N=256 trace roots: compare weighted theta moments, sign/mode counts,
effective sample sizes, and target/proposal log-ratio ranges for train,
validation, and audit partitions, without selecting or tuning on audit data.
That report will determine whether the next repair should generate more
independent support or alter the training objective.

No continuation veto fired. The target is still four-dimensional; the UKF
state remains internal. Poor whitening remains a promotion veto and repair
trigger, not evidence that the research direction is impossible.

## Nonclaims

- No IID Gaussian whitening theorem or posterior correctness claim.
- No exhaustive mode-discovery, normalizer, HMC, SMC-U, or canonical LEDH claim.
- No statistical superiority or default promotion claim.
