# Phase 6 Result: Exact SMC-U Fixture and Particle-Scale Repair

Status: `PASS_GATE_CANDIDATE_PARTICLE_SCALE_REPAIRED`

Phase 6 tested the implemented resampling-plus-mutation estimator on a known
normalizer and then replicated the repaired q=20 M0 candidate. The phase
addresses estimator bookkeeping and finite-run variability; it does not prove
mode discovery, posterior correctness, or an unbiased q=20 normalizer.

## Run receipts

The runner was
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_smc_u_replication_2026_08_25.py`
in `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, with
`CUDA_VISIBLE_DEVICES=-1`, TensorFlow 2.20.0, XLA enabled, and
`TF_FORCE_GPU_ALLOW_GROWTH=true`. Each output directory was unique and retained
the pilot command, source hashes, seed, protocol hash, raw tensor receipts,
and result markdown.

The exact fixture used 64 independent replicates with N=128 and known
unnormalized mass `Z=2.5`. It estimated mean `2.4812129`, MCSE `0.0243`, and
absolute error `0.0188`, within the predeclared `max(0.15, 4*MCSE)` screen.
All estimates were finite and every symmetric-mutation transition-density
residual was zero.

The q=20 candidate used the frozen schedule
`[0,.05,.1,.2,.35,.5,.65,.8,.9,1]`, defensive mixture epsilon `0.2`, one
symmetric random-walk mutation step at scale `0.05`, and fixed protocol hashes.
The six N=100 runs all passed their hard receipts but showed a wide mode
occupancy spread, so the declared particle-count repair was run. Three fresh
N=300 runs then passed the same hard receipts:

| seed | ESS fraction | weighted negative fraction | log mass | roots (-/+) | acceptance range |
|---:|---:|---:|---:|---:|---:|
| 1701 | 0.9772 | 0.4988 | -34.4198 | 72/67 | 0.1583--0.1817 |
| 1801 | 0.9596 | 0.4838 | -34.4603 | 74/75 | 0.1625--0.1808 |
| 1901 | 0.9582 | 0.4013 | -34.1937 | 75/68 | 0.1608--0.1817 |

Across N=300, weighted negative occupancy had mean `0.4613`, sample MCSE
`0.0303`, and range `0.4013--0.4988`; ESS had mean `0.9650` and MCSE
`0.0061`; log-mass MCSE was `0.0830`. Invalid proposals and transition
residuals were zero in all three runs. These summaries are descriptive: three
seeds cannot support a ranking or a default change.

## Three-ledger adjudication

**Engineering ledger.** The fixture and q=20 runners completed with unique
manifests, finite structured artifacts, matching protocol hashes, and complete
proposal/target/root/weight receipts. No HMC or model-file change was run.

**Numerical ledger.** The known-mass screen and symmetric-kernel identity pass.
Finite q=20 mass, support, and status gates pass for all N=300 runs. The
fixture validates the implemented estimator on its stated one-dimensional
target only; it does not identify the q=20 target law.

**Scientific ledger.** Increasing N from 100 to 300 reduced the observed mode
spread in this pilot, but the mutation is local and the seeded roots remain a
finite sample. The result is evidence that the candidate repair is viable for
the next role-limited screen, not evidence of exhaustive mode coverage or IID
Gaussian whitening.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain M0 as a candidate input | exact fixture and all N=300 hard receipts pass | no hard veto | finite mode coverage, q=20 mass uncertainty, target-specific training quality | run Phase 7 target-specific NeuTra retuning on the N=300 bank | no authority/default/posterior/HMC claim |
| Reject normalized replay as authority | fresh independent bank and raw receipts exist | historical replay remains non-admissible | finite-run bias and support outside seeded regions | preserve historical bank only as context | no claim that M0 is already unbiased |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for fixture and three N=300 candidate runs |
| Statistically supported ranking | none; no candidate ranking was attempted |
| Descriptive-only differences | N=300 mode/ESS/mass spread and acceptance |
| Default-readiness | not ready; authority and downstream gates remain open |
| Next evidence needed | target-specific NeuTra retuning, raw-mass/measure audit, and broader independent-seed/mode evidence |

## Red-team note

The strongest alternative explanation is that the apparent balance at N=300 is
inherited from the two seeded sign regions and the defensive proposal, while a
single short local mutation step does not traverse a missing bridge. The
normalizer fixture would not detect that q=20 support problem. Evidence that
would overturn the current candidate disposition is a failed exact fixture,
non-finite or hash-invalid q=20 receipt, reproducible support loss under fresh
seeds, or a target-specific downstream screen that cannot be repaired within
the campaign contract.

## Refresh

The next subplan is
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase7-neutra-retuning-subplan-2026-08-25.md`.
It must consume the N=300 bank explicitly, keep GPU/XLA and batch-native
training, use disjoint tuning/validation/audit rows, and remain outside HMC.
The companion repair note must classify any failure and refresh the next
subplan before continuation. The campaign has not hit a real blocker.
