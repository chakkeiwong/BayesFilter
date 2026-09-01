# Phase 15 Paired Identity-versus-Affine Full-Bank Adjudication

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_PAIRED_DESCRIPTIVE`  
Budget cap: `900 s` from the unchanged `64800 s` campaign cap  
Input: Phase 8 metadata-bound/audited N=300 bank and Phase 14 full-bank metric repair  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase15`

## Question

Does the affine weighted-moment preconditioner improve the *full audited-bank*
NeuTra representation relative to the same runner, profile, seed, architecture,
and update budget with identity coordinates? This is a paired conditioning
diagnostic. It does not test posterior correctness, mode discovery, or an IID
Gaussian theorem.

## Mechanism and exact comparator

The input rows, normalized weights, train/validation/audit indices, target
signature, mode axis (`2`), and random seed are held fixed. The two arms differ
only in `--precondition identity` versus `--precondition affine`. Both use the
target-specific `tuning` profile and 300 updates, inherited as a bounded
comparison hypothesis from Phases 10--14 rather than as a promoted optimizer
default. The affine map is the Phase 13 TensorFlow weighted mean/Cholesky map;
it is not counted as a learned posterior density.

## Evidence contract

**Primary comparison.** For each arm, report the full-bank weighted latent mean,
covariance diagonal/off-diagonal residuals, covariance Frobenius residual, and
the exact transformed-target/parity gates. The paired difference is descriptive
with one seed; no ranking is statistically supported.

**Hard vetoes.** Invalid or non-finite rows; stale protocol or target hash;
wrong mode axis or weight alignment; failed affine/flow round trip; target or
status failure; missing full-bank metrics; failed GPU memory-growth/XLA or
batch-native training receipt; overwritten output root.

**Promotion veto.** A lower residual in one arm cannot promote that arm to an
authority, an IID sampler, a posterior transport, or HMC readiness. A failed
arm blocks only that arm and triggers the repair note.

**Explanatory diagnostics.** Validation loss, ESS, maximum weight, gradient
norm, runtime, and the validation-subset moments explain optimization behavior;
they do not replace full-bank metrics.

## Skeptical audit and pre-mortem

- **Wrong baseline risk:** the identity arm could use a stale mode split or
  positional weights. The runner's explicit `mode_axis=2`, index-aligned
  gathers, protocol hash, and audit receipt are checked before training.
- **Proxy risk:** validation loss could look favorable while full-bank moments
  remain poor. The full-bank diagnostic is the primary comparison object.
- **Seed risk:** one paired seed cannot establish a stochastic ranking. The
  result will say “paired descriptive evidence” and nominate a multi-seed run
  only if the effect is large enough to justify it.
- **Harness risk:** a missing full-bank field or a device-policy failure makes
  the artifact invalid, not evidence against preconditioning. The repair note
  records and retries only the affected component.
- **Scientific overreach:** even exact affine moment whitening does not identify
  a density. The result must retain the no-IID/no-posterior/non-HMC nonclaims.

## Exact execution

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py \
  --precondition identity --profile tuning \
  --plan docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase15-paired-preconditioning-subplan-2026-08-25.md \
  --m0-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --steps 300 --seed 20260825 7305 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase15-attempt1-identity-fullbank2401
```

The Phase 14 affine receipt is the paired comparator. No HMC, package change,
network fetch, model-file edit, or default-policy change is in scope.

## Exit and refresh

If the identity arm completes hard gates, write a result note comparing both
full-bank receipts and refresh Phase 16 toward either (a) a bounded multi-seed
conditioning check if the paired effect is compelling, or (b) a source-faithful
modular proposal/transform contract if conditioning is not the limiting issue.
If the identity run fails a harness gate, repair only that gate and rerun in a
new directory. Poor whitening is a repair/evidence trigger, not a program
blocker.
