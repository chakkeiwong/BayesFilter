# Phase 7 Target-Specific NeuTra Retuning Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `ROLE_LIMITED_EVIDENCE_RETAINED`  
Budget cap: `7200 s` within the unchanged global `64800 s` campaign cap  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase7`

## Objective

Test whether target-specific, batch-native NeuTra training remains numerically
valid when fed by the repaired N=300 M0 candidate bank, using correctly aligned
weights and a frozen train/validation/audit partition. This is a downstream
component screen. It does not promote the particle bank to an authority and it
does not authorize HMC.

## Entry gate

Phase 6's exact fixture and three N=300 q=20 candidate runs passed finite,
status, support, protocol-hash, and mutation-symmetry receipts. The NeuTra
runner's static tests and the new index-aligned weight path pass. The runner
must configure TensorFlow GPU memory growth before logical-device creation,
preserve a leading batch dimension through target/transport/loss/update, and
write a unique manifest. A post-run audit found that the original split used
the last coordinate instead of the pilot's declared signed coordinate
`theta[:, 2]`; the runner is repaired to use `MODE_AXIS = 2`. The earlier
outputs remain transport/status smoke evidence only until rerun.

## Skeptical pre-execution audit

| Risk | Why a successful command could still mislead | Earliest check | Treatment |
|---|---|---|---|
| Weight/row mismatch | a low loss could use weights from different particles | exact index gather and shape/finite assertions | repaired before execution; hard veto on mismatch |
| Normalized terminal weights | training measure is not an SMC-U law | explicit nonclaim and Phase 6 raw receipts | no authority/posterior promotion |
| Whitening proxy drift | latent mean/covariance can improve while target density is wrong | round-trip/logdet and transformed-target status on audit rows | whitening explanatory only |
| Overfitting/tuning leakage | selecting on audit rows inflates downstream result | deterministic 60/20/20 split; audit evaluated once after selection | hard split gate |
| GPU initialization | late memory-growth configuration invalidates run | environment guard plus repository GPU policy receipt | fail closed |
| Stochastic ranking | two short architecture arms and three banks are too small for superiority | per-bank tables and no ranking language | viable/failed only |
| Target implementation | target status can fail independently of transport | finite value/score/status on untouched audit rows | classify as harness/target failure, not method failure |

## Evidence contract

| Field | Predeclared choice |
|---|---|
| Question | Does the repaired N=300 particle bank support a valid batch-native NeuTra training screen under the target-specific objective? |
| Comparator | Two fixed architecture hypotheses (`compact`, `wide_low_lr`) with identical data split, target, seed policy, and update budget; three independent N=300 bank seeds are separate replications, not pooled rows |
| Primary hard criteria | GPU memory policy verified; finite tensors; exact forward/inverse and logdet parity; finite target value/score/status on untouched audit rows; batch size > 1; complete manifest |
| Promotion criterion | Candidate remains role-limited only if every hard criterion passes on each attempted bank; no architecture ranking without uncertainty evidence |
| Vetoes | memory-policy/device failure, scalar/row-mapped training, weight/index mismatch, non-finite values, parity residual > `1e-9`, invalid target status, audit leakage, incomplete artifact, HMC launch |
| Explanatory diagnostics | validation loss, latent weighted mean/covariance, ESS, clipping, runtime, architecture choice |
| Nonclaims | no IID Gaussian whitening, posterior correctness, mode discovery, HMC convergence, predictive improvement, statistical superiority, or default promotion |
| Artifact | one versioned directory per bank, launch/result/failure JSON, raw hashes, result note, repair note, and aggregate decision table |

## Defaults and numeric provenance

The two architecture configurations and `20` update steps are target-specific
screen hypotheses inherited from the earlier short screen; they are not
promoted defaults. The repaired runner derives a deterministic 60/20/20 split
from each bank's particle count (`180/60/60` for N=300), gathers weights by the
same indices as the rows, and uses batch size `180`. The parity tolerance
`1e-9` is inherited from the existing transport identity screen and is a hard
numerical check, not a whitening threshold. A failed screen triggers the
companion repair note before any next bank or architecture interpretation.

## Commands

For each of the three N=300 bank `seed-*` directories (the directory that
contains `pilot.json` and the tensor receipts), run with trusted GPU access and
a fresh output root:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py \
  --plan docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase7-neutra-retuning-subplan-2026-08-25.md \
  --m0-root <phase6-n300-bank> --steps 20 --output-root <phase7-attempt>
```

No `CUDA_VISIBLE_DEVICES=-1`, HMC controller, package install, network fetch,
or model-file edit is permitted in this phase. If the first bank fails for a
harness reason, repair and rerun that bank before moving to another bank.

## Exit and refresh

After each bank, run the focused static tests, classify failures, preserve the
artifact, and update
`...-phase7-repair-and-refresh-2026-08-25.md`. If all hard gates pass, write a
phase result that keeps architecture and bank differences descriptive and
refreshes the next phase toward an explicit raw-mass/measure audit. If a hard
gate fails, repair the smallest cause without changing the target or criteria.
Stop the whole program only under the master real-blocker definition.
