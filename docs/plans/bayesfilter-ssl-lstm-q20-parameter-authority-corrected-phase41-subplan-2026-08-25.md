# Corrected Parameter-Authority Phase 41 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.3-independent-audit-bank`  
Entry gate: Phase 40 passed the root-group boundary but found finite holdout support mismatch  
Status: `PASS_V2_3_INDEPENDENT_AUDIT_REPORT_REPAIR_TRIGGERED`  
Local cap: 1800 s

## Question

Are the Phase 40 NeuTra residuals primarily an artifact of the finite
root-group validation/audit partition, or do they persist when the unchanged
transport is evaluated on a newly generated, independent theta bank?

## Frozen target and training boundary

The target remains the batch-native q=20 SSL-LSTM posterior in
`theta in R^4`; the 60-dimensional UKF state remains internal. The v2.2 N=256
M0 bank remains the frozen training authority. Its deterministic
`root_group_stratified_v1` split and normalized train weights are loaded
read-only. The old validation rows may be used only for the optimizer's
within-run diagnostic; they cannot select a checkpoint or alter the fresh
audit bank.

Generate a new N=256 C0/M0 bank with the same target, proposal family,
schedule, defensive mixture, and protocol, but with a fresh root seed. The
fresh bank is an untouched audit source: no row, weight, target value, root,
or support summary from it may enter training, tuning, checkpoint selection,
or an optimizer decision. The fresh seed pair `(20260826, 4101)` is a
campaign hypothesis chosen to be distinct from prior roots; it is not a
promoted statistical default.

## Evidence contract

**Comparator.** Identity and exact-v2.2-training-measure affine transports,
same architecture/configuration and 200-step budget as Phase 40, trained on
the frozen v2.2 training rows and evaluated on the same frozen v2.2 validation
rows plus the independent fresh M0 bank.

**Primary hard gates.**

1. The fresh pilot is a passing `theta_R4` C0/M0 receipt with the exact Phase
   40 target signature and M0 protocol hash.
2. Old and fresh tensor hashes, root paths, and source manifests are recorded;
   no fresh artifact is the old pilot or a copied tensor.
3. Frozen training split is root-disjoint and complete, and the affine oracle
   is exact on that training measure.
4. All optimizer batches are `[batch,4]`, batch size exceeds one, GPU memory
   growth is verified before TensorFlow initialization, XLA is enabled, and
   all target/status/transport/round-trip values are finite.
5. The fresh-bank audit is evaluated after training and is not used for any
   selection. Its rows, weights, support summaries, and moment residuals are
   reported separately from the old validation diagnostic.

**Promotion vetoes.** Any stale signature, copied/reused fresh root,
non-finite/status-invalid target, split/hash mismatch, GPU policy violation,
or use of fresh rows for optimization/selection. Poor whitening, ESS, mode
occupancy, or a failed candidate is a repair trigger, not a continuation
veto.

## Interpretation ledger

| Observation | Role | Permitted conclusion |
|---|---|---|
| Fresh bank differs from old holdout and fresh residuals improve | explanatory/repair trigger | finite split/support mismatch is plausible; design a broader support or data-generation repair |
| Fresh bank resembles old holdout and residuals persist | explanatory/repair trigger | objective, capacity, or target-weighting mismatch is more plausible; design an objective repair |
| Fresh target/status or protocol gate fails | hard veto | repair the harness/proposal before interpreting transport results |
| Any moment, ESS, loss, or tail difference | descriptive only | no IID, posterior, mode, HMC, LEDH, or superiority claim |

The two branches above are hypotheses, not rankings. A single fresh bank and
one seed cannot establish a statistical preference between explanations.

## Skeptical pre-mortem and defaults

| Risk | Why it could mislead | Earliest check | Status |
|---|---|---|---|
| Fresh bank still follows the same mode-biased proposal | A new seed is not a coverage theorem | compare target/proposal/log-ratio ranges, mode fractions, and weighted theta moments | hypothesis |
| Fresh rows leak into training through an accidental concatenation | Would invalidate the audit | runner records separate tensor hashes and asserts audit rows are absent from optimizer inputs | hard implementation check |
| Old validation selects a checkpoint indirectly | Could overfit the frozen bank | use terminal state only for the primary fresh-audit receipt; report validation descriptively | reviewed design |
| One fresh bank gives noisy tail/moment estimates | Could create a false branch decision | classify all continuous differences as descriptive; require a follow-up replication before promotion | known limitation |
| Reusing the old affine map changes the target measure | Could produce a Jacobian error | exact train oracle and explicit `log_q_theta` composition receipt | hard gate |

Defaults inherited from Phase 40 (N=256, 200 steps, compact/wide arms,
`epsilon=0.20`, `safe_std=2.0`) remain hypotheses and are used only to hold the
comparison fixed. They are not promoted by this phase.

## Pre-execution skeptical audit

This bounded phase passed its pre-run audit on 2026-08-26. The audit checked
the baseline, proxy roles, stop conditions, comparison fairness, stale
metadata, environment policy, and artifact/question alignment:

| Audit item | Finding | Required handling |
|---|---|---|
| Baseline and target | Both banks use the same declared `theta_R4` target and protocol; the UKF state is not loaded as a particle | retain the four-dimensional shape gate |
| Proxy promotion | Latent moments, loss, ESS, and mode fractions cannot promote whitening or posterior correctness | classify them as descriptive/repair diagnostics |
| Fresh-bank leakage | A fresh row could accidentally enter the optimizer or checkpoint selector | keep the old split as the sole optimizer input and assert fresh-use flags are false |
| Checkpoint fairness | Validation-driven selection would make the fresh audit ambiguous | use the terminal state only; report old validation descriptively |
| Independent support | A new seed can still sample the same biased proposal region | compare support/range/mode summaries and require a later replication before ranking explanations |
| Environment | The pilot is CPU-hidden; the audit runner is GPU/XLA with memory growth required before import | fail closed on either policy mismatch |
| Artifact alignment | A successful command alone would not answer the support-versus-objective question | emit separate old-authority, fresh-audit, split, hash, and decision receipts |

The audit passes because no unexamined default is promoted and every remaining
unknown is either an explicit hypothesis or a stated nonclaim. A valid but
poor fresh-audit result is a repair trigger, not a continuation veto.

## Commands

Generate the fresh bank in the CPU diagnostic lane:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/fresh-n256 \
  --particles 256 --calibration-particles 64 --arms both --seed 20260826 4101
```

Then run the GPU/XLA frozen-training audit runner:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase41_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --fresh-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/fresh-n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/frozen-training-audit-attempt3 \
  --precondition both --split-policy root_group_stratified_v1 --steps 200 \
  --seed 20260825 4011
```

Finally run the read-only CPU reporter:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase41_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --fresh-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/fresh-n256 \
  --audit-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/frozen-training-audit-attempt3 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/report-attempt2
```

Each output root must be new. A failed harness attempt is preserved and
repaired in a new attempt root; no artifact is overwritten.

The first report attempt exposed a JSON/tensor dtype conversion defect before
writing a result. Its repair is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase41-report-repair-refresh-2026-08-26.md`.

The terminal receipts are:

- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/frozen-training-audit-attempt3/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/report-attempt2/`

## Refresh and stop

After the receipt, write a result and repair-refresh note. If all hard gates
pass, refresh the next subplan using measured support summaries and hashes. If
the fresh-bank target/common-support gate fails, stop the continuation only
after the adjacent proposal/harness repair has been attempted. Do not promote
NeuTra, HMC, posterior, IID-whitening, exhaustive-mode, or canonical LEDH
status from this phase.
