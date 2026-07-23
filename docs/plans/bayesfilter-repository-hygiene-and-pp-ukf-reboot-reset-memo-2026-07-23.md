# Repository hygiene and PP-UKF reboot reset memo

Date: 2026-07-23 (Asia/Hong_Kong)

Repository branch: `main`

Base commit before this cleanup: `4281adf3c6067b706d83841bfc7a8fba022a65dd`

## Cleanup boundary

This trusted academic repository uses a simple evidence boundary:

- Track authored implementation, tests, benchmark harnesses, plans, result
  notes, source audits, local literature/source copies, and compact artifacts
  that support a promotion or claim.
- Ignore reproducible execution state: tensors, NumPy arrays, event streams,
  logs, stdout captures, mutable progress/checkpoints, private payloads,
  superseded attempts, local databases, and source snapshots.
- Historical generated failures remain described in result notes but are not
  promoted to tracked evidence when they do not support a current claim.
- The selected Zhao-Cui tuning artifact, authoritative CPU replay, final GPU
  claim, higher-moment terminal result/comparison, one-seed terminal result,
  and PP-UKF terminal public result/manifest remain trackable because they
  support stated validation or promotion decisions.

The `.gitignore` rules are intentionally exact for the current superseded
Zhao-Cui and feasibility roots. They do not hide authored `.py`, `.md`, or
tests. After staging, every visible path must be tracked; remaining generated
execution state must be ignored.

## PP-UKF terminal evidence

Attempt 09 completed successfully with exit code `0` and all ten candidate rows:

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/public_result.json
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/run_manifest.json
```

The attempt-08 resume checkpoint remains unchanged and local:

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-08/progress.json
SHA-256: 8590b64b48581d3bb13f8a8c02aa2dee323d6d842a87f83909a8063d3a5c391d
```

Attempt-09 charged `24,978.793005` seconds after a carry-in charge of
`17,424.711535`, for an aggregate `42,403.504540` seconds of the authorized
`86,400`-second campaign cap. Remaining budget is `43,996.495460` seconds.

Observed viable candidates were `L=5,13,14,18,19,24,25`. `L=9,12,17` stopped
at the driver's retained cap of `3,000` draws per chain and must not be treated
as final rejections under the reviewed `10,000`-draw maximum. No hard vetoes or
statistical ranking were established.

## Reboot preparation

No PP-UKF process or tmux session is active. Reboot is prepared but not
launched. Before attempt 10, repair and test the driver so that:

1. retained maximum is `10,000` per chain;
2. only `L=9,12,17` are selected for continuation;
3. failed short-cap rows are not silently treated as complete;
4. attempt 08 and attempt 09 remain immutable; and
5. the fresh attempt-10 result records the prior progress hash, policy, budget,
   target/transport identities, GPU memory policy, and native-divergence
   limitation.

The prepared command skeleton is:

```bash
tmux new-session -d -s pp_ukf_hmc_24h_20260723_reboot \
  "cd /home/chakwong/BayesFilter && \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
    docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py \
    --output-root docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10 \
    --candidate-index 1 --candidate-index 2 --candidate-index 5 \
    --resume-progress docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json \
    --prior-elapsed-seconds 42403.504540"
```

The command is intentionally a skeleton until the retained-cap and continuation
semantics are repaired. A fresh output root is mandatory; no prior output is
overwritten.

## Decision status

| Item | Status | Next action |
|---|---|---|
| Repository hygiene | ready for commit | stage all non-ignored paths and verify zero visible untracked files |
| PP-UKF attempt 09 | terminal, no hard veto | preserve public result and manifest |
| `L=9,12,17` | censored by 3,000-draw implementation cap | repair driver, then continue to 10,000 |
| Candidate ranking | not performed | remain unranked |
| Posterior/default/production claims | not established | require downstream evidence |

This reset changes no scientific target, data, frozen transport, candidate
controls, or claim. It only separates retained evidence from reproducible
runtime state and prepares a fresh, protocol-correct continuation.
